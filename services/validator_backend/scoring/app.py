# TODO: Efficient switching between target models. Currently fixed to mistral-7b-instruct-v0.2.
from flask import Flask, request, jsonify
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
    TextGenerationPipeline,
)
import random
import structlog
import gc
from .datatypes import BatchedScoringRequest
import traceback
from .metric_handlers import metric_handlers
from .anti_exploitation.filter_existance import FilterExistanceChecker
import time

gc.enable()
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer(),
    ]
)

logger = structlog.get_logger("Validator-Backend")


class ScoringService:
    def __init__(self):
        """
        Initializes the ScoringService with model and tokenizer storage, device configuration,
        and a lock for thread-safe operations. Runs a unit test to verify setup.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            "Condense-AI/Mistral-7B-Instruct-v0.2",
            attn_implementation="flash_attention_2",
            torch_dtype=self.dtype,
        ).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Condense-AI/Mistral-7B-Instruct-v0.2"
        )

        j_tokenizer = AutoTokenizer.from_pretrained(
            "upstage/solar-pro-preview-instruct"
        )
        j_model = AutoModelForCausalLM.from_pretrained(
            "upstage/solar-pro-preview-instruct",
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.judge_pipeline = TextGenerationPipeline(
            model=j_model,
            tokenizer=j_tokenizer,
            device=self.device,
            torch_dtype=self.dtype,
        )
        self.filter_existance_checker = FilterExistanceChecker()

    @torch.no_grad()
    def get_metrics(self, request: BatchedScoringRequest) -> dict[str, float]:
        logger.info("Received request")
        criteria = random.choice(request.ground_truth_request.criterias)
        values = []
        metric_handler = metric_handlers[criteria]["handler"]
        preprocess_batch = metric_handlers[criteria]["preprocess_batch"]
        logger.info(
            "positive_chunks",
            positive_chunks=request.ground_truth_request.positive_chunks,
        )
        logger.info(
            "negative_chunks",
            negative_chunks=request.ground_truth_request.negative_chunks,
        )
        for miner_response in request.miner_responses:
            try:
                miner_response.decode()
                kv_cache = DynamicCache.from_legacy_cache(
                    torch.from_numpy(miner_response.compressed_kv).to(
                        device=self.device, dtype=self.dtype
                    )
                )
                start_time = time.time()
                value = metric_handler(
                    filter_existance_checker=self.filter_existance_checker,
                    kv_cache=kv_cache,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    ground_truth_request=request.ground_truth_request,
                    judge_pipeline=self.judge_pipeline,
                )
                end_time = time.time()
                logger.info(
                    "metric_handler_time",
                    handler_name=metric_handler.__name__,
                    time_taken=f"{end_time - start_time:.2f}s",
                )
            except Exception as e:
                logger.error(
                    "metric_handler_error",
                    error=str(e),
                    handler_name=metric_handler.__name__,
                    traceback=traceback.format_exc(),
                )
                value = None
            values.append(value)
            logger.info(
                "metric_value", handler_name=metric_handler.__name__, value=value
            )
        values = preprocess_batch(values)
        return {"metrics": {criteria: values}}


app = Flask(__name__)
scoring_service = ScoringService()


@app.route("/", methods=["GET"])
def is_alive():
    return jsonify({"message": "I'm alive!"})


@app.route("/get_metrics", methods=["POST"])
def get_metrics():
    request_data = BatchedScoringRequest(**request.get_json())
    return jsonify(scoring_service.get_metrics(request_data))
