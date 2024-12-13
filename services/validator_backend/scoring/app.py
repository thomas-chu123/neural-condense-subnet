# TODO: Efficient switching between target models. Currently fixed to mistral-7b-instruct-v0.2.
from flask import Flask, request, jsonify
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DynamicCache,
    AutoModel,
)
from transformers import pipeline
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
            "Condense-AI/Mistral-7B-Instruct-v0.2"
        ).to(dtype=self.dtype, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Condense-AI/Mistral-7B-Instruct-v0.2"
        )
        self.embed_model = AutoModel.from_pretrained(
            "nvidia/NV-Embed-v2", trust_remote_code=True, torch_dtype=self.dtype
        ).to(device=self.device)
        self.filter_existance_checker = FilterExistanceChecker()

    @torch.no_grad()
    def get_metrics(self, request: BatchedScoringRequest) -> dict[str, float]:
        criteria = random.choice(request.ground_truth_request.criterias)
        values = []
        metric_handler = metric_handlers[criteria]["handler"]
        preprocess_batch = metric_handlers[criteria]["preprocess_batch"]
        positive_chunk, negative_chunk = (
            self.filter_existance_checker.get_messages_pair(
                request.ground_truth_request.messages,
                request.ground_truth_request.hidden_messages,
            )
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
                    embed_model=self.embed_model,
                    kv_cache=kv_cache,
                    activation_prompt=request.ground_truth_request.activation_prompt,
                    expected_completion=request.ground_truth_request.expected_completion,
                    tokenizer=self.tokenizer,
                    model=self.model,
                    context=request.ground_truth_request.context,
                    positive_chunk=positive_chunk,
                    negative_chunk=negative_chunk,
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
