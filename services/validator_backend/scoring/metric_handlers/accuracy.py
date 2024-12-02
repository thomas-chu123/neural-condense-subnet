import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, DynamicCache, AutoModelForCausalLM
import structlog
from copy import deepcopy

logger = structlog.get_logger("accuracy")

DEFAULT_VALUE = 0


def accuracy(
    embed_model: AutoModel,
    kv_cache: DynamicCache,
    activation_prompt: str,
    expected_completion: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_tokens: int = 256,
    context: str = None,
    **kwargs,
) -> float:
    device = model.device
    expected_completion_ids = tokenizer(
        expected_completion,
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids.to(device=device, dtype=torch.long)
    n_expected_completion_tokens = expected_completion_ids.shape[1]
    max_new_tokens = int(n_expected_completion_tokens * 1.5)
    prompt_ids = tokenizer(
        activation_prompt,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_tokens,
    ).input_ids.to(device=device, dtype=torch.long)
    num_seen_tokens = kv_cache._seen_tokens
    logger.debug(f"Num seen tokens: {num_seen_tokens}")
    input_ids = torch.cat(
        [
            torch.full(
                (1, num_seen_tokens),
                0,
                dtype=torch.long,
                device=device,
            ),
            prompt_ids,
        ],
        dim=1,
    )
    kv_cache = kv_cache.to(device=device)
    outputs = model.generate(
        input_ids=input_ids, past_key_values=kv_cache, max_new_tokens=max_new_tokens
    )
    completion = tokenizer.decode(
        outputs[0][input_ids.shape[1] :], skip_special_tokens=True
    )
    completion = completion.strip() or "I don't know"
    ground_truth = expected_completion.strip()
    logger.debug(f"Activation prompt: {activation_prompt}")
    logger.debug(f"Completion: {completion}")
    logger.debug(f"Ground truth: {ground_truth}")
    return get_accuracy(completion, ground_truth, embed_model)


def get_accuracy(completion: str, ground_truth: str, embed_model: AutoModel) -> float:
    query_instruction = (
        "Instruct: Given a text, retrieve the text that has similar meaning.\nQuery:"
    )
    queries = [ground_truth]
    passages = [completion]
    max_length = 1024

    query_embeddings = embed_model.encode(
        queries, instruction=query_instruction, max_length=max_length
    )
    passage_embeddings = embed_model.encode(
        passages, instruction="", max_length=max_length
    )
    score = (query_embeddings @ passage_embeddings.T) * 100
    score = int(score[0][0].item())
    if score < 50:
        score = 0
    logger.debug(f"Score: {score}")
    return score


def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]
