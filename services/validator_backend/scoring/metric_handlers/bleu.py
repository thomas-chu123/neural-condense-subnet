import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, TextGenerationPipeline
import time
from torchmetrics.functional.text import bleu_score as bleu_score_fn

DEFAULT_VALUE = 0


def bleu(
    kv_cache: DynamicCache,
    activation_prompt: str,
    expected_completion: str,
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    max_tokens: int = 256,
    **kwargs,
) -> float:
    print(f"Activation prompt: {activation_prompt}")
    print(f"Expected completion: {expected_completion}")
    device = model.device
    expected_completion_ids = tokenizer(
        expected_completion,
        return_tensors="pt",
        add_special_tokens=False,
        max_length=max_tokens,
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
    start_time = time.time()
    outputs = model.generate(input_ids=input_ids, past_key_values=kv_cache, max_new_tokens=max_new_tokens)
    end_time = time.time()
    print(f"Generation time: {end_time - start_time} seconds")
    completion = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
    completion = completion.strip() or "I don't know"
    ground_truth = expected_completion.strip()

    bleu_score = bleu_score_fn(preds=[completion], target=[ground_truth])
    print(f"Completion: {completion}")
    print(f"Ground truth: {ground_truth}")
    print(f"BLEU score: {bleu_score}")
    
    return bleu_score.item()

def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]
