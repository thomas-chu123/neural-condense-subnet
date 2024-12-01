import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache, TextGenerationPipeline
import time
DEFAULT_VALUE = 0


def accuracy(
    judge_pipeline: TextGenerationPipeline,
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
    judge_messages = [
        {
            "role": "user",
            "content": f"Judge the correctness of the answer versus the ground truth. Return concisely 'yes' if the answer is correct, 'no' otherwise, don't need to explain.\n\n### **Ground truth**: {ground_truth}\n\n### **Answer**: {completion}",
        },
    ]
    print(f"Judge messages: {judge_messages}")
    start_time = time.time()
    score = judge_pipeline(judge_messages, max_new_tokens=32, return_full_text=False)[0]["generated_text"]
    end_time = time.time()
    print(f"Judge time: {end_time - start_time} seconds. Judge score: {score}")
    return 1 if 'yes' in score.lower() else 0

def preprocess_batch(values: list[float]) -> list[float]:
    return [value if value is not None else DEFAULT_VALUE for value in values]
