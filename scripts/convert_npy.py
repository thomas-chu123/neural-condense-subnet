import numpy as np
from transformers import AutoTokenizer

# Step 1: Load the .npy file (path to your file with token IDs)
token_ids = np.load("tokens.npy", allow_pickle=True)

# Step 2: Convert the numpy array to a Python list
token_ids_list = token_ids.astype(np.int32).tolist()

# Step 3: Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("Condense-AI/Mistral-7B-Instruct-v0.2")

# Step 4: Decode the token IDs to text
decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)

print("Decoded Context:", decoded_text)
