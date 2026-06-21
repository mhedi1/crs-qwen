import os
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_train.jsonl")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_and_tokenize(examples):
    texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
    return tokenizer(texts, truncation=True, max_length=1500)

dataset = raw_dataset.map(format_and_tokenize, batched=True, remove_columns=raw_dataset.column_names)

max_id = -999999
min_id = 999999
bad_ids = []

for i in range(len(dataset)):
    tokens = dataset[i]["input_ids"]
    mx = max(tokens)
    mn = min(tokens)
    if mx > max_id: max_id = mx
    if mn < min_id: min_id = mn
    if mx >= 152064 or mn < 0:
        bad_ids.append((i, [t for t in tokens if t >= 152064 or t < 0]))

print(f"Checked {len(dataset)} examples.")
print(f"Max token ID: {max_id}")
print(f"Min token ID: {min_id}")
print(f"Bad examples: {len(bad_ids)}")
if bad_ids:
    print(f"First bad example: {bad_ids[0]}")
