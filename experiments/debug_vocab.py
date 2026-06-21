import os
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora_train.jsonl")

print("Loading config...")
config = AutoConfig.from_pretrained(MODEL_ID)
print(f"Model vocab_size from config: {config.vocab_size}")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print("Applying chat template to first 5 examples...")
raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_chat_template(example):
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}
    
dataset = raw_dataset.map(format_chat_template)

max_id = 0
min_id = 999999

print("Tokenizing and finding min/max token ID...")
for i in range(100):
    tokens = tokenizer(dataset[i]["text"])["input_ids"]
    if max(tokens) > max_id:
        max_id = max(tokens)
    if min(tokens) < min_id:
        min_id = min(tokens)

print(f"Max token ID in dataset: {max_id}")
print(f"Min token ID in dataset: {min_id}")
print(f"pad_token_id: {tokenizer.pad_token_id}")
print(f"eos_token_id: {tokenizer.eos_token_id}")
