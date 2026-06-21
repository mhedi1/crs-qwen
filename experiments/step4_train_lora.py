import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

_EXPERIMENTS_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DATASET_PATH = os.path.join(_EXPERIMENTS_DIR, "lora_train.jsonl")
OUTPUT_DIR = os.path.join(_EXPERIMENTS_DIR, "qwen2.5-7b-lora-adapter")

def train():
    print("=== STEP 4: LORA FINE-TUNING ===")
    print(f"Base Model: {MODEL_ID}")
    print(f"Dataset: {DATASET_PATH}")
    
    # 1. Load Dataset
    print("\nLoading dataset...")
    raw_dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
    
    # 2. Load Tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # 3. Format Dataset with Chat Template
    print("Applying chat template to dataset...")
    def format_chat_template(example):
        # apply_chat_template automatically handles formatting the System, User, and Assistant roles
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
        return {"text": text}
        
    dataset = raw_dataset.map(format_chat_template)
    
    # 4. Load Model in 4-bit
    print("\nLoading base model in 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        # Using auto now that the VRAM will be cleared
        device_map="auto", 
        trust_remote_code=True
    )
    
    # Prepare model for QLoRA
    model = prepare_model_for_kbit_training(model)
    
    # 5. Configure LoRA Adapter
    print("\nConfiguring LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        # Targeting all linear layers for maximum learning capability
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # 6. Training Arguments (Using new SFTConfig for latest trl versions)
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # Effective batch size = 8
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1, # 1 epoch over 2000 examples is perfect for LoRA adaptation
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True, # Use bfloat16 for stability
        report_to="none" # Disable wandb/mlflow for this isolated run
    )
    
    # 7. Initialize Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=lora_config
    )
    
    print("\nStarting Training...")
    # Fix for newer PyTorch/Transformers versions occasionally ignoring use_cache
    model.config.use_cache = False 
    
    trainer.train()
    
    print(f"\nTraining Complete! Saving final adapter to {OUTPUT_DIR}/final")
    trainer.model.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    
if __name__ == "__main__":
    train()
