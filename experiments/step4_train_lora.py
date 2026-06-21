import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

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
        
    # 3. Format Dataset with Chat Template and Tokenize
    print("Applying chat template and tokenizing dataset...")
    def format_and_tokenize(examples):
        # We process batched examples
        texts = [tokenizer.apply_chat_template(msg, tokenize=False) for msg in examples["messages"]]
        return tokenizer(texts, truncation=True, max_length=1500)
        
    dataset = raw_dataset.map(format_and_tokenize, batched=True, remove_columns=raw_dataset.column_names)
    
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
    print(f"Model lm_head shape: {model.get_output_embeddings().weight.shape}")
    
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
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4, # Effective batch size = 8
        optim="paged_adamw_32bit",
        save_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        num_train_epochs=1, 
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True, 
        report_to="none" 
    )
    
    # 7. Initialize Trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    class DebugTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.get("labels")
            if labels is not None:
                print(f"DEBUG: labels min={labels.min().item()}, max={labels.max().item()}, shape={labels.shape}")
                # Count negative labels other than -100
                bad_negatives = ((labels < 0) & (labels != -100)).sum().item()
                if bad_negatives > 0:
                    print(f"DEBUG: Found {bad_negatives} negative labels that are not -100!")
            
            # Run forward pass to get logits
            outputs = model(**inputs)
            logits = outputs.get("logits")
            if logits is not None:
                print(f"DEBUG: logits shape={logits.shape}")
            
            # Return standard loss computation
            inputs["labels"] = labels
            if num_items_in_batch is None:
                return super().compute_loss(model, inputs, return_outputs)
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    trainer = DebugTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
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
