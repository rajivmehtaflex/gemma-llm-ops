#!/usr/bin/env python3
"""
SFT Training Script for Gemma Shell Ops using Unsloth QLoRA
Phase C: SFT Pilot & Full Run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

import unsloth
from unsloth import FastLanguageModel
from datasets import Dataset
from trl import SFTTrainer, SFTConfig

def format_prompts(batch, tokenizer):
    texts = []
    for messages in batch["messages"]:
        # messages is list of dicts: system, user, assistant
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    return {"text": texts}

def main():
    parser = argparse.ArgumentParser(description="Train SFT model with Unsloth")
    parser.add_argument("--model-name", default="unsloth/gemma-3-4b-it", help="Base model ID")
    parser.add_argument("--train-data", default="data/splits/shell_sft_train.jsonl", help="Train dataset")
    parser.add_argument("--val-data", default="data/splits/shell_sft_val.jsonl", help="Validation dataset")
    parser.add_argument("--output", default="runs/sft-shell", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (for pilot run)")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing FastLanguageModel ({args.model_name}) ===")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Load dataset
    train_records = []
    with open(args.train_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_records.append(json.loads(line))
    if args.max_samples:
        train_records = train_records[:args.max_samples]

    val_records = []
    if Path(args.val_data).is_file():
        with open(args.val_data, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    val_records.append(json.loads(line))
    if args.max_samples and val_records:
        val_records = val_records[:min(len(val_records), args.max_samples // 4)]

    print(f"Dataset loaded: {len(train_records)} train samples, {len(val_records)} val samples.")

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_list(train_records)
    train_dataset = train_dataset.map(lambda b: format_prompts(b, tokenizer), batched=True)
    
    val_dataset = None
    if val_records:
        val_dataset = Dataset.from_list(val_records)
        val_dataset = val_dataset.map(lambda b: format_prompts(b, tokenizer), batched=True)

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=sft_config,
    )

    print("=== Commencing Training ===")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"=== Saving Model Adapter to {out_dir} ===")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    # Evaluate validation loss
    eval_metrics = {}
    if val_dataset:
        print("=== Evaluating on Validation Set ===")
        eval_metrics = trainer.evaluate()

    # Record train log
    log_file = out_dir / "train_log.md"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# SFT Training Log ({out_dir.name})\n\n")
        f.write(f"- **Base Model:** `{args.model_name}`\n")
        f.write(f"- **Train Samples:** {len(train_records)}\n")
        f.write(f"- **Epochs:** {args.epochs}\n")
        f.write(f"- **Effective Batch Size:** {args.batch_size * args.grad_accum}\n")
        f.write(f"- **Learning Rate:** {args.lr}\n")
        f.write(f"- **Elapsed Time:** {elapsed:.1f}s\n")
        f.write(f"- **Final Train Loss:** {train_result.training_loss:.4f}\n")
        if eval_metrics:
            f.write(f"- **Validation Loss:** {eval_metrics.get('eval_loss', 'N/A')}\n")

    print(f"Training completed in {elapsed:.1f}s. Log written to {log_file}")

if __name__ == "__main__":
    main()
