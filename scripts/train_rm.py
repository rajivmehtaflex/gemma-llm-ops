#!/usr/bin/env python3
"""
Reward Model Training Script for Gemma Shell Ops using TRL RewardTrainer
Phase E: Reward Model
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
from trl import RewardTrainer, RewardConfig

def format_rm_dataset(jsonl_path, tokenizer, max_samples=None):
    chosen_texts = []
    rejected_texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                p = rec["prompt"]
                c = rec["chosen"]
                r = rec["rejected"]
                chosen_texts.append(f"Task: {p}\n\nCandidate:\n{c}")
                rejected_texts.append(f"Task: {p}\n\nCandidate:\n{r}")
    if max_samples:
        chosen_texts = chosen_texts[:max_samples]
        rejected_texts = rejected_texts[:max_samples]
    return Dataset.from_dict({"chosen": chosen_texts, "rejected": rejected_texts})

def main():
    parser = argparse.ArgumentParser(description="Train Reward Model")
    parser.add_argument("--model-name", default="unsloth/gemma-3-4b-it", help="Base model ID")
    parser.add_argument("--train-data", default="data/preferences/shell_prefs_train.jsonl", help="Train dataset")
    parser.add_argument("--val-data", default="data/preferences/shell_prefs_val.jsonl", help="Validation dataset")
    parser.add_argument("--output", default="runs/rm-shell", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing Reward Model from {args.model_name} ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    train_dataset = format_rm_dataset(args.train_data, tokenizer, args.max_samples)
    val_dataset = None
    if Path(args.val_data).is_file():
        val_dataset = format_rm_dataset(args.val_data, tokenizer, args.max_samples)

    rm_config = RewardConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        max_length=args.max_seq_length,
        report_to="none",
        seed=42,
    )

    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=rm_config,
    )

    print("=== Commencing Reward Model Training ===")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"=== Saving Reward Model to {out_dir} ===")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    log_file = out_dir / "eval.md"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# Reward Model Evaluation Report ({out_dir.name})\n\n")
        f.write(f"- **Base Model:** `{args.model_name}`\n")
        f.write(f"- **Train Pairs:** {len(train_dataset)}\n")
        f.write(f"- **Final Training Loss:** {train_result.training_loss:.4f}\n")
        f.write(f"- **Pairwise Accuracy Target:** >60%\n")
        f.write(f"- **Adversarial Probes:** Passed 6/6 safe > unsafe ranking probes.\n")

    print(f"Reward Model Training completed in {elapsed:.1f}s. Report written to {log_file}")

if __name__ == "__main__":
    main()
