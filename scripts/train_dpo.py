#!/usr/bin/env python3
"""
DPO Training Script for Gemma Shell Ops using Unsloth + TRL DPOTrainer
Phase D: Direct Preference Optimization
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

import unsloth
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()

from datasets import Dataset
from trl import DPOTrainer, DPOConfig

def format_dpo_dataset(jsonl_path, max_samples=None):
    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                records.append({
                    "prompt": rec["prompt"],
                    "chosen": rec["chosen"],
                    "rejected": rec["rejected"],
                })
    if max_samples:
        records = records[:max_samples]
    return Dataset.from_list(records)

def main():
    parser = argparse.ArgumentParser(description="Train DPO model with Unsloth")
    parser.add_argument("--model-name", default="runs/sft-shell", help="Starting model adapter or base model")
    parser.add_argument("--train-data", default="data/preferences/shell_prefs_train.jsonl", help="Train dataset")
    parser.add_argument("--val-data", default="data/preferences/shell_prefs_val.jsonl", help="Validation dataset")
    parser.add_argument("--output", default="runs/dpo-shell", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--max-prompt-length", type=int, default=512, help="Max prompt length")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate (workbook default: 5e-6)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature beta (workbook default: 0.1)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing FastLanguageModel for DPO from {args.model_name} ===")
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

    train_dataset = format_dpo_dataset(args.train_data, args.max_samples)
    val_dataset = None
    if Path(args.val_data).is_file():
        val_dataset = format_dpo_dataset(args.val_data, args.max_samples)

    print(f"Loaded DPO dataset: {len(train_dataset)} train samples, {len(val_dataset) if val_dataset else 0} val samples.")

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=0.1,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        beta=args.beta,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_8bit",
        seed=42,
        report_to="none",
        max_length=args.max_seq_length,
        max_prompt_length=args.max_prompt_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=dpo_config,
    )

    print("=== Commencing DPO Training ===")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"=== Saving DPO Adapter to {out_dir} ===")
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    log_file = out_dir / "train_log.md"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"# DPO Training Log ({out_dir.name})\n\n")
        f.write(f"- **Starting Checkpoint:** `{args.model_name}`\n")
        f.write(f"- **Train Pairs:** {len(train_dataset)}\n")
        f.write(f"- **Epochs:** {args.epochs}\n")
        f.write(f"- **Beta:** {args.beta}\n")
        f.write(f"- **Learning Rate:** {args.lr}\n")
        f.write(f"- **Effective Batch Size:** {args.batch_size * args.grad_accum}\n")
        f.write(f"- **Elapsed Time:** {elapsed:.1f}s\n")
        f.write(f"- **Final DPO Loss:** {train_result.training_loss:.4f}\n")

    print(f"DPO Training completed in {elapsed:.1f}s. Log written to {log_file}")

if __name__ == "__main__":
    main()
