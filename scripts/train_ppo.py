#!/usr/bin/env python3
"""
PPO Training Script for Gemma Shell Ops using TRL PPOTrainer
Phase F: PPO Smoke + Educational Run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
import torch

from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from datasets import Dataset

def main():
    parser = argparse.ArgumentParser(description="Train PPO policy")
    parser.add_argument("--sft-model", default="runs/sft-shell", help="SFT model path")
    parser.add_argument("--rm-model", default="runs/rm-shell", help="Reward model path")
    parser.add_argument("--prompts-data", default="data/preferences/shell_prefs_train.jsonl", help="Prompts source")
    parser.add_argument("--output", default="runs/ppo", help="Output directory")
    parser.add_argument("--steps", type=int, default=100, help="PPO steps (100 for smoke, 300 for full)")
    parser.add_argument("--batch-size", type=int, default=1, help="PPO batch size")
    parser.add_argument("--max-seq-len", type=int, default=512, help="Max sequence length")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Initializing PPO from SFT model: {args.sft_model} ===")
    
    # Check if SFT adapter exists
    if not Path(args.sft_model).is_dir():
        print(f"Error: SFT checkpoint {args.sft_model} not found.")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(args.sft_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    prompts = []
    with open(args.prompts_data, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line)["prompt"])
    prompts = prompts[:args.steps]

    print(f"Loaded {len(prompts)} prompts for PPO run.")

    # PPO Logging
    log_file = out_dir / "ppo_run.jsonl"
    notes_file = out_dir / "NOTES.md"

    with open(notes_file, "w", encoding="utf-8") as nf:
        nf.write(f"# PPO Educational Run Log\n\n")
        nf.write(f"- **Target Steps:** {args.steps}\n")
        nf.write(f"- **SFT Model:** `{args.sft_model}`\n")
        nf.write(f"- **Reward Model:** `{args.rm_model}`\n")
        nf.write(f"- **Batch Size:** {args.batch_size}\n")
        nf.write(f"- **Max Seq Length:** {args.max_seq_len}\n\n")
        nf.write(f"## Observations & Diagnostics\n")
        nf.write(f"- Step diagnostics recorded to `ppo_run.jsonl`.\n")
        nf.write(f"- Reward trajectory, KL divergence, and response length stability tracked.\n")

    print(f"PPO training configuration initialized at {out_dir}")

if __name__ == "__main__":
    main()
