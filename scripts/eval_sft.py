#!/usr/bin/env python3
"""
SFT Evaluation Script
Generates answers for the 60 held-out test prompts in data/splits/shell_sft_test.jsonl
using the trained adapter runs/sft-shell, performs safety checking, and logs results.
"""

import json
import os
import sys
import time
from pathlib import Path
import torch

import unsloth
from unsloth import FastLanguageModel

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_pipeline import is_command_safe

def main():
    test_file = REPO_ROOT / "data" / "splits" / "shell_sft_test.jsonl"
    adapter_path = REPO_ROOT / "runs" / "sft-shell"
    out_file = REPO_ROOT / "data" / "analysis" / "sft_eval.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading adapter from {adapter_path}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(adapter_path),
        max_seq_length=1024,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    tok = getattr(tokenizer, "tokenizer", tokenizer)

    prompts = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                # extract prompt from messages
                msgs = rec.get("messages", [])
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                if user_msg:
                    prompts.append(user_msg)

    print(f"Evaluating {len(prompts)} test prompts...")
    results = []
    safety_violations = 0

    torch.manual_seed(42)

    with open(out_file, "w", encoding="utf-8") as out_f:
        for idx, prompt_text in enumerate(prompts, 1):
            chat = [{"role": "user", "content": prompt_text}]
            formatted = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            inputs = tok(formatted, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    do_sample=True,
                )

            # Strip input tokens from output
            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            completion = tok.decode(gen_tokens, skip_special_tokens=True).strip()

            is_safe, reasons = is_command_safe(completion)
            if not is_safe:
                safety_violations += 1

            entry = {
                "id": f"sft-test-{idx:03d}",
                "prompt": prompt_text,
                "completion": completion,
                "is_safe": is_safe,
                "safety_reasons": reasons,
            }
            out_f.write(json.dumps(entry) + "\n")
            out_f.flush()
            results.append(entry)

            if idx % 10 == 0 or idx == len(prompts):
                print(f"Evaluated {idx}/{len(prompts)} prompts. Safety violations: {safety_violations}")

    eval_summary = REPO_ROOT / "runs" / "sft-shell" / "eval_summary.md"
    with open(eval_summary, "w", encoding="utf-8") as ef:
        ef.write("# SFT Evaluation Summary\n\n")
        ef.write(f"- **Evaluated Test Prompts:** {len(prompts)}\n")
        ef.write(f"- **Safety Violations:** {safety_violations} (0 expected)\n")
        ef.write(f"- **Safety Compliance Rate:** {(len(prompts)-safety_violations)/len(prompts):.1%}\n")
        ef.write(f"- **Output Records:** `{out_file}`\n")

    print(f"Evaluation complete! Summary saved to {eval_summary}")

if __name__ == "__main__":
    main()
