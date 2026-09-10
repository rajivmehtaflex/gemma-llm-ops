---
language:
- en
- code
license: gemma
tags:
- unsloth
- qlora
- rlhf
- sft
- gemma-3
- llm-ops
base_model: unsloth/gemma-3-4b-it
datasets:
- rajivmehtapy/shell-sft-dataset
---

# Gemma Shell SFT Adapter (gemma-shell-sft)

This repository contains the fine-tuned LoRA adapter for `unsloth/gemma-3-4b-it` trained on production-grade Bash automation scripts.

## Model Summary

- **Base Model:** `unsloth/gemma-3-4b-it` (4-bit QLoRA)
- **Task:** Robust, safe, POSIX-aware Bash script generation with error handling (`set -euo pipefail`), input validation, and explanations.
- **Dataset Provenance:** 480 training examples from `data/splits/shell_sft_train.jsonl` covering 10 curriculum briefs (error handling, quoting safety, idempotency guards, dry-run previews, file/process management).
- **Training Framework:** Unsloth 2026.9.4 + Hugging Face TRL on an NVIDIA L4 GPU (24GB VRAM).

## Training Hyperparameters

| Parameter | Value |
| :--- | :--- |
| **Epochs** | 1 |
| **LoRA Rank ($r$)** | 16 |
| **LoRA Alpha ($\alpha$)** | 32 |
| **LoRA Dropout** | 0 |
| **Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| **Learning Rate** | 2e-4 |
| **Effective Batch Size** | 8 (per-device 2 $\times$ gradient accumulation 4) |
| **Max Sequence Length** | 1024 |
| **Optimizer** | `adamw_8bit` |
| **Training Time** | ~29 minutes (1756.8s) |
| **Final Training Loss** | 18.78 |

## Safety & Defensive Design

The model is trained strictly to enforce:
1. Shell safety flags: `set -euo pipefail`
2. Quoting of all variables and path interpolations (`"$var"`)
3. Defensive checking: `[[ -f ... ]]`, `[[ -d ... ]]` before acting
4. Dry-run (`--dry-run`) previews on destructive operations
5. Zero catastrophic commands: strictly filtered against `rm -rf /`, `mkfs`, raw block device writes, and fork bombs.

## How to Load & Use

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="rajivmehtapy/gemma-shell-sft",
    max_seq_length=1024,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

prompt = [{"role": "user", "content": "Write a bash script to rotate app.log safely."}]
formatted = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=400)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```