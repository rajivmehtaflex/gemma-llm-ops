# Runbook: Resuming RLHF Training Pipeline on a New Machine

This runbook provides step-by-step instructions to resume the full Gemma RLHF training pipeline on any GPU machine (e.g. NVIDIA L4, A10G, A100, RTX 3090/4090 with $\ge$ 24GB VRAM).

---

## 1. Prerequisites on the New Machine

- **Operating System:** Ubuntu 22.04 / 24.04 LTS
- **GPU:** NVIDIA GPU with $\ge$ 24 GB VRAM (`nvidia-smi` confirms driver and CUDA support)
- **Hugging Face Account:** `rajivmehtapy` (authenticate with a newly generated write-scoped token; never commit or paste it into commands)

---

## 2. Environment Setup

```bash
# 1. Update system packages
sudo apt update && sudo apt install -y git curl jq tmux build-essential python3-venv

# 2. Install uv for fast Python package management
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 3. Clone repository and checkout the training branch
git clone https://github.com/rajivmehtaflex/gemma-llm-ops.git
cd gemma-llm-ops
git checkout block3-training

# 4. Create and activate the locked Python 3.12 training environment
uv sync --group training
source .venv/bin/activate

# 5. Verify the GPU-backed stack and local services
uv run --group training python scripts/training_env.py
```

---

## 3. Hugging Face Authentication

Authenticate using Hugging Face's interactive browser flow. Revoke any token
that was previously exposed, then create a fresh write-scoped token. Never
place an access token in this file, a shell command, or shell history:

```bash
uv run --group training hf auth login --force

# Verify authenticated identity
uv run --group training hf auth whoami
# Expected output: user=rajivmehtapy
```

---

## 4. Download SFT Adapter Artifact

Download the trained SFT model checkpoint from the Hugging Face Hub. The
helper resolves the requested ref to the server-returned commit SHA, downloads
that exact snapshot, and writes an atomic hash manifest:

```bash
mkdir -p runs/sft-shell
uv run --group training python scripts/download_sft_artifact.py \
    --repo-id rajivmehtapy/gemma-shell-sft \
    --adapter runs/sft-shell \
    --manifest runs/sft-shell/evaluation/adapter_manifest.json

# Verify downloaded adapter files
ls -la runs/sft-shell/
# Expected: adapter_model.safetensors, adapter_config.json, tokenizer.json
cat runs/sft-shell/evaluation/adapter_manifest.json
```

---

## 5. Validated SFT Gate (current stopping point)

Run the paired base-versus-adapter evaluation on the 15 frozen prompts and 60
held-out prompts. Every response is graded by the seven-lens rubric through the
local Ollama judge; the structured gate requires complete evidence, no
catastrophic adapter violations, and a strictly lower held-out weighted
failure score. Seed 42 is primary; run seed 43 only if seed 42 fails.

```bash
uv run --group training python scripts/eval_sft.py --seed 42

# Only a structured PASS gate is accepted; Markdown-only claims are rejected.
uv run --group training python scripts/verify_e2e.py --check sft --repo-root .
```

Do not start DPO until this gate is `PASS`. If seed 42 returns
`RETRY_REQUIRED`, inspect the persisted JSONL records, resolve any malformed
human grades, and rerun once with `--seed 43`; a seed-43 pass is recorded as
`INCONCLUSIVE`, not an unconditional promotion.

---

## 6. Execute Remaining Training Stages

### Stage 1: Phase D — Direct Preference Optimization (DPO)

Run DPO on the 400 preference pairs using the SFT checkpoint:

```bash
python3 scripts/train_dpo.py \
    --model-name runs/sft-shell \
    --train-data data/preferences/shell_prefs_train.jsonl \
    --val-data data/preferences/shell_prefs_val.jsonl \
    --output runs/dpo-shell \
    --beta 0.1 \
    --lr 5e-6 \
    --epochs 1

# Upload DPO checkpoint to Hugging Face
hf repo create dpo-shell --repo-type model --private || true
hf upload rajivmehtapy/dpo-shell runs/dpo-shell .
```

### Stage 2: Phase E — Reward Model (RM)

Train the scalar reward head on the preference dataset:

```bash
python3 scripts/train_rm.py \
    --train-data data/preferences/shell_prefs_train.jsonl \
    --val-data data/preferences/shell_prefs_val.jsonl \
    --output runs/rm-shell \
    --lr 2e-5 \
    --epochs 1

# Upload Reward Model checkpoint to Hugging Face
hf repo create rm-shell --repo-type model --private || true
hf upload rajivmehtapy/rm-shell runs/rm-shell .
```

### Stage 3: Phase F — PPO Smoke & Educational Run

Run PPO policy optimization from the SFT and Reward Model checkpoints:

```bash
python3 scripts/train_ppo.py \
    --sft-model runs/sft-shell \
    --rm-model runs/rm-shell \
    --steps 100 \
    --output runs/ppo

# Verify logs and reward trajectory
cat runs/ppo/NOTES.md
```

---

## 6. Verification & Read-Back

Confirm all remote artifacts are accessible and verified on Hugging Face:

```bash
# Verify SFT adapter
hf download rajivmehtapy/gemma-shell-sft README.md --local-dir /tmp/check_sft && cat /tmp/check_sft/README.md

# Verify Preferences dataset
hf download rajivmehtapy/shell-sft-preferences README.md --local-dir /tmp/check_pref && cat /tmp/check_pref/README.md
```
