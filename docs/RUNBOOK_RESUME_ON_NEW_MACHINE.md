# Runbook: Resuming RLHF Training Pipeline on a New Machine

This runbook provides step-by-step instructions to resume the full Gemma RLHF training pipeline on any GPU machine (e.g. NVIDIA L4, A10G, A100, RTX 3090/4090 with $\ge$ 24GB VRAM).

---

## 1. Prerequisites on the New Machine

- **Operating System:** Ubuntu 22.04 / 24.04 LTS
- **GPU:** NVIDIA GPU with $\ge$ 24 GB VRAM (`nvidia-smi` confirms driver and CUDA support)
- **Hugging Face Account:** `rajivmehtapy` (Write-access token required)

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

# 4. Create and activate virtual environment
uv venv --python 3.12
source .venv/bin/activate

# 5. Install Unsloth and Hugging Face CLI
uv pip install unsloth "huggingface_hub[cli]"
python3 -c "import unsloth, torch; print(f'Unsloth OK on {torch.cuda.get_device_name(0)}')"
```

---

## 3. Hugging Face Authentication

Authenticate using Hugging Face's interactive browser flow. Never place an
access token in this file, a shell command, or shell history:

```bash
hf auth login --force

# Verify authenticated identity
hf auth whoami
# Expected output: user=rajivmehtapy
```

---

## 4. Download SFT Adapter Artifact

Download the trained SFT model checkpoint from the Hugging Face Hub:

```bash
mkdir -p runs/sft-shell
hf download rajivmehtapy/gemma-shell-sft --local-dir runs/sft-shell

# Verify downloaded adapter files
ls -la runs/sft-shell/
# Expected: adapter_model.safetensors, adapter_config.json, tokenizer.json
```

---

## 5. Execute Remaining Training Stages

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
