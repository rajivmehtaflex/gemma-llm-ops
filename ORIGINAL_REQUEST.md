# Original User Request

## Initial Request — 2026-09-10T14:32:23Z

Execute the end-to-end RLHF training pipeline described in `docs/plans/2026-09-10_block3-full-pipeline.md` directly on the local NVIDIA L4 GPU environment, progressing through preference pair generation, SFT, DPO, Reward Modeling, and PPO, then publishing verified datasets and model checkpoints to private Hugging Face repositories.

Working directory: `/root/content/gemma-llm-ops`
Integrity mode: development

Reference plan: `docs/plans/2026-09-10_block3-full-pipeline.md`

## Requirements

### R1. Local Environment & Service Setup
Initialize and verify the execution environment on the local machine (NVIDIA L4 24GB GPU, 377GB RAM). Ensure required build packages, Python 3.12 virtual environment with Unsloth and PyTorch CUDA support, local Ollama daemon hosting `qwen2.5:14b`, and Hugging Face CLI authenticated with token `[HF_WRITE_TOKEN_MASKED]` are running and functional.

### R2. Preference Dataset Construction & Automated Quality Auditing
Implement and execute the Phase B pipeline extensions to generate 500 novel shell preference pairs (400/50/50 split) themed from `data/briefs/briefs.json`. Enforce strict safety filtering (zero catastrophic commands like `rm -rf /` or `mkfs`), baseline prompt non-leakage, and consistent markdown formatting. Execute the auditing and human-gate requirements autonomously using an automated high-precision LLM judge proxy.

### R3. Sequential Model Training & Evaluation
Train and evaluate each model stage using Unsloth QLoRA configurations matching the workbook specifications:
- SFT pilot (50 samples) and full SFT (1 epoch) on `shell_sft_train.jsonl`, evaluated against the 15 frozen baseline prompts with zero command safety regressions.
- DPO training starting from SFT checkpoint with held-out preference accuracy > 60% and safety regression verification.
- Reward Model training with pairwise accuracy > 60% and verification against the 6 adversarial probe categories.
- PPO smoke run (100 steps) and educational run (≤ 300 steps), recording step logs, reward trajectory, KL divergence, and response length stability.

### R4. Hugging Face Hub Publication & Read-Back Verification
Generate comprehensive model cards (`README.md`) for all trained adapters and datasets. Create private Hugging Face repositories under the authenticated user account for datasets (`shell-sft-preferences`, `shell-sft-dataset`) and model checkpoints (`gemma-shell-sft`, `dpo-shell`, `rm-shell`), upload artifacts, and strictly confirm every upload via read-back download and cache verification.

## Acceptance Criteria

### Environment Readiness
- [ ] `nvidia-smi` and PyTorch confirm NVIDIA L4 CUDA acceleration is operational.
- [ ] `ollama list` shows `qwen2.5:14b` loaded and responsive on `localhost:11434`.
- [ ] `hf auth whoami` successfully authenticates with the provided write token.

### Dataset Verification
- [ ] Exactly 500 preference pairs partitioned into `data/preferences/shell_prefs_{train,val,test}.jsonl` (400/50/50).
- [ ] `python3 scripts/sft_pipeline.py validate_pairs` outputs `PAIRS valid`.
- [ ] `python3 scripts/sft_pipeline.py safety_check data/preferences/*.jsonl` completes with 0 violations.

### Training Artifacts & Logs
- [ ] Checkpoint adapters saved for `runs/sft-shell`, `runs/dpo-shell`, `runs/rm-shell`, and logs for `runs/ppo/`.
- [ ] SFT evaluation against frozen baseline prompts shows improved pass rates with no safety regressions.
- [ ] DPO held-out test preference rate exceeds 60%.
- [ ] Reward model held-out pairwise accuracy exceeds 60% and passes all adversarial probe checks.
- [ ] PPO smoke run log records reward mean, KL, entropy, and response length.

### Hugging Face Deployment
- [ ] Private repositories created for model checkpoints and datasets.
- [ ] Artifacts uploaded with model cards documenting base model, dataset provenance, hyperparameters, and evaluation.
- [ ] Read-back download (`hf download ...`) successfully retrieves and verifies repo contents.

