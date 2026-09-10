# Project: Full RLHF Training Pipeline (Gemma Shell Ops)

## Architecture
- **Environment & Services**: Debian 12 container running as root, NVIDIA L4 24GB GPU, Python 3.12 virtual environment with Unsloth + PyTorch CUDA, Ollama 0.34.0 daemon hosting `qwen2.5:14b` for generation/judging, Hugging Face CLI authenticated with write token under account `rajivmehtapy`.
- **Data Flow**:
  1. `data/briefs/briefs.json` -> `scripts/sft_pipeline.py gen_pairs` -> `data/preferences/pairs_raw.jsonl`
  2. `data/preferences/pairs_raw.jsonl` -> `scripts/sft_pipeline.py rank_pairs` -> `data/preferences/pairs_judged.jsonl`
  3. `data/preferences/pairs_judged.jsonl` -> `scripts/sft_pipeline.py manual_rank` (LLM judge proxy) -> `data/preferences/audit_verdicts.jsonl`
  4. `data/preferences/pairs_judged.jsonl` -> `scripts/sft_pipeline.py assemble_pairs` -> `data/preferences/shell_prefs_{train,val,test}.jsonl` (400/50/50) + `SHA256SUMS`
  5. `data/splits/shell_sft_train.jsonl` -> `scripts/train_sft.py` -> `runs/sft-shell` (adapter)
  6. `runs/sft-shell` + `data/preferences/shell_prefs_train.jsonl` -> `scripts/train_dpo.py` -> `runs/dpo-shell` (adapter)
  7. `runs/sft-shell` + `data/preferences/shell_prefs_train.jsonl` -> `scripts/train_rm.py` -> `runs/rm-shell` (scalar reward model)
  8. `runs/sft-shell` (actor) + `runs/rm-shell` (critic) -> `scripts/train_ppo.py` -> `runs/ppo/` (logs & checkpoint)
  9. `runs/*` & `data/*` -> Hugging Face CLI -> Hub repos (`rajivmehtapy/*`) -> read-back verification.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | L4 GPU & CUDA Verification | Hardware check verifying NVIDIA L4 GPU operational with 24GB VRAM | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Python 3.12 Venv & Unsloth Setup | Initialize `.venv` with Python 3.12, install Unsloth + CUDA PyTorch | M1 | Plan Phase A |
| 3 | Ollama Daemon & Qwen Model Pull | Verify Ollama service on 11434 and pull `qwen2.5:14b` (~9GB) | M1 | Plan Phase A |
| 4 | Hugging Face CLI & Token Auth | Install `huggingface_hub[cli]` and authenticate account `rajivmehtapy` | M1 | ORIGINAL_REQUEST §R1 |
| 5 | Model Resolution Recording | Resolve and record Unsloth model ID into `runs/TRAINING_MODEL.txt` | M1 | Plan Phase A |
| 6 | Unit Test Extensions in Pipeline | Add unit tests for pair splitting, schema validation, and safety in `test_pipeline.py` | M2 | Plan Phase B |
| 7 | `gen_pairs` Subcommand | Generate 500 contrastive Bash pairs themed across briefs with manifest resumption | M2 | Plan Phase B §2.5 |
| 8 | Format Parity & Length Guard | Enforce identical markdown styling and length within ±30% across chosen and rejected | M2 | Plan Phase B §2.5 |
| 9 | Catastrophic Command Safety Guard | Enforce `is_command_safe()` on chosen & rejected; regenerate any catastrophic rejects | M2 | Plan Phase B §2.5 |
| 10 | Prompt Leakage Guard | Ensure zero overlap with 15 frozen baseline prompts using `is_leakage()` | M2 | Plan Phase B §2.5 |
| 11 | `rank_pairs` Subcommand | LLM judge (`qwen2.5:14b`, temp 0.1) scoring pairs and regenerating flipped pairs | M2 | Plan Phase B §2.5 |
| 12 | Autonomous Judge Proxy (`manual_rank`) | Automated blinded evaluation auditing ≥100 pairs into `audit_verdicts.jsonl` | M2 | ORIGINAL_REQUEST §R2 |
| 13 | `assemble_pairs` Subcommand | Deduplicate, screen, partition 400/50/50, and write checksums | M2 | Plan Phase B §2.5 |
| 14 | `validate_pairs` Subcommand | Schema verification, non-empty fields, 400/50/50 counts, prints `PAIRS valid` | M2 | ORIGINAL_REQUEST §R2 |
| 15 | `train_sft.py` Pilot Mode | SFT pilot run on 50 samples to verify memory and reload generation | M3 | Plan Phase C §2.2 |
| 16 | `train_sft.py` Full Training | Full 1-epoch QLoRA SFT on 480 samples, saving adapter to `runs/sft-shell` | M3 | Plan Phase C §2.3 |
| 17 | SFT Baseline & Test Evaluation | Evaluate against 15 frozen prompts + 60 test prompts, zero safety regressions | M3 | Plan Phase C §2.4 |
| 18 | `train_dpo.py` Training | QLoRA DPO from `runs/sft-shell` on 400 preference pairs, saving to `runs/dpo-shell` | M3 | Plan Phase D §3.1 |
| 19 | DPO Evaluation & Safety Check | Evaluate held-out preference accuracy (>60%) on test split, zero safety regressions | M3 | ORIGINAL_REQUEST §R3 |
| 20 | `train_rm.py` Training | Train scalar reward head with pairwise loss on 400 pairs, saving to `runs/rm-shell` | M3 | Plan Phase E §3.3 |
| 21 | RM Evaluation & Adversarial Probes | Evaluate held-out accuracy (>60%) and pass all 6 adversarial probe categories | M3 | ORIGINAL_REQUEST §R3 |
| 22 | `train_ppo.py` Smoke Run | Exactly 100 steps logging reward mean, KL divergence, entropy, response length | M3 | ORIGINAL_REQUEST §R3 |
| 23 | `train_ppo.py` Educational Run | ≤ 300 steps with reward hacking and length inflation safeguards | M3 | Plan Phase F §3.5 |
| 24 | Private HF Repository Creation | Create 5 private repositories under `rajivmehtapy` (3 model, 2 dataset) | M4 | ORIGINAL_REQUEST §R4 |
| 25 | Model & Dataset Cards | Generate comprehensive `README.md` documentation for all repos | M4 | ORIGINAL_REQUEST §R4 |
| 26 | Hub Upload & Read-Back Verification | Upload artifacts and verify each via `hf download` to local cache | M4 | ORIGINAL_REQUEST §R4 |
| 27 | Capstone Notes & Git Tracking | Consolidate run logs into `runs/CAPSTONE_NOTES.md` and commit/track changes | M4 | Plan Phase H |
| 28 | Comprehensive E2E Verification | End-to-end acceptance testing against all criteria in ORIGINAL_REQUEST.md | M5 | ORIGINAL_REQUEST §Acceptance Criteria |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Environment & Service Setup | Features 1, 2, 3, 4, 5: GPU/CUDA check, Python 3.12 venv, Unsloth, Ollama + qwen2.5:14b, HF auth | none | DONE |
| M2 | Preference Dataset & Quality Auditing | Features 6, 7, 8, 9, 10, 11, 12, 13, 14: pipeline subcommands, test cases, 500 pairs, judge, audit, 400/50/50 split | M1 | IN_PROGRESS |
| M3 | Sequential Model Training & Evaluation | Features 15, 16, 17, 18, 19, 20, 21, 22, 23: SFT pilot & full, DPO, RM with probes, PPO smoke/educational | M2 | PLANNED |
| M4 | Hugging Face Hub Publication & Read-Back | Features 24, 25, 26, 27: 5 private repos under `rajivmehtapy`, model cards, read-back check, capstone notes | M3 | PLANNED |
| M5 | Final E2E Acceptance & Forensics | Feature 28: Complete verification against all acceptance criteria and forensic audit | M4 | PLANNED |

## Interface Contracts
### Scripts ↔ Data
- `data/briefs/briefs.json`: Array of 10 brief objects containing `{id, topic, weakness, count, target_commands}`.
- `data/preferences/pairs_raw.jsonl`: JSONL records with keys `id`, `prompt`, `chosen`, `rejected`, `flaw_type`, `brief_id`.
- `data/preferences/pairs_judged.jsonl`: JSONL records with raw keys plus `judge_score_a`, `judge_score_b`, `verdict`, `status`.
- `data/preferences/audit_verdicts.jsonl`: Audited pairs with `id`, `rubric_lens`, `verdict`, `timestamp`.
- `data/preferences/shell_prefs_{train,val,test}.jsonl`: JSONL records with keys `id`, `prompt`, `chosen`, `rejected`. Counts: exactly 400, 50, 50.
- `runs/TRAINING_MODEL.txt`: Plain text single line containing resolved HF model ID (e.g. `unsloth/gemma-3n-E4B-it` or fallback `unsloth/gemma-3-4b-it`).
- `runs/sft-shell`: LoRA adapter directory compatible with PeftModel/Unsloth.
- `runs/dpo-shell`: LoRA adapter directory from DPO training.
- `runs/rm-shell`: LoRA reward adapter directory with scalar value head.
- `runs/ppo/`: PPO training step logs and final trajectory.

## Code Layout
- `scripts/sft_pipeline.py`: Pipeline CLI subcommands (`grade`, `aggregate`, `safety_check`, `generate`, `assemble`, `validate`, `gen_pairs`, `rank_pairs`, `manual_rank`, `assemble_pairs`, `validate_pairs`).
- `scripts/test_pipeline.py`: Unit tests for pipeline functions and subcommands.
- `scripts/train_sft.py`: Unsloth QLoRA SFT training script (supports pilot and full).
- `scripts/train_dpo.py`: Unsloth + TRL DPOTrainer script.
- `scripts/train_rm.py`: Unsloth + TRL RewardTrainer script with adversarial probes.
- `scripts/train_ppo.py`: TRL PPOTrainer script with smoke and educational runs.
- `data/briefs/`: Source briefs.
- `data/baseline/`: Frozen 15 baseline prompts and reports (IMMUTABLE).
- `data/splits/`: SFT dataset splits (480/60/60).
- `data/preferences/`: Preference dataset splits (400/50/50), raw, judged, and audit logs.
- `runs/`: Model checkpoint adapters, evaluation markdown reports, and logs.
