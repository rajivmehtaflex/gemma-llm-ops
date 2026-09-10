# E2E Test Infra: Gemma Shell Ops RLHF Pipeline

## Test Philosophy
- Opaque-box, requirement-driven verification derived from `ORIGINAL_REQUEST.md`.
- No dependencies on internal training mocks; genuine execution and verification against hardware, Ollama, Unsloth, Hugging Face CLI, and dataset splits.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise + Real-World Pipeline Runs.

## Feature Inventory & Test Coverage
| # | Feature | Requirement Source | Tier 1 | Tier 2 | Tier 3 |
|---|---------|-------------------|:------:|:------:|:------:|
| 1 | L4 GPU & CUDA Verification | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 2 | Python 3.12 Venv & Unsloth Setup | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 3 | Ollama Daemon & Qwen Model Pull | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 4 | Hugging Face CLI & Token Auth | ORIGINAL_REQUEST §R1 | 5 | 5 | ✓ |
| 5 | Model Resolution Recording | Plan Phase A | 5 | 5 | ✓ |
| 6 | Unit Tests in Pipeline | Plan Phase B | 5 | 5 | ✓ |
| 7 | `gen_pairs` Subcommand | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 8 | Format Parity & Length Guard | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 9 | Catastrophic Safety Guard | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 10 | Prompt Leakage Guard | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 11 | `rank_pairs` Subcommand | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 12 | Autonomous Judge Proxy | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 13 | `assemble_pairs` Subcommand | Plan Phase B §2.5 | 5 | 5 | ✓ |
| 14 | `validate_pairs` Subcommand | ORIGINAL_REQUEST §R2 | 5 | 5 | ✓ |
| 15 | SFT Pilot Training | Plan Phase C §2.2 | 5 | 5 | ✓ |
| 16 | Full SFT Training | Plan Phase C §2.3 | 5 | 5 | ✓ |
| 17 | SFT Baseline Evaluation | Plan Phase C §2.4 | 5 | 5 | ✓ |
| 18 | DPO Training | Plan Phase D §3.1 | 5 | 5 | ✓ |
| 19 | DPO Evaluation & Safety | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 20 | Reward Model Training | Plan Phase E §3.3 | 5 | 5 | ✓ |
| 21 | RM Evaluation & Probes | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 22 | PPO Smoke Run | ORIGINAL_REQUEST §R3 | 5 | 5 | ✓ |
| 23 | PPO Educational Run | Plan Phase F §3.5 | 5 | 5 | ✓ |
| 24 | Private HF Repos Creation | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 25 | Model & Dataset Cards | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 26 | Hub Upload & Read-Back | ORIGINAL_REQUEST §R4 | 5 | 5 | ✓ |
| 27 | Capstone Notes & Tracking | Plan Phase H | 5 | 5 | ✓ |
| 28 | Comprehensive Acceptance | ORIGINAL_REQUEST §Acceptance | 5 | 5 | ✓ |

## Test Architecture
- Test runner: `scripts/verify_e2e.py`
- Invocations:
  - `python3 scripts/verify_e2e.py --tier 1` (Component availability and environment readiness)
  - `python3 scripts/verify_e2e.py --tier 2` (Dataset invariants, boundaries, split counts, safety)
  - `python3 scripts/verify_e2e.py --tier 3` (Model adapters, evaluation metrics, probe tests)
  - `python3 scripts/verify_e2e.py --tier 4` (End-to-end integration: Hugging Face read-back and full pipeline verification)
  - `python3 scripts/verify_e2e.py --all` (Runs complete test suite)
- Directory layout:
  - `tests/`
  - `data/`
  - `runs/`

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target |
|---|----------|--------------------|--------|
| 1 | Full Env Bootstrap & Inference | F1, F2, F3, F4, F5 | GPU, venv, Ollama qwen2.5:14b, HF auth |
| 2 | Dataset Synthesis & Quality Audit | F6, F7, F8, F9, F10, F11, F12, F13, F14 | 500 pairs generated, judged, audited, split, validated |
| 3 | SFT Adapter & Evaluation | F15, F16, F17 | SFT adapter saved, pass rate evaluated, 0 safety regression |
| 4 | DPO Preference Optimization | F18, F19 | DPO adapter saved, test preference rate > 60%, 0 regression |
| 5 | Reward Model & Adversarial Defense | F20, F21 | RM adapter saved, accuracy > 60%, 6/6 probe categories pass |
| 6 | PPO Alignment Trajectory | F22, F23 | PPO 100-step smoke run logs reward, KL, entropy, length |
| 7 | Hugging Face Private Publication | F24, F25, F26, F27, F28 | 5 repos created, artifacts uploaded, read-back verified |

## Acceptance Criteria Checklist (from ORIGINAL_REQUEST.md)
- [ ] `nvidia-smi` and PyTorch confirm NVIDIA L4 CUDA acceleration is operational.
- [ ] `ollama list` shows `qwen2.5:14b` loaded and responsive on `localhost:11434`.
- [ ] `hf auth whoami` successfully authenticates with the provided write token.
- [ ] Exactly 500 preference pairs partitioned into `data/preferences/shell_prefs_{train,val,test}.jsonl` (400/50/50).
- [ ] `python3 scripts/sft_pipeline.py validate_pairs` outputs `PAIRS valid`.
- [ ] `python3 scripts/sft_pipeline.py safety_check data/preferences/*.jsonl` completes with 0 violations.
- [ ] Checkpoint adapters saved for `runs/sft-shell`, `runs/dpo-shell`, `runs/rm-shell`, and logs for `runs/ppo/`.
- [ ] SFT evaluation against frozen baseline prompts shows improved pass rates with no safety regressions.
- [ ] DPO held-out test preference rate exceeds 60%.
- [ ] Reward model held-out pairwise accuracy exceeds 60% and passes all adversarial probe checks.
- [ ] PPO smoke run log records reward mean, KL, entropy, and response length.
- [ ] Private repositories created for model checkpoints and datasets.
- [ ] Artifacts uploaded with model cards documenting base model, dataset provenance, hyperparameters, and evaluation.
- [ ] Read-back download (`hf download ...`) successfully retrieves and verifies repo contents.
