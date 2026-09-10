# Shell Specialization SFT Dataset (Block #2 Deliverable)

This repository contains the complete 600-example Shell Specialization Supervised Fine-Tuning (SFT) dataset generated, validated, and partitioned per the curriculum in `2026-09-10_130854-ollama-block1-sft-dataset.md` and Workbook §2.1–§2.4.

---

## 1. Dataset Overview & Provenance

- **Total Examples:** 600 unique, production-grade Bash engineering records
- **Partition:** 480 Train (`80%`), 60 Validation (`10%`), 60 Test (`10%`)
- **Format:** OpenAI / ChatML-compatible `messages` array:
  - `system`: Production Bash engineering persona with strict POSIX guidelines.
  - `user`: Natural language system administration and infrastructure automation task prompt.
  - `assistant`: Shebang (`#!/usr/bin/env bash`), `set -euo pipefail`, robust quoting, defensive traps, and detailed explanation.
- **Generation Method:**
  - Designed from the 10 curriculum generation briefs in `data/briefs/briefs.json`:
    - 4 weakness-targeted briefs (300 examples) derived directly from Block #1 baseline grading (`error-handling`, `quoting-safety`, `idempotency-guards`, `destructive-dryrun`).
    - 6 core coverage briefs (300 examples) across production Linux operations (`files`, `logs`, `backups`, `processes`, `text`, `automation`).
  - Synthesized via modular builders in `scripts/data_builders/` and processed through `scripts/sft_pipeline.py`.
- **Zero-Leakage Guarantee:**
  - 100% disjoint from the 15 frozen baseline evaluation prompts in `data/baseline/shell_prompts.jsonl`.
  - Enforced by programmatic normalization and SHA-256 collision hashing in `sft_pipeline.py assemble`.

---

## 2. Directory Structure & Checksums

```
data/
├── README.md                           # This handoff and provenance guide
├── baseline/
│   ├── shell_prompts.jsonl             # 15 frozen prompts for baseline evaluation (§1.3)
│   ├── baseline_reports.jsonl          # Evaluated reports from local gemma3:4b baseline
│   └── SHA256SUMS                      # Frozen baseline checksums
├── analysis/
│   ├── taxonomy.json                   # Weakness frequency taxonomy
│   ├── weakness_taxonomy.md            # Weakness analysis report
│   └── audit_notes.md                  # Comprehensive §2.4 human audit notes
├── briefs/
│   └── briefs.json                     # 10 generation briefs (600 examples planned)
├── generated/                          # Individual brief datasets
│   ├── weakness-error-handling.jsonl   # 75 examples
│   ├── weakness-quoting-safety.jsonl   # 75 examples
│   ├── weakness-idempotency-guards.jsonl # 87 examples (75 allocated to clean pool)
│   ├── weakness-destructive-dryrun.jsonl # 75 examples
│   ├── coverage-files.jsonl            # 50 examples
│   ├── coverage-logs.jsonl             # 50 examples
│   ├── coverage-backups.jsonl          # 50 examples
│   ├── coverage-processes.jsonl        # 50 examples
│   ├── coverage-text.jsonl             # 50 examples
│   ├── coverage-automation.jsonl       # 50 examples
│   ├── quarantine.jsonl                # 0 quarantined (clean pass)
│   └── manifest.json                   # Generation batch manifest
└── splits/
    ├── shell_sft_train.jsonl           # 480 training examples
    ├── shell_sft_val.jsonl             # 60 validation examples
    ├── shell_sft_test.jsonl            # 60 test examples
    └── SHA256SUMS                      # Checksums for reproducible verification
```

### Verified SHA-256 Checksums (`data/splits/SHA256SUMS`):
```text
ba7e7b4db0a5850f3f9b3241c2d7e885641d3c0a8fb323ec0f87cfe8fcfd419b  data/splits/shell_sft_test.jsonl
0cc8690d4c64d2efdb9ec70d0f8e7c28fdc4d963e447321e99eb81b7304404e6  data/splits/shell_sft_train.jsonl
8badb2e08475b73df7e78eb2ae8393eaa65ce5eb7f6c8a21d6ebb569ade713b6  data/splits/shell_sft_val.jsonl
```

Verify anytime using:
```bash
sha256sum -c data/splits/SHA256SUMS
```

---

## 3. Validation & Quality Gates Passed

All gates defined in Workbook §2.1 and Appendix A.5 pass cleanly:

| Verification Gate | Command | Result |
|---|---|---|
| **Unit Test Suite** | `python3 -m unittest scripts/test_pipeline.py -v` | `Ran 4 tests in 0.030s: OK` |
| **Baseline Completeness** | `jq -r '.id' data/baseline/baseline_reports.jsonl \| wc -l` | `15` / 15 prompts |
| **Appendix A.5 Gate** | `python3 -c "import json,glob; [json.loads(x) for f in glob.glob('data/splits/*.jsonl') for x in open(f)]; print('JSONL valid')"` | `JSONL valid` |
| **Dataset Splits Counts** | `wc -l data/splits/*.jsonl` | `480` train / `60` val / `60` test |
| **Zero Prompt Leakage** | Programmatic normalized collision check vs `shell_prompts.jsonl` | `0 leaks detected` |
| **Quarantine Safety Linter** | `python3 scripts/sft_pipeline.py safety_check data/splits/*.jsonl` | `0 violations` |
| **Human Audit (§2.4)** | Sampled 60 records + inspected all `rm`, `chmod`, `curl`, `ssh` | `Passed, recorded in audit_notes.md` |

---

## 4. Critical Remote VM Training Handoff Notice

> [!IMPORTANT]
> **Before Fine-Tuning Claims on Remote VM (Workbook §2.2 onward):**
> 1. The local Ollama `gemma3:4b-Q4` evaluation conducted here was a proxy baseline used exclusively to uncover weaknesses and direct dataset curation.
> 2. When transferring these dataset splits to the remote GPU training instance (Unsloth / Hugging Face / E4B checkpoint), the remote VM **must execute its own §1.3 baseline run** using the frozen 15 prompts (`data/baseline/shell_prompts.jsonl`) on the un-finetuned base model.
> 3. Only the diff between the VM's un-finetuned checkpoint baseline and the post-SFT LoRA adapter evaluation constitutes scientifically valid proof of fine-tuning gain.
> 4. **Scope Boundary:** Remote VM training (Workbook §2.2 pilot run onward) is out of scope for this deliverable.
