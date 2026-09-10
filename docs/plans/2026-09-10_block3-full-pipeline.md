# Execution Plan: Full RLHF Training Pipeline (Blocks #2-remaining → #3) on L4

**Date:** 2026-09-10 · **Repo:** `github.com/rajivmehtaflex/gemma-llm-ops` (this repo)
**Executor:** an autonomous coding agent (e.g. Antigravity CLI) with SSH access to a fresh NVIDIA L4 (24 GB) Ubuntu VM.
**Training framework (locked):** **Unsloth** — all training stages (SFT §2.2–2.3, DPO §3.1, RM §3.3, PPO §3.4–3.5) use Unsloth-patched QLoRA models. Where Unsloth's notebooks internally delegate to **Hugging Face TRL** trainer classes (DPOTrainer / RewardTrainer / PPOTrainer), that is expected and still the Unsloth workflow (workbook rule: "Unsloth remains the primary workflow"). Ollama is used ONLY for data generation/judging (§2.5), never as a training source. Storage: Hugging Face Hub (checkpoints/datasets) + GitHub (code/small data).
**Curriculum source:** "RLHF Hands-on Workbook — Gemma 4 E4B" (§2.2–§2.5, §3.1–§3.5). Specs quoted in this plan; when this plan and the workbook conflict, **the workbook wins**.

---

## Goal

Take the repo from "SFT dataset ready" to "PPO learning run recorded": build the missing 500 preference pairs (§2.5), run SFT pilot + full SFT (§2.2–§2.4) on the L4, then train DPO (§3.1), Reward Model (§3.3), and PPO smoke + educational run (§3.4–3.5) — checkpointing everything to Hugging Face Hub as we go.

## Current context (what already exists — do NOT redo)

- `data/baseline/` — 15 frozen prompts + graded reports. **Frozen: never train on these prompts; never edit.**
- `data/analysis/` — weakness taxonomy + audit notes. `data/briefs/briefs.json` — 10 briefs (600 examples).
- `data/splits/` — validated 480/60/60 SFT dataset (`shell_sft_{train,val,test}.jsonl`), checksums in `data/splits/SHA256SUMS`.
- `scripts/sft_pipeline.py` — stdlib-only pipeline: `grade | aggregate | safety_check | generate | assemble | validate`. Reuse `call_ollama()`, `is_command_safe()`, `is_leakage()`, `split_records()` — import them, don't copy them.
- **Missing (this plan builds):** `data/shell_preferences.jsonl` (§2.5), SFT/DPO/RM checkpoints, PPO run logs, HF Hub uploads.

## Infrastructure (user allocates; agent verifies)

1. User provides an SSH alias to the L4 VM, e.g. `l4-train` (Lightning.ai or Modal, Ubuntu 22.04/24.04, ≥24 GB VRAM, ≥100 GB disk, ≥32 GB RAM).
2. **Verify before anything:** `ssh l4-train 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'` → expect `NVIDIA L4, 24564 MiB` (or similar). If it fails, STOP and report.

---

## Phase A — VM bootstrap (~20 min)

Run all on the VM. Use `tmux` for anything longer than 2 min.

```bash
ssh l4-train
sudo apt update && sudo apt install -y git curl jq tmux build-essential python3-venv
# uv for python env management
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
# Ollama (for §2.5 pair generation/judging — NOT for training)
curl -fsSL https://ollama.com/install.sh | sh
ollama --version && curl -s localhost:11434/api/version
# Hugging Face CLI
uv tool install "huggingface_hub[cli]"   # provides `hf` and `huggingface-cli`
hf version
```

**Model resolution (5 min, do not skip):** the workbook says "current Unsloth-compatible E4B instruction checkpoint" (~4.5B effective / 8B total, ≤12B hard cap). Check https://docs.unsloth.ai for the current Gemma entry. Default chain:
1. `unsloth/gemma-3n-E4B-it` (matches the workbook's E4B description exactly) — preferred if listed as supported on L4.
2. Fallback: `unsloth/gemma-3-4b-it`.
Record the chosen id in `runs/TRAINING_MODEL.txt`. **Never use an Ollama tag as the training source** (workbook model rule).

**Pull generator model for pair generation:** `ollama pull qwen2.5:14b` (~9 GB, fits 24 GB VRAM beside nothing else — pairs are generated before training starts).

```bash
git clone https://github.com/rajivmehtaflex/gemma-llm-ops.git && cd gemma-llm-ops
uv venv --python 3.12 && source .venv/bin/activate
uv pip install unsloth  # follow docs.unsloth.ai install for CUDA 12.x if the plain name fails
python3 -c "import unsloth; print('unsloth OK')"
git checkout -b block3-training
```

Gate A: `python3 -c "import torch; print(torch.cuda.get_device_name(0))"` → `NVIDIA L4`.

---

## Phase B — §2.5 Preference pairs (3–6 h machine + 1–2 h user) — **BUILT AND RUN BY ANTIGRAVITY ONLY**

> **Ownership rule:** every artifact in this phase (new pipeline subcommands, tests, generated pairs, judge logs, manifests) is authored, executed, and debugged by Antigravity against the repo. No other agent or manual process produces Phase B code or data. The user's ONLY Phase B involvement is the human-ranking gate in step 3 (they answer prompts; they never write code).

Extend `scripts/sft_pipeline.py` with subcommands (TDD: add failing tests to `scripts/test_pipeline.py` first, same style as existing 4 tests):

1. **`gen_pairs`** — for each of 500 NEW prompts (generated like `generate` does, themed from `briefs.json`; must pass `is_leakage()` vs the 15 baseline prompts): ask the generator model **twice** for two candidate scripts. Candidate A prompt enforces the quality bar (shebang, `set -euo pipefail`, quoting, dry-run guards, explanation). Candidate B prompt requests the same task but with the *targeted flaw* (unquoted vars, no error checks, non-idempotent, unguarded destructive op — pick from the weakness taxonomy). **CRITICAL: format must be constant across A/B** — same markdown style, similar length (±30%) — so DPO learns quality, not formatting. Output: `data/preferences/pairs_raw.jsonl` with `{id, prompt, chosen, rejected, flaw_type}` (conversational JSONL per Appendix: prompt as messages, chosen/rejected as assistant messages). Resumable via manifest, same pattern as `generate`.
2. **`rank_pairs`** — judge model (qwen2.5:14b, temp 0.1) scores each pair on the workbook rubric (safety > portability > idempotency > quoting > error_handling > no destructive cmds) and verifies the intended direction (A better). Pairs where the judge disagrees with the intended direction → flagged `FLIPPED`, regenerated. Also run `is_command_safe()` on **both** sides: rejected may contain *teachable* flaws but never catastrophic ones (`rm -rf /`, `mkfs`, raw `dd`) — catastrophic rejects are regenerated, not kept.
3. **`manual_rank`** — interactive CLI that shows the user N sampled pairs (A/B, randomized side) and records human verdicts. **USER GATE — the agent must pause and ask the user.** Workbook requires: **≥100 pairs human-ranked**, **≥50 rubric-assisted labels audited**. Antigravity runs the mechanics and records verdicts; a human provides the verdicts (verdicts only — no code changes by the user or any other agent).
4. **`assemble_pairs`** — dedup (normalized prompt hash), leakage check, safety quarantine, then **split 400/50/50** (seed 42) → `data/preferences/shell_prefs_{train,val,test}.jsonl` + `SHA256SUMS`.
5. **`validate_pairs`** — every record has `prompt`, `chosen`, `rejected`, all non-empty; counts 400/50/50; print `PAIRS valid`.

Gate B: `python3 scripts/sft_pipeline.py validate_pairs` → `PAIRS valid`; `wc -l data/preferences/shell_prefs_*.jsonl` → `400 50 50`; `python3 scripts/sft_pipeline.py safety_check data/preferences/*.jsonl` → 0 violations. Commit: `feat(block2.5): 500 shell preference pairs (400/50/50), judged + human-audited`.

---

## Phase C — SFT on the L4 (§2.2–§2.4, ~2–3 h)

Write `scripts/train_sft.py` (Unsloth + QLoRA). Use `fast_language_model` API; params per workbook §2.3: 4-bit, seq_len 1024, LoRA r=16, alpha=32, lr 2e-4, effective batch 8 (e.g. per-device 2 × grad-accum 4), **1 epoch**, on `data/splits/shell_sft_train.jsonl` (apply the model's chat template — never hand-write Gemma tokens).

- **§2.2 Pilot first:** same script, `--max-samples 50 --output runs/sft-pilot`. Verify: trains without OOM, `trainer.save_model()` + reload produces output. Gate: pilot completes and reload-generates text.
- **§2.3 Full:** `python3 scripts/train_sft.py --output runs/sft-shell`. Record final loss + val loss in `runs/sft-shell/train_log.md`.
- **§2.4 Eval:** generate answers for all 60 `shell_sft_test.jsonl` prompts (temp 0.7, seed 42) → `data/analysis/sft_eval.jsonl`; grade with the existing 7-lens rubric prompt (reuse `RUBRIC_DESCRIPTION` via Ollama) and compare fail-counts vs the Block #1 baseline taxonomy. **Also run the VM's own §1.3 baseline**: the 15 frozen prompts against the *un-finetuned* base checkpoint — this is the honest before/after anchor (mandated by `data/README.md` §4). Gate: SFT fail-rate < baseline fail-rate on the 60 test prompts, no copied training examples (spot-check 10), every suspicious command manually reviewed. Commit adapter + logs: `feat(block2): SFT trained on L4 (runs/sft-shell) + eval vs baseline`.

---

## Phase D — DPO (§3.1, ~1–2 h)

`scripts/train_dpo.py` starting from `runs/sft-shell`: QLoRA 4-bit, **beta 0.1, lr 5e-6, effective batch 8, 1 epoch, seq 1024** on `shell_prefs_train.jsonl`. Memory note: student + frozen reference both resident — on 24 GB this fits without offload.

Eval (workbook checkpoint: "improve preference **without unsafe-command regression**"):
- Held-out pairs (`shell_prefs_test.jsonl`): chosen-vs-rejected preference rate → expect >60%, report exact number.
- Regression probe: the 15 frozen prompts + the safety/adversarial prompts from `is_command_safe` categories → verify no new safety FAILs vs the SFT eval.
Output `runs/dpo-shell` + `runs/dpo-shell/train_log.md`. Commit: `feat(block3): DPO adapter (runs/dpo-shell) + no-regression eval`.

## Phase E — Reward Model (§3.3, ~1 h)

`scripts/train_rm.py`: same base + LoRA, scalar reward head, trained on the same 400 train pairs (standard chosen>rejected pairwise loss). Report in `runs/rm-shell/eval.md`: held-out **pairwise accuracy** (50 val + 50 test pairs), score distribution, validation loss, and the workbook's adversarial probes — unsafe commands, shell injection, hard-coded paths, longer-but-worse scripts. Gate: pairwise accuracy >60% AND the RM ranks a known-safe script above a known-unsafe one on all 6 probe categories. Commit: `feat(block3): reward model (runs/rm-shell) + adversarial probe results`.

## Phase F — PPO smoke + educational run (§3.4–3.5, ~3–5 h)

`scripts/train_ppo.py` from `runs/sft-shell` + `runs/rm-shell`: **seq 512, batch size 1, QLoRA**, KL coefficient from workbook defaults.

- **§3.4 Smoke: exactly 100 steps.** Log per-step: reward mean, KL, entropy, response length, VRAM. If the installed Unsloth build has no PPO path, use TRL `PPOTrainer` under the Unsloth-patched model; if PPO is fundamentally unsupported on this stack, **STOP, record the blocker in `runs/ppo/NOTES.md` and skip to Phase H** — the workbook explicitly honors diagnosed failures as valid learning artifacts.
- **USER GATE before §3.5:** show the user the smoke logs + 10 sample outputs; user says go/no-go.
- **§3.5: ≤300 steps.** Watch: reward rising while sampled quality falls = **reward hacking → stop and record** (workbook checkpoint). KL exploding or length inflating >3× baseline = stop. Output `runs/ppo/` + logs. Commit: `feat(block3): PPO smoke + educational run (runs/ppo) + reward-hacking watch notes`.

---

## Phase G — Hugging Face Hub: checkpoints + datasets (interleave, ~15 min per upload)

**Concept (answer to "convert git repo to HF repo"):** you don't convert — a GitHub repo and an HF repo are two separate git repos serving different content. Convention used here: **code + small data stay on GitHub; weights/checkpoints go to HF *model* repos; the preference dataset goes to an HF *dataset* repo.** (HF repos are git too, so mirroring is possible, but the CLI/LFS route below is the reliable path for large files.)

**One-time auth on the VM:**
```bash
hf auth login          # paste a WRITE-scope token from https://huggingface.co/settings/tokens
hf auth whoami         # expect: rajivmehtaflex (or whichever account the user's token belongs to)
```

**Create repos + upload (run after each phase's gate):**
```bash
# SFT adapter (repeat pattern for dpo-shell, rm-shell with their own repo names)
hf repo create gemma-shell-sft --repo-type model --private
huggingface-cli upload rajivmehtaflex/gemma-shell-sft runs/sft-shell . --repo-type model
# Preference dataset
hf repo create shell-sft-preferences --repo-type dataset --private
huggingface-cli upload rajivmehtaflex/shell-sft-preferences data/preferences . --repo-type dataset
# Optionally the SFT dataset too:
huggingface-cli upload rajivmehtaflex/shell-sft-dataset data/splits . --repo-type dataset
```
**Verify every upload by read-back:** `hf download rajivmehtaflex/gemma-shell-sft README.md --local-dir /tmp/hfcheck && cat /tmp/hfcheck/README.md` (and `huggingface-cli scan-cache` / repo page file listing). A repo is not "uploaded" until its file list on hf.co matches local `ls`.
**Model cards are mandatory:** each checkpoint repo gets a `README.md` (base model id, dataset provenance, hyperparams, eval numbers, safety notes). Write them in `runs/*/model_card.md` before upload.

## Phase H — Wrap-up

`runs/CAPSTONE_NOTES.md`: stage-by-stage table (data → hyperparams → eval numbers → checkpoint HF URL), all negative results recorded honestly, disk/GPU-hours used. Push the `block3-training` branch to GitHub: `git push -u origin block3-training`. Final commit message: `feat(block3): full RLHF chain complete — DPO, RM, PPO + HF artifacts`.

## Standing rules (apply at every step)

- Never execute model-generated shell scripts — generate, lint, store, review. Python verifiers (if any) run sandboxed only.
- The 15 baseline prompts are exam questions: never train on them; every new prompt passes `is_leakage()`.
- Human gates (Phase B ranking, Phase F go/no-go) **require the user** — the agent must stop and ask, never self-approve.
- Long jobs run in `tmux`; checkpoint + manifest writes are atomic (tmp file + rename) so a kill never corrupts state; every phase is resumable.
- Any step that cannot be completed is reported as blocked with its exact error — never fabricated as done.

## Risks / contingencies

| Risk | Mitigation |
|---|---|
| `unsloth/gemma-3n-E4B-it` unsupported on L4 stack | Resolution step in Phase A; fallback `unsloth/gemma-3-4b-it` (still ≤12B rule) |
| Unsloth PPO path missing | TRL PPOTrainer fallback; if still blocked, record + skip (workbook-honored negative result) |
| Judge model disagrees with intended A/B direction too often (flips) | `FLIPPED` regeneration loop; if flip-rate >30%, strengthen the flaw prompt for B and re-run |
| 24 GB tight during DPO (two models) | Both 4-bit; if OOM: halve per-device batch, double grad-accum (workbook: reduce batch before touching the model) |
| HF upload auth/lfs failures | Verify token scope `write` first; retry with `HF_HUB_ENABLE_HF_TRANSFER=0`; never mark uploaded without read-back |
