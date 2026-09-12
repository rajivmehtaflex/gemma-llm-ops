# Resume-on-New-Machine Training Pipeline

## Summary

Make the runbook resume from the already-completed SFT adapter through DPO, reward-model training, PPO smoke testing, private Hub publication, and final evidence verification. Use one locked `uv` environment, immutable artifacts, structured gates, and automatic service cleanup.

## Implementation Changes

- Update the training environment:
  - Pin a resolver-compatible `mergekit`.
  - Modernize TRL integrations to use `processing_class`.
  - Add import/readiness checks for DPO, reward-model, and PPO APIs.
  - Start Ollama on demand and stop it after judging.

- Consume the completed SFT adapter:
  - Reuse adapter revision `e9dcaa7999ed3128f3798ea976ea935d832a1e20`.
  - Validate its existing manifest and exact base-model identity before DPO.
  - Do not redownload, retrain, or rerun the SFT paired evaluation.

- Implement DPO:
  - Train one deterministic adapter with beta `0.1`, learning rate `5e-6`, sequence limit `1024`, and effective batch size `8`.
  - Emit structured training and evaluation metrics.
  - Validate preference quality and safety.
  - Publish a private DPO repository and verify its immutable revision.

- Implement reward-model validation:
  - Use the current `RewardTrainer` API.
  - Compute actual held-out pairwise accuracy; require `>60%`.
  - Run six explicit safety-ranking probes; all must pass.
  - Emit machine-readable gate evidence and publish/read back a private RM repository.

- Implement PPO smoke:
  - Wire current policy, reference, reward, and value models.
  - Run exactly 100 steps with per-step reward, KL, entropy, response length, and VRAM.
  - If unsupported, record an explicit structured blocker; never fabricate metrics.
  - Keep PPO artifacts local.

- Replace fragile Tier 3 text parsing with structured JSON validation.
- Preserve atomic/resumable writes, path safety, hash/size checks, secret redaction, and ignored large model files.
- Commit only manifests, summaries, metrics, and logs; retain model weights/tokenizers outside Git.

## Test Plan

- Unit-test manifest schema, immutable revision enforcement, hash/size validation, path traversal rejection, and resumable writes.
- Test environment readiness parsing, including current Hugging Face identity output.
- Test DPO/RM/PPO import compatibility and constructor signatures.
- Test completed-SFT manifest validation and base-model identity.
- Test real RM accuracy and six-probe validation, including failure cases.
- Test PPO structured-log validation and explicit-blocker handling.
- Run the full test suite, locked dependency checks, compilation, secret scanning, and end-to-end verifier.
- Verify each private Hub repository by immutable revision and file-hash readback.

## Assumptions

- The existing SFT adapter is reused; no SFT retraining is required.
- DPO, SFT, and RM repositories are private; PPO is not uploaded.
- The authenticated Hugging Face token remains available through the CLI credential store and is never written to files or logs.
- Ollama is a transient judging dependency and is stopped once evaluation completes.

## Task 1 — Environment and TRL Compatibility

- Add a resolver-compatible pinned `mergekit` dependency to the training group and refresh the lockfile.
- Update DPO and reward-model scripts to the installed TRL constructor APIs, preserving deterministic settings and adapter configuration.
- Add readiness/import smoke coverage for DPO, reward-model, and PPO entry points.
- Verify with the focused compatibility tests, `uv lock --check`, and a locked sync.

## Task 2 — Completed SFT Handoff and Structured Tier 3 Validation

- Treat the existing SFT adapter and manifest as complete input evidence.
- Validate its immutable revision, required files, hashes, and exact base-model identity before DPO.
- Remove SFT generation, grading, seed fallback, and promotion logic from this execution.
- Make later Tier 3 DPO, RM, and PPO checks consume structured JSON evidence rather than prose-only claims.
- Add tests for completed-SFT handoff, malformed manifests, and tampered artifacts.

## Task 3 — DPO Training, Evaluation, and Private Publication

- Train the preference adapter with beta `0.1`, learning rate `5e-6`, sequence limit `1024`, and effective batch size `8`.
- Emit reproducible logs, metrics, model-card metadata, and a provenance manifest.
- Evaluate held-out preference quality and safety, then upload to a private Hub repository and verify immutable readback.

## Task 4 — Reward Model Training and Safety Gate

- Train with the current `RewardTrainer` API and save the reward adapter/model artifacts.
- Compute actual held-out pairwise accuracy and require `>60%`.
- Execute six explicit safety-ranking probes, require all six to pass, emit structured gate JSON, and publish/read back a private repository.

## Task 5 — PPO 100-Step Smoke and Evidence

- Implement current TRL policy/reference/reward/value wiring for exactly 100 steps.
- Record per-step reward, KL, entropy, response length, and VRAM in structured JSONL.
- On unsupported runtime/API, write a structured blocker with the concrete cause and no fabricated metrics; keep PPO local.

## Task 6 — End-to-End Verification and Evidence Hygiene

- Run the complete runbook sequence and all verification commands.
- Start Ollama only while judging and stop it at the end.
- Verify private SFT/DPO/RM Hub revisions and file hashes; confirm no credentials appear in logs.
- Commit manifests, gate summaries, metrics, and logs while keeping weights/tokenizers ignored.
