# Repository Guidelines

## Project Structure & Module Organization

`src/gemma_llm_ops/` contains the installable package and console entry point. `scripts/` contains the SFT and preference-data pipelines, training entry points (`train_sft.py`, `train_dpo.py`, `train_rm.py`, `train_ppo.py`), evaluation, artifact validation, and environment checks. Reusable synthetic-data builders are in `scripts/data_builders/`. Tests live in `tests/`; `scripts/test_pipeline.py` covers the original data pipeline. `data/` stores datasets and reports, `docs/` has runbooks and plans, and `runs/` holds training metadata and artifacts. Do not commit credentials or unrequested model weights.

## Build, Test, and Development Commands

- `uv sync` installs base project dependencies in a Python 3.12+ environment; `uv sync --group training` installs the larger ML training stack.
- `uv run gemma-llm-ops` runs the package entry point.
- `python3 -m unittest discover -s tests` runs the `tests/` suite; training compatibility checks require the training dependency group.
- `python3 scripts/test_pipeline.py` runs pipeline unit tests.
- `python3 scripts/verify_e2e.py --tier 2` runs dataset and safety checks; `--all` runs every tier and may require GPU, Ollama, and Hugging Face access. See `TEST_INFRA.md` before running environment-dependent tiers.
- `python3 scripts/sft_pipeline.py --help` lists pipeline subcommands.

## Coding Style & Naming Conventions

Use Python 3.12 and standard four-space indentation. Follow the surrounding code’s type hints, descriptive `snake_case` functions and modules, and `Test...`/`test_...` unittest naming. Keep generated JSONL and dataset builders consistent with the existing schemas. No formatter or linter is configured in `pyproject.toml`; avoid adding one without project-level agreement.

## Testing Guidelines

Tests use Python’s built-in `unittest`; no coverage threshold is configured. Add focused tests for behavior changes and run the relevant unit suite. Run E2E tiers only when their external services and hardware are available; distinguish infrastructure failures from code failures.

## Commit & Pull Request Guidelines

Recent history uses conventional prefixes such as `feat:`, `chore:`, and `docs(plan):`, often with a scope (for example, `feat(block2): ...`). Keep commit subjects concise and action-oriented. PRs should explain the behavior or data change, list validation commands and results, link related issues when applicable, and note any new external-service or hardware requirements.
