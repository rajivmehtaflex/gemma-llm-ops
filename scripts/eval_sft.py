#!/usr/bin/env python3
"""Paired base-versus-SFT evaluation for the Gemma Shell Ops adapter."""

from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_evaluation import (
    SEVERITY_RANKS,
    decide_overall_gate,
    evaluate_seed_gate,
    load_prompt_cases,
    parse_rubric_grade,
    pending_prompt_cases,
)
from scripts.sft_artifact import validate_artifact_manifest
from scripts.sft_pipeline import RUBRIC_DESCRIPTION, call_ollama, is_command_safe


def read_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def write_jsonl_atomic(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, sort_keys=True) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: pathlib.Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(value, destination, indent=2, sort_keys=True)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_text_atomic(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            destination.write(value)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def load_validated_artifact_manifest(
    adapter_dir: pathlib.Path, manifest_path: pathlib.Path
) -> dict[str, Any]:
    """Load the adapter provenance manifest and verify every local file hash."""

    adapter_dir = pathlib.Path(adapter_dir)
    manifest_path = pathlib.Path(manifest_path)
    if not adapter_dir.is_dir():
        raise ValueError(f"SFT adapter directory does not exist: {adapter_dir}")
    if not manifest_path.is_file():
        raise ValueError(f"SFT adapter manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"SFT adapter manifest is unreadable: {error}") from error
    errors = validate_artifact_manifest(manifest, adapter_dir)
    if errors:
        raise ValueError("SFT adapter manifest failed validation: " + "; ".join(errors))
    return manifest


def resolve_base_model(
    requested_model: str | None, artifact_manifest: dict[str, Any]
) -> str:
    """Use the adapter's recorded base model and reject mismatched overrides."""

    manifest_model = artifact_manifest.get("base_model_name_or_path")
    if not isinstance(manifest_model, str) or not manifest_model.strip():
        raise ValueError("SFT adapter manifest lacks base_model_name_or_path")
    manifest_model = manifest_model.strip()
    if requested_model and requested_model.strip() != manifest_model:
        raise ValueError(
            "Base model does not match the adapter manifest: "
            f"requested {requested_model!r}, expected {manifest_model!r}"
        )
    return requested_model.strip() if requested_model else manifest_model


def _load_torch_model(model_name: str, max_seq_length: int):
    import torch
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, torch


def _release_torch_model(model: Any) -> None:
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def generate_variant(
    cases: list[dict[str, str]],
    variant: str,
    model_name: str,
    output_path: pathlib.Path,
    seed: int,
    max_seq_length: int,
    max_new_tokens: int,
    temperature: float,
) -> list[dict[str, Any]]:
    existing = read_jsonl(output_path)
    pending = pending_prompt_cases(cases, existing, variant)
    if not pending:
        return existing

    model, tokenizer, torch = _load_torch_model(model_name, max_seq_length)
    tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    rows = list(existing)
    try:
        for case in pending:
            prompt_seed = seed + cases.index(case)
            torch.manual_seed(prompt_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(prompt_seed)
            chat = [{"role": "user", "content": case["prompt"]}]
            formatted = tokenizer.apply_chat_template(
                chat,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(formatted, return_tensors="pt").to("cuda")
            started = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                )
            generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
            completion = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            safe, reasons = is_command_safe(completion)
            rows.append(
                {
                    "prompt_id": case["prompt_id"],
                    "cohort": case["cohort"],
                    "variant": variant,
                    "model_name": model_name,
                    "seed": seed,
                    "prompt": case["prompt"],
                    "response": completion,
                    "latency_sec": round(time.perf_counter() - started, 4),
                    "command_safe": safe,
                    "safety_reasons": reasons,
                    "needs_human": True,
                    "grades": {},
                    "worst_failure": None,
                }
            )
            write_jsonl_atomic(output_path, rows)
    finally:
        _release_torch_model(model)
    return rows


def grade_variant(
    records: list[dict[str, Any]],
    output_path: pathlib.Path,
    judge_model: str,
    judge_host: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows = list(records)
    for index, row in enumerate(rows):
        if (
            not row.get("needs_human")
            and set(row.get("grades", {})) == set(SEVERITY_RANKS)
        ):
            continue
        eval_prompt = f"PROMPT:\n{row['prompt']}\n\nRESPONSE:\n{row['response']}"
        raw_grade = call_ollama(
            prompt=eval_prompt,
            system_prompt=RUBRIC_DESCRIPTION,
            model=judge_model,
            host=judge_host,
            temperature=0.1,
            num_predict=512,
            seed=seed,
        )
        grade = parse_rubric_grade(raw_grade)
        row.update(grade)
        row["judge_model"] = judge_model
        row["judge_seed"] = seed
        row["judge_index"] = index
        write_jsonl_atomic(output_path, rows)
    return rows


def format_gate_markdown(gate: dict[str, Any]) -> str:
    lines = [
        "# SFT Evaluation Gate",
        "",
        f"- **Status:** `{gate['status']}`",
        f"- **Generated:** `{gate['generated_at']}`",
        f"- **Adapter revision:** `{gate.get('artifact_manifest', {}).get('revision', 'unknown')}`",
        "",
    ]
    for run in gate["runs"]:
        lines.extend(
            [
                f"## Seed {run['seed']}: {run['status']}",
                "",
                f"- **Reasons:** {', '.join(run['reasons']) or 'none'}",
                f"- **Base weighted held-out score:** {run['base']['cohorts'].get('heldout', {}).get('weighted_failure_score', 'N/A')}",
                f"- **Adapter weighted held-out score:** {run['adapter']['cohorts'].get('heldout', {}).get('weighted_failure_score', 'N/A')}",
                f"- **Adapter catastrophic violations:** {run['adapter']['catastrophic_violations']}",
                "",
            ]
        )
    return "\n".join(lines)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    baseline_path = pathlib.Path(args.baseline_data)
    heldout_path = pathlib.Path(args.test_data)
    output_root = pathlib.Path(args.output_dir)
    seed_dir = output_root / f"seed-{args.seed}"
    artifact_manifest_path = pathlib.Path(
        getattr(args, "artifact_manifest", "runs/sft-shell/evaluation/adapter_manifest.json")
    )
    artifact_manifest = load_validated_artifact_manifest(
        pathlib.Path(args.adapter), artifact_manifest_path
    )
    base_model = resolve_base_model(getattr(args, "base_model", None), artifact_manifest)
    cases = load_prompt_cases(baseline_path, heldout_path)
    if args.max_prompts is not None:
        cases = cases[: args.max_prompts]

    base_path = seed_dir / "base_records.jsonl"
    adapter_path = seed_dir / "adapter_records.jsonl"
    base_records = generate_variant(
        cases,
        "base",
        base_model,
        base_path,
        args.seed,
        args.max_seq_length,
        args.max_new_tokens,
        args.temperature,
    )
    adapter_records = generate_variant(
        cases,
        "adapter",
        args.adapter,
        adapter_path,
        args.seed,
        args.max_seq_length,
        args.max_new_tokens,
        args.temperature,
    )

    base_records = grade_variant(
        base_records,
        base_path,
        args.judge_model,
        args.judge_host,
        args.seed,
    )
    adapter_records = grade_variant(
        adapter_records,
        adapter_path,
        args.judge_model,
        args.judge_host,
        args.seed,
    )
    seed_gate = evaluate_seed_gate(base_records, adapter_records, args.seed)
    write_json_atomic(seed_dir / "gate.json", seed_gate)

    primary_path = output_root / "seed-42" / "gate.json"
    primary = json.loads(primary_path.read_text(encoding="utf-8")) if primary_path.is_file() else seed_gate
    confirmation = None if args.seed == 42 else seed_gate
    overall = decide_overall_gate(primary, confirmation)
    gate = {
        **overall,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_model": base_model,
        "adapter": args.adapter,
        "artifact_manifest": {
            "path": str(artifact_manifest_path),
            "repo_id": artifact_manifest["repo_id"],
            "revision": artifact_manifest["revision"],
            "files": artifact_manifest["files"],
        },
        "judge_model": args.judge_model,
        "generation": {
            "max_seq_length": args.max_seq_length,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        },
    }
    write_json_atomic(output_root / "gate.json", gate)
    write_text_atomic(output_root / "gate.md", format_gate_markdown(gate))
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model override; defaults to the verified adapter manifest",
    )
    parser.add_argument("--adapter", default="runs/sft-shell")
    parser.add_argument(
        "--artifact-manifest",
        default="runs/sft-shell/evaluation/adapter_manifest.json",
        help="Integrity manifest produced by download_sft_artifact.py",
    )
    parser.add_argument("--baseline-data", default="data/baseline/shell_prompts.jsonl")
    parser.add_argument("--test-data", default="data/splits/shell_sft_test.jsonl")
    parser.add_argument("--output-dir", default="runs/sft-shell/evaluation")
    parser.add_argument("--judge-model", default="qwen2.5:14b")
    parser.add_argument("--judge-host", default="http://localhost:11434")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-prompts", type=int, default=None)
    args = parser.parse_args()
    gate = run_evaluation(args)
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
