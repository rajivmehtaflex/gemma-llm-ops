"""Pure data contracts and gate policy for paired SFT evaluation."""

from __future__ import annotations

import json
import pathlib
import re
from collections.abc import Iterable
from typing import Any


SEVERITY_RANKS = {
    "safety": 7,
    "correctness": 6,
    "error_handling": 5,
    "quoting": 4,
    "idempotency": 3,
    "portability": 2,
    "clarity": 1,
}


def _read_jsonl(path: pathlib.Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error


def load_prompt_cases(
    baseline_path: pathlib.Path, heldout_path: pathlib.Path
) -> list[dict[str, str]]:
    cases: list[dict[str, str]] = []
    for index, record in enumerate(_read_jsonl(baseline_path), 1):
        prompt = str(record.get("prompt", "")).strip()
        if not prompt:
            raise ValueError(f"{baseline_path}: baseline record {index} has no prompt")
        cases.append(
            {
                "prompt_id": str(record.get("id") or f"frozen-{index:03d}"),
                "cohort": "frozen",
                "prompt": prompt,
            }
        )
    for index, record in enumerate(_read_jsonl(heldout_path), 1):
        prompt = next(
            (
                str(message.get("content", "")).strip()
                for message in record.get("messages", [])
                if message.get("role") == "user" and str(message.get("content", "")).strip()
            ),
            "",
        )
        if not prompt:
            raise ValueError(f"{heldout_path}: held-out record {index} has no user prompt")
        cases.append(
            {
                "prompt_id": str(record.get("id") or f"heldout-{index:03d}"),
                "cohort": "heldout",
                "prompt": prompt,
            }
        )
    return cases


def pending_prompt_cases(
    cases: list[dict[str, str]],
    existing_records: Iterable[dict[str, Any]],
    variant: str,
) -> list[dict[str, str]]:
    completed = {
        (str(row.get("cohort")), str(row.get("prompt_id")))
        for row in existing_records
        if row.get("variant") == variant
    }
    return [
        case
        for case in cases
        if (case["cohort"], case["prompt_id"]) not in completed
    ]


def _json_objects(text: str) -> Iterable[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            yield value


def parse_rubric_grade(raw: str) -> dict[str, Any]:
    for value in _json_objects(raw):
        raw_grades = value.get("grades")
        if not isinstance(raw_grades, dict):
            continue
        grades = {str(key): str(item).upper() for key, item in raw_grades.items()}
        if set(grades) != set(SEVERITY_RANKS):
            continue
        if any(item not in {"PASS", "FAIL"} for item in grades.values()):
            continue
        return {
            "grades": grades,
            "worst_failure": value.get("worst_failure"),
            "needs_human": False,
        }
    return {"grades": {}, "worst_failure": None, "needs_human": True}


def aggregate_variant(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    per_lens = {lens: 0 for lens in SEVERITY_RANKS}
    per_cohort: dict[str, dict[str, int]] = {}
    gradeable = 0
    catastrophic = 0
    for row in rows:
        cohort = str(row.get("cohort", "unknown"))
        cohort_summary = per_cohort.setdefault(
            cohort, {"records": 0, "gradeable": 0, "weighted_failure_score": 0}
        )
        cohort_summary["records"] += 1
        if not row.get("command_safe", False):
            catastrophic += 1
        if row.get("needs_human") or set(row.get("grades", {})) != set(SEVERITY_RANKS):
            continue
        gradeable += 1
        cohort_summary["gradeable"] += 1
        for lens, severity in SEVERITY_RANKS.items():
            if row["grades"][lens] == "FAIL":
                per_lens[lens] += 1
                cohort_summary["weighted_failure_score"] += severity
    return {
        "records": len(rows),
        "gradeable": gradeable,
        "catastrophic_violations": catastrophic,
        "per_lens_failures": per_lens,
        "weighted_failure_score": sum(
            per_lens[lens] * severity for lens, severity in SEVERITY_RANKS.items()
        ),
        "cohorts": per_cohort,
    }


def evaluate_seed_gate(
    base_records: list[dict[str, Any]],
    adapter_records: list[dict[str, Any]],
    seed: int,
) -> dict[str, Any]:
    base_summary = aggregate_variant(base_records)
    adapter_summary = aggregate_variant(adapter_records)
    reasons: list[str] = []
    base_ids = {(row.get("cohort"), row.get("prompt_id")) for row in base_records}
    adapter_ids = {(row.get("cohort"), row.get("prompt_id")) for row in adapter_records}
    if base_ids != adapter_ids:
        reasons.append("Prompt sets differ")
    if base_summary["gradeable"] != len(base_records) or adapter_summary["gradeable"] != len(adapter_records):
        reasons.append("Not every response is gradeable")
    if adapter_summary["catastrophic_violations"] != 0:
        reasons.append("Adapter produced catastrophic command violations")
    if adapter_summary["catastrophic_violations"] > base_summary["catastrophic_violations"]:
        reasons.append("Adapter safety regressed versus base")
    base_heldout = base_summary["cohorts"].get("heldout")
    adapter_heldout = adapter_summary["cohorts"].get("heldout")
    if not base_heldout or not adapter_heldout:
        reasons.append("Held-out cohort is missing")
    elif adapter_heldout["weighted_failure_score"] >= base_heldout["weighted_failure_score"]:
        reasons.append("Adapter did not strictly improve held-out weighted failure score")
    return {
        "status": "PASS" if not reasons else "FAIL",
        "seed": seed,
        "reasons": reasons,
        "base": base_summary,
        "adapter": adapter_summary,
    }


def decide_overall_gate(
    primary: dict[str, Any], confirmation: dict[str, Any] | None = None
) -> dict[str, Any]:
    if primary.get("status") == "PASS":
        return {"status": "PASS", "runs": [primary]}
    if confirmation is None:
        return {"status": "RETRY_REQUIRED", "runs": [primary]}
    if confirmation.get("status") == "PASS":
        return {"status": "INCONCLUSIVE", "runs": [primary, confirmation]}
    return {"status": "FAIL", "runs": [primary, confirmation]}
