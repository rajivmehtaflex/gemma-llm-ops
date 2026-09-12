import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_evaluation import (
    aggregate_variant,
    decide_overall_gate,
    evaluate_seed_gate,
    load_prompt_cases,
    parse_rubric_grade,
    pending_prompt_cases,
)
from scripts.eval_sft import load_validated_artifact_manifest
from scripts.sft_artifact import build_artifact_manifest, write_manifest


class TestPromptLoading(unittest.TestCase):
    def test_eval_cli_help_works_from_repository_script_path(self):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "eval_sft.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Paired base-versus-SFT evaluation", result.stdout)

    def test_loads_frozen_and_heldout_prompt_schemas(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            baseline = root / "baseline.jsonl"
            heldout = root / "heldout.jsonl"
            baseline.write_text(json.dumps({"id": "sh-01", "prompt": "List files"}) + "\n")
            heldout.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "system", "content": "Be safe"},
                            {"role": "user", "content": "Rotate a log"},
                            {"role": "assistant", "content": "answer"},
                        ]
                    }
                )
                + "\n"
            )
            cases = load_prompt_cases(baseline, heldout)

            self.assertEqual(
                cases,
                [
                    {"prompt_id": "sh-01", "cohort": "frozen", "prompt": "List files"},
                    {"prompt_id": "heldout-001", "cohort": "heldout", "prompt": "Rotate a log"},
                ],
            )

    def test_resume_only_returns_prompts_without_existing_variant_response(self):
        cases = [
            {"prompt_id": "sh-01", "cohort": "frozen", "prompt": "one"},
            {"prompt_id": "sh-02", "cohort": "frozen", "prompt": "two"},
        ]
        existing = [{"prompt_id": "sh-01", "cohort": "frozen", "variant": "base"}]

        pending = pending_prompt_cases(cases, existing, variant="base")

        self.assertEqual(pending, [cases[1]])

    def test_evaluator_requires_and_validates_adapter_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": "base/model"}),
                encoding="utf-8",
            )
            (adapter / "adapter_model.safetensors").write_bytes(b"weights")
            (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")
            manifest_path = root / "manifest.json"
            write_manifest(
                manifest_path,
                build_artifact_manifest(adapter, repo_id="repo", revision="a" * 40),
            )

            manifest = load_validated_artifact_manifest(adapter, manifest_path)

            self.assertEqual(manifest["revision"], "a" * 40)
            (adapter / "tokenizer.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_validated_artifact_manifest(adapter, manifest_path)


class TestRubricParsing(unittest.TestCase):
    def test_parses_json_from_markdown_and_normalizes_lens_values(self):
        raw = """Result:\n```json\n{"grades":{"safety":"pass","correctness":"FAIL","error_handling":"PASS","quoting":"PASS","idempotency":"PASS","portability":"PASS","clarity":"PASS"},"worst_failure":"correctness"}\n```"""

        grade = parse_rubric_grade(raw)

        self.assertFalse(grade["needs_human"])
        self.assertEqual(grade["grades"]["safety"], "PASS")
        self.assertEqual(grade["grades"]["correctness"], "FAIL")

    def test_marks_malformed_or_incomplete_grade_for_human_review(self):
        grade = parse_rubric_grade('{"grades":{"safety":"PASS"}}')

        self.assertTrue(grade["needs_human"])
        self.assertEqual(grade["grades"], {})


def record(prompt_id, cohort, variant, failures=(), command_safe=True):
    lenses = [
        "safety",
        "correctness",
        "error_handling",
        "quoting",
        "idempotency",
        "portability",
        "clarity",
    ]
    return {
        "prompt_id": prompt_id,
        "cohort": cohort,
        "variant": variant,
        "command_safe": command_safe,
        "needs_human": False,
        "grades": {lens: "FAIL" if lens in failures else "PASS" for lens in lenses},
    }


class TestAggregationAndGate(unittest.TestCase):
    def test_aggregates_weighted_failures_with_fixed_severity(self):
        records = [
            record("h-1", "heldout", "base", failures=("correctness", "clarity")),
            record("f-1", "frozen", "base", failures=("safety",), command_safe=False),
        ]

        summary = aggregate_variant(records)

        self.assertEqual(summary["weighted_failure_score"], 14)
        self.assertEqual(summary["catastrophic_violations"], 1)
        self.assertEqual(summary["per_lens_failures"]["correctness"], 1)

    def test_seed_gate_passes_only_for_complete_safer_improved_adapter(self):
        base = [
            record("h-1", "heldout", "base", failures=("correctness",)),
            record("f-1", "frozen", "base"),
        ]
        adapter = [
            record("h-1", "heldout", "adapter"),
            record("f-1", "frozen", "adapter"),
        ]

        gate = evaluate_seed_gate(base, adapter, seed=42)

        self.assertEqual(gate["status"], "PASS")
        self.assertEqual(gate["reasons"], [])

    def test_seed_gate_rejects_missing_or_ungradeable_records(self):
        base = [record("h-1", "heldout", "base", failures=("correctness",))]
        adapter = [record("other", "heldout", "adapter")]
        adapter[0]["needs_human"] = True
        adapter[0]["grades"] = {}

        gate = evaluate_seed_gate(base, adapter, seed=42)

        self.assertEqual(gate["status"], "FAIL")
        self.assertIn("Prompt sets differ", gate["reasons"])
        self.assertIn("Not every response is gradeable", gate["reasons"])

    def test_failed_primary_and_passing_confirmation_is_inconclusive(self):
        overall = decide_overall_gate(
            {"status": "FAIL", "seed": 42, "reasons": ["not improved"]},
            {"status": "PASS", "seed": 43, "reasons": []},
        )

        self.assertEqual(overall["status"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
