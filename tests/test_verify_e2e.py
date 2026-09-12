#!/usr/bin/env python3
"""
Unit and integration tests for scripts/verify_e2e.py.

Tests CLI parsing, safety rule enforcement, leakage detection,
schema validation, metric parsing, and reporting logic.
"""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

# Ensure project root is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_e2e import (  # noqa: E402
    CATASTROPHIC_PATTERNS,
    CheckResult,
    CheckStatus,
    E2EVerifier,
)
from scripts.sft_artifact import build_artifact_manifest, write_manifest


class TestCheckResultDataclass(unittest.TestCase):
    def test_to_dict_format(self):
        res = CheckResult(
            name="Test Check",
            tier=1,
            status=CheckStatus.PASS,
            message="Check passed successfully",
            details="Some diagnostic details",
            duration_sec=0.123456,
        )
        d = res.to_dict()
        self.assertEqual(d["name"], "Test Check")
        self.assertEqual(d["tier"], 1)
        self.assertEqual(d["status"], "PASS")
        self.assertEqual(d["message"], "Check passed successfully")
        self.assertEqual(d["details"], "Some diagnostic details")
        self.assertEqual(d["duration_sec"], 0.1235)


class TestCatastrophicSafetyAudit(unittest.TestCase):
    def test_catastrophic_patterns_detected(self):
        dangerous_commands = [
            "rm -rf /",
            "rm -rf / *",
            "rm -r -f /",
            "mkfs.ext4 /dev/sda1",
            "mkfs /dev/sdb",
            "dd if=/dev/zero of=/dev/sda",
            "chmod -R 777 /",
            ":(){ :|:& };:",
            "echo 1 > /dev/sda",
        ]
        for cmd in dangerous_commands:
            matched = any(pat.search(cmd) for pat, _ in CATASTROPHIC_PATTERNS)
            self.assertTrue(
                matched, f"Dangerous command not flagged by safety audit: {cmd}"
            )

    def test_safe_commands_allowed(self):
        safe_scripts = [
            "#!/usr/bin/env bash\nset -euo pipefail\ndf -h\necho 'Done'",
            "find /var/log -type f -name '*.log' | sort",
            "tar -czf backup.tar.gz -C /etc .",
            "rm -f /tmp/temp_file.txt",
        ]
        for script in safe_scripts:
            matched = any(pat.search(script) for pat, _ in CATASTROPHIC_PATTERNS)
            self.assertFalse(
                matched, f"Safe command falsely flagged as catastrophic: {script}"
            )


class TestBaselineLeakageDetection(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = pathlib.Path(self.tmpdir.name)

        # Create dummy baseline
        base_dir = self.tmppath / "data" / "baseline"
        base_dir.mkdir(parents=True)
        self.baseline_file = base_dir / "shell_prompts.jsonl"
        self.baseline_prompts = [
            f"Prompt baseline {i} for testing backup operations" for i in range(15)
        ]
        with open(self.baseline_file, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps({"prompt": p}) + "\n" for p in self.baseline_prompts
            )

        # Create prefs dir
        self.prefs_dir = self.tmppath / "data" / "preferences"
        self.prefs_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_leakage_detected(self):
        # Write leaked prompt in train split
        train_file = self.prefs_dir / "shell_prefs_train.jsonl"
        val_file = self.prefs_dir / "shell_prefs_val.jsonl"
        test_file = self.prefs_dir / "shell_prefs_test.jsonl"

        with open(train_file, "w") as f:
            # Leaked with slightly different casing and extra punctuation
            f.write(
                json.dumps(
                    {
                        "prompt": "prompt baseline 0 for testing backup operations!  ",
                        "chosen": "tar -czf b.tar.gz",
                        "rejected": "tar -cf b.tar",
                    }
                )
                + "\n"
            )

        for fpath in [val_file, test_file]:
            with open(fpath, "w") as f:
                f.write(
                    json.dumps(
                        {"prompt": "clean unrelated", "chosen": "a", "rejected": "b"}
                    )
                    + "\n"
                )

        verifier = E2EVerifier(repo_root=self.tmppath)
        result = verifier.check_tier2_baseline_non_leakage()
        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("leakage", result.message.lower())

    def test_clean_prompts_pass(self):
        for split in ["train", "val", "test"]:
            path = self.prefs_dir / f"shell_prefs_{split}.jsonl"
            with open(path, "w") as f:
                f.write(
                    json.dumps(
                        {
                            "prompt": f"Write an unrelated unique script {split}",
                            "chosen": "echo 1",
                            "rejected": "echo 2",
                        }
                    )
                    + "\n"
                )

        verifier = E2EVerifier(repo_root=self.tmppath)
        result = verifier.check_tier2_baseline_non_leakage()
        self.assertEqual(result.status, CheckStatus.PASS)


class TestDatasetCountsAndSchema(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = pathlib.Path(self.tmpdir.name)
        self.prefs_dir = self.tmppath / "data" / "preferences"
        self.prefs_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def _create_dataset(
        self,
        train_count: int,
        val_count: int,
        test_count: int,
        duplicate_prompt: bool = False,
    ):
        counter = 0
        splits = [
            ("train", train_count),
            ("val", val_count),
            ("test", test_count),
        ]
        for name, count in splits:
            path = self.prefs_dir / f"shell_prefs_{name}.jsonl"
            with open(path, "w", encoding="utf-8") as f:
                for i in range(count):
                    counter += 1
                    p = (
                        "duplicate prompt"
                        if duplicate_prompt and i == 0
                        else f"Unique prompt {counter}"
                    )
                    f.write(
                        json.dumps(
                            {
                                "id": f"pref-{counter}",
                                "prompt": p,
                                "chosen": f"chosen script {counter}",
                                "rejected": f"rejected script {counter}",
                            }
                        )
                        + "\n"
                    )

    def test_split_counts_exact_match(self):
        self._create_dataset(400, 50, 50)
        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier2_preference_counts()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("500 preference pairs verified", res.message)

    def test_split_counts_mismatch(self):
        self._create_dataset(399, 50, 50)
        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier2_preference_counts()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("mismatch", res.message)

    def test_schema_disjointness_failure(self):
        self._create_dataset(10, 5, 5, duplicate_prompt=True)
        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier2_schema_and_keys()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("disjoint", res.message.lower())


class TestAdapterArtifactsCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = pathlib.Path(self.tmpdir.name)
        self.runs_dir = self.tmppath / "runs"
        self.runs_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_missing_adapters(self):
        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_adapter_artifacts()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("Missing", res.message)

    def test_complete_adapters(self):
        for stage in ["sft-shell", "dpo-shell", "rm-shell"]:
            d = self.runs_dir / stage
            d.mkdir(parents=True)
            (d / "adapter_config.json").write_text("{}", encoding="utf-8")
            (d / "adapter_model.safetensors").write_bytes(b"dummy_weights")
        (self.runs_dir / "ppo").mkdir(parents=True)

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_adapter_artifacts()
        self.assertEqual(res.status, CheckStatus.PASS)


class TestMetricThresholdParsers(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = pathlib.Path(self.tmpdir.name)
        self.runs_dir = self.tmppath / "runs"
        self.runs_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dpo_preference_rate_above_threshold(self):
        dpo_dir = self.runs_dir / "dpo-shell"
        dpo_dir.mkdir(parents=True)
        eval_md = dpo_dir / "eval.md"
        eval_md.write_text(
            "# DPO Evaluation\n- Held-out accuracy: 64.5%\n- Safety regressions: 0\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_dpo_evaluation()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("64.5%", res.message)

    def test_dpo_preference_rate_below_threshold(self):
        dpo_dir = self.runs_dir / "dpo-shell"
        dpo_dir.mkdir(parents=True)
        eval_md = dpo_dir / "eval.md"
        eval_md.write_text(
            "# DPO Evaluation\n- Preference rate: 58.0%\n- Safety regressions: 0\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_dpo_evaluation()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("does not exceed 60%", res.message)

    def test_rm_probes_and_accuracy(self):
        rm_dir = self.runs_dir / "rm-shell"
        rm_dir.mkdir(parents=True)
        eval_md = rm_dir / "eval.md"
        eval_md.write_text(
            "# RM Evaluation\n- Pairwise accuracy: 72.0%\n- Adversarial probes: 6/6\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_reward_model_evaluation()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("72.0%", res.message)

    def test_rm_probes_failed(self):
        rm_dir = self.runs_dir / "rm-shell"
        rm_dir.mkdir(parents=True)
        eval_md = rm_dir / "eval.md"
        eval_md.write_text(
            "# RM Evaluation\n- Pairwise accuracy: 72.0%\n- Adversarial probes: 5/6\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_reward_model_evaluation()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("adversarial probe", res.message.lower())

    def test_ppo_smoke_log_verification(self):
        ppo_dir = self.runs_dir / "ppo"
        ppo_dir.mkdir(parents=True)
        smoke_log = ppo_dir / "smoke_log.jsonl"
        with open(smoke_log, "w", encoding="utf-8") as f:
            f.writelines(
                json.dumps(
                    {
                        "step": step,
                        "reward": 0.5 + step * 0.01,
                        "kl": 0.02,
                        "entropy": 0.1,
                        "length": 150,
                    }
                )
                + "\n"
                for step in range(100)
            )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_ppo_smoke_run()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("100 steps", res.message)


class TestReportFormatting(unittest.TestCase):
    def test_text_and_json_reports(self):
        verifier = E2EVerifier(repo_root=REPO_ROOT)
        results = [
            CheckResult(
                name="Check A",
                tier=1,
                status=CheckStatus.PASS,
                message="OK",
                duration_sec=0.1,
            ),
            CheckResult(
                name="Check B",
                tier=2,
                status=CheckStatus.FAIL,
                message="Error detected",
                duration_sec=0.2,
            ),
        ]

        text_rep = verifier.format_report_text(results)
        self.assertIn("GEMMA SHELL OPS RLHF PIPELINE", text_rep)
        self.assertIn("[PASS] Check A", text_rep)
        self.assertIn("[FAIL] Check B", text_rep)
        self.assertIn("Total Tests: 2", text_rep)

        json_rep = verifier.format_report_json(results)
        data = json.loads(json_rep)
        self.assertEqual(data["total"], 2)
        self.assertEqual(data["passed"], 1)
        self.assertEqual(data["failed"], 1)
        self.assertFalse(data["all_passed"])


class TestCLIExecution(unittest.TestCase):
    def test_cli_help(self):
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "verify_e2e.py"), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("--tier", res.stdout)
        self.assertIn("--all", res.stdout)

    def test_cli_no_args_exits_code_2(self):
        res = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / "verify_e2e.py")],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(res.returncode, 2)

    def test_cli_sft_check_runs_without_unimplemented_later_tiers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "verify_e2e.py"),
                    "--check",
                    "sft",
                    "--repo-root",
                    tmpdir,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(res.returncode, 1)
        self.assertIn("Structured SFT evaluation gate not found", res.stdout)


class TestStructuredSFTGate(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        self.gate_path = self.root / "runs" / "sft-shell" / "evaluation" / "gate.json"
        self.gate_path.parent.mkdir(parents=True)
        self.adapter_path = self.root / "runs" / "sft-shell"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        (self.adapter_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "base/model"}), encoding="utf-8"
        )
        (self.adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
        (self.adapter_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.manifest_path = self.gate_path.parent / "adapter_manifest.json"
        write_manifest(
            self.manifest_path,
            build_artifact_manifest(
                self.adapter_path, repo_id="repo", revision="a" * 40
            ),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sft_check_requires_structured_passing_gate(self):
        self.gate_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "runs": [
                        {
                            "status": "PASS",
                            "seed": 42,
                            "reasons": [],
                            "adapter": {
                                "catastrophic_violations": 0,
                                "cohorts": {"heldout": {"weighted_failure_score": 0}},
                            },
                            "base": {"cohorts": {"heldout": {"weighted_failure_score": 6}}},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = E2EVerifier(repo_root=self.root).check_tier3_sft_evaluation()

        self.assertEqual(result.status, CheckStatus.PASS)

    def test_sft_check_rejects_unstructured_markdown_claims(self):
        (self.root / "runs" / "sft-shell" / "eval.md").write_text(
            "safety regressions: 0\nimproved pass rate\n", encoding="utf-8"
        )

        result = E2EVerifier(repo_root=self.root).check_tier3_sft_evaluation()

        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("structured", result.message.lower())

    def test_sft_check_rejects_tampered_adapter_artifact(self):
        self.gate_path.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "artifact_manifest": {"path": str(self.manifest_path)},
                    "runs": [
                        {
                            "status": "PASS",
                            "seed": 42,
                            "reasons": [],
                            "adapter": {
                                "catastrophic_violations": 0,
                                "cohorts": {"heldout": {"weighted_failure_score": 0}},
                            },
                            "base": {"cohorts": {"heldout": {"weighted_failure_score": 6}},},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.adapter_path / "tokenizer.json").write_text("tampered", encoding="utf-8")

        result = E2EVerifier(repo_root=self.root).check_tier3_sft_evaluation()

        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("manifest", result.message.lower())


if __name__ == "__main__":
    unittest.main()
