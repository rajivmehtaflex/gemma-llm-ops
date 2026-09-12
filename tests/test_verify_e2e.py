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


class TestStructuredTier3Evidence(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = pathlib.Path(self.tmpdir.name)
        self.runs_dir = self.tmppath / "runs"
        self.runs_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_dpo_accepts_structured_passing_evidence(self):
        dpo_dir = self.runs_dir / "dpo-shell"
        dpo_dir.mkdir(parents=True)
        (dpo_dir / "eval_results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "preference_rate": 0.645,
                    "safety_regressions": 0,
                }
            ),
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_dpo_evaluation()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("64.5%", res.message)

    def test_dpo_rejects_prose_only_claims(self):
        dpo_dir = self.runs_dir / "dpo-shell"
        dpo_dir.mkdir(parents=True)
        (dpo_dir / "eval.md").write_text(
            "# DPO Evaluation\n- Preference rate: 99%\n- Safety regressions: 0\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_dpo_evaluation()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("structured", res.message.lower())

    def test_dpo_rejects_malformed_or_unsafe_evidence(self):
        dpo_dir = self.runs_dir / "dpo-shell"
        dpo_dir.mkdir(parents=True)
        evidence_path = dpo_dir / "eval_results.json"
        cases = [
            ({"schema_version": 1, "status": "PASS"}, "preference_rate"),
            (
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "preference_rate": 0.8,
                    "safety_regressions": 1,
                },
                "safety regressions",
            ),
        ]
        verifier = E2EVerifier(repo_root=self.tmppath)
        for evidence, expected in cases:
            with self.subTest(evidence=evidence):
                evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
                result = verifier.check_tier3_dpo_evaluation()
                self.assertEqual(result.status, CheckStatus.FAIL)
                self.assertIn(expected, result.message.lower())

    def test_rm_accepts_structured_accuracy_and_six_probes(self):
        rm_dir = self.runs_dir / "rm-shell"
        rm_dir.mkdir(parents=True)
        (rm_dir / "probe_results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "pairwise_accuracy": 0.72,
                    "adversarial_probes": [
                        {"name": f"probe-{index}", "passed": True}
                        for index in range(1, 7)
                    ],
                }
            ),
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_reward_model_evaluation()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("72.0%", res.message)

    def test_rm_rejects_prose_only_claims(self):
        rm_dir = self.runs_dir / "rm-shell"
        rm_dir.mkdir(parents=True)
        (rm_dir / "eval.md").write_text(
            "# RM Evaluation\n- Pairwise accuracy: 72.0%\n- Adversarial probes: 5/6\n",
            encoding="utf-8",
        )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_reward_model_evaluation()
        self.assertEqual(res.status, CheckStatus.FAIL)
        self.assertIn("structured", res.message.lower())

    def test_rm_rejects_incomplete_structured_probe_evidence(self):
        rm_dir = self.runs_dir / "rm-shell"
        rm_dir.mkdir(parents=True)
        (rm_dir / "probe_results.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "pairwise_accuracy": 0.72,
                    "adversarial_probes": [
                        {"name": f"probe-{index}", "passed": True}
                        for index in range(1, 6)
                    ],
                }
            ),
            encoding="utf-8",
        )

        result = E2EVerifier(repo_root=self.tmppath).check_tier3_reward_model_evaluation()

        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("six", result.message.lower())

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
                        "response_length": 150,
                        "vram_mb": 12_000,
                    }
                )
                + "\n"
                for step in range(100)
            )

        verifier = E2EVerifier(repo_root=self.tmppath)
        res = verifier.check_tier3_ppo_smoke_run()
        self.assertEqual(res.status, CheckStatus.PASS)
        self.assertIn("100 steps", res.message)

    def test_ppo_rejects_invalid_or_incomplete_jsonl_records(self):
        ppo_dir = self.runs_dir / "ppo"
        ppo_dir.mkdir(parents=True)
        smoke_log = ppo_dir / "smoke_log.jsonl"
        rows = [
            {
                "step": step,
                "reward": 0.5,
                "kl": 0.02,
                "entropy": 0.1,
                "response_length": 150,
                "vram_mb": 12_000,
            }
            for step in range(100)
        ]
        del rows[37]["vram_mb"]
        smoke_log.write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )

        result = E2EVerifier(repo_root=self.tmppath).check_tier3_ppo_smoke_run()

        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("record", result.message.lower())

    def test_ppo_accepts_structured_blocker_and_rejects_notes(self):
        ppo_dir = self.runs_dir / "ppo"
        ppo_dir.mkdir(parents=True)
        (ppo_dir / "NOTES.md").write_text(
            "PPO is blocked by an unsupported trainer API.", encoding="utf-8"
        )
        verifier = E2EVerifier(repo_root=self.tmppath)

        prose_result = verifier.check_tier3_ppo_smoke_run()

        self.assertEqual(prose_result.status, CheckStatus.FAIL)
        self.assertIn("structured", prose_result.message.lower())

        (ppo_dir / "blocker.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "BLOCKED",
                    "stage": "ppo",
                    "reason": "Installed PPOTrainer cannot accept the required value model.",
                }
            ),
            encoding="utf-8",
        )

        blocker_result = verifier.check_tier3_ppo_smoke_run()

        self.assertEqual(blocker_result.status, CheckStatus.PASS)
        self.assertIn("blocker", blocker_result.message.lower())


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
        self.assertIn("SFT adapter artifact manifest not found", res.stdout)


class TestCompletedSFTHandoff(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name)
        self.adapter_path = self.root / "runs" / "sft-shell"
        self.adapter_path.mkdir(parents=True, exist_ok=True)
        (self.adapter_path / "evaluation").mkdir()
        (self.adapter_path / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "base/model"}), encoding="utf-8"
        )
        (self.adapter_path / "adapter_model.safetensors").write_bytes(b"weights")
        (self.adapter_path / "tokenizer.json").write_text("{}", encoding="utf-8")
        self.manifest_path = self.adapter_path / "evaluation" / "adapter_manifest.json"
        write_manifest(
            self.manifest_path,
            build_artifact_manifest(
                self.adapter_path, repo_id="repo", revision="a" * 40
            ),
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_sft_check_accepts_completed_manifest_handoff_without_evaluation(self):
        result = E2EVerifier(repo_root=self.root).check_tier3_sft_evaluation()

        self.assertEqual(result.status, CheckStatus.PASS)
        self.assertIn("handoff", result.message.lower())

    def test_sft_check_rejects_malformed_manifest_fields(self):
        original = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        cases = [
            ({"revision": "main"}, "immutable"),
            ({"required_files": "adapter_config.json"}, "required_files"),
            ({"base_model_name_or_path": "other/model"}, "base_model"),
        ]
        verifier = E2EVerifier(repo_root=self.root)
        for changes, expected in cases:
            with self.subTest(changes=changes):
                manifest = {**original, **changes}
                self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                result = verifier.check_tier3_sft_evaluation()
                self.assertEqual(result.status, CheckStatus.FAIL)
                self.assertIn(expected, result.details.lower())

    def test_sft_check_rejects_tampered_adapter_artifact(self):
        (self.adapter_path / "tokenizer.json").write_text("tampered", encoding="utf-8")

        result = E2EVerifier(repo_root=self.root).check_tier3_sft_evaluation()

        self.assertEqual(result.status, CheckStatus.FAIL)
        self.assertIn("sha-256 mismatch", result.details.lower())


if __name__ == "__main__":
    unittest.main()
