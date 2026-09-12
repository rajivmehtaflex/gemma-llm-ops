import ast
import importlib
import inspect
import subprocess
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestTrainingEntrypointCompatibility(unittest.TestCase):
    def test_training_entrypoints_import_and_display_help(self):
        for script_name in ("train_dpo.py", "train_rm.py", "train_ppo.py"):
            with self.subTest(script=script_name):
                result = subprocess.run(
                    [sys.executable, str(REPO_ROOT / "scripts" / script_name), "--help"],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )

                self.assertEqual(
                    result.returncode,
                    0,
                    msg=f"{script_name} failed its import smoke test:\n{result.stderr}",
                )
                self.assertIn("usage:", result.stdout)

    def test_trainer_calls_match_installed_trl_constructor_signatures(self):
        cases = (
            ("train_dpo.py", "scripts.train_dpo", "DPOTrainer"),
            ("train_rm.py", "scripts.train_rm", "RewardTrainer"),
        )
        for script_name, module_name, call_name in cases:
            with self.subTest(script=script_name):
                try:
                    trainer_class = getattr(importlib.import_module(module_name), call_name)
                except Exception as exc:
                    self.fail(f"{script_name} trainer API is not importable: {exc}")

                source = (REPO_ROOT / "scripts" / script_name).read_text(encoding="utf-8")
                tree = ast.parse(source)
                calls = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == call_name
                ]
                self.assertEqual(len(calls), 1)

                keywords = {keyword.arg for keyword in calls[0].keywords}
                constructor_parameters = set(
                    inspect.signature(trainer_class.__init__).parameters
                )
                self.assertLessEqual(keywords, constructor_parameters)
                self.assertIn("processing_class", keywords)
                self.assertNotIn("tokenizer", keywords)


if __name__ == "__main__":
    unittest.main()
