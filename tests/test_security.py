import pathlib
import sys
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.check_secrets import find_token_locations


class TestSecretScanner(unittest.TestCase):
    def test_reports_hugging_face_token_without_reporting_safe_placeholder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            safe = root / "safe.md"
            leaked = root / "leaked.md"
            safe.write_text("Authenticate with ${HF_TOKEN}\n", encoding="utf-8")
            leaked.write_text("token=" + "hf_" + "a" * 30 + "\n", encoding="utf-8")

            findings = find_token_locations([safe, leaked])

            self.assertEqual(findings, [(leaked, 1)])

    def test_skips_binary_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            binary = pathlib.Path(tmpdir) / "weights.bin"
            binary.write_bytes(b"\x00hf_" + b"a" * 30)

            self.assertEqual(find_token_locations([binary]), [])


if __name__ == "__main__":
    unittest.main()
