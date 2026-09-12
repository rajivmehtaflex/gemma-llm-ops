import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_artifact import (  # noqa: E402
    build_artifact_manifest,
    resolve_hub_revision,
    sha256_file,
    validate_artifact_manifest,
)


class TestSFTArtifactManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tmpdir.name) / "adapter"
        self.root.mkdir()
        (self.root / "adapter_config.json").write_text(
            json.dumps(
                {
                    "base_model_name_or_path": "unsloth/gemma-3-4b-it",
                    "peft_type": "LORA",
                    "r": 16,
                }
            ),
            encoding="utf-8",
        )
        (self.root / "adapter_model.safetensors").write_bytes(b"weights")
        (self.root / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_manifest_records_revision_required_files_and_hashes(self):
        manifest = build_artifact_manifest(
            self.root,
            repo_id="rajivmehtapy/gemma-shell-sft",
            revision="a" * 40,
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["repo_id"], "rajivmehtapy/gemma-shell-sft")
        self.assertEqual(manifest["revision"], "a" * 40)
        self.assertEqual(
            manifest["base_model_name_or_path"], "unsloth/gemma-3-4b-it"
        )
        self.assertEqual(
            manifest["files"]["adapter_model.safetensors"]["sha256"],
            sha256_file(self.root / "adapter_model.safetensors"),
        )
        self.assertEqual(
            manifest["required_files"],
            ["adapter_config.json", "adapter_model.safetensors", "tokenizer.json"],
        )
        self.assertEqual(validate_artifact_manifest(manifest, self.root), [])

    def test_manifest_rejects_missing_revision_and_incomplete_adapter(self):
        with self.assertRaises(ValueError):
            build_artifact_manifest(self.root, repo_id="repo", revision="")

        with self.assertRaises(ValueError):
            build_artifact_manifest(self.root, repo_id="repo", revision="main")

        (self.root / "adapter_model.safetensors").unlink()
        with self.assertRaises(ValueError):
            build_artifact_manifest(self.root, repo_id="repo", revision="b" * 40)

    def test_validation_detects_changed_file_and_untrusted_path(self):
        manifest = build_artifact_manifest(self.root, repo_id="repo", revision="c" * 40)
        (self.root / "tokenizer.json").write_text("changed\n", encoding="utf-8")

        errors = validate_artifact_manifest(manifest, self.root)

        self.assertTrue(any("sha-256 mismatch" in error.lower() for error in errors))

    def test_validation_requires_complete_manifest_for_on_disk_artifact(self):
        manifest = build_artifact_manifest(self.root, repo_id="repo", revision="d" * 40)
        manifest["required_files"] = ["adapter_config.json"]

        errors = validate_artifact_manifest(manifest, self.root)

        self.assertTrue(any("required" in error.lower() for error in errors))

    def test_validation_detects_manifest_base_model_tampering(self):
        manifest = build_artifact_manifest(self.root, repo_id="repo", revision="e" * 40)
        manifest["base_model_name_or_path"] = "different/model"

        errors = validate_artifact_manifest(manifest, self.root)

        self.assertTrue(any("base_model" in error.lower() for error in errors))

    def test_hub_revision_is_resolved_to_server_commit_sha(self):
        expected = "f" * 40

        class FakeInfo:
            sha = expected

        class FakeAPI:
            def __init__(self):
                self.calls = []

            def model_info(self, repo_id, revision=None):
                self.calls.append((repo_id, revision))
                return FakeInfo()

        api = FakeAPI()
        revision = resolve_hub_revision(api, "repo", requested_revision="main")

        self.assertEqual(revision, expected)
        self.assertEqual(api.calls, [("repo", "main")])


if __name__ == "__main__":
    unittest.main()
