#!/usr/bin/env python3
"""Build and validate provenance manifests for a local SFT adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Any, Iterable


SCHEMA_VERSION = 1
WEIGHT_FILENAMES = ("adapter_model.safetensors", "adapter_model.bin")
TOKENIZER_FILENAMES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "tokenizer_config.json",
)
IMMUTABLE_REVISION = re.compile(r"^[0-9a-fA-F]{40}$")


def sha256_file(path: pathlib.Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_artifact_files(adapter_dir: pathlib.Path) -> Iterable[pathlib.Path]:
    for path in sorted(adapter_dir.rglob("*")):
        relative_parts = path.relative_to(adapter_dir).parts
        # Evaluation evidence and Hugging Face's local download bookkeeping
        # live beside the adapter under runs/sft-shell; neither is part of the
        # Hub snapshot and neither must make its hash drift.
        if relative_parts and relative_parts[0] in {"evaluation", ".cache"}:
            continue
        if path.is_symlink():
            raise ValueError(f"Symlink is not allowed in adapter artifact: {path}")
        if path.is_file():
            yield path


def _relative_file_map(adapter_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    files: dict[str, pathlib.Path] = {}
    for path in _iter_artifact_files(adapter_dir):
        relative = path.relative_to(adapter_dir).as_posix()
        files[relative] = path
    return files


def _read_adapter_config(adapter_dir: pathlib.Path) -> dict[str, Any]:
    config_path = adapter_dir / "adapter_config.json"
    if not config_path.is_file():
        raise ValueError(f"Missing required adapter file: {config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid adapter_config.json: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("adapter_config.json must contain a JSON object")
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("adapter_config.json lacks base_model_name_or_path")
    return config


def _require_immutable_revision(revision: str) -> str:
    value = str(revision).strip()
    if not IMMUTABLE_REVISION.fullmatch(value):
        raise ValueError(
            "revision must be the 40-character immutable Hugging Face commit SHA"
        )
    return value


def resolve_hub_revision(api: Any, repo_id: str, requested_revision: str | None = None) -> str:
    """Resolve a Hub branch/tag/ref to the immutable commit SHA returned by the API."""

    info = api.model_info(repo_id, revision=requested_revision)
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str):
        raise ValueError(f"Hugging Face did not return a commit SHA for {repo_id}")
    return _require_immutable_revision(revision)


def _select_required_files(file_map: dict[str, pathlib.Path]) -> list[str]:
    required = ["adapter_config.json"]
    weight = next((name for name in WEIGHT_FILENAMES if name in file_map), None)
    if weight is None:
        names = ", ".join(WEIGHT_FILENAMES)
        raise ValueError(f"Missing adapter weights; expected one of: {names}")
    required.append(weight)
    tokenizer = next((name for name in TOKENIZER_FILENAMES if name in file_map), None)
    if tokenizer is None:
        names = ", ".join(TOKENIZER_FILENAMES)
        raise ValueError(f"Missing tokenizer; expected one of: {names}")
    required.append(tokenizer)
    return required


def build_artifact_manifest(
    adapter_dir: pathlib.Path,
    *,
    repo_id: str,
    revision: str,
) -> dict[str, Any]:
    """Return a deterministic manifest for a complete local adapter snapshot.

    The revision is deliberately restricted to a commit SHA so a mutable Hub
    branch (for example ``main``) cannot be recorded as provenance.
    """

    adapter_dir = pathlib.Path(adapter_dir)
    if not adapter_dir.is_dir():
        raise ValueError(f"Adapter directory does not exist: {adapter_dir}")
    repo_id = str(repo_id).strip()
    if not repo_id:
        raise ValueError("repo_id is required")
    revision = _require_immutable_revision(revision)
    config = _read_adapter_config(adapter_dir)
    file_map = _relative_file_map(adapter_dir)
    required_files = _select_required_files(file_map)
    files = {
        relative: {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for relative, path in sorted(file_map.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "repo_id": repo_id,
        "revision": revision,
        "base_model_name_or_path": config["base_model_name_or_path"].strip(),
        "required_files": required_files,
        "files": files,
    }


def _safe_relative_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value or pathlib.PurePosixPath(value).is_absolute():
        return None
    candidate = pathlib.PurePosixPath(value)
    if any(part in ("", ".", "..") for part in candidate.parts):
        return None
    return candidate.as_posix()


def validate_artifact_manifest(
    manifest: dict[str, Any], adapter_dir: pathlib.Path
) -> list[str]:
    """Return all integrity errors found in ``manifest`` and ``adapter_dir``."""

    errors: list[str] = []
    adapter_dir = pathlib.Path(adapter_dir)
    if not adapter_dir.is_dir():
        return [f"Adapter directory does not exist: {adapter_dir}"]
    if not isinstance(manifest, dict):
        return ["Manifest must contain a JSON object"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"Unsupported manifest schema: {manifest.get('schema_version')!r}")
    if not isinstance(manifest.get("repo_id"), str) or not manifest["repo_id"].strip():
        errors.append("Manifest repo_id is missing")
    revision = manifest.get("revision")
    if not isinstance(revision, str) or not IMMUTABLE_REVISION.fullmatch(revision):
        errors.append("Manifest revision is not an immutable commit SHA")
    if not isinstance(manifest.get("base_model_name_or_path"), str) or not manifest[
        "base_model_name_or_path"
    ].strip():
        errors.append("Manifest base_model_name_or_path is missing")

    required = manifest.get("required_files")
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        errors.append("Manifest required_files is invalid")
        required = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        errors.append("Manifest files is invalid")
        files = {}

    for required_name in required:
        if required_name not in files:
            errors.append(f"Required file is absent from manifest: {required_name}")

    actual_map: dict[str, pathlib.Path] = {}
    try:
        actual_map = _relative_file_map(adapter_dir)
    except ValueError as error:
        errors.append(str(error))

    try:
        actual_config = _read_adapter_config(adapter_dir)
    except ValueError as error:
        errors.append(str(error))
    else:
        actual_base_model = actual_config["base_model_name_or_path"].strip()
        if manifest.get("base_model_name_or_path") != actual_base_model:
            errors.append("Manifest base_model_name_or_path does not match adapter_config.json")
        try:
            actual_required = _select_required_files(actual_map)
        except ValueError as error:
            errors.append(str(error))
        else:
            for required_name in actual_required:
                if required_name not in required:
                    errors.append(f"Manifest omits required file: {required_name}")

    manifest_names: set[str] = set()
    for raw_name, metadata in files.items():
        name = _safe_relative_path(raw_name)
        if name is None:
            errors.append(f"Manifest contains unsafe file path: {raw_name!r}")
            continue
        manifest_names.add(name)
        path = actual_map.get(name)
        if path is None:
            errors.append(f"Manifest file is missing on disk: {name}")
            continue
        if not isinstance(metadata, dict):
            errors.append(f"Manifest metadata is invalid: {name}")
            continue
        expected_hash = metadata.get("sha256")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            errors.append(f"Manifest SHA-256 is invalid: {name}")
        else:
            actual_hash = sha256_file(path)
            if actual_hash.lower() != expected_hash.lower():
                errors.append(f"SHA-256 mismatch: {name}")
        expected_size = metadata.get("size_bytes")
        if not isinstance(expected_size, int) or expected_size < 0:
            errors.append(f"Manifest size is invalid: {name}")
        elif path.stat().st_size != expected_size:
            errors.append(f"Size mismatch: {name}")

    for name in sorted(set(actual_map) - manifest_names):
        errors.append(f"Unrecorded adapter file: {name}")
    return errors


def write_manifest(path: pathlib.Path, manifest: dict[str, Any]) -> None:
    """Atomically write a JSON manifest."""

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as destination:
            json.dump(manifest, destination, indent=2, sort_keys=True)
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="runs/sft-shell")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--manifest", default="runs/sft-shell/evaluation/adapter_manifest.json"
    )
    args = parser.parse_args()
    adapter_dir = pathlib.Path(args.adapter)
    manifest = build_artifact_manifest(
        adapter_dir,
        repo_id=args.repo_id,
        revision=args.revision,
    )
    errors = validate_artifact_manifest(manifest, adapter_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    write_manifest(pathlib.Path(args.manifest), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
