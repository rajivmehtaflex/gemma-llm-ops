#!/usr/bin/env python3
"""Download an SFT adapter at an immutable Hub revision and write its manifest."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_artifact import (  # noqa: E402
    build_artifact_manifest,
    resolve_hub_revision,
    validate_artifact_manifest,
    write_manifest,
)


def download_adapter(
    repo_id: str,
    adapter_dir: pathlib.Path,
    *,
    requested_revision: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve, download, and validate one exact Hub model revision."""

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    revision = resolve_hub_revision(api, repo_id, requested_revision)
    adapter_dir = pathlib.Path(adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision=revision,
        local_dir=adapter_dir,
    )
    manifest = build_artifact_manifest(
        adapter_dir,
        repo_id=repo_id,
        revision=revision,
    )
    return revision, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="rajivmehtapy/gemma-shell-sft")
    parser.add_argument("--adapter", default="runs/sft-shell")
    parser.add_argument(
        "--manifest", default="runs/sft-shell/evaluation/adapter_manifest.json"
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hub ref to resolve; the manifest always records its commit SHA",
    )
    args = parser.parse_args()

    adapter_dir = pathlib.Path(args.adapter)
    revision, manifest = download_adapter(
        args.repo_id,
        adapter_dir,
        requested_revision=args.revision,
    )
    manifest_path = pathlib.Path(args.manifest)
    errors = validate_artifact_manifest(manifest, adapter_dir)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 2
    write_manifest(manifest_path, manifest)
    print(
        json.dumps(
            {
                "adapter": str(adapter_dir),
                "manifest": str(manifest_path),
                "repo_id": args.repo_id,
                "revision": revision,
                "files": len(manifest["files"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
