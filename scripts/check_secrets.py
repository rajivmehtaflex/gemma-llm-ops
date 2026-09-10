#!/usr/bin/env python3
"""Fail when tracked text files contain Hugging Face access tokens."""

from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
from collections.abc import Iterable


HF_TOKEN_PATTERN = re.compile(r"hf_[A-Za-z0-9]{20,}")


def find_token_locations(paths: Iterable[pathlib.Path]) -> list[tuple[pathlib.Path, int]]:
    findings: list[tuple[pathlib.Path, int]] = []
    for path in paths:
        try:
            raw = path.read_bytes()
        except (OSError, UnicodeDecodeError):
            continue
        if b"\0" in raw:
            continue
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(content.splitlines(), 1):
            if HF_TOKEN_PATTERN.search(line):
                findings.append((path, line_number))
    return findings


def tracked_files(repo_root: pathlib.Path) -> list[pathlib.Path]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [repo_root / item.decode() for item in completed.stdout.split(b"\0") if item]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()
    root = args.repo_root.resolve()
    findings = find_token_locations(tracked_files(root))
    for path, line_number in findings:
        print(f"{path.relative_to(root)}:{line_number}: Hugging Face token detected")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
