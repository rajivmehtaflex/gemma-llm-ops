"""Read-only readiness checks for the Gemma Shell Ops training machine."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any


def parse_ollama_models(body: str, expected_model: str) -> bool:
    try:
        models = json.loads(body).get("models", [])
    except (TypeError, json.JSONDecodeError):
        return False
    return any(isinstance(model, dict) and model.get("name") == expected_model for model in models)


def parse_hf_identity(output: str, expected_user: str) -> bool:
    expected = expected_user.strip().lower()
    first_line = next((line.strip() for line in output.splitlines() if line.strip()), "")
    if first_line.lower() == expected:
        return True
    if first_line.lower().startswith("username:"):
        return first_line.split(":", 1)[1].strip().lower() == expected
    match = re.search(r"(?:^|\s)user=([^\s]+)", first_line, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower() == expected
    try:
        payload = json.loads(output)
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict):
        return False
    identity = payload.get("name", payload.get("username", payload.get("user", "")))
    return isinstance(identity, str) and identity.strip().lower() == expected


def _command(argv: list[str], timeout: int = 30) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "command unavailable or timed out"
    return result.returncode == 0, result.stdout.strip()


def _http_get(url: str, timeout: int = 10) -> tuple[bool, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200, response.read().decode("utf-8")
    except (OSError, urllib.error.URLError):
        return False, "HTTP endpoint unavailable"


def check_environment(
    repo_root: pathlib.Path,
    ollama_host: str = "http://localhost:11434",
    expected_model: str = "qwen2.5:14b",
    expected_hf_user: str = "rajivmehtapy",
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    smi_ok, smi_output = _command(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    checks.append({"name": "gpu", "ok": smi_ok and "L4" in smi_output, "details": smi_output if smi_ok else "nvidia-smi failed"})

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
        device = torch.cuda.get_device_name(0) if cuda_ok else ""
        torch_details = f"torch={torch.__version__}, device={device}"
        checks.append({"name": "cuda", "ok": cuda_ok and "L4" in device, "details": torch_details})
        checks.append({"name": "bf16", "ok": cuda_ok and bool(torch.cuda.is_bf16_supported()), "details": "BF16 supported" if cuda_ok and torch.cuda.is_bf16_supported() else "BF16 unavailable"})
    except Exception as error:
        checks.append({"name": "cuda", "ok": False, "details": f"PyTorch import failed: {error.__class__.__name__}"})

    for package in ("unsloth", "datasets", "trl", "transformers", "peft"):
        try:
            version = importlib.metadata.version(package)
            checks.append({"name": f"package:{package}", "ok": True, "details": version})
        except importlib.metadata.PackageNotFoundError:
            checks.append({"name": f"package:{package}", "ok": False, "details": "not installed"})

    version_ok, version_body = _http_get(f"{ollama_host.rstrip('/')}/api/version")
    checks.append({"name": "ollama", "ok": version_ok, "details": version_body if version_ok else "Ollama API unavailable"})
    tags_ok, tags_body = _http_get(f"{ollama_host.rstrip('/')}/api/tags")
    checks.append({"name": f"ollama-model:{expected_model}", "ok": tags_ok and parse_ollama_models(tags_body, expected_model), "details": "model present" if tags_ok and parse_ollama_models(tags_body, expected_model) else "model absent"})

    hf_ok, hf_output = _command(["hf", "auth", "whoami"])
    checks.append({"name": f"hf-user:{expected_hf_user}", "ok": hf_ok and parse_hf_identity(hf_output, expected_hf_user), "details": "authenticated identity verified" if hf_ok and parse_hf_identity(hf_output, expected_hf_user) else "identity unavailable or unexpected"})

    free_bytes = shutil.disk_usage(repo_root).free
    checks.append({"name": "disk", "ok": free_bytes >= 20 * 1024**3, "details": f"{free_bytes / 1024**3:.1f} GiB free"})

    return {
        "python": sys.version.split()[0],
        "repo_root": str(repo_root.resolve()),
        "checks": checks,
        "all_passed": all(check["ok"] for check in checks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="qwen2.5:14b")
    parser.add_argument("--hf-user", default="rajivmehtapy")
    args = parser.parse_args()
    report = check_environment(args.repo_root, args.ollama_host, args.model, args.hf_user)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
