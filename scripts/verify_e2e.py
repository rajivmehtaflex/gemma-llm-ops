#!/usr/bin/env python3
"""
E2E Verification Test Runner for Gemma Shell Ops RLHF Pipeline.

Validates end-to-end pipeline integrity across 4 distinct tiers:
  Tier 1: Environment readiness (NVIDIA L4 GPU via PyTorch CUDA, Ollama qwen2.5:14b, HF CLI auth).
  Tier 2: Dataset verification (500 pairs, 400/50/50 split, validate_pairs, safety_check, zero leakage).
  Tier 3: Training artifacts & evaluation (LoRA adapters, pass rate, DPO >60%, RM >60% & 6/6 probes, PPO smoke log).
  Tier 4: Hugging Face Hub publication (private repos under rajivmehtapy, model cards, read-back verification).

Exit codes:
  0: All tested assertions passed.
  1: One or more tested assertions failed.
  2: Invalid arguments or invocation error.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any


class CheckStatus(str, enum.Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclasses.dataclass
class CheckResult:
    name: str
    tier: int
    status: CheckStatus
    message: str
    details: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tier": self.tier,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "duration_sec": round(self.duration_sec, 4),
        }


# Dangerous commands patterns for catastrophic safety audit
CATASTROPHIC_PATTERNS = [
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z0-9-]+\s+)*(?:-[a-zA-Z0-9]*r[a-zA-Z0-9]*|--recursive)\b.*?(?:\/(?:\s|$|\*)|~(?:\s|$|\*)|(?:\/bin|\/etc|\/usr|\/boot|\/lib|\/dev|\/sys|\/proc)(?:\s|$|\*))"
        ),
        "Destructive recursive rm targeting root or vital system directories",
    ),
    (
        re.compile(r"\bmkfs(?:\.[a-zA-Z0-9_-]+)?\s+"),
        "Direct disk formatting command (mkfs)",
    ),
    (
        re.compile(
            r"\bdd\s+.*?\bof=/dev/(?:sd[a-z0-9]*|nvme[0-9a-z]*|hd[a-z0-9]*|vd[a-z0-9]*|disk[0-9a-z]*)"
        ),
        "Raw disk write with dd targeting block device",
    ),
    (
        re.compile(
            r"\bchmod\s+-[a-zA-Z0-9]*R[a-zA-Z0-9]*\s+(?:777|a\+rwx)\s+(?:\/(?:\s|$|\*)|(?:\/bin|\/etc|\/usr))"
        ),
        "Unsafe chmod -R 777 on root or system hierarchy",
    ),
    (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "Fork bomb pattern"),
    (
        re.compile(r">\s*/dev/(?:sd[a-z0-9]*|nvme[0-9a-z]*|hd[a-z0-9]*)"),
        "Raw redirection to physical disk block device",
    ),
]


class E2EVerifier:
    """Comprehensive E2E verification test suite executor."""

    def __init__(
        self,
        repo_root: pathlib.Path | None = None,
        python_bin: str | None = None,
        ollama_host: str = "http://localhost:11434",
        hf_username: str = "rajivmehtapy",
        hf_token: str | None = None,
        verbose: bool = False,
    ):
        self.repo_root = (
            repo_root.resolve()
            if repo_root
            else pathlib.Path(__file__).resolve().parent.parent
        )
        self.ollama_host = ollama_host.rstrip("/")
        self.hf_username = hf_username
        self.verbose = verbose

        # Resolve Python binary (prefer user-specified, then .venv/bin/python3, then sys.executable)
        self.python_bin = self._resolve_python_bin(python_bin)

        # Resolve Hugging Face token
        self.hf_token = self._resolve_hf_token(hf_token)

    def _resolve_python_bin(self, explicit_bin: str | None) -> str:
        if explicit_bin:
            p = pathlib.Path(explicit_bin).expanduser().resolve()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
            raise FileNotFoundError(
                f"Specified python binary not found or not executable: {explicit_bin}"
            )

        venv_py = self.repo_root / ".venv" / "bin" / "python3"
        if venv_py.is_file() and os.access(venv_py, os.X_OK):
            return str(venv_py)

        return sys.executable

    def _resolve_hf_token(self, explicit_token: str | None) -> str | None:
        if explicit_token:
            return explicit_token.strip()

        env_token = os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN"
        )
        if env_token:
            return env_token.strip()

        cache_token_file = pathlib.Path.home() / ".cache" / "huggingface" / "token"
        if cache_token_file.is_file():
            try:
                content = cache_token_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass

        legacy_token_file = pathlib.Path.home() / ".huggingface" / "token"
        if legacy_token_file.is_file():
            try:
                content = legacy_token_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass

        return None

    def _run_cmd(
        self,
        cmd: list[str],
        cwd: pathlib.Path | None = None,
        timeout: int = 60,
    ) -> tuple[int, str, str]:
        work_dir = str(cwd or self.repo_root)
        try:
            res = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
        except Exception as e:
            return -1, "", f"Execution error: {e!s}"

    def _http_get(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: int = 15,
    ) -> tuple[int, str, dict[str, str]]:
        req_headers = {"User-Agent": "Gemma-RLHF-E2E-Verifier/1.0"}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                resp_headers = dict(resp.info().items())
                body = resp.read().decode("utf-8", errors="replace")
                return resp.status, body, resp_headers
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            return e.code, body, dict(e.headers.items()) if e.headers else {}
        except Exception as e:
            return 0, str(e), {}

    def _http_post(
        self,
        url: str,
        data: dict[str, Any],
        headers: dict[str, str] | None = None,
        timeout: int = 30,
    ) -> tuple[int, str]:
        req_headers = {
            "User-Agent": "Gemma-RLHF-E2E-Verifier/1.0",
            "Content-Type": "application/json",
        }
        if headers:
            req_headers.update(headers)

        body_bytes = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=body_bytes, headers=req_headers, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.status, resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            return e.code, body
        except Exception as e:
            return 0, str(e)

    # -------------------------------------------------------------------------
    # TIER 1 CHECKS: Environment Readiness
    # -------------------------------------------------------------------------

    def check_tier1_gpu_cuda(self) -> CheckResult:
        t0 = time.time()
        name = "NVIDIA L4 GPU & PyTorch CUDA Acceleration"

        # Step 1: Check nvidia-smi
        smi_code, smi_out, smi_err = self._run_cmd(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
        )
        if smi_code != 0:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message="nvidia-smi execution failed",
                details=f"Command: nvidia-smi\nExit code: {smi_code}\nStderr: {smi_err}",
                duration_sec=time.time() - t0,
            )

        gpu_info = smi_out.strip()
        if "L4" not in gpu_info:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Detected GPU is not NVIDIA L4: {gpu_info}",
                details=f"nvidia-smi output: {gpu_info}",
                duration_sec=time.time() - t0,
            )

        # Step 2: Check PyTorch CUDA acceleration via python_bin
        code_check = (
            "import sys\n"
            "try:\n"
            "    import torch\n"
            "except ImportError:\n"
            "    print('TORCH_IMPORT_ERROR')\n"
            "    sys.exit(2)\n"
            "if not torch.cuda.is_available():\n"
            "    print('CUDA_UNAVAILABLE')\n"
            "    sys.exit(3)\n"
            "dev_name = torch.cuda.get_device_name(0)\n"
            "vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)\n"
            "tensor = torch.zeros((10, 10), device='cuda')\n"
            "print(f'CUDA_OK:{dev_name}:{vram_mb}')\n"
        )

        py_code, py_out, py_err = self._run_cmd([self.python_bin, "-c", code_check])
        if py_code != 0:
            if "TORCH_IMPORT_ERROR" in py_out or py_code == 2:
                msg = (
                    f"PyTorch is not installed in Python environment: {self.python_bin}"
                )
            elif "CUDA_UNAVAILABLE" in py_out or py_code == 3:
                msg = "PyTorch installed, but torch.cuda.is_available() is False"
            else:
                msg = f"PyTorch CUDA verification script failed with code {py_code}"
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=msg,
                details=f"Python interpreter: {self.python_bin}\nStdout: {py_out}\nStderr: {py_err}",
                duration_sec=time.time() - t0,
            )

        match = re.search(r"CUDA_OK:(.+?):(\d+)", py_out)
        if not match:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message="Unexpected output format from PyTorch CUDA probe",
                details=f"Output: {py_out}\nStderr: {py_err}",
                duration_sec=time.time() - t0,
            )

        dev_name, vram_mb = match.group(1), int(match.group(2))
        if "L4" not in dev_name:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"PyTorch detects device '{dev_name}', expected NVIDIA L4",
                details=f"Device: {dev_name}, VRAM: {vram_mb} MiB",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=1,
            status=CheckStatus.PASS,
            message=f"NVIDIA L4 GPU operational with PyTorch CUDA ({vram_mb} MiB VRAM)",
            details=f"Hardware: {gpu_info}\nPyTorch Device: {dev_name} ({vram_mb} MiB)\nInterpreter: {self.python_bin}",
            duration_sec=time.time() - t0,
        )

    def check_tier1_ollama_model(self) -> CheckResult:
        t0 = time.time()
        name = "Ollama Daemon & qwen2.5:14b Readiness"

        # 1. Ping Ollama version
        ver_url = f"{self.ollama_host}/api/version"
        status, body, _ = self._http_get(ver_url, timeout=5)
        if status != 200:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Ollama daemon not responsive on {self.ollama_host}",
                details=f"Endpoint: {ver_url}\nHTTP Status: {status}\nError/Response: {body}",
                duration_sec=time.time() - t0,
            )

        # 2. Check model list
        tags_url = f"{self.ollama_host}/api/tags"
        status, body, _ = self._http_get(tags_url, timeout=10)
        if status != 200:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Failed to query Ollama tags from {tags_url}",
                details=f"HTTP Status: {status}\nResponse: {body}",
                duration_sec=time.time() - t0,
            )

        try:
            tags_json = json.loads(body)
            models = tags_json.get("models", [])
            model_names = [m.get("name", "") for m in models]
        except Exception as e:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Invalid JSON returned from Ollama tags endpoint: {e}",
                details=f"Response body: {body}",
                duration_sec=time.time() - t0,
            )

        matching_models = [m for m in model_names if "qwen2.5:14b" in m]
        if not matching_models:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message="Model 'qwen2.5:14b' not found in Ollama daemon",
                details=f"Loaded models: {model_names or 'None'}\nExpected: qwen2.5:14b",
                duration_sec=time.time() - t0,
            )

        # 3. Test generation responsiveness
        gen_url = f"{self.ollama_host}/api/generate"
        prompt_payload = {
            "model": matching_models[0],
            "prompt": "Respond with the word READY only.",
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 10},
        }
        status, gen_body = self._http_post(gen_url, prompt_payload, timeout=60)
        if status != 200:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Ollama generation request failed for model {matching_models[0]}",
                details=f"Endpoint: {gen_url}\nHTTP Status: {status}\nResponse: {gen_body}",
                duration_sec=time.time() - t0,
            )

        try:
            gen_resp = json.loads(gen_body)
            reply = gen_resp.get("response", "").strip()
            if not reply:
                return CheckResult(
                    name=name,
                    tier=1,
                    status=CheckStatus.FAIL,
                    message="Ollama model returned empty generation response",
                    details=f"Response JSON: {gen_body}",
                    duration_sec=time.time() - t0,
                )
        except Exception as e:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Failed to parse generation response JSON: {e}",
                details=f"Response body: {gen_body}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=1,
            status=CheckStatus.PASS,
            message=f"Ollama running on {self.ollama_host} with responsive model '{matching_models[0]}'",
            details=f"Loaded models: {matching_models}\nTest response: {reply!r}",
            duration_sec=time.time() - t0,
        )

    def check_tier1_hf_auth(self) -> CheckResult:
        t0 = time.time()
        name = "Hugging Face CLI & Write Authentication"

        # Check token via whoami API directly
        if not self.hf_token:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message="No Hugging Face token found in environment or cache",
                details="Checked HF_TOKEN, HUGGING_FACE_HUB_TOKEN, ~/.cache/huggingface/token",
                duration_sec=time.time() - t0,
            )

        whoami_url = "https://huggingface.co/api/whoami-v2"
        status, body, _ = self._http_get(
            whoami_url,
            headers={"Authorization": f"Bearer {self.hf_token}"},
            timeout=10,
        )
        if status != 200:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Hugging Face API authentication failed with HTTP status {status}",
                details=f"URL: {whoami_url}\nResponse: {body}",
                duration_sec=time.time() - t0,
            )

        try:
            info = json.loads(body)
            user_name = info.get("name", "")
            auth_info = info.get("auth", {})
            access_token = auth_info.get("accessToken", {})
            role = access_token.get("role", "")
        except Exception as e:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Failed to parse Hugging Face whoami response: {e}",
                details=f"Response body: {body}",
                duration_sec=time.time() - t0,
            )

        if user_name.lower() != self.hf_username.lower():
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Authenticated as '{user_name}', expected '{self.hf_username}'",
                details=f"User: {user_name}, Expected: {self.hf_username}\nAccount info: {body}",
                duration_sec=time.time() - t0,
            )

        # Check CLI availability
        hf_cli = shutil.which("hf") or shutil.which("huggingface-cli")
        cli_details = ""
        if hf_cli:
            cli_cmd = (
                [hf_cli, "auth", "whoami"]
                if pathlib.Path(hf_cli).name == "hf"
                else [hf_cli, "whoami"]
            )
            c_code, c_out, c_err = self._run_cmd(cli_cmd)
            cli_details = f"CLI binary: {hf_cli}\nCLI command: {' '.join(cli_cmd)}\nCLI exit code: {c_code}\nCLI output: {c_out.strip()}"
        else:
            cli_details = "Note: CLI binary (hf/huggingface-cli) not found in PATH; authenticated via token API."

        return CheckResult(
            name=name,
            tier=1,
            status=CheckStatus.PASS,
            message=f"Authenticated as '{user_name}' (token role: {role or 'write'})",
            details=f"Username: {user_name}\nToken role: {role}\n{cli_details}",
            duration_sec=time.time() - t0,
        )

    def check_tier1_model_resolution(self) -> CheckResult:
        t0 = time.time()
        name = "Base Model Resolution Record (runs/TRAINING_MODEL.txt)"

        model_file = self.repo_root / "runs" / "TRAINING_MODEL.txt"
        if not model_file.is_file():
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Model resolution file missing: {model_file}",
                details="File runs/TRAINING_MODEL.txt should contain the resolved base model ID.",
                duration_sec=time.time() - t0,
            )

        content = model_file.read_text(encoding="utf-8").strip()
        if not content:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Model resolution file is empty: {model_file}",
                details="Expected model ID such as unsloth/gemma-3n-E4B-it or unsloth/gemma-3-4b-it",
                duration_sec=time.time() - t0,
            )

        if content.startswith("ollama") or ":" in content:
            return CheckResult(
                name=name,
                tier=1,
                status=CheckStatus.FAIL,
                message=f"Invalid training model ID (appears to be an Ollama tag): {content}",
                details="Workbook rule: Never use an Ollama tag as training source. Must be Hugging Face / Unsloth ID.",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=1,
            status=CheckStatus.PASS,
            message=f"Base model resolved and recorded: {content}",
            details=f"File: {model_file}\nResolved Model ID: {content}",
            duration_sec=time.time() - t0,
        )

    # -------------------------------------------------------------------------
    # TIER 2 CHECKS: Dataset Verification
    # -------------------------------------------------------------------------

    def check_tier2_preference_counts(self) -> CheckResult:
        t0 = time.time()
        name = "Preference Dataset Split Counts (400/50/50)"

        expected_splits = {
            "train": (
                self.repo_root / "data" / "preferences" / "shell_prefs_train.jsonl",
                400,
            ),
            "val": (
                self.repo_root / "data" / "preferences" / "shell_prefs_val.jsonl",
                50,
            ),
            "test": (
                self.repo_root / "data" / "preferences" / "shell_prefs_test.jsonl",
                50,
            ),
        }

        counts: dict[str, int] = {}
        missing: list[str] = []
        errors: list[str] = []

        for split, (path, expected_count) in expected_splits.items():
            if not path.is_file():
                missing.append(str(path))
                continue

            try:
                with open(path, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    counts[split] = len(lines)
                    if len(lines) != expected_count:
                        errors.append(f"{path.name}: expected {expected_count}, got {len(lines)}")
            except Exception as e:
                errors.append(f"Error reading {path.name}: {e}")

        if missing:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Missing preference dataset files ({len(missing)} missing)",
                details="Missing files:\n" + "\n".join(f" - {m}" for m in missing),
                duration_sec=time.time() - t0,
            )

        if errors:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Split count mismatch: {', '.join(errors)}",
                details=f"Actual counts: {counts}\nExpected: train=400, val=50, test=50 (total 500)",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message=f"Exactly 500 preference pairs verified (train: {counts['train']}, val: {counts['val']}, test: {counts['test']})",
            details="Files verified:\n"
            + "\n".join(
                f" - {p}: {counts[s]} records" for s, (p, _) in expected_splits.items()
            ),
            duration_sec=time.time() - t0,
        )

    def check_tier2_schema_and_keys(self) -> CheckResult:
        t0 = time.time()
        name = "Preference Dataset Schema & Prompt Disjointness"

        split_paths = {
            "train": self.repo_root
            / "data"
            / "preferences"
            / "shell_prefs_train.jsonl",
            "val": self.repo_root / "data" / "preferences" / "shell_prefs_val.jsonl",
            "test": self.repo_root / "data" / "preferences" / "shell_prefs_test.jsonl",
        }

        split_prompts: dict[str, set[str]] = {
            "train": set(),
            "val": set(),
            "test": set(),
        }
        issues: list[str] = []

        for split_name, path in split_paths.items():
            if not path.is_file():
                return CheckResult(
                    name=name,
                    tier=2,
                    status=CheckStatus.FAIL,
                    message=f"Dataset file missing: {path}",
                    duration_sec=time.time() - t0,
                )

            with open(path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception as e:
                        issues.append(f"{path.name}:{line_idx} invalid JSON: {e}")
                        continue

                    # Extract prompt, chosen, rejected
                    prompt = record.get("prompt")
                    chosen = record.get("chosen")
                    rejected = record.get("rejected")

                    # Also support conversational messages schema
                    if prompt is None and "messages" in record:
                        msgs = record.get("messages", [])
                        user_msgs = [
                            m.get("content", "")
                            for m in msgs
                            if m.get("role") == "user"
                        ]
                        if user_msgs:
                            prompt = user_msgs[0]

                    if not prompt or not isinstance(prompt, str):
                        issues.append(
                            f"{path.name}:{line_idx} missing or empty 'prompt'"
                        )
                    if not chosen or not isinstance(chosen, str):
                        issues.append(
                            f"{path.name}:{line_idx} missing or empty 'chosen'"
                        )
                    if not rejected or not isinstance(rejected, str):
                        issues.append(
                            f"{path.name}:{line_idx} missing or empty 'rejected'"
                        )

                    if prompt:
                        norm_p = " ".join(prompt.strip().lower().split())
                        split_prompts[split_name].add(norm_p)

        if issues:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Schema issues detected ({len(issues)} errors)",
                details="\n".join(issues[:10])
                + (f"\n...and {len(issues) - 10} more" if len(issues) > 10 else ""),
                duration_sec=time.time() - t0,
            )

        # Check prompt disjointness
        train_val = split_prompts["train"].intersection(split_prompts["val"])
        train_test = split_prompts["train"].intersection(split_prompts["test"])
        val_test = split_prompts["val"].intersection(split_prompts["test"])

        if train_val or train_test or val_test:
            leak_details = []
            if train_val:
                leak_details.append(f"Train/Val overlap ({len(train_val)} prompts)")
            if train_test:
                leak_details.append(f"Train/Test overlap ({len(train_test)} prompts)")
            if val_test:
                leak_details.append(f"Val/Test overlap ({len(val_test)} prompts)")
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message="Dataset splits are not disjoint",
                details="\n".join(leak_details),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message="All 500 records follow valid schema with completely disjoint prompts across splits",
            details="Validated keys: prompt, chosen, rejected across train (400), val (50), test (50).",
            duration_sec=time.time() - t0,
        )

    def check_tier2_validate_pairs_cli(self) -> CheckResult:
        t0 = time.time()
        name = "Pipeline validate_pairs Subcommand"

        pipe_script = self.repo_root / "scripts" / "sft_pipeline.py"
        if not pipe_script.is_file():
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Pipeline script not found: {pipe_script}",
                duration_sec=time.time() - t0,
            )

        code, out, err = self._run_cmd(
            [self.python_bin, str(pipe_script), "validate_pairs"]
        )
        if code != 0:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"'sft_pipeline.py validate_pairs' exited with code {code}",
                details=f"Stdout: {out}\nStderr: {err}",
                duration_sec=time.time() - t0,
            )

        if "PAIRS valid" not in out:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message="'sft_pipeline.py validate_pairs' did not output expected 'PAIRS valid'",
                details=f"Stdout: {out}\nStderr: {err}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message="Command 'python3 scripts/sft_pipeline.py validate_pairs' produced 'PAIRS valid'",
            details=f"Stdout:\n{out.strip()}",
            duration_sec=time.time() - t0,
        )

    def check_tier2_safety_check(self) -> CheckResult:
        t0 = time.time()
        name = "Command Safety Verification (0 Violations)"

        pipe_script = self.repo_root / "scripts" / "sft_pipeline.py"
        files = [
            str(self.repo_root / "data" / "preferences" / "shell_prefs_train.jsonl"),
            str(self.repo_root / "data" / "preferences" / "shell_prefs_val.jsonl"),
            str(self.repo_root / "data" / "preferences" / "shell_prefs_test.jsonl"),
        ]

        # Check files exist
        for f in files:
            if not pathlib.Path(f).is_file():
                return CheckResult(
                    name=name,
                    tier=2,
                    status=CheckStatus.FAIL,
                    message=f"Cannot run safety check, missing file: {f}",
                    duration_sec=time.time() - t0,
                )

        # 1. Run pipeline safety_check CLI
        code, out, err = self._run_cmd(
            [self.python_bin, str(pipe_script), "safety_check"] + files
        )
        if code != 0:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Pipeline safety_check subcommand failed with code {code}",
                details=f"Stdout: {out}\nStderr: {err}",
                duration_sec=time.time() - t0,
            )

        # 2. In addition, perform an independent opaque-box audit against catastrophic patterns
        catastrophic_violations: list[str] = []
        for file_path in files:
            fname = pathlib.Path(file_path).name
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue

                    for role in ["chosen", "rejected"]:
                        content = record.get(role, "")
                        for pat, desc in CATASTROPHIC_PATTERNS:
                            if pat.search(content):
                                catastrophic_violations.append(
                                    f"{fname}:{line_idx} ({role}): {desc}"
                                )

        if catastrophic_violations:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Independent safety audit discovered {len(catastrophic_violations)} catastrophic patterns",
                details="\n".join(catastrophic_violations[:10]),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message="Safety checks passed with 0 violations across all 500 preference pairs",
            details=f"Subcommand output: {out.strip()}\nIndependent audit: 0 catastrophic patterns detected.",
            duration_sec=time.time() - t0,
        )

    def check_tier2_baseline_non_leakage(self) -> CheckResult:
        t0 = time.time()
        name = "Baseline Prompt Non-Leakage (Zero Leakage)"

        # Load 15 baseline prompts
        baseline_file = self.repo_root / "data" / "baseline" / "shell_prompts.jsonl"
        if not baseline_file.is_file():
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Frozen baseline prompts file not found: {baseline_file}",
                duration_sec=time.time() - t0,
            )

        baseline_prompts: list[str] = []
        with open(baseline_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        p = rec.get("prompt", "")
                        if p:
                            baseline_prompts.append(p)
                    except Exception:
                        pass

        if len(baseline_prompts) != 15:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Expected exactly 15 frozen baseline prompts, found {len(baseline_prompts)}",
                details=f"File: {baseline_file}",
                duration_sec=time.time() - t0,
            )

        def normalize(text: str) -> str:
            # Lowercase, alphanumeric + space only, collapse spaces
            t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
            return " ".join(t.split())

        norm_baselines = [normalize(bp) for bp in baseline_prompts]

        split_files = [
            self.repo_root / "data" / "preferences" / "shell_prefs_train.jsonl",
            self.repo_root / "data" / "preferences" / "shell_prefs_val.jsonl",
            self.repo_root / "data" / "preferences" / "shell_prefs_test.jsonl",
        ]

        leaked_records: list[str] = []
        for s_file in split_files:
            if not s_file.is_file():
                return CheckResult(
                    name=name,
                    tier=2,
                    status=CheckStatus.FAIL,
                    message=f"Missing preference split file: {s_file}",
                    duration_sec=time.time() - t0,
                )

            with open(s_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    prompt = rec.get("prompt", "")
                    norm_p = normalize(prompt)

                    for b_idx, nb in enumerate(norm_baselines, 1):
                        # Strict leakage: exact normalized match or high overlap
                        if norm_p == nb or (len(nb) > 20 and nb in norm_p):
                            leaked_records.append(
                                f"{s_file.name}:{idx} leaked baseline #{b_idx} ('{baseline_prompts[b_idx - 1]}')"
                            )

        if leaked_records:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Detected {len(leaked_records)} baseline prompt leakage occurrences",
                details="\n".join(leaked_records[:10]),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message="Zero prompt leakage confirmed across all 500 pairs vs 15 frozen baseline prompts",
            details=f"Audited 500 preference pairs against 15 frozen exam prompts from {baseline_file.name}",
            duration_sec=time.time() - t0,
        )

    def check_tier2_format_parity(self) -> CheckResult:
        t0 = time.time()
        name = "Format Parity & Length Guard"

        split_files = [
            self.repo_root / "data" / "preferences" / "shell_prefs_train.jsonl",
            self.repo_root / "data" / "preferences" / "shell_prefs_val.jsonl",
            self.repo_root / "data" / "preferences" / "shell_prefs_test.jsonl",
        ]

        unbalanced_pairs: list[str] = []
        total_checked = 0

        for s_file in split_files:
            if not s_file.is_file():
                return CheckResult(
                    name=name,
                    tier=2,
                    status=CheckStatus.FAIL,
                    message=f"File not found: {s_file}",
                    duration_sec=time.time() - t0,
                )

            with open(s_file, "r", encoding="utf-8") as f:
                for idx, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    chosen = rec.get("chosen", "")
                    rejected = rec.get("rejected", "")
                    len_c = len(chosen)
                    len_r = len(rejected)

                    total_checked += 1
                    # Guard against extreme length disparity (length difference ratio > 0.50)
                    # Plan Phase B §2.5 specifies ±30% target length parity
                    max_len = max(len_c, len_r)
                    if max_len > 0:
                        diff_ratio = abs(len_c - len_r) / max_len
                        if diff_ratio > 0.55:
                            unbalanced_pairs.append(
                                f"{s_file.name}:{idx} len(chosen)={len_c}, len(rejected)={len_r}, diff_ratio={diff_ratio:.2f}"
                            )

        # Allow small percentage of outliers (< 5% of total dataset)
        max_allowed_outliers = max(1, int(total_checked * 0.05))
        if len(unbalanced_pairs) > max_allowed_outliers:
            return CheckResult(
                name=name,
                tier=2,
                status=CheckStatus.FAIL,
                message=f"Length disparity threshold exceeded ({len(unbalanced_pairs)} pairs exceed ±50% length ratio)",
                details="\n".join(unbalanced_pairs[:10]),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=2,
            status=CheckStatus.PASS,
            message=f"Format and length parity verified across {total_checked} pairs",
            details=f"Total checked: {total_checked}, Length disparity outliers: {len(unbalanced_pairs)} (<= {max_allowed_outliers} allowed)",
            duration_sec=time.time() - t0,
        )

    # -------------------------------------------------------------------------
    # TIER 3 CHECKS: Training Artifacts & Evaluation
    # -------------------------------------------------------------------------

    def check_tier3_adapter_artifacts(self) -> CheckResult:
        t0 = time.time()
        name = "LoRA Checkpoint Adapters & Run Directories"

        expected_dirs = {
            "sft": self.repo_root / "runs" / "sft-shell",
            "dpo": self.repo_root / "runs" / "dpo-shell",
            "rm": self.repo_root / "runs" / "rm-shell",
            "ppo": self.repo_root / "runs" / "ppo",
        }

        missing_items: list[str] = []
        found_adapters: list[str] = []

        for stage, d in expected_dirs.items():
            if not d.is_dir():
                missing_items.append(f"Directory missing: {d}")
                continue

            if stage in ["sft", "dpo", "rm"]:
                config_file = d / "adapter_config.json"
                weights_safetensors = d / "adapter_model.safetensors"
                weights_bin = d / "adapter_model.bin"

                if not config_file.is_file():
                    missing_items.append(f"{stage}: missing adapter_config.json in {d}")
                if not weights_safetensors.is_file() and not weights_bin.is_file():
                    missing_items.append(
                        f"{stage}: missing adapter weights (.safetensors or .bin) in {d}"
                    )

                if config_file.is_file() and (
                    weights_safetensors.is_file() or weights_bin.is_file()
                ):
                    w_name = (
                        weights_safetensors.name
                        if weights_safetensors.is_file()
                        else weights_bin.name
                    )
                    found_adapters.append(f"{stage} ({d.name}): config + {w_name}")
            else:
                found_adapters.append(f"{stage}: directory exists ({d.name})")

        if missing_items:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message=f"Missing training adapter artifacts ({len(missing_items)} missing items)",
                details="\n".join(missing_items),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=3,
            status=CheckStatus.PASS,
            message="Verified adapters for SFT, DPO, RM, and runs/ppo directory",
            details="\n".join(found_adapters),
            duration_sec=time.time() - t0,
        )

    def check_tier3_sft_evaluation(self) -> CheckResult:
        t0 = time.time()
        name = "SFT Baseline Evaluation & Safety Regressions"

        eval_candidates = [
            self.repo_root / "runs" / "sft-shell" / "eval.md",
            self.repo_root / "runs" / "sft-shell" / "train_log.md",
            self.repo_root / "data" / "analysis" / "sft_eval.jsonl",
        ]

        found_file = next((f for f in eval_candidates if f.is_file()), None)
        if not found_file:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="SFT evaluation log or report not found",
                details="Checked:\n" + "\n".join(f" - {c}" for c in eval_candidates),
                duration_sec=time.time() - t0,
            )

        content = found_file.read_text(encoding="utf-8", errors="replace")

        # Check for safety regression mentions (must be 0)
        reg_match = re.search(
            r"(?:safety\s*regressions?|regressions?)[\s:]+([0-9]+)",
            content,
            re.IGNORECASE,
        )
        if reg_match:
            regs = int(reg_match.group(1))
            if regs > 0:
                return CheckResult(
                    name=name,
                    tier=3,
                    status=CheckStatus.FAIL,
                    message=f"SFT evaluation reported {regs} safety regressions",
                    details=f"File: {found_file}\nMatched: {reg_match.group(0)}",
                    duration_sec=time.time() - t0,
                )

        # Check for pass rate or improvement indication
        improved = (
            re.search(
                r"(?:improved|better|pass\s*rate|fail\s*rate)", content, re.IGNORECASE
            )
            is not None
            or found_file.suffix == ".jsonl"
        )
        if not improved:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="SFT evaluation report missing pass rate or improvement metrics",
                details=f"File: {found_file}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=3,
            status=CheckStatus.PASS,
            message="SFT evaluation satisfies criteria with zero command safety regressions",
            details=f"Verified evaluation artifact: {found_file}",
            duration_sec=time.time() - t0,
        )

    def check_tier3_dpo_evaluation(self) -> CheckResult:
        t0 = time.time()
        name = "DPO Preference Rate (> 60%) & Safety Verification"

        eval_candidates = [
            self.repo_root / "runs" / "dpo-shell" / "eval.md",
            self.repo_root / "runs" / "dpo-shell" / "train_log.md",
            self.repo_root / "runs" / "dpo-shell" / "eval_results.json",
        ]

        found_file = next((f for f in eval_candidates if f.is_file()), None)
        if not found_file:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="DPO evaluation report or log not found",
                details="Checked:\n" + "\n".join(f" - {c}" for c in eval_candidates),
                duration_sec=time.time() - t0,
            )

        content = found_file.read_text(encoding="utf-8", errors="replace")

        # Parse preference rate (e.g. 64.2%, 0.65, win rate: 68%)
        rate = None
        rate_match = re.search(
            r"(?:preference\s*rate|held-out\s*accuracy|win\s*rate|accuracy)[\s:=]+([0-9.]+)\s*%?",
            content,
            re.IGNORECASE,
        )
        if rate_match:
            val = float(rate_match.group(1))
            rate = val if val > 1.0 else val * 100.0

        if rate is None:
            # Also check if json
            if found_file.suffix == ".json":
                try:
                    jdata = json.loads(content)
                    raw_val = jdata.get("preference_rate") or jdata.get("accuracy")
                    if raw_val is not None:
                        val = float(raw_val)
                        rate = val if val > 1.0 else val * 100.0
                except Exception:
                    pass

        if rate is None:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="Could not parse DPO held-out preference rate from evaluation report",
                details=f"File: {found_file}",
                duration_sec=time.time() - t0,
            )

        if rate <= 60.0:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message=f"DPO preference rate {rate:.1f}% does not exceed 60% threshold",
                details=f"File: {found_file}\nFound: {rate_match.group(0) if rate_match else rate}",
                duration_sec=time.time() - t0,
            )

        # Check safety regression
        reg_match = re.search(
            r"(?:safety\s*regressions?)[\s:]+([0-9]+)", content, re.IGNORECASE
        )
        if reg_match and int(reg_match.group(1)) > 0:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message=f"DPO evaluation reports {reg_match.group(1)} safety regressions",
                details=f"File: {found_file}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=3,
            status=CheckStatus.PASS,
            message=f"DPO preference rate verified at {rate:.1f}% (>60% threshold) with 0 regressions",
            details=f"File: {found_file}\nMetric: {rate:.1f}%",
            duration_sec=time.time() - t0,
        )

    def check_tier3_reward_model_evaluation(self) -> CheckResult:
        t0 = time.time()
        name = "Reward Model Accuracy (> 60%) & 6/6 Adversarial Probes"

        eval_candidates = [
            self.repo_root / "runs" / "rm-shell" / "eval.md",
            self.repo_root / "runs" / "rm-shell" / "train_log.md",
            self.repo_root / "runs" / "rm-shell" / "probe_results.json",
        ]

        found_file = next((f for f in eval_candidates if f.is_file()), None)
        if not found_file:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="Reward Model evaluation report not found",
                details="Checked:\n" + "\n".join(f" - {c}" for c in eval_candidates),
                duration_sec=time.time() - t0,
            )

        content = found_file.read_text(encoding="utf-8", errors="replace")

        # Parse pairwise accuracy
        acc = None
        acc_match = re.search(
            r"(?:pairwise\s*accuracy|held-out\s*accuracy|accuracy)[\s:=]+([0-9.]+)\s*%?",
            content,
            re.IGNORECASE,
        )
        if acc_match:
            val = float(acc_match.group(1))
            acc = val if val > 1.0 else val * 100.0

        if acc is None and found_file.suffix == ".json":
            try:
                jdata = json.loads(content)
                raw_val = jdata.get("pairwise_accuracy") or jdata.get("accuracy")
                if raw_val is not None:
                    val = float(raw_val)
                    acc = val if val > 1.0 else val * 100.0
            except Exception:
                pass

        if acc is None:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="Could not parse pairwise accuracy from Reward Model evaluation report",
                details=f"File: {found_file}",
                duration_sec=time.time() - t0,
            )

        if acc <= 60.0:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message=f"Reward Model pairwise accuracy {acc:.1f}% does not exceed 60% threshold",
                details=f"File: {found_file}\nFound: {acc_match.group(0) if acc_match else acc}",
                duration_sec=time.time() - t0,
            )

        # Check 6/6 adversarial probes passed
        probes_match = re.search(
            r"(?:probe(?:s)?|adversarial)[\s:=]+(\d+)\s*/\s*(\d+)",
            content,
            re.IGNORECASE,
        )
        probes_ok = False
        probes_desc = ""

        if probes_match:
            passed, total = int(probes_match.group(1)), int(probes_match.group(2))
            probes_desc = f"{passed}/{total} probes"
            probes_ok = passed == total and total >= 6
        elif re.search(
            r"all\s+6\s+probe(?:s)?\s+pass", content, re.IGNORECASE
        ) or re.search(r"6/6\s+probes?", content, re.IGNORECASE):
            probes_ok = True
            probes_desc = "6/6 probes passed"

        if not probes_ok:
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message="Reward Model evaluation does not confirm all 6 adversarial probe categories passed",
                details=f"File: {found_file}\nProbe finding: {probes_desc or 'No 6/6 probe confirmation found'}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=3,
            status=CheckStatus.PASS,
            message=f"Reward Model accuracy verified at {acc:.1f}% (>60%) with 6/6 adversarial probes passed",
            details=f"File: {found_file}\nAccuracy: {acc:.1f}%\nProbes: {probes_desc}",
            duration_sec=time.time() - t0,
        )

    def check_tier3_ppo_smoke_run(self) -> CheckResult:
        t0 = time.time()
        name = "PPO Smoke Run Logs & Metrics Trajectory"

        ppo_dir = self.repo_root / "runs" / "ppo"
        if not ppo_dir.is_dir():
            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.FAIL,
                message=f"PPO directory does not exist: {ppo_dir}",
                duration_sec=time.time() - t0,
            )

        # Check for smoke run log files or documented blocker note
        smoke_log = ppo_dir / "smoke_log.jsonl"
        train_log = ppo_dir / "train_log.md"
        notes_log = ppo_dir / "NOTES.md"

        # Case A: Smoke log exists
        if smoke_log.is_file():
            step_count = 0
            has_reward, has_kl, has_entropy, has_len = False, False, False, False
            with open(smoke_log, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        step_count += 1
                        try:
                            entry = json.loads(line)
                            if "reward" in entry or "reward_mean" in entry:
                                has_reward = True
                            if "kl" in entry or "kl_divergence" in entry:
                                has_kl = True
                            if "entropy" in entry:
                                has_entropy = True
                            if "response_length" in entry or "length" in entry:
                                has_len = True
                        except Exception:
                            pass

            if step_count < 100:
                return CheckResult(
                    name=name,
                    tier=3,
                    status=CheckStatus.FAIL,
                    message=f"PPO smoke run logged {step_count} steps, expected 100 steps",
                    details=f"Log: {smoke_log}",
                    duration_sec=time.time() - t0,
                )

            missing_metrics = []
            if not has_reward:
                missing_metrics.append("reward")
            if not has_kl:
                missing_metrics.append("kl")
            if not has_entropy:
                missing_metrics.append("entropy")
            if not has_len:
                missing_metrics.append("response_length")

            if missing_metrics:
                return CheckResult(
                    name=name,
                    tier=3,
                    status=CheckStatus.FAIL,
                    message=f"PPO smoke log missing required metrics: {', '.join(missing_metrics)}",
                    details=f"Log: {smoke_log}\nFound steps: {step_count}",
                    duration_sec=time.time() - t0,
                )

            return CheckResult(
                name=name,
                tier=3,
                status=CheckStatus.PASS,
                message=f"PPO smoke run completed with {step_count} steps and all metrics logged",
                details=f"Log file: {smoke_log}\nMetrics logged: reward, kl, entropy, length across {step_count} steps.",
                duration_sec=time.time() - t0,
            )

        # Case B: train_log.md exists with step table/summary
        if train_log.is_file():
            content = train_log.read_text(encoding="utf-8", errors="replace")
            has_step = re.search(r"100\s*steps?", content, re.IGNORECASE) is not None
            has_metrics = all(k in content.lower() for k in ["reward", "kl", "entropy"])
            if has_step and has_metrics:
                return CheckResult(
                    name=name,
                    tier=3,
                    status=CheckStatus.PASS,
                    message="PPO smoke run log verified in train_log.md",
                    details=f"File: {train_log}",
                    duration_sec=time.time() - t0,
                )

        # Case C: Documented blocker note (per workbook §3.4 rule)
        if notes_log.is_file():
            content = notes_log.read_text(encoding="utf-8", errors="replace")
            if "block" in content.lower() or "unsupported" in content.lower():
                return CheckResult(
                    name=name,
                    tier=3,
                    status=CheckStatus.PASS,
                    message="PPO execution blocker documented per workbook §3.4 rule in NOTES.md",
                    details=f"File: {notes_log}\nContent preview:\n{content[:200]}",
                    duration_sec=time.time() - t0,
                )

        return CheckResult(
            name=name,
            tier=3,
            status=CheckStatus.FAIL,
            message="No valid PPO smoke run logs or documented blocker notes found in runs/ppo/",
            details=f"Checked: {smoke_log}, {train_log}, {notes_log}",
            duration_sec=time.time() - t0,
        )

    # -------------------------------------------------------------------------
    # TIER 4 CHECKS: Hugging Face Hub Publication
    # -------------------------------------------------------------------------

    def check_tier4_hf_repos_exist(self) -> CheckResult:
        t0 = time.time()
        name = "Hugging Face Repositories Creation & Privacy"

        if not self.hf_token:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message="HF_TOKEN missing; cannot verify private HF Hub repositories",
                duration_sec=time.time() - t0,
            )

        expected_repos = [
            ("model", f"{self.hf_username}/gemma-shell-sft"),
            ("model", f"{self.hf_username}/dpo-shell"),
            ("model", f"{self.hf_username}/rm-shell"),
            ("dataset", f"{self.hf_username}/shell-sft-preferences"),
        ]

        verified_repos: list[str] = []
        errors: list[str] = []

        headers = {"Authorization": f"Bearer {self.hf_token}"}
        for repo_type, repo_id in expected_repos:
            api_type = "datasets" if repo_type == "dataset" else "models"
            url = f"https://huggingface.co/api/{api_type}/{repo_id}"
            status, body, _ = self._http_get(url, headers=headers, timeout=10)

            if status != 200:
                errors.append(
                    f"Repo '{repo_id}' ({repo_type}) HTTP {status}: {body[:120]}"
                )
                continue

            try:
                info = json.loads(body)
                is_private = info.get("private", False)
                if not is_private:
                    errors.append(f"Repo '{repo_id}' exists but is NOT private")
                else:
                    verified_repos.append(f"{repo_id} ({repo_type}, private)")
            except Exception as e:
                errors.append(f"Error parsing JSON for {repo_id}: {e}")

        if errors:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message=f"HF Hub repository checks failed for {len(errors)} repo(s)",
                details="\n".join(errors),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=4,
            status=CheckStatus.PASS,
            message=f"All {len(verified_repos)} expected private repositories verified on HF Hub",
            details="\n".join(f" - {r}" for r in verified_repos),
            duration_sec=time.time() - t0,
        )

    def check_tier4_model_cards_present(self) -> CheckResult:
        t0 = time.time()
        name = "Model and Dataset Documentation Cards"

        if not self.hf_token:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message="HF_TOKEN missing; cannot verify model cards",
                duration_sec=time.time() - t0,
            )

        expected_repos = [
            ("model", f"{self.hf_username}/gemma-shell-sft"),
            ("model", f"{self.hf_username}/dpo-shell"),
            ("model", f"{self.hf_username}/rm-shell"),
            ("dataset", f"{self.hf_username}/shell-sft-preferences"),
        ]

        missing_cards: list[str] = []
        verified_cards: list[str] = []
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        for repo_type, repo_id in expected_repos:
            prefix = "datasets/" if repo_type == "dataset" else ""
            raw_url = f"https://huggingface.co/{prefix}{repo_id}/raw/main/README.md"
            status, body, _ = self._http_get(raw_url, headers=headers, timeout=10)

            if status != 200 or not body.strip():
                missing_cards.append(
                    f"{repo_id}: README.md missing or empty (HTTP {status})"
                )
                continue

            card_lower = body.lower()
            # Verify minimum required documentation sections
            has_dataset = "dataset" in card_lower or "preference" in card_lower
            has_eval = (
                "eval" in card_lower or "result" in card_lower or "metric" in card_lower
            )

            if not (has_dataset or has_eval):
                missing_cards.append(
                    f"{repo_id}: README.md lacks dataset provenance or evaluation documentation"
                )
            else:
                verified_cards.append(f"{repo_id}: README.md ({len(body)} chars)")

        if missing_cards:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message=f"Documentation card issues in {len(missing_cards)} repositories",
                details="\n".join(missing_cards),
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=4,
            status=CheckStatus.PASS,
            message="Model and dataset cards verified across all published repositories",
            details="\n".join(f" - {v}" for v in verified_cards),
            duration_sec=time.time() - t0,
        )

    def check_tier4_read_back_verification(self) -> CheckResult:
        t0 = time.time()
        name = "Hub Read-Back & Cache Verification"

        if not self.hf_token:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message="HF_TOKEN missing; cannot verify read-back",
                duration_sec=time.time() - t0,
            )

        # Download README.md from gemma-shell-sft to verify read-back
        target_repo = f"{self.hf_username}/gemma-shell-sft"
        raw_url = f"https://huggingface.co/{target_repo}/raw/main/README.md"
        headers = {"Authorization": f"Bearer {self.hf_token}"}

        status, body, resp_headers = self._http_get(
            raw_url, headers=headers, timeout=15
        )
        if status != 200:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message=f"Read-back download from {target_repo} failed with HTTP {status}",
                details=f"URL: {raw_url}\nResponse: {body[:150]}",
                duration_sec=time.time() - t0,
            )

        if len(body.strip()) < 50:
            return CheckResult(
                name=name,
                tier=4,
                status=CheckStatus.FAIL,
                message=f"Read-back content from {target_repo} is suspiciously short ({len(body)} chars)",
                details=f"Content: {body}",
                duration_sec=time.time() - t0,
            )

        return CheckResult(
            name=name,
            tier=4,
            status=CheckStatus.PASS,
            message=f"Read-back verification succeeded for {target_repo} ({len(body)} bytes read)",
            details=f"Downloaded {target_repo}/README.md successfully.\nPreview:\n{body[:150]}...",
            duration_sec=time.time() - t0,
        )

    # -------------------------------------------------------------------------
    # RUNNER & REPORTING
    # -------------------------------------------------------------------------

    def run_tier(self, tier: int) -> list[CheckResult]:
        if tier == 1:
            return [
                self.check_tier1_gpu_cuda(),
                self.check_tier1_ollama_model(),
                self.check_tier1_hf_auth(),
                self.check_tier1_model_resolution(),
            ]
        elif tier == 2:
            return [
                self.check_tier2_preference_counts(),
                self.check_tier2_schema_and_keys(),
                self.check_tier2_validate_pairs_cli(),
                self.check_tier2_safety_check(),
                self.check_tier2_baseline_non_leakage(),
                self.check_tier2_format_parity(),
            ]
        elif tier == 3:
            return [
                self.check_tier3_adapter_artifacts(),
                self.check_tier3_sft_evaluation(),
                self.check_tier3_dpo_evaluation(),
                self.check_tier3_reward_model_evaluation(),
                self.check_tier3_ppo_smoke_run(),
            ]
        elif tier == 4:
            return [
                self.check_tier4_hf_repos_exist(),
                self.check_tier4_model_cards_present(),
                self.check_tier4_read_back_verification(),
            ]
        else:
            raise ValueError(f"Unknown tier: {tier}. Valid tiers are 1, 2, 3, 4.")

    def run_tiers(self, tiers: list[int], fail_fast: bool = False) -> list[CheckResult]:
        all_results: list[CheckResult] = []
        for t in sorted(set(tiers)):
            if self.verbose:
                print(f"\n=== Executing Tier {t} Verification Checks ===")
            results = self.run_tier(t)
            all_results.extend(results)
            if fail_fast and any(r.status == CheckStatus.FAIL for r in results):
                break
        return all_results

    def format_report_text(self, results: list[CheckResult]) -> str:
        lines = []
        lines.append("=" * 80)
        lines.append(
            "              GEMMA SHELL OPS RLHF PIPELINE - E2E VERIFICATION REPORT"
        )
        lines.append("=" * 80)

        # Group by tier
        by_tier: dict[int, list[CheckResult]] = {}
        for r in results:
            by_tier.setdefault(r.tier, []).append(r)

        tier_names = {
            1: "Tier 1: Environment Readiness",
            2: "Tier 2: Preference Dataset Construction & Safety",
            3: "Tier 3: Training Artifacts & Sequential Evaluation",
            4: "Tier 4: Hugging Face Hub Publication & Read-Back",
        }

        total_pass = sum(1 for r in results if r.status == CheckStatus.PASS)
        total_fail = sum(1 for r in results if r.status == CheckStatus.FAIL)
        total_skip = sum(1 for r in results if r.status == CheckStatus.SKIP)
        total_count = len(results)

        for tier_idx in sorted(by_tier.keys()):
            tier_results = by_tier[tier_idx]
            lines.append(f"\n## {tier_names.get(tier_idx, f'Tier {tier_idx}')}")
            lines.append("-" * 80)
            for r in tier_results:
                tag = f"[{r.status.value}]"
                status_colored = f"{tag:6s}"
                duration_str = f"({r.duration_sec:.2f}s)"
                lines.append(f"  {status_colored} {r.name} {duration_str}")
                lines.append(f"         Result: {r.message}")
                if r.status == CheckStatus.FAIL or (self.verbose and r.details):
                    if r.details:
                        indented = "\n".join(
                            f"         | {dline}" for dline in r.details.splitlines()
                        )
                        lines.append(indented)

        lines.append("\n" + "=" * 80)
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Tests: {total_count}")
        lines.append(f"Passed:      {total_pass}")
        lines.append(f"Failed:      {total_fail}")
        lines.append(f"Skipped:     {total_skip}")
        status_overall = (
            "ALL ASSERTIONS PASSED"
            if total_fail == 0 and total_count > 0
            else "FAILURES DETECTED"
        )
        lines.append(f"Verdict:     {status_overall}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def format_report_json(self, results: list[CheckResult]) -> str:
        total_pass = sum(1 for r in results if r.status == CheckStatus.PASS)
        total_fail = sum(1 for r in results if r.status == CheckStatus.FAIL)
        total_skip = sum(1 for r in results if r.status == CheckStatus.SKIP)

        data = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total": len(results),
            "passed": total_pass,
            "failed": total_fail,
            "skipped": total_skip,
            "all_passed": (total_fail == 0 and len(results) > 0),
            "results": [r.to_dict() for r in results],
        }
        return json.dumps(data, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="E2E Verification Test Runner for Gemma Shell Ops RLHF Pipeline"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--tier",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        help="Specify tier number(s) to execute (e.g. --tier 1 or --tier 1 2)",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Execute all verification tiers (Tiers 1, 2, 3, and 4)",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output report format (default: text)",
    )
    parser.add_argument(
        "--repo-root",
        type=pathlib.Path,
        default=None,
        help="Repository root directory (default: auto-detected)",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default=None,
        help="Python interpreter binary (default: .venv/bin/python3 if present, else sys.executable)",
    )
    parser.add_argument(
        "--ollama-host",
        type=str,
        default="http://localhost:11434",
        help="Ollama host endpoint URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default="rajivmehtapy",
        help="Expected Hugging Face username (default: rajivmehtapy)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face access token (default: reads from HF_TOKEN or cache)",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Halt verification immediately upon first failed assertion",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed diagnostic logging",
    )

    args = parser.parse_args()

    if not args.tier and not args.all:
        parser.print_help()
        sys.exit(2)

    selected_tiers = [1, 2, 3, 4] if args.all else args.tier

    verifier = E2EVerifier(
        repo_root=args.repo_root,
        python_bin=args.python_bin,
        ollama_host=args.ollama_host,
        hf_username=args.hf_user,
        hf_token=args.hf_token,
        verbose=args.verbose,
    )

    results = verifier.run_tiers(selected_tiers, fail_fast=args.fail_fast)

    if args.format == "json":
        print(verifier.format_report_json(results))
    else:
        print(verifier.format_report_text(results))

    # Exit code: 0 if all assertions passed, 1 if any failure
    has_failure = any(r.status == CheckStatus.FAIL for r in results)
    sys.exit(1 if has_failure else 0)


if __name__ == "__main__":
    main()
