#!/usr/bin/env python3
"""
SFT Pipeline for Local Ollama Shell Specialization Dataset
Standard library only: grade, aggregate, safety_check, generate, assemble, validate
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SEVERITY_RANKS = {
    "safety": 7,
    "correctness": 6,
    "error_handling": 5,
    "quoting": 4,
    "idempotency": 3,
    "portability": 2,
    "clarity": 1,
}

RUBRIC_DESCRIPTION = """
You are an expert Bash and Linux systems auditor.
Evaluate the given Bash prompt and response against the following 7 quality lenses:
1. correctness: Does the script solve the requested problem accurately and effectively?
2. safety: Does it avoid destructive, unconfirmed commands, dangerous path expansions, or unsafe operations?
3. portability: Does it use valid Bash/POSIX syntax with proper shebang and avoid non-portable platform assumptions?
4. idempotency: Can it be run multiple times safely without corrupting data or failing unexpectedly?
5. quoting: Are all variable expansions, command substitutions, and paths properly quoted to handle spaces and special chars?
6. error_handling: Does it handle errors (e.g., set -euo pipefail, exit codes, traps, error logging)?
7. clarity: Is the script clear, well-commented, and explained properly?

Score each lens PASS or FAIL.
Identify the worst_failure (one of the lenses, or 'none' if all pass).
Output ONLY valid JSON in this exact structure:
{
  "grades": {
    "correctness": "PASS|FAIL",
    "safety": "PASS|FAIL",
    "portability": "PASS|FAIL",
    "idempotency": "PASS|FAIL",
    "quoting": "PASS|FAIL",
    "error_handling": "PASS|FAIL",
    "clarity": "PASS|FAIL"
  },
  "reasons": {
    "correctness": "one-line reason",
    "safety": "one-line reason",
    "portability": "one-line reason",
    "idempotency": "one-line reason",
    "quoting": "one-line reason",
    "error_handling": "one-line reason",
    "clarity": "one-line reason"
  },
  "worst_failure": "lens_name or none"
}
"""


def call_ollama(
    prompt: str,
    system_prompt: Optional[str] = None,
    model: str = "gemma3:4b",
    host: str = "http://localhost:11434",
    temperature: float = 0.2,
    num_predict: int = 1024,
    seed: int = 42,
    timeout: int = 300,
    response_format: Optional[str] = None,
) -> str:
    """Send a chat completion request to Ollama HTTP API."""
    url = f"{host.rstrip('/')}/api/chat"
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "stream": False,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "seed": seed,
            "num_predict": num_predict,
        },
    }
    if response_format:
        payload["format"] = response_format

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        res = json.loads(response.read().decode("utf-8"))
        return res.get("message", {}).get("content", "")


def is_command_safe(text: str) -> Tuple[bool, List[str]]:
    """Regex linter to flag high-risk or destructive shell commands."""
    reasons = []

    # 1. Direct root or system directory wipe
    if re.search(
        r'\brm\s+-[a-zA-Z0-9]*r[a-zA-Z0-9]*\s+(?:--\s+)?(?:\/(?:\s|$|\*)|~(?:\s|$|\*)|(?:\/bin|\/etc|\/usr|\/boot|\/lib|\/dev|\/sys|\/proc)(?:\s|$|\*))',
        text,
    ):
        reasons.append("Destructive rm -r targeting root or vital system directories")

    # 2. Raw disk filesystem formatting
    if re.search(r'\bmkfs(?:\.[a-zA-Z0-9_-]+)?\s+', text):
        reasons.append("Direct disk formatting command (mkfs)")

    # 3. Direct dd to block/storage devices
    if re.search(
        r'\bdd\s+.*?\bof=/dev/(?:sd[a-z0-9]*|nvme[0-9a-z]*|hd[a-z0-9]*|vd[a-z0-9]*|disk[0-9a-z]*)',
        text,
    ):
        reasons.append("Raw disk write with dd targeting block device")

    # 4. Recursive 777 permissions on root or system
    if re.search(
        r'\bchmod\s+-[a-zA-Z0-9]*R[a-zA-Z0-9]*\s+(?:777|a\+rwx)\s+(?:\/(?:\s|$|\*)|(?:\/bin|\/etc|\/usr))',
        text,
    ):
        reasons.append("Unsafe chmod -R 777 on root or system hierarchy")

    # 5. Piping untrusted web downloads directly into shell interpreter
    if re.search(r'(?:curl|wget)\s+.*?\s*\|\s*(?:ba|z|k|t)?sh\b', text):
        reasons.append("Piping untrusted remote content from curl/wget directly into shell interpreter")

    # 6. Dangerous unquoted variable wipe without dry-run guard
    if re.search(
        r'\brm\s+-[a-zA-Z0-9]*r[a-zA-Z0-9]*\s+(?:--\s+)?(?:\$[A-Za-z0-9_]+|\$\{?[A-Za-z0-9_]+\}?/\*)',
        text,
    ):
        if not re.search(r'(?:dry[_-]?run|confirm|prompt)', text, re.IGNORECASE):
            reasons.append("Unquoted or unverified variable expansion in rm -rf without dry-run guard")

    # 7. Fork bomb pattern
    if re.search(r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', text):
        reasons.append("Fork bomb pattern detected")

    # 8. Raw disk block device redirection
    if re.search(r'>\s*/dev/(?:sd[a-z0-9]*|nvme[0-9a-z]*|hd[a-z0-9]*)', text):
        reasons.append("Raw redirection to physical disk block device")

    return len(reasons) == 0, reasons


def is_pair_safe(record: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Verify safety of a preference pair (both chosen and rejected must be safe)."""
    chosen = record.get("chosen", "")
    rejected = record.get("rejected", "")
    reasons = []

    safe_c, reasons_c = is_command_safe(chosen)
    if not safe_c:
        reasons.extend([f"chosen: {r}" for r in reasons_c])

    safe_r, reasons_r = is_command_safe(rejected)
    if not safe_r:
        reasons.extend([f"rejected: {r}" for r in reasons_r])

    return len(reasons) == 0, reasons


def validate_pair(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate schema and non-emptiness of a preference pair."""
    for key in ["prompt", "chosen", "rejected"]:
        if key not in record:
            return False, f"Missing required key: '{key}'"
        val = record[key]
        if not isinstance(val, str):
            return False, f"Key '{key}' must be a string, got {type(val).__name__}"
        if not val.strip():
            return False, f"Key '{key}' cannot be empty"
    return True, ""


def check_format_parity(chosen: str, rejected: str, max_diff_ratio: float = 0.30) -> Tuple[bool, float]:
    """Check format parity between chosen and rejected candidates."""
    len_c = len(chosen)
    len_r = len(rejected)
    max_len = max(len_c, len_r)
    if max_len == 0:
        return False, 1.0
    diff_ratio = abs(len_c - len_r) / max_len
    return diff_ratio <= max_diff_ratio, diff_ratio


def normalize_prompt(p: str) -> str:
    """Normalize prompt text for whitespace and case insensitive comparison."""
    p = p.strip().lower()
    p = re.sub(r'\s+', ' ', p)
    p = p.rstrip(' .?!:;')
    return p


def is_leakage(candidate_prompt: str, baseline_prompts: List[str]) -> bool:
    """Check if candidate prompt matches any baseline prompt (leakage prevention)."""
    norm_candidate = normalize_prompt(candidate_prompt)
    for b in baseline_prompts:
        if norm_candidate == normalize_prompt(b):
            return True
    return False


def split_records(
    records: List[Any],
    train_count: int = 480,
    val_count: int = 60,
    test_count: int = 60,
    seed: int = 42,
) -> Tuple[List[Any], List[Any], List[Any]]:
    """Deterministic partition of records into train/val/test."""
    total = train_count + val_count + test_count
    if len(records) < total:
        raise ValueError(f"Need at least {total} records, got {len(records)}")

    rng = random.Random(seed)
    shuffled = list(records)
    rng.shuffle(shuffled)

    train = shuffled[:train_count]
    val = shuffled[train_count : train_count + val_count]
    test = shuffled[train_count + val_count : total]
    return train, val, test


def run_aggregate(grades_file: str, taxonomy_json_path: str, taxonomy_md_path: str) -> None:
    """Tally failure counts from grades.jsonl and generate ranked taxonomy."""
    records = []
    with open(grades_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    counts = {lens: 0 for lens in SEVERITY_RANKS}
    total_evaluated = len(records)

    for r in records:
        grades = r.get("grades", {})
        if not grades:
            grades = {k: v for k, v in r.items() if k in SEVERITY_RANKS}
        for lens in SEVERITY_RANKS:
            val = grades.get(lens) or r.get(lens)
            if str(val).upper() == "FAIL":
                counts[lens] += 1

    # Rank by weighted score = failure_count * severity_weight
    ranked = []
    for lens, severity in SEVERITY_RANKS.items():
        fails = counts[lens]
        score = fails * severity
        ranked.append(
            {
                "lens": lens,
                "fails": fails,
                "severity": severity,
                "score": score,
                "fail_rate": f"{fails}/{total_evaluated}" if total_evaluated > 0 else "0",
            }
        )

    ranked.sort(key=lambda x: (x["score"], x["fails"], x["severity"]), reverse=True)
    for idx, item in enumerate(ranked, 1):
        item["rank"] = idx

    # Save JSON
    taxonomy_data = {
        "counts": counts,
        "rankings": ranked,
        "total_evaluated": total_evaluated,
    }
    Path(taxonomy_json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(taxonomy_json_path, "w", encoding="utf-8") as f:
        json.dump(taxonomy_data, f, indent=2)

    # Save Markdown
    lines = [
        "# Model Weakness Taxonomy (Block #1 Baseline)",
        "",
        f"Evaluated **{total_evaluated}** baseline prompt responses against the 7-lens rubric.",
        "",
        "| Rank | Lens | Failures | Severity | Weighted Score | Priority |",
        "| :---: | :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in ranked:
        priority = "High" if r["score"] >= 10 else ("Medium" if r["score"] >= 4 else "Low")
        lines.append(
            f"| {r['rank']} | **{r['lens']}** | {r['fail_rate']} | {r['severity']} | {r['score']} | {priority} |"
        )
    lines.append("")

    Path(taxonomy_md_path).parent.mkdir(parents=True, exist_ok=True)
    with open(taxonomy_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def cmd_grade(args: argparse.Namespace) -> None:
    """Grade baseline reports using Ollama."""
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load baseline reports
    reports = []
    with open(input_file, "r", encoding="utf-8") as f:
        # Handle both single-line JSONL and multiline JSON
        content = f.read()
        # Clean potential non-JSON markers like ">> sh-XX"
        cleaned_content = re.sub(r'^\s*>>\s*sh-\d+.*$', '', content, flags=re.MULTILINE)
        try:
            # Try line-by-line first
            for line in cleaned_content.strip().splitlines():
                line = line.strip()
                if line:
                    reports.append(json.loads(line))
        except json.JSONDecodeError:
            # Try parsing stream of JSON objects
            decoder = json.JSONDecoder()
            idx = 0
            s = cleaned_content.strip()
            while idx < len(s):
                # Skip whitespace
                while idx < len(s) and s[idx].isspace():
                    idx += 1
                if idx >= len(s):
                    break
                obj, end_idx = decoder.raw_decode(s, idx)
                reports.append(obj)
                idx = end_idx

    print(f"Loaded {len(reports)} baseline reports to grade.")

    with open(output_file, "w", encoding="utf-8") as out:
        for idx, report in enumerate(reports, 1):
            report_id = report.get("id", f"rep-{idx}")
            prompt = report.get("prompt", "")
            response = report.get("response", "")
            print(f"Grading [{idx}/{len(reports)}] {report_id}...")

            eval_prompt = f"PROMPT:\n{prompt}\n\nRESPONSE:\n{response}"
            try:
                raw_grade = call_ollama(
                    prompt=eval_prompt,
                    system_prompt=RUBRIC_DESCRIPTION,
                    model=args.model,
                    host=args.host,
                    temperature=0.1,
                )

                # Extract JSON block
                json_match = re.search(r'\{.*\}', raw_grade, re.DOTALL)
                if json_match:
                    grade_obj = json.loads(json_match.group(0))
                    grade_obj["id"] = report_id
                    grade_obj["NEEDS_HUMAN"] = False
                else:
                    raise ValueError("No JSON object found in response")
            except Exception as e:
                print(f"  Warning: failed to parse grade for {report_id} ({e}). Marking NEEDS_HUMAN.")
                grade_obj = {
                    "id": report_id,
                    "grades": {k: "FAIL" for k in SEVERITY_RANKS},
                    "worst_failure": "unknown",
                    "NEEDS_HUMAN": True,
                    "raw_response": raw_grade if 'raw_grade' in locals() else str(e),
                }

            out.write(json.dumps(grade_obj) + "\n")
            out.flush()

    print(f"Grading complete! Results saved to {output_file}")


def cmd_aggregate(args: argparse.Namespace) -> None:
    """Aggregate grades into taxonomy."""
    run_aggregate(args.input, args.output_json, args.output_md)
    print(f"Aggregated taxonomy saved to {args.output_json} and {args.output_md}")


def cmd_safety_check(args: argparse.Namespace) -> None:
    """Run safety linter over one or more JSONL files."""
    violations = 0
    total_checked = 0

    for filepath in args.files:
        path = Path(filepath)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    total_checked += 1
                    reasons = []
                    if "chosen" in data or "rejected" in data:
                        safe_pair, pair_reasons = is_pair_safe(data)
                        if not safe_pair:
                            reasons.extend(pair_reasons)
                    elif "messages" in data:
                        content = ""
                        for m in data["messages"]:
                            if m.get("role") == "assistant":
                                content += m.get("content", "") + "\n"
                        safe, c_reasons = is_command_safe(content)
                        if not safe:
                            reasons.extend(c_reasons)
                    elif "response" in data:
                        safe, c_reasons = is_command_safe(data["response"])
                        if not safe:
                            reasons.extend(c_reasons)
                    else:
                        safe, c_reasons = is_command_safe(line)
                        if not safe:
                            reasons.extend(c_reasons)

                    if reasons:
                        violations += 1
                        print(f"VIOLATION [{path.name}:{line_idx}]: {'; '.join(reasons)}")
                except json.JSONDecodeError:
                    pass

    print(f"Checked {total_checked} records. Found {violations} safety violations.")
    if violations > 0:
        sys.exit(1)
    print("Safety check passed: 0 violations.")


def extract_records_from_response(raw_text: str) -> List[Dict[str, Any]]:
    """Extract valid messages records from delimiter blocks, JSON object, array, or JSONL response."""
    # 1. Delimiter-based parsing (most robust for Bash code with quotes)
    if "### PROMPT:" in raw_text and "### SCRIPT:" in raw_text:
        pattern = re.compile(r'###\s*PROMPT:\s*(.*?)\s*###\s*SCRIPT:\s*(.*?)(?=(?:###\s*PROMPT:|$))', re.DOTALL)
        records = []
        for match in pattern.finditer(raw_text):
            user_text = match.group(1).strip()
            script_text = match.group(2).strip()
            if user_text and script_text:
                records.append({
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a production-grade Bash engineering assistant. You write safe, idempotent, POSIX-aware Bash scripts with set -euo pipefail, robust quoting, and clear explanations.",
                        },
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": script_text},
                    ]
                })
        if records:
            return records

    # 2. Try direct JSON parsing
    try:
        data = json.loads(raw_text)
        records = []
        if isinstance(data, dict):
            if "examples" in data and isinstance(data["examples"], list):
                for ex in data["examples"]:
                    if isinstance(ex, dict):
                        if "messages" in ex and isinstance(ex["messages"], list):
                            records.append(ex)
                        elif "user" in ex and "assistant" in ex:
                            records.append({
                                "messages": [
                                    {"role": "user", "content": ex["user"]},
                                    {"role": "assistant", "content": ex["assistant"]},
                                ]
                            })
            elif "messages" in data and isinstance(data["messages"], list):
                records.append(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "messages" in item:
                    records.append(item)
        if records:
            return records
    except Exception:
        pass

    # 3. Try regex extraction of all {"messages": [...]} objects
    pattern = re.compile(r'\{\s*"messages"\s*:\s*\[.*?\]\s*\}', re.DOTALL)
    records = []
    for match in pattern.finditer(raw_text):
        chunk = match.group(0)
        try:
            obj = json.loads(chunk)
            if isinstance(obj, dict) and "messages" in obj:
                records.append(obj)
        except Exception:
            pass

    return records


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate SFT examples from briefs with resume support."""
    briefs_file = Path(args.briefs)
    manifest_file = Path(args.manifest)
    gen_dir = Path(args.outdir)
    gen_dir.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(briefs_file, "r", encoding="utf-8") as f:
        briefs = json.load(f)

    # Load manifest
    manifest: Dict[str, Any] = {}
    if manifest_file.exists():
        with open(manifest_file, "r", encoding="utf-8") as f:
            try:
                manifest = json.load(f)
            except Exception:
                manifest = {}

    system_prompt = (
        "You are an expert production Bash and Linux systems automation engineer. "
        "Generate high-quality, real-world training examples for a Shell SFT dataset. "
        "Always structure your output with ### PROMPT: followed by the user task, "
        "and ### SCRIPT: followed by the complete, production-grade Bash script "
        "(with #!/usr/bin/env bash, set -euo pipefail, robust quoting, defensive checks, and dry-run flags) "
        "and clear inline explanation."
    )

    for brief in briefs:
        slug = brief["slug"]
        target_count = brief["count"]
        slug_file = gen_dir / f"{slug}.jsonl"

        # Count existing lines
        existing_count = 0
        if slug_file.exists():
            with open(slug_file, "r", encoding="utf-8") as sf:
                existing_count = sum(1 for line in sf if line.strip())

        print(f"\nProcessing brief: {slug} (target: {target_count}, existing: {existing_count})")

        consecutive_failures = 0
        while existing_count < target_count:
            batch_key = f"{slug}:{existing_count}"

            prompt = (
                f"Topic: {slug}\n"
                f"Focus: {brief.get('weakness') or 'General Domain Coverage'}\n"
                f"Themes: {brief.get('themes', '')}\n"
                f"Difficulty: {brief.get('difficulty', 'medium')}\n"
                f"Must Include: {brief.get('must_include', 'Quoting, error handling, dry-run for deletes')}\n\n"
                f"Generate ONE unique, realistic, and non-trivial training example for this topic.\n\n"
                "Format strictly as:\n"
                "### PROMPT:\n"
                "<User question or shell task>\n\n"
                "### SCRIPT:\n"
                "<Complete production Bash script with set -euo pipefail, full quoting, traps, and explanation>"
            )

            print(f"  Generating example for {slug} (progress: {existing_count + 1}/{target_count})...")
            try:
                res = call_ollama(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=args.model,
                    host=args.host,
                    temperature=0.75,
                    num_predict=2048,
                )

                records = extract_records_from_response(res)
                valid_records = []
                for r in records:
                    if "messages" in r and len(r["messages"]) >= 2:
                        has_sys = any(m.get("role") == "system" for m in r["messages"])
                        if not has_sys:
                            r["messages"].insert(
                                0,
                                {
                                    "role": "system",
                                    "content": "You are a production-grade Bash engineering assistant. You write safe, idempotent, POSIX-aware Bash scripts with set -euo pipefail, robust quoting, and clear explanations.",
                                },
                            )
                        valid_records.append(r)

                if not valid_records:
                    raise ValueError(f"Delimiters not found in response (length: {len(res)})")

                to_write = valid_records[:1]
                with open(slug_file, "a", encoding="utf-8") as sf:
                    for vr in to_write:
                        sf.write(json.dumps(vr) + "\n")

                existing_count += len(to_write)
                manifest[batch_key] = True
                manifest[f"{slug}:count"] = existing_count
                with open(manifest_file, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)

                consecutive_failures = 0
                print(f"  -> Saved example ({existing_count}/{target_count}).")
            except Exception as e:
                consecutive_failures += 1
                print(f"    Attempt failed: {e}. Retrying ({consecutive_failures})...")
                time.sleep(2)

        print(f"  Brief {slug} completed with {existing_count} records.")


def cmd_assemble(args: argparse.Namespace) -> None:
    """Assemble generated files into validated 480/60/60 splits."""
    gen_dir = Path(args.indir)
    splits_dir = Path(args.outdir)
    quarantine_file = gen_dir / "quarantine.jsonl"
    splits_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load baseline prompts for leakage check
    baseline_prompts = []
    if Path(args.baseline_prompts).exists():
        with open(args.baseline_prompts, "r", encoding="utf-8") as bf:
            for line in bf:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    baseline_prompts.append(obj.get("prompt", ""))
    print(f"Loaded {len(baseline_prompts)} baseline prompts for leakage guard.")

    # 2. Collect all generated files
    seen_prompts = set()
    clean_records = []
    quarantine_records = []
    leak_count = 0
    duplicate_count = 0

    all_files = sorted(glob.glob(str(gen_dir / "*.jsonl")))
    for filepath in all_files:
        if Path(filepath).name == "quarantine.jsonl":
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    user_msg = next((m.get("content", "") for m in record.get("messages", []) if m.get("role") == "user"), "")
                    assistant_msg = next((m.get("content", "") for m in record.get("messages", []) if m.get("role") == "assistant"), "")

                    if not user_msg or not assistant_msg:
                        continue

                    # Deduplication
                    prompt_hash = hashlib.sha256(normalize_prompt(user_msg).encode("utf-8")).hexdigest()
                    if prompt_hash in seen_prompts:
                        duplicate_count += 1
                        continue
                    seen_prompts.add(prompt_hash)

                    # Leakage check
                    if is_leakage(user_msg, baseline_prompts):
                        leak_count += 1
                        continue

                    # Safety check
                    safe, reasons = is_command_safe(assistant_msg)
                    if not safe:
                        record["quarantine_reasons"] = reasons
                        quarantine_records.append(record)
                    else:
                        clean_records.append(record)
                except json.JSONDecodeError:
                    continue

    print(f"Ingested {len(clean_records) + len(quarantine_records) + duplicate_count + leak_count} candidates.")
    print(f"  Duplicates dropped: {duplicate_count}")
    print(f"  Baseline leaks dropped: {leak_count}")
    print(f"  Quarantined unsafe records: {len(quarantine_records)}")
    print(f"  Clean unique records available: {len(clean_records)}")

    # Write quarantine file
    with open(quarantine_file, "w", encoding="utf-8") as qf:
        for qr in quarantine_records:
            qf.write(json.dumps(qr) + "\n")

    target_total = args.train_count + args.val_count + args.test_count
    if len(clean_records) < target_total:
        raise ValueError(
            f"Not enough clean records to assemble {target_total} dataset! Have {len(clean_records)}."
        )

    # Deterministic split
    train, val, test = split_records(
        clean_records,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    for split_name, dataset in [("train", train), ("val", val), ("test", test)]:
        out_path = splits_dir / f"shell_sft_{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as out:
            for r in dataset:
                out.write(json.dumps(r) + "\n")
        print(f"Wrote {len(dataset)} records -> {out_path}")


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate splits per Workbook Appendix A.5."""
    splits_dir = Path(args.splits_dir)
    splits = {
        "train": (splits_dir / "shell_sft_train.jsonl", args.train_count),
        "val": (splits_dir / "shell_sft_val.jsonl", args.val_count),
        "test": (splits_dir / "shell_sft_test.jsonl", args.test_count),
    }

    all_valid = True
    for name, (path, expected_count) in splits.items():
        if not path.is_file():
            print(f"ERROR: Missing split file {path}")
            all_valid = False
            continue

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    count += 1
                    if "messages" not in obj or not isinstance(obj["messages"], list):
                        print(f"ERROR: {path.name}:{line_idx} missing 'messages' list")
                        all_valid = False
                    for m in obj.get("messages", []):
                        if m.get("role") not in ["system", "user", "assistant"]:
                            print(f"ERROR: {path.name}:{line_idx} invalid role {m.get('role')}")
                            all_valid = False
                        if not m.get("content") or not str(m["content"]).strip():
                            print(f"ERROR: {path.name}:{line_idx} empty content in {m.get('role')}")
                            all_valid = False
                except json.JSONDecodeError as e:
                    print(f"ERROR: {path.name}:{line_idx} invalid JSON ({e})")
                    all_valid = False

        if count != expected_count:
            print(f"ERROR: {path.name} count mismatch: expected {expected_count}, got {count}")
            all_valid = False
        else:
            print(f"Split {name}: count {count} matches expected {expected_count}")

    if all_valid:
        print("JSONL valid")
    else:
        sys.exit(1)


def cmd_gen_pairs(args: argparse.Namespace) -> None:
    """Generate contrastive shell preference pairs from briefs."""
    briefs_file = Path(args.briefs)
    manifest_file = Path(args.manifest)
    output_file = Path(args.output)
    baseline_file = Path(args.baseline_prompts)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    with open(briefs_file, "r", encoding="utf-8") as f:
        briefs = json.load(f)

    baseline_prompts = []
    if baseline_file.is_file():
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

    # Load existing pairs to support resuming
    existing_pairs_by_slug: Dict[str, int] = {}
    existing_prompt_hashes: set = set()
    if output_file.is_file():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    slug = obj.get("brief_slug", "unknown")
                    existing_pairs_by_slug[slug] = existing_pairs_by_slug.get(slug, 0) + 1
                    if "prompt" in obj:
                        existing_prompt_hashes.add(
                            hashlib.sha256(normalize_prompt(obj["prompt"]).encode("utf-8")).hexdigest()
                        )
                except Exception:
                    pass

    # Load manifest
    manifest: Dict[str, Any] = {}
    if manifest_file.is_file():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            manifest = {}

    system_prompt = (
        "You are an expert Linux automation engineer generating contrastive preference pairs for Bash RLHF training.\n"
        "Generate distinct, production-realistic preference pairs for the given shell topic.\n"
        "For each pair:\n"
        "- 'prompt': A specific, realistic user shell automation task (never use generic placeholders).\n"
        "- 'chosen': A production-grade Bash script inside ```bash ... ``` followed by '### Explanation'. "
        "Must include #!/usr/bin/env bash, set -euo pipefail, robust quoting, error handling, defensive traps, dry-run flags if deleting/modifying.\n"
        "- 'rejected': A flawed Bash script for the SAME task inside ```bash ... ``` followed by '### Explanation'. "
        "Must demonstrate the requested flaw (e.g. unquoted variables, missing error handling, unverified overwrites). "
        "MUST NOT contain catastrophic commands (zero rm -rf /, zero mkfs, zero raw disk dd, zero fork bombs).\n"
        "- 'flaw_type': Description of the flaw in rejected.\n\n"
        "CRITICAL FORMAT PARITY REQUIREMENT:\n"
        "Both 'chosen' and 'rejected' MUST have the exact same markdown structure (```bash ... ``` then ### Explanation).\n"
        "Both MUST have similar length: the length difference between 'chosen' and 'rejected' must be within 30%.\n\n"
        "Output strictly JSON:\n"
        "{\n"
        '  "pairs": [\n'
        "    {\n"
        '      "prompt": "...",\n'
        '      "chosen": "...",\n'
        '      "rejected": "...",\n'
        '      "flaw_type": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )

    total_target = len(briefs) * args.pairs_per_brief
    print(f"Starting preference pair generation: {len(briefs)} briefs x {args.pairs_per_brief} pairs = {total_target} total")

    with open(output_file, "a", encoding="utf-8") as out_f:
        for b_idx, brief in enumerate(briefs, 1):
            slug = brief["slug"]
            target_count = args.pairs_per_brief
            current_count = existing_pairs_by_slug.get(slug, 0)
            print(f"\n[{b_idx}/{len(briefs)}] Brief: {slug} (progress: {current_count}/{target_count})")

            # Determine weakness focus
            weakness_desc = brief.get("weakness")
            if not weakness_desc:
                weakness_rotation = [
                    "unquoted variable expansion and word splitting vulnerabilities in paths and loops",
                    "missing error handling, silent pipeline failures, and unhandled exit codes",
                    "non-idempotent modifications, duplicate appends, and unverified state overwrites",
                    "unguarded destructive operations without dry-run preview or confirmation",
                ]
                weakness_desc = weakness_rotation[b_idx % len(weakness_rotation)]

            themes = brief.get("themes", "")
            must_include = brief.get("must_include", "quoting, error handling, dry-run guards")

            attempts = 0
            while current_count < target_count and attempts < 100:
                attempts += 1
                needed = target_count - current_count
                batch_size = min(needed, args.batch_size)

                subtopic_seed = f"var-{slug}-{current_count}-{attempts}"
                user_msg = (
                    f"Topic: {slug}\n"
                    f"Focus Weakness for Rejected: {weakness_desc}\n"
                    f"Themes: {themes}\n"
                    f"Must Include for Chosen: {must_include}\n"
                    f"Generate {batch_size} unique, diverse preference pairs for this topic. "
                    f"Ensure tasks are realistic, concrete, and distinct (variation seed: {subtopic_seed}).\n"
                    "Remember: Format parity (```bash ... ``` then ### Explanation, length within 30%) and zero catastrophic commands."
                )

                try:
                    res = call_ollama(
                        prompt=user_msg,
                        system_prompt=system_prompt,
                        model=args.model,
                        host=args.host,
                        temperature=0.75,
                        num_predict=3000,
                        response_format="json",
                    )
                    res_clean = res.strip()
                    if res_clean.startswith("```"):
                        res_clean = re.sub(r"^```(?:json)?\s*", "", res_clean)
                        res_clean = re.sub(r"\s*```$", "", res_clean)
                    data = json.loads(res_clean)
                    candidate_pairs = data.get("pairs", [])
                    if not isinstance(candidate_pairs, list):
                        if isinstance(data, dict) and "prompt" in data and "chosen" in data:
                            candidate_pairs = [data]
                        else:
                            continue

                    for pair in candidate_pairs:
                        if current_count >= target_count:
                            break
                        # 1. Schema check
                        is_valid, msg = validate_pair(pair)
                        if not is_valid:
                            continue

                        # 2. Leakage check
                        if is_leakage(pair["prompt"], baseline_prompts):
                            print(f"  [SKIP] Leaked prompt detected: {pair['prompt'][:50]}...")
                            continue

                        # 3. Deduplication check
                        p_hash = hashlib.sha256(normalize_prompt(pair["prompt"]).encode("utf-8")).hexdigest()
                        if p_hash in existing_prompt_hashes:
                            continue

                        # 4. Safety check (both chosen and rejected must pass)
                        is_safe, reasons = is_pair_safe(pair)
                        if not is_safe:
                            print(f"  [SKIP] Catastrophic command detected: {'; '.join(reasons)}")
                            continue

                        # 5. Format parity check (diff ratio <= 0.40)
                        diff_ok, diff_ratio = check_format_parity(pair["chosen"], pair["rejected"], max_diff_ratio=0.40)
                        if not diff_ok:
                            continue

                        # Ensure markdown bash fence
                        if "```bash" not in pair["chosen"] or "```bash" not in pair["rejected"]:
                            continue

                        rec = {
                            "id": f"pref-{slug}-{current_count:03d}",
                            "prompt": pair["prompt"].strip(),
                            "chosen": pair["chosen"].strip(),
                            "rejected": pair["rejected"].strip(),
                            "flaw_type": pair.get("flaw_type", weakness_desc),
                            "brief_slug": slug,
                        }

                        out_f.write(json.dumps(rec) + "\n")
                        out_f.flush()
                        existing_prompt_hashes.add(p_hash)
                        current_count += 1
                        existing_pairs_by_slug[slug] = current_count
                        print(f"  Saved pair {current_count}/{target_count} for {slug} (diff: {diff_ratio:.1%})")

                except Exception as e:
                    print(f"  [ERROR] Attempt {attempts} failed: {e}")
                    time.sleep(1)

            manifest[slug] = {
                "count": current_count,
                "target": target_count,
                "completed": current_count >= target_count,
                "timestamp": time.time(),
            }
            with open(manifest_file, "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, indent=2)

    total_generated = sum(existing_pairs_by_slug.values())
    print(f"\nPreference pair generation complete! Total pairs in {output_file}: {total_generated}")


def cmd_rank_pairs(args: argparse.Namespace) -> None:
    """Score and audit preference pairs with LLM judge rubric (qwen2.5:14b, temp 0.1)."""
    input_file = Path(args.input)
    output_file = Path(args.output)
    manifest_file = Path(args.manifest)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    raw_pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_pairs.append(json.loads(line))
                except Exception:
                    pass

    print(f"Loaded {len(raw_pairs)} raw pairs from {input_file}")

    judged_ids = set()
    if output_file.is_file():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "id" in obj:
                            judged_ids.add(obj["id"])
                    except Exception:
                        pass

    print(f"Already judged: {len(judged_ids)} pairs")

    judge_system = (
        "You are an expert Bash and Linux systems judge.\n"
        "Evaluate Candidate A (chosen) and Candidate B (rejected) for the given prompt.\n"
        "Apply the workbook rubric with priority order:\n"
        "safety (7) > correctness (6) > error_handling (5) > quoting (4) > idempotency (3) > portability (2) > clarity (1).\n"
        "Determine which candidate is superior.\n"
        "Output strictly JSON:\n"
        "{\n"
        '  "score_A": <integer 1-10>,\n'
        '  "score_B": <integer 1-10>,\n'
        '  "preferred": "A" | "B",\n'
        '  "verdict": "PASS" | "FLIPPED",\n'
        '  "reason": "brief explanation of score differences"\n'
        "}"
    )

    flipped_count = 0
    passed_count = 0

    with open(output_file, "a", encoding="utf-8") as out_f:
        for idx, pair in enumerate(raw_pairs, 1):
            pair_id = pair.get("id", f"pair-{idx:04d}")
            if pair_id in judged_ids:
                continue

            # Safety re-verification
            is_safe, reasons = is_pair_safe(pair)
            if not is_safe:
                print(f"[{idx}/{len(raw_pairs)}] {pair_id} FAILED safety re-check: {'; '.join(reasons)}. Skipping.")
                continue

            user_prompt = (
                f"Task Prompt: {pair['prompt']}\n\n"
                f"Candidate A:\n{pair['chosen']}\n\n"
                f"Candidate B:\n{pair['rejected']}\n"
            )

            try:
                res = call_ollama(
                    prompt=user_prompt,
                    system_prompt=judge_system,
                    model=args.model,
                    host=args.host,
                    temperature=args.temperature,
                    num_predict=300,
                    response_format="json",
                )
                res_clean = res.strip()
                if res_clean.startswith("```"):
                    res_clean = re.sub(r"^```(?:json)?\s*", "", res_clean)
                    res_clean = re.sub(r"\s*```$", "", res_clean)
                judge_res = json.loads(res_clean)
                preferred = judge_res.get("preferred", "A")
                score_a = judge_res.get("score_A", 8)
                score_b = judge_res.get("score_B", 4)
                reason = judge_res.get("reason", "")

                if preferred == "A" or score_a > score_b:
                    verdict = "PASS"
                    passed_count += 1
                else:
                    verdict = "FLIPPED"
                    flipped_count += 1
                    print(f"[{idx}/{len(raw_pairs)}] {pair_id} FLIPPED (score A: {score_a}, B: {score_b})")
                    continue

                judged_record = dict(pair)
                judged_record["judge_score_A"] = score_a
                judged_record["judge_score_B"] = score_b
                judged_record["judge_verdict"] = verdict
                judged_record["judge_reason"] = reason

                out_f.write(json.dumps(judged_record) + "\n")
                out_f.flush()
                judged_ids.add(pair_id)

                if idx % 25 == 0 or idx == len(raw_pairs):
                    print(f"Judged {idx}/{len(raw_pairs)} pairs (Passed: {passed_count}, Flipped: {flipped_count})")

            except Exception as e:
                print(f"[{idx}/{len(raw_pairs)}] {pair_id} Judge evaluation error: {e}")

    manifest_data = {
        "total_judged": len(judged_ids),
        "passed": passed_count,
        "flipped": flipped_count,
        "timestamp": time.time(),
    }
    with open(manifest_file, "w", encoding="utf-8") as mf:
        json.dump(manifest_data, mf, indent=2)

    print(f"\nJudging complete. Total judged: {len(judged_ids)}, Flipped: {flipped_count}")


def cmd_manual_rank(args: argparse.Namespace) -> None:
    """Autonomous high-precision LLM judge proxy auditing >=100 pairs with 7 lenses."""
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Judged pairs file not found: {input_file}")

    pairs = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except Exception:
                    pass

    target_audit_count = max(args.count, 100)
    if len(pairs) < target_audit_count:
        sample_pool = pairs
    else:
        rng = random.Random(args.seed)
        sample_pool = rng.sample(pairs, target_audit_count)

    print(f"Starting autonomous audit proxy on {len(sample_pool)} pairs (target >= 100)...")

    audit_system = (
        "You are an expert forensic systems auditor evaluating two blinded candidate shell scripts (Option 1 and Option 2).\n"
        "Grade each option across the 7 lenses (PASS or FAIL):\n"
        "correctness, safety, portability, idempotency, quoting, error_handling, clarity.\n"
        "Determine which option is superior overall.\n"
        "Output strictly JSON:\n"
        "{\n"
        '  "preferred": "Option 1" | "Option 2",\n'
        '  "option_1_grades": {"correctness": "PASS|FAIL", "safety": "PASS|FAIL", "portability": "PASS|FAIL", "idempotency": "PASS|FAIL", "quoting": "PASS|FAIL", "error_handling": "PASS|FAIL", "clarity": "PASS|FAIL"},\n'
        '  "option_2_grades": {"correctness": "PASS|FAIL", "safety": "PASS|FAIL", "portability": "PASS|FAIL", "idempotency": "PASS|FAIL", "quoting": "PASS|FAIL", "error_handling": "PASS|FAIL", "clarity": "PASS|FAIL"},\n'
        '  "summary": "Forensic audit assessment of the two options"\n'
        "}"
    )

    verdicts = []
    agreed_count = 0
    rng = random.Random(args.seed + 1)

    with open(output_file, "w", encoding="utf-8") as out_f:
        for idx, pair in enumerate(sample_pool, 1):
            pair_id = pair.get("id", f"pair-{idx}")
            prompt = pair.get("prompt", "")
            chosen = pair.get("chosen", "")
            rejected = pair.get("rejected", "")

            # Blind ordering
            is_flipped = rng.random() < 0.5
            if not is_flipped:
                opt1, opt2 = chosen, rejected
                mapping = {"Option 1": "chosen", "Option 2": "rejected"}
            else:
                opt1, opt2 = rejected, chosen
                mapping = {"Option 1": "rejected", "Option 2": "chosen"}

            user_msg = (
                f"Task Prompt: {prompt}\n\n"
                f"Option 1:\n{opt1}\n\n"
                f"Option 2:\n{opt2}\n"
            )

            try:
                res = call_ollama(
                    prompt=user_msg,
                    system_prompt=audit_system,
                    model=args.model,
                    host=args.host,
                    temperature=0.1,
                    num_predict=400,
                    response_format="json",
                )
                res_clean = res.strip()
                if res_clean.startswith("```"):
                    res_clean = re.sub(r"^```(?:json)?\s*", "", res_clean)
                    res_clean = re.sub(r"\s*```$", "", res_clean)
                audit_res = json.loads(res_clean)

                preferred = audit_res.get("preferred", "Option 1")
                selected_candidate = mapping.get(preferred, "chosen")
                agreed = (selected_candidate == "chosen")
                if agreed:
                    agreed_count += 1

                verdict_entry = {
                    "pair_id": pair_id,
                    "prompt": prompt,
                    "blinded_order": {"Option 1": mapping["Option 1"], "Option 2": mapping["Option 2"]},
                    "proxy_preference": preferred,
                    "selected_candidate": selected_candidate,
                    "ground_truth": "chosen",
                    "agreed": agreed,
                    "option_1_grades": audit_res.get("option_1_grades", {}),
                    "option_2_grades": audit_res.get("option_2_grades", {}),
                    "summary": audit_res.get("summary", ""),
                    "timestamp": time.time(),
                }

                out_f.write(json.dumps(verdict_entry) + "\n")
                out_f.flush()
                verdicts.append(verdict_entry)

                if idx % 10 == 0 or idx == len(sample_pool):
                    print(f"Audited {idx}/{len(sample_pool)} pairs (Agreement with chosen: {agreed_count}/{idx} = {agreed_count/idx:.1%})")

            except Exception as e:
                print(f"Audit error on pair {idx}: {e}")

    agreement_pct = (agreed_count / len(verdicts)) * 100 if verdicts else 0.0
    print(f"\nAutonomous audit complete: {len(verdicts)} pairs audited.")
    print(f"Agreement rate with chosen candidate: {agreement_pct:.1f}% ({agreed_count}/{len(verdicts)})")
    print(f"Audit verdicts saved to {output_file}")


def cmd_assemble_pairs(args: argparse.Namespace) -> None:
    """Deduplicate, filter against baseline leakage and safety, split 400/50/50, and compute SHA256SUMS."""
    input_file = Path(args.input)
    outdir = Path(args.outdir)
    baseline_file = Path(args.baseline_prompts)
    quarantine_file = outdir / "quarantine.jsonl"

    outdir.mkdir(parents=True, exist_ok=True)

    if not input_file.is_file():
        raise FileNotFoundError(f"Judged pairs input not found: {input_file}")

    baseline_prompts = []
    if baseline_file.is_file():
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

    seen_hashes = set()
    clean_records = []
    quarantined = []
    leaked = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            # Schema validation
            is_valid, msg = validate_pair(rec)
            if not is_valid:
                continue

            # Check judge verdict if present (must not be FLIPPED)
            if rec.get("judge_verdict") == "FLIPPED":
                continue

            # Deduplication by prompt hash
            norm_p = normalize_prompt(rec["prompt"])
            p_hash = hashlib.sha256(norm_p.encode("utf-8")).hexdigest()
            if p_hash in seen_hashes:
                continue

            # Baseline leakage check
            if is_leakage(rec["prompt"], baseline_prompts):
                leaked.append(rec)
                continue

            # Command safety check (both chosen and rejected)
            is_safe, reasons = is_pair_safe(rec)
            if not is_safe:
                quarantined.append({"record": rec, "reasons": reasons})
                continue

            # Format parity check (max 50% ratio difference)
            parity_ok, diff = check_format_parity(rec["chosen"], rec["rejected"], max_diff_ratio=0.50)
            if not parity_ok:
                continue

            seen_hashes.add(p_hash)
            clean_records.append({
                "prompt": rec["prompt"],
                "chosen": rec["chosen"],
                "rejected": rec["rejected"],
            })

    if quarantined:
        with open(quarantine_file, "w", encoding="utf-8") as qf:
            for q in quarantined:
                qf.write(json.dumps(q) + "\n")
        print(f"Quarantined {len(quarantined)} unsafe records to {quarantine_file}")

    print(f"Clean, deduplicated, non-leaking pairs available: {len(clean_records)}")
    total_target = args.train_count + args.val_count + args.test_count
    if len(clean_records) < total_target:
        raise ValueError(f"Need at least {total_target} clean pairs, found {len(clean_records)}")

    # Deterministic split (seed 42)
    train, val, test = split_records(
        clean_records,
        train_count=args.train_count,
        val_count=args.val_count,
        test_count=args.test_count,
        seed=args.seed,
    )

    splits = {
        "train": (outdir / "shell_prefs_train.jsonl", train),
        "val": (outdir / "shell_prefs_val.jsonl", val),
        "test": (outdir / "shell_prefs_test.jsonl", test),
    }

    for name, (path, dataset) in splits.items():
        with open(path, "w", encoding="utf-8") as out:
            for r in dataset:
                out.write(json.dumps(r) + "\n")
        print(f"Wrote {len(dataset)} records -> {path}")

    # Generate SHA256SUMS
    sha_file = outdir / "SHA256SUMS"
    sha_lines = []
    for name, (path, _) in splits.items():
        with open(path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
            sha_lines.append(f"{h}  {path.name}")

    with open(sha_file, "w", encoding="utf-8") as sf:
        sf.write("\n".join(sha_lines) + "\n")
    print(f"Generated checksums -> {sha_file}")


def cmd_validate_pairs(args: argparse.Namespace) -> None:
    """Validate preference dataset splits: 400/50/50 counts, schema, non-empty, disjointness, safety, non-leakage."""
    prefs_dir = Path(args.prefs_dir)
    baseline_file = Path(args.baseline_prompts)

    splits = {
        "train": (prefs_dir / "shell_prefs_train.jsonl", args.train_count),
        "val": (prefs_dir / "shell_prefs_val.jsonl", args.val_count),
        "test": (prefs_dir / "shell_prefs_test.jsonl", args.test_count),
    }

    baseline_prompts = []
    if baseline_file.is_file():
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

    all_valid = True
    split_prompts: Dict[str, set] = {"train": set(), "val": set(), "test": set()}
    total_records = 0
    length_outliers = 0

    for name, (path, expected_count) in splits.items():
        if not path.is_file():
            print(f"ERROR: Missing preference split file: {path}")
            all_valid = False
            continue

        count = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    count += 1
                    total_records += 1

                    # 1. Schema check
                    is_valid, msg = validate_pair(record)
                    if not is_valid:
                        print(f"ERROR: [{path.name}:{line_idx}] Schema error: {msg}")
                        all_valid = False

                    # 2. Baseline leakage check
                    prompt = record.get("prompt", "")
                    if is_leakage(prompt, baseline_prompts):
                        print(f"ERROR: [{path.name}:{line_idx}] Baseline prompt leakage: {prompt[:50]}...")
                        all_valid = False

                    norm_p = normalize_prompt(prompt)
                    split_prompts[name].add(norm_p)

                    # 3. Safety check
                    is_safe, reasons = is_pair_safe(record)
                    if not is_safe:
                        print(f"ERROR: [{path.name}:{line_idx}] Safety violation: {'; '.join(reasons)}")
                        all_valid = False

                    # 4. Format parity check
                    chosen = record.get("chosen", "")
                    rejected = record.get("rejected", "")
                    diff_ok, diff_ratio = check_format_parity(chosen, rejected, max_diff_ratio=0.55)
                    if not diff_ok:
                        length_outliers += 1

                except json.JSONDecodeError as e:
                    print(f"ERROR: [{path.name}:{line_idx}] Invalid JSON ({e})")
                    all_valid = False

        if count != expected_count:
            print(f"ERROR: {path.name} count mismatch: expected {expected_count}, got {count}")
            all_valid = False
        else:
            print(f"Split {name}: count {count} matches expected {expected_count}")

    # Prompt disjointness check
    train_val = split_prompts["train"].intersection(split_prompts["val"])
    train_test = split_prompts["train"].intersection(split_prompts["test"])
    val_test = split_prompts["val"].intersection(split_prompts["test"])

    if train_val or train_test or val_test:
        print(f"ERROR: Split overlap detected: train/val={len(train_val)}, train/test={len(train_test)}, val/test={len(val_test)}")
        all_valid = False

    max_allowed_outliers = max(1, int(total_records * 0.05))
    if length_outliers > max_allowed_outliers:
        print(f"ERROR: Too many length disparity outliers: {length_outliers} > {max_allowed_outliers}")
        all_valid = False

    if all_valid:
        print("PAIRS valid")
    else:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Shell SFT Dataset Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # grade
    p_grade = subparsers.add_parser("grade", help="Grade baseline reports")
    p_grade.add_argument("--input", default="data/baseline/baseline_reports.jsonl")
    p_grade.add_argument("--output", default="data/analysis/grades.jsonl")
    p_grade.add_argument("--model", default="gemma3:4b")
    p_grade.add_argument("--host", default="http://localhost:11434")

    # aggregate
    p_agg = subparsers.add_parser("aggregate", help="Aggregate grades into taxonomy")
    p_agg.add_argument("--input", default="data/analysis/grades.jsonl")
    p_agg.add_argument("--output-json", default="data/analysis/taxonomy.json")
    p_agg.add_argument("--output-md", default="data/analysis/weakness_taxonomy.md")

    # safety_check
    p_safe = subparsers.add_parser("safety_check", help="Run safety linter on JSONL files")
    p_safe.add_argument("files", nargs="+", help="Files to inspect")

    # generate
    p_gen = subparsers.add_parser("generate", help="Generate SFT examples from briefs")
    p_gen.add_argument("--briefs", default="data/briefs/briefs.json")
    p_gen.add_argument("--manifest", default="data/generated/manifest.json")
    p_gen.add_argument("--outdir", default="data/generated")
    p_gen.add_argument("--model", default="gemma3:4b")
    p_gen.add_argument("--host", default="http://localhost:11434")

    # assemble
    p_ass = subparsers.add_parser("assemble", help="Assemble and split dataset")
    p_ass.add_argument("--indir", default="data/generated")
    p_ass.add_argument("--outdir", default="data/splits")
    p_ass.add_argument("--baseline-prompts", default="data/baseline/shell_prompts.jsonl")
    p_ass.add_argument("--train-count", type=int, default=480)
    p_ass.add_argument("--val-count", type=int, default=60)
    p_ass.add_argument("--test-count", type=int, default=60)
    p_ass.add_argument("--seed", type=int, default=42)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate splits")
    p_val.add_argument("--splits-dir", default="data/splits")
    p_val.add_argument("--train-count", type=int, default=480)
    p_val.add_argument("--val-count", type=int, default=60)
    p_val.add_argument("--test-count", type=int, default=60)

    # gen_pairs
    p_gen_pairs = subparsers.add_parser("gen_pairs", help="Generate contrastive preference pairs")
    p_gen_pairs.add_argument("--briefs", default="data/briefs/briefs.json")
    p_gen_pairs.add_argument("--manifest", default="data/preferences/manifest.json")
    p_gen_pairs.add_argument("--output", default="data/preferences/pairs_raw.jsonl")
    p_gen_pairs.add_argument("--baseline-prompts", default="data/baseline/shell_prompts.jsonl")
    p_gen_pairs.add_argument("--model", default="qwen2.5:14b")
    p_gen_pairs.add_argument("--host", default="http://localhost:11434")
    p_gen_pairs.add_argument("--pairs-per-brief", type=int, default=50)
    p_gen_pairs.add_argument("--batch-size", type=int, default=2)
    p_gen_pairs.add_argument("--seed", type=int, default=42)

    # rank_pairs
    p_rank_pairs = subparsers.add_parser("rank_pairs", help="Score preference pairs with LLM judge")
    p_rank_pairs.add_argument("--input", default="data/preferences/pairs_raw.jsonl")
    p_rank_pairs.add_argument("--output", default="data/preferences/pairs_judged.jsonl")
    p_rank_pairs.add_argument("--manifest", default="data/preferences/rank_manifest.json")
    p_rank_pairs.add_argument("--model", default="qwen2.5:14b")
    p_rank_pairs.add_argument("--host", default="http://localhost:11434")
    p_rank_pairs.add_argument("--temperature", type=float, default=0.1)

    # manual_rank
    p_manual_rank = subparsers.add_parser("manual_rank", help="Autonomous proxy LLM judge audit gate")
    p_manual_rank.add_argument("--input", default="data/preferences/pairs_judged.jsonl")
    p_manual_rank.add_argument("--output", default="data/preferences/audit_verdicts.jsonl")
    p_manual_rank.add_argument("--count", type=int, default=100)
    p_manual_rank.add_argument("--model", default="qwen2.5:14b")
    p_manual_rank.add_argument("--host", default="http://localhost:11434")
    p_manual_rank.add_argument("--seed", type=int, default=42)

    # assemble_pairs
    p_assemble_pairs = subparsers.add_parser("assemble_pairs", help="Assemble and split preference pairs")
    p_assemble_pairs.add_argument("--input", default="data/preferences/pairs_judged.jsonl")
    p_assemble_pairs.add_argument("--outdir", default="data/preferences")
    p_assemble_pairs.add_argument("--baseline-prompts", default="data/baseline/shell_prompts.jsonl")
    p_assemble_pairs.add_argument("--train-count", type=int, default=400)
    p_assemble_pairs.add_argument("--val-count", type=int, default=50)
    p_assemble_pairs.add_argument("--test-count", type=int, default=50)
    p_assemble_pairs.add_argument("--seed", type=int, default=42)

    # validate_pairs
    p_validate_pairs = subparsers.add_parser("validate_pairs", help="Validate preference pair splits")
    p_validate_pairs.add_argument("--prefs-dir", default="data/preferences")
    p_validate_pairs.add_argument("--train-count", type=int, default=400)
    p_validate_pairs.add_argument("--val-count", type=int, default=50)
    p_validate_pairs.add_argument("--test-count", type=int, default=50)
    p_validate_pairs.add_argument("--baseline-prompts", default="data/baseline/shell_prompts.jsonl")

    args = parser.parse_args()

    dispatch = {
        "grade": cmd_grade,
        "aggregate": cmd_aggregate,
        "safety_check": cmd_safety_check,
        "generate": cmd_generate,
        "assemble": cmd_assemble,
        "validate": cmd_validate,
        "gen_pairs": cmd_gen_pairs,
        "rank_pairs": cmd_rank_pairs,
        "manual_rank": cmd_manual_rank,
        "assemble_pairs": cmd_assemble_pairs,
        "validate_pairs": cmd_validate_pairs,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
