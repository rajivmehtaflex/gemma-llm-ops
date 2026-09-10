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

    return len(reasons) == 0, reasons


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
                    # Extract script content
                    content = ""
                    if "messages" in data:
                        for m in data["messages"]:
                            if m.get("role") == "assistant":
                                content += m.get("content", "") + "\n"
                    elif "response" in data:
                        content = data["response"]
                    else:
                        content = line

                    safe, reasons = is_command_safe(content)
                    if not safe:
                        violations += 1
                        print(f"VIOLATION [{path.name}:{line_idx}]: {'; '.join(reasons)}")
                except json.JSONDecodeError:
                    pass

    print(f"Checked {total_checked} records. Found {violations} safety violations.")
    if violations > 0:
        sys.exit(1)
    print("Safety check passed: 0 violations.")


def extract_records_from_response(raw_text: str) -> List[Dict[str, Any]]:
    """Extract valid messages records from JSON object, array, or JSONL response."""
    # 1. Try direct JSON parsing
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
                        elif "prompt" in ex and ("script" in ex or "response" in ex):
                            resp = ex.get("script") or ex.get("response")
                            if "explanation" in ex:
                                resp = f"{resp}\n\n{ex['explanation']}"
                            records.append({
                                "messages": [
                                    {"role": "user", "content": ex["prompt"]},
                                    {"role": "assistant", "content": resp},
                                ]
                            })
                        elif "code" in ex and ("message" in ex or "description" in ex):
                            prompt_text = ex.get("description") or "Write a Bash script."
                            resp = ex["code"]
                            if "message" in ex:
                                resp = f"{resp}\n\n{ex['message']}"
                            records.append({
                                "messages": [
                                    {"role": "user", "content": prompt_text},
                                    {"role": "assistant", "content": resp},
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

    # 2. Try JSON Lines extraction
    cleaned = re.sub(r'^```(?:jsonl?|json)?\s*$', '', raw_text, flags=re.MULTILINE)
    records = []
    for line in cleaned.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            repaired = re.sub(r'\\(?![/"\\bfnrtu])', r'\\\\', line)
            obj = json.loads(repaired)
            if isinstance(obj, dict) and "messages" in obj:
                records.append(obj)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict) and "messages" in item:
                        records.append(item)
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
        "Output ONLY a valid JSON object with key 'examples', where each item has format: "
        '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}. '
        "The assistant response must contain a robust, production-grade Bash script (with #!/usr/bin/env bash, "
        "set -euo pipefail, full variable quoting, defensive checks, and --dry-run/confirmation "
        "where destructive) followed by concise, production-grade explanations."
    )

    batch_size = 3
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

        batch_idx = existing_count // batch_size
        consecutive_failures = 0

        while existing_count < target_count and consecutive_failures < 5:
            needed = min(batch_size, target_count - existing_count)
            batch_key = f"{slug}:{batch_idx}"

            prompt = (
                f"Topic: {slug}\n"
                f"Focus: {brief.get('weakness') or 'General Domain Coverage'}\n"
                f"Themes: {brief.get('themes', '')}\n"
                f"Difficulty: {brief.get('difficulty', 'medium')}\n"
                f"Must Include: {brief.get('must_include', 'Quoting, error handling, dry-run for deletes')}\n\n"
                f"Generate exactly {needed} unique, non-trivial, and diverse training examples for this topic.\n"
                'Output schema: {"examples": [{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}]}'
            )

            print(f"  Generating batch {batch_idx + 1} ({needed} examples needed, total so far: {existing_count}/{target_count})...")
            try:
                res = call_ollama(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    model=args.model,
                    host=args.host,
                    temperature=0.75,
                    num_predict=2500,
                    response_format="json",
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
                    raise ValueError("No valid records extracted from response")

                to_write = valid_records[:needed]
                with open(slug_file, "a", encoding="utf-8") as sf:
                    for vr in to_write:
                        sf.write(json.dumps(vr) + "\n")

                existing_count += len(to_write)
                manifest[batch_key] = True
                manifest[f"{slug}:count"] = existing_count
                with open(manifest_file, "w", encoding="utf-8") as mf:
                    json.dump(manifest, mf, indent=2)

                consecutive_failures = 0
                print(f"  -> Batch {batch_idx + 1} saved ({len(to_write)} records, total now: {existing_count}/{target_count}).")
                batch_idx += 1
            except Exception as e:
                consecutive_failures += 1
                print(f"    Failed batch {batch_idx + 1} (error: {e}). Consecutive failures: {consecutive_failures}")
                time.sleep(2)

        if existing_count >= target_count:
            print(f"  Brief {slug} completed with {existing_count} records.")
        else:
            print(f"  Brief {slug} paused at {existing_count}/{target_count} after consecutive failures.")


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

    args = parser.parse_args()

    dispatch = {
        "grade": cmd_grade,
        "aggregate": cmd_aggregate,
        "safety_check": cmd_safety_check,
        "generate": cmd_generate,
        "assemble": cmd_assemble,
        "validate": cmd_validate,
    }

    dispatch[args.command](args)


if __name__ == "__main__":
    main()
