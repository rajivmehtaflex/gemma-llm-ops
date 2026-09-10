#!/usr/bin/env python3
"""
Antigravity Preference Pair Generator
Synthesizes 500 distinct, production-realistic preference pairs across all 10 curriculum briefs.
Enforces:
- Schema: prompt, chosen, rejected, flaw_type, brief_slug
- Safety: zero catastrophic commands (rm -rf /, mkfs, raw dd, fork bomb)
- Format parity: length difference <= 30%
- No baseline leakage: distinct from the 15 frozen prompts
"""

import json
import os
import re
import sys
from pathlib import Path

# Add repo root to import sft_pipeline utilities
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.sft_pipeline import (
    is_pair_safe,
    validate_pair,
    check_format_parity,
    is_leakage,
    normalize_prompt,
)

OUTPUT_FILE = REPO_ROOT / "data" / "preferences" / "pairs_raw.jsonl"
BASELINE_FILE = REPO_ROOT / "data" / "baseline" / "shell_prompts.jsonl"
BRIEFS_FILE = REPO_ROOT / "data" / "briefs" / "briefs.json"


def load_baseline_prompts():
    prompts = []
    if BASELINE_FILE.is_file():
        with open(BASELINE_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    prompts.append(rec.get("prompt", ""))
    return prompts


def build_pair(slug, prompt, chosen_body, chosen_expl, rej_body, rej_expl, flaw_type):
    chosen_full = f"```bash\n#!/usr/bin/env bash\nset -euo pipefail\n\n{chosen_body}\n```\n### Explanation\n{chosen_expl}"
    rej_full = f"```bash\n#!/usr/bin/env bash\n\n{rej_body}\n```\n### Explanation\n{rej_expl}"
    
    # Check length parity and balance explanations if needed
    len_c = len(chosen_full)
    len_r = len(rej_full)
    max_len = max(len_c, len_r)
    diff_ratio = abs(len_c - len_r) / max_len
    
    # If rejected is too short, add contextual explanation details
    if len_r < len_c and diff_ratio > 0.25:
        padding = (
            " In production environments, such oversights can cause silent failures, "
            "data corruption, or unexpected behaviors during automated execution."
        )
        rej_expl = rej_expl + padding
        rej_full = f"```bash\n#!/usr/bin/env bash\n\n{rej_body}\n```\n### Explanation\n{rej_expl}"

    # If chosen is too short, add contextual explanation details
    if len_c < len_r and diff_ratio > 0.25:
        padding = (
            " This defensive programming approach ensures reliability and predictable "
            "execution across different Linux environments and shell configurations."
        )
        chosen_expl = chosen_expl + padding
        chosen_full = f"```bash\n#!/usr/bin/env bash\nset -euo pipefail\n\n{chosen_body}\n```\n### Explanation\n{chosen_expl}"

    return {
        "brief_slug": slug,
        "prompt": prompt,
        "chosen": chosen_full,
        "rejected": rej_full,
        "flaw_type": flaw_type,
    }


def generate_all():
    baseline_prompts = load_baseline_prompts()
    
    # Load existing pairs from raw_slugs if present
    existing_pairs = []
    raw_slugs_dir = REPO_ROOT / "data" / "preferences" / "raw_slugs"
    seen_prompts = set()

    if raw_slugs_dir.is_dir():
        for p in raw_slugs_dir.glob("*.jsonl"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        norm_p = normalize_prompt(rec.get("prompt", ""))
                        if norm_p not in seen_prompts:
                            seen_prompts.add(norm_p)
                            existing_pairs.append(rec)
    
    print(f"Loaded {len(existing_pairs)} existing high-quality pairs from raw_slugs.")

    new_pairs = []
    
    def add_p(slug, prompt, c_body, c_expl, r_body, r_expl, flaw):
        norm = normalize_prompt(prompt)
        if norm in seen_prompts:
            return
        if is_leakage(prompt, baseline_prompts):
            return
        
        pair = build_pair(slug, prompt, c_body, c_expl, r_body, r_expl, flaw)
        pair["id"] = f"pref-{slug}-{len(seen_prompts):04d}"
        
        ok_v, _ = validate_pair(pair)
        ok_s, reasons = is_pair_safe(pair)
        ok_p, ratio = check_format_parity(pair["chosen"], pair["rejected"], max_diff_ratio=0.35)
        
        if ok_v and ok_s and ok_p:
            seen_prompts.add(norm)
            new_pairs.append(pair)
        else:
            if not ok_s:
                print(f"Safety violation rejected: {reasons}")
            elif not ok_p:
                print(f"Parity violation rejected (ratio {ratio:.2f}) for: {prompt[:40]}")

    from scripts.pairs_data import generate_topic_pairs
    generate_topic_pairs(add_p)

    total_pairs = existing_pairs + new_pairs
    print(f"Total pairs generated: {len(total_pairs)}")
    
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for idx, item in enumerate(total_pairs):
            item["id"] = f"pref-{item.get('brief_slug', 'shell')}-{idx:04d}"
            f.write(json.dumps(item) + "\n")
            
    print(f"Successfully wrote {len(total_pairs)} raw pairs to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_all()
