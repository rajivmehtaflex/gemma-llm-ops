#!/usr/bin/env python3
"""
Direct High-Quality SFT Dataset Synthesizer for Shell Specialization
Generates comprehensive, diverse, production-grade Bash examples conforming to:
- Shebang: #!/usr/bin/env bash
- Strict mode: set -euo pipefail
- Full quoting for variable expansions and command substitutions
- Defensive checks and traps
- Production explanations
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
DATA_DIR = Path("/content/gemma-llm-ops/data/generated")

DATA_DIR.mkdir(parents=True, exist_ok=True)

SYS_MSG = {
    "role": "system",
    "content": "You are a production-grade Bash engineering assistant. You write safe, idempotent, POSIX-aware Bash scripts with set -euo pipefail, robust quoting, and clear explanations.",
}


def make_record(user_prompt: str, script_code: str, explanation: str) -> dict:
    assistant_content = f"```bash\n{script_code.strip()}\n```\n\n### Explanation\n\n{explanation.strip()}"
    return {
        "messages": [
            SYS_MSG,
            {"role": "user", "content": user_prompt.strip()},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def write_records(slug: str, records: list):
    out_file = DATA_DIR / f"{slug}.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} records -> {out_file}")


def generate_all():
    print("Starting generation of SFT dataset...")
    # Import the generators
    from data_builders import (
        build_error_handling,
        build_quoting_safety,
        build_idempotency_guards,
        build_destructive_dryrun,
        build_coverage_files,
        build_coverage_logs,
        build_coverage_backups,
        build_coverage_processes,
        build_coverage_text,
        build_coverage_automation,
    )

    write_records("weakness-error-handling", build_error_handling())
    write_records("weakness-quoting-safety", build_quoting_safety())
    write_records("weakness-idempotency-guards", build_idempotency_guards())
    write_records("weakness-destructive-dryrun", build_destructive_dryrun())
    write_records("coverage-files", build_coverage_files())
    write_records("coverage-logs", build_coverage_logs())
    write_records("coverage-backups", build_coverage_backups())
    write_records("coverage-processes", build_coverage_processes())
    write_records("coverage-text", build_coverage_text())
    write_records("coverage-automation", build_coverage_automation())
    print("All brief datasets generated successfully!")


if __name__ == "__main__":
    generate_all()
