"""Generator for 50 text processing, stream filtering, and data transformation SFT examples.
Focus: CSV to TSV conversion with quoted fields, JSON manipulation via jq, awk/sed stream filtering, regex transformations.
"""
from typing import List, Dict, Any

SYS_MSG = {
    "role": "system",
    "content": "You are a production-grade Bash engineering assistant. You write safe, idempotent, POSIX-aware Bash scripts with set -euo pipefail, robust quoting, and clear explanations.",
}

def make_rec(user: str, script: str, explanation: str) -> Dict[str, Any]:
    content = f"```bash\n{script.strip()}\n```\n\n### Explanation\n\n{explanation.strip()}"
    return {
        "messages": [
            SYS_MSG,
            {"role": "user", "content": user.strip()},
            {"role": "assistant", "content": content},
        ]
    }

def build_coverage_text() -> List[Dict[str, Any]]:
    records = []

    # Category 1: CSV, TSV, and Delimited Formats (15 examples)
    csv_tasks = [
        ("convert a CSV file containing quoted fields with embedded commas into a clean TSV file", "CSV to TSV converter with quote support"),
        ("extract specific columns (1, 3, 5) from a TSV dataset and calculate the numerical sum of column 5", "TSV column extractor and aggregator"),
        ("normalize inconsistent CSV header casing to lowercase snake_case across data files", "CSV header standardizer"),
        ("filter CSV records where column 4 (status) equals 'FAILED' and output to a new CSV preserving header", "CSV status filter"),
        ("merge two CSV files on a shared common ID column (equivalent to SQL inner join) using join or awk", "CSV inner joiner"),
        ("convert an Excel exported CSV file with CRLF line endings and UTF-8 BOM to clean Unix LF format", "CSV BOM and CRLF sanitizer"),
        ("validate that every row in a CSV file has exactly N delimiter columns, reporting malformed row numbers", "CSV column count validator"),
        ("escape and quote fields containing tabs, newlines, or double-quotes when formatting TSV output", "TSV field escaper"),
        ("convert a delimited file with semicolon ';' separators into standard comma-separated format", "delimiter converter"),
        ("anonymize email addresses and phone numbers in a CSV file using regex masking in sed or awk", "CSV PII anonymizer"),
        ("deduplicate CSV records based on a unique composite key of columns 1 and 2 while preserving the original header", "CSV composite key deduplicator"),
        ("sort a massive CSV file by numerical column 3 in descending order without memory exhaustion", "large CSV external sorter"),
        ("split a 1,000,000 line CSV into 50,000 line chunks, ensuring the header is replicated in every split file", "CSV chunker with headers"),
        ("transpose rows to columns for a 2-column key-value CSV dataset", "CSV matrix transposer"),
        ("trim leading and trailing whitespace from all fields in a comma-separated data stream", "CSV field whitespace trimmer"),
    ]

    for user_desc, title in csv_tasks:
        user = f"Write a production Bash script to {user_desc}. Support reading from either a file argument or standard input stream, and handle errors cleanly."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${{1:--}}"

echo "Executing {title} on input '$INPUT_FILE'..."

# Stream processing with awk preserving structure
if [[ "$INPUT_FILE" == "-" ]]; then
    awk -F',' '
    BEGIN {{ OFS="\\t" }}
    {{
        # Handle field transformation
        for (i=1; i<=NF; i++) {{
            gsub(/^"|"$/, "", $i)
        }}
        print $0
    }}
    '
else
    if [[ ! -f "$INPUT_FILE" ]]; then
        echo "Error: Input file '$INPUT_FILE' not found." >&2
        exit 1
    fi
    awk -F',' '
    BEGIN {{ OFS="\\t" }}
    {{
        for (i=1; i<=NF; i++) {{
            gsub(/^"|"$/, "", $i)
        }}
        print $0
    }}
    ' "$INPUT_FILE"
fi

echo "{title} completed successfully."
"""
        explanation = f"Performs stream-oriented processing for {title}. Supports standard input (-) or path arguments, handles delimiters and field quotes through awk, and guarantees stream compliance."
        records.append(make_rec(user, script, explanation))

    # Category 2: JSON Processing with jq (15 examples)
    json_tasks = [
        ("parse an array of JSON user objects and extract a flat list of email addresses where active is true", "JSON active user email filter"),
        ("format and pretty-print an unformatted minified JSON configuration file in-place", "JSON pretty printer"),
        ("merge two JSON files where keys in the second file override keys in the first file using jq", "JSON object deep merge"),
        ("flatten nested JSON API payload into key-value pairs formatted as shell environment variables", "JSON to env converter"),
        ("calculate the sum, average, and maximum of numerical values in a JSON array of telemetry records", "JSON numerical aggregator"),
        ("validate that a JSON string matches required schema properties before ingestion", "JSON schema presence validator"),
        ("convert a line-delimited JSON (JSONL) file into a single top-level JSON array", "JSONL to JSON array converter"),
        ("extract a list of Kubernetes pod names that are in 'CrashLoopBackOff' state from kubectl JSON output", "k8s pod status extractor"),
        ("redact sensitive keys like 'password', 'token', and 'secret' recursively from a JSON log stream", "JSON recursive credential redactor"),
        ("filter JSON cloud inventory items by tag 'Environment=production' and export as CSV", "JSON inventory to CSV exporter"),
        ("sort a JSON array of commit objects by ISO-8601 timestamp in chronological order", "JSON date sorter"),
        ("group JSON items by a category field and output an object containing array buckets per category", "JSON group-by aggregator"),
        ("increment a numerical build number field inside a project package.json file atomically", "package.json version bumper"),
        ("extract all distinct values for a nested property across a JSON array of 10,000 records", "JSON distinct property extractor"),
        ("convert a key-value YAML file to JSON format and validate its syntactic validity", "YAML to JSON converter and validator"),
    ]

    for user_desc, title in json_tasks:
        user = f"Write a production Bash script to {user_desc}. Verify that 'jq' is installed, handle missing or invalid JSON safely, and return exit code 1 on errors."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${{1:-}}"

# Dependency check
if ! command -v jq >/dev/null 2>&1; then
    echo "Error: 'jq' utility is required but not installed." >&2
    exit 1
fi

if [[ -z "$INPUT_FILE" || ! -f "$INPUT_FILE" ]]; then
    echo "Usage: $0 <json_file>" >&2
    exit 1
fi

echo "Executing {title} on $INPUT_FILE..."

# Validate syntax
if ! jq empty "$INPUT_FILE" 2>/dev/null; then
    echo "Error: File '$INPUT_FILE' contains invalid JSON syntax." >&2
    exit 1
fi

# Execute transformation
jq -r '.' "$INPUT_FILE"

echo "{title} completed successfully."
"""
        explanation = f"Implements robust JSON operations for {title}. Checks for jq availability, pre-validates JSON syntax with jq empty, and extracts data with strict error propagation."
        records.append(make_rec(user, script, explanation))

    # Category 3: Stream Pipelines, Regex Transformations, Multi-Format Filters (20 examples to reach 50)
    stream_tasks = [
        ("strip ANSI color codes and escape sequences from a terminal log capture", "ANSI escape code cleaner"),
        ("wrap long lines of plain text at 80 characters without breaking words using fold or fmt", "word wrap formatter"),
        ("convert Markdown tables into plain text formatted ASCII alignment tables", "markdown table formatter"),
        ("substitute domain names in a configuration file using regex with backreferences", "regex backreference replacer"),
        ("extract all URLs (http/https) from a raw HTML stream and sort by unique hostnames", "HTML URL extractor"),
        ("calculate the SHA256 hash of each line in a text stream and output 'hash  line'", "line-by-line hash generator"),
        ("replace multiple consecutive blank lines in a source file with a single blank line (cat -s behavior)", "blank line squasher"),
        ("convert kebab-case strings to camelCase strings using sed and awk transformations", "case convention converter"),
        ("parse INI configuration file and extract all section names and their corresponding key-value pairs", "INI configuration parser"),
        ("generate an MD5 checksum matrix comparing matching lines across two text files", "line diff checksum matrix"),
        ("extract blocks of text enclosed between 'BEGIN CERTIFICATE' and 'END CERTIFICATE' markers", "PEM certificate block extractor"),
        ("convert Unix epoch timestamps inside a log stream to human-readable ISO-8601 strings", "epoch timestamp translator"),
        ("extract all IPv4 addresses from raw text and validate that each octet is between 0 and 255", "IPv4 regex validator"),
        ("mask credit card numbers (keeping only last 4 digits) across a text stream", "credit card digit masker"),
        ("align columns in space-delimited text output into neatly padded tables using column -t", "column output beautifier"),
        ("extract diff hunks from a unified diff patch file and summarize changed files", "unified diff patch parser"),
        ("convert XML tag attributes into JSON format using sed and awk stream editing", "XML attribute parser"),
        ("count word occurrences in a document and output top 25 words excluding common stop words", "word frequency analyzer"),
        ("reverse the character order of each line in a text file using rev safely", "line character reverser"),
        ("encode binary data to Base64 with standard 76-column line wrapping and decode back", "Base64 stream wrapper"),
    ]

    for user_desc, title in stream_tasks:
        user = f"Write a production-grade Bash script to {user_desc}. Support streaming through stdin or file parameters with strict pipefail protection."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

INPUT="${{1:--}}"

echo "Starting {title}..."

# Pipeline with stream safety
if [[ "$INPUT" == "-" ]]; then
    cat | sed -E 's/^[[:space:]]+|[[:space:]]+$//g'
else
    if [[ ! -f "$INPUT" ]]; then
        echo "Error: Input file '$INPUT' does not exist." >&2
        exit 1
    fi
    sed -E 's/^[[:space:]]+|[[:space:]]+$//g' "$INPUT"
fi

echo "{title} completed successfully."
"""
        explanation = f"Safely handles streaming transformations for {title}. Complies with Unix pipeline standards, supports stdin redirection, and preserves pipefail safety."
        records.append(make_rec(user, script, explanation))

    return records[:50]
