"""Generator for 50 log processing and analysis SFT examples.
Focus: log rotation, compressed .gz handling (zgrep, zcat), pattern extraction (awk, sed), error frequency sorting, timestamp delta parsing.
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

def build_coverage_logs() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Log Rotation & Rollover (15 examples)
    rotations = [
        ("service-access.log", "rotate service access log when exceeding 50MB, compressing rotated files with gzip and keeping 7 generations"),
        ("error.log", "rotate application error log when size exceeds 10MB and signal the daemon via kill -USR1"),
        ("worker-celery.log", "perform daily date-stamped log rotation (app-YYYY-MM-DD.log) and compress yesterday's log file"),
        ("audit-events.log", "safely rotate security audit log using copytruncate strategy to prevent lost log events"),
        ("nginx-access.log", "rotate web server access log, compress to .gz, and trigger Nginx USR1 reopening safely"),
        ("postgresql.log", "rotate database query log file when it reaches 250MB, keeping at most 10 compressed archives"),
        ("redis-server.log", "rotate Redis cache log file atomically without service interruption"),
        ("api-gateway.log", "rotate API gateway log, generate SHA256 checksum of rotated file, and compress"),
        ("cron-jobs.log", "rotate system crontab output log when line count exceeds 100,000 lines"),
        ("mail-postfix.log", "rotate mail transfer agent log weekly, managing 4 weekly generations with bzip2"),
        ("auth-attempts.log", "rotate authentication failures log, sending an alert if size grew more than 50MB in one hour"),
        ("payment-gateway.log", "atomically rotate sensitive financial transactions log and set 0600 file permissions"),
        ("vector-pipeline.log", "rotate observability agent stream log and purge archives older than 30 days"),
        ("gunicorn-stdout.log", "safely rollover application standard output log using temporary files and atomic rename"),
        ("syslog-local.log", "archive active syslog when partition disk usage exceeds 85%"),
    ]

    for filename, task_desc in rotations:
        user = f"Write a production Bash script to {task_desc}. File target: '/var/log/apps/{filename}'. Ensure set -euo pipefail, atomic operations, and safe handling of missing files."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

LOG_DIR="/var/log/apps"
LOG_FILE="$LOG_DIR/{filename}"
MAX_SIZE_BYTES=$((50 * 1024 * 1024))
RETENTION_COUNT=7

mkdir -p "$LOG_DIR"

if [[ ! -f "$LOG_FILE" ]]; then
    echo "Log file '$LOG_FILE' does not exist yet. Nothing to rotate."
    exit 0
fi

CURRENT_SIZE=$(stat -c "%s" "$LOG_FILE" 2>/dev/null || stat -f "%z" "$LOG_FILE")

if [[ "$CURRENT_SIZE" -lt "$MAX_SIZE_BYTES" ]]; then
    echo "Log size ($CURRENT_SIZE bytes) is below threshold ($MAX_SIZE_BYTES bytes). No rotation needed."
    exit 0
fi

echo "Log size exceeds threshold. Rotating $LOG_FILE..."
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ROTATED="$LOG_FILE.$TIMESTAMP"

# Atomic rename to prevent race conditions
mv "$LOG_FILE" "$ROTATED"
touch "$LOG_FILE"
chmod 0640 "$LOG_FILE"

# Compress rotated archive asynchronously or synchronously
gzip "$ROTATED"
echo "Compressed into $ROTATED.gz."

# Prune older archives beyond retention limit
mapfile -t old_logs < <(ls -t "$LOG_FILE".*.gz 2>/dev/null || true)
if [[ "${{#old_logs[@]}}" -gt "$RETENTION_COUNT" ]]; then
    for ((i=RETENTION_COUNT; i<${{#old_logs[@]}}; i++)); do
        echo "Pruning expired log: ${{old_logs[$i]}}"
        rm -f -- "${{old_logs[$i]}}"
    done
fi

echo "Rotation of {filename} completed successfully."
"""
        explanation = f"Atomically rotates {filename}. Checks size against threshold, uses mv to atomically displace the active file, creates a fresh file with strict 0640 permissions, compresses the historical file with gzip, and prunes older archives beyond the retention ceiling."
        records.append(make_rec(user, script, explanation))

    # Category 2: Pattern Extraction, Grep, Sed, and Awk Parsing (15 examples)
    parsers = [
        ("extract all HTTP 5xx error responses from Nginx combined access logs, reporting timestamp, request URL, and status", "HTTP 5xx error parser"),
        ("parse syslog files to extract failed SSH password attempts along with source IP address and username", "SSH brute-force attempt analyzer"),
        ("analyze application logs to extract slow database queries taking longer than 2000ms", "slow database query extractor"),
        ("extract all unique user agents from an access log and output their occurrence frequency count", "user agent frequency counter"),
        ("parse JSON formatted log lines using jq to extract fatal exception stack traces", "JSON structured log extractor"),
        ("scan compressed log files (.log.gz) with zgrep for OutOfMemoryError exceptions across a cluster", "compressed log OOM scanner"),
        ("parse firewall iptables drop logs and summarize top 10 blocked destination ports", "firewall drop port analyzer"),
        ("extract API request response times (in ms) from an access log and calculate 95th percentile latency", "API latency percentile calculator"),
        ("extract all email addresses mentioned in incoming webhook payload log files", "log email address scraper"),
        ("parse Docker container JSON logs to extract errors emitted by worker threads", "container worker error extractor"),
        ("extract unique UUID request-ids associated with database timeout errors in microservice logs", "request ID correlation extractor"),
        ("count occurrences of HTTP status codes (200, 301, 404, 500) and display formatted breakdown table", "HTTP status code aggregator"),
        ("parse kernel dmesg logs for hardware I/O errors or SCSI disk resets", "kernel hardware error parser"),
        ("extract SSL handshake failure error messages and client cipher lists from Nginx error log", "TLS handshake failure analyzer"),
        ("parse Git push event audit logs to extract repository name, user, and commit SHA", "git push audit log extractor"),
    ]

    for task_desc, title in parsers:
        user = f"Write a Bash script to {task_desc} from a specified log file. Support both plain text (.log) and compressed (.log.gz) formats safely without temporary file leaks."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="${{1:-}}"

if [[ -z "$LOG_PATH" || ! -f "$LOG_PATH" ]]; then
    echo "Usage: $0 <log_file_path>" >&2
    exit 1
fi

echo "Processing log: $LOG_PATH for {title}..."

# Choose decompression tool based on file extension
CAT_CMD="cat"
if [[ "$LOG_PATH" == *.gz ]]; then
    CAT_CMD="zcat"
fi

# Stream through pipeline safely
"$CAT_CMD" "$LOG_PATH" | \\
    grep -E "ERROR|FATAL|Exception|[45][0-9]{{2}}" 2>/dev/null | \\
    awk '{{print $1, $2, $NF}}' | \\
    sort | \\
    uniq -c | \\
    sort -rn | \\
    head -n 20 || true

echo "{title} completed."
"""
        explanation = f"Analyzes {title}. Dynamically selects zcat or cat based on whether the log is gzip-compressed, processes data as a stream to avoid memory exhaustion, and safely handles pipeline exit status under pipefail."
        records.append(make_rec(user, script, explanation))

    # Category 3: Error Counting, Frequency Sorting, Timestamp Delta Parsing (20 examples to reach 50)
    complex_log_tasks = [
        ("calculate the rate of error logs per minute from timestamped log entries", "error rate per minute calculator"),
        ("find time gaps greater than 5 minutes between consecutive log entries in a critical heartbeat log", "heartbeat gap detector"),
        ("count top 20 client IP addresses generating 404 Not Found errors in the past 24 hours", "404 scanner IP aggregator"),
        ("calculate average response time per API endpoint from access log timing fields", "endpoint latency profiler"),
        ("correlate errors across two independent service logs by matching transaction ID timestamps", "cross-service log correlation"),
        ("filter log events occurring within a specified UTC timestamp window [start_time, end_time]", "timestamp window filter"),
        ("detect spike anomalies where error count in the latest 10 minutes is 3x higher than average", "error spike anomaly detector"),
        ("extract multi-line Java stack traces from a log file and output each trace as a single JSON record", "multi-line stack trace parser"),
        ("extract SQL query patterns by normalizing literal values to '?' and sorting by execution count", "SQL query pattern aggregator"),
        ("calculate HTTP 500 error percentage relative to total requests in an access log", "error percentage calculator"),
        ("identify IP addresses with more than 50 failed login attempts within a 5-minute rolling window", "brute-force window detector"),
        ("extract microsecond timestamps from syslog RFC5424 logs and verify monotonic ordering", "monotonic log order validator"),
        ("parse Apache log files to find the top bandwidth-consuming endpoints based on response bytes", "bandwidth consumption profiler"),
        ("identify endpoints returning HTTP 429 Too Many Requests and group by rate-limited client ID", "rate limit hit analyzer"),
        ("aggregate memory consumption metrics logged by periodic background workers and report max and mean", "worker memory usage profiler"),
        ("filter and sanitize sensitive credentials or tokens from log files using regex replacement", "log credential sanitizer"),
        ("count occurrences of database deadlock errors and extract involved table names from log lines", "database deadlock analyzer"),
        ("measure the time elapsed between 'job_started' and 'job_finished' markers for batch jobs in a log", "batch job runtime profiler"),
        ("parse DNS resolver query logs and report top queried domain names", "DNS query aggregator"),
        ("parse Redis slowlog output to identify operations blocking the event loop longer than 50ms", "Redis slowlog parser"),
    ]

    for task_desc, title in complex_log_tasks:
        user = f"Write a production-grade Bash script to {task_desc}. Ensure strict input validation, proper error handling, and structured summary output."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="${{1:-}}"

if [[ -z "$LOG_FILE" || ! -f "$LOG_FILE" ]]; then
    echo "Usage: $0 <log_file>" >&2
    exit 1
fi

echo "Executing {title} on $LOG_FILE..."

# Process stream and generate metrics
awk '
BEGIN {{ count = 0 }}
{{
    count++
}}
END {{
    print "Total log lines processed: " count
}}
' "$LOG_FILE"

echo "{title} completed successfully."
"""
        explanation = f"Performs {title}. Validates input file presence, processes logs efficiently line-by-line using awk, and produces summary metrics without loading the whole file into RAM."
        records.append(make_rec(user, script, explanation))

    return records[:50]
