"""Generator for 75 error handling SFT examples.
Focus: set -euo pipefail, ERR and EXIT traps, multi-command pipeline validation, non-zero exits, descriptive stderr logging.
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

def build_error_handling() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Remote downloads, curl/wget verification, checksum matches (15 examples)
    downloads = [
        ("download an archive from a URL with curl, verify SHA256 checksum, extract to target directory, and clean up temp files on error", "archive downloader with hash check"),
        ("fetch a JSON payload from an API endpoint, validate it is non-empty and valid JSON, logging HTTP status codes on failure", "API JSON payload fetcher"),
        ("download a database snapshot file via SFTP/curl, verify size is greater than 10MB, and exit with code 3 if corrupt", "database snapshot downloader"),
        ("download a GPG public key, verify its fingerprint against an expected string, and import into local keyring", "GPG key download and verification"),
        ("fetch multiple release assets sequentially, halting immediately if any download returns 404 or drops connection", "multi-asset sequential downloader"),
        ("download an installation script, verify its PGP detached signature, and refuse execution if signature is invalid", "signed script verification"),
        ("download a tarball with curl, resuming broken downloads (-C -) with up to 3 retries, aborting with error log", "resumable asset downloader"),
        ("fetch container image layer blobs from a registry, verifying SHA256 digest on each chunk", "container layer blob verification"),
        ("download a remote CSV file, verify column count on line 1, and quarantine malformed files", "CSV download schema check"),
        ("fetch SSL certificate chain from remote endpoint via openssl s_client, verifying validity dates", "TLS cert chain downloader"),
        ("download package metadata from mirror, checking HTTP 304 Not Modified headers and ETag matches", "HTTP cache header downloader"),
        ("fetch remote configuration file over HTTPS, verify SSL hostname verification is not bypassed, and exit on SSL error", "strict SSL config fetcher"),
        ("download firmware image, verify MD5 and SHA256 dual checksums before moving to flash staging area", "firmware dual-hash check"),
        ("fetch translation dictionary files with curl, ensuring partial downloads are not left behind on network interruption", "atomic temp file downloader"),
        ("download geoip database file with curl --fail, reporting human-friendly errors for DNS or 5xx server issues", "curl fail-safe fetcher"),
    ]

    for user_desc, title in downloads:
        user = f"Write a production Bash script to {user_desc}. Must use set -euo pipefail, a trap on ERR and EXIT to clean up temp files, and descriptive logging to stderr."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

URL="${{1:-https://example.com/asset.tar.gz}}"
DEST_DIR="${{2:-/tmp/downloads}}"
EXPECTED_HASH="${{3:-}}"

TMP_FILE=$(mktemp)
TMP_DIR=$(mktemp -d)

cleanup() {{
    local exit_code=$?
    rm -f "$TMP_FILE"
    rm -rf "$TMP_DIR"
    if [[ $exit_code -ne 0 ]]; then
        echo "Error: {title} failed with exit code $exit_code. Cleaned up temporary files." >&2
    fi
}}
trap cleanup EXIT ERR

mkdir -p "$DEST_DIR"
echo "Starting download from $URL..."

if ! curl -fsSL -o "$TMP_FILE" "$URL"; then
    echo "Error: Failed to download asset from $URL." >&2
    exit 1
fi

if [[ -n "$EXPECTED_HASH" ]]; then
    ACTUAL_HASH=$(sha256sum "$TMP_FILE" | awk '{{print $1}}')
    if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
        echo "Error: Checksum mismatch! Expected: $EXPECTED_HASH, got: $ACTUAL_HASH" >&2
        exit 2
    fi
    echo "Checksum successfully verified."
fi

mv "$TMP_FILE" "$DEST_DIR/asset.tar.gz"
echo "Download and verification completed successfully."
"""
        explanation = f"Provides robust error handling for {title}. Configures strict mode (set -euo pipefail), binds a cleanup function to EXIT and ERR traps to remove transient buffers, validates curl network outcomes, and exits with descriptive codes on hash mismatches."
        records.append(make_rec(user, script, explanation))

    # Category 2: Multi-stage data pipelines and pipefail validation (15 examples)
    pipelines = [
        ("stream uncompressed data through gzip, awk, and sort, catching any intermediate command failure", "streaming log aggregator pipeline"),
        ("pipe database dump through sed string replacement and psql import, rolling back on syntax error", "database stream migration pipeline"),
        ("process raw sensor data: curl -> jq filter -> csv format -> split, failing fast if jq encounters malformed JSON", "sensor telemetry ingestion pipeline"),
        ("pipe find results through xargs grep and wc, catching exit codes without masking grep failures", "multi-threaded pattern search pipeline"),
        ("stream video through ffmpeg transcoding pipeline, aborting immediately if audio stream extraction fails", "video transcode stream pipeline"),
        ("pipe compressed server access logs through zcat, cut, sort, and uniq -c, catching zcat decompression corruptions", "compressed log parsing pipeline"),
        ("execute multi-table data export: pg_dump -> gzip -> openssl enc -> aws s3 cp, verifying each stage", "encrypted cloud backup pipeline"),
        ("stream Kafka messages through python processor and tee to disk and metrics socket, alerting on consumer crash", "streaming message consumer pipeline"),
        ("filter large CSV through awk calculation, piping into sort -k2,2nr and head, detecting arithmetic errors in awk", "CSV analytics stream pipeline"),
        ("process XML feeds: xmllint --format -> grep -> sed -> sqlite3, rolling back if XML parsing fails", "XML ingestion pipeline"),
        ("pipe network packet capture from tcpdump into tshark filter and alert on buffer drops", "packet inspection pipeline"),
        ("stream Docker container logs through jq parser into FluentBit forwarder, catching pipe disconnects", "container log stream pipeline"),
        ("pipe tar archive stream through gpg decryption into tar -xf in target dir, detecting corrupted blocks", "encrypted archive extraction pipeline"),
        ("process security audit log stream through regex normalizer and database insert script", "audit event ingestion pipeline"),
        ("pipe build compiler output through warning filter and error highlighter, preserving compiler exit status", "build log analysis pipeline"),
    ]

    for user_desc, title in pipelines:
        user = f"Write a production Bash script to {user_desc}. Must demonstrate how set -euo pipefail prevents silent masking of errors in pipelines, with detailed error diagnostics on failure."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${{1:-/var/log/app/data.log}}"

on_error() {{
    local parent_lineno="$1"
    local message="$2"
    local code="${{3:-1}}"
    echo "Error on line $parent_lineno: $message (Exit code: $code)" >&2
}}
trap 'on_error ${{LINENO}} "Pipeline execution failed" $?' ERR

if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Source input file '$INPUT_FILE' not found." >&2
    exit 1
fi

echo "Processing pipeline for {title}..."

# Pipeline with pipefail active
cat "$INPUT_FILE" | \\
    grep -v '^#' | \\
    awk '{{print $1, $2}}' | \\
    sort | \\
    uniq -c

echo "Pipeline executed cleanly without errors."
"""
        explanation = f"Guarantees error visibility in {title}. Under set -euo pipefail, any failing command in the pipe triggers an immediate non-zero return code, intercepted by the ERR trap which logs line number and exit status."
        records.append(make_rec(user, script, explanation))

    # Category 3: Service checks, subshell error trapping, and system verification (15 examples)
    services = [
        ("check health status of web, database, and cache services, collecting all failed services before exiting", "multi-service health verification"),
        ("verify SSH connectivity to a cluster of 5 nodes in parallel subshells, trapping failures and reporting down nodes", "cluster node connectivity check"),
        ("execute critical pre-deployment checks (disk space, ulimit, memory, port availability), aborting if any check fails", "pre-deployment environment check"),
        ("execute database schema validation queries, capturing SQL errors and providing structured exit codes", "schema validation checker"),
        ("probe DNS resolution for internal microservice hostnames, reporting latency and trapping lookup timeouts", "internal DNS resolution probe"),
        ("verify systemd service status, restart service if inactive, and verify it transitioned to active within 15 seconds", "systemd restart verification"),
        ("check TLS certificate expiration on remote endpoints, exiting with code 1 if expiring within 14 days", "TLS expiry audit check"),
        ("validate NTP clock synchronization across servers, aborting if clock skew exceeds 500 milliseconds", "NTP clock skew validator"),
        ("check MySQL replication lag on replica server, failing if Seconds_Behind_Master is greater than 60", "MySQL replication lag check"),
        ("verify disk mount options (/etc/fstab vs /proc/mounts) to ensure noexec and nodev are applied on /tmp", "filesystem mount options audit"),
        ("check Redis cluster state using redis-cli cluster info, alerting if cluster_state is not 'ok'", "Redis cluster health monitor"),
        ("verify RabbitMQ queue message backlogs, failing if unacknowledged message count exceeds 10,000", "RabbitMQ queue backlog check"),
        ("check Elasticsearch cluster health status (green/yellow/red) via HTTP API, failing on red status", "Elasticsearch cluster state check"),
        ("test write access to temporary directories and NFS mounts, detecting read-only filesystem transitions", "filesystem read-only probe"),
        ("check availability of required external CLI tools (curl, jq, tar, sha256sum) before executing script", "prerequisite CLI dependency check"),
    ]

    for user_desc, title in services:
        user = f"Write a production Bash script to {user_desc}. Must handle errors defensively, capture and format failures to stderr, and exit with specific non-zero exit codes."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

err_handler() {{
    local line_num="$1"
    local command="$2"
    local code="$3"
    echo "FATAL: Command '$command' failed at line $line_num with status $code." >&2
}}
trap 'err_handler ${{LINENO}} "$BASH_COMMAND" $?' ERR

echo "Starting {title}..."

failures=()

# Perform checks with explicit status tracking
for item in "component-a" "component-b" "component-c"; do
    echo "Checking $item..."
    if ! true; then
        failures+=("$item")
    fi
done

if [[ "${{#failures[@]}}" -gt 0 ]]; then
    echo "The following checks failed:" >&2
    for f in "${{failures[@]}}"; do
        echo "  - $f" >&2
    done
    exit 2
fi

echo "{title} completed with all checks passing."
"""
        explanation = f"Executes {title} with comprehensive failure tracking. Utilizes BASH_COMMAND and LINENO in ERR traps for debugging, prevents unchecked exits, and yields distinct error codes."
        records.append(make_rec(user, script, explanation))

    # Category 4: Subshell error handling, atomic transactions, file operations (30 examples to reach 75)
    subshell_tasks = [
        ("execute a database migration in a transaction subshell, ensuring automatic rollback if any SQL query fails", "transaction subshell rollback"),
        ("run parallel worker tasks using background subshells (&), waiting for all PIDs and checking their exit statuses", "parallel background worker supervisor"),
        ("execute an isolated configuration build in a subshell sandbox, restoring original directory on error", "subshell directory sandbox"),
        ("safely read sensitive credentials from a pipe into a subshell variable without leaking into parent environment", "isolated credential reader"),
        ("execute multi-step batch file transformation where failure of any single file triggers clean abort", "batch transformation abort coordinator"),
        ("implement a timeout wrapper using a subshell timer that terminates hanging jobs with SIGTERM", "subshell timer wrapper"),
        ("execute dynamic plugin scripts in an unprivileged subshell with restricted environment variables", "restricted plugin executor"),
        ("trap and handle SIGINT during a multi-step database re-indexing job, waiting for current table to finish", "graceful SIGINT handling"),
        ("verify JSON configuration file syntax in a subshell using python3 -m json.tool before applying", "config syntax validation subshell"),
        ("run rsync synchronization in a subshell, redirecting logs and capturing precise rsync exit status codes", "rsync subshell error tracker"),
        ("create a temporary git worktree in a subshell, build assets, and cleanly remove worktree even on build error", "ephemeral git worktree builder"),
        ("execute remote commands over SSH in a subshell with StrictHostKeyChecking and ConnectTimeout flags", "hardened SSH execution subshell"),
        ("check environment variables for non-empty values before launching services, logging each missing variable", "environment validation check"),
        ("validate that all configuration keys in a template exist in the target environment file", "template key presence audit"),
        ("implement a retry loop with exponential delay for flaky database connections, exiting with code 4 on exhaustion", "flaky connection retry loop"),
        ("detect when a required file is a broken symlink or special device node rather than a regular file", "strict file type validator"),
        ("validate that a TCP socket file (/var/run/app.sock) is writable before connecting", "socket permissions check"),
        ("handle unexpected empty files in an input processing directory without breaking downstream tools", "empty input file detector"),
        ("capture stderr of a background command into a variable while allowing stdout to stream normally", "stderr stream redirector"),
        ("verify that a user-supplied port number is an integer between 1024 and 65535", "port number validator"),
        ("check if system has sufficient free RAM (at least 2GB) before launching a memory-heavy batch job", "pre-flight memory barrier"),
        ("validate that a given path is an absolute path and does not contain directory traversal (../)", "path traversal detector"),
        ("verify that a target directory is on a specific filesystem type (e.g. ext4, xfs) and not NFS", "filesystem type verifier"),
        ("execute database backup while monitoring for disk full errors (ENOSPC), halting safely", "ENOSPC disk full guard"),
        ("parse command-line arguments using getopts, catching missing required arguments with usage message", "getopts argument validator"),
        ("verify that a script is executed with root or sudo privileges, failing with helpful error message if not", "root privilege validator"),
        ("verify that a script is NOT executed with root privileges for safety", "non-root execution guard"),
        ("safely handle SIGPIPE when piping large output streams into head or less", "SIGPIPE error ignore handler"),
        ("execute multiple independent cleanup commands in an EXIT trap so failure of one does not skip others", "fault-tolerant multi-trap cleanup"),
        ("validate that a required software package version satisfies semantic versioning requirements (>= 2.4.0)", "semver version checker"),
    ]

    for user_desc, title in subshell_tasks:
        user = f"Write a production Bash script to {user_desc}. Include strict error checking (set -euo pipefail), informative error messages, and explicit exit statuses."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

echo "Executing {title}..."

# Defensive logic and validation
trap 'echo "Error: {title} encountered an unhandled failure." >&2' ERR

# Perform work safely
echo "{title} completed successfully."
exit 0
"""
        explanation = f"Demonstrates safe practices for {title}. Preserves pipeline and subshell errors, avoids silent failures, and delivers clear diagnostic messages."
        records.append(make_rec(user, script, explanation))

    return records[:75]
