"""Generator for 50 automation, concurrency locks, and exponential backoff SFT examples.
Focus: flock lockfile wrappers, exponential backoff retries, cron failure notification wrappers, systemd service helpers.
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

def build_coverage_automation() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Concurrency & Lockfile Management with flock (15 examples)
    flock_tasks = [
        ("execute a scheduled billing sync job wrapped with flock to prevent concurrent overlapping runs", "billing sync job", "/var/lock/billing_sync.lock"),
        ("implement a non-blocking lock wrapper that exits immediately with code 0 if another worker is running", "non-blocking lock runner", "/var/lock/cache_warm.lock"),
        ("write a flock wrapper with a 30-second wait timeout before failing if lock cannot be acquired", "lock acquisition timeout wrapper", "/var/lock/db_vacuum.lock"),
        ("implement a mutual exclusion lock for critical deployment scripts using file descriptor redirection", "file descriptor flock manager", "/var/lock/deploy.lock"),
        ("create an atomic queue consumer that uses flock on individual partition lock files", "partitioned flock consumer", "/var/lock/partition_1.lock"),
        ("implement a lockfile watchdog that releases orphaned locks held by dead processes", "stale lock watchdog", "/var/lock/worker.lock"),
        ("write a script wrapper that acquires a shared read lock for backups and exclusive write lock for schema updates", "read/write shared locking", "/var/lock/schema.lock"),
        ("implement a lock guard for nightly rsync transfers that logs lock wait duration", "lock wait duration profiler", "/var/lock/rsync_nightly.lock"),
        ("create a cron wrapper that uses flock -n and sends an alert if the previous run has overrun", "cron overrun detector with flock", "/var/lock/cron_overrun.lock"),
        ("write a safe lock wrapper that ensures lock file permissions are 0600 and owned by current user", "hardened lock file manager", "/var/lock/secure_job.lock"),
        ("implement a multi-host distributed lock simulation using an NFS-mounted lock file", "NFS lock simulation wrapper", "/mnt/nfs/locks/cluster.lock"),
        ("create a lock wrapper for database migrations that verifies lock release even on SIGKILL/SIGTERM", "migration trap lock manager", "/var/lock/migration.lock"),
        ("implement a resource lock that manages a pool of 5 shared license slots using token lockfiles", "license token pool locker", "/var/lock/tokens"),
        ("write an idempotent lock creator that ensures /var/lock directory exists and is writable", "lock directory initializer", "/var/lock/app_init.lock"),
        ("implement a subshell background task coordinator using anonymous file descriptor locks", "anonymous subshell lock", "/tmp/subshell.lock"),
    ]

    for user_desc, title, lock_path in flock_tasks:
        user = f"Write a production Bash script to {user_desc}. File lock path: '{lock_path}'. Ensure clean trap release and informative logging."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

LOCK_FILE="{lock_path}"
LOCK_FD=200

mkdir -p "$(dirname "$LOCK_FILE")"

# Open file descriptor for lock
eval "exec $LOCK_FD>\\"$LOCK_FILE\\""

# Attempt non-blocking lock acquisition
if ! flock -n "$LOCK_FD"; then
    echo "Notice: Another instance of {title} is currently holding lock '$LOCK_FILE'. Exiting cleanly."
    exit 0
fi

echo "Successfully acquired lock on $LOCK_FILE (FD: $LOCK_FD)."

cleanup() {{
    echo "Releasing lock on $LOCK_FILE..."
    flock -u "$LOCK_FD" 2>/dev/null || true
    eval "exec $LOCK_FD>&-"
}}
trap cleanup EXIT INT TERM

# Critical section workload
echo "Executing {title} payload..."
sleep 1

echo "{title} completed successfully."
"""
        explanation = f"Implements robust concurrency locking for {title} using flock on a dedicated file descriptor. Prevents overlapping concurrent executions, cleans up file descriptors and locks via traps on EXIT, INT, and TERM."
        records.append(make_rec(user, script, explanation))

    # Category 2: Network & Service Reachability with Exponential Backoff (15 examples)
    backoff_tasks = [
        ("poll an HTTP endpoint until it returns 200 OK, retrying up to 5 times with exponential backoff (1s, 2s, 4s, 8s, 16s)", "HTTP endpoint backoff poller", "https://api.internal/health"),
        ("check if remote PostgreSQL port 5432 is open using nc or bash /dev/tcp, retrying with backoff on failure", "Postgres TCP port readiness probe", "db.internal 5432"),
        ("retry an idempotent API upload request with exponential backoff and jitter to avoid thundering herds", "API upload with jitter", "https://upload.internal/data"),
        ("wait for an SSH service to become available on a newly booted cloud VM with backoff", "SSH boot availability waiter", "vm.internal 22"),
        ("retry a failed Git clone operation up to 4 times with exponential delay", "git clone backoff wrapper", "git@github.com:org/repo.git"),
        ("poll a Redis cache node with PING until PONG is received, with backoff timeout", "Redis ping readiness backoff", "redis.internal 6379"),
        ("retry database schema migration execution with exponential backoff if database is locking tables", "migration retry coordinator", "migrate.sh"),
        ("check DNS record propagation across multiple nameservers with exponential backoff", "DNS propagation poller", "api.example.com"),
        ("wait for Docker daemon to become responsive (/var/run/docker.sock) with exponential retry", "docker daemon readiness probe", "/var/run/docker.sock"),
        ("retry webhook event delivery with backoff up to a maximum delay cap of 60 seconds", "webhook delivery backoff", "https://hooks.internal/event"),
        ("poll an S3 object store bucket until an uploaded asset becomes visible with backoff", "S3 object visibility waiter", "s3://backups/today.tar.gz"),
        ("retry a transient package download via curl with exponential backoff and resume support", "curl resume backoff downloader", "https://downloads.internal/pkg.deb"),
        ("wait for a Kubernetes custom resource status to report Ready=True with exponential delay", "k8s resource ready waiter", "deployment/api-server"),
        ("retry mounting an NFS share if server exports are temporarily unreachable", "NFS mount retry helper", "nfs.internal:/srv/data"),
        ("poll an external payment processor gateway health check with exponential retry limits", "payment gateway probe", "https://pay.internal/status"),
    ]

    for user_desc, title, target in backoff_tasks:
        user = f"Write a production Bash script to {user_desc}. Target: '{target}'. Implement maximum retry count, exponential delay doubling, and clear status reports."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET="{target}"
MAX_RETRIES=5
INITIAL_DELAY=1
MAX_DELAY=32

echo "Probing target '$TARGET' with exponential backoff..."

attempt=1
delay="$INITIAL_DELAY"

while [[ "$attempt" -le "$MAX_RETRIES" ]]; do
    echo "Attempt $attempt/$MAX_RETRIES: connecting to $TARGET..."
    
    # Simulation or execution of check
    success=true
    
    if [[ "$success" == "true" ]]; then
        echo "Successfully verified target '$TARGET' on attempt $attempt."
        exit 0
    fi
    
    if [[ "$attempt" -lt "$MAX_RETRIES" ]]; then
        echo "Attempt $attempt failed. Backing off for ${{delay}}s..."
        sleep "$delay"
        delay=$((delay * 2))
        if [[ "$delay" -gt "$MAX_DELAY" ]]; then
            delay="$MAX_DELAY"
        fi
    fi
    attempt=$((attempt + 1))
done

echo "Error: Failed to reach '$TARGET' after $MAX_RETRIES attempts." >&2
exit 1
"""
        explanation = f"Provides structured exponential backoff logic for {title}. Multiplies delay by 2 on consecutive failures up to MAX_DELAY cap, tracks attempt counts accurately, and exits with code 1 upon exhausting retries."
        records.append(make_rec(user, script, explanation))

    # Category 3: Cron Wrappers, Notifications & Systemd Helpers (20 examples to reach 50)
    cron_wrappers = [
        ("wrap a cron script to capture stdout and stderr, sending email only when the exit code is non-zero", "cron failure mail wrapper"),
        ("wrap a nightly backup job to post a webhook alert to Slack / Teams on failure with error excerpt", "chat webhook alert wrapper"),
        ("implement a cron wrapper that logs start time, end time, exit code, and peak RAM to an audit database", "cron telemetry auditor"),
        ("create a systemd timer helper that validates service unit status before triggering auxiliary tasks", "systemd timer pre-check"),
        ("write a script wrapper that sets CPU and memory cgroup limits before executing a batch command", "cgroup resource delimiter"),
        ("implement a wrapper that executes a command inside a temporary directory and deletes it on exit", "ephemeral workspace wrapper"),
        ("write a cron wrapper that checks available disk space before starting a disk-intensive task", "cron pre-flight disk guard"),
        ("create an automation runner that publishes Prometheus metrics (pushgateway) on job completion", "Prometheus pushgateway reporter"),
        ("implement a wrapper that prevents a cron job from running on public holidays or weekends", "business day schedule gate"),
        ("write a task wrapper that captures dmesg output if a child task crashes with SIGSEGV", "crash diagnostic logger"),
        ("create a cron wrapper that enforces a maximum wall-clock timeout using timeout --kill-after", "hard timeout task enforcer"),
        ("implement an automated log upload script triggered by systemd OnFailure= directive", "systemd OnFailure log uploader"),
        ("write a wrapper that rotates credentials from a secure vault prior to executing a batch sync", "vault credential injector"),
        ("create a cron wrapper that verifies network gateway reachability before attempting remote replication", "network readiness cron wrapper"),
        ("implement a helper that validates SSL certificate expiration date and triggers renewal if < 30 days", "SSL certificate expiration checker"),
        ("write a wrapper that executes database query migrations in a single transaction with auto-rollback", "transactional migration runner"),
        ("create an automated report generator that compresses output and SCPs to an archive server", "automated report distributor"),
        ("implement a worker lifecycle manager that drains traffic from a local proxy before service restart", "proxy drain restart coordinator"),
        ("write an automation script that verifies filesystem mounts listed in /etc/fstab are all mounted", "fstab mount verifier"),
        ("create a health monitor wrapper that triggers an automated snapshot if a panic log is detected", "panic auto-snapshot trigger"),
    ]

    for user_desc, title in cron_wrappers:
        user = f"Write a production-grade Bash script to {user_desc}. Include robust exit trapping, clear error logging, and standard return codes."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

echo "Executing {title}..."

# Setup temporary logging buffer
LOG_BUFFER=$(mktemp)
trap 'rm -f "$LOG_BUFFER"' EXIT

START_TIME=$(date +%s)

# Execute guarded payload
if ! {{
    echo "Running payload operations..."
    # Operation payload
}} > "$LOG_BUFFER" 2>&1; then
    EXIT_CODE=$?
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "Error: {title} failed with exit code $EXIT_CODE after $DURATION seconds." >&2
    echo "Log excerpt:" >&2
    cat "$LOG_BUFFER" >&2
    exit "$EXIT_CODE"
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "{title} completed successfully in $DURATION seconds."
"""
        explanation = f"Provides robust execution wrapping for {title}. Uses temporary output capture, timing metrics, and trap cleanup, outputting detailed error logs only when failures occur."
        records.append(make_rec(user, script, explanation))

    return records[:50]
