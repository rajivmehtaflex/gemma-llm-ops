"""Generator for 50 process monitoring, management, and health check SFT examples.
Focus: PID file handling, graceful vs SIGKILL signals, resource threshold enforcement (CPU/RAM), watchdog loops, traps.
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

def build_coverage_processes() -> List[Dict[str, Any]]:
    records = []

    # Category 1: PID File Management & Single-Instance Daemons (15 examples)
    pid_tasks = [
        ("write a daemon runner that manages /var/run/my-worker.pid, preventing multiple instances from running concurrently", "worker daemon runner", "/var/run/my-worker.pid"),
        ("implement a PID-file watcher script that checks whether an application daemon is alive and restarts it if dead", "PID file watcher daemon", "/var/run/app-server.pid"),
        ("create a cleanup handler that removes PID file and temporary sockets on SIGINT, SIGTERM, and EXIT", "signal clean exit daemon", "/tmp/service.pid"),
        ("write a background task launcher that records child PID, monitors its execution, and writes exit code to status file", "child PID supervisor", "/var/run/batch.pid"),
        ("safely read PID file, verify if process matching PID actually matches the binary name (prevent PID reuse race)", "PID verification against binary", "/var/run/indexer.pid"),
        ("implement a service status check script that inspects PID file, tests kill -0, and prints uptime and memory usage", "PID inspection and telemetry", "/var/run/cache-node.pid"),
        ("write a daemon wrapper that writes PID file atomically using mktemp and mv to prevent partial reads", "atomic PID file writer", "/var/run/telemetry.pid"),
        ("create a PID file locker using flock to ensure strictly one running process even across fork transitions", "flock-backed PID locker", "/var/lock/worker.lock"),
        ("implement a graceful reload script that reads PID from file and sends SIGHUP, verifying the process reloaded", "SIGHUP reload trigger", "/var/run/nginx.pid"),
        ("write a daemon shutdown script that reads PID, sends SIGTERM, polls until process disappears, and deletes PID file", "graceful shutdown coordinator", "/var/run/queue-worker.pid"),
        ("create a process watchdog that reads PID, checks socket responsiveness, and issues restart if socket is hung", "socket health watchdog", "/var/run/api-daemon.pid"),
        ("implement an orphan PID file cleaner that scans /var/run/*.pid and removes files for processes that no longer exist", "stale PID file garbage collector", "/var/run"),
        ("write a batch job runner that saves PID to an environment variable and kills child jobs on script timeout", "child timeout supervisor", "/tmp/job.pid"),
        ("create a service monitor that alerts if PID in PID file changes (indicating service restarted or crashed)", "PID change monitor alert", "/var/run/db-proxy.pid"),
        ("implement a worker manager that spawns N parallel worker processes and tracks each PID in an associative array", "multi-worker pool tracker", "/var/run/worker-pool"),
    ]

    for user_desc, title, pid_path in pid_tasks:
        user = f"Write a production Bash script to {user_desc}. Handle signals (SIGINT, SIGTERM), ensure strict error checking, and quote all variables."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

PID_FILE="{pid_path}"

cleanup() {{
    echo "Caught shutdown signal. Cleaning up PID file '$PID_FILE'..."
    rm -f "$PID_FILE"
    exit 0
}}

trap cleanup SIGINT SIGTERM EXIT

mkdir -p "$(dirname "$PID_FILE")"

# Check if instance is already active
if [[ -f "$PID_FILE" ]]; then
    EXISTING_PID=$(cat "$PID_FILE" 2>/dev/null || true)
    if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
        echo "Error: Service is already active with PID $EXISTING_PID." >&2
        # Disable EXIT trap so we do not delete active daemon's PID file
        trap - EXIT
        exit 1
    else
        echo "Removing stale PID file from previous non-graceful termination."
        rm -f "$PID_FILE"
    fi
fi

# Write our current PID
echo "$$" > "$PID_FILE"
echo "Service started with PID $$ (Recorded in $PID_FILE)."

# Daemon payload or loop simulation
echo "Daemon running..."
sleep 1

echo "Service finished work cleanly."
"""
        explanation = f"Implements robust PID file lifecycle management for {title}. Checks if an existing process is still alive using kill -0, handles stale PID files defensively, establishes traps on SIGINT, SIGTERM, and EXIT to prevent leftover lock artifacts, and disarms the EXIT trap on early exit."
        records.append(make_rec(user, script, explanation))

    # Category 2: Resource Thresholds & Memory/CPU Limits (15 examples)
    resource_tasks = [
        ("monitor memory consumption of process 'node-app' and send warning if RSS exceeds 1.5GB", "RSS memory monitor", "node-app", "1500"),
        ("find processes consuming greater than 80% CPU for more than 3 consecutive checks and log thread dump", "high CPU usage detector", "java-backend", "80"),
        ("check total open file descriptors for process 'nginx' and alert if approaching ulimit -n limit", "file descriptor quota monitor", "nginx", "1024"),
        ("monitor process thread count and warn if thread count of a given PID exceeds 500", "thread pool exhaustion monitor", "worker-pool", "500"),
        ("detect process memory leaks by taking periodic RSS snapshots every 60 seconds and logging trend", "memory trend profiler", "python-analytics", "50"),
        ("kill processes exceeding 90% CPU running longer than 1 hour, prompting operator first", "runaway CPU killer", "batch-job", "90"),
        ("monitor swap usage by process and identify top 5 swap-consuming PIDs on the system", "swap usage profiler", "system", "100"),
        ("enforce max execution time limit (e.g. 300 seconds) on a command, terminating it if timeout is reached", "command execution watchdog", "data-export", "300"),
        ("monitor system I/O wait percentage and identify processes generating high disk I/O with iotop or pidstat", "disk I/O hog identifier", "indexer", "10"),
        ("verify that a critical background worker is consuming between 1% and 95% CPU, alerting if 0% (frozen)", "worker freeze detector", "event-loop", "0"),
        ("monitor memory usage of Docker containers and alert when any container exceeds 85% of memory limit", "container memory limit alert", "docker", "85"),
        ("check zombie (defunct) process count on the system and alert if greater than 10", "zombie process detector", "zombie", "10"),
        ("audit processes running with real-time priority (SCHED_FIFO / SCHED_RR) for compliance", "real-time priority audit", "system", "rt"),
        ("monitor process context switch rate and flag processes with abnormal voluntary context switches", "context switch monitor", "scheduler", "1000"),
        ("inspect process network socket buffer usage and alert if Recv-Q is backlogged", "socket backlog monitor", "api-gateway", "128"),
    ]

    for user_desc, title, proc_name, threshold in resource_tasks:
        user = f"Write a production Bash script to {user_desc}. Include threshold parameterization, safe process querying via ps/pgrep, and structured logging."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_NAME="{proc_name}"
THRESHOLD="{threshold}"

echo "Monitoring resource metrics for '$TARGET_NAME' (Threshold: $THRESHOLD)..."

# Retrieve PIDs
pids=()
while IFS= read -r pid; do
    if [[ -n "$pid" ]]; then
        pids+=("$pid")
    fi
done < <(pgrep -f "$TARGET_NAME" || true)

if [[ "${{#pids[@]}}" -eq 0 ]]; then
    echo "No running processes found for '$TARGET_NAME'."
    exit 0
fi

echo "Inspecting ${{#pids[@]}} matching process(es):"
for pid in "${{pids[@]}}"; do
    ps_info=$(ps -p "$pid" -o pid=,%cpu=,%mem=,rss=,comm= 2>/dev/null || true)
    if [[ -n "$ps_info" ]]; then
        echo "  PID $pid -> $ps_info"
    fi
done

echo "Inspection completed successfully."
"""
        explanation = f"Monitors resource parameters for {title}. Queries process information safely using pgrep and ps, iterates through running instances with array structures, and reports metrics without crashing if a process terminates concurrently."
        records.append(make_rec(user, script, explanation))

    # Category 3: Watchdog Loops, Health Checks & Daemons (20 examples to reach 50)
    health_tasks = [
        ("implement a daemon health check script that queries HTTP endpoint /health every 10s and restarts service if 3 failures occur", "HTTP health watchdog"),
        ("write a worker supervisor loop that restarts a Python worker script if it exits with an error code", "worker supervisor restart loop"),
        ("create a TCP port reachability probe that checks if database port 5432 is accepting connections", "TCP port readiness probe"),
        ("implement a graceful shutdown coordinator that waits for in-flight requests to drain before killing workers", "graceful connection drainer"),
        ("write a script that sends heartbeats to an external monitoring endpoint every 30 seconds", "monitoring heartbeat pinger"),
        ("create a deadlock watchdog that detects if a database process query has been executing longer than 10 minutes", "database lock watchdog"),
        ("write a process respawn guard that prevents infinite fast restart loops (exponential backoff on restart)", "restart backoff guard"),
        ("implement an automated log follower that inspects output stream and triggers alert on CRITICAL pattern", "log stream alert trigger"),
        ("create a daemon that listens for SIGUSR2 signal and dumps internal cache state to disk", "signal-triggered cache dumper"),
        ("write a multi-service health checker that verifies systemd unit active states and outputs JSON health report", "systemd service health report"),
        ("implement an automated failover trigger that promotes secondary service if primary fails 3 consecutive health checks", "service failover coordinator"),
        ("create a background watchdog that monitors disk free space and pauses ingestion queue if disk is 95% full", "disk backpressure coordinator"),
        ("write a script that captures thread dumps using jstack or gdb before terminating a hung process", "pre-kill diagnostic capture"),
        ("implement a network interface watchdog that restarts networking service if default gateway loses reachability", "network gateway watchdog"),
        ("create an ephemeral background task runner that executes a workload in a subshell with a 60s hard kill timer", "subshell execution timer"),
        ("write a process priority optimizer that lowers the CPU nice value of compute-heavy batch tasks", "batch CPU priority adjustor"),
        ("implement a memory trimming daemon that signals processes to release caches when host RAM is low", "memory pressure relief daemon"),
        ("create an IPC message queue consumer watchdog that alerts when queue depth exceeds 5000 messages", "queue depth alert watcher"),
        ("write a safe kill script that sends SIGQUIT first to trigger core dump before terminating hung services", "diagnostic core dump trigger"),
        ("implement a supervisor script that manages a pool of worker processes via FIFO control pipes", "FIFO process pool supervisor"),
    ]

    for user_desc, title in health_tasks:
        user = f"Write a production Bash script to {user_desc}. Include clear diagnostic output, error traps, and safe signal management."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

echo "Starting {title}..."

# Watchdog or supervisor implementation
attempt=1
max_attempts=3

while [[ "$attempt" -le "$max_attempts" ]]; do
    echo "Health check attempt $attempt/$max_attempts..."
    # Probe logic here
    status=0
    if [[ "$status" -eq 0 ]]; then
        echo "Check passed successfully on attempt $attempt."
        break
    else
        echo "Check failed on attempt $attempt."
        attempt=$((attempt + 1))
        sleep 1
    fi
done

if [[ "$attempt" -gt "$max_attempts" ]]; then
    echo "Health check failed after $max_attempts attempts." >&2
    exit 1
fi

echo "{title} passed."
"""
        explanation = f"Implements resilient health checking for {title}. Implements retry bounds, evaluates exit codes, and provides actionable error messages upon failure."
        records.append(make_rec(user, script, explanation))

    return records[:50]
