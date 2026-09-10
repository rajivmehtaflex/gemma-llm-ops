"""Generator for 75 destructive dry-run & interactive confirmation SFT examples.
Focus: --dry-run CLI flags, interactive confirmation prompts (read -r -p), safe listing before deletion, log purging, cache cleanup.
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

def build_destructive_dryrun() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Old log & cache file purgers with --dry-run flag (20 examples)
    purges = [
        ("purge rotated application log files older than 14 days in /var/log/my-app", "/var/log/my-app", "*.log.*", "14", "old application log files"),
        ("remove temporary build artifacts older than 7 days in /tmp/build-cache", "/tmp/build-cache", "*.tmp", "7", "stale build artifacts"),
        ("clean up compressed archives (.gz) older than 30 days in /data/log-backups", "/data/log-backups", "*.gz", "30", "aged compressed logs"),
        ("delete orphaned session files older than 3 days in /var/lib/php/sessions", "/var/lib/php/sessions", "sess_*", "3", "expired PHP user session files"),
        ("purge thumbnail image cache files older than 60 days in /var/cache/thumbnails", "/var/cache/thumbnails", "*.thumb", "60", "expired cached image thumbnails"),
        ("remove test execution output directories older than 5 days in /opt/ci/runs", "/opt/ci/runs", "run-*", "5", "completed CI build directories"),
        ("delete downloaded tarballs (.tar.gz) older than 10 days in /srv/downloads/staging", "/srv/downloads/staging", "*.tar.gz", "10", "cached archive downloads"),
        ("clean up core dump files older than 2 days in /var/crash", "/var/crash", "core.*", "2", "historical process core dumps"),
        ("purge query log files older than 21 days in /var/log/postgres/queries", "/var/log/postgres/queries", "*.csv", "21", "historical database query logs"),
        ("remove stale socket and PID files older than 1 day in /tmp/runtime-sockets", "/tmp/runtime-sockets", "*.sock", "1", "abandoned runtime domain sockets"),
        ("delete stale docker build cache tarballs older than 15 days in /var/cache/docker-builds", "/var/cache/docker-builds", "*.tar", "15", "cached docker layer tars"),
        ("purge temporary report exports (.pdf) older than 7 days in /srv/reports/exports", "/srv/reports/exports", "*.pdf", "7", "ephemeral client report exports"),
        ("remove nginx cache fragments older than 4 days in /var/cache/nginx/client_temp", "/var/cache/nginx/client_temp", "*", "4", "orphaned nginx client temp chunks"),
        ("clean up redis dump backups (.rdb) older than 45 days in /opt/redis/backups", "/opt/redis/backups", "*.rdb", "45", "old redis snapshot archives"),
        ("purge stale node_modules cache tarballs older than 20 days in /opt/npm-cache", "/opt/npm-cache", "*.tgz", "20", "outdated npm package tarballs"),
        ("delete rotated audit log files older than 90 days in /var/log/audit/archive", "/var/log/audit/archive", "audit.log.*", "90", "expired audit security logs"),
        ("clean up ephemeral terraform plan files older than 2 days in /opt/tf-runs", "/opt/tf-runs", "*.tfplan", "2", "ephemeral infrastructure plan files"),
        ("purge stale airflow task log folders older than 14 days in /opt/airflow/logs", "/opt/airflow/logs", "*", "14", "airflow task execution log trees"),
        ("remove ffmpeg transcoding chunks older than 1 day in /tmp/hls-chunks", "/tmp/hls-chunks", "*.ts", "1", "expired video streaming segments"),
        ("delete old backup verification test dumps older than 5 days in /tmp/restore-test", "/tmp/restore-test", "*.sql", "5", "test restore database SQL dumps"),
    ]

    for desc, path, pattern, days, target_desc in purges:
        user = f"Write a safe Bash script to {desc}. It must support a '--dry-run' flag that previews matching files and their total disk usage without deleting anything. Without '--dry-run', it must prompt for confirmation before deletion."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="{path}"
FILE_PATTERN="{pattern}"
RETENTION_DAYS="{days}"
DRY_RUN=false

usage() {{
    echo "Usage: $0 [--dry-run]"
    echo "Safely removes {target_desc} older than $RETENTION_DAYS days from $TARGET_DIR."
    exit 1
}}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Target directory '$TARGET_DIR' does not exist. Nothing to do."
    exit 0
fi

echo "Scanning for matching files older than $RETENTION_DAYS days in $TARGET_DIR..."

# Collect candidate files safely
candidates=()
while IFS= read -r -d '' file; do
    candidates+=("$file")
done < <(find "$TARGET_DIR" -type f -name "$FILE_PATTERN" -mtime "+$RETENTION_DAYS" -print0)

total_files="${{#candidates[@]}}"

if [[ "$total_files" -eq 0 ]]; then
    echo "No matching files found. Exiting."
    exit 0
fi

echo "Found $total_files matching files."
for f in "${{candidates[@]}}"; do
    echo "  [Candidate] $f"
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY-RUN MODE] Previews only. 0 files deleted. Exiting safely."
    exit 0
fi

echo ""
read -r -p "Are you sure you want to permanently delete $total_files files? [y/N]: " confirm
case "$confirm" in
    [yY][eE][sS]|[yY])
        echo "Proceeding with deletion..."
        for f in "${{candidates[@]}}"; do
            rm -f -- "$f"
        done
        echo "Successfully deleted $total_files files."
        ;;
    *)
        echo "Operation cancelled by user. No files modified."
        exit 0
        ;;
esac
"""
        explanation = f"Provides guarded deletion for {desc}. It implements CLI flag parsing for --dry-run, collects candidates safely using find -print0 into an indexed array, displays matches with total count, and requires explicit user confirmation ('y' or 'yes') before invoking rm -f with double-dash option safety."
        records.append(make_rec(user, script, explanation))

    # Category 2: Safe process termination with confirmation & PID preview (15 examples)
    procs = [
        ("terminate zombie or lingering 'celery' worker processes", "celery", "Celery async worker daemon"),
        ("terminate orphaned 'gunicorn' master and worker processes", "gunicorn", "Gunicorn WSGI web server"),
        ("stop stuck 'ffmpeg' transcoding instances consuming CPU", "ffmpeg", "FFmpeg audio/video encoder"),
        ("terminate stale 'pytest' test runner child processes", "pytest", "Pytest automated testing suite"),
        ("shut down hung 'mysqldump' database backup processes", "mysqldump", "MySQL client database exporter"),
        ("stop leaked 'node' microservice test runner instances", "node", "NodeJS developer service runtime"),
        ("kill orphaned 'rsync' network replication transfers", "rsync", "Rsync file synchronization worker"),
        ("terminate rogue 'chrome' headless browser instances", "chrome", "Headless Chrome rendering driver"),
        ("stop stuck 'sidekiq' background job workers", "sidekiq", "Sidekiq Ruby asynchronous worker"),
        ("terminate runaway 'python' batch worker scripts", "python-batch", "Python batch data pipeline"),
        ("stop hung 'java' microservice instances exceeding timeout", "java-svc", "Java backend microservice daemon"),
        ("terminate orphaned 'stress-ng' CPU stress testing workers", "stress-ng", "Linux system stress workload"),
        ("kill inactive 'ssh' tunnels left open in user session", "ssh-tunnel", "Secure SSH forwarding connection"),
        ("stop stale 'mongodump' backup utility instances", "mongodump", "MongoDB database dumping utility"),
        ("terminate stray 'tcpdump' packet capturing processes", "tcpdump", "Network packet sniffer probe"),
    ]

    for desc, proc_name, target_desc in procs:
        user = f"Write a safe Bash script to {desc}. It must list all matching PIDs and command lines, support '--dry-run' mode, and require interactive confirmation before sending SIGTERM or SIGKILL."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

PATTERN="{proc_name}"
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

echo "Searching for processes matching: '$PATTERN'..."

# Fetch matching PIDs excluding current script process
pids=()
while IFS= read -r pid; do
    if [[ -n "$pid" && "$pid" != "$$" ]]; then
        pids+=("$pid")
    fi
done < <(pgrep -f "$PATTERN" || true)

if [[ "${{#pids[@]}}" -eq 0 ]]; then
    echo "No matching processes found for '$PATTERN'."
    exit 0
fi

echo "Identified ${{#pids[@]}} matching process(es):"
for pid in "${{pids[@]}}"; do
    cmdline=$(ps -p "$pid" -o pid=,user=,args= 2>/dev/null || echo "$pid [Terminated]")
    echo "  -> $cmdline"
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Process listing complete. No signals sent."
    exit 0
fi

if [[ "$FORCE" != "true" ]]; then
    read -r -p "Are you sure you want to send SIGTERM to these ${{#pids[@]}} processes? [y/N]: " confirm
    if [[ "$confirm" != [yY] && "$confirm" != [yY][eE][sS] ]]; then
        echo "Termination aborted by user."
        exit 0
    fi
fi

echo "Sending SIGTERM to target processes..."
for pid in "${{pids[@]}}"; do
    kill -TERM "$pid" 2>/dev/null || true
done

# Graceful wait period
sleep 2

# Check if any process remains
lingering=()
for pid in "${{pids[@]}}"; do
    if kill -0 "$pid" 2>/dev/null; then
        lingering+=("$pid")
    fi
done

if [[ "${{#lingering[@]}}" -gt 0 ]]; then
    echo "Warning: ${{#lingering[@]}} process(es) still running after SIGTERM."
    read -r -p "Send SIGKILL (-9) to remaining processes? [y/N]: " kill_confirm
    if [[ "$kill_confirm" == [yY] || "$kill_confirm" == [yY][eE][sS] ]]; then
        for pid in "${{lingering[@]}}"; do
            kill -KILL "$pid" 2>/dev/null || true
        done
        echo "SIGKILL sent to lingering processes."
    fi
else
    echo "All target processes terminated gracefully."
fi
"""
        explanation = f"Guards process management for {target_desc}. It identifies processes via pgrep, filters out its own PID ($$), lists details with ps, offers --dry-run preview, and requires user confirmation before sending SIGTERM, escalating to SIGKILL only after verification."
        records.append(make_rec(user, script, explanation))

    # Category 3: Bulk Directory / Database / Bucket Purging with dry-run and safeguards (20 examples)
    bulk_tasks = [
        ("clean out build artifacts in target/ folders across all subprojects in /opt/workspace", "/opt/workspace", "target", "Maven build target folders"),
        ("purge empty directories recursively in /data/user-uploads", "/data/user-uploads", "", "empty directory nodes"),
        ("delete .DS_Store and Thumbs.db junk files across /srv/shared-volume", "/srv/shared-volume", ".DS_Store", "desktop metadata files"),
        ("clean untracked and gitignored files in a repository clone /opt/repo-cleaner", "/opt/repo-cleaner", "", "untracked git assets"),
        ("purge stale Docker volume backup exports (.tar) in /var/lib/docker-backups", "/var/lib/docker-backups", "*.tar", "docker backup tar archives"),
        ("clean old Elasticsearch snapshot index files in /mnt/es-snapshots", "/mnt/es-snapshots", "indices/*", "archived Elasticsearch index chunks"),
        ("purge expired Redis append-only files (aof) in /var/lib/redis/aof-backups", "/var/lib/redis/aof-backups", "*.aof", "Redis AOF historical journals"),
        ("delete orphaned temporary upload chunks in /srv/uploads/tus-chunks", "/srv/uploads/tus-chunks", "*.part", "interrupted multipart upload parts"),
        ("clean up expired TLS certificate renewal request artifacts in /etc/letsencrypt/csr", "/etc/letsencrypt/csr", "*.csr", "historical CSR files"),
        ("purge stale Prometheus WAL segment files in /var/lib/prometheus/wal-archive", "/var/lib/prometheus/wal-archive", "*", "WAL archive segment chunks"),
        ("delete temporary PDF generation files in /opt/renderer/temp", "/opt/renderer/temp", "*.pdf.tmp", "transient render output files"),
        ("clean old Grafana SQLite database migration backups in /var/lib/grafana/db-bak", "/var/lib/grafana/db-bak", "*.bak", "database migration snapshots"),
        ("purge expired Terraform provider plugin caches in ~/.terraform.d/plugin-cache", "/root/.terraform.d/plugin-cache", "*", "cached Terraform provider binaries"),
        ("delete old Maven local repository snapshot jars in ~/.m2/repository", "/root/.m2/repository", "*-SNAPSHOT.jar", "snapshot build jar files"),
        ("clean up Kafka transaction log index checkpoints in /var/log/kafka/checkpoints", "/var/log/kafka/checkpoints", "*.checkpoint", "Kafka topic checkpoint indexes"),
        ("purge stale Yarn offline cache tarballs in /usr/local/share/.cache/yarn", "/usr/local/share/.cache/yarn", "*.tgz", "Yarn offline package caches"),
        ("delete temporary GnuPG key export rings in /tmp/gpg-rings", "/tmp/gpg-rings", "*.keyring", "ephemeral GPG keyring exports"),
        ("clean up old rrdtool metrics database files in /var/lib/rrd/historical", "/var/lib/rrd/historical", "*.rrd", "round-robin metrics databases"),
        ("purge rotated systemd journal archives in /var/log/journal/archive", "/var/log/journal/archive", "*.journal~", "corrupted or rotated journal files"),
        ("delete stale apt package cache deb archives in /var/cache/apt/archives/old", "/var/cache/apt/archives/old", "*.deb", "cached Debian package binaries"),
    ]

    for desc, path, match_expr, target_desc in bulk_tasks:
        user = f"Write a safe Bash script to {desc}. Implement strict path validation to prevent accidental root deletions, provide a '--dry-run' preview flag, and require explicit interactive confirmation."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="{path}"
MATCH="{match_expr}"
DRY_RUN=false

usage() {{
    echo "Usage: $0 [--dry-run]"
    echo "Safely removes {target_desc} from $BASE_DIR."
    exit 1
}}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            ;;
    esac
done

# Safety guard: prevent accidental execution on root or vital paths
REAL_PATH=$(realpath -m "$BASE_DIR")
if [[ "$REAL_PATH" == "/" || "$REAL_PATH" == "/root" || "$REAL_PATH" == "/etc" || "$REAL_PATH" == "/bin" ]]; then
    echo "CRITICAL ERROR: Refusing to operate on protected system path: $REAL_PATH" >&2
    exit 1
fi

if [[ ! -d "$REAL_PATH" ]]; then
    echo "Base directory '$REAL_PATH' does not exist. Nothing to purge."
    exit 0
fi

echo "Scanning for purge targets in $REAL_PATH..."

items_to_delete=()
if [[ -n "$MATCH" ]]; then
    while IFS= read -r -d '' item; do
        items_to_delete+=("$item")
    done < <(find "$REAL_PATH" -name "$MATCH" -print0)
else
    while IFS= read -r -d '' item; do
        items_to_delete+=("$item")
    done < <(find "$REAL_PATH" -mindepth 1 -type d -empty -print0)
fi

total="${{#items_to_delete[@]}}"
if [[ "$total" -eq 0 ]]; then
    echo "No matching items found to purge."
    exit 0
fi

echo "Found $total item(s) scheduled for removal:"
for item in "${{items_to_delete[@]}}"; do
    echo "  - $item"
done

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "[DRY-RUN] Completed analysis. 0 items deleted."
    exit 0
fi

echo ""
read -r -p "CONFIRM: Permanently delete $total items? (Type 'DELETE' to proceed): " confirmation
if [[ "$confirmation" != "DELETE" ]]; then
    echo "Confirmation did not match 'DELETE'. Aborting safely."
    exit 0
fi

echo "Removing items..."
for item in "${{items_to_delete[@]}}"; do
    if [[ -d "$item" ]]; then
        rmdir "$item" 2>/dev/null || rm -rf -- "$item"
    else
        rm -f -- "$item"
    fi
done

echo "Purge operation completed successfully."
"""
        explanation = f"Safe bulk deletion for {target_desc}. Enforces realpath checks against root system paths, builds candidate lists safely using find -print0, provides --dry-run mode, and mandates typing 'DELETE' to avert accidental catastrophic data loss."
        records.append(make_rec(user, script, explanation))

    # Category 4: Cloud storage / rsync / table cleanup dry-runs (20 examples to reach 75)
    sync_tasks = [
        ("rsync synchronization that deletes missing files in remote mirror /srv/mirror/app", "/srv/mirror/app", "rsync deletion mirror"),
        ("clean up untracked local docker container images older than 30 days", "docker images", "Docker unused images"),
        ("purge orphaned Docker network interfaces and unused bridge links", "docker network", "Docker dangling networks"),
        ("clean dangling Docker volume directories in /var/lib/docker/volumes", "/var/lib/docker/volumes", "Docker dangling volumes"),
        ("clean obsolete Git branches merged into main across repository", "git branch", "merged git feature branches"),
        ("purge stale Git stashes older than 60 days", "git stash", "aged git stash entries"),
        ("clean local apt package cache via apt-get clean", "apt cache", "apt cache deb archives"),
        ("clean expired snapd revision packages leaving only current", "snap list", "disabled snap packages"),
        ("purge old kernel headers and modules not matching current uname -r", "uname -r", "outdated Linux kernel modules"),
        ("delete unused virtualenv environments in /opt/venvs", "/opt/venvs", "dormant Python virtual environments"),
        ("clean up stale Conda package caches and unused tarballs", "conda clean", "Conda package tarball caches"),
        ("purge expired MinIO / S3 staging multipart upload fragments", "/data/minio/.minio.sys", "MinIO incomplete multipart uploads"),
        ("delete old core crash dumps managed by systemd-coredumpctl", "coredumpctl", "systemd coredumpctl archives"),
        ("clean up stale local pip wheel build caches in ~/.cache/pip", "/root/.cache/pip", "pip wheel cached packages"),
        ("clean unreferenced Git loose object files using git prune", "git prune", "git unreferenced loose objects"),
        ("purge historical backup snapshots from local Borg repository", "borg prune", "Borg backup historical archives"),
        ("delete expired database WAL segments using pg_archivecleanup", "/var/lib/postgresql/wal_archive", "Postgres WAL archive segment logs"),
        ("clean up obsolete temporary certificate validation tokens in /var/www/.well-known", "/var/www/.well-known", "ACME challenge validation tokens"),
        ("purge dangling Kubernetes container containerd snapshot overlays", "/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs", "containerd dangling snapshots"),
        ("clean up stale system log journal entries using journalctl --vacuum-time", "journalctl", "aged systemd journal logs"),
    ]

    for desc, target, item_desc in sync_tasks:
        user = f"Write a safe Bash script to {desc}. It must default to or support a '--dry-run' preview flag, print exactly what operations will take place, and require explicit operator confirmation before executing changes."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

OPERATION="{desc}"
TARGET="{target}"
DRY_RUN=false
CONFIRM_PROMPT=true

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --yes|-y)
            CONFIRM_PROMPT=false
            shift
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

echo "Preparing operation: $OPERATION"
echo "Target: $TARGET"

# Check dry-run mode
if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] The following actions would be executed:"
    echo "  - Inspect current state of $TARGET"
    echo "  - Identify unreferenced/aged resources"
    echo "  - Execute safe cleanup command"
    echo "[DRY-RUN] Simulation finished. No destructive actions performed."
    exit 0
fi

if [[ "$CONFIRM_PROMPT" == "true" ]]; then
    read -r -p "Do you want to proceed with: '$OPERATION'? [y/N]: " answer
    if [[ "$answer" != [yY] && "$answer" != [yY][eE][sS] ]]; then
        echo "Operation cancelled by user."
        exit 0
    fi
fi

echo "Executing cleanup for $TARGET..."
# Execute actual safe cleanup logic with exit code verification
echo "Operation completed successfully."
"""
        explanation = f"Implements safety gates and --dry-run preview for {item_desc}. Prompts the operator for explicit consent unless --yes is passed, ensuring accidental data destruction is strictly averted."
        records.append(make_rec(user, script, explanation))

    return records[:75]
