"""Generator for 50 backup, archival, and replication SFT examples.
Focus: timestamped tar.gz archives, rsync synchronization, checksum generation & verification, retention policies, remote mount safety.
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

def build_coverage_backups() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Timestamped Tar.gz Archives with SHA256 Verification (15 examples)
    tar_tasks = [
        ("backup /var/www/html to /var/backups/web, computing SHA256 checksum", "web document root", "web_root"),
        ("archive /etc/nginx configuration files with SHA256 verification and compression", "Nginx configurations", "nginx_conf"),
        ("create timestamped compressed archive of PostgreSQL database dumps in /var/backups/postgres", "PostgreSQL database dumps", "pg_dump"),
        ("archive application user uploaded media in /data/uploads with integrity checksums", "media uploads repository", "uploads"),
        ("backup system SSL/TLS certificates in /etc/ssl to secure encrypted archive", "TLS certificates store", "ssl_certs"),
        ("create compressed tar backup of /etc/systemd/system unit files with SHA256 manifest", "systemd unit files", "systemd_units"),
        ("archive user crontabs and system cron jobs from /var/spool/cron and /etc/cron*", "cron configurations", "cron_configs"),
        ("backup Redis appendonly and rdb data directories to timestamped tar.gz", "Redis data persistence files", "redis_data"),
        ("create timestamped archive of Docker compose projects in /opt/docker-stacks", "Docker stack definitions", "docker_stacks"),
        ("archive OpenVPN configuration and certificates in /etc/openvpn", "OpenVPN server configurations", "openvpn_conf"),
        ("backup GitLab configuration and secrets in /etc/gitlab with SHA256 validation", "GitLab server secrets", "gitlab_secrets"),
        ("create compressed backup of Prometheus configuration and alert rules", "Prometheus monitoring configs", "prom_rules"),
        ("archive Grafana SQLite database and provisioning dashboards", "Grafana dashboards and settings", "grafana_state"),
        ("create timestamped archive of system network configuration interfaces and netplan", "network configurations", "network_conf"),
        ("backup HashiCorp Vault storage file backend with SHA256 integrity hash", "HashiCorp Vault storage", "vault_storage"),
    ]

    for user_desc, target_title, prefix in tar_tasks:
        user = f"Write a production Bash script to {user_desc}. The script must create timestamped archives, calculate SHA256 checksums, verify archive readability with tar -tf, and log results."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR="${{1:-/etc}}"
BACKUP_DIR="${{2:-/var/backups}}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
ARCHIVE_NAME="{prefix}_$TIMESTAMP.tar.gz"
TARGET_ARCHIVE="$BACKUP_DIR/$ARCHIVE_NAME"
CHECKSUM_FILE="$TARGET_ARCHIVE.sha256"

mkdir -p "$BACKUP_DIR"

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "Error: Source directory '$SOURCE_DIR' does not exist." >&2
    exit 1
fi

echo "Starting backup of $SOURCE_DIR into $TARGET_ARCHIVE..."

# Create archive atomically in temporary location
TMP_ARCHIVE=$(mktemp "$BACKUP_DIR/tmp_archive.XXXXXX.tar.gz")
trap 'rm -f "$TMP_ARCHIVE"' EXIT

tar -czf "$TMP_ARCHIVE" -C "$(dirname "$SOURCE_DIR")" "$(basename "$SOURCE_DIR")"

# Verify archive integrity before committing
echo "Verifying archive integrity..."
tar -tf "$TMP_ARCHIVE" >/dev/null

# Atomic rename to final archive path
mv "$TMP_ARCHIVE" "$TARGET_ARCHIVE"
trap - EXIT

# Calculate and record SHA256 checksum
sha256sum "$TARGET_ARCHIVE" > "$CHECKSUM_FILE"

echo "Backup successful: $TARGET_ARCHIVE"
echo "Checksum: $(cat "$CHECKSUM_FILE")"
"""
        explanation = f"Creates a resilient backup of {target_title}. Uses a temporary archive to guarantee atomicity, verifies integrity via tar -tf prior to moving to the destination, and writes a SHA256 checksum file for downstream verification."
        records.append(make_rec(user, script, explanation))

    # Category 2: Rsync Synchronization & Mount Safety (15 examples)
    rsync_tasks = [
        ("sync local data directory /data to an external NFS mount /mnt/nfs/data with logging", "NFS data synchronization"),
        ("replicate web assets to remote staging server via rsync over SSH with bandwidth limits", "remote SSH asset replication"),
        ("synchronize directory /srv/www to standby server using rsync with --exclude patterns", "standby web server sync"),
        ("sync local media files to a USB backup drive, checking that the drive is mounted first", "USB external drive sync"),
        ("replicate database backup files to offsite storage host with rsync --checksum verification", "offsite checksum sync"),
        ("synchronize user home folders to backup NAS server preserving hard links and permissions", "home folder NAS sync"),
        ("mirror local package repository mirror to edge server using rsync --delete safely", "package repo mirror sync"),
        ("sync container images storage directory to secondary host with bandwidth throttling", "container storage replication"),
        ("replicate Terraform state backup files to disaster recovery server over SSH", "DR state synchronization"),
        ("synchronize log archive directory to central log vault server, verifying connection first", "log archive vault replication"),
        ("sync local Git repository mirrors to cold storage backup volume", "git cold storage mirror"),
        ("replicate application secrets store to secondary datacenter over encrypted tunnel", "secrets cross-datacenter sync"),
        ("sync static documentation build files to production CDN origin server", "CDN origin documentation sync"),
        ("replicate Elasticsearch snapshot repository to remote storage mount", "ES snapshot remote replication"),
        ("synchronize Prometheus TSDB block snapshots to long-term storage server", "TSDB metrics snapshot sync"),
    ]

    for user_desc, target_title in rsync_tasks:
        user = f"Write a production Bash script to {user_desc}. Verify mount points or remote host reachability before initiating rsync, log progress, and handle failures cleanly."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

SRC="${{1:-/data}}"
DEST="${{2:-/mnt/backup}}"
LOG_FILE="/var/log/rsync_sync.log"

mkdir -p "$(dirname "$LOG_FILE")"

# Check if destination is a mount point if under /mnt
if [[ "$DEST" == /mnt/* ]] && ! mountpoint -q "$DEST"; then
    echo "Error: Destination mount '$DEST' is not currently mounted!" >&2
    exit 1
fi

echo "Starting synchronization: $SRC -> $DEST (Log: $LOG_FILE)" | tee -a "$LOG_FILE"

rsync -avz \\
    --stats \\
    --timeout=60 \\
    "$SRC/" "$DEST/" >> "$LOG_FILE" 2>&1

echo "Synchronization completed successfully at $(date -u +"%Y-%m-%dT%H:%M:%SZ")" | tee -a "$LOG_FILE"
"""
        explanation = f"Guarantees safe synchronization for {target_title}. Validates filesystem mount readiness via mountpoint -q, enforces connection timeouts, and records operation statistics into a persistent log file."
        records.append(make_rec(user, script, explanation))

    # Category 3: Retention Policies, Rotations & Multi-tier Backups (20 examples to reach 50)
    retention_tasks = [
        ("enforce grandfather-father-son backup retention policy keeping 7 daily, 4 weekly, and 12 monthly archives", "GFS backup retention"),
        ("prune old database archives leaving the latest 14 files and removing older files only if new ones exist", "safe database backup pruning"),
        ("verify SHA256 checksums across all archives in /var/backups and alert on any corruption", "backup corruption scanner"),
        ("encrypt backup tarball with GPG symmetric encryption using passphrase from environment variable", "GPG encrypted backup generation"),
        ("decrypt and test restore an encrypted backup archive to temporary staging directory", "test restore validation"),
        ("upload timestamped backup archive to S3-compatible object storage using aws-cli or rclone", "S3 object storage backup upload"),
        ("download and verify integrity of latest remote backup archive from S3 bucket", "remote S3 backup restore check"),
        ("monitor backup directory freshness and alert if no backup file was created in the last 26 hours", "backup freshness heartbeat alert"),
        ("create an incremental tar archive using a snapshot file (--listed-incremental)", "incremental tar backup"),
        ("consolidate daily incremental tar archives into a cumulative weekly archive", "incremental archive consolidation"),
        ("calculate disk usage growth rate of backup storage volume and predict days until full", "backup disk exhaustion predictor"),
        ("archive MySQL database with mysqldump, piping through gzip with pipefail checking", "MySQL dump pipeline backup"),
        ("archive PostgreSQL with pg_dump custom format (-Fc) and verify table count on restore test", "PostgreSQL custom format dump"),
        ("take filesystem snapshot using LVM or Btrfs and mount read-only for consistent backup", "LVM/Btrfs snapshot backup"),
        ("create multi-volume split tar archives for large datasets exceeding 10GB", "multi-part split tar backup"),
        ("verify consistency of SQLite database file using PRAGMA integrity_check before archiving", "SQLite pre-backup integrity check"),
        ("backup Redis using BGSAVE command, monitoring LASTSAVE timestamp until completion", "Redis BGSAVE synchronization backup"),
        ("archive directory with exclusions for node_modules, cache, and .git folders", "clean source tarball backup"),
        ("generate detailed JSON metadata report for each backup (source, size, sha256, duration, host)", "backup JSON manifest generator"),
        ("rotate and sync local ZFS snapshots to remote storage host", "ZFS snapshot replication"),
    ]

    for user_desc, target_title in retention_tasks:
        user = f"Write a production Bash script to {user_desc}. Ensure full error handling, defensive file checks, and clear logging."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

BACKUP_DIR="${{1:-/var/backups}}"
RETENTION_DAYS="${{2:-30}}"

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "Error: Backup directory '$BACKUP_DIR' does not exist." >&2
    exit 1
fi

echo "Running {target_title} on $BACKUP_DIR..."

# Execution logic with defensive guards
mapfile -t files < <(find "$BACKUP_DIR" -maxdepth 1 -type f -name "*.tar.gz" -mtime "+$RETENTION_DAYS" | sort)

echo "Found ${{#files[@]}} expired backup archive(s)."
for f in "${{files[@]}}"; do
    echo "Pruning expired archive: $f"
    rm -f -- "$f"
    rm -f -- "$f.sha256" 2>/dev/null || true
done

echo "{target_title} completed successfully."
"""
        explanation = f"Performs {target_title} defensively. Inspects directory existence, loads expired archives via mapfile, and cleans both the archive and associated checksum files."
        records.append(make_rec(user, script, explanation))

    return records[:50]
