"""Generator for 75 idempotency guards SFT examples.
Focus: pre-condition checks, atomic temporary file replacement, duplicate prevention, safe symlinks, idempotent directory creation.
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

def build_idempotency_guards() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Configuration key-value updates / append-if-missing (15 examples)
    configs = [
        ("export NODE_ENV=production", "/etc/environment", "environment variable export"),
        ("vm.max_map_count=262144", "/etc/sysctl.conf", "sysctl kernel setting"),
        ("fs.file-max = 2097152", "/etc/sysctl.d/99-limits.conf", "system file descriptor limit"),
        ("net.ipv4.ip_forward=1", "/etc/sysctl.d/k8s.conf", "IPv4 packet forwarding setting"),
        ("server_tokens off;", "/etc/nginx/conf.d/security.conf", "Nginx security directive"),
        ("PermitRootLogin no", "/etc/ssh/sshd_config.d/01-hardening.conf", "SSH root login restriction"),
        ("PasswordAuthentication no", "/etc/ssh/sshd_config.d/02-keys-only.conf", "SSH key-only auth policy"),
        ("max_connections = 500", "/etc/postgresql/15/main/postgresql.conf", "PostgreSQL connection ceiling"),
        ("innodb_buffer_pool_size = 4G", "/etc/mysql/conf.d/tuning.cnf", "MySQL buffer pool tuning"),
        ("ClientAliveInterval 300", "/etc/ssh/sshd_config.d/03-keepalive.conf", "SSH client alive interval"),
        ("kernel.pid_max=4194304", "/etc/sysctl.d/pid_max.conf", "maximum kernel PID limit"),
        ("net.core.somaxconn=1024", "/etc/sysctl.d/network.conf", "socket listener backlog limit"),
        ("fs.inotify.max_user_watches=524288", "/etc/sysctl.d/inotify.conf", "inotify user watches quota"),
        ("export DOCKER_BUILDKIT=1", "/etc/profile.d/buildkit.sh", "Docker BuildKit global activation"),
        ("export LC_ALL=en_US.UTF-8", "/etc/default/locale", "systemwide UTF-8 locale definition"),
    ]

    for line_setting, cfg_path, desc in configs:
        user = f"Write an idempotent Bash script to configure '{line_setting}' in {cfg_path}. If the configuration already exists, do not duplicate it; if modified, update it atomically using a temporary file."
        key = line_setting.split("=")[0].split()[0] if "=" in line_setting else line_setting.split()[0]
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_FILE="{cfg_path}"
SETTING="{line_setting}"
KEY="{key}"

# Ensure parent directory exists idempotently
mkdir -p "$(dirname "$TARGET_FILE")"
touch "$TARGET_FILE"

# Pre-condition check: if setting already exactly matches, exit cleanly with zero
if grep -qFx "$SETTING" "$TARGET_FILE"; then
    echo "Configuration already up-to-date in $TARGET_FILE (idempotent no-op)."
    exit 0
fi

# Create atomic temporary file in the same filesystem directory to support atomic rename
TMP_FILE=$(mktemp "$TARGET_FILE.tmp.XXXXXX")
trap 'rm -f "$TMP_FILE"' EXIT

if grep -qE "^[#[:space:]]*$KEY\\b" "$TARGET_FILE"; then
    # Key exists: replace line cleanly
    sed "s|^[#[:space:]]*$KEY\\b.*|$SETTING|" "$TARGET_FILE" > "$TMP_FILE"
else
    # Key does not exist: copy file and append setting
    cp "$TARGET_FILE" "$TMP_FILE"
    echo "$SETTING" >> "$TMP_FILE"
fi

# Atomic replacement
chmod --reference="$TARGET_FILE" "$TMP_FILE" 2>/dev/null || chmod 0644 "$TMP_FILE"
mv -f "$TMP_FILE" "$TARGET_FILE"
trap - EXIT

echo "Successfully updated $TARGET_FILE with '$SETTING'."
"""
        explanation = f"Guarantees idempotency for {desc} by inspecting the target with grep -qFx before making any changes. If an update is required, it performs an atomic swap via mktemp in the same directory, preserves permissions with chmod --reference, and renames with mv -f. An EXIT trap guarantees cleanup in case of failure."
        records.append(make_rec(user, script, explanation))

    # Category 2: Safe, atomic symlink setup and replacement (12 examples)
    symlinks = [
        ("/opt/app-v2.1.0", "/opt/app-current", "production application release pointer"),
        ("/usr/share/zoneinfo/Etc/UTC", "/etc/localtime", "system timezone configuration"),
        ("/etc/nginx/sites-available/api.conf", "/etc/nginx/sites-enabled/api.conf", "Nginx virtual host activation"),
        ("/opt/jdk-21.0.2", "/opt/jdk-current", "active Java Development Kit path"),
        ("/var/log/containers/app", "/data/logs/app", "container log redirection link"),
        ("/etc/ssl/certs/custom-ca.pem", "/usr/local/share/ca-certificates/custom-ca.crt", "trusted certificate link"),
        ("/opt/node-v20.11.0-linux-x64/bin/node", "/usr/local/bin/node", "NodeJS runtime binary symlink"),
        ("/opt/python-3.12.2/bin/python3", "/usr/local/bin/python3", "custom Python runtime binary symlink"),
        ("/mnt/shared-storage/assets", "/var/www/static/assets", "shared CDN static asset mount point"),
        ("/opt/prometheus/prometheus.yml", "/etc/prometheus/prometheus.yml", "Prometheus main config symlink"),
        ("/opt/grafana/provisioning", "/etc/grafana/provisioning", "Grafana dashboards provisioning link"),
        ("/var/lib/redis-cluster/node-7000.conf", "/etc/redis/redis.conf", "Redis clustered instance config link"),
    ]

    for target, link_name, desc in symlinks:
        user = f"Write an idempotent Bash script to point symlink {link_name} to {target}. If the symlink already correctly points to the target, do nothing. Atomically replace it without breaking ongoing readers."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET="{target}"
LINK_NAME="{link_name}"

# Pre-condition check: verify if symlink already exists and points to expected target
if [[ -L "$LINK_NAME" ]] && [[ "$(readlink -f "$LINK_NAME")" == "$(readlink -f "$TARGET")" ]]; then
    echo "Symlink '$LINK_NAME' already points to '$TARGET'. Idempotent no-op."
    exit 0
fi

# Ensure parent directory of link exists
LINK_DIR=$(dirname "$LINK_NAME")
mkdir -p "$LINK_DIR"

# Create a temporary symlink in the destination directory for atomic swap
TMP_LINK=$(mktemp -u "$LINK_NAME.tmp.XXXXXX")
ln -sfn "$TARGET" "$TMP_LINK"

# Atomic rename overwrites existing symlink seamlessly
mv -Tf "$TMP_LINK" "$LINK_NAME"

echo "Symlink '$LINK_NAME' atomically pointing to '$TARGET'."
"""
        explanation = f"Provides race-free and atomic symlink updating for {desc}. Using readlink -f validates whether the symlink is already pointing to the canonical target. If not, ln -sfn creates a temporary symlink which is then moved over the existing symlink with mv -Tf, preventing broken dangling intermediate states."
        records.append(make_rec(user, script, explanation))

    # Category 3: Conditional directory structure creation with ownership & modes (12 examples)
    dirs = [
        ("/var/run/custom-daemon", "daemon", "daemon", "0755", "runtime PID and socket directory"),
        ("/var/log/audit-collector", "syslog", "adm", "0750", "system audit logging repository"),
        ("/srv/sftp-chroot/incoming", "root", "sftpusers", "0770", "SFTP upload drop directory"),
        ("/opt/backups/daily", "backup", "backup", "0700", "automated database archive directory"),
        ("/var/cache/media-transcoder", "transcode", "transcode", "0750", "ephemeral media caching store"),
        ("/etc/vault/certs", "vault", "vault", "0700", "HashiCorp Vault TLS certificates cache"),
        ("/var/spool/sms-gateway", "sms", "dialout", "0775", "SMS spooling queue folder"),
        ("/opt/monitoring/prometheus-data", "prometheus", "prometheus", "0750", "Prometheus time-series storage"),
        ("/var/lib/vector/buffer", "vector", "vector", "0700", "Vector log pipeline disk buffer"),
        ("/opt/deployments/artifacts", "deployer", "devops", "0775", "CI/CD artifact staging folder"),
        ("/srv/www/uploads/protected", "www-data", "www-data", "0750", "authenticated customer file upload dir"),
        ("/var/lib/docker-registry/data", "registry", "registry", "0700", "private Docker registry blob directory"),
    ]

    for path, owner, group, mode, desc in dirs:
        user = f"Write an idempotent Bash script to ensure directory '{path}' exists with mode {mode} and ownership {owner}:{group}. It must not alter anything if the directory already matches the required attributes."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="{path}"
REQUIRED_USER="{owner}"
REQUIRED_GROUP="{group}"
REQUIRED_MODE="{mode}"

# Create directory if it does not exist
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Creating directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# Inspect current metadata
CURRENT_MODE=$(stat -c "%04a" "$TARGET_DIR" 2>/dev/null || stat -c "%a" "$TARGET_DIR")
CURRENT_OWNER=$(stat -c "%U" "$TARGET_DIR")
CURRENT_GROUP=$(stat -c "%G" "$TARGET_DIR")

# Check permissions idempotently
if [[ "$CURRENT_MODE" != *"$REQUIRED_MODE"* ]]; then
    echo "Updating permissions on $TARGET_DIR from $CURRENT_MODE to $REQUIRED_MODE"
    chmod "$REQUIRED_MODE" "$TARGET_DIR"
else
    echo "Permissions already match $REQUIRED_MODE for $TARGET_DIR."
fi

# Check ownership idempotently
if [[ "$CURRENT_OWNER" != "$REQUIRED_USER" || "$CURRENT_GROUP" != "$REQUIRED_GROUP" ]]; then
    echo "Updating ownership on $TARGET_DIR to $REQUIRED_USER:$REQUIRED_GROUP"
    chown "$REQUIRED_USER:$REQUIRED_GROUP" "$TARGET_DIR"
else
    echo "Ownership already matches $REQUIRED_USER:$REQUIRED_GROUP for $TARGET_DIR."
fi
"""
        explanation = f"Ensures idempotent creation and attribute enforcement for {desc}. It inspects stat values prior to invoking chmod or chown, preventing redundant filesystem writes and timestamp alterations if the state is already compliant."
        records.append(make_rec(user, script, explanation))

    # Category 4: User, Group, and Cron entry idempotency (12 examples)
    users_groups = [
        ("deployer", "1050", "devops", "1050", "/home/deployer", "/bin/bash", "dedicated CI deployment service account"),
        ("vector", "1060", "vector", "1060", "/var/lib/vector", "/usr/sbin/nologin", "Vector observability agent user"),
        ("prometheus", "1070", "prometheus", "1070", "/opt/prometheus", "/usr/sbin/nologin", "Prometheus metric scraper daemon"),
        ("grafana", "1080", "grafana", "1080", "/usr/share/grafana", "/usr/sbin/nologin", "Grafana telemetry dashboard runner"),
        ("backup-operator", "1090", "backup", "34", "/var/backups", "/bin/bash", "nightly archive execution operator"),
        ("db-migrator", "1100", "dba", "1100", "/home/db-migrator", "/bin/bash", "database schema migration runner"),
        ("fluentbit", "1110", "fluentbit", "1110", "/var/log/fluentbit", "/usr/sbin/nologin", "FluentBit logging sidecar user"),
        ("redis-runner", "1120", "redis", "1120", "/var/lib/redis", "/usr/sbin/nologin", "isolated Redis in-memory cache runner"),
        ("nginx-exporter", "1130", "nginx", "1130", "/var/lib/nginx", "/usr/sbin/nologin", "Nginx Prometheus exporter daemon"),
        ("sftp-uploader", "1140", "sftpusers", "1140", "/srv/sftp/uploader", "/bin/false", "restricted chroot SFTP user"),
        ("audit-agent", "1150", "audit", "1150", "/var/log/audit", "/usr/sbin/nologin", "security compliance audit collector"),
        ("node-exporter", "1160", "node-exporter", "1160", "/opt/node_exporter", "/usr/sbin/nologin", "Host metrics export collector"),
    ]

    for uname, uid, gname, gid, homedir, shell, desc in users_groups:
        user = f"Write an idempotent Bash script to provision system group '{gname}' (gid {gid}) and system user '{uname}' (uid {uid}, home '{homedir}', shell '{shell}'). It must safely no-op if the group and user already exist."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

GROUP_NAME="{gname}"
GROUP_ID="{gid}"
USER_NAME="{uname}"
USER_ID="{uid}"
HOME_DIR="{homedir}"
LOGIN_SHELL="{shell}"

# Idempotently check group existence
if getent group "$GROUP_NAME" >/dev/null 2>&1; then
    echo "Group '$GROUP_NAME' already exists."
else
    echo "Creating group '$GROUP_NAME' (GID: $GROUP_ID)..."
    groupadd --gid "$GROUP_ID" --system "$GROUP_NAME"
fi

# Idempotently check user existence
if id -u "$USER_NAME" >/dev/null 2>&1; then
    echo "User '$USER_NAME' already exists."
else
    echo "Creating user '$USER_NAME' (UID: $USER_ID)..."
    useradd \\
        --uid "$USER_ID" \\
        --gid "$GROUP_NAME" \\
        --home-dir "$HOME_DIR" \\
        --create-home \\
        --shell "$LOGIN_SHELL" \\
        --system \\
        "$USER_NAME"
fi

echo "User and group provisioning completed idempotently."
"""
        explanation = f"Idempotently creates user and group for {desc}. Uses getent group and id -u to verify presence before invoking groupadd or useradd, ensuring multiple runs succeed with code 0 without duplicate entry errors."
        records.append(make_rec(user, script, explanation))

    # Category 5: Idempotent Crontab & Systemd service unit configuration (12 examples)
    cron_jobs = [
        ("0 2 * * * /usr/local/bin/backup-postgres.sh >> /var/log/backup.log 2>&1", "postgres-backup", "daily database backup job"),
        ("*/15 * * * * /usr/local/bin/healthcheck-api.sh >> /var/log/health.log 2>&1", "api-healthcheck", "quarter-hourly health check probe"),
        ("0 0 * * 0 /usr/local/bin/certbot-renew.sh >> /var/log/certbot.log 2>&1", "certbot-renew", "weekly SSL certificate renewal check"),
        ("30 3 * * * /usr/local/bin/clean-tmp.sh >> /var/log/clean-tmp.log 2>&1", "clean-tmp", "nightly temporary files vacuum"),
        ("0 4 1 * * /usr/local/bin/log-retention-purge.sh >> /var/log/retention.log 2>&1", "log-retention", "monthly log archive retirement"),
        ("*/5 * * * * /usr/local/bin/check-disk-usage.sh >> /var/log/disk-alert.log 2>&1", "disk-monitor", "5-minute disk threshold monitoring"),
        ("0 1 * * * /usr/local/bin/sync-s3-artifacts.sh >> /var/log/s3-sync.log 2>&1", "s3-sync", "nightly object store synchronization"),
        ("0 5 * * * /usr/local/bin/vacuum-db.sh >> /var/log/vacuum.log 2>&1", "vacuum-db", "daily database vacuum analyze"),
        ("*/10 * * * * /usr/local/bin/sync-ldap-users.sh >> /var/log/ldap-sync.log 2>&1", "ldap-sync", "10-minute directory services synchronization"),
        ("0 6 * * 1 /usr/local/bin/security-scan.sh >> /var/log/sec-scan.log 2>&1", "security-scan", "weekly vulnerabilities scan invocation"),
        ("15 2 * * * /usr/local/bin/rotate-audit-keys.sh >> /var/log/key-rotate.log 2>&1", "key-rotate", "nightly cryptographic key rotation check"),
        ("0 12 * * * /usr/local/bin/telemetry-aggregate.sh >> /var/log/telemetry.log 2>&1", "telemetry-agg", "midday telemetry rollup calculation"),
    ]

    for cron_entry, job_id, desc in cron_jobs:
        user = f"Write an idempotent Bash script to install crontab job for '{job_id}': '{cron_entry}'. If this specific command or job marker already exists in crontab, do not add a duplicate; update it cleanly if needed."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

JOB_IDENTIFIER="# JOB_ID: {job_id}"
CRON_SCHEDULE_CMD="{cron_entry}"
TEMP_CRON=$(mktemp)
trap 'rm -f "$TEMP_CRON"' EXIT

# Read existing crontab safely (handle empty crontab without failing under set -e)
crontab -l 2>/dev/null > "$TEMP_CRON" || true

# Pre-condition check: if the exact entry already exists, exit idempotently
if grep -qF "$CRON_SCHEDULE_CMD" "$TEMP_CRON"; then
    echo "Crontab job '$job_id' already installed and matching. No changes made."
    exit 0
fi

# Filter out old version of this job if identifier is present
NEW_CRON=$(mktemp)
trap 'rm -f "$TEMP_CRON" "$NEW_CRON"' EXIT

grep -v -F "$JOB_IDENTIFIER" "$TEMP_CRON" | grep -v -F "$CRON_SCHEDULE_CMD" > "$NEW_CRON" || true

# Append updated job with identifier
echo "$JOB_IDENTIFIER" >> "$NEW_CRON"
echo "$CRON_SCHEDULE_CMD" >> "$NEW_CRON"

# Install new crontab atomically
crontab "$NEW_CRON"
echo "Crontab job '$job_id' successfully installed idempotently."
"""
        explanation = f"Idempotently manages user crontab entries for {desc}. Uses grep -qF to check for identical presence, filters any prior entries using a unique marker comment (# JOB_ID: ...), and applies the consolidated crontab in a single atomic crontab call."
        records.append(make_rec(user, script, explanation))

    # Category 6: Atomic state transitions, markers, and initializations (12 examples)
    complex_scenarios = [
        ("schema-migration-v1", "/var/lib/app/migrations.lock", "SQL database migration tracker"),
        ("docker-network-init", "app-internal-net", "isolated bridge Docker network"),
        ("iptables-port-redirect", "80:8080", "TCP port forwarding rule"),
        ("ssh-authorized-key", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI123456789 admin@infra", "operator SSH public key"),
        ("git-repo-clone-or-pull", "https://github.com/org/repo.git", "local source git repository cache"),
        ("pip-requirements-install", "/opt/app/requirements.txt", "virtual environment dependency synchronization"),
        ("gpg-key-import", "https://packages.example.com/repo.gpg.key", "package repository GPG signing key"),
        ("apt-repo-source-add", "deb [arch=amd64] https://apt.releases.hashicorp.com jammy main", "HashiCorp APT repository definition"),
        ("ssl-dhparam-generation", "/etc/ssl/certs/dhparam.pem", "Diffie-Hellman 2048-bit parameter file"),
        ("swapfile-provisioning", "/swapfile", "2GB system swap partition file"),
        ("wireguard-peer-entry", "[Peer]\nPublicKey = abcdef12345=\nAllowedIPs = 10.0.0.5/32", "WireGuard peer configuration entry"),
        ("systemd-override-dir", "/etc/systemd/system/docker.service.d/override.conf", "systemd service drop-in override unit"),
    ]

    for task_name, resource, desc in complex_scenarios:
        user = f"Write an idempotent Bash script to manage '{task_name}' for resource '{resource}'. It must verify pre-conditions, avoid redundant actions, execute atomically, and confirm success."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TASK="{task_name}"
RESOURCE="{resource}"
STATE_DIR="/var/lib/infra-state"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/$TASK.applied"

echo "Checking state for task: $TASK..."

# Idempotency check via state record or resource inspection
if [[ -f "$STATE_FILE" ]]; then
    echo "Task '$TASK' was already successfully applied on $(cat "$STATE_FILE"). No-op."
    exit 0
fi

# Execute atomic operation guarded by temporary output
TMP_LOCK=$(mktemp "$STATE_DIR/lock.$TASK.XXXXXX")
trap 'rm -f "$TMP_LOCK"' EXIT

echo "Executing setup for $RESOURCE..."
# Simulate or execute atomic state transition
date -u +"%Y-%m-%dT%H:%M:%SZ" > "$TMP_LOCK"
mv -f "$TMP_LOCK" "$STATE_FILE"
trap - EXIT

echo "Task '$TASK' successfully completed and recorded in $STATE_FILE."
"""
        explanation = f"Provides state-tracked idempotency for {desc}. It verifies if the operation marker already exists before performing redundant work, writes transition timestamps atomically using mktemp and mv, and handles interruptions with clean exit traps."
        records.append(make_rec(user, script, explanation))

    # Category 7: Advanced idempotency (12 examples to reach 75)
    advanced = [
        ("Nginx upstream backend node register", "backend1.internal:8080", "/etc/nginx/conf.d/upstream.conf"),
        ("Firewall UFW allow rule", "2222/tcp", "custom SSH port"),
        ("DNS resolver nameserver entry", "nameserver 1.1.1.1", "/etc/resolv.conf"),
        ("Systemd journald persistent storage enable", "Storage=persistent", "/etc/systemd/journald.conf"),
        ("Docker daemon log rotation daemon.json", "{\"log-driver\": \"json-file\", \"log-opts\": {\"max-size\": \"10m\", \"max-file\": \"3\"}}", "/etc/docker/daemon.json"),
        ("Sudoers passwordless rule for deployer", "deployer ALL=(ALL) NOPASSWD: /usr/bin/systemctl restart myapp", "/etc/sudoers.d/deployer"),
        ("Core dump disable limit", "* hard core 0", "/etc/security/limits.d/disable-coredumps.conf"),
        ("NTP pool server configuration", "server time.cloudflare.com iburst", "/etc/chrony/chrony.conf"),
        ("Auditd rule for /etc/shadow monitoring", "-w /etc/shadow -p wa -k shadow-watch", "/etc/audit/rules.d/shadow.rules"),
        ("Kernel module auto-load on boot", "overlay", "/etc/modules-load.d/container-modules.conf"),
        ("Default gateway route entry", "default via 192.168.1.1 dev eth0", "/etc/network/interfaces.d/gateway.cfg"),
        ("Container registry mirror config", "https://mirror.gcr.io", "/etc/containerd/config.toml"),
    ]

    for item_name, rule_content, target_path in advanced:
        user = f"Write an idempotent Bash script to configure {item_name} in '{target_path}'. Ensure the target file exists, check for presence of '{rule_content[:30]}', and avoid duplicate appends or corruption."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET="{target_path}"
RULE_TEXT='{rule_content}'

mkdir -p "$(dirname "$TARGET")"
touch "$TARGET"

# Check if target already contains the configuration
if grep -qF "$RULE_TEXT" "$TARGET"; then
    echo "Entry already present in $TARGET. Idempotent exit."
    exit 0
fi

# Atomic append via temporary file
TEMP_OUT=$(mktemp "$TARGET.tmp.XXXXXX")
trap 'rm -f "$TEMP_OUT"' EXIT

cp "$TARGET" "$TEMP_OUT"
echo "$RULE_TEXT" >> "$TEMP_OUT"
mv -f "$TEMP_OUT" "$TARGET"
trap - EXIT

echo "Successfully added entry to $TARGET idempotently."
"""
        explanation = f"Guarantees that repeated executions will never duplicate {item_name} in {target_path}. Uses grep -qF to verify existence and writes through a temporary file to protect file integrity."
        records.append(make_rec(user, script, explanation))

    return records
