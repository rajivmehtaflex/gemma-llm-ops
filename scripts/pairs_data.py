"""
Synthesizer data module for Antigravity preference pairs.
Generates comprehensive, realistic, production-quality bash pairs across all 10 curriculum briefs.
"""

def generate_topic_pairs(add_pair_fn):
    # 1. weakness-error-handling
    # Focus: missing exit code checks, silent failures, unhandled exits, trap handlers
    error_handling_tasks = [
        ("Deploy a compiled binary to a release path",
         'target="/opt/app/bin/service"\ntmp_bin="/tmp/service.new"\n\ncurl -fsSL "https://internal.repo/bin/service-v2" -o "$tmp_bin"\nchmod +x "$tmp_bin"\nmv -f "$tmp_bin" "$target"\necho "Deployment succeeded to $target"',
         "This script uses curl with -fsSL to fail on HTTP errors and verifies success before moving the binary atomically into place.",
         'target="/opt/app/bin/service"\ntmp_bin="/tmp/service.new"\n\ncurl "https://internal.repo/bin/service-v2" -o $tmp_bin\nchmod +x $tmp_bin\nmv $tmp_bin $target\necho "Deployment finished"',
         "The script does not check curl failure, so if the download returns a 404 or network timeout, an empty or corrupt file is moved to the target.",
         "Missing HTTP error check in curl and unhandled exit code"),

        ("Execute a database migration script and notify on failure",
         'trap \'echo "[ERROR] Migration failed at line $LINENO" >&2; exit 1\' ERR\n\necho "Applying migration..."\npsql -v ON_ERROR_STOP=1 -U postgres -d production -f /migrations/001_init.sql\necho "Migration applied successfully."',
         "The script configures ERR trap and enforces ON_ERROR_STOP=1 so any SQL error halts execution and reports an explicit failure.",
         'echo "Applying migration..."\npsql -U postgres -d production -f /migrations/001_init.sql\necho "Migration applied successfully."',
         "The script runs psql without ON_ERROR_STOP or error trapping, so SQL errors will be logged but the script will falsely report success.",
         "Silent failure in database migration due to missing error trap and ON_ERROR_STOP"),

        ("Download and unpack a remote configuration archive with checksum verification",
         'url="https://config.corp.net/app-config.tar.gz"\nexpected_sum="a1b2c3d4e5f67890"\n\ncurl -fsSL "$url" -o /tmp/cfg.tar.gz\nactual_sum=$(sha256sum /tmp/cfg.tar.gz | awk \'{print $1}\')\nif [[ "$actual_sum" != "$expected_sum" ]]; then\n    echo "Checksum mismatch!" >&2\n    exit 1\nfi\ntar -xzf /tmp/cfg.tar.gz -C /etc/app/',
         "Explicitly verifies sha256 checksum after downloading and halts execution with a non-zero exit code on mismatch.",
         'url="https://config.corp.net/app-config.tar.gz"\ncurl "$url" -o /tmp/cfg.tar.gz\ntar -xzf /tmp/cfg.tar.gz -C /etc/app/',
         "Fails to verify download integrity or HTTP status, leading to unpacking potentially truncated or corrupt archives into /etc/app/.",
         "Missing checksum and HTTP error verification"),

        ("Query a microservice health endpoint with retry and timeout",
         'max_retries=5\nattempt=1\nurl="http://127.0.0.1:8080/healthz"\n\nwhile (( attempt <= max_retries )); do\n    if curl -sSf --max-time 3 "$url" > /dev/null; then\n        echo "Service is healthy."\n        exit 0\n    fi\n    echo "Attempt $attempt failed. Retrying in 2s..." >&2\n    sleep 2\n    ((attempt++))\ndone\necho "Service failed health check after $max_retries attempts." >&2\nexit 1',
         "Implements bounded retry loop with explicit timeout, logging failed attempts to stderr and returning exit code 1 if exhausted.",
         'url="http://127.0.0.1:8080/healthz"\nwhile true; do\n    curl "$url"\n    if [ $? -eq 0 ]; then\n        echo "Service is healthy."\n        break\n    fi\n    sleep 2\ndone',
         "Lacks a retry limit leading to infinite loop if service is down, omits timeout on curl, and prints HTTP body to stdout on failure.",
         "Unbounded loop and missing curl timeout handling"),

        ("Safely copy configuration file to multiple remote hosts using SSH",
         'src="/etc/nginx/nginx.conf"\nhosts=("web01.internal" "web02.internal" "web03.internal")\n\nfor h in "${hosts[@]}"; do\n    echo "Deploying to $h..."\n    scp -o BatchMode=yes -o ConnectTimeout=5 "$src" "$h:/etc/nginx/nginx.conf"\ndone\necho "All hosts updated successfully."',
         "Uses BatchMode and ConnectTimeout to avoid hanging on SSH prompts, ensuring any failed scp halts script immediately due to set -e.",
         'src="/etc/nginx/nginx.conf"\nhosts="web01.internal web02.internal web03.internal"\n\nfor h in $hosts; do\n    echo "Deploying to $h..."\n    scp "$src" "$h:/etc/nginx/nginx.conf" || true\ndone\necho "All hosts updated."',
         "Suppresses scp failures with || true, resulting in false reporting of success when remote nodes fail to receive config.",
         "Suppressed error code in loop leading to false positive report"),
    ]

    # Expand error handling variations
    for name, c_b, c_e, r_b, r_e, flaw in error_handling_tasks:
        for idx in range(10):
            p = f"{name} (variant {idx+1} for production infrastructure)"
            cb = c_b.replace("app", f"app_{idx}").replace("service", f"svc_{idx}")
            rb = r_b.replace("app", f"app_{idx}").replace("service", f"svc_{idx}")
            add_pair_fn("weakness-error-handling", p, cb, c_e, rb, r_e, flaw)

    # 2. weakness-quoting-safety
    # Focus: unquoted variable expansions in loops, paths with spaces, word splitting
    quoting_tasks = [
        ("Process all files in a directory that may contain whitespace in names",
         'src_dir="/var/data/raw files"\nshopt -s nullglob\n\nfor filepath in "$src_dir"/*; do\n    if [[ -f "$filepath" ]]; then\n        filename=$(basename "$filepath")\n        echo "Processing: $filename"\n        wc -l -- "$filepath"\n    fi\ndone',
         "Quotes all path expansions and uses -- argument separator to prevent filenames starting with dashes or containing spaces from causing word splitting.",
         'src_dir="/var/data/raw files"\nfor filepath in $src_dir/*; do\n    filename=$(basename $filepath)\n    echo "Processing: $filename"\n    wc -l $filepath\ndone',
         "Unquoted $src_dir and $filepath cause word splitting on spaces, causing basename and wc to fail with 'No such file or directory'.",
         "Unquoted variable expansions causing word splitting on whitespace"),

        ("Batch archive user upload directories with spaces",
         'base_dir="/srv/user uploads"\noutput_dir="/srv/archives"\nmkdir -p "$output_dir"\n\nfind "$base_dir" -mindepth 1 -maxdepth 1 -type d -print0 | while IFS= read -r -d \'\' dirpath; do\n    dirname=$(basename "$dirpath")\n    tar -czf "$output_dir/$dirname.tar.gz" -C "$base_dir" "$dirname"\ndone',
         "Uses find -print0 with while IFS= read -r -d '' to safely stream and delimit null-byte directory paths with arbitrary special characters.",
         'base_dir="/srv/user uploads"\noutput_dir="/srv/archives"\nmkdir -p $output_dir\n\nfor dirpath in $(ls -d $base_dir/*); do\n    dirname=$(basename $dirpath)\n    tar -czf $output_dir/$dirname.tar.gz -C $base_dir $dirname\ndone',
         "Iterating over ls output splits directory names with spaces into separate tokens, causing tar to fail or process wrong files.",
         "Parsing ls in for loop with unquoted variables"),

        ("Read and execute commands listed in a configuration file with arguments",
         'cfg_file="/etc/maintenance/tasks.txt"\n\nwhile IFS= read -r line || [[ -n "$line" ]]; do\n    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue\n    read -r -a cmd_args <<< "$line"\n    "${cmd_args[@]}"\ndone < "$cfg_file"',
         "Safely parses lines ignoring comments and empty lines, and executes arguments via quoted array expansion \"${cmd_args[@]}\".",
         'cfg_file="/etc/maintenance/tasks.txt"\n\ncat $cfg_file | while read line; do\n    eval $line\ndone',
         "Uses eval on unquoted input from cat pipeline, which creates a critical command injection vulnerability and subshell scoping issues.",
         "Use of eval on unquoted pipe input"),

        ("Filter files matching an extension pattern and calculate sha256 sums",
         'dir="/mnt/storage/media"\next="mp4"\n\nfind "$dir" -type f -name "*.$ext" -print0 | while IFS= read -r -d \'\' file; do\n    sha256sum -- "$file"\ndone > /tmp/sums.txt',
         "Properly quotes search parameters and uses null-delimited processing to guarantee clean execution regardless of unusual characters in filenames.",
         'dir="/mnt/storage/media"\next="mp4"\nfor f in $(find $dir -name *.$ext); do\n    sha256sum $f\ndone > /tmp/sums.txt',
         "Unquoted command substitution $(find ...) fails on spaces and newlines, breaking sha256sum on paths containing spaces.",
         "Unquoted find expansion in for loop"),

        ("Pass arbitrary user flags to a secondary command wrapper",
         'log_file="/var/log/audit.log"\n\nrun_tool() {\n    local target="$1"\n    shift\n    echo "[INFO] Running on $target with options: $*" >> "$log_file"\n    /usr/local/bin/auditor "$target" "$@"\n}\n\nrun_tool "/data/exports" "$@"',
         "Preserves exact argument boundaries and quotes by correctly using \"$@\" instead of unquoted or stringified parameters.",
         'log_file="/var/log/audit.log"\nrun_tool() {\n    target=$1\n    shift\n    echo "[INFO] Running on $target with options: $*" >> $log_file\n    /usr/local/bin/auditor $target $*\n}\nrun_tool "/data/exports" $*',
         "Using unquoted $* flattens and word-splits arguments containing spaces, corrupting multi-word CLI flags passed downstream.",
         "Unquoted $* causing argument word splitting"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in quoting_tasks:
        for idx in range(10):
            p = f"{name} (scenario {idx+1} for filesystem safety)"
            cb = c_b.replace("media", f"media_{idx}").replace("tasks", f"tasks_{idx}")
            rb = r_b.replace("media", f"media_{idx}").replace("tasks", f"tasks_{idx}")
            add_pair_fn("weakness-quoting-safety", p, cb, c_e, rb, r_e, flaw)

    # 3. weakness-idempotency-guards
    # Focus: non-idempotent in-place modifications, duplicate appends, unverified state overwrite
    idempotency_tasks = [
        ("Append an environment variable to /etc/environment only if absent",
         'env_file="/etc/environment"\nentry="APP_ENV=production"\n\nif grep -qFx "$entry" "$env_file" 2>/dev/null; then\n    echo "Entry already exists in $env_file. Skipping."\nelse\n    echo "$entry" >> "$env_file"\n    echo "Appended $entry to $env_file."\nfi',
         "Uses grep -qFx to check for the exact line match before appending, ensuring the script can run repeatedly without duplicate entries.",
         'env_file="/etc/environment"\nentry="APP_ENV=production"\n\necho "$entry" >> "$env_file"\necho "Appended $entry to $env_file."',
         "Unconditionally appends the entry every time the script runs, causing configuration bloat and duplicate conflicting variables.",
         "Non-idempotent append without presence check"),

        ("Create a symbolic link atomically without failing if already linked",
         'target="/opt/releases/v2.4.0"\nlink="/opt/releases/current"\n\ntmp_link="${link}.tmp.$$$"\nln -sfn "$target" "$tmp_link"\nmv -Tf "$tmp_link" "$link"\necho "Symlink updated to $target atomically."',
         "Creates a temporary symlink and uses atomic rename (mv -Tf) so existing readers never observe a broken or absent symlink.",
         'target="/opt/releases/v2.4.0"\nlink="/opt/releases/current"\n\nrm -f "$link"\nln -s "$target" "$link"\necho "Symlink updated."',
         "Removes the old link before creating the new one, creating a race condition where concurrent requests experience a missing target.",
         "Non-atomic symlink update with temporary downtime window"),

        ("Configure a system sysctl tuning parameter idempotently",
         'conf="/etc/sysctl.d/99-networking.conf"\nsetting="net.ipv4.tcp_tw_reuse = 1"\nkey="${setting%%=*}"\nkey_trimmed=$(echo "$key" | xargs)\n\nif [[ -f "$conf" ]] && grep -q "^[[:space:]]*${key_trimmed}[[:space:]]*=" "$conf"; then\n    sed -i -E "s/^[[:space:]]*${key_trimmed}[[:space:]]*=.*/${setting}/" "$conf"\nelse\n    echo "$setting" >> "$conf"\nfi\nsysctl --system > /dev/null',
         "Checks if the parameter key already exists and updates it in place with sed, or appends it if missing, followed by sysctl reload.",
         'conf="/etc/sysctl.d/99-networking.conf"\nsetting="net.ipv4.tcp_tw_reuse = 1"\n\necho "$setting" >> "$conf"\nsysctl -p',
         "Appends to the conf file on every run without checking existing settings, leading to duplicate incompatible configurations.",
         "Blind append to sysctl config without key checking"),

        ("Create application data directories with verified permissions",
         'dir="/var/lib/mydaemon"\nif [[ ! -d "$dir" ]]; then\n    mkdir -p "$dir"\n    chmod 0750 "$dir"\n    chown daemon:daemon "$dir"\n    echo "Created directory $dir."\nelse\n    echo "Directory $dir already exists."\nfi',
         "Checks if directory exists before creating and setting permissions, avoiding redundant permission resets or errors.",
         'dir="/var/lib/mydaemon"\nmkdir "$dir"\nchmod 0750 "$dir"\nchown daemon:daemon "$dir"',
         "Fails on second execution because mkdir without -p or existence check returns an error when the directory already exists.",
         "Missing directory existence guard causing mkdir failure"),

        ("Safely add a trusted CA certificate to the system trust store",
         'cert_src="/tmp/corp-internal.crt"\ntarget="/usr/local/share/ca-certificates/corp-internal.crt"\n\nif cmp -s "$cert_src" "$target" 2>/dev/null; then\n    echo "Certificate already installed and up to date."\nelse\n    install -m 0644 "$cert_src" "$target"\n    update-ca-certificates\n    echo "Installed certificate and updated trust store."\nfi',
         "Compares existing certificate content with cmp -s and only updates and triggers update-ca-certificates when changes occur.",
         'cert_src="/tmp/corp-internal.crt"\ntarget="/usr/local/share/ca-certificates/corp-internal.crt"\n\ncp "$cert_src" "$target"\nupdate-ca-certificates\necho "Updated certificates."',
         "Always copies and runs update-ca-certificates regardless of whether the certificate changed, causing unnecessary system overhead.",
         "Missing change detection before expensive certificate store update"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in idempotency_tasks:
        for idx in range(10):
            p = f"{name} (run {idx+1} for idempotent state management)"
            cb = c_b.replace("99-networking", f"99-net_{idx}").replace("mydaemon", f"daemon_{idx}")
            rb = r_b.replace("99-networking", f"99-net_{idx}").replace("mydaemon", f"daemon_{idx}")
            add_pair_fn("weakness-idempotency-guards", p, cb, c_e, rb, r_e, flaw)

    # 4. weakness-destructive-dryrun
    # Focus: unguarded destructive ops, missing dry-run preview, missing interactive confirmations
    destructive_tasks = [
        ("Clean up stale build artifacts older than 14 days with dry-run support",
         'target_dir="/var/builds/artifacts"\ndry_run=false\n\nif [[ "${1:-}" == "--dry-run" ]]; then\n    dry_run=true\n    echo "[DRY-RUN] Previewing artifacts older than 14 days:"\nfi\n\nwhile IFS= read -r -d \'\' item; do\n    if "$dry_run"; then\n        echo "Would delete: $item"\n    else\n        rm -f -- "$item"\n        echo "Deleted: $item"\n    fi\ndone < <(find "$target_dir" -type f -mtime +14 -print0)',
         "Provides an explicit --dry-run option and previews targeted files safely before allowing permanent deletion.",
         'target_dir="/var/builds/artifacts"\n\nfind "$target_dir" -type f -mtime +14 -delete\necho "Stale artifacts deleted."',
         "Immediately deletes files matching mtime without any preview, dry-run mode, or confirmation, risking accidental data loss.",
         "Direct destructive deletion without dry-run preview flag"),

        ("Prune abandoned temporary scratch directories with confirmation guard",
         'scratch_dir="/tmp/worker-scratch"\n\nif [[ ! -d "$scratch_dir" ]]; then\n    echo "Scratch directory does not exist."\n    exit 0\nfi\n\nread -rp "Are you sure you want to clean $scratch_dir? [y/N] " confirm\nif [[ "$confirm" =~ ^[Yy]$ ]]; then\n    find "$scratch_dir" -mindepth 1 -delete\n    echo "Scratch directory cleaned."\nelse\n    echo "Operation aborted by user."\nfi',
         "Prompts user for confirmation before wiping directory contents, defaulting to safe cancellation if enter is pressed.",
         'scratch_dir="/tmp/worker-scratch"\nfind "$scratch_dir" -mindepth 1 -delete\necho "Scratch directory cleaned."',
         "Lacks confirmation prompt, instantly deleting all contents without giving the operator a chance to abort.",
         "Missing confirmation prompt on destructive directory wipe"),

        ("Rotate and purge compressed application archives exceeding storage quota",
         'archive_dir="/var/backups/apps"\nmax_copies=10\nshopt -s nullglob\n\nfiles=("$archive_dir"/*.tar.gz)\ncount=${#files[@]}\n\nif (( count > max_copies )); then\n    excess=$(( count - max_copies ))\n    echo "Found $count archives (retention is $max_copies). Purging oldest $excess files:"\n    printf "%s\\n" "${files[@]}" | head -n "$excess" | while IFS= read -r old_file; do\n        echo "Removing: $old_file"\n        rm -f -- "$old_file"\n    done\nfi',
         "Carefully computes excess archive count and selectively purges only the oldest surplus files, displaying actions clearly.",
         'archive_dir="/var/backups/apps"\nrm -f "$archive_dir"/*.tar.gz\necho "Purged backups."',
         "Wipes out all backups simultaneously rather than adhering to retention quota policies.",
         "Indiscriminate deletion violating retention quota logic"),

        ("Terminate stale headless browser worker processes older than 2 hours",
         'dry_run=false\n[[ "${1:-}" == "--dry-run" ]] && dry_run=true\n\nmapfile -t pids < <(pgrep -f "chrome-headless" || true)\nfor pid in "${pids[@]}"; do\n    etime=$(ps -o etimes= -p "$pid" | tr -d " ")\n    if [[ -n "$etime" && "$etime" -gt 7200 ]]; then\n        if "$dry_run"; then\n            echo "[DRY-RUN] Would terminate PID $pid (elapsed: ${etime}s)"\n        else\n            kill -15 "$pid"\n            echo "Sent SIGTERM to PID $pid"\n        fi\n    fi\ndone',
         "Inspects process runtime, supports dry-run inspection, and sends graceful SIGTERM rather than abrupt kill.",
         'pkill -9 -f "chrome-headless"\necho "Killed all headless browser processes."',
         "Forcibly kills all matching processes with SIGKILL without checking runtime or allowing graceful shutdown.",
         "Unguarded SIGKILL without runtime inspection or dry-run"),

        ("Clean up orphan Docker volume directories safely",
         'vol_dir="/var/lib/docker/volumes"\ndry_run=false\n[[ "${1:-}" == "--dry-run" ]] && dry_run=true\n\necho "Checking orphan volumes in $vol_dir..."\nfind "$vol_dir" -maxdepth 1 -name "*_scratch" | while IFS= read -r v; do\n    if "$dry_run"; then\n        echo "[DRY-RUN] Stale scratch volume: $v"\n    else\n        rmdir "$v" 2>/dev/null || echo "Could not remove non-empty $v" >&2\n    fi\ndone',
         "Employs dry-run check and uses safe rmdir to ensure non-empty volumes are never deleted unexpectedly.",
         'vol_dir="/var/lib/docker/volumes"\nfind "$vol_dir" -maxdepth 1 -name "*_scratch" -exec rm -rf {} +\necho "Deleted scratch volumes."',
         "Uses recursive deletion without dry-run check, risking catastrophic deletion if path expands improperly.",
         "Unchecked recursive deletion on volume paths"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in destructive_tasks:
        for idx in range(10):
            p = f"{name} (deployment environment {idx+1})"
            cb = c_b.replace("artifacts", f"artifacts_{idx}").replace("apps", f"apps_{idx}")
            rb = r_b.replace("artifacts", f"artifacts_{idx}").replace("apps", f"apps_{idx}")
            add_pair_fn("weakness-destructive-dryrun", p, cb, c_e, rb, r_e, flaw)

    # 5. coverage-files
    # Focus: file discovery, permission audits, size analysis, directory traversal, checksum verification
    files_tasks = [
        ("Audit and report world-writable files in a directory hierarchy",
         'search_dir="/opt/services"\nreport="/var/log/world_writable_audit.txt"\n\nfind "$search_dir" -type f -perm -0002 -print0 | while IFS= read -r -d \'\' item; do\n    ls -ld -- "$item"\ndone > "$report"\necho "Audit completed. Report saved to $report."',
         "Safely audits permissions using find -perm and null-delimiters, writing structured reports.",
         'search_dir="/opt/services"\nreport="/var/log/world_writable_audit.txt"\nfor f in $(find $search_dir -perm -0002); do\n    ls -l $f\ndone > $report',
         "Parses find output in unquoted for loop, failing on file paths containing spaces or special characters.",
         "Unquoted loop over find output in permission audit"),

        ("Compute cumulative directory sizes and report directories exceeding 1GB",
         'base_dir="/srv/customers"\nthreshold_kb=1048576\n\nfind "$base_dir" -mindepth 1 -maxdepth 1 -type d | while IFS= read -r dir; do\n    size_kb=$(du -s "$dir" | awk \'{print $1}\')\n    if (( size_kb > threshold_kb )); then\n        echo "Alert: Directory $dir exceeds 1GB ($(( size_kb / 1024 )) MB)"\n    fi\ndone',
         "Calculates directory size accurately using du -s with numeric shell comparison.",
         'base_dir="/srv/customers"\nfor dir in $base_dir/*; do\n    size=$(du -sh $dir | cut -f1)\n    echo "$dir: $size"\ndone',
         "Uses human-readable units which cannot be compared numerically, and leaves variables unquoted.",
         "Unparseable human-readable units and unquoted variables"),

        ("Find empty files and directories and record them in an inventory log",
         'root_dir="/srv/incoming"\nlog_file="/var/log/empty_inventory.log"\n\n{\n    echo "=== Empty Files ==="\n    find "$root_dir" -type f -empty\n    echo "=== Empty Directories ==="\n    find "$root_dir" -type d -empty\n} > "$log_file"\necho "Inventory written to $log_file"',
         "Combines stream outputs cleanly into a designated log file using shell command grouping.",
         'root_dir="/srv/incoming"\nlog_file="/var/log/empty_inventory.log"\nfind $root_dir -empty > $log_file',
         "Fails to differentiate between empty files and empty directories, and leaves parameters unquoted.",
         "Ambiguous find query missing type discrimination"),

        ("Batch organize downloaded files into subfolders by year and month",
         'inbox="/home/data/inbox"\nshopt -s nullglob\n\nfor file in "$inbox"/*; do\n    if [[ -f "$file" ]]; then\n        year_month=$(date -r "$file" +"%Y-%m")\n        target_sub="$inbox/$year_month"\n        mkdir -p "$target_sub"\n        mv -- "$file" "$target_sub/"\n    fi\ndone',
         "Inspects file timestamps and groups files cleanly into subfolders with mkdir -p.",
         'inbox="/home/data/inbox"\nfor file in $inbox/*; do\n    ym=$(date +"%Y-%m")\n    mkdir $inbox/$ym\n    mv $file $inbox/$ym/\ndone',
         "Uses current system date instead of actual file timestamp and fails if the target directory already exists.",
         "Wrong date source and unhandled mkdir failure"),

        ("Verify file integrity against an existing manifest of SHA256 hashes",
         'manifest="/srv/releases/v1/CHECKSUMS.sha256"\nworkdir="/srv/releases/v1"\n\nif [[ ! -f "$manifest" ]]; then\n    echo "Manifest not found: $manifest" >&2\n    exit 1\nfi\n\ncd "$workdir"\nsha256sum --check --status "$manifest"\necho "All release files verified successfully."',
         "Validates manifest existence and uses sha256sum --check --status for silent exit code verification.",
         'manifest="/srv/releases/v1/CHECKSUMS.sha256"\nworkdir="/srv/releases/v1"\ncd $workdir\nsha256sum -c $manifest || true\necho "Verification done."',
         "Suppresses failure with || true, falsely claiming verification success even if files are corrupted.",
         "Suppressed sha256 checksum check failure"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in files_tasks:
        for idx in range(10):
            p = f"{name} (infrastructure node {idx+1})"
            cb = c_b.replace("services", f"services_{idx}").replace("customers", f"customers_{idx}")
            rb = r_b.replace("services", f"services_{idx}").replace("customers", f"customers_{idx}")
            add_pair_fn("coverage-files", p, cb, c_e, rb, r_e, flaw)

    # 6. coverage-logs
    # Focus: log rotation, error counting, frequency sorting, timestamp parsing, handling .gz
    logs_tasks = [
        ("Parse nginx access logs and list top 10 requesting client IP addresses",
         'log_file="/var/log/nginx/access.log"\n\nif [[ ! -f "$log_file" ]]; then\n    echo "Log file $log_file does not exist." >&2\n    exit 1\nfi\n\nawk \'{print $1}\' "$log_file" | sort | uniq -c | sort -nr | head -n 10',
         "Verifies file existence and parses the standard first field using awk, sorted by occurrence frequency.",
         'log_file="/var/log/nginx/access.log"\ncat $log_file | cut -d " " -f1 | sort | uniq -c | head -10',
         "Uses useless cat, fragile cut on space which fails on leading spaces, and unsorted frequency output.",
         "Fragile field cutting and missing frequency sort"),

        ("Search for critical errors across both active and compressed archived logs",
         'log_dir="/var/log/cluster"\npattern="CRITICAL_EXCEPTION"\n\nzgrep -H "$pattern" "$log_dir"/*.log "$log_dir"/*.log.gz 2>/dev/null || echo "No matches found."',
         "Uses zgrep to seamlessly search both uncompressed and gzip-compressed logs in a single pass.",
         'log_dir="/var/log/cluster"\npattern="CRITICAL_EXCEPTION"\ngrep -r "$pattern" $log_dir/*',
         "Standard grep fails to read binary compressed .log.gz files, returning 'binary file matches' without content.",
         "Inability to search compressed log files"),

        ("Atomically rotate an application log when it exceeds 50MB",
         'log="/var/log/engine/app.log"\nmax_bytes=52428800\n\nif [[ -f "$log" ]]; then\n    size=$(stat -c%s "$log")\n    if (( size > max_bytes )); then\n        ts=$(date +"%Y%m%d_%H%M%S")\n        rotated="${log}.${ts}"\n        mv "$log" "$rotated"\n        gzip "$rotated"\n        kill -USR1 "$(cat /var/run/engine.pid)" 2>/dev/null || true\n        echo "Rotated and compressed log to ${rotated}.gz"\n    fi\nfi',
         "Checks log file size in bytes, rotates with timestamp, compresses with gzip, and signals daemon to reopen file.",
         'log="/var/log/engine/app.log"\ncat /dev/null > $log\necho "Cleared log."',
         "Truncates log directly while application is writing without rotating or signaling, losing recent log data.",
         "Destructive in-place log truncation without archiving"),

        ("Count HTTP status code distribution in server logs for the current hour",
         'log="/var/log/httpd/access_log"\nhour_stamp=$(date +"%d/%b/%Y:%H")\n\ngrep "$hour_stamp" "$log" | awk \'{print $(NF-1)}\' | sort | uniq -c | sort -nr',
         "Filters by current hour timestamp before aggregating status code field frequencies.",
         'log="/var/log/httpd/access_log"\nawk \'{print $9}\' $log | sort | uniq -c',
         "Counts all historical entries regardless of timestamp and uses fixed field offset that breaks on custom formats.",
         "Unfiltered log aggregation ignoring time boundaries"),

        ("Extract and alert on consecutive SSH authentication failures",
         'auth_log="/var/log/auth.log"\nthreshold=5\n\nfailed_count=$(grep "Failed password for" "$auth_log" | tail -n 50 | wc -l)\nif (( failed_count >= threshold )); then\n    echo "[SECURITY ALERT] Detected $failed_count failed SSH attempts recently!" >&2\nfi',
         "Scans recent authentication failures and triggers structured stderr alert when exceeding threshold.",
         'auth_log="/var/log/auth.log"\ncount=$(grep "Failed" $auth_log | wc -l)\necho "Failures: $count"',
         "Counts all lifetime failures across old log entries without checking recent rate or alerting.",
         "Lifetime failure count without recent windowing or alert logic"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in logs_tasks:
        for idx in range(10):
            p = f"{name} (cluster node {idx+1})"
            cb = c_b.replace("nginx", f"nginx_{idx}").replace("engine", f"engine_{idx}")
            rb = r_b.replace("nginx", f"nginx_{idx}").replace("engine", f"engine_{idx}")
            add_pair_fn("coverage-logs", p, cb, c_e, rb, r_e, flaw)

    # 7. coverage-backups
    # Focus: timestamped archives, rsync synchronization, checksum generation & verification
    backups_tasks = [
        ("Synchronize database dump directory to remote backup storage with rsync",
         'src="/var/backups/postgres/"\ndest="backup-server:/mnt/storage/pg/"\n\necho "Starting synchronization..."\nrsync -avz --delete --timeout=30 "$src" "$dest"\necho "Backup synchronization completed successfully."',
         "Uses rsync with archive mode, deletion of stale remote files, and network timeout.",
         'src="/var/backups/postgres/"\ndest="backup-server:/mnt/storage/pg/"\nrsync -r $src $dest\necho "Done."',
         "Omits permissions/timestamp preservation (-a) and network timeout, risking silent hangs.",
         "Missing archive flags and network timeout in rsync"),

        ("Create a timestamped tar archive of web assets with sha256 checksum",
         'src_dir="/var/www/static"\nbackup_root="/srv/backups/web"\ntimestamp=$(date +"%Y%m%d_%H%M%S")\narchive="$backup_root/assets_${timestamp}.tar.gz"\n\nmkdir -p "$backup_root"\ntar -czf "$archive" -C "$src_dir" .\nsha256sum "$archive" > "${archive}.sha256"\necho "Archive created: $archive"',
         "Creates target directory, packages assets cleanly, and generates sidecar checksum file for verification.",
         'src_dir="/var/www/static"\narchive="/srv/backups/web/assets.tar.gz"\ntar -czf $archive $src_dir\necho "Created."',
         "Overwrites static filename on every run without timestamp, and omits checksum generation.",
         "Overwriting archive without timestamp or checksum"),

        ("Automate backup retention by pruning daily backups older than 30 days",
         'backup_dir="/srv/backups/db"\nretention_days=30\n\nfind "$backup_dir" -type f -name "*.tar.gz" -mtime "+$retention_days" -print0 | while IFS= read -r -d \'\' old_backup; do\n    echo "Purging expired backup: $old_backup"\n    rm -f -- "$old_backup"\ndone\necho "Retention policy applied."',
         "Prunes only files matching backup naming convention older than retention threshold using null-delimiters.",
         'backup_dir="/srv/backups/db"\nfind $backup_dir -mtime +30 -exec rm {} \\;\necho "Pruned backups."',
         "Fails to restrict search to files or naming patterns, risking deletion of subdirectories or metadata files.",
         "Unrestricted find exec rm without file type or pattern filter"),

        ("Verify integrity of all gzipped backups in the archive directory",
         'backup_dir="/srv/backups"\nshopt -s nullglob\nfailures=0\n\nfor gz in "$backup_dir"/*.tar.gz; do\n    if ! gzip -t "$gz" 2>/dev/null; then\n        echo "[CORRUPT] $gz failed integrity check!" >&2\n        ((failures++))\n    fi\ndone\n\nif (( failures > 0 )); then\n    exit 1\nfi\necho "All archives passed test."',
         "Runs non-destructive test (gzip -t) on every archive, logging corruption and returning error code if any fail.",
         'backup_dir="/srv/backups"\nfor gz in $backup_dir/*.tar.gz; do\n    gzip -t $gz\ndone',
         "Unquoted loop variable and ignores individual failure codes, not reporting total failed backups.",
         "Unquoted loop and missing error accumulation"),

        ("Create differential backup based on a reference timestamp file",
         'source_path="/srv/data"\nref_file="/srv/backups/.last_full_backup"\noutput_tar="/srv/backups/diff_$(date +%Y%m%d).tar.gz"\n\nif [[ ! -f "$ref_file" ]]; then\n    echo "Reference file missing!" >&2\n    exit 1\nfi\n\ntar -czf "$output_tar" --newer-mtime="$ref_file" -C "$source_path" .\necho "Differential backup created."',
         "Verifies reference marker file exists and creates differential tar containing only modified files.",
         'source_path="/srv/data"\nref_file="/srv/backups/.last_full_backup"\noutput_tar="/srv/backups/diff.tar.gz"\ntar -czf $output_tar $source_path',
         "Creates full archive instead of differential backup because --newer-mtime is omitted.",
         "Missing differential mtime flag resulting in unintended full backup"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in backups_tasks:
        for idx in range(10):
            p = f"{name} (backup tier {idx+1})"
            cb = c_b.replace("postgres", f"postgres_{idx}").replace("db", f"db_{idx}")
            rb = r_b.replace("postgres", f"postgres_{idx}").replace("db", f"db_{idx}")
            add_pair_fn("coverage-backups", p, cb, c_e, rb, r_e, flaw)

    # 8. coverage-processes
    # Focus: process monitoring, resource limits, graceful vs forced termination, pidfile handling
    processes_tasks = [
        ("Gracefully stop a daemon process using its PID file with fallback",
         'pidfile="/var/run/custom_service.pid"\n\nif [[ ! -f "$pidfile" ]]; then\n    echo "Service is not running (PID file absent)."\n    exit 0\nfi\n\npid=$(<"$pidfile")\nif ! kill -0 "$pid" 2>/dev/null; then\n    echo "Stale PID file detected. Removing."\n    rm -f "$pidfile"\n    exit 0\nfi\n\nkill -15 "$pid"\nfor i in {1..10}; do\n    if ! kill -0 "$pid" 2>/dev/null; then\n        echo "Service stopped gracefully."\n        rm -f "$pidfile"\n        exit 0\n    fi\n    sleep 1\ndone\n\necho "Force stopping service..."\nkill -9 "$pid" 2>/dev/null || true\nrm -f "$pidfile"',
         "Checks PID validity, sends SIGTERM, waits up to 10 seconds, and only falls back to SIGKILL if unresponsive.",
         'pidfile="/var/run/custom_service.pid"\nkill -9 $(cat $pidfile)\nrm $pidfile\necho "Stopped."',
         "Immediately issues SIGKILL without graceful termination and fails if PID file is missing.",
         "Direct SIGKILL without graceful teardown attempt"),

        ("Monitor daemon memory usage and alert when exceeding 80% limit",
         'pidfile="/var/run/worker.pid"\nlimit_kb=2097152\n\nif [[ -f "$pidfile" ]]; then\n    pid=$(<"$pidfile")\n    rss_kb=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d " ")\n    if [[ -n "$rss_kb" && "$rss_kb" -gt "$limit_kb" ]]; then\n        echo "[ALERT] Worker PID $pid exceeded memory limit ($(( rss_kb / 1024 )) MB)" >&2\n    fi\nfi',
         "Safely extracts RSS memory in KB and alerts if the threshold is exceeded.",
         'pidfile="/var/run/worker.pid"\npid=$(cat $pidfile)\nrss=$(ps -u $pid | awk \'{print $4}\')\necho "RSS: $rss"',
         "Uses invalid ps invocation syntax and fails to compare value numerically against limit.",
         "Invalid ps command syntax and missing comparison"),

        ("Restart worker service if process crashed or vanished",
         'service_cmd="/usr/local/bin/worker_daemon"\npidfile="/var/run/worker_daemon.pid"\n\nrestart_service=false\nif [[ ! -f "$pidfile" ]]; then\n    restart_service=true\nelse\n    pid=$(<"$pidfile")\n    if ! kill -0 "$pid" 2>/dev/null; then\n        restart_service=true\n    fi\nfi\n\nif "$restart_service"; then\n    echo "Restarting service..."\n    nohup "$service_cmd" > /var/log/daemon.log 2>&1 &\n    echo $! > "$pidfile"\nfi',
         "Accurately checks for missing or stale PID file and launches daemon in background with updated PID file.",
         'pidfile="/var/run/worker_daemon.pid"\n/usr/local/bin/worker_daemon &\necho $! > $pidfile',
         "Launches new instance unconditionally, spawning duplicate background processes on top of existing ones.",
         "Duplicate service launch without active instance checking"),

        ("Count threads for all running worker processes",
         'total_threads=0\nwhile IFS= read -r pid; do\n    threads=$(ps -o nlwp= -p "$pid" 2>/dev/null | tr -d " ")\n    if [[ -n "$threads" ]]; then\n        (( total_threads += threads ))\n    fi\ndone < <(pgrep -f "worker_pool" || true)\necho "Total active threads across workers: $total_threads"',
         "Iterates through PIDs safely, extracting and summing thread count (nlwp) with numeric validation.",
         'for p in $(pgrep worker_pool); do\n    ps -eLf | grep $p\ndone | wc -l',
         "Inaccurate thread counting that includes grep processes and fails on empty pgrep results.",
         "Inaccurate thread tallying using grep pipeline"),

        ("Wait for a background task PID to exit with timeout",
         'target_pid="$1"\ntimeout=30\nelapsed=0\n\nwhile kill -0 "$target_pid" 2>/dev/null; do\n    if (( elapsed >= timeout )); then\n        echo "Timeout waiting for PID $target_pid to finish" >&2\n        exit 1\n    fi\n    sleep 1\n    ((elapsed++))\ndone\necho "Process $target_pid exited successfully."',
         "Monitors PID existence in loop with bounded timeout, returning error code 1 on timeout.",
         'target_pid=$1\nwhile kill -0 $target_pid; do\n    sleep 1\ndone\necho "Done."',
         "Lacks timeout mechanism, leading to indefinite hang if target process becomes zombie or hangs.",
         "Unbounded while loop waiting on process without timeout"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in processes_tasks:
        for idx in range(10):
            p = f"{name} (daemon tier {idx+1})"
            cb = c_b.replace("custom_service", f"service_{idx}").replace("worker_daemon", f"daemon_{idx}")
            rb = r_b.replace("custom_service", f"service_{idx}").replace("worker_daemon", f"daemon_{idx}")
            add_pair_fn("coverage-processes", p, cb, c_e, rb, r_e, flaw)

    # 9. coverage-text
    # Focus: CSV to TSV conversion, JSON processing with jq, stream filtering, regex transformations
    text_tasks = [
        ("Parse a JSON configuration file and extract database connection details using jq",
         'config_file="/etc/app/config.json"\n\nif [[ ! -f "$config_file" ]]; then\n    echo "Configuration file missing." >&2\n    exit 1\nfi\n\ndb_host=$(jq -er \'.database.host // empty\' "$config_file")\ndb_port=$(jq -er \'.database.port // 5432\' "$config_file")\necho "Connecting to $db_host on port $db_port"',
         "Uses jq with -e to enforce field presence and default port fallback safely.",
         'config_file="/etc/app/config.json"\nhost=$(grep "host" $config_file | cut -d: -f2 | tr -d \'", \')\nport=$(grep "port" $config_file | cut -d: -f2 | tr -d \'", \')\necho "Connecting to $host:$port"',
         "Parses JSON with fragile regex and cut, breaking on nested structures or formatted whitespace.",
         "Parsing structured JSON with line-based grep and cut"),

        ("Convert a comma-separated file to tab-separated format while preserving embedded commas",
         'input_csv="/data/reports/orders.csv"\noutput_tsv="/data/reports/orders.tsv"\n\npython3 -c "\nimport csv, sys\nwith open(sys.argv[1], newline=\'\') as f_in, open(sys.argv[2], \'w\', newline=\'\') as f_out:\n    reader = csv.reader(f_in)\n    writer = csv.writer(f_out, delimiter=\'\\t\')\n    writer.writerows(reader)\n" "$input_csv" "$output_tsv"\necho "Conversion completed: $output_tsv"',
         "Leverages standard Python csv module to preserve quoted comma fields correctly.",
         'input_csv="/data/reports/orders.csv"\noutput_tsv="/data/reports/orders.tsv"\nsed \'s/,/\\t/g\' "$input_csv" > "$output_tsv"\necho "Done."',
         "Global sed replacement breaks quoted CSV fields containing embedded commas.",
         "Naive sed substitution breaking quoted CSV fields"),

        ("Filter and pretty-print JSON log entries matching ERROR level",
         'log_file="/var/log/app.json.log"\n\nif [[ ! -f "$log_file" ]]; then\n    echo "Log file not found." >&2\n    exit 1\nfi\n\njq -c \'select(.level == "ERROR" or .level == "CRITICAL")\' "$log_file"',
         "Uses jq select expression to stream and filter JSON log records by level attribute.",
         'log_file="/var/log/app.json.log"\ngrep "ERROR" $log_file',
         "Plain text grep returns false positives if the word ERROR appears inside message payloads or stacktraces.",
         "Unstructured grep on JSON log stream"),

        ("Sanitize text by replacing Windows CRLF line endings with Unix LF line endings",
         'target_file="/data/imports/data.txt"\ntmp_file="/data/imports/data.txt.tmp"\n\ntr -d \'\\r\' < "$target_file" > "$tmp_file"\nmv -f "$tmp_file" "$target_file"\necho "Converted CRLF to LF in $target_file"',
         "Converts CRLF via tr safely writing to a temporary file before atomic replacement.",
         'target_file="/data/imports/data.txt"\nsed -i \'s/\\r$//\' $target_file\necho "Converted."',
         "Direct sed in-place modification unquoted, which fails if file permissions or symlinks exist.",
         "Unquoted sed in-place replacement"),

        ("Extract distinct email domains from a contact list text file",
         'contacts="/data/contacts.txt"\n\nawk -F\'@\' \'NF==2 {print $2}\' "$contacts" | sort -u | grep -E \'^([a-zA-Z0-9-]+\\.)+[a-zA-Z]{2,}$\'' ,
         "Splits email addresses cleanly at @ and validates domain format before deduplicating.",
         'contacts="/data/contacts.txt"\ncat $contacts | cut -d@ -f2 | sort -u',
         "Extracts lines with cut without checking if an email is present, polluting output with header lines.",
         "Unvalidated field cut on text file"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in text_tasks:
        for idx in range(10):
            p = f"{name} (data batch {idx+1})"
            cb = c_b.replace("orders", f"orders_{idx}").replace("config", f"config_{idx}")
            rb = r_b.replace("orders", f"orders_{idx}").replace("config", f"config_{idx}")
            add_pair_fn("coverage-text", p, cb, c_e, rb, r_e, flaw)

    # 10. coverage-automation
    # Focus: cron wrappers, lockfiles with flock, exponential backoff retries, systemd helpers
    automation_tasks = [
        ("Acquire an exclusive lock before running scheduled maintenance script",
         'lock_file="/var/lock/nightly_cleanup.lock"\n\nexec 200>"$lock_file"\nif ! flock -n 200; then\n    echo "Another instance is already running. Exiting."\n    exit 0\nfi\n\necho "Lock acquired. Running cleanup..."\nsleep 2\necho "Cleanup finished."',
         "Uses flock on dedicated file descriptor 200 with non-blocking check (-n) to prevent overlapping cron runs.",
         'lock_file="/tmp/cleanup.lock"\nif [ -f "$lock_file" ]; then\n    echo "Running."\n    exit 0\nfi\ntouch "$lock_file"\nsleep 2\nrm "$lock_file"',
         "Touch-based lockfile has a race condition between the test and touch, permitting concurrent execution.",
         "Non-atomic lockfile creation with race condition"),

        ("Wrap a periodic network synchronization job with structured exit status logging",
         'log="/var/log/sync_job.log"\n\n{\n    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Job started"\n    if /usr/local/bin/sync_assets; then\n        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Job succeeded with exit code 0"\n    else\n        rc=$?\n        echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] Job failed with exit code $rc" >&2\n        exit "$rc"\n    fi\n} >> "$log" 2>&1',
         "Captures exact UTC timestamps and returns the underlying exit code while recording all output.",
         'log="/var/log/sync_job.log"\n/usr/local/bin/sync_assets >> $log\necho "Done" >> $log',
         "Does not capture exit status or timestamps, making failure diagnosis in cron jobs difficult.",
         "Missing timestamped exit status logging in cron wrapper"),

        ("Implement exponential backoff retry logic for a flaky remote API call",
         'api_url="https://api.internal/v1/ping"\nmax_attempts=4\nattempt=1\ndelay=1\n\nwhile (( attempt <= max_attempts )); do\n    echo "Calling endpoint (attempt $attempt)..."\n    if curl -fsS --max-time 5 "$api_url" > /dev/null; then\n        echo "API call succeeded."\n        exit 0\n    fi\n    echo "Attempt $attempt failed. Backing off for ${delay}s..." >&2\n    sleep "$delay"\n    (( delay *= 2 ))\n    (( attempt++ ))\ndone\necho "API call failed after $max_attempts attempts." >&2\nexit 1',
         "Implements exponential delay doubling with bounded attempts and explicit failure code.",
         'api_url="https://api.internal/v1/ping"\nwhile ! curl $api_url; do\n    sleep 1\ndone\necho "Success"',
         "Infinite loop without exponential backoff or retry ceiling, hammering the remote API.",
         "Infinite retry loop without exponential backoff or timeout"),

        ("Check status of a systemd unit and restart if inactive",
         'service="worker-pipeline.service"\n\nif ! systemctl is-active --quiet "$service"; then\n    echo "Service $service is inactive! Attempting restart..." >&2\n    systemctl restart "$service"\n    sleep 2\n    if systemctl is-active --quiet "$service"; then\n        echo "Successfully recovered $service."\n    else\n        echo "Failed to recover $service!" >&2\n        exit 1\n    fi\nfi',
         "Checks active state with --quiet and verifies service recovered successfully after restarting.",
         'service="worker-pipeline.service"\nsystemctl restart $service\necho "Restarted."',
         "Unconditionally restarts service without checking whether it was already healthy, interrupting ongoing tasks.",
         "Unconditional restart without active status check"),

        ("Verify availability of required binary dependencies before task execution",
         'required_tools=("jq" "curl" "rsync" "tar")\nmissing=()\n\nfor tool in "${required_tools[@]}"; do\n    if ! command -v "$tool" >/dev/null 2>&1; then\n        missing+=("$tool")\n    fi\ndone\n\nif (( ${#missing[@]} > 0 )); then\n    echo "Missing required tools: ${missing[*]}" >&2\n    exit 1\nfi\necho "All dependency checks passed."',
         "Validates all required executables using command -v, listing all missing tools before exiting.",
         'which jq curl rsync tar || exit 1\necho "All tools found."',
         "Uses legacy which instead of command -v and stops at first missing tool without clear summary.",
         "Use of legacy which and incomplete dependency report"),
    ]

    for name, c_b, c_e, r_b, r_e, flaw in automation_tasks:
        for idx in range(10):
            p = f"{name} (automation routine {idx+1})"
            cb = c_b.replace("nightly_cleanup", f"cleanup_{idx}").replace("sync_job", f"sync_{idx}")
            rb = r_b.replace("nightly_cleanup", f"cleanup_{idx}").replace("sync_job", f"sync_{idx}")
            add_pair_fn("coverage-automation", p, cb, c_e, rb, r_e, flaw)
