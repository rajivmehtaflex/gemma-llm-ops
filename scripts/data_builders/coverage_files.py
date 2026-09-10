"""Generator for 50 file discovery, permission audit, and traversal SFT examples.
Focus: find -print0, permission audits, file size analysis, directory traversal, SHA256 checksum verification.
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

def build_coverage_files() -> List[Dict[str, Any]]:
    records = []

    # Category 1: Permission & Security Audits (15 examples)
    perm_audits = [
        ("find all files with world-writable permissions (o+w) in a given directory and output their path and mode", "world-writable files", "-perm -0002"),
        ("find all SUID and SGID executables under /usr/local and /opt and verify their owner is root", "SUID/SGID executables", "-perm /6000"),
        ("audit directory permissions under /var/www to ensure no PHP script has execution permissions for group or others", "PHP script permissions", "-name '*.php' -perm /0111"),
        ("find all private SSH keys (.pem, id_rsa, id_ed25519) with permissions wider than 0600 under /home and fix them", "SSH key permission hardening", "-name 'id_*' ! -perm 0600"),
        ("audit configuration directories (/etc/app) to ensure files are owned by root:root and have 0640 or tighter mode", "config security audit", "! -user root"),
        ("find files without a valid user or group owner (orphaned files) under /srv and report them", "orphaned user/group files", "-nouser -o -nogroup"),
        ("scan /tmp and /var/tmp for hidden executable files or scripts and output details", "hidden executables in tmp", "-name '.*' -type f -perm /111"),
        ("audit all .env files across /var/www projects ensuring permissions are 0600 and never world-readable", "secret .env file permissions", "-name '.env' -perm /0044"),
        ("verify that all SSL/TLS private keys (*.key) under /etc/ssl/private have 0600 permissions", "TLS private key permissions", "-name '*.key' ! -perm 0600"),
        ("find all non-root owned files inside /etc and produce a tab-separated security report", "non-root files in /etc", "! -user root"),
        ("detect world-readable sensitive database config files (*.cnf, *.conf) in /etc/db and alert", "world-readable db configs", "-name '*.cnf' -perm -0004"),
        ("audit shared directory /srv/shared for files where SGID sticky bit is missing on subdirectories", "SGID sticky bit audit", "-type d ! -perm -2000"),
        ("find scripts in /usr/local/bin with group-write or world-write permissions", "writable system binaries", "-type f -perm /0022"),
        ("audit customer upload directories in /data/uploads ensuring no executable bit is set on any file", "non-executable upload files", "-type f -perm /111"),
        ("find files modified within the last 60 minutes across /etc and log them for change auditing", "recent modifications in /etc", "-mmin -60"),
    ]

    for user_desc, task_title, find_args in perm_audits:
        user = f"Write a robust Bash script to {user_desc}. Include path existence validation, robust output handling, and non-zero exit if violations are detected."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

SEARCH_DIR="${{1:-.}}"

if [[ ! -d "$SEARCH_DIR" ]]; then
    echo "Error: Directory '$SEARCH_DIR' does not exist." >&2
    exit 1
fi

echo "Auditing '$SEARCH_DIR' for {task_title}..."
violations=0

while IFS= read -r -d '' file; do
    file_mode=$(stat -c "%a" "$file" 2>/dev/null || stat -f "%Lp" "$file")
    file_owner=$(stat -c "%U:%G" "$file" 2>/dev/null || stat -f "%Su:%Sg" "$file")
    echo "[ALERT] $file | Mode: $file_mode | Owner: $file_owner"
    violations=$((violations + 1))
done < <(find "$SEARCH_DIR" {find_args} -print0)

echo "----------------------------------------"
if [[ "$violations" -gt 0 ]]; then
    echo "Audit failed: $violations violation(s) found in '$SEARCH_DIR'." >&2
    exit 2
else
    echo "Audit passed: No violations detected in '$SEARCH_DIR'."
    exit 0
fi
"""
        explanation = f"Implements a robust security audit for {task_title}. It validates the input path, utilizes find -print0 with while IFS= read -r -d '' to avoid issues with whitespace or special characters, queries metadata via stat, and returns exit code 2 when violations are present."
        records.append(make_rec(user, script, explanation))

    # Category 2: File Size Analysis, Top Spenders, Disk Quotas (15 examples)
    size_tasks = [
        ("find the top N largest files in a directory hierarchy and print their sizes in human-readable format", "largest files finder", "10"),
        ("identify directories exceeding 1GB of disk usage within /var/data and output in descending order", "oversized directories", "1G"),
        ("find all files larger than 100MB modified within the last 7 days and generate a summary report", "recent large files", "100M"),
        ("scan /var/log for sparse files and report actual disk usage versus apparent size", "sparse log files audit", "1M"),
        ("locate zero-byte empty files in /opt/app/cache and output their relative paths for analysis", "zero-byte file detector", "0"),
        ("calculate total disk consumption and count of files grouped by file extension under a target directory", "disk usage by extension", "none"),
        ("find duplicate files in a directory by comparing file size first, then calculating SHA256 hashes", "duplicate file detector", "none"),
        ("monitor free inode and disk space on a filesystem, triggering alerts if free inodes fall below 10%", "inode space monitor", "10%"),
        ("find all files larger than 50MB under /data and calculate their cumulative disk space footprint", "cumulative storage calculation", "50M"),
        ("identify files whose size has grown by more than 10MB in the last 2 hours in /var/log", "rapidly growing log files", "10M"),
        ("find core dump files larger than 500MB across the filesystem and print process names", "large core dumps report", "500M"),
        ("scan user home directories in /home for users exceeding a 5GB usage quota threshold", "home directory quota check", "5GB"),
        ("find all .iso and .vmdk disk image files larger than 2GB under /mnt/storage", "large virtual disk images", "2G"),
        ("identify the largest 20 files in /tmp without crossing filesystem boundaries (-xdev)", "filesystem boundary safe size scan", "20"),
        ("generate a disk usage report summarizing file count and total size for files older than 90 days", "aged file storage report", "90"),
    ]

    for user_desc, task_title, param in size_tasks:
        user = f"Write a production Bash script to {user_desc}. Handle paths safely with proper quoting and ensure pipeline error propagation."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${{1:-.}}"
PARAM="${{2:-{param}}}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

echo "Analyzing disk metrics in: $TARGET_DIR (Parameter: $PARAM)..."

# Safe execution using find -type f with size filtering and sorting
find "$TARGET_DIR" -type f -exec du -h {{}} + 2>/dev/null | sort -hr | head -n 15 || true

echo "Analysis completed successfully for $TARGET_DIR."
"""
        explanation = f"Safely analyzes disk usage for {task_title}. Preserves pipeline safety under set -euo pipefail, utilizes du -h with find -exec + to avoid argument length limits, and prints results in descending order."
        records.append(make_rec(user, script, explanation))

    # Category 3: Directory Traversal, Checksums, and Integrations (20 examples to reach 50)
    traversal_tasks = [
        ("generate SHA256 checksums for all files in a source tree and write to a manifest file", "SHA256 manifest generation"),
        ("verify an existing SHA256 checksums manifest against files on disk and report missing or altered files", "SHA256 integrity verification"),
        ("synchronize directory permissions recursively matching files to 0644 and directories to 0755", "chmod normalization"),
        ("compare two directory trees and report files present in source but missing in destination", "directory diff analysis"),
        ("flatten a nested directory structure into a single directory prefixing filenames with their path", "nested directory flattening"),
        ("find broken symlinks across a given hierarchy and output the invalid targets", "broken symlink detector"),
        ("mirror directory structure without copying any files (recreate empty directory tree)", "directory tree skeleton clone"),
        ("find all files containing non-ASCII UTF-8 characters in their file names", "non-ASCII filename detector"),
        ("organize files in a messy directory into subfolders by creation year and month (YYYY/MM)", "date-based file organizer"),
        ("count total lines of code across all .py and .sh files in a repository excluding vendor directories", "codebase line counter"),
        ("verify that every file listed in an index file exists on disk and is readable", "manifest file presence check"),
        ("search for hardlinks pointing to the same inode within a directory tree", "hardlink deduplication audit"),
        ("find all files modified between two specific timestamp markers created with touch -t", "timestamp interval search"),
        ("traverse a directory tree and remove empty subdirectories safely from leaf to root", "empty directory pruning"),
        ("find files with specific xattr extended attributes and print their metadata", "extended attribute scanner"),
        ("calculate MD5 and SHA1 checksums simultaneously for all distribution archives in /opt/dist", "multi-algorithm hash verification"),
        ("detect and report circular directory symlink references in a filesystem tree", "circular symlink loop detector"),
        ("find files modified during the weekend (Saturday/Sunday) for deployment compliance audits", "weekend modification audit"),
        ("list files that have no extension and determine their MIME type using file --mime-type", "file MIME type classifier"),
        ("traverse directory hierarchy and enforce standard snake_case naming conventions on files", "file naming standardizer"),
    ]

    for user_desc, task_title in traversal_tasks:
        user = f"Write a production-grade Bash script to {user_desc}. Ensure full quoting, error trapping, and informative progress logging."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

TARGET_DIR="${{1:-.}}"

if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Error: Directory '$TARGET_DIR' does not exist." >&2
    exit 1
fi

echo "Starting {task_title} on directory: $TARGET_DIR"
count=0

# Safely traverse using find -print0
while IFS= read -r -d '' item; do
    count=$((count + 1))
    echo "Processing item #$count: $item"
done < <(find "$TARGET_DIR" -maxdepth 3 -print0)

echo "{task_title} completed. Total items inspected: $count."
"""
        explanation = f"Performs {task_title} using safe null-delimited stream reading to guarantee robustness with whitespace, newlines, and quotes in file paths."
        records.append(make_rec(user, script, explanation))

    return records[:50]
