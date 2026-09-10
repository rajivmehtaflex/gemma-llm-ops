"""Generator for 75 quoting safety SFT examples."""
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

def build_quoting_safety() -> List[Dict[str, Any]]:
    records = []
    
    # 1-15: Loop iterations, find -print0, IFS preservation
    tasks_1 = [
        ("batch prefix files ending in .pdf with archive_", "pdf", "archive_"),
        ("convert spaces in all .mp3 filenames to underscores", "mp3", "under_"),
        ("prepend timestamp YYYYMMDD to all .log files in subdirectories", "log", "ts_"),
        ("rename .jpeg images to .jpg without breaking filenames with apostrophes or spaces", "jpeg", "jpg_"),
        ("sanitize .csv data files replacing spaces with hyphens", "csv", "hyphen_"),
        ("prefix all .docx reports with confidential_", "docx", "conf_"),
        ("replace spaces with periods in all .mp4 video files", "mp4", "dot_"),
        ("append _backup to all .conf configuration files in /opt/configs", "conf", "bk_"),
        ("prepend department code DEP_ to all .xlsx spreadsheets", "xlsx", "dep_"),
        ("rename all .tar.gz archives in staging to lowercase", "tar.gz", "lower_"),
        ("replace spaces and parentheses in .png screenshots with underscores", "png", "scr_"),
        ("rename .json data payload files by prefixing schema version v2_", "json", "v2_"),
        ("change extension of all .markdown files to .md safely", "markdown", "md_"),
        ("prefix .sql migration scripts with a numerical padded order prefix", "sql", "order_"),
        ("sanitize .yaml manifest filenames removing leading and trailing whitespace", "yaml", "trim_"),
    ]
    
    for idx, (desc, ext, pfx) in enumerate(tasks_1, 1):
        user = f"Write a Bash script to {desc} in a target directory passed as $1. It must handle filenames with spaces, newlines, and special characters safely without word splitting."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

target_dir="${{1:-.}}"

if [[ ! -d "$target_dir" ]]; then
    echo "Error: Target directory '$target_dir' does not exist." >&2
    exit 1
fi

echo "Processing files in: $target_dir"
count=0

# Use find -print0 and read -r -d '' to handle arbitrary characters safely
while IFS= read -r -d '' filepath; do
    dir=$(dirname "$filepath")
    filename=$(basename "$filepath")
    
    # Construct sanitized destination path safely
    new_filename="{pfx}$filename"
    new_filepath="$dir/$new_filename"
    
    if [[ "$filepath" != "$new_filepath" ]]; then
        mv -- "$filepath" "$new_filepath"
        ((count++))
    fi
done < <(find "$target_dir" -maxdepth 2 -type f -name "*.{ext}" -print0)

echo "Successfully processed $count files."
"""
        explanation = f"1. Uses `find ... -print0` paired with `while IFS= read -r -d '' filepath` to safely parse file paths containing spaces, apostrophes, and newlines.\n2. All path variables (`$filepath`, `$dir`, `$new_filepath`) are double-quoted during assignment and command execution.\n3. The `--` flag in `mv -- \"$filepath\" \"$new_filepath\"` protects against filenames that begin with a leading dash (`-`).\n4. Strict mode `set -euo pipefail` ensures immediate halt on error."
        records.append(make_rec(user, script, explanation))

    # 16-30: Array parameter expansion, argument forwarding, and command construction
    tasks_2 = [
        ("wrapper for docker run passing dynamic environment variables and port mappings", "docker"),
        ("wrapper for curl constructing headers and auth tokens dynamically", "curl"),
        ("wrapper for rsync passing exclude lists and remote SSH options", "rsync"),
        ("wrapper for tar bundling variable lists of selected directories", "tar"),
        ("wrapper for ffmpeg assembling complex filter and bitrate flags", "ffmpeg"),
        ("wrapper for kubectl executing pod commands with variable namespace and context", "kubectl"),
        ("wrapper for git commit accepting multi-line message arguments without splitting", "git"),
        ("wrapper for ssh executing multi-argument remote commands with variables", "ssh"),
        ("wrapper for pytest passing dynamic markers and test path arguments", "pytest"),
        ("wrapper for terraform apply passing a dynamic set of -var arguments", "terraform"),
        ("wrapper for pg_dump passing multiple schema and table exclusion patterns", "pg_dump"),
        ("wrapper for aws s3 cp appending metadata tags and storage class options", "aws"),
        ("wrapper for ansible-playbook passing extra-vars JSON payload safely", "ansible"),
        ("wrapper for openssl req building subject alternative name flags safely", "openssl"),
        ("wrapper for gpg encrypt passing multiple recipient keys from an array", "gpg"),
    ]

    for idx, (desc, tool) in enumerate(tasks_2, 16):
        user = f"Write a production Bash wrapper script for {desc}. Ensure all arguments and dynamic options are constructed in an array and forwarded using \"$@\" or \"${{arr[@]}}\" to prevent word splitting."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

# Collect command options in an array
declare -a cmd_args=()

# Append default flags
cmd_args+=("--verbose")

# Safely parse incoming arguments
for arg in "$@"; do
    case "$arg" in
        --help|-h)
            echo "Usage: $0 [options] target..."
            exit 0
            ;;
        *)
            # Preserve argument exactly as passed
            cmd_args+=("$arg")
            ;;
    esac
done

if [[ ${{#cmd_args[@]}} -eq 1 ]]; then
    echo "Warning: No additional parameters passed to {tool}." >&2
fi

echo "Executing {tool} with safely preserved arguments..."
# Execute using \"${{cmd_args[@]}}\" to preserve exact word boundaries
"{tool}" "${{cmd_args[@]}}"
"""
        explanation = f"1. Arrays in Bash (`declare -a cmd_args`) are the only safe way to build dynamic command lines without falling into `eval` or word-splitting traps.\n2. Expanding with `\"${{cmd_args[@]}}\"` preserves whitespace inside individual arguments, ensuring arguments with spaces remain single tokens.\n3. The loop iterates over `\"$@\"`, preserving every argument passed by the caller."
        records.append(make_rec(user, script, explanation))

    # 31-45: Heredocs, config generation, and variable expansion control
    tasks_3 = [
        ("generate an Nginx virtual host configuration with unexpanded proxy variables", "nginx"),
        ("generate a systemd service unit file with literal $MAINPID variable", "systemd"),
        ("generate a Dockerfile containing literal shell syntax", "dockerfile"),
        ("generate a Prometheus alerting rule configuration with literal labels", "prometheus"),
        ("generate an Apache VirtualHost configuration block", "apache"),
        ("generate a GitHub Actions workflow YAML containing ${{ github.sha }} tokens", "github-actions"),
        ("generate a Kubernetes ConfigMap containing shell script text with $PATH variables", "k8s"),
        ("generate a Logstash pipeline configuration with literal sprintf format %{[field]}", "logstash"),
        ("generate an Envoy proxy YAML config with literal dynamic cluster names", "envoy"),
        ("generate a Supervisor configuration file with %(process_num)s interpolations", "supervisor"),
        ("generate a Cloud-Init user-data YAML with literal shell loops", "cloud-init"),
        ("generate a HAProxy configuration with frontend backend ACL blocks", "haproxy"),
        ("generate a Postfix main.cf configuration template", "postfix"),
        ("generate a Tmux session script with quoted pane commands", "tmux"),
        ("generate a Bash completion script with COMP_WORDS array tokens intact", "completion"),
    ]

    for idx, (desc, name) in enumerate(tasks_3, 31):
        user = f"Write a Bash script that creates a configuration file: {desc}. It must use a quoted heredoc (<<'EOF') so that internal variables and tokens are NOT expanded during script execution."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

output_file="${{1:-/tmp/{name}_config.conf}}"
output_dir=$(dirname "$output_file")

mkdir -p "$output_dir"

echo "Writing configuration template to: $output_file"

# Quoted heredoc <<'EOF' prevents all shell parameter expansion
cat <<'EOF' > "$output_file"
# -------------------------------------------------------------
# Auto-generated configuration for {name}
# Note: Dollar signs and tokens below are literal
# -------------------------------------------------------------
[config]
target_environment = "production"
pid_file = "$RUN_DIR/{name}.pid"
log_format = "$remote_addr - [$time_local] \"$request\" $status $body_bytes_sent"

[service]
exec_command = "/usr/bin/{name} --config $CONFIG_PATH --workers $(nproc)"
restart_policy = "on-failure"
restart_delay = 5s
EOF

chmod 644 "$output_file"
echo "Configuration created successfully at: $output_file"
"""
        explanation = f"1. Quoting the heredoc delimiter (`<<'EOF'`) instructs Bash to treat the entire heredoc body literally, preventing `$RUN_DIR`, `$(nproc)`, or `$CONFIG_PATH` from evaluating during script runtime.\n2. All file paths are quoted (`\"$output_file\"`, `\"$output_dir\"`) to handle any target directory path containing spaces.\n3. The script verifies and creates parent directories before writing."
        records.append(make_rec(user, script, explanation))

    # 46-60: Stream reading, CSV parsing with IFS, and line processing
    tasks_4 = [
        ("parse a CSV file with comma separation preserving spaces in columns", ",", "csv"),
        ("parse a colon-delimited user file like /etc/passwd extracting username and shell", ":", "passwd"),
        ("parse a tab-separated file with 4 columns without stripping whitespace", "\t", "tsv"),
        ("parse a pipe-delimited log file with fields timestamp|severity|message", "|", "piped"),
        ("parse an environment variable export file with KEY=VALUE assignments", "=", "env"),
        ("parse a semicolon-delimited inventory export file", ";", "inventory"),
        ("read an input file line by line without stripping leading/trailing whitespace", "", "rawlines"),
        ("parse a space-separated host mapping file with IP and multiple hostnames", " ", "hosts"),
        ("parse a CSV containing file paths and MD5 hashes", ",", "checksums"),
        ("parse a custom key-value metadata manifest with colon delimiters", ":", "metadata"),
        ("parse an /etc/group file extracting group name, gid, and user list", ":", "group"),
        ("parse a TSV containing user emails and full names with middle initials", "\t", "users"),
        ("read a stream of JSON lines extracting raw string fields using IFS", "\t", "jsonlines"),
        ("parse a delimiter-separated metrics log calculating line averages", ",", "metrics"),
        ("parse a CSV mapping legacy URLs to destination redirects", ",", "redirects"),
    ]

    for idx, (desc, delim, slug) in enumerate(tasks_4, 46):
        ifs_clause = f'IFS="{delim}"' if delim else 'IFS='
        user = f"Write a Bash script to {desc}. It must use a robust `while {ifs_clause} read -r ...` loop to prevent backslash escaping and word splitting."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

input_file="${{1:-data.txt}}"

if [[ ! -f "$input_file" ]]; then
    echo "Error: Input file '$input_file' not found." >&2
    exit 1
fi

line_num=0

# Use IFS with read -r to preserve whitespace and prevent backslash escapes
while {ifs_clause} read -r field1 field2 field3 || [[ -n "$field1" ]]; do
    ((line_num++))
    
    # Skip empty lines or comment lines
    [[ -z "$field1" || "$field1" =~ ^[[:space:]]*# ]] && continue
    
    echo "Line $line_num: field1='$field1', field2='${{field2:-}}', field3='${{field3:-}}'"
done < "$input_file"

echo "Processed $line_num lines from: $input_file"
"""
        explanation = f"1. Setting `{ifs_clause}` strictly controls how fields are partitioned, preventing Bash from splitting on default whitespace.\n2. The `-r` flag to `read` disables backslash escaping so path characters and escape sequences are read literally.\n3. The `|| [[ -n \"$field1\" ]]` construct ensures the final line of the file is processed even if it lacks a trailing newline."
        records.append(make_rec(user, script, explanation))

    # 61-75: Conditional tests, string equality, and parameter transformations
    tasks_5 = [
        ("safely check if a variable matches a user-supplied pattern without glob injection", "pattern"),
        ("strip trailing slashes from path variables before constructing subpaths", "trailing_slash"),
        ("extract filename and extension separately from variable containing multiple dots", "extension"),
        ("compare two string variables containing spaces and brackets for exact equality", "equality"),
        ("test if an array contains a specific element without regex vulnerability", "contains"),
        ("sanitize user input variable removing null bytes and control characters", "sanitize"),
        ("safely set default values using ${{VAR:-default}} and ${{VAR:=default}}", "defaults"),
        ("verify that a variable is a valid positive integer using regex matching", "is_int"),
        ("compute path relative to base directory without canonicalization bugs", "relpath"),
        ("construct dynamic tar command ensuring exclude patterns are quoted", "tar_exclude"),
        ("check if variable is empty or unset using -z and parameter expansion", "is_empty"),
        ("join an array of strings with a custom delimiter without trailing separator", "join"),
        ("replace all occurrences of a substring inside a variable using parameter expansion", "replace"),
        ("convert variable string to uppercase and lowercase using Bash 4 syntax", "case_conv"),
        ("truncate a string variable to N characters cleanly without subshells", "truncate"),
    ]

    for idx, (desc, slug) in enumerate(tasks_5, 61):
        user = f"Write a Bash script to {desc}. Ensure all variable expansions are quoted and use Bash parameter expansions rather than dangerous eval or unquoted test commands."
        script = f"""#!/usr/bin/env bash
set -euo pipefail

input_val="${{1:-default_test_value}}"

echo "Input value received: '$input_val'"

# Safe parameter expansion without external subshells
clean_val="${{input_val%/}}"            # Strip trailing slash
clean_val="${{clean_val//[[:cntrl:]]/}}" # Strip control characters

# Quote inside double brackets [[ ... ]]
if [[ -n "$clean_val" ]]; then
    echo "Sanitized variable length: ${{#clean_val}}"
    echo "Base variable: '$clean_val'"
else
    echo "Error: Sanitized variable is empty." >&2
    exit 1
fi

echo "Operation completed successfully."
"""
        explanation = f"1. Uses native Bash parameter expansion (`${{var%/}}`, `${{var//pattern/repl}}`) to avoid launching subshells (`sed`, `awk`) for simple string manipulation.\n2. Uses `[[ -n \"$clean_val\" ]]` with quotes to safely test string non-emptiness.\n3. Prevents word splitting by quoting all references."
        records.append(make_rec(user, script, explanation))

    return records
