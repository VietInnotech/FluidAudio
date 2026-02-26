#!/usr/bin/env bash
set -euo pipefail

input_json="$(cat)"
file_path="$(jq -r '.tool_input.file_path // empty' <<<"$input_json")"

if [[ -z "$file_path" || "$file_path" != *.swift ]]; then
    exit 0
fi

project_dir="${CLAUDE_PROJECT_DIR:-$(pwd)}"

if [[ "$file_path" != /* ]]; then
    file_path="$project_dir/$file_path"
fi

if [[ "$file_path" != "$project_dir/"* ]]; then
    echo "Skipping format outside project: $file_path" >&2
    exit 0
fi

if [[ "$file_path" == "$project_dir/.build/"* || "$file_path" == "$project_dir/.git/"* ]]; then
    exit 0
fi

if [[ ! -f "$file_path" ]]; then
    exit 0
fi

swift format --in-place --configuration "$project_dir/.swift-format" "$file_path"
echo "Formatted $file_path"
