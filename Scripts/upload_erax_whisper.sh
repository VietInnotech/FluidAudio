#!/usr/bin/env bash
# Upload EraX-WoW-Turbo V1.1 CoreML models to HuggingFace Hub.
#
# Usage:
#   export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'
#   bash Tools/upload_erax_whisper.sh
#
# Prerequisites:
#   - Run Tools/convert_erax_whisper_to_coreml.py first
#   - HF_TOKEN must have WRITE access to the FluidInference org

set -e

HF_REPO="FluidInference/erax-wow-turbo-v1.1-coreml"
MODEL_DIR="./build/erax-wow-turbo-v1.1-coreml"
COMMIT_MSG="Upload EraX-WoW-Turbo V1.1 CoreML models (whisper-large-v3-turbo fine-tune)"

echo "========================================================"
echo "EraX-WoW-Turbo V1.1 CoreML → HuggingFace Upload"
echo "========================================================"
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable not set"
    echo ""
    echo "  export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'"
    echo ""
    exit 1
fi

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: Model directory not found: $MODEL_DIR"
    echo ""
    echo "Run the conversion script first:"
    echo "  uv run Tools/convert_erax_whisper_to_coreml.py"
    echo ""
    exit 1
fi

echo "✓ HF_TOKEN is set"
echo "✓ Model directory: $MODEL_DIR"
echo "→ Target repo:     $HF_REPO"
echo ""

# Check for huggingface_hub
PYTHON_BIN="${PYTHON:-python3}"
if [ -f ".venv/bin/python" ]; then
    PYTHON_BIN=".venv/bin/python"
fi

$PYTHON_BIN -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing huggingface-hub..."
    $PYTHON_BIN -m pip install --quiet huggingface-hub
}

echo "Uploading..."
$PYTHON_BIN - <<EOF
import os
from huggingface_hub import HfApi

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = "$HF_REPO"
local_dir = "$MODEL_DIR"

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, token=token)
    print(f"  Repository ready: {repo_id}")
except Exception as e:
    print(f"  Warning: {e}")

api.upload_folder(
    folder_path=local_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message="$COMMIT_MSG",
    token=token,
    ignore_patterns=["*.py", "__pycache__/*"],
)
print(f"  ✅ Upload complete: https://huggingface.co/{repo_id}")
EOF
