#!/bin/bash
# VibeVoice-ASR CoreML upload to HuggingFace using uv environment
# Usage: cd /Users/vit/offasr/FluidAudio && export HF_TOKEN="your_token" && bash Tools/upload_final.sh

set -e

echo "======================================="
echo "VibeVoice-ASR CoreML Upload"
echo "======================================="
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable not set"
    echo ""
    echo "Set your token:"
    echo "  export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'"
    echo ""
    exit 1
fi

echo "✓ HF_TOKEN is set"
echo ""

OUTPUT_DIR="./build/vibevoice-asr-coreml/f32"
REPO_ID="leakless/vibevoice-asr-coreml"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Models not found at $OUTPUT_DIR"
    exit 1
fi

echo "✓ Found models at $OUTPUT_DIR"
echo ""
echo "Repository: $REPO_ID"
echo ""
echo "Files to upload:"
ls -lh "$OUTPUT_DIR"/*.mlmodelc "$OUTPUT_DIR"/*.bin "$OUTPUT_DIR"/*.json 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""

# Use uv run to ensure we're in the right environment
cd /Users/vit/offasr/FluidAudio

echo "Installing huggingface-hub in uv environment..."
cd Tools
uv pip install huggingface-hub 2>&1 | tail -3
cd ..

echo ""
echo "Starting upload..."
echo ""

# Run upload via uv environment
cd Tools
uv run python << 'UPLOAD_SCRIPT'
import os
import sys
from pathlib import Path

os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")
if not os.environ["HF_TOKEN"]:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

from huggingface_hub import HfApi, create_repo

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = "leakless/vibevoice-asr-coreml"

print(f"Repository: {repo_id}")
print("")

# Create repo (idempotent)
try:
    create_repo(repo_id, repo_type="model", token=token, private=False, exist_ok=True)
    print(f"✓ Repository ready: {repo_id}")
except Exception as e:
    print(f"⚠ {e}")

print("")
print("Uploading files...")
print("")

# Base directory
base_dir = Path("../build/vibevoice-asr-coreml/f32")

# List of files to upload
files_to_upload = [
    ("vibevoice_acoustic_encoder.mlmodelc", "vibevoice_acoustic_encoder.mlmodelc"),
    ("vibevoice_semantic_encoder.mlmodelc", "vibevoice_semantic_encoder.mlmodelc"),
    ("vibevoice_decoder_stateful.mlmodelc", "vibevoice_decoder_stateful.mlmodelc"),
    ("vibevoice_embeddings.bin", "vibevoice_embeddings.bin"),
    ("vocab.json", "vocab.json"),
    ("metadata.json", "metadata.json"),
]

for i, (local_name, remote_name) in enumerate(files_to_upload, 1):
    local_path = base_dir / local_name
    
    if not local_path.exists():
        print(f"  [{i}/{len(files_to_upload)}] ⚠ SKIP {local_name} (not found)")
        continue
    
    size_gb = local_path.stat().st_size / 1e9
    print(f"  [{i}/{len(files_to_upload)}] 📤 {local_name} ({size_gb:.2f} GB)...", end=" ", flush=True)
    
    try:
        # For directories (.mlmodelc), upload recursively
        if local_path.is_dir():
            api.upload_folder(
                folder_path=str(local_path),
                repo_id=repo_id,
                repo_type="model",
                path_in_repo=remote_name,
                token=token,
            )
        else:
            # For files, upload directly
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=repo_id,
                repo_type="model",
                token=token,
            )
        
        print("✓")
    except Exception as e:
        print(f"✗ {e}")
        sys.exit(1)

print("")
print("======================================")
print("✓ Upload Complete!")
print("======================================")
print("")
print(f"Models available at:")
print(f"  https://huggingface.co/{repo_id}")
print("")

UPLOAD_SCRIPT

echo "🎉 Done!"
