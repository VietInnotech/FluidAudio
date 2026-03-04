#!/bin/bash
# Upload VibeVoice-ASR CoreML models to HuggingFace
# Usage: export HF_TOKEN="your_token_here" && bash upload_vibevoice.sh

set -e

if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN not set. Usage: export HF_TOKEN='your_token' && bash $0"
    exit 1
fi

OUTPUT_DIR="/Users/vit/offasr/FluidAudio/build/vibevoice-asr-coreml/f32"
REPO_NAME="vibevoice-asr-coreml"
ORG="FluidInference"
REPO_ID="$ORG/$REPO_NAME"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "ERROR: Models not found at $OUTPUT_DIR"
    exit 1
fi

echo "========================================="
echo "VibeVoice-ASR CoreML Upload"
echo "========================================="
echo "Repository: $REPO_ID"
echo "Models:"
ls -lh "$OUTPUT_DIR" | grep -E "\.mlmodelc|\.bin|\.json"
echo ""

# Create the repo if it doesn't exist (this command is idempotent in huggingface-cli)
echo "Checking/creating repository..."
cd "$OUTPUT_DIR"

# Upload using huggingface_hub
echo "Uploading models..."
uv run python << 'EOF'
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
token = os.environ["HF_TOKEN"]
repo_id = "FluidInference/vibevoice-asr-coreml"

# Try to create the repo (will error if exists, which is fine)
try:
    create_repo(repo_id, repo_type="model", token=token, private=False)
    print(f"✓ Created repository {repo_id}")
except:
    print(f"✓ Repository {repo_id} already exists")

# Upload the f32 folder
import os
output_dir = "/Users/vit/offasr/FluidAudio/build/vibevoice-asr-coreml/f32"

# Get all files to upload
files_to_upload = []
for root, dirs, files in os.walk(output_dir):
    for file in files:
        file_path = os.path.join(root, file)
        rel_path = os.path.relpath(file_path, output_dir)
        files_to_upload.append((file_path, f"f32/{rel_path}"))

print(f"\nUploading {len(files_to_upload)} files...")
for i, (local_path, remote_path) in enumerate(files_to_upload, 1):
    size_mb = os.path.getsize(local_path) / 1e6
    print(f"  [{i}/{len(files_to_upload)}] {remote_path} ({size_mb:.1f} MB)")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type="model",
        token=token,
    )

print(f"\n✓ Upload complete!")
print(f"✓ Models available at: https://huggingface.co/{repo_id}")
EOF

echo ""
echo "========================================="
echo "✓ Upload successful!"
echo "========================================="
echo ""
echo "Models are now available at:"
echo "  https://huggingface.co/FluidInference/vibevoice-asr-coreml"
echo ""
echo "To use in FluidAudio, models will auto-download on first transcription."
