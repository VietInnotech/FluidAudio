#!/bin/bash
# Quick setup for VibeVoice-ASR CoreML upload
# Just set HF_TOKEN and run this in the FluidAudio root directory

echo "======================================="
echo "VibeVoice-ASR CoreML Upload"
echo "======================================="
echo ""

if [ -z "$HF_TOKEN" ]; then
    echo "❌ ERROR: HF_TOKEN environment variable not set"
    echo ""
    echo "To get your token:"
    echo "  1. Visit: https://huggingface.co/settings/tokens"
    echo "  2. Create a new token with WRITE access"
    echo "  3. Copy the token"
    echo ""
    echo "Then set it:"
    echo "  export HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'"
    echo ""
    exit 1
fi

echo "✓ HF_TOKEN is set"
echo ""

OUTPUT_DIR="./build/vibevoice-asr-coreml/f32"

if [ ! -d "$OUTPUT_DIR" ]; then
    echo "❌ ERROR: Models not found at $OUTPUT_DIR"
    echo "Run conversion first: cd Tools && uv run convert_vibevoice_asr_to_coreml.py"
    exit 1
fi

echo "✓ Found models at $OUTPUT_DIR"
echo ""

# Check files
echo "Files to upload:"
du -sh "$OUTPUT_DIR"/*.mlmodelc "$OUTPUT_DIR"/*.bin "$OUTPUT_DIR"/*.json 2>/dev/null | sed 's/^/  /'
TOTAL=$(du -sh "$OUTPUT_DIR" | cut -f1)
echo "  Total: $TOTAL"
echo ""

echo "Starting upload..."
echo ""

cd "$OUTPUT_DIR"

# Use uv to run the Python upload script with huggingface_hub
uv run --with huggingface_hub python << 'PYTHON_EOF'
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, create_repo

token = os.environ.get("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not set")
    sys.exit(1)

api = HfApi()
repo_id = "FluidInference/vibevoice-asr-coreml"

print(f"Repository: {repo_id}")
print(f"Author: FluidInference")
print(f"License: MIT")
print("")

# Create repo (idempotent)
try:
    create_repo(repo_id, repo_type="model", token=token, private=False)
    print(f"✓ Created repository {repo_id}")
except Exception as e:
    if "exists" in str(e) or "already exists" in str(e):
        print(f"✓ Repository {repo_id} already exists")
    else:
        print(f"⚠ {e}")

print("")
print("Uploading files...")
print("")

# Upload all files from f32/
cwd = Path.cwd()
files = list(cwd.glob("**/*"))
files = [f for f in files if f.is_file()]

for i, file_path in enumerate(files, 1):
    rel_path = file_path.relative_to(cwd)
    size_mb = file_path.stat().st_size / 1e6
    
    # Skip large files > 4GB (GitHub limitation)
    if file_path.stat().st_size > 4.5e9:
        print(f"  [{i}/{len(files)}] ⚠ SKIPPED {rel_path} ({size_mb:.0f} MB) - too large for single upload")
        print(f"              (Consider splitting: split -b 4G filename parts)")
        continue
    
    print(f"  [{i}/{len(files)}] 🔄 {rel_path} ({size_mb:.1f} MB)...", end=" ", flush=True)
    try:
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=f"f32/{rel_path}",
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
print("The models will auto-download on first use:")
print("  let models = try await VibeVoiceAsrModels.download(variant: .f32)")
print("")

PYTHON_EOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Upload failed"
    exit 1
fi

echo "🎉 Done!"
