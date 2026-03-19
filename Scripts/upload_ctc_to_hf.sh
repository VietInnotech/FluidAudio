#!/usr/bin/env bash
# Upload CTC Vietnamese model to HuggingFace Hub

set -e

HF_REPO="leakless/parakeet-ctc-0.6b-Vietnamese-coreml"
MODEL_DIR="../Models/parakeet-ctc-0.6b-vietnamese-coreml"
COMMIT_MSG="Upload CTC Vietnamese model: raw logits, 100% argmax agreement"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: Model directory not found: $MODEL_DIR"
    echo "Run convert_nemo_ctc_to_coreml.py first to generate models (output goes to ../Models/)."
    exit 1
fi

echo "=========================================="
echo "Uploading to: $HF_REPO"
echo "Files to upload:"
echo "  ✓ README.md                 (model documentation)"
echo "  ✓ MelSpectrogram.mlmodelc/  (compiled CoreML)"
echo "  ✓ AudioEncoder.mlmodelc/    (compiled CoreML)"
echo "  ✓ vocab.json                (1024 BPE tokens)"
echo "  ✓ config.json               (model metadata)"
echo "=========================================="
echo ""

# Check if huggingface_hub is installed
.venv/bin/python -c "import huggingface_hub" 2>/dev/null || {
    echo "Installing huggingface-hub..."
    .venv/bin/python -m pip install huggingface-hub
}

# Create HF repo if needed (requires HF_TOKEN env var)
echo "[1/3] Ensuring HuggingFace repo exists: $HF_REPO"
.venv/bin/python << 'EOF'
import os
from huggingface_hub import create_repo, HfApi, logging

logging.set_verbosity_info()
repo_id = os.environ.get("HF_REPO", "leakless/parakeet-ctc-0.6b-Vietnamese-coreml")

try:
    create_repo(repo_id, exist_ok=True, private=False)
    print(f"✓ Repo ready: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"⚠ Repo check: {e}")
    print(f"⚠ Make sure HF_TOKEN env var is set: export HF_TOKEN=hf_...")
EOF

echo ""
echo "[2/3] Uploading model files..."
.venv/bin/python << 'PYTHON_EOF'
import os
from pathlib import Path
from huggingface_hub import HfApi

repo_id = "leakless/parakeet-ctc-0.6b-Vietnamese-coreml"
model_dir = Path("parakeet-ctc-0.6b-vietnamese-coreml")

api = HfApi()

# Files to upload
files_to_upload = [
    ("README.md", "README.md"),
    ("config.json", "config.json"),
    ("vocab.json", "vocab.json"),
    ("MelSpectrogram.mlmodelc", "MelSpectrogram.mlmodelc"),
    ("AudioEncoder.mlmodelc", "AudioEncoder.mlmodelc"),
]

print(f"Uploading to: {repo_id}")
for local_path, hf_path in files_to_upload:
    local_file = model_dir / local_path
    if not local_file.exists():
        print(f"  ⚠ Skipping (not found): {local_path}")
        continue
    
    if local_file.is_dir():
        print(f"  📁 Uploading folder: {local_path}/")
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_file),
            path_in_repo=hf_path,
            commit_message=f"Add {local_path}",
        )
    else:
        print(f"  📄 Uploading file: {local_path}")
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(local_file),
            path_in_repo=hf_path,
            commit_message=f"Add {local_path}",
        )
    print(f"     ✓ {local_path}")

print(f"\n✓ Upload complete: https://huggingface.co/{repo_id}")
PYTHON_EOF

echo ""
echo "[3/3] Verifying upload..."
.venv/bin/python << 'EOF'
from huggingface_hub import list_repo_files
repo_id = "leakless/parakeet-ctc-0.6b-Vietnamese-coreml"
try:
    files = list_repo_files(repo_id)
    required = ["README.md", "config.json", "vocab.json", "MelSpectrogram.mlmodelc", "AudioEncoder.mlmodelc"]
    found = [f for f in required if any(f in file for file in files)]
    print(f"Found {len(found)}/{len(required)} required files:")
    for f in required:
        has_it = any(f in file for file in files)
        print(f"  {'✓' if has_it else '✗'} {f}")
    print(f"\nRepo: https://huggingface.co/{repo_id}")
except Exception as e:
    print(f"Could not verify: {e}")
    print("Please check manually: https://huggingface.co/leakless/parakeet-ctc-0.6b-Vietnamese-coreml")
EOF

echo ""
echo "=========================================="
echo "✓ Done! Your model is now on HuggingFace"
echo "=========================================="
