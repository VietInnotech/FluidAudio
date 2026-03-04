#!/bin/bash
# VibeVoice-ASR CoreML upload to HuggingFace
# Usage: export HF_TOKEN="your_token" && bash upload_vibevoice_v2.sh

set -e

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
REPO_ID="leakless/vibevoice-asr-coreml"

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

echo "Repository: $REPO_ID"
echo "License: MIT"
echo ""

# Login to HuggingFace
echo "Authenticating with HuggingFace..."
echo "$HF_TOKEN" | uv run huggingface-cli login --token-type "user_access_token" 2>&1 | grep -v "WARNING\|will be ignored" || true
echo ""

# Create the repository if it doesn't exist
echo "Setting up repository..."
uv run huggingface-cli repo create "$REPO_ID" --type model 2>&1 | grep -v "WARNING\|already exists" || true
echo ""

# Upload files using huggingface-cli
echo "Uploading files to $REPO_ID..."
echo ""

cd "$OUTPUT_DIR"

# Upload individually for better control and progress visibility
echo "📤 Uploading vibevoice_acoustic_encoder.mlmodelc..."
uv run huggingface-cli upload "$REPO_ID" vibevoice_acoustic_encoder.mlmodelc vibevoice_acoustic_encoder.mlmodelc --repo-type model 2>&1 | tail -3

echo "📤 Uploading vibevoice_semantic_encoder.mlmodelc..."
uv run huggingface-cli upload "$REPO_ID" vibevoice_semantic_encoder.mlmodelc vibevoice_semantic_encoder.mlmodelc --repo-type model 2>&1 | tail -3

echo "📤 Uploading vibevoice_decoder_stateful.mlmodelc (~13 GB)..."
uv run huggingface-cli upload "$REPO_ID" vibevoice_decoder_stateful.mlmodelc vibevoice_decoder_stateful.mlmodelc --repo-type model 2>&1 | tail -3

echo "📤 Uploading vibevoice_embeddings.bin (~1 GB)..."
uv run huggingface-cli upload "$REPO_ID" vibevoice_embeddings.bin vibevoice_embeddings.bin --repo-type model 2>&1 | tail -3

echo "📤 Uploading vocab.json..."
uv run huggingface-cli upload "$REPO_ID" vocab.json vocab.json --repo-type model 2>&1 | tail -3

echo "📤 Uploading metadata.json..."
uv run huggingface-cli upload "$REPO_ID" metadata.json metadata.json --repo-type model 2>&1 | tail -3

cd - > /dev/null

echo ""
echo "======================================="
echo "✓ Upload Complete!"
echo "======================================="
echo ""
echo "Models available at:"
echo "  https://huggingface.co/$REPO_ID"
echo ""
