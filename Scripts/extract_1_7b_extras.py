"""Extract embeddings binary and vocab.json for Qwen3-ASR-1.7B."""
import json
import struct
import shutil
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

model_id = "Qwen/Qwen3-ASR-1.7B"
output_dir = Path("/Users/vit/offasr/FluidAudio/Tools/build/qwen3-asr-1.7b")

# Load index and weights
index_path = hf_hub_download(model_id, "model.safetensors.index.json")
with open(index_path) as f:
    index = json.load(f)

# Find which shard has embed_tokens
embed_shard = index["weight_map"]["thinker.model.embed_tokens.weight"]
shard_path = hf_hub_download(model_id, embed_shard)
weights = load_file(shard_path)
embed_weight = weights["thinker.model.embed_tokens.weight"]
print(f"embed_tokens shape: {embed_weight.shape}")

# Save as binary: uint32 vocab, uint32 hidden, float16 data
vocab_size, hidden_size = embed_weight.shape
embed_fp16 = embed_weight.half().numpy()
out_path = output_dir / "qwen3_asr_embeddings.bin"
with open(out_path, "wb") as f:
    f.write(struct.pack("<II", vocab_size, hidden_size))
    f.write(embed_fp16.tobytes())
print(f"Saved: {out_path} ({out_path.stat().st_size:,} bytes)")

# Copy vocab.json
vocab_path = hf_hub_download(model_id, "vocab.json")
dest = output_dir / "vocab.json"
shutil.copy2(vocab_path, dest)
print(f"Copied vocab.json ({dest.stat().st_size:,} bytes)")

# Write metadata.json
metadata = {
    "model_id": model_id,
    "architecture": "Qwen3ASRForConditionalGeneration",
    "sample_rate": 16000,
    "num_mel_bins": 128,
    "max_audio_seconds": 30.0,
    "max_seq_length": 4096,
    "audio_encoder": {
        "n_window": 50,
        "n_window_infer": 800,
        "mel_window_size": 100,
        "conv_downsample_factor": 8,
        "d_model": 1024,
        "output_dim": 2048,
        "num_layers": 24,
        "num_heads": 16,
    },
    "text_decoder": {
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_layers": 28,
        "num_attention_heads": 16,
        "num_kv_heads": 8,
        "head_dim": 128,
        "vocab_size": 151936,
        "rope_theta": 1000000,
        "mrope_section": [24, 20, 20],
    },
    "special_tokens": {
        "audio_start_token_id": 151669,
        "audio_end_token_id": 151670,
        "audio_token_id": 151676,
        "im_start_token_id": 151644,
        "im_end_token_id": 151645,
        "system_token_id": 8948,
        "user_token_id": 872,
        "assistant_token_id": 77091,
        "newline_token_id": 198,
        "eos_token_ids": [151645, 151643],
    },
    "components": {
        "audio_encoder": {
            "path": "qwen3_asr_audio_encoder.mlpackage",
            "precision": "float16",
        },
        "decoder_stateful": {
            "path": "qwen3_asr_decoder_stateful.mlpackage",
            "precision": "float16",
            "note": "Fused decoder with lmHead, stateful KV cache",
        },
    },
    "export_settings": {
        "compute_units": "CPU_AND_GPU",
        "deployment_target": "macOS15",
    },
}
meta_path = output_dir / "metadata.json"
meta_path.write_text(json.dumps(metadata, indent=2))
print(f"Wrote metadata.json")
