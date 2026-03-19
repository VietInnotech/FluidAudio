#!/usr/bin/env python3
"""Check Qwen3-ASR-1.7B model configuration by loading config.json directly."""
import json
from huggingface_hub import hf_hub_download

# Download config.json directly
config_path = hf_hub_download("Qwen/Qwen3-ASR-1.7B", "config.json")
with open(config_path) as f:
    config = json.load(f)

tc = config["thinker_config"]
ac = tc["audio_config"]
dc = tc["text_config"]

print("=== Audio Encoder ===")
print(f"d_model: {ac['d_model']}")
print(f"encoder_layers: {ac['encoder_layers']}")
print(f"encoder_attention_heads: {ac['encoder_attention_heads']}")
print(f"encoder_ffn_dim: {ac['encoder_ffn_dim']}")
print(f"output_dim: {ac['output_dim']}")
print(f"n_window: {ac['n_window']}")
print(f"n_window_infer: {ac['n_window_infer']}")
print(f"num_mel_bins: {ac['num_mel_bins']}")
print(f"conv_chunksize: {ac['conv_chunksize']}")
print(f"downsample_hidden_size: {ac['downsample_hidden_size']}")
print()
print("=== Text Decoder ===")
print(f"hidden_size: {dc['hidden_size']}")
print(f"intermediate_size: {dc['intermediate_size']}")
print(f"num_hidden_layers: {dc['num_hidden_layers']}")
print(f"num_attention_heads: {dc['num_attention_heads']}")
print(f"num_key_value_heads: {dc['num_key_value_heads']}")
print(f"head_dim: {dc['head_dim']}")
print(f"vocab_size: {dc['vocab_size']}")
print(f"rope_theta: {dc['rope_theta']}")
print(f"rms_norm_eps: {dc['rms_norm_eps']}")
print()
print("=== Special Tokens ===")
print(f"audio_start_token_id: {tc['audio_start_token_id']}")
print(f"audio_end_token_id: {tc['audio_end_token_id']}")
print(f"audio_token_id: {tc['audio_token_id']}")
