#!/usr/bin/env python3
"""Debug: test tracing and CoreML conversion of a single decoder layer."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import coremltools as ct
from transformers import Qwen3Config, Qwen3Model

HIDDEN_SIZE = 1024  # Test with 0.6B dimensions first
NUM_Q_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
GQA_REPEAT = NUM_Q_HEADS // NUM_KV_HEADS

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def repeat_kv(hidden_states, n_rep):
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class MiniStatefulDecoder(nn.Module):
    """Test: 1-layer stateful decoder."""
    def __init__(self, layers, final_norm, lm_head, max_seq_len=64):
        super().__init__()
        self.layers = layers
        self.final_norm = final_norm
        self.lm_head = lm_head
        self.scale = 1.0 / math.sqrt(HEAD_DIM)
        self.register_buffer("k_cache_0",
            torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16))
        self.register_buffer("v_cache_0",
            torch.zeros(1, NUM_KV_HEADS, max_seq_len, HEAD_DIM, dtype=torch.float16))

    def forward(self, hidden_states, position_cos, position_sin, attention_mask):
        q_len = hidden_states.shape[1]
        end_step = attention_mask.shape[-1]
        past_kv_len = end_step - q_len
        
        cos = position_cos.unsqueeze(1)
        sin = position_sin.unsqueeze(1)
        
        layer = self.layers[0]
        k_cache = self.k_cache_0
        v_cache = self.v_cache_0
        
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        
        attn = layer.self_attn
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)
        
        q = q.view(1, q_len, NUM_Q_HEADS, HEAD_DIM).transpose(1, 2)
        k = k.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = v.view(1, q_len, NUM_KV_HEADS, HEAD_DIM).transpose(1, 2)
        
        if hasattr(attn, "q_norm"):
            q = attn.q_norm(q)
            k = attn.k_norm(k)
        
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        
        k_cache[:, :, past_kv_len:end_step, :] = k.half()
        v_cache[:, :, past_kv_len:end_step, :] = v.half()
        
        k_full = k_cache[:, :, :end_step, :].float()
        v_full = v_cache[:, :, :end_step, :].float()
        
        k_full = repeat_kv(k_full, GQA_REPEAT)
        v_full = repeat_kv(v_full, GQA_REPEAT)
        
        attn_weights = torch.matmul(q, k_full.transpose(2, 3)) * self.scale
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v_full)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(1, q_len, NUM_Q_HEADS * HEAD_DIM)
        hidden_states = attn.o_proj(attn_output)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        mlp = layer.mlp
        gate = mlp.gate_proj(hidden_states)
        up = mlp.up_proj(hidden_states)
        hidden_states = mlp.down_proj(F.silu(gate) * up)
        hidden_states = residual + hidden_states
        
        last_hidden = hidden_states[:, -1:, :]
        last_hidden = self.final_norm(last_hidden)
        logits = self.lm_head(last_hidden)
        return logits

# Create model
VOCAB_SIZE = 1000  # Small vocab for testing

print("Creating 1-layer Qwen3 model (small vocab for speed)...")
tc = Qwen3Config(
    hidden_size=HIDDEN_SIZE,
    intermediate_size=3072,  # 0.6B size
    num_hidden_layers=1,
    num_attention_heads=NUM_Q_HEADS,
    num_key_value_heads=NUM_KV_HEADS,
    head_dim=HEAD_DIM,
    vocab_size=VOCAB_SIZE,
    max_position_embeddings=65536,
    rms_norm_eps=1e-6,
    rope_theta=1000000,
    hidden_act="silu",
    attention_bias=False,
    tie_word_embeddings=True,
)
text_model = Qwen3Model(tc)
text_model.eval()

lm_head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
lm_head.weight = nn.Parameter(text_model.embed_tokens.weight.data.clone())
lm_head.eval()

print(f"  Layer type: {type(text_model.layers[0]).__name__}")
print(f"  Attn type: {type(text_model.layers[0].self_attn).__name__}")
print(f"  LayerNorm type: {type(text_model.layers[0].input_layernorm).__name__}")
print(f"  Has q_norm: {hasattr(text_model.layers[0].self_attn, 'q_norm')}")
if hasattr(text_model.layers[0].self_attn, 'q_norm'):
    print(f"  q_norm type: {type(text_model.layers[0].self_attn.q_norm).__name__}")

MAX_SEQ = 64
wrapper = MiniStatefulDecoder(text_model.layers, text_model.norm, lm_head, max_seq_len=MAX_SEQ)
wrapper.eval()

# Trace
hidden = torch.randn(1, 1, HIDDEN_SIZE)
cos_in = torch.randn(1, 1, HEAD_DIM)
sin_in = torch.randn(1, 1, HEAD_DIM)
mask = torch.zeros(1, 1, 1, 5)

print("\nTracing...")
with torch.no_grad():
    ref_out = wrapper(hidden, cos_in, sin_in, mask)
    print(f"  Eager output: {ref_out.shape}")
    traced = torch.jit.trace(wrapper, (hidden, cos_in, sin_in, mask))
traced.eval()

# Print a snippet of the trace graph to find the problematic op
graph_str = str(traced.graph)
lines = graph_str.split('\n')
print(f"\nTrace graph: {len(lines)} lines")
# Find 'aten::Int' ops
for i, line in enumerate(lines):
    if 'Int' in line:
        print(f"  Line {i}: {line.strip()[:120]}")

# Try converting
print("\nConverting to CoreML...")
q_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ, default=1)
e_dim = ct.RangeDim(lower_bound=1, upper_bound=MAX_SEQ, default=1)

inputs = [
    ct.TensorType("hidden_states", shape=(1, q_dim, HIDDEN_SIZE), dtype=np.float32),
    ct.TensorType("position_cos", shape=(1, q_dim, HEAD_DIM), dtype=np.float32),
    ct.TensorType("position_sin", shape=(1, q_dim, HEAD_DIM), dtype=np.float32),
    ct.TensorType("attention_mask", shape=(1, 1, q_dim, e_dim), dtype=np.float32),
]
outputs = [ct.TensorType("logits", dtype=np.float32)]
states = [
    ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float16), name="k_cache_0"),
    ct.StateType(wrapped_type=ct.TensorType(shape=(1, NUM_KV_HEADS, MAX_SEQ, HEAD_DIM), dtype=np.float16), name="v_cache_0"),
]

try:
    mlmodel = ct.convert(
        traced, inputs=inputs, outputs=outputs, states=states,
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_GPU,
    )
    print("  SUCCESS!")
    state = mlmodel.make_state()
    out = mlmodel.predict({
        "hidden_states": np.random.randn(1, 1, HIDDEN_SIZE).astype(np.float32),
        "position_cos": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
        "position_sin": np.random.randn(1, 1, HEAD_DIM).astype(np.float32),
        "attention_mask": np.zeros((1, 1, 1, 1), dtype=np.float32),
    }, state=state)
    print(f"  Output: {out['logits'].shape}")
except Exception as e:
    print(f"  FAILED: {e}")
    # Try to show more context
    import traceback
    traceback.print_exc()
