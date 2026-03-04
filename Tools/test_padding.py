"""Test what padding the VibeVoice encoder applies for different input sizes."""
import torch
import copy

from vibevoice.modular.modeling_vibevoice_asr import VibeVoiceASRForConditionalGeneration

print("Loading model (cached)...")
model = VibeVoiceASRForConditionalGeneration.from_pretrained(
    "microsoft/VibeVoice-ASR",
    torch_dtype=torch.bfloat16,
    attn_implementation="eager",
    trust_remote_code=True,
)
acoustic_tok = copy.deepcopy(model.model.acoustic_tokenizer).float()
semantic_tok = copy.deepcopy(model.model.semantic_tokenizer).float()
del model

import torch.nn.functional as F

pad_calls = []
original_pad = F.pad


def capture_pad(input, pad, mode="constant", value=0):
    pad_calls.append({"shape": tuple(input.shape), "pad": pad, "mode": mode})
    return original_pad(input, pad, mode, value)


F.pad = capture_pad

SAMPLE_RATE = 24000

for n_sec in [10, 30, 60]:
    n_samp = n_sec * SAMPLE_RATE
    pad_calls.clear()
    audio = torch.randn(1, 1, n_samp)
    with torch.no_grad():
        out = acoustic_tok.encode(audio)
    print(f"\nAcoustic {n_sec}s ({n_samp} samples): output={tuple(out.mean.shape)}")
    if pad_calls:
        for c in pad_calls:
            print(f"  F.pad: shape={c['shape']}, pad={c['pad']}, mode={c['mode']}")
    else:
        print("  No F.pad calls!")

print("\n--- Semantic encoder ---")
for n_sec in [10, 30, 60]:
    n_samp = n_sec * SAMPLE_RATE
    pad_calls.clear()
    audio = torch.randn(1, 1, n_samp)
    with torch.no_grad():
        out = semantic_tok.encode(audio)
    print(f"\nSemantic {n_sec}s ({n_samp} samples): output={tuple(out.mean.shape)}")
    if pad_calls:
        for c in pad_calls:
            print(f"  F.pad: shape={c['shape']}, pad={c['pad']}, mode={c['mode']}")
    else:
        print("  No F.pad calls!")
