#!/usr/bin/env python3
"""Check coremltools ONNX support and test onnxsim for graph simplification."""

import coremltools as ct
import onnx
import onnxsim

print(f"coremltools: {ct.__version__}")
print(f"onnxsim: {onnxsim.__version__}")

# Check if onnx sub-module exists
try:
    from coremltools.converters import onnx as ct_onnx
    print("ct.converters.onnx: EXISTS")
except ImportError as e:
    print(f"ct.converters.onnx: MISSING ({e})")

# Check if 'onnx' is a valid source for ct.convert
import inspect
sig = inspect.signature(ct.convert)
print(f"ct.convert signature source param: {sig.parameters.get('source', 'N/A')}")

# Try onnxsim on the decoder to resolve dynamic Clip
FP32_DIR = "build/zipformer-vi-onnx-fp32"
INT8_DIR = "build/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"

print("\n=== Simplify decoder.onnx with onnxsim ===")
decoder = onnx.load(f"{FP32_DIR}/decoder-epoch-20-avg-10.onnx")
simplified_decoder, check = onnxsim.simplify(
    decoder,
    overwrite_input_shapes={'y': [1, 2]},
)
print(f"onnxsim decoder: ok={check}, nodes: {len(decoder.graph.node)} -> {len(simplified_decoder.graph.node)}")
import collections
ops = collections.Counter(n.op_type for n in simplified_decoder.graph.node)
print(f"Simplified ops: {dict(ops)}")
# Save simplified
onnx.save(simplified_decoder, "build/decoder_simplified.onnx")
print("Saved: build/decoder_simplified.onnx")

# Try onnx2torch on simplified decoder
print("\n=== onnx2torch on simplified decoder ===")
import onnx2torch
import torch
import numpy as np
try:
    torch_decoder = onnx2torch.convert("build/decoder_simplified.onnx")
    torch_decoder.eval()
    sample_y = torch.zeros(1, 2, dtype=torch.int64)
    with torch.no_grad():
        out = torch_decoder(sample_y)
    print(f"Forward pass OK: {out.shape}")
    traced = torch.jit.trace(torch_decoder, (sample_y,))
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='y', shape=(1, 2), dtype=np.int64)],
        minimum_deployment_target=ct.target.iOS17,
    )
    print("CoreML conversion: SUCCESS!")
    print(f"  spec inputs: {[i.name for i in mlmodel.get_spec().description.input]}")
    print(f"  spec outputs: {[o.name for o in mlmodel.get_spec().description.output]}")
    mlmodel.save("build/ZipformerDecoder.mlpackage")
    print("Saved: build/ZipformerDecoder.mlpackage")
except Exception as e:
    import traceback
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

# Try simplifying encoder
print("\n=== Simplify encoder fp32 with onnxsim ===")
try:
    encoder = onnx.load(f"{FP32_DIR}/encoder-epoch-20-avg-10.onnx")
    simplified_enc, check = onnxsim.simplify(
        encoder,
        overwrite_input_shapes={'x': [1, 300, 80], 'x_lens': [1]},
    )
    print(f"onnxsim encoder: ok={check}, nodes: {len(encoder.graph.node)} -> {len(simplified_enc.graph.node)}")
    ops_enc = collections.Counter(n.op_type for n in simplified_enc.graph.node)
    bad = {op: c for op, c in ops_enc.items() if op in {'If', 'Loop', 'Scan'}}
    print(f"Control flow ops remaining: {bad}")
    onnx.save(simplified_enc, "build/encoder_simplified.onnx")
    print("Saved: build/encoder_simplified.onnx")
except Exception as e:
    import traceback
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
