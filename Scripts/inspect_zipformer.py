#!/usr/bin/env python3
"""Inspect ZipFormer ONNX models and test CoreML conversion feasibility."""

import coremltools as ct
import onnx
import numpy as np

print(f"coremltools version: {ct.__version__}")

MODEL_DIR = "build/sherpa-onnx-zipformer-vi-30M-int8-2026-02-09"

# Test decoder conversion using modern ct.convert() API
print("\n=== Testing decoder.onnx -> CoreML ===")
try:
    mlmodel = ct.convert(
        f"{MODEL_DIR}/decoder.onnx",
        inputs=[ct.TensorType(name='y', shape=(1, 2), dtype=np.int64)],
        minimum_deployment_target=ct.target.iOS17,
    )
    print("Decoder conversion: SUCCESS")
    spec = mlmodel.get_spec()
    print(f"  Inputs: {[i.name for i in spec.description.input]}")
    print(f"  Outputs: {[o.name for o in spec.description.output]}")
except Exception as e:
    print(f"Decoder conversion: FAILED - {type(e).__name__}: {e}")

# Test joiner conversion
print("\n=== Testing joiner.int8.onnx -> CoreML ===")
try:
    mlmodel = ct.convert(
        f"{MODEL_DIR}/joiner.int8.onnx",
        inputs=[
            ct.TensorType(name='encoder_out', shape=(1, 512), dtype=np.float32),
            ct.TensorType(name='decoder_out', shape=(1, 512), dtype=np.float32),
        ],
        minimum_deployment_target=ct.target.iOS17,
    )
    print("Joiner conversion: SUCCESS")
    spec = mlmodel.get_spec()
    print(f"  Inputs: {[i.name for i in spec.description.input]}")
    print(f"  Outputs: {[o.name for o in spec.description.output]}")
except Exception as e:
    print(f"Joiner conversion: FAILED - {type(e).__name__}: {e}")

# Check what ops in encoder are NOT supported
print("\n=== Checking encoder.int8.onnx op compatibility ===")
m = onnx.load(f"{MODEL_DIR}/encoder.int8.onnx")
import collections
ops = collections.Counter(n.op_type for n in m.graph.node)
print("All ops used:")
for op, count in ops.most_common():
    print(f"  {op}: {count}")

# coremltools supported ONNX opset 13 ops (approximate list of problematic ones)
KNOWN_UNSUPPORTED = {"MatMulInteger", "DynamicQuantizeLinear", "QLinearMatMul", "QLinearConv"}
unsupported = {op: count for op, count in ops.items() if op in KNOWN_UNSUPPORTED}
print(f"\nKnown unsupported ops: {unsupported}")
