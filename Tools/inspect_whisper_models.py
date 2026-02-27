#!/usr/bin/env python3
"""Inspect WhisperKit CoreML model I/O signatures."""
import coremltools as ct
import json

MODEL_DIR = "Models/whisperkit-coreml/openai_whisper-large-v3_turbo"

def inspect_model(name):
    path = f"{MODEL_DIR}/{name}.mlmodelc"
    try:
        model = ct.models.MLModel(path)
    except Exception as e:
        print(f"  ERROR loading: {e}")
        return
    
    spec = model.get_spec()
    
    print(f"\n=== {name} ===")
    
    # Check if stateful
    if hasattr(spec, 'description') and spec.description.state:
        print("  States:")
        for state in spec.description.state:
            mtype = state.type.multiArrayType
            shape = [d.value for d in mtype.shape] if mtype.shape else "flexible"
            print(f"    {state.name}: shape={shape}")
    
    print("  Inputs:")
    for inp in spec.description.input:
        mtype = inp.type.multiArrayType
        if mtype.shape:
            shape = [d.value for d in mtype.shape]
        elif mtype.shapeRange and mtype.shapeRange.sizeRanges:
            shape = []
            for dim_range in mtype.shapeRange.sizeRanges:
                lo = dim_range.lowerBound
                hi = dim_range.upperBound
                if lo == hi:
                    shape.append(lo)
                else:
                    shape.append(f"{lo}-{hi}")
        else:
            shape = "unknown"
        dtype_map = {65536: "float16", 65568: "float32", 131072: "int32"}
        dtype = dtype_map.get(mtype.dataType, f"type_{mtype.dataType}")
        print(f"    {inp.name}: shape={shape} dtype={dtype}")
    
    print("  Outputs:")
    for out in spec.description.output:
        mtype = out.type.multiArrayType
        if mtype.shape:
            shape = [d.value for d in mtype.shape]
        elif mtype.shapeRange and mtype.shapeRange.sizeRanges:
            shape = []
            for dim_range in mtype.shapeRange.sizeRanges:
                lo = dim_range.lowerBound
                hi = dim_range.upperBound
                if lo == hi:
                    shape.append(lo)
                else:
                    shape.append(f"{lo}-{hi}")
        else:
            shape = "unknown"
        dtype_map = {65536: "float16", 65568: "float32", 131072: "int32"}
        dtype = dtype_map.get(mtype.dataType, f"type_{mtype.dataType}")
        print(f"    {out.name}: shape={shape} dtype={dtype}")

for name in ["MelSpectrogram", "AudioEncoder", "TextDecoder", "TextDecoderContextPrefill"]:
    inspect_model(name)
