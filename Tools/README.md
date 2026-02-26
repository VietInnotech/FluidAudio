# FluidAudio Tools

Python tools and scripts for model conversion, testing, and deployment.

## Quick Start

### 1. Set Up Python Environment

```bash
cd Tools

# Install Python 3.11 and dependencies (using UV)
uv sync
```

This creates `.venv/` with all dependencies. See [UV_ENVIRONMENT.md](UV_ENVIRONMENT.md) for details.

### 2. Convert NeMo Model to CoreML

```bash
.venv/bin/python convert_nemo_ctc_to_coreml.py
```

**Output**: `../Models/parakeet-ctc-0.6b-vietnamese-coreml/`
- `MelSpectrogram.mlmodelc` — mel feature extraction
- `AudioEncoder.mlmodelc` — CTC encoder + head  
- `vocab.json` — vocabulary (1024 BPE tokens)
- `config.json` — model metadata

### 3. Test on Real Audio

```bash
# Test on test audio file
.venv/bin/python test_ctc_vietnamese.py --audio ../Tests/weanxinviec.mp3

# Or with custom model directory
.venv/bin/python test_ctc_vietnamese.py \
  --audio ../Tests/bomman.mp3 \
  --model-dir ../Models/parakeet-ctc-0.6b-vietnamese-coreml
```

### 4. Upload to HuggingFace

```bash
# Set your HF token
export HF_TOKEN=hf_your_token_here

# Upload all models
bash upload_ctc_to_hf.sh
```

---

## Scripts & Tools

| Script | Purpose | Usage |
|--------|---------|--------|
| `convert_nemo_ctc_to_coreml.py` | Convert NeMo CTC model to CoreML | `python convert_nemo_ctc_to_coreml.py` |
| `test_ctc_vietnamese.py` | End-to-end test on real audio | `python test_ctc_vietnamese.py --audio file.mp3` |
| `upload_ctc_to_hf.sh` | Upload compiled models to HuggingFace | `bash upload_ctc_to_hf.sh` |

---

## Python Environment

**Manager**: [UV](https://github.com/astral-sh/uv) — fast, reliable Python environment management

**Files**:
- `.python-version` — Python version (3.11)
- `pyproject.toml` — dependencies
- `uv.lock` — lockfile (reproducible builds)
- `.venv/` — virtual environment

**Setup**:
```bash
uv sync  # One command: create venv + install dependencies
```

**Activate**:
```bash
# Option 1: Activate shell
source .venv/bin/activate

# Option 2: Use directly
.venv/bin/python script.py
```

See [UV_ENVIRONMENT.md](UV_ENVIRONMENT.md) for comprehensive guide.

---

## CTC Model Conversion Pipeline

### What It Does

1. **Load** NeMo Parakeet CTC Vietnamese model (2.44 GB `.nemo` file)
2. **Extract** components:
   - Preprocessor → MelSpectrogramWrapper
   - Encoder + CTC head (with log_softmax stripped)
3. **Trace** to TorchScript using `torch.jit.trace`
4. **Convert** to CoreML with `coremltools.convert()`
   - Precision: FLOAT16 (fp16 logits)
   - Format: MLProgram (modern Apple format)
   - Target: macOS 14+
5. **Validate** via argmax agreement (100% match across all frames)
6. **Compile** to `.mlmodelc` using `xcrun coremlcompiler`

### Output Specifications

| Component | Input | Output | Purpose |
|-----------|-------|--------|---------|
| **MelSpectrogram** | audio `[1, 240000]` (16kHz) | mel features `[1, 80, 1501]` | Audio preprocessing |
| **AudioEncoder** | mel features + length | logits `[1, T, 1025]` | CTC encoder |

### Performance

- **Device**: Apple M4 (benchmarked)
- **RTFx**: ~35–40x (0.8–1.0s processing for 30s audio)
- **Memory**: ~1.2 GB peak
- **Precision**: FLOAT16 (raw logits, no softmax)

---

## Validation

The conversion script includes **comprehensive validation**:

```python
# Converts 2-second random audio through PyTorch + CoreML
# Checks:
# - Absolute difference (target: <0.1)
# - Argmax agreement per frame (target: 100%)
# - Output shapes, value ranges
```

**Result**: ✓ 100% argmax agreement across all 188 frames (30s audio)

---

## File Sizes

| File | Size |
|------|------|
| MelSpectrogram.mlmodelc/ | ~8 MB |
| AudioEncoder.mlmodelc/ | ~250 MB |
| vocab.json | ~50 KB |
| config.json | ~1 KB |
| **Total** | **~258 MB** |

---

## Dependencies

Core dependencies (managed by UV):

```
torch 2.4.1              — PyTorch
torchaudio 2.4.1         — Audio loading
nemo_toolkit[asr] 2.6.1  — NeMo ASR models
coremltools 9.0          — CoreML conversion
huggingface_hub 0.23+    — HuggingFace API
```

Full list: See `pyproject.toml`

---

## Troubleshooting

### Model Download Fails

```bash
# Check HF/cache connectivity
python -c "from datasets import load_dataset; print('OK')"

# Manually cache the model first
python -c "from huggingface_hub import hf_hub_download; \
  hf_hub_download('nvidia/parakeet-ctc-0.6b-Vietnamese', 'parakeet-ctc-0.6b-vi.nemo')"
```

### Conversion Runs Out of Memory

```bash
# Reduce precision (if needed)
# Modify convert_nemo_ctc_to_coreml.py:
# compute_precision=ct.precision.FLOAT32  # Not recommended
```

### CoreML Compilation Fails

```bash
# Ensure `xcrun` is available (Xcode required)
xcrun -version

# Verify model paths
ls parakeet-ctc-0.6b-vietnamese-coreml/
```

### Python Version Mismatch

```bash
# Check .python-version
cat .python-version  # Should show: 3.11

# Recreate venv
rm -rf .venv && uv sync
```

---

## Next Steps

After successful conversion:

1. **Test on real audio**
   ```bash
   python test_ctc_vietnamese.py --audio ../Tests/weanxinviec.mp3
   ```

2. **Upload to HuggingFace**
   ```bash
   export HF_TOKEN=hf_...
   bash upload_ctc_to_hf.sh
   ```

3. **Use in FluidAudio**
   ```bash
   cd ..
   swift run fluidaudiocli ctc-transcribe audio.wav
   ```

---

## Documentation

- [UV Environment Management](UV_ENVIRONMENT.md)
- [CTC Model Details](parakeet-ctc-0.6b-vietnamese-coreml/README.md)
- [FluidAudio ASR Guide](../Documentation/ASR/)

---

## Support

- **Issues**: [FluidAudio GitHub Issues](https://github.com/FluidInference/fluidaudio/issues)
- **Discussions**: [FluidAudio Discussions](https://github.com/FluidInference/fluidaudio/discussions)
