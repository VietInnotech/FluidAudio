# Python Environment Management with UV

This project uses **[UV](https://github.com/astral-sh/uv)** for fast, reliable Python environment management.

## Quick Start

### 1. Install UV (if not already installed)

```bash
# macOS
brew install uv

# Linux / Other
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Up Python Environment

The venv and dependencies are managed entirely by UV:

```bash
cd Tools

# Create/update venv and sync dependencies (one command)
uv sync
```

This will:
- Create `.venv/` directory with Python 3.11 (as specified in `.python-version`)
- Install all dependencies from `pyproject.toml`
- Create `.venv/pyvenv.cfg` and activate environment

### 3. Activate & Use

```bash
# Activate the venv
source .venv/bin/activate

# Or run commands directly without activating
.venv/bin/python convert_nemo_ctc_to_coreml.py
.venv/bin/pip list
```

## Project Structure

```
Tools/
├── .python-version              # UV pins Python 3.11
├── pyproject.toml              # Dependencies (uv reads this)
├── uv.lock                      # Lockfile (reproducible installs)
├── .venv/                       # Virtual environment (auto-created by uv)
├── convert_nemo_ctc_to_coreml.py
├── test_ctc_vietnamese.py
└── upload_ctc_to_hf.sh
```

## Common Commands

### Install/Update Packages

```bash
# Install new package and update lock
uv pip install package_name

# Sync all dependencies from pyproject.toml
uv sync

# Sync without installing the project itself
uv sync --no-install-project

# Update dependencies to latest compatible versions
uv sync --upgrade
```

### Run Python Scripts

```bash
# Using activated venv
source .venv/bin/activate
python convert_nemo_ctc_to_coreml.py

# Or directly (no activation needed)
.venv/bin/python convert_nemo_ctc_to_coreml.py
```

### Check Environment

```bash
# Show installed packages
.venv/bin/pip list

# Show venv info
.venv/bin/python --version
which python  # shows .venv path
```

## Dependencies

### Current (`pyproject.toml`)

```toml
[project]
name = "fluidaudio-model-conversion"
requires-python = "==3.11.*"
dependencies = [
    "torch>=2.3.0,<2.5",
    "torchaudio>=2.3.0,<2.5",
    "nemo_toolkit[asr]>=2.1.0",
    "coremltools>=8.0",
    "onnx>=1.16",
    "onnxruntime>=1.18",
    "numpy>=1.24,<2.0",
    "huggingface_hub>=0.23",
    "tqdm",
    "scikit-learn>=0.17,<=1.5.1",  # pinned for coremltools compatibility
]
```

### Pin Strategy

- Python: **exactly 3.11** (via `.python-version`)
- Key packages: **ranges** with upper bounds (e.g., `torch>=2.3.0,<2.5`)
- Conflicts: **explicit pins** (e.g., `scikit-learn<=1.5.1`)

### Lock File (`uv.lock`)

UV creates `uv.lock` containing all transitive dependencies with **exact versions**. This ensures reproducible builds across machines:

```bash
# Reproducible install (uses lock file)
uv sync

# Fresh resolve (ignores lock, may update versions)
uv sync --upgrade
```

## Adding New Dependencies

```bash
# Add a package (updates pyproject.toml + uv.lock)
uv add package_name

# Add with version constraint
uv add "package_name>=1.0,<2.0"

# Remove a package
uv remove package_name
```

## Virtual Environment Details

### Location

```
Tools/.venv/
├── bin/
│   ├── python           # Python executable
│   ├── pip              # Package manager
│   ├── python3.11       # Specific version
│   └── [installed scripts]
├── lib/python3.11/site-packages/
│   ├── torch/
│   ├── nemo/
│   ├── coremltools/
│   └── [all dependencies]
└── pyvenv.cfg
```

### Deletion & Recreation

```bash
# Remove venv (safe, auto-recreated by uv sync)
rm -rf .venv

# Recreate
uv sync
```

## Troubleshooting

### Issue: "No module named X"

```bash
# Check if module is installed
.venv/bin/pip list | grep module_name

# Reinstall from lock
uv sync

# Or fresh install
rm -rf .venv && uv sync
```

### Issue: Python version mismatch

```bash
# Check current Python version
.venv/bin/python --version

# Should be 3.11.x (check .python-version)
cat .python-version

# UV automatically uses correct version from .python-version
uv sync
```

### Issue: Dependency conflict

```bash
# Show dependency tree
uv pip show package_name

# Resolve conflicts by locking versions in pyproject.toml
# Then: uv sync --upgrade
```

## Benefits of UV

| Aspect | UV | Traditional venv + pip |
|--------|----|----|
| **Speed** | ~10–100x faster | Slow pip installs |
| **Locking** | Automatic `uv.lock` | Manual requirements.txt |
| **Python Management** | Built-in pyenv | Separate tool needed |
| **Consistency** | Lock file reproducible | Varies across machines |
| **Syntax** | Simpler CLI | More verbose |

## Tips

1. **Always commit `uv.lock` to git**
   ```bash
   git add uv.lock
   ```

2. **Use `uv sync` to sync team environment**
   ```bash
   # Same dependencies everywhere
   uv sync
   ```

3. **Only edit `pyproject.toml` directly** (not `uv.lock`)
   - Let `uv` manage the lock file

4. **For CI/CD**, use `uv sync` in workflows
   ```yaml
   - run: uv sync
   - run: .venv/bin/python script.py
   ```

## Related Documentation

- [UV Official Docs](https://github.com/astral-sh/uv)
- [CTC Model Conversion](./convert_nemo_ctc_to_coreml.py)
- [FluidAudio ASR](../Documentation/ASR/)

## Next Steps

After setting up the venv:

```bash
# 1. Verify setup
source .venv/bin/activate
python --version  # Should show 3.11.x

# 2. Run conversion
python convert_nemo_ctc_to_coreml.py

# 3. Test models
python test_ctc_vietnamese.py --audio ../Tests/weanxinviec.mp3

# 4. Upload to HuggingFace
bash upload_ctc_to_hf.sh
```
