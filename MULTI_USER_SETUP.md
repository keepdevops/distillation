# Multi-User Model Storage Setup

This setup allows all user profiles on the Mac to access and use distilled models from a shared location.

## Architecture

**Shared Storage:** `/Users/Shared/models`

All users can read/write models here. Benefits:
- No duplication of large model files
- Consistent model paths across users
- Works with `mlx_lm.generate`, Gradio, and all distillation scripts

## Quick Setup

### For Each User Profile

Run this once per user:

```bash
cd /Users/caribou/distill
./scripts/setup_shared_models.sh
```

Or manually add to `~/.zshrc`:

```bash
echo 'export MODEL_PATH=/Users/Shared/models' >> ~/.zshrc
source ~/.zshrc
```

### Verify Setup

```bash
echo $MODEL_PATH
# Should print: /Users/Shared/models

ls $MODEL_PATH
# Should show: distilled-minillm  hf_cache
```

## Directory Structure

```
/Users/Shared/models/
├── distilled-minillm/          # Trained models
│   ├── metrics.jsonl
│   └── trial_00/
├── hf_cache/                   # HuggingFace model cache
│   ├── models--bert-large-uncased/
│   ├── models--distilbert-base-uncased/
│   └── models--Qwen--Qwen2-0.5B-Instruct/
└── [future distilled models]
```

## Usage Examples

### MLX Generation

```bash
# Using MODEL_PATH variable
mlx_lm.generate \
  --model-dir $MODEL_PATH/hf_cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/* \
  --prompt "Hello, how are you?"

# Direct path
mlx_lm.generate \
  --model-dir /Users/Shared/models/hf_cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/* \
  --prompt "Explain quantum computing."
```

### Gradio UI

```bash
# Uses $MODEL_PATH/distilled-minillm by default
python scripts/eval_gradio.py

# Specify different model
python scripts/eval_gradio.py --model_path $MODEL_PATH/distilled-sft

# Use cached HF model
python scripts/eval_gradio.py \
  --model_path $MODEL_PATH/hf_cache/models--bert-large-uncased/snapshots/*
```

### Dashboard

```bash
# Scans MODEL_PATH for training runs
export MODEL_PATH=/Users/Shared/models
python scripts/dashboard.py --runs_dir $MODEL_PATH
```

### Distillation Training

All distillation scripts now check `MODEL_PATH`:

```bash
# Output goes to $MODEL_PATH/distilled-minillm by default
python scripts/run_distillation_agent.py --open

# Or specify output explicitly
python scripts/distill_minillm.py \
  --open \
  --output_dir $MODEL_PATH/my-custom-model
```

## Environment Variables

### Primary Variable

```bash
MODEL_PATH=/Users/Shared/models
```

Set in `~/.zshrc` for convenience. All scripts check this variable first.

### HuggingFace Cache

To use the shared HF cache:

```bash
export HF_HOME=$MODEL_PATH/hf_cache
```

Add to `~/.zshrc` if you want this persistent.

## Permissions

The shared directory has these permissions:

```bash
drwxr-xr-x  /Users/Shared/models  # 755: readable by all, writable by owner
```

All users in the `wheel` group can write to `/Users/Shared/`.

## Multi-User Workflow

### User 1: Train a Model

```bash
# User: alice
export MODEL_PATH=/Users/Shared/models
python scripts/distill_minillm.py \
  --open \
  --output_dir $MODEL_PATH/distilled-by-alice \
  --epochs 2
```

### User 2: Evaluate the Model

```bash
# User: bob
export MODEL_PATH=/Users/Shared/models
python scripts/eval_gradio.py \
  --model_path $MODEL_PATH/distilled-by-alice
```

### User 3: Export to GGUF

```bash
# User: charlie
export MODEL_PATH=/Users/Shared/models
python scripts/export_student_gguf.py \
  --model_path $MODEL_PATH/distilled-by-alice \
  --output_dir $MODEL_PATH/distilled-by-alice
```

Everyone can now use the GGUF file!

## Advanced Configuration

### Custom Model Paths in Scripts

The scripts automatically resolve model paths:

```python
from model_path_helper import resolve_model_path, get_model_base_path

# Resolves relative to MODEL_PATH if set
path = resolve_model_path("distilled-minillm")
# Returns: /Users/Shared/models/distilled-minillm

# Get base path
base = get_model_base_path()
# Returns: Path('/Users/Shared/models')
```

### List Available Models

```bash
python scripts/model_path_helper.py
```

Output:
```
Model Path Helper
============================================================
MODEL_PATH env: /Users/Shared/models
Base path: /Users/Shared/models
HF cache: /Users/Shared/models/hf_cache

Available models:
  distilled-minillm: /Users/Shared/models/distilled-minillm
  hf_cache: /Users/Shared/models/hf_cache
```

## Migration

### Moving Existing Models

If you have models in your home directory:

```bash
# Copy to shared location
cp -r ~/distill/distilled-minillm /Users/Shared/models/

# Or move (saves space)
mv ~/distill/distilled-minillm /Users/Shared/models/

# Create symlink for compatibility
ln -s /Users/Shared/models/distilled-minillm ~/distill/distilled-minillm
```

### Updating Cache Location

Point HuggingFace to shared cache:

```bash
# Add to ~/.zshrc
export HF_HOME=/Users/Shared/models/hf_cache
export HF_DATASETS_CACHE=/Users/Shared/models/hf_cache/datasets
```

## Troubleshooting

### "Permission denied" when writing

**Issue:** User cannot write to `/Users/Shared/models`

**Solution:**
```bash
chmod 777 /Users/Shared/models  # Allow all users to write
```

### "MODEL_PATH not set"

**Issue:** Variable not available in current shell

**Solution:**
```bash
# Reload shell config
source ~/.zshrc

# Or export manually
export MODEL_PATH=/Users/Shared/models
```

### Scripts still using old paths

**Issue:** Scripts look in wrong directory

**Solution:**
```bash
# Check if MODEL_PATH is set
echo $MODEL_PATH

# Explicitly pass path
python scripts/eval_gradio.py --model_path /Users/Shared/models/my-model
```

### MLX can't find model

**Issue:** `mlx_lm.generate` says model not found

**Solution:**
```bash
# Use full path with wildcard for snapshot
mlx_lm.generate \
  --model-dir /Users/Shared/models/hf_cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/* \
  --prompt "test"

# Or find the exact snapshot
ls /Users/Shared/models/hf_cache/models--Qwen--Qwen2-0.5B-Instruct/snapshots/
```

## Security Notes

- `/Users/Shared/` is designed for multi-user access on macOS
- All users on the system can read models
- Consider file permissions for sensitive models
- No network sharing by default (local only)

## Backup

Backup shared models:

```bash
# Time Machine automatically backs up /Users/Shared/

# Manual backup
rsync -av /Users/Shared/models/ /Volumes/Backup/models/

# Or use tar
tar -czf ~/models-backup-$(date +%Y%m%d).tar.gz -C /Users/Shared models
```

## Cleanup

Remove old models:

```bash
# List sizes
du -sh /Users/Shared/models/*

# Remove specific model
rm -rf /Users/Shared/models/old-distilled-model

# Clear HF cache (keeps only recent)
rm -rf /Users/Shared/models/hf_cache/hub/models--*/.locks
```

## Summary

✅ **Shared storage:** `/Users/Shared/models`
✅ **Environment variable:** `MODEL_PATH=/Users/Shared/models`
✅ **Works with:** MLX, PyTorch, GGUF, Gradio, Dashboard
✅ **Multi-user:** All profiles can read/write
✅ **Automatic detection:** Scripts check `MODEL_PATH` first

For questions or issues, see the main [USER_MANUAL.md](docs/USER_MANUAL.md).
