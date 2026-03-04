# Air-Gapped Setup Guide

Complete guide for running distillation in air-gapped environments.

## Phase 1: Cache Everything (Run with Internet)

### Step 1: Cache Models (Required)

Cache the Qwen2 models (no HF login needed):

```bash
cd /Users/caribou/distill

# Cache to default HF location (~/.cache/huggingface/)
python scripts/cache_models.py --open

# OR cache to custom location for transfer
python scripts/cache_models.py --open --output ./hf_cache
```

This will download:
- `Qwen/Qwen2-0.5B-Instruct` (student, ~1GB)
- `Qwen/Qwen2-1.5B-Instruct` (teacher, ~3GB)
- `distilbert-base-uncased` (for benchmarks)
- `bert-large-uncased` (for benchmarks)

**Time: ~5-10 minutes** (depending on connection speed)

### Step 2: Cache Datasets (Required)

```bash
# Cache to default location
python scripts/cache_datasets.py

# OR cache to custom location with disk copies
python scripts/cache_datasets.py --output ./datasets_cache --disk
```

This will download:
- `tatsu-lab/alpaca` (~50K examples, ~52MB)
- `glue:sst2` (for benchmarks)

**Time: ~2-3 minutes**

### Step 3: Verify Cache

```bash
# Check models
ls -lh ~/.cache/huggingface/hub/models--Qwen--Qwen2-*/

# Check datasets
ls -lh ~/.cache/huggingface/datasets/tatsu-lab___alpaca/
```

---

## Phase 2: Air-Gapped Operation

### Environment Variables

Set these for guaranteed offline operation:

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Or add to `~/.zshrc` or `~/.bashrc`:

```bash
echo 'export HF_HUB_OFFLINE=1' >> ~/.zshrc
echo 'export HF_DATASETS_OFFLINE=1' >> ~/.zshrc
echo 'export TRANSFORMERS_OFFLINE=1' >> ~/.zshrc
source ~/.zshrc
```

### Run Distillation Offline

**Always use `--offline` flag:**

```bash
python scripts/run_distillation_agent.py \
    --open \
    --offline \
    --n_trials 3 \
    --epochs 2 \
    --export gguf \
    --log_experiment
```

**All supported offline commands:**

```bash
# Minimal run
python scripts/run_distillation_agent.py \
    --open --offline \
    --epochs 2 \
    --export gguf

# Full featured
python scripts/run_distillation_agent.py \
    --open --offline \
    --epochs 2 \
    --max_samples 2000 \
    --export gguf \
    --compare_teacher \
    --benchmarks \
    --log_experiment

# With config file
python scripts/run_distillation_agent.py \
    --config configs/agent_config.json \
    --offline
```

### Update Config for Offline

Edit `configs/agent_config.json`:

```json
{
  "output_dir": "./distilled-minillm",
  "open": true,
  "offline": true,    ← Add this
  "watchdog": false,
  "backend": "pytorch",
  "export": "gguf",
  "epochs": 2,
  "max_samples": 2000
}
```

---

## Phase 3: Transfer to Air-Gapped Machine (Optional)

If caching on a different machine:

### On Internet-Connected Machine

```bash
# Cache to portable directory
python scripts/cache_models.py --open --output ./airgap_transfer/hf_cache
python scripts/cache_datasets.py --output ./airgap_transfer/datasets_cache --disk

# Compress for transfer
cd airgap_transfer
tar -czf hf_cache.tar.gz hf_cache/
tar -czf datasets_cache.tar.gz datasets_cache/

# Transfer these files to air-gapped machine
```

### On Air-Gapped Machine

```bash
# Extract caches
tar -xzf hf_cache.tar.gz
tar -xzf datasets_cache.tar.gz

# Point to local cache
export HF_HOME=/path/to/airgap_transfer/hf_cache
export HF_DATASETS_CACHE=/path/to/airgap_transfer/datasets_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Run distillation
python scripts/run_distillation_agent.py --open --offline --epochs 2 --export gguf
```

---

## Troubleshooting

### "Could not find file" or Network Errors

**Problem:** Script tries to download despite `--offline`

**Solutions:**
1. Verify cache exists: `ls ~/.cache/huggingface/hub/`
2. Set environment variables: `export HF_HUB_OFFLINE=1`
3. Check you used `--open` flag (for Qwen models)
4. Re-run cache scripts

### "Repository not found" with `--offline`

**Problem:** Model not in cache

**Solution:**
```bash
# Cache missing model
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
model_id = 'Qwen/Qwen2-0.5B-Instruct'
AutoTokenizer.from_pretrained(model_id)
AutoModelForCausalLM.from_pretrained(model_id)
"
```

### Benchmark Failures in Offline Mode

**Problem:** WikiText-2 not cached

**Solution:**
```bash
# Cache WikiText-2 for benchmarks
python -c "
from datasets import load_dataset
load_dataset('wikitext', 'wikitext-2-raw-v1')
"
```

### Cache Size Verification

```bash
# Check HF cache size
du -sh ~/.cache/huggingface/

# Expected sizes:
# Models: ~5-8 GB (Qwen2 + BERT models)
# Datasets: ~100-200 MB (alpaca + glue)
# Total: ~5-10 GB
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| Cache models | `python scripts/cache_models.py --open` |
| Cache datasets | `python scripts/cache_datasets.py` |
| Run offline | Add `--offline` flag to ALL commands |
| Set env vars | `export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1` |
| Check cache | `ls ~/.cache/huggingface/hub/` |
| Cache size | `du -sh ~/.cache/huggingface/` |

---

## Verification Checklist

Before going fully air-gapped, verify:

- [ ] Models cached: `ls ~/.cache/huggingface/hub/ | grep Qwen`
- [ ] Datasets cached: `ls ~/.cache/huggingface/datasets/ | grep alpaca`
- [ ] Test run works: `python scripts/run_distillation_agent.py --open --offline --epochs 1 --max_samples 10 --skip_eval --export none`
- [ ] No network warnings in logs
- [ ] `--offline` flag added to all commands
- [ ] Environment variables set (optional but recommended)

Once verified, you can disconnect from the network! 🔒
