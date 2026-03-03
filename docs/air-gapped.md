# Air-Gapped Setup Guide
## Knowledge Distillation on Apple M3

This guide covers preparing and deploying the distillation pipeline in an **air-gapped** (offline) environment. No internet is used on the target M3 machine after setup.

---

## Overview

```
Online Staging Machine          Physical Transfer          Air-Gapped M3 Target
┌─────────────────────┐        ┌──────────────┐           ┌─────────────────────┐
│ • Miniforge install │   →    │ USB / SSD    │    →      │ • Unpack env         │
│ • conda env create  │        │ SHA-256 sum  │           │ • Set HF caches      │
│ • HF model cache    │        │              │           │ • Run distill.py     │
│ • Dataset cache     │        │              │           │                     │
│ • conda pack        │        │              │           │                     │
└─────────────────────┘        └──────────────┘           └─────────────────────┘
```

---

## Step 1: Staging Machine (Online)

### 1.1 Install Miniforge (ARM64)

```bash
# Download from GitHub releases
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
```

### 1.2 Create Environment

```bash
cd /path/to/distill
conda env create -f environment.yml
conda activate distillation_m3
```

### 1.3 Cache Models and Datasets

Run the cache scripts on staging (with network):

**Meta Llama (requires HF login + license):**
```bash
python scripts/cache_models.py --output ./hf_cache
```

**Open models (no auth):**
```bash
python scripts/cache_models.py --open --output ./hf_cache
```

**Datasets** (use `--disk` for air-gapped: creates load_from_disk paths):
```bash
python scripts/cache_datasets.py --output ./datasets_cache --disk
```

**Bartowski GGUF (curl, US origin):** Quantized models from Bartowski — HuggingFace US region, resolves to cdn-lfs-us-1:

```bash
./scripts/download_bartowski_gguf.sh ./gguf_models
```

Single-file curl (US origin):

```bash
# Dolphin3.0 (public)
curl -L -o Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Dolphin3.0-Llama3.2-3B-GGUF/resolve/main/Dolphin3.0-Llama3.2-3B-Q4_K_M.gguf"

# Llama 3.2 1B (gated)
curl -L -H "Authorization: Bearer $HF_TOKEN" -o Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
```

Or manually (safetensors for distillation):

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Meta Llama (requires HF login + license) or open models (no auth)
models = ["meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-8B-Instruct"]
# Open alternative: ["Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct"]
for m in models:
    AutoTokenizer.from_pretrained(m, cache_dir="./hf_cache")
    AutoModelForCausalLM.from_pretrained(m, cache_dir="./hf_cache")

load_dataset("tatsu-lab/alpaca", cache_dir="./datasets_cache")
```

### 1.4 Package Environment

```bash
conda pack -n distillation_m3 -o distill-offline.tar.gz
```

### 1.5 Verify and Transfer

```bash
sha256sum distill-offline.tar.gz > SHA256SUMS
sha256sum hf_cache/* >> SHA256SUMS  # or recursive
sha256sum datasets_cache/* >> SHA256SUMS
# Copy to USB/SSD: distill-offline.tar.gz, hf_cache/, datasets_cache/, gguf_models/, code, SHA256SUMS
```

---

## Step 2: Target M3 (Air-Gapped)

### 2.1 Install Miniforge Offline

Transfer `Miniforge3-MacOSX-arm64.sh` and run:

```bash
bash Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3-offline
source ~/miniforge3-offline/bin/activate
```

### 2.2 Restore Environment

```bash
mkdir -p ~/envs/distill-offline
tar -xzf distill-offline.tar.gz -C ~/envs/distill-offline
source ~/envs/distill-offline/bin/activate
```

### 2.3 Verify Transfer

```bash
cd /path/to/copied/artifacts
sha256sum -c SHA256SUMS
```

### 2.4 Set Cache Paths

```bash
export HF_HOME=/path/to/copied/hf_cache
export HF_DATASETS_CACHE=/path/to/copied/datasets_cache
```

### 2.5 Verify MPS (M3 GPU)

```bash
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Expected: `MPS: True`

### 2.6 Run Distillation (Offline)

```bash
export HF_HOME=/path/to/copied/hf_cache
export HF_DATASETS_CACHE=/path/to/copied/datasets_cache

python scripts/distill_minillm.py --offline \
  --teacher meta-llama/Llama-3.2-8B-Instruct \
  --student meta-llama/Llama-3.2-1B-Instruct \
  --dataset /path/to/copied/datasets_cache/tatsu-lab_alpaca \
  --output_dir ./distilled
```

**Open models (no Meta license):**
```bash
python scripts/distill_minillm.py --offline --open \
  --dataset /path/to/copied/datasets_cache/tatsu-lab_alpaca \
  --output_dir ./distilled
```

`--offline` sets `local_files_only` for models; use pre-cached paths for `--dataset` (from `cache_datasets.py --disk`).

---

## Step 2.7 (optional): llama.cpp + student

Convert distilled student to GGUF and run with llama.cpp (no Python on target):

```bash
# On staging: after distill
python ../llama.cpp/convert_hf_to_gguf.py ./distilled-minillm --outfile distilled-student.gguf --outtype f16
# Transfer: distilled-student.gguf + llama.cpp build
# On target: ./llama-server -m distilled-student.gguf
```

See [LLAMA_CPP_STUDENT.md](LLAMA_CPP_STUDENT.md).

---

## Checklist

- [ ] Miniforge ARM64 installer on USB
- [ ] `distill-offline.tar.gz` (conda env)
- [ ] `hf_cache/` (models; use `cache_models.py --open` for no-auth)
- [ ] `datasets_cache/` (use `cache_datasets.py --disk` for load_from_disk paths)
- [ ] `gguf_models/` (optional; from `download_bartowski_gguf.sh`)
- [ ] Project code (`scripts/`, `configs/`)
- [ ] `SHA256SUMS` for verification
- [ ] Write-once media preferred for security

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `HF_HOME` not found | Ensure path is absolute and exists |
| Offline: connection error | Use `--offline`; set HF_HOME, HF_DATASETS_CACHE; use `--dataset /path/to/disked` |
| Dataset not found offline | Run `cache_datasets.py --disk` on staging; use the printed path for `--dataset` |
| MPS returns False | macOS Ventura+, check `torch.backends.mps.is_built()` |
| OOM during training | Reduce `batch_size`, enable `gradient_checkpointing` |
| Missing package | Re-run `conda pack` on staging with that package |
