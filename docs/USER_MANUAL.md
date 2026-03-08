# User Manual — Knowledge Distillation Toolkit

**Apple M3 · Air-gapped compatible · MLX / PyTorch / Unsloth backends**

---

## Table of Contents

1. [What This Toolkit Does](#1-what-this-toolkit-does)
2. [Prerequisites](#2-prerequisites)
3. [Installation](#3-installation)
4. [Choosing a Backend](#4-choosing-a-backend)
5. [Step-by-Step: First Run (PyTorch)](#5-step-by-step-first-run-pytorch)
6. [Step-by-Step: MLX Backend (Recommended for M3)](#6-step-by-step-mlx-backend)
7. [Step-by-Step: Unsloth Backend](#7-step-by-step-unsloth-backend)
8. [Export Formats](#8-export-formats)
9. [Running the Full Pipeline (Agent)](#9-running-the-full-pipeline-agent)
10. [Dashboard & Visualization](#10-dashboard--visualization)
11. [Thermal Protection (System-Wide)](#11-thermal-protection-system-wide)
12. [Training Watchdog (Plateau Detection)](#12-training-watchdog-plateau-detection)
13. [Starting All Services at Once](#13-starting-all-services-at-once)
14. [Air-Gapped / Offline Operation](#14-air-gapped--offline-operation)
15. [Configuration Reference](#15-configuration-reference)
16. [Script Reference](#16-script-reference)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. What This Toolkit Does

**Knowledge distillation** compresses a large "teacher" model into a smaller "student" model that retains most of the teacher's quality. This toolkit:

- Trains the student to mimic the teacher's output distribution (KL divergence loss)
- Applies **LoRA** adapters so only a fraction of parameters are updated
- Runs entirely on your Mac — no cloud, no GPU server required
- Exports the result to **GGUF** (llama.cpp), **CoreML** (.mlpackage, Apple Neural Engine), or **MLX** quantized weights

**Three training backends:**

| Backend | Speed on M3 | Memory | Best for |
|---------|-------------|--------|----------|
| `pytorch` | 1× (baseline) | ~8–12 GB | Compatibility, existing workflows |
| `mlx` | 2–5× faster | ~6–10 GB | Daily use on Apple Silicon |
| `unsloth` | 2–4× faster | ~4–8 GB | Lowest memory, optional |

**Default models (no login required):**
- Teacher: `Qwen/Qwen2-1.5B-Instruct`
- Student: `Qwen/Qwen2-0.5B-Instruct`
- Dataset: `tatsu-lab/alpaca`

---

## 2. Prerequisites

| Requirement | Notes |
|-------------|-------|
| macOS Ventura 13+ | Required for MPS and CoreML |
| Apple Silicon (M1/M2/M3) | MLX backend requires ARM64 |
| Miniforge / Conda | For environment management |
| 16 GB RAM minimum | 36 GB recommended for 8B teacher |
| ~20 GB free disk | Models + outputs |
| Python 3.11 | Specified in `environment.yml` |

Check your macOS version: `sw_vers -productVersion`

---

## 3. Installation

### 3.1 Install Miniforge (if not installed)

```bash
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh -b -p ~/miniforge3
source ~/miniforge3/bin/activate
```

### 3.2 Create the Conda Environment

```bash
cd /path/to/distill
conda env create -f environment.yml
conda activate distillation_m3
```

This installs PyTorch, Transformers, PEFT, TRL, Gradio, MLX, mlx-lm, and coremltools.

### 3.3 Verify the Setup

```bash
# Check MPS (M3 GPU)
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"

# Check MLX
python -c "import mlx.core as mx; print('MLX:', mx.__version__)"

# Check CoreML
python -c "import coremltools as ct; print('CoreML:', ct.__version__)"
```

Expected output:
```
MPS: True
MLX: 0.31.0
CoreML: 9.0
```

### 3.4 Install Unsloth (Optional)

Unsloth provides additional kernel optimizations. Install separately:

```bash
pip install "unsloth[mlx]"   # Apple Silicon with MLX backend
# or
pip install unsloth           # CPU/CUDA fallback
```

If Unsloth is not installed, `distill_unsloth.py` prints clear install instructions and exits — no crash.

---

## 4. Choosing a Backend

**Rule of thumb:**
- New to the toolkit → start with `pytorch` (safe, well-tested)
- Daily use on M3 → use `mlx` (2–5× faster, same output quality)
- Lowest memory footprint → try `unsloth` (requires separate install)

All three backends produce:
- `metrics.jsonl` — same format, dashboard works unchanged
- Respect `pause.flag` — watchdog works unchanged
- Compatible with all export formats

---

## 5. Step-by-Step: First Run (PyTorch)

### Step 1 — Activate the environment

```bash
conda activate distillation_m3
cd /path/to/distill
```

### Step 2 — Run a quick test distillation (20 samples, 1 epoch)

```bash
python scripts/distill_minillm.py \
  --open \
  --max_samples 20 \
  --epochs 1 \
  --output_dir ./test-run
```

`--open` uses the Qwen2 models (no HuggingFace login needed).

You should see:
```
Device: mps
Using open models: Qwen/Qwen2-1.5B-Instruct → Qwen/Qwen2-0.5B-Instruct
{'loss': 2.34, 'epoch': 0.5, ...}
```

### Step 3 — Check the output

```bash
ls test-run/
# config.json  metrics.jsonl  pytorch_model.bin  tokenizer.json ...
```

### Step 4 — View the loss curve

```bash
python scripts/plot_training.py ./test-run -o test-run/curves.png
open test-run/curves.png
```

### Step 5 — Run a full production distillation

```bash
python scripts/distill_minillm.py \
  --open \
  --max_samples 2000 \
  --epochs 2 \
  --lora_r 64 \
  --output_dir ./distilled-minillm
```

This takes 30–90 minutes on M3 Max (2000 samples, 2 epochs).

---

## 6. Step-by-Step: MLX Backend

MLX is Apple's native ML framework — lazy evaluation, unified memory, 2–5× faster than PyTorch/MPS for this workload.

### Step 1 — Verify MLX is installed

```bash
python -c "import mlx_lm; print('OK')"
```

If you see `ModuleNotFoundError`: `pip install mlx mlx-lm`

### Step 2 — Run a quick test

```bash
python scripts/distill_mlx.py \
  --open \
  --max_samples 20 \
  --epochs 1 \
  --lora_r 4 \
  --no_export \
  --output_dir ./test-mlx
```

You should see per-step loss lines and an `eval_loss`:
```
step=2  epoch=0.10  loss=0.70  0.59 steps/s
step=4  epoch=0.30  loss=4.56  0.72 steps/s
  eval_loss=3.30
```

### Step 3 — Run a full MLX distillation

```bash
python scripts/distill_mlx.py \
  --open \
  --max_samples 2000 \
  --epochs 2 \
  --lora_r 64 \
  --output_dir ./distilled-mlx
```

### Step 4 — With MLX quantization (optional, runs after training)

```bash
python scripts/distill_mlx.py \
  --open \
  --max_samples 2000 \
  --epochs 2 \
  --q_bits 4 \
  --output_dir ./distilled-mlx
```

This saves both raw weights (`mlx_student_weights.npz`) and a 4-bit quantized copy (`distilled-mlx/mlx_q4/`).

### Step 5 — Check outputs

```bash
ls distilled-mlx/
# distill_config.json  metrics.jsonl  mlx_student_weights.npz  mlx_q4/
```

---

## 7. Step-by-Step: Unsloth Backend

Unsloth uses optimized LoRA kernels. Uses MLX for the teacher, PyTorch for the student.

### Step 1 — Install Unsloth

```bash
pip install "unsloth[mlx]"
```

### Step 2 — Verify installation

```bash
python -c "import unsloth; print('Unsloth OK')"
```

### Step 3 — Run distillation

```bash
python scripts/distill_unsloth.py \
  --open \
  --max_samples 2000 \
  --epochs 2 \
  --q_bits 4 \
  --output_dir ./distilled-unsloth
```

`--q_bits 4` loads the student in 4-bit (saves ~50% memory). Use `--q_bits 8` for higher quality.

### Step 4 — Graceful fallback test

If Unsloth is not installed, the script exits cleanly:

```bash
python scripts/distill_unsloth.py --open
# ERROR: unsloth is not installed.
# Install instructions: pip install 'unsloth[mlx]'
# Tip: Use --backend pytorch or --backend mlx if Unsloth is unavailable.
```

---

## 8. Export Formats

After distillation, convert the student model for deployment.

### 8.1 GGUF (llama.cpp, llama-server)

Requires llama.cpp in `../llama.cpp/` or `./llama.cpp/`.

```bash
python llama.cpp/convert_hf_to_gguf.py ./distilled-minillm \
  --outfile ./distilled-minillm/student-f16.gguf \
  --outtype f16

# Serve it:
./llama.cpp/build/bin/llama-server \
  -m ./distilled-minillm/student-f16.gguf
```

GGUF quantization types (smaller = faster inference, slightly lower quality):

| `--outtype` | Size (0.5B model) | Quality |
|-------------|-------------------|---------|
| `f16` | ~1.0 GB | Best |
| `q8_0` | ~0.5 GB | Very good |
| `q4_K_M` | ~0.3 GB | Good |

### 8.2 CoreML (.mlpackage, Apple Neural Engine)

```bash
# Basic export
python scripts/export_coreml.py \
  --model_dir ./distilled-minillm

# With int4 quantization (~75% smaller)
python scripts/export_coreml.py \
  --model_dir ./distilled-minillm \
  --quantize int4

# Target specific compute units
python scripts/export_coreml.py \
  --model_dir ./distilled-minillm \
  --compute_units ALL         # CPU + GPU + ANE (default)
  # or: CPU_ONLY, CPU_AND_GPU, CPU_AND_NE
```

The script prints a Swift inference snippet for the generated `.mlpackage`.

Output: `distilled-minillm/distilled-minillm.mlpackage`

Quantization options:

| `--quantize` | Compression | Notes |
|--------------|-------------|-------|
| *(none)* | None | Float32 weights |
| `float16` | ~2× | Good for all ANE targets |
| `int8` | ~4× | Fast on ANE |
| `int4` | ~8× | Smallest; use for edge |

### 8.3 MLX Quantized Weights

```bash
# Standalone (from any existing model directory)
python -c "
from mlx_lm import convert
convert('Qwen/Qwen2-0.5B-Instruct', quantize=True, q_bits=4, mlx_path='./mlx_q4')
"

# Or built into distill_mlx.py (runs automatically unless --no_export)
python scripts/distill_mlx.py --open --q_bits 4 --output_dir ./distilled-mlx
```

---

## 9. Running the Full Pipeline (Agent)

The agent chains distillation → export in one command. Use this for unattended runs.

### 9.1 Basic usage

```bash
# PyTorch backend + GGUF export (existing behavior, unchanged)
python scripts/run_distillation_agent.py --open --export gguf

# MLX backend + GGUF export
python scripts/run_distillation_agent.py --open --backend mlx --export gguf

# MLX backend + all exports (GGUF + CoreML + MLX quant)
python scripts/run_distillation_agent.py --open --backend mlx --export all

# Unsloth backend + CoreML export
python scripts/run_distillation_agent.py --open --backend unsloth --export coreml
```

### 9.2 With watchdog (plateau + thermal protection)

```bash
python scripts/run_distillation_agent.py \
  --open \
  --backend mlx \
  --export all \
  --watchdog \
  --epochs 3 \
  --max_samples 5000
```

The watchdog runs as a separate process (started by `start.sh`). The `--watchdog` flag tells the trainer to check for `pause.flag` every step.

### 9.3 Using a config file

Edit `configs/agent_config.json`:

```json
{
  "output_dir": "./distilled-minillm",
  "open": true,
  "offline": false,
  "watchdog": true,
  "backend": "mlx",
  "export": "all",
  "outtype": "f16",
  "q_bits": 4,
  "coreml_quantize": "int4",
  "epochs": 2,
  "max_samples": 2000,
  "temperature": 1.0,
  "lora_r": 64
}
```

```bash
python scripts/run_distillation_agent.py --config configs/agent_config.json
```

### 9.4 All agent flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `pytorch` | Training backend: `pytorch`, `mlx`, `unsloth` |
| `--export` | `gguf` | Export format: `gguf`, `coreml`, `mlx`, `all`, `none` |
| `--open` | off | Use Qwen2 open models (no HF login) |
| `--offline` | off | Air-gapped: local cache only |
| `--watchdog` | off | Enable pause.flag / plateau detection |
| `--outtype` | `f16` | GGUF quant type: `f16`, `q8_0`, `q4_K_M` |
| `--q_bits` | `4` | MLX quant bits: `4` or `8` |
| `--coreml_quantize` | none | CoreML quant: `int4`, `int8`, `float16` |
| `--epochs` | `2` | Training epochs |
| `--max_samples` | `2000` | Max training samples |
| `--temperature` | `1.0` | KD softmax temperature |
| `--lora_r` | `64` | LoRA rank |
| `--config` | — | JSON config file (overrides CLI) |
| `--verbose` / `-v` | off | Debug logging |

---

## 10. Dashboard & Visualization

The Gradio dashboard shows live training curves, eval perplexity, artifact sizes, and a model eval UI.

### 10.1 Start the dashboard

```bash
python scripts/dashboard.py
# Opens at http://127.0.0.1:7860
```

### 10.2 What you see

**Plots tab** — auto-refreshes every 30 s:
- Training loss curve (from `metrics.jsonl`)
- Eval perplexity
- Model artifacts bar chart: GGUF (blue), CoreML (purple), MLX weights (green)
- Inference speed (if `llama-cpp-python` installed)

**Evaluate tab** — interactive prompt testing:
- Load any model from a run directory
- Type a prompt, get the student's response
- Compare teacher vs student side by side

### 10.3 Generate a standalone pipeline plot

```bash
python scripts/plot_gguf_pipeline.py ./distilled-minillm
open distilled-minillm/pipeline_summary.png
```

### 10.4 Plot training curves only

```bash
python scripts/plot_training.py ./distilled-minillm -o curves.png
open curves.png
```

---

## 11. Thermal Protection (System-Wide)

The **thermal agent** is an autonomous system service that monitors CPU/GPU temperatures and automatically pauses ALL running jobs when thermal limits are exceeded. Unlike the training watchdog (plateau detection), the thermal agent provides system-wide hardware protection.

### 11.1 Why use the thermal agent?

**System-wide protection:**
- Protects ALL GPU workloads: training, inference, export, benchmarks
- Monitors multiple jobs simultaneously
- Always-on protection via LaunchAgent (survives reboots)
- No need to remember `--watchdog` flags

**Auto-resume:**
- Pauses jobs when temp ≥ 85°C (configurable)
- Resumes when temp drops to 80°C (5°C hysteresis)
- Prevents thermal throttling and hardware damage

### 11.2 Installation (recommended)

Install as macOS LaunchAgent for always-on protection:

```bash
./scripts/install_thermal_agent.sh

# Verify installation
launchctl list | grep thermal_agent
```

The agent starts automatically on login and protects all jobs.

### 11.3 Manual usage

```bash
# Watch single job
python scripts/thermal_agent.py --watch ./distilled-minillm

# Watch multiple jobs (system-wide)
python scripts/thermal_agent.py --watch ./distilled-minillm ./distilled-mlx

# Custom threshold (default: 85°C)
python scripts/thermal_agent.py --watch . --threshold 70 --interval 15

# Daemon mode (background process)
python scripts/thermal_agent.py --daemon --watch . --log thermal_agent.jsonl
```

### 11.4 Fan control GUI

Manual fan speed control with temperature display:

```bash
python scripts/fan_control_popup.py

# With warning threshold
python scripts/fan_control_popup.py --threshold 75
```

**Requirements:** [Macs Fan Control](https://crystalidea.com/macs-fan-control) app installed

### 11.5 Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--watch` | (required) | Directories to monitor |
| `--threshold` | 85 | Temperature threshold (°C) |
| `--metric` | `soc_temp_c` | Metric: `soc_temp_c`, `cpu_temp_c`, `gpu_temp_c` |
| `--hysteresis` | 5 | Resume delta (°C) |
| `--interval` | 30 | Poll interval (seconds) |
| `--log` | — | Log file for thermal events (JSONL) |
| `--daemon` | off | Run as background daemon |

**Recommended thresholds:**
- M1/M2: 80°C (conservative) to 85°C (balanced)
- M3: 85°C (conservative) to 90°C (balanced)
- M3 Max/Pro: 90°C (conservative) to 95°C (balanced)

See [THERMAL_AGENT.md](../THERMAL_AGENT.md) for complete documentation.

---

## 12. Training Watchdog (Plateau & Divergence Detection)

The **training watchdog** monitors `trainer_state.json` during training and writes `pause.flag` when it detects problems. The trainer then saves and exits gracefully. This is separate from thermal protection and focuses on ML-specific monitoring.

**Two detection modes:**

| Mode | Trigger | Action |
|------|---------|--------|
| **Plateau** | Loss changes < 0.001 for N consecutive steps | Writes `watchdog_suggestions.json` with scaled LR |
| **Divergence** | Recent avg loss > early baseline avg × 1.5 | Writes `pause.flag` immediately to stop training |

### 12.1 Run watchdog in a separate terminal

```bash
# Terminal 1: start training with watchdog support
python scripts/distill_mlx.py --open --watchdog --output_dir ./distilled-mlx

# Terminal 2: run the watchdog
python scripts/training_watchdog.py ./distilled-mlx --interval 60
```

### 12.2 C++ watchdog (lighter, no Python)

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build .
./watchdog ../../distilled-mlx --interval 60 --config ../../configs/watchdog_rules.json
```

### 12.3 Watchdog config

`configs/watchdog_rules.json`:

```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "min_points": 5,
    "lr_scale": 0.8
  }
}
```

### 12.4 LaunchAgent (survives reboot)

```bash
# Edit paths in the plist
cp scripts/launch_agent/com.caribou.distill-watchdog.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
```

See [WATCHDOG.md](WATCHDOG.md) for full LaunchAgent setup.

---

## 13. Starting All Services at Once

`start.sh` launches the watchdog + dashboard (+ optional thermal monitor) in one command.

```bash
# Watchdog + dashboard only
./scripts/start.sh

# With thermal monitor
./scripts/start.sh --monitor

# Eval UI instead of dashboard
./scripts/start.sh --eval

# Pass backend to training launcher
./scripts/start.sh --backend=mlx

# Pass export format
./scripts/start.sh --backend=mlx --export=all
```

PIDs are saved to `.distill.pids`. Stop everything:

```bash
./scripts/stop.sh   # or: kill $(cat .distill.pids | awk '{print $1}')
```

Services started:

| Service | Log | URL |
|---------|-----|-----|
| `training_watchdog.py` | `watchdog.log` | — |
| `dashboard.py` | `dashboard.log` | http://127.0.0.1:7860 |
| `monitor_cpu_gpu_temp.py` | `thermal.log` | — (optional) |

---

## 14. Air-Gapped / Offline Operation

Full offline workflow — no internet on the target machine after setup.

### 14.1 Stage on an online machine

```bash
# 1. Cache models
python scripts/cache_models.py --open --output ./hf_cache

# 2. Cache datasets
python scripts/cache_datasets.py --output ./datasets_cache --disk

# 3. Package the conda env
conda activate distillation_m3
conda pack -n distillation_m3 -o distill-offline.tar.gz

# 4. Verify checksums
sha256sum distill-offline.tar.gz hf_cache/* > SHA256SUMS
```

### 14.2 Transfer to air-gapped machine

Copy via USB/SSD:
- `distill-offline.tar.gz`
- `hf_cache/`
- `datasets_cache/`
- Project code (`scripts/`, `configs/`)
- `SHA256SUMS`

### 14.3 Set up on air-gapped machine

```bash
# Restore environment
mkdir -p ~/envs/distill-offline
tar -xzf distill-offline.tar.gz -C ~/envs/distill-offline
source ~/envs/distill-offline/bin/activate

# Set cache paths
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/datasets_cache
```

### 14.4 Run offline

```bash
# PyTorch backend
python scripts/distill_minillm.py --offline --open --output_dir ./distilled

# MLX backend
python scripts/distill_mlx.py --offline --open --output_dir ./distilled-mlx

# Full agent
python scripts/run_distillation_agent.py \
  --offline --open --backend mlx --export gguf
```

`--offline` sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` automatically.

See [air-gapped.md](air-gapped.md) for the full checklist.

---

## 15. Configuration Reference

### 15.1 `configs/agent_config.json`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `output_dir` | string | `./distilled-minillm` | Where to save results |
| `open` | bool | `true` | Use Qwen2 models (no HF login) |
| `offline` | bool | `false` | Air-gapped mode |
| `watchdog` | bool | `false` | Enable pause.flag support |
| `backend` | string | `pytorch` | `pytorch`, `mlx`, or `unsloth` |
| `export` | string | `gguf` | `gguf`, `coreml`, `mlx`, `all`, `none` |
| `outtype` | string | `f16` | GGUF type: `f16`, `q8_0`, `q4_K_M` |
| `q_bits` | int | `4` | MLX quant bits: `4` or `8` |
| `coreml_quantize` | string/null | `null` | CoreML quant: `int4`, `int8`, `float16` |
| `epochs` | int | `2` | Training epochs |
| `max_samples` | int | `2000` | Max training samples |
| `temperature` | float | `1.0` | KD temperature |
| `lora_r` | int | `64` | LoRA rank |

### 15.2 `configs/watchdog_rules.json`

**Plateau detection:**

| Key | Default | Description |
|-----|---------|-------------|
| `plateau.window` | `3` | Number of recent loss deltas to check |
| `plateau.max_delta` | `0.001` | Loss change below this for all N steps → plateau |
| `plateau.min_points` | `5` | Minimum loss points before checking |
| `plateau.lr_scale` | `0.8` | Multiply LR suggestion by this on plateau |

**Divergence detection (new):**

| Key | Default | Description |
|-----|---------|-------------|
| `divergence.window` | `3` | Recent avg = mean of last N losses |
| `divergence.threshold` | `1.5` | Pause if recent avg > baseline avg × this |
| `divergence.baseline_window` | `5` | Baseline = mean of first N losses |
| `divergence.min_points` | `8` | Minimum points before divergence check starts |

Example config with both rules:

```json
{
  "plateau": { "window": 3, "max_delta": 0.001, "min_points": 5, "lr_scale": 0.8 },
  "divergence": { "window": 3, "threshold": 1.5, "baseline_window": 5, "min_points": 8 }
}
```

**Note:** Thermal monitoring is handled by the standalone `thermal_agent.py` (see section 11).

---

## 16. Script Reference

### Training scripts

| Script | Backend | When to use |
|--------|---------|-------------|
| `scripts/distill_minillm.py` | PyTorch/MPS | Existing workflows, maximum compatibility |
| `scripts/distill_mlx.py` | MLX | Daily use, fastest on M3 |
| `scripts/distill_unsloth.py` | Unsloth+MLX | Lowest memory, optional install |
| `scripts/distill_sft.py` | PyTorch/MPS | Supervised fine-tuning |
| `scripts/distill_forward.py` | PyTorch/MPS | Classification models (BERT, etc.) |

### Export scripts

| Script | Output | When to use |
|--------|--------|-------------|
| `scripts/export_coreml.py` | `.mlpackage` | iOS/macOS apps, Apple Neural Engine |
| `scripts/export_student_gguf.sh` | `.gguf` | Wrapper for llama.cpp conversion |
| `llama.cpp/convert_hf_to_gguf.py` | `.gguf` | llama-server, cross-platform inference |
| `mlx_lm.convert` (built-in) | `mlx_q4/` dir | MLX inference, smallest on Apple Silicon |

### Orchestration & agents

| Script | Purpose |
|--------|---------|
| `scripts/run_distillation_agent.py` | End-to-end pipeline: distill → export |
| `scripts/thermal_agent.py` | System-wide thermal monitoring & protection |
| `scripts/training_watchdog.py` | Plateau detection (ML-specific) |
| `scripts/watchdog_callbacks.py` | PauseFlagCallback + MetricsCallback |
| `cpp/build/watchdog` | C++ watchdog (no Python) |

### Thermal monitoring & control

| Script | Purpose |
|--------|---------|
| `scripts/monitor_cpu_gpu_temp.py` | Thermal logging via mactop |
| `scripts/fan_control_popup.py` | Fan control GUI with temp display |
| `scripts/install_thermal_agent.sh` | Install thermal agent as LaunchAgent |
| `scripts/test_fan_control.py` | Test fan control integration |

### Dashboard & visualization

| Script | Purpose |
|--------|---------|
| `scripts/dashboard.py` | Gradio UI: curves + eval + artifacts |
| `scripts/eval_gradio.py` | Standalone model evaluation UI |
| `scripts/plot_training.py` | Plot loss/LR curves |
| `scripts/plot_gguf_pipeline.py` | Full pipeline summary figure |

### Evaluation & quality

| Script | Purpose |
|--------|---------|
| `scripts/eval_quality.py` | Quality gates, diversity metrics, per-sample teacher PPL, LLM-as-judge |
| `scripts/early_stopping_callback.py` | Stop training early if loss diverges from baseline (HF Trainer callback) |
| `scripts/run_eval.py` | Validation loss evaluation |
| `scripts/run_benchmarks.py` | Benchmark suite |
| `scripts/experiment_log.py` | Experiment tracking |

### Data & offline support

| Script | Purpose |
|--------|---------|
| `scripts/data_pipeline.py` | Shared dataset loading, prompt formatting, and tokenization utilities |
| `scripts/cache_models.py` | Pre-download HF models for offline use |
| `scripts/cache_datasets.py` | Pre-download datasets for offline use |
| `scripts/setup_airgap.py` | Air-gap setup automation |
| `scripts/generate_synthetic_data.py` | Synthetic instruction-response data generation (batched, dedup-filtered) |

### Service management

| Script | Purpose |
|--------|---------|
| `scripts/start.sh` | Launch all services in background |
| `scripts/stop.sh` | Stop all background services |

---

## 17. Troubleshooting

### Thermal agent issues

| Problem | Fix |
|---------|-----|
| `mactop not found` | `brew install context-labs/tap/mactop` |
| Thermal readings unavailable | Ensure running on Apple Silicon; test `mactop --headless --format json --count 1` |
| pause.flag not cleared after cooldown | Check flag content with `cat pause.flag \| jq`; thermal agent only clears flags it created |
| Multiple thermal agents running | `pkill -f thermal_agent.py` then restart one instance |
| Fan control popup doesn't work | Install [Macs Fan Control](https://crystalidea.com/macs-fan-control) app |

### MLX issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'mlx'` | `pip install mlx mlx-lm` |
| `TokenizerWrapper not callable` | Already fixed in `distill_mlx.py` (uses `._tokenizer`) |
| `LoRALinear has no attribute 'from_linear'` | Already fixed — uses `linear_to_lora_layers` |
| Low step rate (<0.5 steps/s) | Normal for first epoch (model loading); increases after |
| OOM during MLX training | Reduce `--batch_size` to 1 or 2; reduce `--lora_r` to 4 |

### PyTorch / MPS issues

| Problem | Fix |
|---------|-----|
| `MPS: False` | Requires macOS Ventura 13+; check `torch.backends.mps.is_built()` |
| OOM on MPS | Reduce `--batch_size`; add `--use_4bit_teacher` |
| Slow training on MPS | Normal; try `--backend mlx` for 2–5× speedup |

### CoreML export issues

| Problem | Fix |
|---------|-----|
| `torch.jit.trace` fails | Model must be in `eval()` mode with float32 weights |
| `coremltools not found` | `pip install 'coremltools>=8.0'` |
| Torch version warning | Expected — torch 2.10 is newer than tested; usually works |
| `.mlpackage` too large | Add `--quantize int4` or `int8` |

### Unsloth issues

| Problem | Fix |
|---------|-----|
| `unsloth is not installed` | `pip install 'unsloth[mlx]'` — or use `--backend mlx` |
| CUDA error on Apple Silicon | Use `unsloth[mlx]` not bare `unsloth` |

### Dataset issues

| Problem | Fix |
|---------|-----|
| `trust_remote_code not supported` | Already fixed — removed from `load_dataset` call |
| Dataset not found offline | Run `cache_datasets.py --disk` on staging machine first |
| `KeyError: 'train'` | Dataset may use a different split name; check with `ds.keys()` |

### Dashboard issues

| Problem | Fix |
|---------|-----|
| No runs appear in Plots tab | Ensure `metrics.jsonl` exists in the run directory |
| Empty artifacts panel | Ensure `.gguf`, `.mlpackage`, or `.npz` files exist in the run dir |
| Port 7860 in use | Another Gradio app is running; kill it or change port |

### Watchdog issues

| Problem | Fix |
|---------|-----|
| `pause.flag` not detected | Trainer must be started with `--watchdog` flag |
| Watchdog exits immediately | Check `trainer_state.json` exists (needs at least 1 training step) |
| Divergence not triggering | Needs `divergence.min_points` (default 8) losses logged before checking |
| False divergence at start | First few losses may be high; increase `divergence.baseline_window` or `min_points` |

### General

| Problem | Fix |
|---------|-----|
| `HF_HOME not found` | Use absolute path: `export HF_HOME=/Users/you/distill/hf_cache` |
| Models re-downloading | Set `HF_HOME` before running; use `--offline` to block network |
| `RuntimeError: exit code 1` | Run the subscript directly for full error message |
| Package version conflicts | Re-create env: `conda env remove -n distillation_m3 && conda env create -f environment.yml` |
