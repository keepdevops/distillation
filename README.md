# Knowledge Distillation Toolkit

**Compress large language models with high-quality transfer on Apple M3**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A bare-metal, **air-gapped** knowledge distillation pipeline for Apple M3 (ARM64). Supports MiniLLM-style reverse KL, vanilla forward KL, MLX training (2–5× faster), CoreML export, and optional C++ inference. No Docker required; Conda for reproducibility.

**→ [Full User Manual](docs/USER_MANUAL.md)**
**→ [Pipeline Architecture & Phases](ARCHITECTURE.md)**

---

## Features

- **Three training backends** — PyTorch/MPS (default), MLX (2–5× faster, Apple-native), Unsloth (lowest memory) → [Backend Optimization Guide](BACKEND_OPTIMIZATIONS.md)
- **MiniLLM** (reverse KL) — mode-seeking, high-quality LLM distillation via TRL
- **Forward KL** — classic distillation for classification (BERT, ResNet)
- **Air-gapped** — full offline workflow with `conda-pack` and pre-cached models
- **M3-optimized** — MPS/MLX backend, gradient checkpointing, LoRA, 4-bit teacher
- **Three export formats** — GGUF (llama.cpp), CoreML (.mlpackage, Apple Neural Engine), MLX quantized weights
- **UX** — Universal Gradio UI (PyTorch/MLX/GGUF/vLLM), pipeline summary plots
- **Multi-user** — Shared model storage at `/Users/Shared/models` for all profiles
- **Thermal protection** — system-wide autonomous thermal agent monitors all jobs, auto-pause/resume
- **Training watchdog** — plateau detection, learning rate adjustment, LaunchAgent-ready
- **Agentic hyperparameter search** — hill-climbing over multi-trial runs, self-improving via experiment history
- **Production quality gates** — batch generation, refusal detection, embedding diversity, LLM-as-judge
- **Optional C++** — LibTorch + C++ watchdog for no-Python deployments

---

## Pipeline Phases

The system is organized around two sets of phases:

### Pipeline Stages (run every distillation)

| Stage | Name | Key scripts |
|-------|------|-------------|
| 0 | Preparation | `cache_models.py`, `generate_synthetic_data.py`, `distill_sft.py` |
| 1 | Distillation Core | `distill_mlx.py`, `distill_minillm.py`, `distill_unsloth.py` |
| 2 | Evaluation | `run_eval.py`, `eval_quality.py`, `run_benchmarks.py` |
| 3 | Export | `export_coreml.py`, `export_student_gguf.sh`, `mlx_lm.convert` |
| 4 | Agentic Feedback | `experiment_log.py`, `run_distillation_agent.py` |

### Optimization Phases (cumulative improvements to the pipeline)

| Phase | Speedup | Key optimizations |
|-------|---------|-------------------|
| Phase 1 | 40–50% | Flash Attention 2, torch.compile(), reduced eval_steps |
| Phase 1.5 | +13–20% | DataLoader workers, batch tuning, early stopping, memory clearing |
| **Cumulative** | **51–58%** | 7.2 hrs → 3.0–3.5 hrs for 5-trial autonomous run |

### Quality Gate Phases (layered evaluation depth)

| Phase | Focus | Key metrics |
|-------|-------|-------------|
| QG Phase 1 | Batch inference + basic filters | Pass rate, refusal rate, length |
| QG Phase 2 | Quality depth | Category balance, teacher PPL, embedding diversity |
| QG Phase 3 | Optimization | MLX backend, UMAP viz, volume guidance |

**→ Full in-depth explanation of all phases: [ARCHITECTURE.md](ARCHITECTURE.md)**

---

## Quick Start

### 1. Environment (Online)

```bash
conda env create -f environment.yml
conda activate distillation_m3
```

### 2. Offline / Air-Gapped

See [docs/air-gapped.md](docs/air-gapped.md) for full instructions.

**Staging:** `conda pack`, cache models/datasets, transfer via USB.  
**Target:** Unpack env, set `HF_HOME` and `HF_DATASETS_CACHE`, run.

### 3. Run Distillation

**Recommended — MLX backend (2–5× faster on M3):**
```bash
python scripts/distill_mlx.py --open --output_dir ./distilled-mlx
```

**PyTorch/MPS (default, maximum compatibility):**
```bash
# Without login: use open models (Qwen2 1.5B→0.5B)
python scripts/distill_minillm.py --open --output_dir ./distilled-minillm

# With Meta Llama (requires: huggingface-cli login + Meta license)
python scripts/distill_minillm.py \
  --teacher meta-llama/Llama-3.2-8B-Instruct \
  --student meta-llama/Llama-3.2-1B-Instruct \
  --output_dir ./distilled-minillm
```

**Forward KL (Classification):**
```bash
python scripts/distill_forward.py \
  --teacher bert-large-uncased \
  --student distilbert-base-uncased \
  --dataset glue --dataset_config sst2 \
  --output_dir ./distilled-forward
```

### 4. Plotting & Dashboard

**Plot training curves** (loss, learning rate) from any run:

```bash
python scripts/plot_training.py ./distilled-minillm -o training_curves.png
```

**Unified dashboard** — plots + model evaluation in one Gradio app:

```bash
python scripts/dashboard.py --runs_dir .
```

Opens at `http://127.0.0.1:7860` with **Plots** and **Evaluate** tabs.

**Standalone eval** (model-only):

```bash
python scripts/eval_gradio.py --model_path ./distilled-minillm
```

### 5. Autonomous Agent (full pipeline)

Run distill → export end-to-end without manual steps:

```bash
# PyTorch → GGUF (original behavior, unchanged)
python scripts/run_distillation_agent.py --open --export gguf

# MLX → all exports (GGUF + CoreML + MLX quant)
python scripts/run_distillation_agent.py --open --backend mlx --export all

# With plateau detection watchdog + config file
python scripts/run_distillation_agent.py --config configs/agent_config.json --watchdog
```

**Note:** For thermal protection, install the thermal agent separately (see section 6). It protects all jobs system-wide.

See [docs/AUTONOMOUS_AGENT.md](docs/AUTONOMOUS_AGENT.md) for config and LaunchAgent setup.

### 5b. Export to CoreML (.mlpackage, Apple Neural Engine)

```bash
python scripts/export_coreml.py --model_dir ./distilled-minillm --quantize int4
```

Produces `distilled-minillm.mlpackage` targeting CPU + GPU + ANE. Prints a Swift inference snippet.

### 6. Thermal Protection (system-wide)

**Thermal Agent** — Autonomous hardware monitoring that pauses ALL jobs when temps exceed threshold:

```bash
# Install as always-on system service (recommended)
./scripts/install_thermal_agent.sh

# Or run manually
python scripts/thermal_agent.py --watch ./distilled-minillm --threshold 85

# Fan control GUI (requires Macs Fan Control app)
python scripts/fan_control_popup.py
```

See [THERMAL_AGENT.md](THERMAL_AGENT.md) for full details.

### 7. Training Watchdog (plateau detection)

Monitor training for loss plateau and adjust learning rate:

```bash
python scripts/training_watchdog.py ./distilled-minillm --interval 60
python scripts/distill_minillm.py --output_dir ./distilled-minillm --watchdog  # pause.flag support
```

**C++ (standalone, no Python):**
```bash
cd cpp && mkdir -p build && cd build && cmake .. && cmake --build .
./watchdog ../distilled-minillm --interval 60   # or use absolute path
```
See [docs/WATCHDOG.md](docs/WATCHDOG.md).

---

## Project Structure

```
distill/
├── configs/
│   ├── agent_config.json     # Agent defaults (backend, export, epochs, ...)
│   └── watchdog_rules.json   # Plateau + divergence thresholds
├── cpp/                      # C++ LibTorch distillation + watchdog
│   ├── CMakeLists.txt
│   ├── main.cpp
│   ├── watchdog_main.cpp     # Watchdog (no LibTorch required)
│   └── third_party/nlohmann/json.hpp
├── docs/
│   ├── USER_MANUAL.md        # ← Full step-by-step user manual
│   ├── AUTONOMOUS_AGENT.md   # Agent usage & LaunchAgent setup
│   ├── SOFTWARE_DESIGN_DOCUMENT.md
│   ├── air-gapped.md         # Offline operation checklist
│   ├── WATCHDOG.md           # Plateau detection reference
│   ├── QUANTIZATION.md       # Post-training quantization guide
│   ├── LLAMA_CPP_STUDENT.md  # GGUF/llama-server deployment
│   └── UNIVERSAL_GRADIO.md   # Dashboard usage guide
├── scripts/
│   # Training scripts
│   ├── distill_minillm.py    # PyTorch/MPS backend (MiniLLM, reverse KL)
│   ├── distill_mlx.py        # MLX backend (Apple-native, 2-5× faster)
│   ├── distill_unsloth.py    # Unsloth backend (optimized LoRA + KD)
│   ├── distill_sft.py        # Supervised fine-tuning warmup
│   ├── distill_forward.py    # Forward KL (classification models)
│   ├── data_pipeline.py      # Shared dataset loading + tokenization
│   # Orchestration & agents
│   ├── run_distillation_agent.py  # End-to-end pipeline orchestrator
│   ├── experiment_log.py     # Agentic experiment memory (JSONL)
│   ├── thermal_agent.py      # System-wide thermal monitoring & protection
│   ├── training_watchdog.py  # Plateau + divergence detection (Python)
│   ├── watchdog_callbacks.py # PauseFlagCallback + MetricsCallback (HF Trainer)
│   ├── early_stopping_callback.py # Early stop on diverging trials
│   # Thermal monitoring & control
│   ├── monitor_cpu_gpu_temp.py    # Thermal logging via mactop
│   ├── fan_control_popup.py  # Fan control GUI with temp display
│   ├── install_thermal_agent.sh   # Install thermal agent as LaunchAgent
│   # Export scripts
│   ├── export_coreml.py      # CoreML export → .mlpackage (ANE)
│   ├── export_student_gguf.sh     # GGUF export wrapper
│   # Dashboard & visualization
│   ├── dashboard.py          # Gradio: plots + eval + artifacts + thermal
│   ├── eval_gradio.py        # Standalone eval UI
│   ├── plot_training.py      # Loss/LR curves
│   ├── plot_gguf_pipeline.py # Pipeline summary (GGUF/CoreML/MLX artifacts)
│   # Evaluation & quality
│   ├── eval_quality.py       # Quality gates: batch gen, refusal, diversity, judge
│   ├── run_eval.py           # Validation perplexity evaluation
│   ├── run_benchmarks.py     # WikiText-2 benchmark suite
│   # Data & offline support
│   ├── cache_models.py       # Pre-download HF models for offline use
│   ├── cache_datasets.py     # Pre-download datasets for offline use
│   ├── setup_airgap.py       # Air-gap bundle automation
│   ├── generate_synthetic_data.py # Synthetic instruction-response generation
│   ├── filter_dataset.py     # Dataset deduplication + quality filtering
│   # Service management
│   ├── start.sh              # Launch all services in tmux session 'distill' with caffeinate
│   └── stop.sh               # Stop all services and kill distill tmux sessions
├── # Root-level documentation
├── ARCHITECTURE.md           # ← In-depth pipeline phases & design
├── BACKEND_OPTIMIZATIONS.md  # Backend feature matrix + optimization guide
├── SPEEDUP_PHASE1.md         # Phase 1 speedup: Flash Attn + compile (40-50%)
├── SPEEDUP_PHASE1_5.md       # Phase 1.5 speedup: DataLoader + early stop (+15-20%)
├── PRODUCTION_QUALITY_GATES.md  # Quality gates Phases 1-3 implementation
├── QUALITY_GATES_QUICKSTART.md  # Quick start for quality evaluation
├── THERMAL_AGENT.md          # Thermal protection reference
├── STARTUP_GUIDE.md          # Terminal startup sequence
├── AIRGAP_SETUP.md           # Air-gap staging guide
├── MULTI_USER_SETUP.md       # Shared model storage setup
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Inference Engine?

**For distillation:** Not needed. PyTorch/MLX handles training and inference.

**For production deployment:** Recommended.

| Target        | Option                    |
|---------------|---------------------------|
| M3 Neural Engine | Core ML (coremltools)   |
| Cross-platform   | ONNX Runtime           |
| MLX-native       | MLX (Apple Silicon)     |
| C++ binary       | LibTorch / llama.cpp    |

**llama.cpp + distilled student:** Convert student to GGUF, run with `llama-server`. See [docs/LLAMA_CPP_STUDENT.md](docs/LLAMA_CPP_STUDENT.md).

---

## Quantization After Distillation

Best order: **Prune → Distill → Quantize (P-KD-Q)**. Distillation first produces a compressible student; then apply GPTQ or AWQ for 2–4× further compression. See [docs/QUANTIZATION.md](docs/QUANTIZATION.md).

---

## Citation

```bibtex
@article{gu2024minillm,
  title={MiniLLM: Knowledge Distillation of Large Language Models},
  author={Gu, Yuxian and Dong, Li and Wei, Furu and Huang, Minlie},
  journal={ICLR},
  year={2024}
}
```

---

## License

MIT
