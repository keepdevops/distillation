# Distillation Pipeline Architecture

## Overview

This is an **autonomous agentic knowledge distillation system** (2025-2026 pattern) where an LLM-powered agent acts as an autonomous ML engineer, self-improving through:

1. **Hyperparameter proposal** (via historical performance analysis)
2. **Automated execution** (distillation → evaluation → export)
3. **Quality assessment** (perplexity, diversity, instruction-following)
4. **Adaptive retry** (hill-climbing toward better configurations)

---

## Core Components

### 1. Autonomous Agent (`run_distillation_agent.py`)

The orchestrator that runs the full pipeline end-to-end:

```bash
python scripts/run_distillation_agent.py \
    --open \
    --n_trials 3 \        # Run 3 trials, pick best
    --epochs 2 \
    --export all \        # GGUF + CoreML + MLX
    --log_experiment      # Feed history to agentic loop
```

**Pipeline Stages:**

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0: Preparation (optional)                                │
├─────────────────────────────────────────────────────────────────┤
│  • Cache models/datasets (--offline mode)                       │
│  • Generate synthetic data (--synthetic_data)                   │
│  • SFT warmup (--curriculum)                                    │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Distillation Core                                     │
├─────────────────────────────────────────────────────────────────┤
│  Backend-specific training:                                     │
│  • pytorch:  distill_minillm.py  (MiniLLM reverse-KL)          │
│  • mlx:      distill_mlx.py      (Apple Silicon unified mem)   │
│  • unsloth:  distill_unsloth.py  (GPU-optimized 2x speedup)    │
│                                                                  │
│  Outputs:                                                        │
│  • Trained model (HF format or MLX .npz)                        │
│  • metrics.jsonl (loss, perplexity curves)                      │
│  • trainer_state.json (live training state)                     │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Evaluation                                            │
├─────────────────────────────────────────────────────────────────┤
│  • run_eval.py:        Cross-entropy on validation set          │
│    → eval_perplexity, teacher comparison, quant comparison      │
│                                                                  │
│  • eval_quality.py:    Generation quality metrics               │
│    → distinct-1/2 (diversity), max-rep (repetition loops)       │
│    → LLM-as-judge scoring (--judge)                             │
│                                                                  │
│  • run_benchmarks.py:  WikiText-2 perplexity (--benchmarks)     │
│    → Detect catastrophic forgetting                             │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Export (--export all)                                 │
├─────────────────────────────────────────────────────────────────┤
│  • GGUF:    llama.cpp format for llama-server                   │
│    → f16, q8_0, q4_K_M quantization                             │
│                                                                  │
│  • CoreML:  .mlpackage for Apple Neural Engine                  │
│    → iPhone/macOS hardware acceleration                         │
│    → Optional int4/int8/float16 post-training quant             │
│                                                                  │
│  • MLX:     Quantized .npz for Apple Silicon                    │
│    → 4-bit or 8-bit weights via mlx_lm.convert                  │
└─────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Agentic Feedback Loop                                 │
├─────────────────────────────────────────────────────────────────┤
│  • experiment_log.py writes result to experiment_log.jsonl      │
│  • diagnose() analyzes metrics, suggests fixes:                 │
│    - High ppl_gap → "Try: --temperature 1.5 --epochs 3"         │
│    - Low distinct-1 → "Try: --synthetic_data, raise temp"       │
│    - Poor judge → "Try: --curriculum, larger dataset"           │
│                                                                  │
│  • propose_next() for multi-trial runs:                         │
│    - < 3 runs: Random exploration                               │
│    - ≥ 3 runs: Hill-climbing (move toward best configs)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agentic Hyperparameter Search

### Search Space

```python
_SEARCH_SPACE = {
    "temperature": [0.5, 0.7, 1.0, 1.3, 1.5, 2.0],  # KD temperature
    "lora_r":      [16, 32, 64, 128],                # LoRA rank
    "epochs":      [1, 2, 3, 4],                     # Training epochs
}
```

### Multi-Trial Flow

```bash
# Run 3 trials with autonomous hyperparameter tuning
python scripts/run_distillation_agent.py \
    --open \
    --n_trials 3 \
    --epochs 2 \
    --export gguf \
    --log_experiment
```

**What happens:**

1. **Trial 1**: Base config `{temperature: 1.0, lora_r: 64, epochs: 2}`
   - Trains → Evaluates → Records `eval_perplexity: 8.5`

2. **Trial 2**: `propose_next()` suggests `{temperature: 1.5, lora_r: 128, epochs: 2}`
   - (If history shows high temp helped previous runs)
   - Trains → Evaluates → Records `eval_perplexity: 7.2`

3. **Trial 3**: `propose_next()` hill-climbs toward best range
   - Trains → Evaluates → Records `eval_perplexity: 6.8`

4. **Winner Selection**: Trial 3 has lowest perplexity (6.8)
   - Exports Trial 3 to GGUF
   - Logs Trial 3 to `experiment_log.jsonl`
   - Shows diagnostic: `[OK] Perplexity gap 15.3% (acceptable)`

---

## Backend Comparison

| Backend   | Speed     | Memory   | Quantization | Best For                  |
|-----------|-----------|----------|--------------|---------------------------|
| `pytorch` | Baseline  | High     | Post-export  | General use, reproducible |
| `mlx`     | 2-3x      | Low      | During train | Apple Silicon, large models|
| `unsloth` | 2x (GPU)  | Medium   | Post-export  | CUDA GPUs, fast iteration |

**MLX Advantages (Apple Silicon):**
- Unified memory (shares RAM with GPU)
- Native `.npz` weights (no conversion needed)
- On-the-fly quantization during training
- Fast iteration for hyperparameter search

---

## Export Format Comparison

### GGUF (llama.cpp)
```bash
# Serve locally
cd llama.cpp
./build/bin/llama-server -m ../distill/distilled-minillm/model-q4_K_M.gguf
```

**Use Cases:**
- Cross-platform CPU/GPU inference
- Large-scale deployment (TGI, vLLM compatible)
- Embedded systems (Raspberry Pi, etc.)

### CoreML (.mlpackage)
```swift
// iOS/macOS app
import CoreML
let model = try! distilled_model_coreml_int4(configuration: .init())
let output = try! model.prediction(input: tokens)
```

**Use Cases:**
- iPhone/iPad apps with Neural Engine acceleration
- macOS native apps
- On-device privacy (no server needed)

### MLX (.npz)
```python
from mlx_lm import load, generate
model, tokenizer = load("./distilled-minillm-mlx-4bit")
generate(model, tokenizer, prompt="...", max_tokens=100)
```

**Use Cases:**
- Apple Silicon development
- Fast local iteration
- Memory-constrained M1/M2/M3 Macs

---

## Monitoring & Observability

### Live Training Monitoring

**Terminal 1: Watchdog (plateau detection)**
```bash
python scripts/training_watchdog.py \
    ./distilled-minillm \
    --config configs/watchdog_rules.json
```

**Terminal 2: Thermal monitoring + fan control**
```bash
python scripts/monitor_cpu_gpu_temp.py \
    --interval 10 \
    --log thermal.log \
    --fan-control
```

**Terminal 3: Training**
```bash
python scripts/run_distillation_agent.py --open --watchdog --epochs 2
```

**Terminal 4: Dashboard (http://127.0.0.1:7860)**
```bash
python scripts/dashboard.py --runs_dir . --port 7860
```

### Dashboard Tabs

| Tab         | Data Source                | Updates  | Purpose                        |
|-------------|----------------------------|----------|--------------------------------|
| Plots       | `metrics.jsonl`            | Live     | Training/eval loss curves      |
| Pipeline    | `trainer_state.json`, GGUF | On save  | End-to-end summary             |
| Thermal     | `thermal.log`              | Live     | CPU/GPU temps, power, fan RPM  |
| Evaluate    | Auto-discovered models     | On load  | Interactive generation testing |
| Quality     | `quality_metrics.json`     | Post-run | Diversity + judge scores       |
| Experiments | `experiment_log.jsonl`     | Post-run | Multi-run history & trends     |

---

## Air-Gapped Operation

**Phase 1: Cache (with internet)**
```bash
python scripts/setup_airgap.py --open
```

**Phase 2: Run offline**
```bash
export HF_HUB_OFFLINE=1
python scripts/run_distillation_agent.py --open --offline [args]
```

**Phase 3: Transfer to isolated machine**
```bash
tar -czf airgap_cache.tar.gz ~/.cache/huggingface/
# Transfer airgap_cache.tar.gz via USB/airgap
```

See: `AIRGAP_SETUP.md` for complete guide.

---

## Configuration Files

### `configs/agent_config.json`
```json
{
  "output_dir": "./distilled-minillm",
  "open": true,
  "offline": true,
  "backend": "pytorch",
  "export": "gguf",
  "epochs": 2,
  "max_samples": 2000,
  "temperature": 1.0,
  "lora_r": 64
}
```

### `configs/watchdog_rules.json`
```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "lr_scale": 0.8
  },
  "thermal": {
    "enabled": true,
    "pause_if_over": 90,
    "metric": "soc_temp_c"
  }
}
```

---

## Advanced Features

### Curriculum Learning (SFT warmup)
```bash
python scripts/run_distillation_agent.py \
    --open \
    --curriculum \          # SFT warmup before KD
    --sft_epochs 1 \
    --epochs 2
```

### Synthetic Data Augmentation
```bash
python scripts/run_distillation_agent.py \
    --open \
    --synthetic_data \
    --n_synthetic 2000      # Generate 2000 synthetic pairs
```

### Teacher Comparison
```bash
python scripts/run_distillation_agent.py \
    --open \
    --compare_teacher \     # Log teacher perplexity gap
    --benchmarks            # Add WikiText-2 benchmark
```

### Regression Detection
```bash
python scripts/run_distillation_agent.py \
    --open \
    --baseline_dir ./previous-run \
    --benchmarks            # Compare against baseline
```

---

## Research Context

This implements the **2025-2026 "agentic knowledge distillation" pattern** where:

1. **Autonomous ML Engineering**: Agent proposes hyperparameters based on historical performance (hill-climbing + random exploration)

2. **Self-Improving Loop**: Each run informs the next via `experiment_log.jsonl` and `propose_next()`

3. **Multi-Metric Optimization**: Not just perplexity, but diversity (distinct-1/2), instruction-following (LLM-as-judge), and generalization (WikiText-2)

4. **Diagnostic Feedback**: Human-readable suggestions ("Try: --temperature 1.5") bridge agent and engineer

5. **Multi-Format Export**: Single distillation → deploy anywhere (llama.cpp servers, iOS apps, local MLX)

---

## Example: Complete Autonomous Run

```bash
# Let the agent find best config over 5 trials
python scripts/run_distillation_agent.py \
    --open --offline \
    --n_trials 5 \
    --export all \
    --compare_teacher \
    --benchmarks \
    --log_experiment \
    --watchdog

# Agent will:
# 1. Run trial_00 with base config
# 2. Evaluate → diagnose → propose trial_01 config
# 3. Repeat for trial_02, trial_03, trial_04
# 4. Select winner (lowest perplexity)
# 5. Export winner to GGUF + CoreML + MLX
# 6. Log all trials to experiment_log.jsonl
# 7. Print diagnostic: [OK/WARN/ERROR] with suggestions

# View results
python scripts/dashboard.py --runs_dir ./distilled-minillm

# Check experiment history
python scripts/experiment_log.py --show 10
```

---

## File Structure

```
distill/
├── scripts/
│   ├── run_distillation_agent.py    # Orchestrator (autonomous agent)
│   ├── experiment_log.py             # Agentic feedback loop
│   ├── distill_minillm.py            # PyTorch MiniLLM backend
│   ├── distill_mlx.py                # MLX backend (Apple Silicon)
│   ├── distill_unsloth.py            # Unsloth backend (GPU 2x speedup)
│   ├── run_eval.py                   # Perplexity evaluation
│   ├── eval_quality.py               # Diversity + judge metrics
│   ├── run_benchmarks.py             # WikiText-2 benchmark
│   ├── export_coreml.py              # CoreML .mlpackage export
│   ├── dashboard.py                  # Gradio monitoring UI
│   ├── training_watchdog.py          # Plateau detection
│   ├── monitor_cpu_gpu_temp.py       # Thermal + fan control
│   └── setup_airgap.py               # Air-gapped caching
│
├── configs/
│   ├── agent_config.json             # Default agent settings
│   └── watchdog_rules.json           # Plateau/thermal thresholds
│
├── experiment_log.jsonl              # Multi-run history (agentic memory)
├── ARCHITECTURE.md                   # This file
├── STARTUP_GUIDE.md                  # Terminal startup sequence
└── AIRGAP_SETUP.md                   # Offline operation guide
```

---

## Key Design Principles

1. **Autonomous by Default**: Minimal human intervention after initial config
2. **Multi-Backend**: Choose speed vs. compatibility (MLX vs. PyTorch vs. Unsloth)
3. **Multi-Format Export**: One distillation → deploy anywhere
4. **Observability**: Live monitoring (dashboard, watchdog, thermal)
5. **Air-Gapped Ready**: Full offline operation for secure environments
6. **Self-Improving**: Historical performance guides future runs

This is a production-ready autonomous ML system, not just a research prototype. 🚀
