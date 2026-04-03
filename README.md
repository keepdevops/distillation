# Distill — On-Device LLM Knowledge Distillation

A bare-metal knowledge distillation toolkit for compressing large language models
on Apple Silicon (M1/M2/M3) and NVIDIA GPUs. Supports multiple training backends,
agentic hyperparameter search, domain-specialist CoT distillation, thermal protection,
and multi-format export to GGUF, CoreML, and MLX quantized.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Hardware Requirements](#hardware-requirements)
4. [Quick Start](#quick-start)
5. [Installation](#installation)
6. [Training Backends](#training-backends)
   - [MLX — Forward KL (Recommended for Apple Silicon)](#mlx--forward-kl)
   - [MiniLLM — Reverse KL / GRPO](#minillm--reverse-kl--grpo)
   - [SFT Warmup (Stage 1)](#sft-warmup-stage-1)
   - [Forward KD for Classification](#forward-kd-for-classification)
7. [Orchestrator — Autonomous Pipeline](#orchestrator--autonomous-pipeline)
8. [Data Pipeline](#data-pipeline)
   - [Dataset Formats](#dataset-formats)
   - [Quality Filters](#quality-filters)
   - [Magpie Synthesis](#magpie-synthesis)
   - [Dataset Filtering](#dataset-filtering)
9. [Expert Pipeline — Domain Distillation with CoT](#expert-pipeline--domain-distillation-with-cot)
10. [Evaluation](#evaluation)
    - [Generation Quality Metrics](#generation-quality-metrics)
    - [Validation Loss](#validation-loss)
    - [WikiText-2 Benchmarks](#wikitext-2-benchmarks)
11. [Monitoring & Protection](#monitoring--protection)
    - [Thermal Agent](#thermal-agent)
    - [Training Watchdog](#training-watchdog)
12. [Export](#export)
    - [GGUF (llama.cpp)](#gguf-llamacpp)
    - [CoreML (Apple Neural Engine)](#coreml-apple-neural-engine)
    - [MLX Quantized](#mlx-quantized)
13. [Gradio UIs](#gradio-uis)
    - [Distillation Launcher](#distillation-launcher)
    - [Universal Model Evaluator](#universal-model-evaluator)
14. [Algorithm Reference](#algorithm-reference)
15. [Configuration Reference](#configuration-reference)
16. [Air-Gap Mode](#air-gap-mode)
17. [Multi-User Shared Storage](#multi-user-shared-storage)
18. [Session Management](#session-management)
19. [Output Artifacts](#output-artifacts)
20. [Troubleshooting](#troubleshooting)
21. [Project Structure](#project-structure)

---

## Overview

Distill compresses a larger **teacher** model into a smaller **student** model through
knowledge distillation. The student learns to mimic the teacher's output distribution
rather than fitting hard labels, capturing nuance that simple fine-tuning cannot.

**Key capabilities:**

| Feature | Details |
|---------|---------|
| Training methods | Forward KL (MLX), Reverse KL / GRPO (MiniLLM), SFT, Classification KD |
| Backends | MLX (Apple-native), PyTorch/MPS, Unsloth, vLLM |
| Acceleration | 2–5× faster via MLX lazy evaluation; 8–15× faster PPL eval via llama.cpp |
| Data synthesis | Magpie self-synthesis, Self-Instruct, domain CoT generation via GGUF teacher |
| Domain support | Tax, Legal, Medical, Finance, Coding, General |
| Export formats | GGUF (llama.cpp), CoreML (.mlpackage), MLX Q4/Q8 quantized |
| Protection | Thermal agent (hardware temps/power), Training watchdog (plateau/divergence) |
| UI | Two Gradio interfaces — full launcher + standalone evaluator |
| Air-gap | Full offline mode — pre-cache models & datasets, train without network |

---

## Architecture

Long-form design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Documentation index: [docs/INDEX.md](docs/INDEX.md).

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     run_distillation_agent.py                           │
│                       (Autonomous Orchestrator)                          │
└──────────────┬──────────────────────────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Stage 1 (opt.)    │   distill_sft.py
    │   SFT Warmup        │   Teacher labels → student cross-entropy
    └──────────┬──────────┘
               │
    ┌──────────▼──────────────────────────────────────┐
    │   Stage 2 — Choose one:                         │
    │                                                 │
    │   2A/B  distill_mlx.py      Forward KL + CE    │
    │         (MLX, Apple-native, 2–5× faster)        │
    │                                                 │
    │   2C    distill_minillm.py  Reverse KL / GRPO  │
    │         (PyTorch/MPS, Unsloth)                  │
    └──────────┬──────────────────────────────────────┘
               │
    ┌──────────▼──────────┐    ┌─────────────────────┐
    │   Evaluation        │    │   Export             │
    │   run_eval.py       │    │   GGUF               │
    │   eval_quality.py   │    │   CoreML             │
    │   run_benchmarks.py │    │   MLX Q4/Q8          │
    └─────────────────────┘    └─────────────────────┘
               │
    ┌──────────▼─────────────────────────────────────┐
    │   Monitoring (always-on, parallel)             │
    │   thermal_agent.py     Hardware (temp/power)   │
    │   training_watchdog.py ML (plateau/divergence) │
    │   ─── communicate via pause.flag ──────────    │
    └────────────────────────────────────────────────┘
```

**Forward vs. Reverse KL:**

| | Forward KL `D_KL(p_T ‖ p_S)` | Reverse KL `D_KL(p_S ‖ p_T)` |
|-|-------------------------------|-------------------------------|
| Behaviour | Mean-seeking — covers all teacher modes | Mode-seeking — sharpens on best mode |
| Script | `distill_mlx.py` | `distill_minillm.py` |
| Speed | 2–5× faster (MLX) | Baseline (PyTorch) |
| Quality | Good general coverage | Highest single-mode quality |

---

## Hardware Requirements

| Config | Minimum | Recommended |
|--------|---------|-------------|
| Apple Silicon | M1 (8 GB) | M3 Max (36 GB) |
| NVIDIA | 8 GB VRAM + CUDA 11 | 24 GB VRAM + CUDA 12 |
| Disk | 50 GB | 200 GB (models + exports) |
| macOS | 13 Ventura | 15 Sequoia |

**Memory guide for M-series (unified):**

| RAM | Recommended pair |
|-----|-----------------|
| 8 GB | Student 0.5B + Teacher 1.5B (Q4) |
| 16 GB | Student 1B + Teacher 3B (Q4) |
| 36 GB | Student 1–3B + Teacher up to 70B (Q4_K_M) |

**Observed thermal envelope on M3 Max:**

| State | CPU | GPU/SoC |
|-------|-----|---------|
| Idle | ~44°C | ~39°C |
| Under MPS load | ~50°C | ~57°C |
| Watchdog pause threshold | — | 85°C (configurable) |

---

## Quick Start

```bash
# Clone and install
git clone <repo>
cd distill
./scripts/install.sh   # auto-detects Apple Silicon / NVIDIA / CPU

# One-command golden production run (~3 hours on M3 Max)
bash scripts/run.sh golden

# Headless MLX run (recommended for M3 Max, no UI needed)
pixi run python -m distill.run_distillation_agent \
  --open --backend mlx --export all --curriculum --watchdog

# Launch the full Gradio UI
pixi run python -m distill.launch_ui
# → http://127.0.0.1:7861

# Quick smoke test (10 samples, ~5 min)
pixi run python -m distill.run_distillation_agent \
  --open --backend mlx --max_samples 10 --epochs 1 \
  --skip_eval --export none
```

---

## Installation

`scripts/install.sh` detects your hardware and installs the appropriate packages automatically:

```bash
./scripts/install.sh
```

**Detection logic:**

| Detected | Installed |
|----------|-----------|
| macOS arm64 (Apple Silicon) | PyTorch (MPS), MLX + mlx-lm, coremltools, mactop, llama.cpp |
| Linux + `nvidia-smi` | PyTorch cu121/cu118 (auto-selected by `nvcc` version), bitsandbytes |
| Fallback (CPU) | PyTorch CPU build |
| All platforms | transformers, datasets, trl, peft, gradio, evaluate, rich, tqdm, psutil |

The script also:
- Installs [pixi](https://pixi.sh) if missing
- Runs `pixi install` to create the base conda environment
- Prints RAM-aware model size guidance
- Runs a Python verification snippet confirming MPS / CUDA / MLX availability

**Manual install:**

```bash
# Base environment
pixi install

# Apple Silicon
pixi run pip install torch torchvision torchaudio
pixi run pip install mlx "mlx-lm>=0.20" "coremltools>=8.0"

# Common packages
pixi run pip install "transformers>=4.38" "datasets>=2.14" "trl>=0.10" \
  "accelerate>=0.24" peft evaluate gradio rich tqdm psutil
```

---

## Training Backends

### MLX — Forward KL

**Module:** `distill.distill_mlx` (`python -m distill.distill_mlx`)
**Best for:** Apple Silicon — 2–5× faster than PyTorch/MPS

Teacher logits are pre-computed once as top-K sparse tensors (~300 MB vs ~311 GB
full vocab), the teacher is freed from memory, then the student trains on the frozen
cache. Loss is a blend of forward KL and cross-entropy with optional linear annealing
of temperature and CE weight.

```
L = α_CE · L_CE + (1 − α_CE) · L_KD
L_KD = D_KL(p_teacher ‖ p_student)    [forward KL — mean-seeking]
```

```bash
pixi run python -m distill.distill_mlx \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --student Qwen/Qwen2-0.5B \
  --output_dir ./distilled-mlx \
  --epochs 3 --batch_size 2 --grad_acc 8 \
  --lora_r 8 --kd_temp 1.0 --ce_alpha 0.2 \
  --topk_logits 50 --q_bits 4 --watchdog
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher` | Llama-3.2-8B-Instruct | Teacher model (HF hub ID or local path) |
| `--student` | Llama-3.2-1B-Instruct | Student model |
| `--open` | — | Use open Qwen2 models (no HF login) |
| `--epochs` | 2 | Training epochs |
| `--batch_size` | 2 | Physical batch size |
| `--grad_acc` | 4 | Gradient accumulation steps |
| `--lora_r` | 8 | LoRA rank (adapters on Q, K, V, O projections) |
| `--kd_temp` | 1.0 | Distillation temperature (higher = softer targets) |
| `--ce_alpha` | 0.1 | CE loss weight: 0 = pure KD, 1 = pure CE |
| `--topk_logits` | 50 | Top-K teacher logits per token (>99% probability mass) |
| `--q_bits` | 4 | Post-training quantization bits (4 or 8) |
| `--precomp_bs` | auto | Batch size for teacher logit pre-computation |
| `--temp_start/end` | — | Anneal KD temperature linearly over training |
| `--hard_weight_start/end` | — | Anneal CE alpha linearly over training |
| `--multi_turn_ratio` | 0.0 | Fraction of multi-turn (ShareGPT) samples per batch |
| `--resume` | — | Continue from last epoch checkpoint |
| `--watchdog` | — | Honour `pause.flag` for thermal / watchdog control |
| `--offline` | — | Air-gapped mode |

**Outputs:** `mlx_student_weights.npz`, `mlx_q4/` or `mlx_q8/`, `metrics.jsonl`

---

### MiniLLM — Reverse KL / GRPO

**Module:** `distill.distill_minillm` (`python -m distill.distill_minillm`)
**Best for:** Highest quality distillation; PyTorch/MPS backend

Trains via **Group Relative Policy Optimization (GRPO)**: sample G completions per
prompt, score with a reward function, compute group-normalised advantage, clip
importance ratio. Approximates the reverse-KL gradient without full teacher rollouts.

```
Â_i = (r_i − μ_G) / (σ_G + ε)       [group-normalised advantage]
L_GRPO = −1/G Σ_i 1/|y_i| Σ_t min(ρ · Â, clip(ρ, 1±ε) · Â)
Reward: +0.5 (clean EOS)  −0.5 (clipped)  −1.0 (< 10 tokens, collapse)
```

```bash
pixi run python -m distill.distill_minillm \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --student Qwen/Qwen2-0.5B \
  --output_dir ./distilled-minillm \
  --epochs 2 --batch_size 8 --grad_acc 8 \
  --lora_r 64 --learning_rate 2e-5 \
  --num_generations 4 --max_new_tokens 256 \
  --watchdog
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher` | Llama-3.2-8B-Instruct | Teacher model |
| `--student` | Llama-3.2-1B-Instruct | Student model |
| `--open` | — | Use open Qwen2 models |
| `--epochs` | 2 | Training epochs |
| `--batch_size` | 8 | Samples per device step |
| `--grad_acc` | 8 | Gradient accumulation (effective batch = 64) |
| `--lora_r` | 64 | LoRA rank |
| `--learning_rate` | 2e-5 | 10× lower than SFT — GRPO is sensitive to LR |
| `--num_generations` | 4 | Completions sampled per prompt (≥4 for GRPO reward variance) |
| `--max_new_tokens` | 256 | Hard completion length cutoff |
| `--minillm_temp` | 1.0 | KD temperature |
| `--eval_steps` | 20 | Evaluation frequency |
| `--eval_split` | 0.02 | Fraction held for validation |
| `--use_4bit_teacher` | — | Load teacher in 4-bit (saves memory) |
| `--watchdog` | — | Honour `pause.flag` |
| `--offline` | — | Air-gapped mode |

**Outputs:** HF checkpoint, `adapter_model.bin`, `trainer_state.json`, `metrics.jsonl`

**Healthy metrics to watch:**

| Metric | Healthy | Action if outside |
|--------|---------|-------------------|
| `reward` | Trending toward +0.5 | Stuck at −1.0 → increase `max_new_tokens` or add SFT warmup |
| `clipped_ratio` | < 30% | > 60% → lower `max_new_tokens` |
| `frac_reward_zero_std` | < 20% after step 50 | > 50% → increase `num_generations` |
| `kl` | Finite, decreasing | NaN → training diverged, lower LR |

---

### SFT Warmup (Stage 1)

**Module:** `distill.distill_sft` (`python -m distill.distill_sft`)
**Purpose:** Curriculum step before MiniLLM to prevent cold-start reward collapse

Teacher generates greedy completions; student minimises cross-entropy on response
tokens only (prompt + padding masked to −100). Teacher labels are cached to
`sft_labels.jsonl` and reused on reruns.

```bash
pixi run python -m distill.distill_sft \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --student Qwen/Qwen2-0.5B \
  --output_dir ./distilled-sft \
  --epochs 1 --batch_size 4 --lora_r 64
```

**When to use SFT first:**
- MiniLLM reward stuck at −1.0 for > 50 steps
- `frac_reward_zero_std` > 0.5 after step 30
- Using a new domain dataset the student hasn't seen before

After SFT finishes, point MiniLLM's `--student` at `distilled-minillm/sft_checkpoint`.

**Outputs:** `sft_checkpoint/`, `sft_labels.jsonl`

---

### Forward KD for Classification

**Module:** `distill.distill_forward` (`python -m distill.distill_forward`)
**Purpose:** Vanilla temperature-scaled KD for encoder classification models (BERT, DistilBERT)

Combines soft targets (temperature-scaled teacher logits) with hard cross-entropy
on GLUE / custom classification datasets. Not for causal LLMs.

```bash
pixi run python -m distill.distill_forward \
  --teacher bert-base-uncased \
  --student distilbert-base-uncased \
  --dataset glue --dataset_config sst2 \
  --temperature 5.0 --alpha 0.5
```

---

## Orchestrator — Autonomous Pipeline

**Module:** `distill.run_distillation_agent` (`python -m distill.run_distillation_agent`)

Runs all stages end-to-end, streams live metrics, supports multi-trial hyperparameter
search, and logs every run to `experiment_log.jsonl`.

```bash
pixi run python -m distill.run_distillation_agent \
  --open \
  --backend mlx \
  --export all \
  --curriculum \
  --n_trials 3 \
  --synthetic_data \
  --benchmarks \
  --watchdog \
  --config configs/golden_pipeline.json
```

**Key flags:**

| Flag | Description |
|------|-------------|
| `--open` | Use open Qwen2 models (no HF login) |
| `--backend` | `mlx` \| `pytorch` \| `unsloth` |
| `--export` | `gguf` \| `coreml` \| `all` \| `none` |
| `--curriculum` | Run SFT warmup (Stage 1) before KD (Stage 2) |
| `--n_trials` | Hyperparameter sweep — N independent trials, keeps best eval loss |
| `--synthetic_data` | Augment with Magpie-synthesised pairs before training |
| `--benchmarks` | Run WikiText-2 perplexity after training |
| `--watchdog` | Enable loss plateau / divergence detection |
| `--config` | JSON config file; CLI flags override config values |
| `--offline` | Air-gapped mode |
| `--log_experiment` | Append run to `experiment_log.jsonl` with `propose_next()` suggestions |

**Multi-trial search** perturbs `learning_rate`, `batch_size`, `kd_temp`, and
`ce_alpha` across N runs, selects the trial with the lowest final eval loss, and
writes hyperparameter suggestions for the next run.

**Pipeline choice guide:**

| Goal | Recommended command |
|------|---------------------|
| Fastest on M3 Max | `--backend mlx --export all` |
| Highest quality | `--curriculum --backend pytorch --export gguf` |
| Domain expert | `expert_pipeline.py --mode distill` |
| Hyperparameter search | `--n_trials 3 --curriculum` |
| Full production run | `bash scripts/run.sh golden` |

---

## Data Pipeline

### Dataset Formats

**Module:** `distill.data_pipeline` (`python -m distill.data_pipeline`)

Auto-detects schema and normalises to `instruction / input / output` triples.

| Schema | Detection | Example datasets |
|--------|-----------|-----------------|
| `alpaca` | `instruction` + `output` fields | `tatsu-lab/alpaca`, `yahma/alpaca-cleaned` |
| `sharegpt` | `conversations` list `{from, value}` | `teknium/OpenHermes-2.5` |
| `messages` | `messages` list `{role, content}` | `HuggingFaceH4/no_robots` |
| `dpo` | `instruction` + `chosen` list | `argilla/distilabel-capybara-dpo-7k-binarized` |
| `guanaco` | `text` with `### Human:` markers | `mlabonne/guanaco-llama2-1k` |
| Local JSONL | Any `.jsonl` with the above fields | Custom domain data |

**Multi-turn support:** ShareGPT / messages format normalised to ChatML, capped at
4 turns. Use `--multi_turn_ratio` in `distill_mlx.py` to blend multi-turn samples
into each batch.

### Quality Filters

Applied automatically during dataset loading:

| Filter | Behaviour |
|--------|-----------|
| Refusal detection | Removes "I'm sorry", "I cannot", "I'm not able", etc. |
| Noise detection | Removes "N/A", "none", "I don't know", `[...]` |
| Distinct-2 gate | Removes responses with bigram diversity < 0.35 |
| Length gate | Removes responses < 20 or > 600 words (configurable) |

### Magpie Synthesis

**Module:** `distill.magpie_synth` (`python -m distill.magpie_synth`)

Generates synthetic instruction-response pairs by conditioning the teacher on the
chat-template user-turn prefix. The model auto-completes a realistic user question
and then generates the answer. No seed dataset required (Xu et al., 2024).

```bash
pixi run python -m distill.magpie_synth \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --domain math \
  --n 10000 \
  --target 3000 \
  --output_dir ./magpie_data \
  --filter --teacher_score
```

**Key flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--teacher` | Qwen2-1.5B-Instruct | Generator model |
| `--domain` | general | Domain key from `configs/domain_prompts.json` |
| `--n` | 10000 | Pairs to generate before filtering |
| `--target` | — | Keep only top N after filtering |
| `--batch_size` | 32 | Generation batch size |
| `--inst_temp` | 0.9 | Instruction temperature |
| `--resp_temp` | 0.7 | Response temperature |
| `--filter` | — | Run quality filter post-generation |
| `--teacher_score` | — | Use teacher NLL scoring in filter |
| `--resume` | — | Continue from existing `magpie_raw.jsonl` |
| `--offline` | — | Air-gapped mode |

**Built-in domains** (system prompts + filter configs in `configs/domain_prompts.json`):
`general`, `medical`, `math`, `legal`, `tax`, `coding`, `finance`

**Outputs:** `magpie_raw.jsonl`, `hf_dataset/`

### Dataset Filtering

**Module:** `distill.filter_dataset` (`python -m distill.filter_dataset`)

Reduces a raw dataset to a high-quality subset using composite scoring and near-dedup.

```bash
pixi run python -m distill.filter_dataset \
  --dataset teknium/OpenHermes-2.5 \
  --output_dir ./filtered_data \
  --target 8000
```

Scoring criteria: distinct-2, instruction complexity, response variety, Jaccard
near-dedup, optional teacher NLL re-ranking.

---

## Expert Pipeline — Domain Distillation with CoT

**Module:** `distill.expert_pipeline` (`python -m distill.expert_pipeline`)

Builds domain-specialist models (tax, legal, medical, finance, coding) in four steps:
remap a dataset → generate Chain-of-Thought rationales via a GGUF teacher → distill.

```bash
# 1. Inspect dataset columns
pixi run python -m distill.expert_pipeline \
  --mode inspect --dataset nelson-liu/legalbench

# 2. Remap columns to instruction/input/output
pixi run python -m distill.expert_pipeline \
  --mode remap \
  --dataset Atome-LLM/Tax-Policy-Analysis \
  --instruction_col question --output_col answer \
  --output_dir ./domain_data/tax

# 3. Generate CoT rationales via GGUF teacher
pixi run python -m distill.expert_pipeline \
  --mode cot \
  --dataset ./domain_data/tax \
  --teacher /Users/Shared/llama/models/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf \
  --domain tax --n_samples 2000 \
  --output_dir ./domain_data/tax_cot

# 4. Distill on CoT data
pixi run python -m distill.expert_pipeline \
  --mode distill \
  --dataset ./domain_data/tax_cot/hf_dataset \
  --output_dir ./runs/tax-expert \
  --backend mlx --open --lora_r 32 --epochs 3
```

**CoT output format** (teacher wraps every response):
```
<reasoning>
[step-by-step domain analysis]
</reasoning>
<answer>
[final answer]
</answer>
```

**Domain system prompts:**

| Domain | Teacher focus |
|--------|--------------|
| `tax` | IRC sections, thresholds, phase-outs, filing requirements |
| `legal` | Statute / regulation / case law application, statutory tests |
| `medical` | Clinical reasoning, pharmacology, pathophysiology |
| `finance` | Financial concepts, regulations, risk frameworks |
| `coding` | Problem-solving, design patterns, edge cases, complexity |
| `general` | Balanced reasoning and explanation |

**Recommended GGUF teachers:**

| Domain | Model | Why |
|--------|-------|-----|
| Tax / Legal | `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf` | Strongest reasoning; fits 36 GB unified |
| Legal (fast) | `law-chat.Q4_K_M.gguf` | Statutory interpretation specialist |
| Medical | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Best 8B medical reasoning |
| Coding | `granite-3.1-8b-instruct-Q4_K_M.gguf` | IBM Granite code quality |
| General | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Fast, general purpose |

All GGUF files go to `/Users/Shared/llama/models/`.

**CoT generation flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--n_samples` | 1000 | CoT examples to generate |
| `--temperature` | 0.3 | Lower = more precise domain reasoning |
| `--max_tokens` | 1024 | Max CoT response length |
| `--ctx_size` | 8192 | llama-server context window |
| `--n_parallel` | 4 | Parallel server slots |

**Outputs:** `remapped.jsonl`, `cot_data.jsonl`, `hf_dataset/`
**Logs:** `runs/ep_<stage>_<timestamp>.log` + `.jsonl`

---

## Evaluation

### Generation Quality Metrics

**Module:** `distill.eval_quality` (`python -m distill.eval_quality`)

Generates N completions and measures diversity, quality, and instruction-following.

```bash
pixi run python -m distill.eval_quality ./distilled-mlx \
  --judge \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --judge-teacher-ppl \
  --n_samples 200
```

| Metric | Good threshold | Description |
|--------|---------------|-------------|
| `distinct_1` | > 0.3 | Unique unigram fraction |
| `distinct_2` | > 0.5 | Unique bigram fraction |
| `3-gram entropy` | > 8.0 | Cross-output generation variety |
| `refusal_rate` | < 5% | Fraction of refusal outputs |
| `quality_gate_pass` | > 90% | Length + refusal filter pass rate |
| `judge_score` (1–10) | > 7.0 | LLM-as-judge instruction-following |
| `teacher_ppl` | lower = better | Per-sample teacher perplexity on student outputs |

Results saved to `quality_metrics.json`.

**Recommended eval sequence:**
```
1. run_eval.py         → quick perplexity sanity check (~2 min)
2. eval_quality.py     → generation diversity check (~5 min)
3. run_benchmarks.py   → WikiText-2 standardised comparison (~5 min)
4. eval_quality.py --judge  → final LLM-as-judge quality gate (~15 min)
```

### Validation Loss

**Module:** `distill.run_eval` (`python -m distill.run_eval`)

Computes cross-entropy on the validation split. Skipped for MLX backend (handled
internally by `distill_mlx.py`).

```bash
pixi run python -m distill.run_eval ./distilled-minillm \
  --backend auto --compare_teacher --max_val_samples 500
```

**Perplexity guide:**

| PPL | Assessment |
|-----|-----------|
| < 5 | Excellent |
| 5–15 | Good — typical well-distilled small model |
| 15–30 | Fair — consider more epochs or better data |
| > 30 | Poor — check training config |

### WikiText-2 Benchmarks

**Module:** `distill.run_benchmarks` (`python -m distill.run_benchmarks`)

Evaluates on WikiText-2-raw-v1 (500 sequences by default). A >15% PPL increase over
a baseline triggers a regression warning.

```bash
pixi run python -m distill.run_benchmarks ./distilled-mlx \
  --baseline_dir ./reference-model --n_sequences 500
```

**Reference numbers:**

| Model | WikiText-2 PPL |
|-------|---------------|
| Qwen2-0.5B-Instruct (no distillation) | ~18–22 |
| Well-distilled student (1.5B teacher) | ~14–18 |
| Qwen2-1.5B-Instruct (teacher) | ~10–13 |

---

## Monitoring & Protection

Two independent monitors run in parallel and communicate through a shared `pause.flag`
file. Every training loop checks for this flag before each step and responds with a
clean checkpoint-and-pause — no data is lost.

### Thermal Agent

**Module:** `distill.thermal_agent` (`python -m distill.thermal_agent`)

Polls `mactop` for CPU/GPU/SoC temperatures and power draw every N seconds. Writes
`pause.flag` to all watched directories when any metric exceeds the threshold, then
clears it automatically once temperatures drop by the hysteresis delta.

```bash
# One-time session
pixi run python -m distill.thermal_agent \
  --watch ./distilled-mlx ./distilled-minillm \
  --threshold 85 --metric soc_temp_c --interval 30

# Persistent — install as macOS LaunchAgent (survives reboot)
bash scripts/install_thermal_agent.sh
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--watch` | required | Directories to protect |
| `--threshold` | 85°C | Pause trigger |
| `--metric` | `soc_temp_c` | `soc_temp_c` \| `cpu_temp_c` \| `gpu_temp_c` \| `total_power_w` |
| `--hysteresis` | 5°C | Temperature delta required to resume |
| `--interval` | 30 s | Poll interval |
| `--log_file` | — | Optional thermal event log |
| `--daemon` | — | Run as background process |

Uses `mactop` (`/opt/homebrew/bin/mactop`) — M3 Max compatible, no `sudo` required.
Provides: `cpu_temp`, `gpu_temp`, `soc_temp`, `cpu_power`, `gpu_power`, `total_power`.

### Training Watchdog

**Module:** `distill.training_watchdog` (`python -m distill.training_watchdog`)

Reads `trainer_state.json` every N seconds. Writes `pause.flag` on:

- **Plateau:** last-N loss deltas all < `max_delta` (default 0.001)
- **Divergence:** recent avg loss > early baseline × 1.5

```bash
pixi run python -m distill.training_watchdog ./distilled-mlx \
  --interval 60 --config scripts/watchdog_rules.json
```

**Default rules** (`configs/watchdog_rules.json`):

```json
{
  "plateau":    { "window": 3, "max_delta": 0.001, "min_points": 5,  "lr_scale": 0.8 },
  "divergence": { "window": 3, "threshold": 1.5,   "baseline_window": 5, "min_points": 8 },
  "validator":  { "backup_before_write": true, "max_lr_scale": 0.5 }
}
```

Writes `watchdog_suggestions.json` with learning rate scaling recommendations.

---

## Export

### GGUF (llama.cpp)

```bash
bash scripts/export_student_gguf.sh ./distilled-minillm
# → /Users/Shared/llama/models/distilled-minillm-Q4_K_M.gguf
```

Calls `convert_hf_to_gguf.py` from llama.cpp (`/Users/Shared/llama/`). Runs inside
`tmux` with `caffeinate` to prevent sleep. Set `OUTTYPE=q8_0` to change precision.

### CoreML (Apple Neural Engine)

```bash
pixi run python -m distill.export_coreml \
  --model_dir ./distilled-minillm \
  --quantize int4 \
  --compute_units CPU_AND_NE \
  --output_dir ./coreml_export
# → ./coreml_export/model.mlpackage + Swift snippet
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model_dir` | required | HF model directory |
| `--quantize` | — | `int4` \| `int8` \| `float16` |
| `--seq_len` | 128 | Sequence length for tracing |
| `--compute_units` | `CPU_AND_NE` | `ALL` \| `CPU_ONLY` \| `CPU_AND_GPU` \| `CPU_AND_NE` |

### MLX Quantized

Pass `--q_bits 4` or `--q_bits 8` to `distill_mlx.py` — quantization runs
automatically at the end of training and saves to `mlx_q4/` or `mlx_q8/`.

```bash
# Manual conversion
pixi run mlx_lm.convert \
  --hf-path ./distilled-minillm \
  --mlx-path ./distilled-minillm-mlx \
  -q --q-bits 4
```

**Export format comparison:**

| Format | Size (1B model) | Inference speed | Use case |
|--------|----------------|-----------------|---------|
| GGUF Q4_K_M | ~700 MB | Fastest (llama.cpp, Metal) | Production inference, Ollama |
| MLX Q4 | ~700 MB | Fast (Apple-native) | On-device Apple Silicon |
| CoreML int4 | ~700 MB | Very fast (ANE) | iOS / macOS app embedding |
| SafeTensors (fp16) | ~2 GB | Baseline | Fine-tuning, LoRA merge |

---

## Gradio UIs

### Distillation Launcher

**Module:** `distill.launch_ui` (`python -m distill.launch_ui`)

Full-featured parameter form for launching, monitoring, and evaluating distillation
runs from a browser — no terminal required.

```bash
pixi run python -m distill.launch_ui
# → http://127.0.0.1:7861
```

**Seven tabs:**

| Tab | Purpose |
|-----|---------|
| **Configure & Launch** | All training parameters with backend-aware defaults |
| **Data Prep** | Magpie synthesis, Self-Instruct generation, Dataset Filter |
| **Domain Synthesis** | Domain-specific Magpie with domain selector + system prompt editor |
| **Eval** | Perplexity, Quality Eval, WikiText-2 benchmark runner |
| **Expert Pipeline** | 4-step CoT workflow: inspect → remap → CoT generation → distill |
| **Live Logs** | Real-time log stream + training loss / gradient norm charts |
| **Help** | Full operation guide, parameter reference, algorithm reference |

**UI features:**
- Progress bar on every tab (parses `Step X/Y`, `Epoch X/Y`, tqdm `75%|`)
- Live loss and gradient norm charts updated every 2 seconds
- Stop button (SIGKILL) — last checkpoint always preserved
- Refresh buttons for model / dataset dropdowns (auto-scans HF cache + local dirs)
- Backend toggle auto-updates `batch_size`, `grad_acc`, `lora_r` defaults
- All runs log to `runs/` with paired `.log` and `.jsonl` files

### Universal Model Evaluator

**Module:** `distill.eval_gradio` (`python -m distill.eval_gradio`)

Standalone evaluator with auto-format detection — works with PyTorch, MLX, GGUF,
and vLLM backends.

```bash
pixi run python -m distill.eval_gradio
pixi run python -m distill.eval_gradio --model_path ./distilled-mlx --backend mlx
pixi run python -m distill.eval_gradio --model_path /Users/Shared/llama/models/model.gguf
# → http://127.0.0.1:7860
```

**Five tabs:**

| Tab | Purpose |
|-----|---------|
| **Model Info** | Artifact summary, training method, formats, checkpoint list |
| **Generate** | Interactive generation with temperature + max-token sliders |
| **Batch Eval** | Quality eval command reference for the loaded model |
| **Algorithms** | LaTeX algorithm reference rendered via MathJax |
| **Help** | Full pipeline reference + appended algorithm reference |

**Format auto-detection priority:**
1. File ends in `.gguf` → GGUF
2. Dir contains `mlx_model.npz` / `mlx_student_weights.npz` → MLX
3. Dir contains `mlx_q4/` subdir → MLX
4. Dir contains `adapter_config.json` + `*.safetensors` → PyTorch (LoRA)
5. Dir contains `*.safetensors` / `pytorch_model.bin` → PyTorch

**Backend speed on M3 Max:**

| Backend | Speed | Memory | Best for |
|---------|-------|--------|---------|
| **gguf** | 8–15× baseline | ~2 GB Q4 | Production inference, PPL benchmarks |
| **mlx** | 3–5× baseline | 4–8 GB | Training + eval on Apple Silicon |
| **pytorch** | baseline | 4–16 GB | Full fine-tuning, LoRA merge |
| **vllm** | 5–10× baseline | NVIDIA VRAM | High-throughput GPU serving |

---

## Algorithm Reference

Rendered with LaTeX / MathJax in both Gradio UIs (Help tab + Algorithms tab).
Run standalone to open an interactive browser view:

```bash
pixi run python -m distill.show_algorithms
pixi run python -m distill.show_algorithms --output algorithms.html  # save only
pixi run python -m distill.show_algorithms --latex algorithms.tex    # export LaTeX source
```

**Covered algorithms:**

| # | Algorithm | Script |
|---|-----------|--------|
| 1 | Pipeline Overview — forward vs. reverse KL | `run_distillation_agent.py` |
| 2 | Stage 1 — SFT Warmup (response-only cross-entropy) | `distill_sft.py` |
| 3 | Stage 2A/B — Forward KL + CE with top-K sparse logits + annealing | `distill_mlx.py` |
| 4 | Stage 2C — Reverse KL / GRPO with group-normalised advantage | `distill_minillm.py` |
| 5 | LoRA Parameterization (`h = W₀x + (α_r/r)·B(Ax)`) | all backends |
| 6 | AdamW + Cosine LR with 3% linear warmup | all backends |

---

## Configuration Reference

All configs live in `configs/`.

### `configs/golden_pipeline.json`

Full production run config — loaded by `scripts/run.sh golden`. Edit to change dataset,
epochs, LoRA rank, export targets, etc.

### `configs/mlx_recommended.json`

Recommended MLX settings for M3 Max (36 GB):

```json
{
  "backend": "mlx",
  "export": "gguf",
  "epochs": 3,
  "max_samples": 2000,
  "batch_size": 2,
  "grad_acc": 8,
  "lora_r": 16,
  "ce_alpha": 0.2,
  "topk_logits": 50,
  "q_bits": 4
}
```

### `configs/agent_config.json`

Default PyTorch agent config:

```json
{
  "output_dir": "./distilled-minillm",
  "open": true,
  "backend": "pytorch",
  "export": "gguf",
  "epochs": 2,
  "batch_size": 8,
  "grad_acc": 8
}
```

### `configs/watchdog_rules.json`

Plateau / divergence detection thresholds (see [Training Watchdog](#training-watchdog)).

### `configs/domain_prompts.json`

Magpie domain registry: per-domain system prompts and filter configs
(`min_response_words`, `max_response_words`, `min_distinct2`, `require_code`,
`require_numbers`) for `general`, `medical`, `math`, `legal`, `tax`, `coding`, `finance`.

**Using a config with the agent:**
```bash
pixi run python -m distill.run_distillation_agent --config configs/golden_pipeline.json
# CLI flags supplied alongside --config override the file values
```

---

## Air-Gap Mode

Train with zero network access after a one-time download.

**Step 1 — Pre-cache (with internet):**
```bash
# All-in-one
pixi run python -m distill.setup_airgap --open

# Or separately
pixi run python -m distill.cache_models     # → ~/.cache/huggingface/
pixi run python -m distill.cache_datasets  # → datasets_cache/
```

**Step 2 — Set offline env vars:**
```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Or add to ~/.zshrc for persistence
```

**Step 3 — Train offline:**
```bash
pixi run python -m distill.run_distillation_agent \
  --open --offline --backend mlx --epochs 2
```

To transfer a cache to an air-gapped machine, copy `~/.cache/huggingface/` and
`datasets_cache/` to the target host before running.

---

## Multi-User Shared Storage

Share models and GGUF files across macOS user profiles.

```bash
# Primary setup
bash scripts/setup_shared_models.sh
# → Sets MODEL_PATH=/Users/Shared/models in ~/.zshrc

# For additional users on the same Mac
cd /path/to/distill && bash scripts/setup_shared_models.sh
```

**Shared locations:**

| Path | Contents |
|------|---------|
| `/Users/Shared/models/` | HF model cache, distilled checkpoints |
| `/Users/Shared/llama/models/` | GGUF files for llama.cpp / Ollama |
| `/Users/Shared/llama/llama-server` | llama-server binary |

---

## Session Management

```bash
# Start background services in tmux (recommended for long runs)
bash scripts/start.sh                # watchdog + dashboard
bash scripts/start.sh --monitor      # + thermal agent
bash scripts/start.sh --eval         # use eval_gradio instead of dashboard

# Attach to running session
tmux attach -t distill

# Stop everything cleanly
bash scripts/stop.sh

# Kill UI only
bash scripts/kill_ui.sh
```

`start.sh` launches `caffeinate` to prevent the machine sleeping during long runs and
opens three tmux windows: **watchdog**, **dashboard/eval**, and optionally **thermal**.

---

## Output Artifacts

After a full pipeline run your output directory contains:

```
your-model/
├── config.json                  # HuggingFace model config
├── *.safetensors                # PyTorch weights (merged LoRA)
├── adapter_config.json          # LoRA config (if not merged)
├── adapter_model.bin            # LoRA weights (if not merged)
├── mlx_student_weights.npz      # MLX LoRA weights
├── mlx_q4/                      # MLX 4-bit quantized
├── model-q4_K_M.gguf            # GGUF 4-bit (llama.cpp)
├── model.mlpackage/             # CoreML (Apple Neural Engine)
├── metrics.jsonl                # Loss, LR, grad norm, eval PPL per step
├── trainer_state.json           # HF Trainer checkpoint state
├── quality_metrics.json         # eval_quality.py results
├── benchmark_results.json       # WikiText-2 perplexity
├── sft_labels.jsonl             # Cached teacher greedy completions (Stage 1)
├── watchdog_suggestions.json    # Watchdog LR scaling recommendations
└── experiment_log.jsonl         # Multi-trial history + propose_next() suggestions
```

---

## Troubleshooting

**Training is slow (> 300 s/iter)**
- Reduce `--max_new_tokens` to 64–96 — generation is the bottleneck
- Reduce `--num_generations` to 2
- Switch to `--backend mlx` (2–5× faster on M-series)

**Reward stuck at −1.0 (mode collapse)**
- Increase `--max_new_tokens` — student is hitting the hard limit before EOS
- Raise `--minillm_temp` to 1.5 (softer targets)
- Add SFT warmup: use `--curriculum` flag

**`clipped_ratio` > 80%**
- Lower `--max_new_tokens` to 64 or 96

**Loss NaN or exploding**
- Learning rate too high — lower 10×: `2e-5 → 2e-6`
- If `grad_norm` > 10 before the loss spike, LR is the cause

**`frac_reward_zero_std` > 0.6 after step 20**
- All completions get identical reward → GRPO gradient is zero
- Increase `--num_generations` to 4–8

**`MPS backend out of memory`**
- Reduce `--batch_size` (try 1 or 2) and `--max_new_tokens` (try 64)

**MLX → numpy float16 `PEP 3118` error**
```python
# Wrong:  np.array(arr, dtype=np.float16)
# Correct:
np.array(arr.astype(mx.float32)).astype(np.float16)
```

**Cosine scheduler `ZeroDivisionError` when `total_steps=1`**
```python
decay_steps = max(1, total_steps - warmup_steps)
```

**`mx.metal.clear_cache()` deprecated**
Use `mx.clear_cache()` (MLX ≥ 0.18).

**`No module named 'trl'` / `gradio`**
```bash
pixi run pip install trl transformers peft datasets gradio
# Or relaunch: pixi run python -m distill.launch_ui
```

**Thermal pauses too frequent**
- Raise `--threshold` to 90°C (observed M3 Max peaks under MPS load: GPU 57°C)
- Increase `--interval` to 60 s to reduce overhead

---

## Project Structure

Python sources live in the installable **`distill/`** package (`pip install -e .` / Pixi `pypi-dependencies`). Run CLIs with **`python -m distill.<module>`** or the console scripts from `pyproject.toml` (`distill-agent`, `distill-ui`, …). See [docs/INDEX.md](docs/INDEX.md).

```
├── distill/                       Python package (training, eval, UIs)
│   ├── launch_ui/                 Gradio launcher (split subpackage)
│   ├── data_pipeline.py           Dataset loading & formatting
│   ├── run_distillation_agent.py  Autonomous orchestrator
│   └── …                          See repo tree
├── scripts/                       Shell helpers (.sh), LaunchAgent plist, etc.
│   ├── install.sh                 Hardware-aware installer (Apple Silicon / NVIDIA / CPU)
│   ├── run.sh                     Distillation runs: golden, production, phase2, smoke, download, export
│   └── services.sh                Service management: start, stop, monitor
├── docs/                          Guides (see docs/INDEX.md)
├── configs/
│   ├── golden_pipeline.json       Production run config
│   ├── mlx_recommended.json       MLX recommended settings
│   ├── agent_config.json          Default agent config
│   ├── watchdog_rules.json        Plateau / divergence rules
│   └── domain_prompts.json        Magpie domain registry
├── pyproject.toml                 setuptools / console entry points
├── pixi.toml                      Conda + editable local package
└── requirements.txt               Pip dependency list
```
