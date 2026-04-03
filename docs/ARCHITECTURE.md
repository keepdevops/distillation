# Distillation Pipeline Architecture

## Overview

This is an **autonomous agentic knowledge distillation system** designed for Apple M3 hardware. An orchestrator agent acts as an autonomous ML engineer, running full distillation pipelines, evaluating results, and self-improving via historical performance analysis.

The pipeline has five stages, plus an orthogonal set of optimization phases and quality gate phases that cross-cut the pipeline.

---

## System Design Principles

1. **Autonomous by Default** — After initial config, the system runs unattended from data loading to deployment-ready export
2. **Multi-Backend** — PyTorch/MPS, MLX (2–5× faster), or Unsloth; choose at runtime
3. **Multi-Format Export** — One distillation → GGUF (llama.cpp), CoreML (.mlpackage, ANE), and MLX quantized weights
4. **Observability** — Live dashboard, thermal monitoring, plateau detection, experiment history
5. **Air-Gapped Ready** — Full offline operation; all dependencies packageable via conda-pack
6. **Self-Improving** — Each run's results are logged and feed into the next run's hyperparameter proposal

---

## Five-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 0: Preparation (optional)                                │
│  • Cache models/datasets offline    (cache_models.py)           │
│  • Generate synthetic data          (generate_synthetic_data.py)│
│  • SFT warmup / curriculum          (distill_sft.py)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Distillation Core                                     │
│  pytorch  → distill_minillm.py   (MiniLLM reverse-KL + TRL)    │
│  mlx      → distill_mlx.py       (MLX native, unified memory)  │
│  unsloth  → distill_unsloth.py   (optimized LoRA kernels)       │
│  Output: trained weights + metrics.jsonl + trainer_state.json   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Evaluation                                            │
│  • run_eval.py         → validation perplexity + teacher gap    │
│  • eval_quality.py     → quality gates, diversity, judge score  │
│  • run_benchmarks.py   → WikiText-2 perplexity                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 3: Export  (--export all)                                │
│  • GGUF      → llama.cpp convert → f16 / q8_0 / q4_K_M         │
│  • CoreML    → coremltools → .mlpackage (ANE / on-device)       │
│  • MLX       → mlx_lm.convert → quantized .npz (4-bit / 8-bit) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Stage 4: Agentic Feedback Loop                                 │
│  • experiment_log.py    → append run record to JSONL            │
│  • diagnose()           → human-readable suggestions            │
│  • propose_next()       → next hyperparameter set               │
│  • Winner selection     → lowest perplexity across trials       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stage 0: Preparation

### 0.1 Model & Dataset Caching

For air-gapped or repeated runs, pre-download models and datasets once:

```bash
python -m distill.cache_models --open --output ./hf_cache
python -m distill.cache_datasets --output ./datasets_cache --disk
```

`cache_models.py` uses `snapshot_download()` from `huggingface_hub` to pull all model files (weights, tokenizer, config) into the local HuggingFace cache (`~/.cache/huggingface/hub` or `$HF_HOME/hub`). The directory structure is:

```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen2-1.5B-Instruct/
    └── snapshots/
        └── <sha>/
            ├── config.json
            ├── model.safetensors
            └── tokenizer.json
```

`cache_datasets.py` downloads and optionally saves datasets to disk (Arrow format) so they load without network access. Set `HF_HUB_OFFLINE=1` to block all network calls during training.

The project also maintains a pre-converted MLX bundle for the default open models:

```
airgap_bundle/
└── mlx_models/
    ├── qwen2-1.5b-instruct/   # pre-converted via mlx_lm.convert
    └── qwen2-0.5b-instruct/
```

`distill_mlx.py` checks `_resolve_mlx_path()` at startup — if a local MLX path exists for the requested model ID, it loads from there directly (no conversion needed, much faster startup).

### 0.2 Synthetic Data Augmentation

```bash
python -m distill.run_distillation_agent --open --synthetic_data --n_synthetic 2000
```

`generate_synthetic_data.py` uses the teacher model to generate instruction-response pairs from seed prompts. The process:

1. Load a small set of seed instructions from the training dataset
2. Prompt the teacher to generate variations (with diverse topics, styles, lengths)
3. Filter by length (min 50 tokens), deduplicate via n-gram hash comparison
4. Merge with original dataset

This is most valuable when the base dataset is small (<1000 samples) or narrow in topic coverage. The teacher's generated responses inherently have low perplexity from the teacher's perspective, which makes distillation more stable.

### 0.3 SFT Warmup (Curriculum Learning)

```bash
python -m distill.run_distillation_agent --open --curriculum --sft_epochs 1 --epochs 2
```

`distill_sft.py` runs a supervised fine-tuning pass on the student before knowledge distillation begins. It trains on ground-truth responses using standard cross-entropy loss (not KL divergence).

**Why it helps:** Cold-starting KD from a randomly-initialized student forces large initial KL divergence and unstable gradients. SFT warmup gives the student a reasonable starting point — it already knows to produce plausible text before being asked to match the teacher's exact logit distribution. This is particularly important for:

- Large teacher-to-student size ratios (e.g., 8B → 1B)
- Instruction-following tasks where the output format matters
- Low refusal-rate targets (student needs format knowledge before KD)

The `--curriculum` flag passes `--sft_warmup_dir` to the main distillation script, which initializes from the SFT checkpoint instead of base weights.

---

## Stage 1: Distillation Core

### 1.1 Knowledge Distillation Loss Theory

Knowledge distillation trains the student to mimic the teacher's **output probability distribution**, not just the hard labels. The teacher's softened logits encode richer information than one-hot targets — e.g., the teacher assigns small but non-zero probability to near-synonyms and semantically related tokens, and near-zero probability to unrelated ones.

**KL Divergence Loss (Forward KL):**

```
L_KD = KL(P_teacher || P_student) = Σ P_teacher(t) * log(P_teacher(t) / P_student(t))
```

This is "mean-seeking" — the student is penalized for putting zero probability anywhere the teacher puts non-zero probability. The student spreads probability to cover all modes the teacher assigns weight to.

**Reverse KL (MiniLLM, `distill_minillm.py`):**

```
L_KD = KL(P_student || P_teacher) = Σ P_student(t) * log(P_student(t) / P_teacher(t))
```

This is "mode-seeking" — the student concentrates on the teacher's highest-probability modes and ignores low-probability tails. For language generation, this produces sharper, more confident outputs with less repetition. MiniLLM uses TRL's `KDTrainer` to implement this efficiently.

**Temperature Scaling:**

Both losses use a temperature parameter T to soften the probability distributions:

```
P(t | x) = softmax(logits(t | x) / T)
```

Higher T (e.g., 1.5–2.0) flattens the distribution, amplifying the signal from low-probability tokens and making the student learn more from the teacher's uncertainty. Lower T (e.g., 0.5) sharpens the distribution, making the student focus on the teacher's top choices. Default: T=1.0.

**Mixed Loss (CE alpha):**

In `distill_mlx.py`, the total loss is a weighted mix:

```
L_total = (1 - alpha) * L_KD + alpha * L_CE
```

Where `L_CE` is the standard cross-entropy against ground-truth tokens. This stabilizes training — pure KD can be sensitive to the teacher's distribution at initialization, while CE provides a consistent training signal. Default: `--ce_alpha 0.1` (10% CE, 90% KD).

### 1.2 Top-K Logit Sparsification

Full-vocabulary logits are memory-prohibitive at scale:

- Vocabulary size: ~150,000 tokens (Llama3 tokenizer)
- Sequence length: 512 tokens
- Float16 precision: 2 bytes
- Per-sample storage: 150,000 × 512 × 2 = **~150 MB**
- For 2,000 training samples: **~300 GB**

The solution: keep only the top-K logits per token position (default K=50):

```bash
python -m distill.distill_mlx --topk_logits 50
```

Top-50 logits capture >99% of the teacher's probability mass (the long tail of near-zero logits contributes negligible KL signal). Storage becomes:

- 50 logits + 50 indices × 512 tokens × 2000 samples = **~100 MB**

The teacher logits are pre-computed before training begins (`_precompute_teacher_logits()` in `distill_mlx.py`), stored in memory, then the teacher model is freed with `del teacher_model; mx.clear_cache()`. This makes the GPU exclusively available for student training.

### 1.3 LoRA (Low-Rank Adaptation)

LoRA keeps the base model frozen and injects small trainable rank-r matrices into the attention layers:

```
W = W_base + B * A     (where A ∈ R^{r×d}, B ∈ R^{d×r})
```

The number of trainable parameters scales as `2 * r * d` per layer rather than `d²`, reducing trainable parameters by ~99% (e.g., from 500M to 5M). This makes distillation feasible on consumer hardware.

**Rank selection:**

| Backend | Default `lora_r` | Notes |
|---------|-----------------|-------|
| MLX | 8 | Lower rank fits in unified memory; increase to 32+ for quality |
| PyTorch | 64 | MPS handles higher rank; matches TRL KDTrainer defaults |
| Unsloth | 64 | Unsloth's optimized kernels work best at higher ranks |

**Layers targeted:** By default, `q_proj`, `v_proj`, `k_proj`, `o_proj`, and `gate_proj` layers. The `mlx_lm.tuner.utils.linear_to_lora_layers()` function applies LoRA to all linear layers matching the target pattern.

### 1.4 Teacher Logit Precomputation

All backends pre-compute teacher logits **once** before training begins:

```
Phase 1a: Teacher forward pass (all data, batched)
  → Store top-K logits per position in memory
  → Free teacher from GPU

Phase 1b: Student training loop (logits already in memory)
  → Load one batch of student inputs
  → Load corresponding pre-computed teacher logits
  → Compute KD loss → Backward → Update LoRA weights
```

This is the most important memory optimization: the teacher and student are never simultaneously on the GPU. Without this, an 8B teacher + 1B student would require ~20 GB VRAM.

**Batching:** Teacher logit computation uses larger batch sizes than training (the teacher is in eval mode, no gradients). `distill_mlx.py` uses `--batch_size 4` for teacher logit batching by default. The logit computation speed depends primarily on the teacher's size and the number of samples.

### 1.5 Gradient Accumulation

To achieve large effective batch sizes without running out of memory, gradients are accumulated over multiple micro-batches before each optimizer step:

```
effective_batch_size = batch_size × grad_acc
```

Default in `distill_mlx.py`: `--batch_size 2 --grad_acc 4` → effective batch = 8

This matters for training stability: KD loss tends to be noisy at small batch sizes because individual samples' logit distributions vary widely. Larger effective batches smooth out this noise, leading to more stable convergence.

In MLX, each micro-batch requires `mx.eval(loss_val, grads)` to force computation before accumulation (MLX uses lazy evaluation and won't compute until forced). After all micro-batches are accumulated, `mx.eval(accum_grads)` forces the final accumulated gradients, then the optimizer step runs.

### 1.6 Training Loop & Monitoring

Each training step:

1. Load micro-batch (student inputs + pre-computed teacher logits)
2. Forward pass through student (with LoRA applied)
3. Compute loss (KD + optional CE)
4. Backward pass (gradient computation through LoRA matrices only)
5. Accumulate gradients
6. After `grad_acc` micro-batches: optimizer step + zero gradients
7. Log loss to `metrics.jsonl` every `--log_steps` steps
8. Evaluate on validation set every `--eval_steps` steps
9. Check `pause.flag` — if it exists, save checkpoint and exit gracefully

**pause.flag protocol:** Both the training watchdog (plateau detection) and thermal agent write a `pause.flag` file in the output directory when they detect a problem. All training scripts check for this file at each step. When detected, the trainer saves the current checkpoint and exits with code 0 (clean exit). The orchestrator (`run_distillation_agent.py`) detects this and marks the trial as interrupted rather than failed.

### 1.7 Checkpoint Saving & Resume

At the end of each epoch, the trainer saves:

- **MLX:** `mlx_student_weights.npz` (all model weights as NumPy arrays), plus `distill_config.json` and `metrics.jsonl`
- **PyTorch:** Standard HuggingFace format (safetensors + config + tokenizer)

Resume from checkpoint with `--resume`:

```bash
python -m distill.distill_mlx --open --resume --output_dir ./distilled-mlx
```

The script finds the latest epoch checkpoint and continues from there, loading the optimizer state (for MLX: `Adam` state is serialized alongside weights).

---

## Stage 1 Optimization Phases

### Phase 1 Speedup (Implemented 2026-03-03)

**1. Flash Attention 2 (2–3× attention speedup)**

Flash Attention rewrites the standard attention computation to be I/O-optimal: instead of materializing the full N×N attention matrix in HBM (high-bandwidth memory), it tiles the computation in SRAM. This reduces memory bandwidth by ~10× and attention wall-clock time by 2–3×.

```bash
pip install flash-attn --no-build-isolation
```

**Important: Flash Attention requires CUDA.** It does NOT work with MPS (Apple Silicon PyTorch) or MLX. The script auto-detects and falls back gracefully:

```python
try:
    attn_impl = "flash_attention_2"  # 2-3x speedup
except ImportError:
    attn_impl = "sdpa"               # Fallback: scaled dot-product attention
```

**2. torch.compile() (20–40% overall speedup)**

`torch.compile()` JIT-compiles the model's forward and backward passes, fusing operations and eliminating Python overhead in the inner training loop. Enabled automatically for PyTorch 2.0+ on CUDA.

**Important: torch.compile() does NOT work with MPS** due to an `InductorError` on Apple Silicon (as of PyTorch 2.5). Scripts automatically disable it on MPS:

```python
if device.type != "mps":
    model = torch.compile(model, mode="reduce-overhead")
```

First-run overhead: ~1–2 min for graph compilation. Subsequent runs use the cached compiled graph.

**3. Reduced eval_steps (10% speedup)**

Evaluating too frequently wastes GPU time on validation passes. Default eval frequency changed from every step to every 2 steps. Still sufficient granularity for monitoring (~25–30 evaluations per epoch on 2000 samples).

**Total Phase 1 impact (5-trial run):**

| Configuration | Time | vs. Original |
|---------------|------|-------------|
| Original | 7.2 hours | — |
| Phase 1 (CUDA) | 4.1 hours | 43% faster |
| Phase 1 (MPS, no Flash/compile) | 5.5 hours | 24% faster |

---

### Phase 1.5 Speedup (Implemented 2026-03-03)

**1. DataLoader Optimization (5–10% speedup)**

Parallel data loading with worker processes pre-fetching batches while the GPU trains on the current batch:

```python
dataloader_num_workers=4,
dataloader_prefetch_factor=2,
```

Eliminates the data loading bottleneck where the GPU waits idle between batches.

**2. Gradient Accumulation Tuning (10–15% speedup)**

Increased physical batch size from 4 to 8 (halves the accumulation steps needed for the same effective batch). The M3 Max's large unified memory pool handles this without OOM. More data per GPU step = better hardware utilization.

```python
# Old: batch_size=4, grad_acc=16 → effective batch = 64
# New: batch_size=8, grad_acc=8  → effective batch = 64 (same)
```

**3. Memory Cache Clearing (2–5% speedup)**

Explicit cache clearing between stages prevents memory fragmentation from causing slowdowns mid-training:

```python
# After teacher logit precompute
del teacher_model
torch.mps.empty_cache()  # PyTorch/MPS
mx.clear_cache()          # MLX
```

**4. Early Stopping for Diverging Trials (55 min savings per failed trial)**

At training step 20 (~2–3 minutes into training), the `EarlyStoppingCallback` checks if the current loss has diverged beyond a threshold:

```python
if step == 20 and current_loss > baseline_loss * 1.5:
    raise EarlyStoppingException("Trial diverging, stopping early")
```

Step 20 is chosen because:
- Early enough to save most of a trial's time (saves ~55 min of a 60-min trial)
- Late enough for the loss to have stabilized past initialization noise (typically stable after step 10–15)
- False positive rate: ~5–10% (conservative 1.5× threshold allows significant variation)

In a 5-trial autonomous run, ~1 trial typically diverges early, saving an additional ~55 minutes.

**5. Quality Eval on Winner Only (already implemented)**

`eval_quality.py` runs once after winner selection, not after every trial. Saves 36 minutes for 5 trials (9 min × 4 non-winning trials skipped).

**Cumulative Phase 1 + 1.5 impact:**

| Scenario | Time |
|----------|------|
| Original | 7.2 hours |
| Phase 1 only | 4.1 hours (43% faster) |
| Phase 1 + 1.5 (no early stops) | 3.5 hours (51% faster) |
| Phase 1 + 1.5 (1 early stop) | 3.0 hours (58% faster) |

---

## Stage 2: Evaluation

### 2.1 Validation Perplexity (`run_eval.py`)

Standard cross-entropy evaluation on a held-out validation split:

```
perplexity = exp(mean(-log P_student(token | context)))
```

Reports:
- `eval_perplexity` — student's perplexity on validation set
- `teacher_perplexity` — teacher's perplexity on same set (comparison baseline)
- `ppl_gap_pct` — `(student_ppl - teacher_ppl) / teacher_ppl * 100`

A perplexity gap <30% is generally acceptable; <15% is good; >50% indicates the distillation is struggling.

### 2.2 Quality Gates — Phase 1: Critical Fixes

`eval_quality.py` generates responses to validation prompts and applies production-ready quality filters.

**Batch inference (8× speedup):**

Old sequential generation: 50 samples × 2s = 100s
Batched generation (batch_size=8): 7 batches × 2s = 14s

The model generates all batch samples in parallel. LLM-as-judge scoring is also batched.

**Refusal detection:**

Five regex patterns flag common refusal language:

```python
REFUSAL_PATTERNS = [
    r"I (cannot|can't|am unable to|must decline)",
    r"(This|That) (request|question|prompt) (is|seems|appears) (harmful|inappropriate|unethical)",
    r"I('m| am) (not|unable)",
    r"(For|As) an AI (language model|assistant|system)",
    r"(I|This) (do|does|can't|cannot|shouldn't|won't) (assist|help|provide|support|generate)",
]
```

Samples matching any pattern are marked as refusals and counted in `refusal_rate_pct`. Target: <5%.

**Length filtering:**

| Condition | Action |
|-----------|--------|
| < 10 tokens | REJECT (too_short) |
| 10–199 tokens | WARN (below_target) |
| 200–2000 tokens | PASS |
| > 2000 tokens | REJECT (too_long) |

**Pass rate:** Samples passing all filters. Target: ≥80%.

### 2.3 Quality Gates — Phase 2: Quality Enhancements

**Category detection (6 categories):**

Keyword-based classification of prompts into:
- `math` — numerical, calculation, algebra, geometry keywords
- `code` — programming language names, function/class/loop keywords
- `creative` — story/poem/creative/fiction/write keywords
- `reasoning` — logic/inference/deduce/conclude keywords
- `qa` — what/how/why/explain question keywords
- `other` — catch-all

Used for balance tracking. An unbalanced distribution (e.g., >40% in one category) can indicate dataset bias that the student may over-specialize on.

**Teacher perplexity on student outputs (`--judge-teacher-ppl`):**

Runs the teacher model on the student's *generated* responses (not ground truth). If the teacher assigns high perplexity to the student's outputs, it means the student is generating text the teacher considers unlikely — a signal of quality degradation.

Threshold: `avg_teacher_ppl > 100` → ERROR (student is generating near-gibberish relative to teacher's distribution).

**Embedding diversity:**

Compute mean pairwise cosine distance between student response embeddings. Captures semantic diversity beyond token-level metrics:

- `mean_pairwise_distance` — average distance between all response pairs (higher = more diverse)
- `coverage_radius_95` — 95th percentile distance from the centroid (how spread out responses are)

Semantic collapse (all responses mean the same thing in different words) doesn't appear in token-level distinct-N metrics but shows up as low pairwise distances.

**Diversity metrics (token-level):**

- `distinct-1` — fraction of unique unigrams across all responses. Target: ≥0.55 (≥0.65 preferred)
- `distinct-2` — fraction of unique bigrams. Usually higher than distinct-1.
- `avg_max_rep` — average maximum n-gram repetition length. >3 indicates repetition loops.
- `3-gram entropy` — Shannon entropy of the trigram distribution. Target: ≥6.0 bits.

These detect mode collapse — when the student learns to repeat a small set of phrases regardless of the input.

### 2.4 Quality Gates — Phase 3: Optimization

**MLX backend for evaluation:**

`eval_quality.py --backend mlx` runs generation on the MLX backend, giving 2–3× speedup on Apple Silicon. The MLX backend loads the quantized `mlx_q4/` directory rather than the full float16 weights.

**UMAP visualization (`embedding_viz.json`):**

Requires `pip install umap-learn`. Projects the high-dimensional response embeddings to 2D:

```json
{
  "points": [
    {"x": -2.3, "y": 1.5, "category": "math"},
    {"x": 0.8, "y": -1.2, "category": "code"}
  ]
}
```

The dashboard renders this as a scatter plot. Clustered by category = good categorical organization; one massive blob = possible mode collapse; widely scattered = good diversity.

**Volume guidance:**

Based on LIMA (Zhou et al., 2023) and Orca (Mukherjee et al., 2023) findings: quality of training data matters more than quantity. The system reports:

- High pass rate (≥80%): suggest expanding to 50k–100k samples
- Low pass rate (<80%): focus on filtering first, then scale

**LLM-as-judge scoring (`--judge`):**

Uses the teacher itself as a zero-shot evaluator. The judge prompt asks the teacher to score each student response on a 1–10 scale based on:
- Instruction following (does it answer the question?)
- Accuracy and coherence
- Formatting and completeness

Target: ≥6/10 (≥7/10 preferred). Score <4/10 triggers a WARN with a suggestion to add curriculum learning.

### 2.5 WikiText-2 Benchmark (`run_benchmarks.py`)

Evaluates perplexity on WikiText-2, a standard NLP benchmark. This detects catastrophic forgetting — when the student's overall language modeling ability degrades even if distillation task-specific metrics look good.

A student that over-fits to the distillation dataset (e.g., all instruction-following data) may lose general text coherence. WikiText-2 perplexity rising significantly vs. the base student checkpoint signals this.

Run with `--benchmarks` flag or `--compare_teacher` flag in the agent.

---

## Stage 3: Export

### 3.1 GGUF Export (llama.cpp)

GGUF (GPT-Generated Unified Format) is the standard format for llama.cpp inference. The export pipeline for MLX-trained models has several steps:

**Step 1: Convert MLX .npz → HuggingFace directory**

`_mlx_weights_to_hf_dir()` in `run_distillation_agent.py`:

1. Loads `mlx_student_weights.npz` (contains both base weights and LoRA adapter weights)
2. Detects LoRA layers by naming convention: `lm.model.layers.N.self_attn.q_proj.lora_a` etc.
3. Merges LoRA adapters into base weights: `W_merged = W_base + B * A`
4. Converts from MLX arrays to NumPy float16
5. Saves merged weights as safetensors
6. Copies HF config files (`config.json`, tokenizer files) from the HF cache

**Step 2: Convert HF format → GGUF**

```bash
python /Users/Shared/llama/convert_hf_to_gguf.py ./hf_export_dir \
    --outfile student-f16.gguf \
    --outtype f16
```

The `find_llama_cpp()` function searches for llama.cpp in priority order:
1. `/Users/Shared/llama` (project standard location)
2. `./llama.cpp` (relative to project root)
3. `../llama.cpp` (sibling directory)

**Step 3: Quantize (optional)**

```bash
# Built into llama.cpp
./llama-quantize student-f16.gguf student-q4_K_M.gguf q4_K_M
```

| Quantization | Size (0.5B) | Quality loss | Use case |
|--------------|------------|-------------|----------|
| `f16` | ~1.0 GB | None | Reference, high-accuracy inference |
| `q8_0` | ~0.5 GB | Minimal | Default production |
| `q4_K_M` | ~0.3 GB | Small | Edge deployment, mobile servers |

**Serving:**

```bash
/Users/Shared/llama/llama-server -m student-q4_K_M.gguf --port 8080
```

See [docs/LLAMA_CPP_STUDENT.md](docs/LLAMA_CPP_STUDENT.md) for full llama-server setup.

### 3.2 CoreML Export (.mlpackage)

`export_coreml.py` converts the HuggingFace model to CoreML format for Apple Neural Engine deployment:

1. Load model in float32 (CoreML tools require float32 inputs)
2. Create example inputs (dummy token tensor)
3. `torch.jit.trace()` — trace the model's forward pass to a TorchScript graph
4. `ct.convert()` — convert TorchScript to CoreML using coremltools
5. Optional: apply `ct.optimize.coreml.palettize_weights()` for int4 quantization
6. Save `.mlpackage` bundle

**Compute units:**

```bash
python -m distill.export_coreml --model_dir ./distilled --compute_units ALL
```

| Option | Hardware used | Latency | Power |
|--------|--------------|---------|-------|
| `CPU_ONLY` | CPU only | Slowest | Lowest |
| `CPU_AND_GPU` | CPU + GPU | Medium | Medium |
| `ALL` (default) | CPU + GPU + ANE | Fastest | Medium |

The Apple Neural Engine (ANE) provides 11–38 TOPS depending on chip generation, making it ideal for inference on quantized models.

**Post-training quantization:**

| `--quantize` | Size reduction | Notes |
|--------------|---------------|-------|
| None | 1× | Float32, largest |
| `float16` | ~2× | Good ANE compatibility |
| `int8` | ~4× | Fast on ANE |
| `int4` | ~8× | Smallest, for edge/mobile |

The script also prints a Swift inference snippet for the generated model, ready to paste into an iOS/macOS project.

### 3.3 MLX Quantized Export

Built into `distill_mlx.py` via `mlx_lm.convert`:

```bash
python -m distill.distill_mlx --open --q_bits 4 --output_dir ./distilled-mlx
# Produces: distilled-mlx/mlx_q4/ (4-bit quantized weights)
```

MLX uses group quantization: weights are split into groups of 64 elements, each group has its own scale and zero-point. This preserves more accuracy than naive per-tensor quantization.

The `mlx_q4/` directory contains:
- `config.json` — model architecture
- `model.safetensors` — quantized weights (MLX format, NOT compatible with PyTorch AutoModelForCausalLM)
- `tokenizer.json` — tokenizer
- `quantization.json` — quantization parameters

Inference:
```python
from mlx_lm import load, generate
model, tokenizer = load("./distilled-mlx/mlx_q4")
output = generate(model, tokenizer, prompt="...", max_tokens=100)
```

---

## Stage 4: Agentic Feedback Loop

### 4.1 Experiment Logging (`experiment_log.py`)

Every run appends one JSON record to `experiment_log.jsonl`:

```json
{
  "timestamp": "2026-03-08T14:23:11",
  "run_id": "distilled-mlx-20260308-1423",
  "config": {
    "teacher": "Qwen/Qwen2-1.5B-Instruct",
    "student": "Qwen/Qwen2-0.5B-Instruct",
    "dataset": "tatsu-lab/alpaca",
    "backend": "mlx",
    "temperature": 1.0,
    "lora_r": 64,
    "epochs": 2,
    "max_samples": 2000
  },
  "metrics": {
    "eval_perplexity": 8.24,
    "teacher_perplexity": 6.91,
    "ppl_gap_pct": 19.2,
    "distinct_1": 0.71,
    "judge_score": 7.1,
    "wikitext2_ppl": 9.45
  },
  "outcome": "success"
}
```

The JSONL (one JSON object per line) format allows incremental appends without rewriting the file and is trivially parseable in streaming fashion.

### 4.2 Diagnostic Analysis (`diagnose()`)

After each run, `diagnose()` analyzes the metrics and prints human-readable suggestions:

```
[OK]    Perplexity gap 19.2% (acceptable)
[OK]    Diversity distinct-1=0.71
[OK]    Judge score 7.1/10
[WARN]  ppl_gap > 30%? Try: --temperature 1.5 --epochs 3 --lora_r 128
[INFO]  Volume guidance: pass_rate=82% → expand to 50k-100k samples
```

Diagnostic rules (simplified):

| Condition | Message | Suggested Fix |
|-----------|---------|--------------|
| `ppl_gap > 30%` | High perplexity gap | `--temperature 1.5 --epochs 3` |
| `ppl_gap > 50%` | Very high gap | `--curriculum --lora_r 128` |
| `distinct_1 < 0.55` | Low diversity | `--synthetic_data --temperature 1.3` |
| `judge_score < 4` | Poor instruction following | `--curriculum --max_samples 5000` |
| `refusal_rate > 5%` | High refusals | `--curriculum` or filter dataset |
| `wt2_ppl increase > 20%` | Catastrophic forgetting | `--ce_alpha 0.3` |

### 4.3 Hyperparameter Proposal (`propose_next()`)

For multi-trial runs (`--n_trials 3+`), `propose_next()` proposes the next hyperparameter set by looking at historical performance:

**Search space:**
```python
_SEARCH_SPACE = {
    "temperature": [0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
    "lora_r":      [16, 32, 64, 128],
    "epochs":      [1, 2, 3, 4],
}
```

**Algorithm:**

- **Trials 1–2 (exploration phase):** Randomly sample from the search space. Ensures diverse coverage before committing to any direction.

- **Trials 3+ (hill-climbing phase):** Find the best previous run (lowest `eval_perplexity`) for this teacher/student/dataset combination. Move each hyperparameter toward the best run's value:
  - If best used `temperature=1.5` and current is `1.0` → try `1.3` (halfway)
  - If best used `lora_r=128` and current is `64` → try `96` or `128`
  - Add random perturbation (±1 step in the search space) to avoid getting stuck

The algorithm is hill-climbing with random restarts, not full Bayesian optimization. For the typical run count (3–10 trials), the overhead of a Gaussian Process or TPE sampler isn't justified — simple hill-climbing converges fast enough.

**Winner selection:** After all trials, the orchestrator selects the trial with the lowest `eval_perplexity`. Only this trial's model is exported (GGUF/CoreML/MLX). All trials' metrics are logged to `experiment_log.jsonl`.

---

## Backend Comparison

| Feature | PyTorch/MPS | MLX | Unsloth |
|---------|------------|-----|---------|
| Speed on M3 Max | 1× (baseline) | 2–5× | 2–4× |
| Memory usage | ~8–12 GB | ~6–10 GB | ~4–8 GB |
| Flash Attention 2 | ❌ (MPS incompatible) | ❌ (own kernels) | ✅ (CUDA only) |
| torch.compile() | ❌ (InductorError on MPS) | ❌ (not PyTorch) | ✅ (CUDA) |
| MLX-native kernels | ❌ | ✅ | ❌ |
| Unified memory | Partial (MPS) | ✅ (full) | ❌ |
| Lazy evaluation | ❌ | ✅ | ❌ |
| LoRA implementation | PEFT library | mlx_lm.tuner | Unsloth optimized |
| pause.flag support | ✅ | ✅ | ✅ |
| Offline mode | ✅ | ✅ | ✅ |
| Best for | Compatibility | Daily use on M3 | CUDA deployment |

**Why MLX is faster on Apple Silicon:**

1. **Unified memory**: Apple Silicon shares RAM between CPU and GPU. MLX exploits this — tensors live in one unified address space with no CPU↔GPU transfer overhead. PyTorch/MPS still has to manage transfers between the CPU-side PyTorch allocator and the MPS device.

2. **Lazy evaluation**: MLX builds a computation graph and executes it only when results are needed (`mx.eval()`). This enables operation fusion, dead code elimination, and memory reuse across the graph.

3. **Native Metal kernels**: MLX is written specifically for Apple Silicon's GPU microarchitecture. PyTorch/MPS uses Metal Shaders translated from CUDA ops, which can be suboptimal for the GPU's specific instruction set.

---

## Monitoring & Observability

### Training Watchdog (`training_watchdog.py`)

Runs in a separate process alongside training, polling `trainer_state.json` every 60 seconds:

```bash
# Terminal 1: Training
python -m distill.distill_mlx --open --watchdog --output_dir ./distilled-mlx

# Terminal 2: Watchdog
python -m distill.training_watchdog ./distilled-mlx --interval 60
```

**Plateau detection:**

Reads the last N losses from `trainer_state.json` (default N=3). If the maximum absolute difference across consecutive pairs is below `max_delta` (default 0.001), the learning rate has likely converged to a local minimum. The watchdog writes `watchdog_suggestions.json` with a scaled learning rate recommendation (`lr_scale: 0.8`).

**Divergence detection:**

Computes the mean of the first `baseline_window` (default 5) losses as the baseline. If the recent N-loss average exceeds `baseline × divergence_threshold` (default 1.5×), something is wrong (learning rate too high, wrong hyperparameters). The watchdog writes `pause.flag` to stop training immediately.

Config (`configs/watchdog_rules.json`):

```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "min_points": 5,
    "lr_scale": 0.8
  },
  "divergence": {
    "window": 3,
    "threshold": 1.5,
    "baseline_window": 5,
    "min_points": 8
  }
}
```

A C++ watchdog (`cpp/build/watchdog`) is also available for environments where Python isn't available (e.g., embedded systems or minimal containers):

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build .
./watchdog ../../distilled-mlx --interval 60 --config ../../configs/watchdog_rules.json
```

### Thermal Agent (`thermal_agent.py`)

System-wide hardware monitoring that pauses ALL GPU workloads (training, evaluation, export) when thermal limits are exceeded. Uses `mactop` for accurate M3 readings (SMC-based monitoring, which `osx-cpu-temp` cannot do on M3):

```bash
mactop --headless --format json --count 1
# Returns: {"cpu_temp": 49.2, "gpu_temp": 57.1, "soc_temp": 51.3, "total_power": 38.2}
```

The thermal agent:

1. Polls temperature every 30 seconds (configurable)
2. If `soc_temp_c` ≥ 85°C (default threshold): writes `pause.flag` in all monitored directories
3. All running trainers detect `pause.flag` and checkpoint + exit
4. When temperature drops to 80°C (5°C hysteresis): deletes `pause.flag`, trainers can resume

Install as always-on LaunchAgent (recommended):

```bash
./scripts/install_thermal_agent.sh
launchctl list | grep thermal_agent
```

Observed M3 Max temperatures:
- Idle: CPU ~44°C, GPU ~39°C
- Under MPS training load: CPU ~49°C, GPU ~57°C
- Thermal limit (85°C default): well above typical peaks — triggers only under extreme sustained load

### Gradio Dashboard (`dashboard.py`)

Multi-tab live monitoring UI:

| Tab | Data source | Refresh |
|-----|-------------|---------|
| Plots | `metrics.jsonl` | 30s auto |
| Evaluate | Auto-discovered model dirs | On demand |
| Artifacts | `.gguf`, `.mlpackage`, `.npz` files | On save |
| Thermal | `thermal.log` | 30s auto |
| Experiments | `experiment_log.jsonl` | Post-run |

```bash
python -m distill.dashboard --port 7860
# Opens at http://127.0.0.1:7860
```

---

## Air-Gapped Operation

Three-phase workflow for secure environments:

**Phase 1: Stage (online machine)**

```bash
python -m distill.setup_airgap --open   # Or manually:
python -m distill.cache_models --open
python -m distill.cache_datasets --disk
conda pack -n distillation_m3 -o distill-offline.tar.gz
sha256sum distill-offline.tar.gz > SHA256SUMS
```

**Phase 2: Transfer**

Copy via USB/SSD: the conda pack, `hf_cache/`, `datasets_cache/`, project code, and `SHA256SUMS`.

**Phase 3: Run (offline machine)**

```bash
# Restore environment
mkdir -p ~/envs/distill
tar -xzf distill-offline.tar.gz -C ~/envs/distill
source ~/envs/distill/bin/activate

# Set cache paths
export HF_HOME=/path/to/hf_cache
export HF_DATASETS_CACHE=/path/to/datasets_cache

# Run with --offline flag
python -m distill.run_distillation_agent \
    --open --offline --backend mlx --export gguf
```

The `--offline` flag sets `HF_HUB_OFFLINE=1` and `HF_DATASETS_OFFLINE=1`, which makes the HuggingFace libraries raise an error immediately if any network access is attempted (rather than hanging).

---

## Configuration Reference

### `configs/agent_config.json`

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
  "coreml_quantize": null,
  "epochs": 2,
  "max_samples": 2000,
  "temperature": 1.0,
  "lora_r": 64,
  "batch_size": 8,
  "grad_acc": 8,
  "eval_steps": 2,
  "ce_alpha": 0.1,
  "topk_logits": 50
}
```

### `configs/watchdog_rules.json`

```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "min_points": 5,
    "lr_scale": 0.8
  },
  "divergence": {
    "window": 3,
    "threshold": 1.5,
    "baseline_window": 5,
    "min_points": 8
  }
}
```

---

## Example: Complete Autonomous Run

```bash
# Let the agent find best config over 5 trials
python -m distill.run_distillation_agent \
    --open --offline \
    --n_trials 5 \
    --export all \
    --compare_teacher \
    --benchmarks \
    --log_experiment \
    --watchdog \
    --judge

# Agent execution flow:
# Trial 1: base config (temp=1.0, lora_r=64, epochs=2)
#   → trains 2000 samples → evaluates → ppl=8.5
#
# Trial 2: random exploration (temp=1.5, lora_r=128, epochs=2)
#   → trains → evaluates → ppl=7.2
#
# Trial 3: hill-climb toward best (temp=1.3, lora_r=128, epochs=3)
#   → trains → evaluates → ppl=6.9
#
# Trial 4: hill-climb (temp=1.4, lora_r=128, epochs=3)
#   → early stop at step 20 (diverging) → ppl=N/A
#
# Trial 5: random restart (temp=1.0, lora_r=64, epochs=3)
#   → trains → evaluates → ppl=7.1
#
# Winner: Trial 3 (ppl=6.9)
#   → Export GGUF + CoreML + MLX
#   → Log all 5 trials to experiment_log.jsonl
#   → Print: [OK] Perplexity gap 17.3% (acceptable)

# View results
python -m distill.dashboard --runs_dir ./distilled-minillm
python -m distill.experiment_log --show 10
```

---

## Research Context

This implements the **2025–2026 agentic knowledge distillation pattern**:

1. **Autonomous ML Engineering**: The system proposes hyperparameters, runs experiments, evaluates results, and selects the best configuration without human intervention between trials.

2. **Self-Improving Loop**: Each run's results are logged in `experiment_log.jsonl` and feed into `propose_next()`, which improves subsequent trials via hill-climbing.

3. **Multi-Metric Optimization**: Not just perplexity — diversity (distinct-N), instruction-following (LLM-as-judge), safety (refusal rate), and generalization (WikiText-2).

4. **Diagnostic Feedback**: Human-readable suggestions bridge the agent and engineer when manual intervention is needed.

5. **Production-Ready Export**: One distillation → three deployment formats (llama.cpp, iOS/macOS, Apple Silicon inference).

Key references:
- MiniLLM (Gu et al., 2024) — Reverse KL for LLM distillation
- LIMA (Zhou et al., 2023) — Quality over quantity for instruction data
- Orca (Mukherjee et al., 2023) — Progressive learning signals for distillation
- LoRA (Hu et al., 2022) — Low-rank adaptation for parameter-efficient fine-tuning
