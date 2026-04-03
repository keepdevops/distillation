"""Help tab widget layout — all accordion markdown sections."""
from __future__ import annotations

import base64
import logging

import gradio as gr

logger = logging.getLogger(__name__)


def build_tab_help():
    """Build the Help tab.

    Returns
    -------
    Empty dict — the Help tab contains no interactive widgets that need wiring.
    """
    with gr.Tab("Help"):
        gr.Markdown("# Distillation Launcher — Operation Guide")
        gr.Markdown(
            "**Seven tabs:** Configure & Launch · Data Prep · Domain Synthesis · "
            "Eval · **Expert Pipeline** · Live Logs · Help.  "
            "A progress bar appears on every tab — no need to switch to Live Logs to monitor a run. "
            "Click a section below to expand it."
        )

        # ── Golden pipeline ──────────────────────────────────────────────
        with gr.Accordion("🏆  Golden Pipeline — Recommended End-to-End Sequence", open=True):
            gr.Markdown("""
```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        GOLDEN PIPELINE  (best quality)                         ║
╠════╦═══════════════════════╦══════════════════════════════════════════╦═════════╣
║ 1  ║ Data Prep  (optional) ║ Filter dataset or run Magpie synthesis   ║  5–30 m ║
║ 2  ║ Domain Synth (opt.)   ║ Generate domain pairs: code/math/legal/… ║ 30–90 m ║
║ 3  ║ Configure & Launch    ║ Stage: SFT · 1 epoch → sft_checkpoint    ║  ~30 m  ║
║ 4  ║ Configure & Launch    ║ Stage: MiniLLM · Student=sft_checkpoint  ║   ~2 h  ║
║ 5  ║ Eval                  ║ Perplexity → Quality → WikiText-2        ║  ~15 m  ║
║ 6  ║ Auto (agent)          ║ scripts/run_golden.sh → full pipeline headless ║  ~3 h   ║
╚════╩═══════════════════════╩══════════════════════════════════════════╩═════════╝
```

**Steps 1–2 are optional** but significantly improve output quality. Step 3 (SFT) is strongly
recommended before MiniLLM — it gives the student a warm start so GRPO rewards don't
open negative and stay there.

### One-command golden run (terminal)
```bash
./scripts/run_golden.sh                                    # foreground
./scripts/run_golden.sh > runs/golden_pipeline.log 2>&1 &  # background with log
```
Config is at `configs/golden_pipeline.json` — edit it to change dataset, epochs, LoRA rank, etc.

---

### Quickstart (no HF login · ~2 hr)
1. **Configure & Launch** → Stage: **MiniLLM** · Backend: **MLX** · ☑ **Use open Qwen2 models** → **Launch**
2. Watch progress bar on same tab. Switch to **Live Logs** for full stream + loss/grad charts.
3. **Eval** → Run Perplexity Eval once training completes.

---

### When to add SFT warmup
Run SFT first when any of these appear in Live Logs during MiniLLM:
- `reward` stuck at −1.0 for more than 50 steps
- `frac_reward_zero_std` > 0.5 after step 30
- You are using a new domain dataset the student hasn't seen before

After SFT finishes, set the MiniLLM **Student** field to `distilled-minillm/sft_checkpoint`.

---

### Decision guide
| Goal | Recommended path |
|------|-----------------|
| Fastest first result | Quickstart (skip steps 1–3) |
| Best general quality | `./scripts/run_golden.sh` |
| Domain specialist (tax / legal / medical / finance / coding) | **Expert Pipeline** tab |
| Fastest training on M-series Mac | Backend: **MLX**, batch=2, grad_acc=8 |
| Largest pair that fits 36 GB unified memory | Teacher ≤ 3B + student ≤ 1B |
| Resume an interrupted run | MLX: tick **Resume**; PyTorch: relaunch from latest checkpoint |

---

### Outputs written to disk
| File | Contents |
|------|----------|
| `output_dir/*.safetensors` | Merged final model weights (PyTorch backend) |
| `output_dir/mlx_student_weights.npz` | Trained weights (MLX backend, LoRA + base fused) |
| `output_dir/metrics.jsonl` | Per-step loss, reward, eval metrics (JSON lines) |
| `output_dir/train_log.jsonl` | Structured JSON log from agent runs (step, loss, epoch, ts) |
| `output_dir/sft_labels.jsonl` | Cached teacher labels (SFT reuses on re-run) |
| `output_dir/quality_metrics.json` | Generation diversity + judge scores |
| `output_dir/benchmark_results.json` | WikiText-2 perplexity result |
| `runs/ep_<stage>_<timestamp>.log` | Plain-text log from Expert Pipeline runs |
| `runs/ep_<stage>_<timestamp>.jsonl` | Structured JSON log from Expert Pipeline runs |
| `/Users/Shared/llama/models/*.gguf` | GGUF exports for llama.cpp / Ollama |
""")

        # ── Tab 1 help ───────────────────────────────────────────────────
        with gr.Accordion("Configure & Launch — Parameter Reference", open=False):
            gr.Markdown("""
### Stage
| Stage | Script | When to use |
|-------|--------|-------------|
| **SFT** | `distill_sft.py` | First-pass warmup. Teacher generates response labels; student trains on them with standard cross-entropy. Use before MiniLLM for best results. |
| **MiniLLM** | `distill_minillm.py` / `distill_mlx.py` | Main distillation stage. Student generates completions; GRPO advantage signal pushes it toward teacher distribution (reverse-KL). |

### Backend
| Backend | Speed | Memory | When to use |
|---------|-------|--------|-------------|
| **PyTorch / MPS** | Baseline | ~8–12 GB unified | Stable, supports all features. Use when debugging or running SFT. |
| **MLX** | 2–5× faster | ~4–8 GB unified | Apple-native lazy evaluation. Best for long MiniLLM runs. Uses lower batch defaults (2/4/8). |

> **Switching backends** auto-updates Batch size / Grad acc / LoRA rank to the recommended defaults for that backend. You can still adjust them manually.

### Models
- **Use open Qwen2 models** — Ticks `--open` flag. Forces teacher = `Qwen/Qwen2-1.5B-Instruct`, student = `Qwen/Qwen2-0.5B-Instruct`. No HuggingFace account or license needed.
- **Teacher** — Larger model that provides soft targets or hard labels. Must fit in unified memory alongside the student. Rule of thumb: teacher ≤ 3× student parameters.
- **Student** — Smaller model being trained. For MiniLLM, point this at your **SFT checkpoint** (`distilled-minillm/sft_checkpoint`) for best results.
- Click **Refresh** after a model finishes downloading to see it in the dropdown.

### Training (common to all stages)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Epochs** | 2 | SFT: 1 is usually enough. MiniLLM: 2–3. More can overfit on small datasets. |
| **Max samples** | 2000 | Samples drawn from the dataset. 2000 trains in ~1–2 hours. Use 500 for a smoke test. |
| **Batch size** | 8 (PyTorch), 2 (MLX) | Physical samples per device step. Increase until Activity Monitor shows ~80% GPU pressure. |
| **Gradient accumulation** | 8 | Multiply by batch for effective batch (8×8=64 PyTorch, 2×8=16 MLX). Larger effective batch = smoother gradients. |
| **LoRA rank** | 16 | Higher = more trainable params = slower but higher capacity. 16 is a good balance for both backends. 64 for SFT. |

### SFT options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Learning rate** | 2e-4 | Standard SFT rate. Reduce to 1e-4 if loss oscillates. |
| **Teacher max new tokens** | 128 | Tokens the teacher generates per prompt for the label cache. 128 covers most alpaca responses. |
| **Max sequence length** | 384 | Total tokens (prompt + response) kept for training. Sequences longer than this are truncated. |

### MiniLLM options (PyTorch)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Softens the teacher distribution. Higher (1.5–2.0) = smoother targets, can stabilize early training. |
| **Learning rate** | 2e-5 | Intentionally 10× lower than SFT. Increase to 5e-5 only if rewards plateau after 100 steps. |
| **Generations per prompt** | 4 | GRPO samples per prompt. **At least 4** to get reward variance within each group (fewer → frac_reward_zero_std stays high → no gradient). 8 gives richer signal but 2× slower. |
| **Max completion length** | 256 | Hard cutoff for student generations. **Critical:** too small → model hits limit before EOS → 80%+ clipped_ratio → reward collapses. 256 tokens ≈ 800 characters is the calibrated default. |
| **Eval every N steps** | 20 | Lower = more detail in metrics.jsonl, higher = faster overall run. 20 is a good balance. |

### MLX options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Same as PyTorch. |
| **Learning rate** | 2e-4 | MLX uses forward-KL + CE, not GRPO, so it tolerates a higher LR. |
| **CE alpha** | 0.2 | Weight of cross-entropy (hard label) loss. 0 = pure KD, 1 = pure CE. 0.2 stabilises early training without losing KD signal (increased from 0.1 for better convergence). |
| **Top-K teacher logits** | 50 | Keeps only top-50 teacher token probabilities. Captures >99% of probability mass while reducing logit memory from ~300 GB to ~300 MB per dataset. Teacher is freed from memory immediately after precompute. |
| **Export quantization bits** | 4 | Bits for the MLX quantized export. 4-bit is standard for llama.cpp-compatible GGUF. |
| **Resume** | off | Continue from the last epoch checkpoint in output_dir if a previous MLX run was interrupted. |

### Watchdog
Creates a `pause.flag` file callback. While training is running, you can pause it by creating `output_dir/pause.flag` from the terminal (`touch distilled-minillm/pause.flag`) and resume by deleting it. Useful for thermal management without losing progress.

### Stop button
Sends **SIGKILL** (immediate termination). The last saved checkpoint is preserved. Use it any time — the run can be resumed via **Resume** (MLX) or by relaunching from the latest checkpoint (PyTorch).
""")

        # ── Tab 2 help ───────────────────────────────────────────────────
        with gr.Accordion("Data Prep & Domain Synthesis — When and How to Use Each Tool", open=False):
            gr.Markdown("""
### When to use Data Prep
Out-of-the-box datasets (Alpaca, Guanaco) work fine for quickstart runs. Use Data Prep when:
- You want domain-specific data not in the standard datasets
- Existing dataset quality is low (short responses, repetition, refusals)
- You want to maximize distillation quality with teacher-generated data

### Domain Synthesis (separate tab)
For domain-specialist models (coding, math, medical, legal, finance, tax) use the
**Domain Synthesis** tab instead of — or in addition to — the tools below. It runs
Magpie synthesis with curated system prompts and domain-specific quality filters.
Output: `domain_data/<domain>/hf_dataset/` — point **Dataset** here in Configure & Launch.

### Magpie Synthesis
**What it does:** Loads the teacher model and repeatedly samples from it, conditioning on just the chat-template user-turn prefix. The model "auto-completes" realistic user instructions, then generates responses. No seed dataset needed.

**Output:** `output_dir/hf_dataset/` — an HF dataset you can point directly at the **Dataset** field in Configure & Launch.

**When to use:** Best quality synthetic data. Requires the teacher to be available locally (large download first time). Use when you want data tailored to the teacher's style.

| Parameter | Guidance |
|-----------|----------|
| **Pairs to generate** | Generate 2–3× your target keep count to account for filtering attrition. |
| **Filter output** | Always keep this on — removes duplicates and low-quality pairs. |
| **Target keep** | Final dataset size after filtering. 2000–5000 pairs is enough for a 2-epoch run. |

### Self-Instruct Synthesis
**What it does:** Shows the teacher `seed_examples` instructions from a base dataset, then asks it to generate a new, diverse instruction. The teacher also generates the response. Filtered by perplexity bounds and distinct-2 score.

**Output:** `output_dir/synthetic_data/` — an HF dataset directory.

**When to use:** More diverse than Magpie (prompts are anchored to varied seed examples). Slower per-pair than Magpie. Good for covering topic diversity.

| Parameter | Guidance |
|-----------|----------|
| **Temperature** | 0.9 is a good balance — high enough for diversity, low enough for coherence. Don't go above 1.2. |
| **Seed examples** | 5 is standard. Higher seeds = more context for the teacher but diminishing returns. |

### Dataset Filter
**What it does:** Takes any alpaca-format dataset (HF hub ID or local path) and removes:
- Responses shorter than `min_response_words`
- Low-coherence responses (distinct-2 bigram diversity < `min_distinct2`)
- Near-duplicate pairs (Jaccard similarity above threshold)
- Common refusal patterns ("I cannot...", "As an AI...")

**Output:** `output_dir/` — an HF dataset directory. Use this path directly in Configure & Launch.

**When to use:** Always run this before training on any public dataset. `yahma/alpaca-cleaned` is already filtered but running it again with stricter thresholds is harmless.

| Parameter | Guidance |
|-----------|----------|
| **Target top-N** | How many pairs to keep after scoring. Keeps the highest-quality pairs. |
| **Min distinct-2** | 0.35 is standard. Increase to 0.45 to keep only highly diverse responses. |
| **Min response words** | 20 words eliminates one-liners. Increase to 30–40 for richer training signal. |

### Combining tools (recommended data workflow)
```
General quality:
  1. Filter        yahma/alpaca-cleaned  →  filtered_data/              (2–3 min)
  2. Self-Instruct Generate 3000 pairs   →  synthetic_data/             (30–60 min)
  3. Configure     Dataset = filtered_data/

Domain specialist:
  1. Domain Synth  coding / math / medical …  →  domain_data/<x>/      (30–90 min)
  2. (Optional) mix with filtered general data using datasets.concatenate_datasets()
  3. Configure     Dataset = domain_data/<x>/hf_dataset/
```
""")

        # ── Tab 3 help ───────────────────────────────────────────────────
        with gr.Accordion("Eval — Interpreting Results", open=False):
            gr.Markdown("""
### Shared inputs
- **Model output dir** — The same directory you used as Output directory during training. This is where metrics.jsonl and result files are written.
- **Checkpoint** — Leave blank to eval the final merged model. Enter a path like `distilled-minillm/checkpoint-80` to eval a mid-training checkpoint.
- **Base model** — Only used as a fallback if the checkpoint directory has no tokenizer. Set to the student model you trained.

---

### Perplexity Eval (`run_eval.py`)
Loads the student model and computes **cross-entropy loss** on a held-out validation split of your training dataset.

**Output appended to:** `metrics.jsonl` as `{"step": N, "eval_loss": X, "perplexity": Y}`

**How to interpret:**
| Perplexity | Interpretation |
|------------|----------------|
| < 5 | Excellent — model fits the data distribution well |
| 5–15 | Good — typical range for a well-distilled small model |
| 15–30 | Fair — may need more epochs or a better dataset |
| > 30 | Poor — check for training bugs, wrong checkpoint, or data mismatch |

- **Compare teacher** — Also runs the teacher through the same eval and logs the gap. A good distilled student should get within 1.5–2× the teacher's perplexity.
- Run this after every training run as a quick sanity check.

---

### Quality Eval (`eval_quality.py`)
Generates responses from the student on `n_samples` prompts and measures **generation diversity**.

**Output:** `quality_metrics.json` in the model output dir.

**Metrics explained:**
| Metric | Good range | What it means |
|--------|-----------|----------------|
| `distinct_1` | > 0.15 | Fraction of unique unigrams. Low = repetitive vocabulary. |
| `distinct_2` | > 0.40 | Fraction of unique bigrams. Low = repetitive phrasing. |
| `max_repetition` | < 0.30 | Highest n-gram repetition rate in any single response. High = mode collapse. |
| `avg_length_tokens` | 50–200 | Average response length. Very short may indicate mode collapse. |
| `judge_score_mean` | 6–8 | LLM-as-judge score (1–10). Enable **LLM-as-judge** for this. |

- **LLM-as-judge** — Uses the teacher model to score each response 1–10. Adds ~5–10 minutes. Worth running once before deploying.
- Run this after perplexity eval confirms a reasonable loss.

---

### WikiText-2 Benchmark (`run_benchmarks.py`)
Evaluates the student on the **WikiText-2-raw-v1** test split — a standard open-domain NLP benchmark independent of your training data.

**Output:** `benchmark_results.json` + `metrics.jsonl` entry `{"wikitext2_perplexity": X}`

**Reference numbers (lower is better):**
| Model | WikiText-2 PPL |
|-------|---------------|
| Qwen2-0.5B-Instruct (base, no distillation) | ~18–22 |
| Well-distilled student (from 1.5B teacher) | ~14–18 |
| Qwen2-1.5B-Instruct (teacher) | ~10–13 |

- **Baseline dir** — Point at a previous run's output dir to get a regression comparison. If the new model is more than `threshold`% worse than the baseline, the benchmark prints a warning.
- Run this as a final check before exporting to GGUF.

---

### Recommended eval sequence
```
1. Perplexity Eval   → quick sanity check, ~2 min
2. Quality Eval      → verify generation diversity, ~5 min
3. WikiText-2        → standardized benchmark for comparison, ~5 min
4. Quality Eval with judge enabled  → final quality gate, ~15 min
```
""")

        # ── Expert Pipeline help ─────────────────────────────────────────
        with gr.Accordion("Expert Pipeline — Domain-Specific Distillation with CoT", open=False):
            gr.Markdown("""
The **Expert Pipeline** tab implements a 4-step workflow for building domain-specialist models
(tax, legal, medical, finance, coding) using any HuggingFace dataset and a GGUF teacher model.

---

### Step 1 — Dataset & Column Mapping
Load any HF dataset and remap its columns to the standard `instruction / input / output` format.

1. Enter the HF dataset ID (e.g. `nelson-liu/legalbench`) or a local path
2. Click **Inspect** — columns are auto-detected and dropdowns populated
3. Adjust column assignments if needed
4. Click **Remap & Save Dataset** → saves to the path in *Save remapped dataset to*

The remapped dataset is saved as an HF dataset on disk **and** as a `remapped.jsonl` for inspection.

**Supported datasets (tested):**

| Dataset | Domain | instruction_col | output_col |
|---------|--------|-----------------|------------|
| `Atome-LLM/Tax-Policy-Analysis` | Tax | `question` | `answer` |
| `nelson-liu/legalbench` | Legal | `instruction` | `output` |
| `medalpaca/medical_meadow_medical_flashcards` | Medical | `input` | `output` |
| `yahma/alpaca-cleaned` | General | `instruction` | `output` |
| `tatsu-lab/alpaca` | General | `instruction` | `output` |

---

### Step 2 — GGUF Teacher
Select a Metal-accelerated GGUF model from `/Users/Shared/llama/models/` to act as the teacher
for CoT generation. The model is served via `llama-server` during generation.

| Setting | Guidance |
|---------|----------|
| **Context size** | 8192 is safe for most models. Reduce to 4096 if the server OOMs. |
| **Parallel slots** | 4 keeps the GPU saturated. Reduce to 2 if getting timeouts. |
| **Temperature** | 0.3 for precise domains (tax, legal, medical). 0.7 for creative/coding. |
| **Max tokens** | 1024 covers most CoT responses. Increase to 2048 for complex reasoning. |

**Recommended GGUF teachers by domain:**

| Domain | Model | Quant | Why |
|--------|-------|-------|-----|
| Tax / Legal (best) | `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf` | Q4_K_M | Strongest reasoning at 70B; fits 36 GB unified |
| Legal (fast) | `law-chat.Q4_K_M.gguf` | Q4_K_M | Fine-tuned on statutory interpretation |
| Legal (alt) | `Llama-3-8B-Instruct-Legal-Q8_0.gguf` | Q8_0 | Highest-accuracy 8B legal model |
| Medical | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Q4_K_M | Best medical reasoning at 8B |
| Coding | `granite-3.1-8b-instruct-Q4_K_M.gguf` | Q4_K_M | IBM Granite excels at code |
| General | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Q4_K_M | Fast, good general quality |

> **Quantization guide:** Q4_K_M = best balance of size and quality. Q8_0 = highest accuracy, ~2× larger.
> All GGUF files go to `/Users/Shared/llama/models/` — click **Refresh** in the teacher dropdown to pick them up.

**Downloading GGUF models** — save all files to `/Users/Shared/llama/models/`:
```bash
cd /Users/Shared/llama/models

# 70B Llama-3 teacher (bartowski) — best for tax/legal CoT, needs ~42 GB RAM
curl -L -o Meta-Llama-3-70B-Instruct-Q4_K_M.gguf \\
  "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"

# Law-Chat 7B — statutory interpretation specialist
curl -L -o law-chat.Q4_K_M.gguf \\
  "https://huggingface.co/ricdomoliver/law-chat-GGUF/resolve/main/law-chat.Q4_K_M.gguf"

# Llama-3-8B Legal fine-tune (bartowski) — faster alternative
curl -L -o Llama-3-8B-Instruct-Legal-Q8_0.gguf \\
  "https://huggingface.co/bartowski/Llama-3-8B-Instruct-Legal-GGUF/resolve/main/Llama-3-8B-Instruct-Legal-Q8_0.gguf"
```
> **Note:** Verify the exact repo paths at huggingface.co before downloading — quantizer repos occasionally rename files.

**Downloading datasets:**
```bash
# LegalBench (Nguha, ~100 legal reasoning tasks) — use dataset ID in Step 1:
#   nguha/legalbench

# Tax-Policy-Analysis instruction data — use dataset ID in Step 1:
#   Atome-LLM/Tax-Policy-Analysis

# Download LegalBench as zip if you need an offline copy:
curl -L -o legalbench.zip \\
  "https://huggingface.co/datasets/nguha/legalbench/resolve/main/legalbench.zip"

# Tax parquet (if offline):
curl -L -o tax_data.parquet \\
  "https://huggingface.co/datasets/Atome-LLM/Tax-Policy-Analysis/resolve/main/data/train-00000-of-00001.parquet"
```
> **Tip:** For online runs just paste the HF dataset ID directly into the *Dataset ID* field — no download needed.

---

### Step 3 — CoT Rationale Generation
Prompts the GGUF teacher to generate **Chain-of-Thought reasoning traces** for each sample.

Each output is formatted as:
```
<reasoning>
[step-by-step domain analysis]
</reasoning>
<answer>
[final answer]
</answer>
```

The domain selector auto-fills a domain-specific system prompt (tax cites IRC sections,
legal applies statutory tests, medical follows clinical guidelines, etc.). You can override it.

**Output:** `cot_output_dir/cot_data.jsonl` + `cot_output_dir/hf_dataset/`

**Logs saved automatically to:**
- `runs/ep_cot_<domain>_<timestamp>.log` — plain text
- `runs/ep_cot_<domain>_<timestamp>.jsonl` — structured JSON (step, loss, ts, msg)

Both the **Training Loss** and **Gradient Norm** charts update live during this step,
and the full output streams in the **Live output** box below the accordions.

---

### Step 4 — Distillation
Launches `run_distillation_agent.py` on the CoT dataset. Leave *Dataset path* blank to
automatically use the `hf_dataset/` produced by Step 3.

**Key settings for domain expert models:**

| Setting | Recommended | Why |
|---------|-------------|-----|
| **LoRA rank** | 32 | Higher than default (16) — domain terminology requires more capacity |
| **Epochs** | 3 | CoT data is dense; 3 epochs ensures thorough absorption |
| **Backend** | MLX | Fastest on M3 Max; CoT outputs are long so MLX memory efficiency matters |

Distillation logs also save to `runs/ep_distill_<timestamp>.log/.jsonl`.

---

### Full expert pipeline (CLI)
```bash
# 1. Remap
python -m distill.expert_pipeline --mode remap \\
    --dataset Atome-LLM/Tax-Policy-Analysis \\
    --instruction_col question --output_col answer \\
    --output_dir ./domain_data/tax_expert --max_samples 5000

# 2. Generate CoT (phi-4 teacher, tax domain)
python -m distill.expert_pipeline --mode cot \\
    --dataset ./domain_data/tax_expert \\
    --teacher /Users/Shared/llama/models/phi-4-Q5_K_M.gguf \\
    --domain tax --n_samples 2000 \\
    --output_dir ./domain_data/tax_cot

# 3. Distill
python -m distill.expert_pipeline --mode distill \\
    --dataset ./domain_data/tax_cot/hf_dataset \\
    --output_dir ./runs/tax-expert \\
    --backend mlx --open --lora_r 32 --epochs 3
```
""")

        # ── Tab 4 help ───────────────────────────────────────────────────
        with gr.Accordion("Live Logs & Progress Bars — Reading Training Output", open=False):
            gr.Markdown("""
### Progress bars
A compact progress bar appears **on every tab** — updates every 2 seconds by parsing log output
for `Step X/Y`, `Epoch X/Y`, tqdm `75%|` patterns, or `Progress: N%`. Turns green on completion.

### Loss & Gradient charts
The **Live Logs** tab and the **Expert Pipeline** tab both show live **Training Loss** and
**Gradient Norm** line charts, updated every 2 seconds. Both MLX (`step=N  loss=X`) and
PyTorch (`{'loss': 'X', 'grad_norm': 'Y'}`) log formats are parsed automatically.

### Log format
Full training output streams in real time (polled every 2 seconds). stdout and stderr merged.

### Key metrics to watch (MiniLLM / GRPO)

| Metric | What it is | Healthy range | Action if outside range |
|--------|-----------|---------------|------------------------|
| `loss` | Training loss (reverse-KL) | Decreasing over first 50 steps | If rising after step 30, LR may be too high |
| `eval_loss` | Validation cross-entropy (logged every eval_steps) | Should decrease and track training loss | Large gap = overfitting |
| `reward` / `eval_reward` | Mean reward across completions in a batch | Should trend from negative toward positive | Stuck at -1.0 = clipping or mode collapse |
| `clipped_ratio` | Fraction of completions that hit max_completion_length | < 30% | If > 60%, lower Max completion length or increase it if responses should be long |
| `frac_reward_zero_std` | Fraction of prompt groups where all completions have identical reward (no GRPO gradient) | < 20% after step 50 | If > 50%, reduce Generations per prompt or check reward function |
| `kl` | KL divergence between student and teacher | Should be finite and decreasing | NaN or exploding = training diverged, stop and reduce LR |

### Key metrics (SFT)
| Metric | Healthy | Notes |
|--------|---------|-------|
| `loss` | Decreasing from ~3–5 to ~1–2 over 1 epoch | Fast decrease early = good. Plateau at >2 = check data quality |
| `grad_norm` | 0.5–2.0 | Spike to >10 = LR too high or data issue |

### Key metrics (MLX)
| Metric | Notes |
|--------|-------|
| `kd_loss` | Forward-KL distillation loss, should decrease |
| `ce_loss` | Cross-entropy loss component (scaled by ce_alpha) |
| `total_loss` | Weighted sum: `(1-ce_alpha)*kd_loss + ce_alpha*ce_loss` |
| `eval_ppl` | Validation perplexity, logged every eval_steps |

### Warning signs
- **`[Process exited with code -9]`** — You clicked Stop (SIGKILL). Normal.
- **`[Process exited with code 1]`** — Script crashed. Scroll up in logs for the Python traceback.
- **`OutOfMemoryError` / `MPS backend out of memory`** — Reduce Batch size or Max completion length. On M3 Max, batch=4 and completion=64 always fits.
- **`RuntimeError: Expected all tensors on same device`** — MPS + bfloat16 edge case. Usually resolves after a restart.
- **`ImportError: trl`** — TRL not installed. Run `pixi run pip install trl`.
- **`nan` in loss after step 1** — Learning rate too high. Reduce by 10×.
- **Reward stuck at -0.5 from step 1** — All completions are clipping. Increase Max completion length to 256+ so responses have room to terminate before the hard limit.
- **`frac_reward_zero_std` > 0.6 after step 20** — All completions in a group get identical reward so GRPO advantage is zero. Increase Generations per prompt to 4–8.

### Thermal note
On M3 Max, GPU temperature under MPS load typically stays at 50–60°C. If you see the machine throttling (iterations getting slower over time), training will continue correctly — the pause.flag watchdog can be used to cool it down without losing progress.

### After training completes
The log ends with:
```
Distilled model saved to ./distilled-minillm
[Process exited with code 0]
```
The merged weights are in the output directory. Go to **Eval** to validate, or run the export scripts (`scripts/export_student_gguf.sh`) to produce a GGUF for llama.cpp.
""")

        # ── Troubleshooting ──────────────────────────────────────────────
        with gr.Accordion("Troubleshooting & FAQ", open=False):
            gr.Markdown("""
### Q: Training is very slow (>300s per iteration)
- Reduce **Max completion length** to 64–96. Generation is the bottleneck — shorter completions = dramatically faster.
- Reduce **Generations per prompt** to 2 (minimum).
- Reduce **Eval every N steps** to 50+ to spend less time on evaluation.
- Switch backend to **MLX** (2–5× faster on M3).

### Q: Reward is stuck at -1.0 (mode collapse)
- Lower **Max completion length** — the model is generating nothing meaningful before hitting the hard limit.
- Increase **KD temperature** to 1.5 — softens the teacher targets, easier for student to match.
- Use SFT warmup first — gives the student a starting distribution to build from.

### Q: clipped_ratio is >80%
- The student is hitting max_completion_length on almost every generation.
- Lower **Max completion length** to 64 or 96 tokens.
- Or: the model is in a loop/repetition mode — check distinct-2 with Quality Eval.

### Q: Loss is NaN or exploding after a few steps
- **Learning rate too high.** Lower by 10×: 2e-5 → 2e-6 for MiniLLM, 2e-4 → 2e-5 for SFT.
- Check **grad_norm** in logs — if it's >10 before loss explodes, LR is definitely too high.

### Q: "No module named 'trl'" or "No module named 'transformers'"
```bash
cd /Users/caribou/distill
pixi run pip install trl transformers peft datasets
```
Or relaunch the UI through pixi: `pixi run python -m distill.launch_ui`

### Q: Can I run multiple jobs at once?
No — only one subprocess is managed by the UI at a time. If you need parallel runs, open a second terminal and call the scripts directly. The UI will show "A run is already in progress" if you try to launch while one is active.

### Q: Where are HuggingFace models cached?
Default: `~/.cache/huggingface/hub/`. Set `HF_HOME` environment variable to redirect. The dropdowns auto-scan this cache.

### Q: How do I use a locally downloaded GGUF model?
GGUFs are for inference only (llama.cpp / Ollama), not training. The training scripts use HF-format models. The pipeline exports to GGUF *after* training via `scripts/export_student_gguf.sh`.

### Q: The progress bar shows 0% but a run is active
The bar parses `Step X/Y`, `Epoch X/Y`, or tqdm `N%|` from the log. Scripts that don't emit
those patterns (e.g. model download phase) show 0% until the first progress line appears.
The Run status textbox ("running (pid …)") confirms the process is alive regardless.

### Q: The UI shows "idle" but I launched a run
Check **Live Logs** tab — the process may have crashed immediately. The status polling updates every 2 seconds.
Also check: if you launched from a tab other than Configure & Launch, only that tab's status textbox
updates on click; all tabs' progress bars update via the timer once the process writes output.
""")

        # ── Algorithm Reference ──────────────────────────────────────────
        gr.Markdown("---\n### Algorithm Reference")
        try:
            from ...ui.show_algorithms import ALGORITHMS, build_html as _build_html_help
            _help_html = _build_html_help(ALGORITHMS)
            _help_b64 = base64.b64encode(_help_html.encode("utf-8")).decode("ascii")
            gr.HTML(
                f'<div style="border-radius:10px;overflow:hidden;border:1px solid #2a2d3e;">'
                f'<iframe src="data:text/html;base64,{_help_b64}" '
                f'style="width:100%;height:80vh;border:none;" '
                f'sandbox="allow-scripts"></iframe>'
                f'</div>'
            )
        except Exception as _e:
            logger.error("Could not load algorithms for Help tab: %s", _e)
            gr.Markdown(f"⚠️ Could not load algorithms: `{_e}`")

    return {}
