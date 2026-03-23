#!/usr/bin/env python3
"""
Universal Gradio UI for evaluating distilled models.
Supports PyTorch, MLX, GGUF, and vLLM backends.
Auto-detects model format and available artifacts.
Runs locally on 127.0.0.1 only (no public share).
"""

import argparse
import os
import sys
from pathlib import Path

# Add scripts dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from universal_model_loader import UniversalModelLoader, detect_model_format, ModelFormat
from artifact_detector import detect_artifacts, format_artifact_summary


def parse_args():
    from model_path_helper import resolve_model_path, get_model_base_path

    p = argparse.ArgumentParser(description="Universal model evaluation UI")

    # Default to MODEL_PATH/distilled-minillm if MODEL_PATH is set
    default_path = str(get_model_base_path() / "distilled-minillm")

    p.add_argument("--model_path", type=str, default=default_path,
                   help="Path to model directory or GGUF file (default: $MODEL_PATH/distilled-minillm)")
    p.add_argument("--backend", type=str, default=None,
                   choices=["pytorch", "mlx", "gguf", "vllm"],
                   help="Force specific backend (auto-detects if not specified)")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def main():
    args = parse_args()
    from model_path_helper import resolve_model_path
    path = resolve_model_path(args.model_path)

    # Check path exists
    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        print("Provide an existing model directory or GGUF file with --model_path")
        raise SystemExit(1)

    print(f"Analyzing path: {path}")

    # Detect artifacts if it's a directory
    artifacts_info = None
    if os.path.isdir(path):
        artifacts_info = detect_artifacts(path)
        print(f"\nDetected formats: {', '.join(artifacts_info['formats']) if artifacts_info['formats'] else 'None'}")
        print(f"Training method: {artifacts_info['training_method']}")
        if artifacts_info['artifacts']:
            print(f"\n{format_artifact_summary(artifacts_info['artifacts'])}")

    # Detect model format
    detected_format = detect_model_format(path)
    backend = args.backend if args.backend else detected_format.value

    print(f"\nBackend: {backend}")
    print(f"Starting Gradio UI on http://127.0.0.1:{args.port}")

    # Create universal loader
    loader = UniversalModelLoader()

    import gradio as gr
    import base64

    # State to track loaded model info
    model_loaded = {"loaded": False, "message": "No model loaded"}

    def load_model_fn(selected_backend):
        """Load model with selected backend."""
        nonlocal model_loaded

        if not selected_backend:
            selected_backend = backend

        print(f"\nLoading model with backend: {selected_backend}")
        success, message = loader.load(path, backend=selected_backend)

        if success:
            info = loader.get_info()
            model_loaded["loaded"] = True
            model_loaded["message"] = message
            status_msg = f"✅ {message}\n\nBackend: {info['backend']}"
            if 'device' in info:
                status_msg += f"\nDevice: {info['device']}"
            if 'dtype' in info:
                status_msg += f"\nDtype: {info['dtype']}"
            return status_msg, gr.update(interactive=True)
        else:
            model_loaded["loaded"] = False
            model_loaded["message"] = message
            return f"❌ {message}", gr.update(interactive=False)

    def generate_fn(prompt, max_tokens, temperature):
        """Generate text from prompt."""
        if not model_loaded["loaded"]:
            return "⚠️ Please load a model first"

        if not prompt.strip():
            return ""

        print(f"\nGenerating (max_tokens={max_tokens}, temp={temperature})")
        result = loader.generate(prompt, max_new_tokens=int(max_tokens), temperature=temperature)
        return result

    def get_artifact_info():
        """Get formatted artifact information."""
        if artifacts_info is None:
            return "Not a directory - single file loaded"

        lines = [
            f"**Path:** `{Path(path).name}`",
            f"**Training method:** {artifacts_info['training_method']}",
            f"**Available formats:** {', '.join(artifacts_info['formats']) if artifacts_info['formats'] else 'None'}",
        ]

        if artifacts_info['has_metrics']:
            lines.append(f"**Metrics:** ✅ Available")
        else:
            lines.append(f"**Metrics:** ❌ Not found")

        if artifacts_info['checkpoints']:
            ckpts = ", ".join([f"step {c['step']}" for c in artifacts_info['checkpoints'][:5]])
            if len(artifacts_info['checkpoints']) > 5:
                ckpts += f" (+{len(artifacts_info['checkpoints']) - 5} more)"
            lines.append(f"**Checkpoints:** {ckpts}")

        if artifacts_info['artifacts']:
            lines.append("\n**Artifacts:**")
            for name, typ, _, size_gb in artifacts_info['artifacts']:
                lines.append(f"- `{name}` ({typ}, {size_gb:.2f} GB)")

        return "\n".join(lines)

    CUSTOM_CSS = """
    /* ── Global ─────────────────────────────────────────────── */
    .gradio-container { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important; }

    /* ── Header banner ──────────────────────────────────────── */
    .app-header {
        background: linear-gradient(135deg, #1a1d2e 0%, #0f1117 100%);
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.4rem 2rem;
        margin-bottom: 0.5rem;
    }
    .app-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(135deg, #7c6af7, #4fc3f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.25rem 0;
    }
    .app-header p { color: #8b8fa8; font-size: 0.9rem; margin: 0; }

    /* ── Tabs ────────────────────────────────────────────────── */
    .tab-nav button {
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em;
    }
    .tab-nav button.selected {
        border-bottom: 3px solid #7c6af7 !important;
        color: #7c6af7 !important;
    }

    /* ── Cards / panels ─────────────────────────────────────── */
    .card {
        background: #1a1d27;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
    }

    /* ── Status box ─────────────────────────────────────────── */
    .status-ok  textarea { border-left: 4px solid #4caf50 !important; color: #4caf50 !important; }
    .status-err textarea { border-left: 4px solid #f44336 !important; color: #f44336 !important; }

    /* ── Buttons ─────────────────────────────────────────────── */
    .btn-primary { border-radius: 8px !important; font-weight: 600 !important; }
    .generate-row { gap: 1rem; align-items: flex-end; }

    /* ── Output box ─────────────────────────────────────────── */
    .output-box textarea {
        background: #13151f !important;
        border: 1px solid #2a2d3e !important;
        border-radius: 8px !important;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        font-size: 0.88rem !important;
    }

    /* ── Algo iframe container ───────────────────────────────── */
    .algo-frame { border-radius: 10px; overflow: hidden; border: 1px solid #2a2d3e; }
    """

    # Build Gradio interface
    with gr.Blocks(title="Universal Model Evaluator", css=CUSTOM_CSS,
                   theme=gr.themes.Soft(
                       primary_hue="violet",
                       secondary_hue="cyan",
                       neutral_hue="slate",
                   )) as iface:

        with gr.Column(elem_classes="app-header"):
            gr.HTML("<h1>Universal Model Evaluator</h1>"
                    "<p>Supports <b>PyTorch</b>, <b>MLX</b>, <b>GGUF</b> (llama.cpp), and <b>vLLM</b> backends. "
                    "Auto-detects model format &nbsp;·&nbsp; See <b>Help</b> tab for reference.</p>")

        with gr.Tabs():
            # ── Tab 1: Model Info & Loading ─────────────────────────
            with gr.Tab("📋 Model Info"):
                gr.Markdown(get_artifact_info())

                with gr.Row(equal_height=True):
                    backend_selector = gr.Dropdown(
                        choices=["pytorch", "mlx", "gguf", "vllm"],
                        value=backend,
                        label="Backend",
                        info="Override auto-detected backend",
                        scale=2,
                    )
                    load_btn = gr.Button("🔄 Load Model", variant="primary",
                                        elem_classes="btn-primary", scale=1, min_width=140)

                load_status = gr.Textbox(
                    label="Status",
                    value=f"Ready to load: {Path(path).name}",
                    interactive=False,
                    lines=3,
                )

            # ── Tab 2: Generation ───────────────────────────────────
            with gr.Tab("✨ Generate"):
                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here…",
                    lines=6,
                )

                with gr.Row(elem_classes="generate-row"):
                    max_tokens_slider = gr.Slider(32, 2048, value=256, step=32, label="Max tokens", scale=3)
                    temperature_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature", scale=3)
                    generate_btn = gr.Button("🚀 Generate", variant="primary",
                                            elem_classes="btn-primary", interactive=False, scale=1, min_width=140)

                output_box = gr.Textbox(
                    label="Generated text",
                    lines=14,
                    interactive=False,
                    elem_classes="output-box",
                    show_copy_button=True,
                )

                gr.Examples(
                    examples=[
                        ["Explain quantum computing in simple terms.", 256, 0.7],
                        ["Write a haiku about machine learning.", 64, 0.9],
                        ["What are the benefits of knowledge distillation?", 200, 0.7],
                        ["Write a Python function to compute Fibonacci numbers.", 300, 0.3],
                    ],
                    inputs=[prompt_box, max_tokens_slider, temperature_slider],
                    label="Quick examples",
                )

            # ── Tab 3: Batch / Quality Evaluation ──────────────────
            with gr.Tab("🧪 Batch Eval"):
                gr.Markdown(f"""
### Batch Quality Evaluation

Run `eval_quality.py` on **{Path(path).name}** for comprehensive quality metrics:

```bash
# Basic quality eval (diversity + quality gates)
python scripts/eval_quality.py {path}

# With LLM-as-judge scoring
python scripts/eval_quality.py {path} --judge --teacher Qwen/Qwen2-1.5B-Instruct

# With teacher perplexity on student outputs (per-sample, not batch-average)
python scripts/eval_quality.py {path} --judge-teacher-ppl

# All metrics together
python scripts/eval_quality.py {path} --judge --judge-teacher-ppl
```

**What it measures:**

| Metric | Description |
|--------|-------------|
| Distinct-1 / Distinct-2 | Lexical diversity (higher = more varied) |
| 3-gram entropy | Generation variety across all outputs |
| Refusal rate | % of outputs that are refusals (gate: <5%) |
| Quality gate pass rate | % passing length + refusal filters |
| Category distribution | math / code / creative / reasoning / qa / other |
| Teacher PPL (per sample) | Per-sample teacher perplexity on student outputs |
| Judge score (1–10) | LLM-as-judge instruction-following quality |

Results saved to: `{path}/quality_metrics.json`
""")

            # ── Tab 4: Algorithm Reference ──────────────────────────
            with gr.Tab("📐 Algorithms"):
                try:
                    import sys as _sys
                    _sys.path.insert(0, str(Path(__file__).parent))
                    from show_algorithms import ALGORITHMS, build_html as _build_html
                    _algo_html = _build_html(ALGORITHMS)
                    # Base64-encode to avoid escaping issues with LaTeX backslashes
                    _b64 = base64.b64encode(_algo_html.encode("utf-8")).decode("ascii")
                    gr.HTML(
                        f'<div class="algo-frame">'
                        f'<iframe src="data:text/html;base64,{_b64}" '
                        f'style="width:100%;height:84vh;border:none;" '
                        f'sandbox="allow-scripts"></iframe>'
                        f'</div>'
                    )
                except Exception as _e:
                    gr.Markdown(f"⚠️ Could not load algorithms: `{_e}`\n\nRun `python scripts/show_algorithms.py` directly.")

            # ── Tab 5: Help & Reference ─────────────────────────────
            with gr.Tab("📖 Help"):
                gr.Markdown("""
# Distillation System — Complete Reference

---

## 1. Pipeline Architecture

The system runs in sequential stages orchestrated by `run_distillation_agent.py`:

```
[Data] → [Stage 1: SFT Warmup] → [Stage 2: KD Training] → [Eval] → [Export]
            distill_sft.py         distill_mlx.py               run_eval.py    export_student_gguf.sh
                                   distill_minillm.py            run_benchmarks.py
```

**Choose your path:**

| Goal | Recommended command |
|------|---------------------|
| Fastest on M3 Max | `--backend mlx --export all` |
| Largest student, best quality | `--backend pytorch --export gguf` |
| Domain expert (tax/legal/medical) | `expert_pipeline.py --mode distill` |
| Multi-trial hyperparameter search | `--n_trials 3 --curriculum` |
| Production run (all features) | `run_golden.sh` |

```bash
# Standard MLX run (recommended for M3 Max)
python scripts/run_distillation_agent.py \\
  --open --backend mlx --export all --curriculum --watchdog

# Full production (3 trials, synthetic data, benchmarks, judge)
bash scripts/run_golden.sh
```

---

## 2. Training Backends

### `distill_mlx.py` — MLX Forward KL (recommended for M3 Max)

Apple-native MLX training. **2–5× faster** than PyTorch MPS. Teacher logits are
pre-computed once as top-K sparse tensors (~300 MB vs 311 GB full vocab), then frozen.
Training runs the combined forward KL + CE loss with optional linear annealing.

```bash
python scripts/distill_mlx.py \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-mlx \\
  --epochs 3 --batch_size 2 --grad_acc 8 \\
  --kd_temp 1.0 --ce_alpha 0.1 --topk_logits 50 \\
  --lora_r 8 --q_bits 4 --watchdog
```

Key flags:
- `--topk_logits 50` — only keep top-50 teacher logits per token (>99% of teacher mass)
- `--ce_alpha 0.1` — blend: 10% CE + 90% KD loss; 0 = pure KD, 1 = pure CE
- `--kd_temp` — distillation temperature (higher = softer teacher distribution)
- `--q_bits 4` — quantize to 4-bit after training (MLX Q4)
- `--multi_turn_ratio` — fraction of multi-turn chat samples in each batch
- `--resume` — continue from last checkpoint if interrupted

Outputs: `mlx_student_weights.npz`, `mlx_q4/` quantized dir, `metrics.jsonl`

---

### `distill_minillm.py` — Reverse KL / GRPO (MiniLLM)

Trains via **Group Relative Policy Optimization**: sample G completions per prompt,
score with a reward function, compute group-normalized advantage, clip importance ratio.
Approximates the intractable reverse KL gradient without requiring full teacher rollouts.

```bash
python scripts/distill_minillm.py \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-minillm \\
  --epochs 2 --batch_size 8 --grad_acc 8 \\
  --lora_r 64 --learning_rate 2e-5 --watchdog
```

Reward function: `+0.5` (clean output), `-1.0` (< 10 tokens, mode collapse),
`-0.5` (> 800 tokens, runaway). Group advantage normalises scores within each prompt's
G completions before clipping the importance ratio at `±ε`.

Outputs: HuggingFace checkpoint, `adapter_model.bin`, `trainer_state.json`, `metrics.jsonl`

---

### `distill_sft.py` — SFT Warmup (Stage 1)

Teacher generates greedy completions for every prompt. Student minimises cross-entropy
**on response tokens only** — prompt and padding positions are masked to -100.
Run before Stage 2 KD to prevent cold-start instability.

```bash
python scripts/distill_sft.py \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-sft \\
  --epochs 1 --batch_size 4 --lora_r 64
```

Outputs: `sft_checkpoint/`, `sft_labels.jsonl` (cached teacher completions — reused
on reruns to avoid re-generating).

---

### `distill_forward.py` — Hinton KD for Classification

Vanilla forward KL distillation for **encoder classification models** (BERT, DistilBERT).
Temperature-scaled soft targets combined with hard CE loss. Not for causal LLMs.

```bash
python scripts/distill_forward.py \\
  --teacher bert-base-uncased --student distilbert-base-uncased \\
  --dataset glue --dataset_config sst2 \\
  --temperature 5.0 --alpha 0.5
```

---

## 3. Orchestrator

### `run_distillation_agent.py` — Autonomous Pipeline

Runs all stages end-to-end, streams live metrics, supports multi-trial hyperparameter
search, and logs every run to `experiment_log.jsonl` for later comparison.

```bash
python scripts/run_distillation_agent.py \\
  --open                     # use OpenHermes dataset
  --backend mlx              # mlx | pytorch | unsloth
  --export all               # gguf | coreml | all
  --curriculum               # SFT warmup before KD
  --n_trials 3               # hyperparameter sweep (3 independent runs)
  --synthetic_data           # augment with Magpie-synthesised pairs
  --benchmarks               # run WikiText-2 perplexity after training
  --watchdog                 # enable loss plateau / divergence detection
  --config configs/golden_pipeline.json  # override defaults with JSON
```

Multi-trial search perturbs `learning_rate`, `batch_size`, `kd_temp`, `ce_alpha`
and picks the trial with the lowest final eval loss. Results logged to
`experiment_log.jsonl` with `propose_next()` suggestions for the following run.

---

## 4. Data Pipeline

### `data_pipeline.py` — Dataset Loading & Formatting

Auto-detects schema (alpaca, sharegpt, messages, DPO, guanaco) and normalises to
`instruction / input / output` triples. Used by every training script.

Supported datasets (pass by HF hub ID or local path):
- `tatsu-lab/alpaca`, `yahma/alpaca-cleaned`
- `teknium/OpenHermes-2.5`
- `HuggingFaceH4/no_robots`
- `argilla/distilabel-capybara-dpo-7k-binarized`
- `mlabonne/guanaco-llama2-1k`
- Any local JSONL with `instruction`/`output` or `messages` fields

Quality filters applied automatically: min 20 / max 600 response words,
`distinct-2 > 0.35`, refusal detection (regex), noise detection (N/A, "I don't know", etc.).

### `filter_dataset.py` — Quality Filtering

Reduces a raw dataset to 8–20k high-quality pairs using composite scoring
(distinct-2, instruction complexity, response variety) + near-dedup via Jaccard similarity.
Optional teacher NLL re-ranking keeps only examples the teacher finds easy (high confidence).

```bash
python scripts/filter_dataset.py \\
  --dataset teknium/OpenHermes-2.5 \\
  --output_dir ./filtered_data \\
  --target 8000
```

### `magpie_synth.py` — Self-Synthesis (Magpie)

Generates synthetic instruction-response pairs by conditioning the teacher on a
user-turn prefix and letting it complete both the instruction and the response.
Inline filters reject short, repetitive, or refusal outputs.

```bash
python scripts/magpie_synth.py \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --n 5000 --target 3000 \\
  --domain math --output_dir ./magpie_data
```

### `expert_pipeline.py` — Domain Expert Distillation

Four modes for domain-specific CoT distillation (tax, legal, medical):

```bash
# 1. Inspect a dataset's columns
python scripts/expert_pipeline.py --mode inspect --dataset my_data.jsonl

# 2. Remap columns to instruction/output
python scripts/expert_pipeline.py --mode remap \\
  --dataset my_data.jsonl --instruction_col question --output_col answer

# 3. Generate Chain-of-Thought via GGUF teacher
python scripts/expert_pipeline.py --mode cot --domain tax

# 4. Run full distillation on CoT data
python scripts/expert_pipeline.py --mode distill --domain tax
```

CoT outputs go to `domain_data/expert_cot/` and `domain_data/expert_remapped/`.

---

## 5. Evaluation

### `eval_quality.py` — Generation Quality Metrics

Generates `n_samples` completions from the model and measures diversity and quality.

```bash
python scripts/eval_quality.py ./distilled-mlx \\
  --judge --teacher Qwen/Qwen2-1.5B-Instruct \\
  --judge-teacher-ppl --n_samples 200
```

| Metric | Description | Good threshold |
|--------|-------------|----------------|
| Distinct-1 | Unique unigram fraction | > 0.3 |
| Distinct-2 | Unique bigram fraction | > 0.5 |
| 3-gram entropy | Cross-output variety | > 8.0 |
| Refusal rate | % refusals | < 5% |
| Quality gate pass | Length + refusal filter | > 90% |
| Judge score (1–10) | LLM-as-judge | > 7.0 |
| Teacher PPL | Per-sample perplexity | lower = better |

Results saved to `quality_metrics.json`.

### `run_eval.py` — Validation Loss

Computes cross-entropy loss on the validation split and appends `eval_loss` /
`perplexity` to `metrics.jsonl`. Skipped automatically for MLX backend
(distill_mlx.py handles eval internally).

```bash
python scripts/run_eval.py ./distilled-minillm \\
  --backend auto --compare_teacher --max_val_samples 500
```

### `run_benchmarks.py` — WikiText-2 Perplexity

Evaluates on WikiText-2 test set (500 sequences by default) and detects regressions
vs a baseline run. A >15% increase in perplexity triggers a warning.

```bash
python scripts/run_benchmarks.py ./distilled-mlx \\
  --baseline_dir ./reference-model --n_sequences 500
```

Results appended to `metrics.jsonl` as `wikitext2_perplexity`.

---

## 6. Inference Backends

### `mlx_eval_utils.py` — MLX (Metal GPU)

Loads MLX `.npz` weights or `mlx_q4/` directory via `mlx-lm`. Handles LoRA adapter
merging automatically. **3–5× faster** than PyTorch MPS for forward-only eval.

### `cpp_eval_utils.py` — llama.cpp (fastest)

Uses binaries at `/Users/Shared/llama/`: `llama-server` for parallel slot-based
generation, `llama-perplexity` for chunked PPL. **8–15× faster** than MLX for PPL
eval on M3 Max. Auto-detects GGUF files (prefers Q4_K_M > Q8_0 > F16).

### `universal_model_loader.py` — Auto-detection

Format detection priority:
1. File ends in `.gguf` → **gguf**
2. Dir contains `mlx_model.npz` / `mlx_student_weights.npz` → **mlx**
3. Dir contains `mlx_q4/` subdir → **mlx**
4. Dir contains `adapter_config.json` + `*.safetensors` → **pytorch** (LoRA)
5. Dir contains `*.safetensors` / `pytorch_model.bin` → **pytorch**

Override with `--backend` flag or the dropdown in this UI.

### Backend comparison

| Backend | Speed (M3 Max) | Memory | Use case |
|---------|---------------|--------|----------|
| **gguf** | 8–15× baseline | ~2 GB Q4 | Production inference, PPL benchmarks |
| **mlx** | 3–5× baseline | unified 4–8 GB | Training + eval on Apple Silicon |
| **pytorch** | baseline | 4–16 GB | Full fine-tuning, LoRA merge |
| **vllm** | 5–10× baseline | NVIDIA VRAM | High-throughput NVIDIA GPU serving |

---

## 7. Monitoring & Protection

### `training_watchdog.py` — Loss Monitor

Reads `trainer_state.json` every `--interval` seconds and writes `pause.flag` if:
- **Plateau**: last N loss deltas all < 0.001 (configurable in `watchdog_rules.json`)
- **Divergence**: recent avg loss > early baseline × 1.5

All training loops call `check_pause_flag()` before each step and save-then-exit cleanly
when `pause.flag` is present. The flag file contains JSON with the reason and metadata.

```bash
python scripts/training_watchdog.py ./distilled-mlx \\
  --interval 60 --config scripts/watchdog_rules.json
```

### `thermal_agent.py` — Hardware Protection

Polls `mactop` every 30 s for CPU/GPU/SoC temperatures. Writes `pause.flag` to all
watched directories when any metric exceeds `--threshold` (default 85°C), then clears
it automatically once temps drop by 5°C (hysteresis).

```bash
# One-time session
python scripts/thermal_agent.py \\
  --watch ./distilled-mlx ./distilled-minillm \\
  --threshold 85 --metric soc_temp_c --interval 30

# Install as always-on LaunchAgent (survives reboot)
bash scripts/install_thermal_agent.sh
```

Observed idle: CPU ~44°C, GPU ~39°C. Under MPS load: CPU ~50°C, GPU ~57°C.
Threshold of 85–90°C gives substantial headroom.

### `monitor_cpu_gpu_temp.py` — Live Thermal Display

Real-time temperature + fan control. Logs to CSV. Optionally drives Macs Fan Control.

```bash
python scripts/monitor_cpu_gpu_temp.py --interval 3 --log thermal.log
```

---

## 8. Outputs & Artifacts

After a full pipeline run your output directory contains:

```
your-model/
├── config.json                  # HuggingFace model config
├── *.safetensors                # PyTorch model weights
├── adapter_config.json          # LoRA config (if not merged)
├── adapter_model.bin            # LoRA weights (if not merged)
├── mlx_student_weights.npz      # MLX LoRA weights
├── mlx_q4/                      # MLX 4-bit quantized (safetensors + config)
├── model-q4_K_M.gguf            # GGUF 4-bit quantized (llama.cpp)
├── model.mlpackage/             # CoreML (Apple Neural Engine, if exported)
├── metrics.jsonl                # Loss, LR, grad norm, eval PPL per step
├── trainer_state.json           # HF Trainer checkpoint state
├── quality_metrics.json         # eval_quality.py results
├── benchmark_results.json       # WikiText-2 perplexity
├── sft_labels.jsonl             # Cached teacher greedy completions (Stage 1)
└── experiment_log.jsonl         # Multi-trial run history
```

---

## 9. Export

### GGUF (llama.cpp)

```bash
bash scripts/export_student_gguf.sh ./distilled-minillm
# Output: /Users/Shared/llama/models/distilled-minillm-Q4_K_M.gguf
```

Calls `convert_hf_to_gguf.py` from llama.cpp at `/Users/Shared/llama/`.
All GGUF exports go to `/Users/Shared/llama/models/`.

### CoreML (Apple Neural Engine)

```bash
python scripts/export_coreml.py \\
  --model_dir ./distilled-minillm \\
  --quantize int4 --output_dir ./coreml_export
# Output: ./coreml_export/model.mlpackage
```

### MLX Quantize (in-pipeline)

Pass `--q_bits 4` or `--q_bits 8` to `distill_mlx.py` — quantization runs automatically
at the end of training and saves to `mlx_q4/` or `mlx_q8/`.

---

## 10. Session Management

```bash
# Start watchdog + dashboard in tmux (recommended)
bash scripts/start.sh --backend mlx --eval

# Attach to running session
tmux attach -t distill

# Stop everything cleanly
bash scripts/stop.sh

# Emergency kill UI only
bash scripts/kill_ui.sh
```

`start.sh` opens tmux windows: **watchdog** (training_watchdog.py),
**dashboard** (dashboard.py or eval_gradio.py), and optionally **thermal** monitor.
`caffeinate` keeps the machine awake during long training runs.

---

## 11. Offline / Air-Gap Setup

```bash
# Pre-download models + datasets
python scripts/cache_models.py      # → hf_cache/
python scripts/cache_datasets.py    # → datasets_cache/

# Or all-in-one
python scripts/setup_airgap.py

# Setup shared model directory
bash scripts/setup_shared_models.sh
# Sets MODEL_PATH=/Users/Shared/models in shell profile
```

Pass `--offline` to any training or eval script to disable network access entirely.

---

## 12. Temperature Guide

| Value | Effect |
|-------|--------|
| 0.0 | Greedy — fully deterministic |
| 0.1–0.3 | Near-deterministic, focused |
| 0.5–0.8 | Balanced — default 0.7 |
| 1.0–1.5 | Creative, more varied outputs |
| > 1.5 | Exploratory / experimental |
""")
                gr.Markdown("---\n### Algorithm Reference")
                try:
                    from show_algorithms import ALGORITHMS, build_html as _build_html_help
                    _help_html = _build_html_help(ALGORITHMS)
                    _help_b64 = base64.b64encode(_help_html.encode("utf-8")).decode("ascii")
                    gr.HTML(
                        f'<div class="algo-frame">'
                        f'<iframe src="data:text/html;base64,{_help_b64}" '
                        f'style="width:100%;height:80vh;border:none;" '
                        f'sandbox="allow-scripts"></iframe>'
                        f'</div>'
                    )
                except Exception as _e:
                    gr.Markdown(f"⚠️ Could not load algorithms: `{_e}`")

        # Event handlers
        load_btn.click(
            fn=load_model_fn,
            inputs=[backend_selector],
            outputs=[load_status, generate_btn]
        )

        generate_btn.click(
            fn=generate_fn,
            inputs=[prompt_box, max_tokens_slider, temperature_slider],
            outputs=[output_box]
        )

        # Auto-load on startup if backend is specified
        if args.backend or detected_format != ModelFormat.UNKNOWN:
            iface.load(
                fn=lambda: load_model_fn(backend),
                outputs=[load_status, generate_btn]
            )

    iface.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False
    )


if __name__ == "__main__":
    main()
