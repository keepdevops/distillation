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

    # Build Gradio interface
    with gr.Blocks(title="Universal Model Evaluator") as iface:
        gr.Markdown("# Universal Model Evaluator")
        gr.Markdown(
            "Supports **PyTorch**, **MLX**, **GGUF** (llama.cpp), and **vLLM** backends. "
            "Auto-detects model format. See the **Help** tab for usage reference."
        )

        with gr.Tabs():
            # Tab 1: Model Info & Loading
            with gr.Tab("📋 Model Info"):
                gr.Markdown(get_artifact_info())

                with gr.Row():
                    backend_selector = gr.Dropdown(
                        choices=["pytorch", "mlx", "gguf", "vllm"],
                        value=backend,
                        label="Backend",
                        info="Select model backend"
                    )
                    load_btn = gr.Button("🔄 Load Model", variant="primary")

                load_status = gr.Textbox(
                    label="Status",
                    value=f"Ready to load: {Path(path).name}",
                    interactive=False,
                    lines=4
                )

            # Tab 2: Generation
            with gr.Tab("✨ Generate"):
                gr.Markdown("### Enter a prompt to generate text")

                prompt_box = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your prompt here...",
                    lines=5
                )

                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        32, 512,
                        value=128,
                        step=32,
                        label="Max tokens"
                    )
                    temperature_slider = gr.Slider(
                        0.1, 2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )

                generate_btn = gr.Button("🚀 Generate", variant="primary", interactive=False)

                output_box = gr.Textbox(
                    label="Generated text",
                    lines=10,
                    interactive=False
                )

                gr.Examples(
                    examples=[
                        ["Explain quantum computing in simple terms.", 128, 0.7],
                        ["Write a haiku about machine learning.", 64, 0.9],
                        ["What are the benefits of distillation?", 150, 0.7],
                    ],
                    inputs=[prompt_box, max_tokens_slider, temperature_slider],
                )

            # Tab 3: Batch / Quality Evaluation
            with gr.Tab("🧪 Batch Eval"):
                gr.Markdown("### Batch Quality Evaluation")
                gr.Markdown(f"""
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

            # Tab 4: Help & Reference
            with gr.Tab("📖 Help"):
                gr.Markdown("""
## Quick Reference

### Backend Selection

| Backend | When to use |
|---------|-------------|
| **pytorch** | Standard HF model, LoRA adapter, or merged weights |
| **mlx** | MLX `.npz` weights or `mlx_q4/` directory — Apple Silicon only |
| **gguf** | `.gguf` quantized file (llama.cpp / llama-server) |
| **vllm** | High-throughput inference on NVIDIA GPU |

Auto-detection picks the right backend based on the files present.
Override with the dropdown if auto-detection is wrong.

### Model Path Examples

```
./distilled-minillm              # PyTorch full model
./distilled-minillm/sft_checkpoint  # SFT warmup checkpoint
./distilled-mlx                  # MLX training output
./distilled-mlx/mlx_q4          # MLX 4-bit quantized
./distilled-minillm/model-q4_K_M.gguf  # GGUF file
```

### Temperature Guide

| Range | Effect |
|-------|--------|
| 0.1–0.3 | Focused, near-deterministic |
| 0.5–0.8 | Balanced (default 0.7) |
| 1.0–1.5 | Creative, more varied |
| >1.5 | Exploratory / experimental |

### After a Full Pipeline Run

Your model directory contains:
```
your-model/
├── config.json              # HF model config
├── *.safetensors            # Model weights
├── metrics.jsonl            # Training loss/eval per step
├── quality_metrics.json     # eval_quality.py results
├── mlx_q4/                  # MLX 4-bit quantized weights
└── *.gguf                   # GGUF quantized file
```

### Running the Full Pipeline

```bash
# MLX backend → all export formats (recommended for M3)
python scripts/run_distillation_agent.py --open --backend mlx --export all

# PyTorch + GGUF only
python scripts/run_distillation_agent.py --open --export gguf

# With curriculum warmup + multi-trial search
python scripts/run_distillation_agent.py --open --backend mlx \\
  --curriculum --n_trials 3 --export all
```

### Training Watchdog

The watchdog monitors training and writes `pause.flag` on:
- **Plateau** — loss changes < 0.001 for 3+ steps
- **Divergence** — recent avg loss > early baseline × 1.5

```bash
python scripts/training_watchdog.py ./your-model --interval 60
```

### Thermal Protection

```bash
# System-wide (recommended — protects all jobs)
./scripts/install_thermal_agent.sh

# Manual (single session)
python scripts/thermal_agent.py --watch ./your-model --threshold 85
```
""")

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
