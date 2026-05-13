"""
Golden Pipeline tab for the Universal Gradio UI.

Renders controls for ``configs/golden_pipeline.json`` and streams
``distill.orchestration.agent`` output in real time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent.parent / "configs" / "golden_pipeline.json"


def _load_defaults() -> dict:
    """Return golden pipeline defaults from disk, or built-in fallback."""
    if _DEFAULT_CONFIG.exists():
        try:
            with open(_DEFAULT_CONFIG) as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Could not read golden_pipeline.json: %s", exc, exc_info=True)
    return {
        "backend": "mlx",
        "dataset": "yahma/alpaca-cleaned",
        "max_samples": 4000,
        "epochs": 3,
        "batch_size": 2,
        "grad_acc": 8,
        "learning_rate": 2e-4,
        "lora_r": 16,
        "temperature": 1.2,
        "ce_alpha": 0.2,
        "topk_logits": 50,
        "export": "gguf",
        "output_dir": "./runs/golden",
        "filter": True,
        "filter_target": 3000,
        "watchdog": True,
        "benchmarks": True,
        "skip_eval": False,
        "skip_judge": False,
        "seed": 42,
    }


def build_golden_tab() -> dict:
    """Render the 'Golden Pipeline' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys: ``dataset``, ``max_samples``, ``epochs``,
        ``batch_size``, ``grad_acc``, ``lr``, ``lora_r``, ``temperature``,
        ``ce_alpha``, ``export``, ``output_dir``, ``filter_chk``,
        ``filter_target``, ``watchdog_chk``, ``benchmarks_chk``,
        ``skip_eval``, ``skip_judge``, ``config_json``,
        ``run_btn``, ``stop_btn``, ``log_box``.
    """
    d = _load_defaults()

    gr.Markdown(
        "### \U0001f3c6 Golden Pipeline\n"
        "Full MLX distillation → filter → eval → GGUF export in one click. "
        "Edits here override `configs/golden_pipeline.json` for this run only."
    )

    with gr.Row():
        dataset = gr.Textbox(label="Dataset", value=d.get("dataset", "yahma/alpaca-cleaned"), scale=3)
        output_dir = gr.Textbox(label="Output dir", value=d.get("output_dir", "./runs/golden"), scale=2)

    with gr.Row():
        max_samples = gr.Slider(500, 20000, value=d.get("max_samples", 4000), step=500,
                                label="Max samples", scale=2)
        epochs = gr.Slider(1, 10, value=d.get("epochs", 3), step=1, label="Epochs", scale=1)
        batch_size = gr.Slider(1, 32, value=d.get("batch_size", 2), step=1, label="Batch size", scale=1)
        grad_acc = gr.Slider(1, 32, value=d.get("grad_acc", 8), step=1, label="Grad accum", scale=1)

    with gr.Row():
        lr = gr.Number(label="Learning rate", value=d.get("learning_rate", 2e-4), precision=6, scale=1)
        lora_r = gr.Slider(4, 64, value=d.get("lora_r", 16), step=4, label="LoRA r", scale=1)
        temperature = gr.Slider(0.5, 2.5, value=d.get("temperature", 1.2), step=0.05,
                                label="KD temperature", scale=2)
        ce_alpha = gr.Slider(0.0, 1.0, value=d.get("ce_alpha", 0.2), step=0.05,
                             label="CE alpha", scale=2)

    with gr.Row():
        export = gr.Dropdown(
            choices=["gguf", "coreml", "mlx_quant", "all", "none"],
            value=d.get("export", "gguf"),
            label="Export format",
            scale=1,
        )
        filter_chk = gr.Checkbox(label="Filter dataset", value=d.get("filter", True), scale=1)
        filter_target = gr.Number(label="Filter target samples", value=d.get("filter_target", 3000),
                                  precision=0, scale=1)

    with gr.Row():
        watchdog_chk = gr.Checkbox(label="Watchdog (plateau stop)", value=d.get("watchdog", True))
        benchmarks_chk = gr.Checkbox(label="Run benchmarks", value=d.get("benchmarks", True))
        skip_eval = gr.Checkbox(label="Skip eval", value=d.get("skip_eval", False))
        skip_judge = gr.Checkbox(label="Skip judge", value=d.get("skip_judge", False))

    with gr.Accordion("Raw config JSON (read-only preview)", open=False):
        config_json = gr.JSON(value=d, label="Resolved config")

    with gr.Row():
        run_btn = gr.Button("\U0001f3c6 Run Golden Pipeline", variant="primary",
                            elem_classes="btn-primary", min_width=220)
        stop_btn = gr.Button("\u23f9 Stop", variant="stop", min_width=100)

    log_box = gr.Textbox(label="Pipeline log", lines=20, interactive=False,
                         elem_classes="output-box")

    return {
        "dataset": dataset,
        "max_samples": max_samples,
        "epochs": epochs,
        "batch_size": batch_size,
        "grad_acc": grad_acc,
        "lr": lr,
        "lora_r": lora_r,
        "temperature": temperature,
        "ce_alpha": ce_alpha,
        "export": export,
        "output_dir": output_dir,
        "filter_chk": filter_chk,
        "filter_target": filter_target,
        "watchdog_chk": watchdog_chk,
        "benchmarks_chk": benchmarks_chk,
        "skip_eval": skip_eval,
        "skip_judge": skip_judge,
        "config_json": config_json,
        "run_btn": run_btn,
        "stop_btn": stop_btn,
        "log_box": log_box,
    }
