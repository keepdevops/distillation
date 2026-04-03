"""Data Prep tab widget layout."""
from __future__ import annotations

import gradio as gr
from pathlib import Path

from ...infra.paths import project_dir

PROJECT_DIR = project_dir()


def build_tab_data_prep(teachers, datasets):
    """Build the Data Prep tab.

    Parameters
    ----------
    teachers:  list of teacher model choices
    datasets:  list of dataset choices

    Returns
    -------
    dict of all data prep widgets required for event wiring.
    """
    with gr.Tab("Data Prep"):
        data_prep_progress = gr.HTML(value="")

        # ── Magpie synthesis ─────────────────────────────────────────────
        gr.Markdown("### Magpie Synthesis")
        gr.Markdown(
            "Generate instruction-response pairs from the teacher by conditioning "
            "on its chat template. Produces an HF dataset in `output_dir/hf_dataset/` "
            "that can be used directly as a distillation dataset."
        )
        with gr.Row():
            mag_teacher = gr.Dropdown(
                choices=teachers,
                value="Qwen/Qwen2-1.5B-Instruct",
                label="Teacher model",
                allow_custom_value=True,
                scale=4,
            )
            mag_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
        with gr.Row():
            mag_output_dir = gr.Textbox(
                value=str(PROJECT_DIR / "magpie_data"),
                label="Output directory",
                scale=4,
            )
        with gr.Row():
            mag_n          = gr.Slider(500, 50000, value=5000, step=500,
                                       label="Pairs to generate (before filtering)")
            mag_batch_size = gr.Slider(1, 64, value=32, step=1,
                                       label="Generation batch size",
                                       info="For MLX this is loop chunk size (generation is sequential but fast)")
        with gr.Row():
            mag_backend = gr.Radio(
                ["auto", "mlx", "mps"], value="auto",
                label="Backend",
                info="auto picks MLX on Apple Silicon (2-4× faster than MPS)",
            )
            mag_filter  = gr.Checkbox(value=True, label="Filter output (dedup + quality)")
            mag_target  = gr.Slider(500, 20000, value=2000, step=500,
                                    label="Target keep (top-N after filter)")
            mag_offline = gr.Checkbox(value=False, label="Offline (use cached model)")
        with gr.Row():
            mag_launch_btn = gr.Button("Generate", variant="primary", scale=3)
            mag_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        mag_status = gr.Textbox(value="idle", label="Status", interactive=False)

        gr.Markdown("---")

        # ── Self-Instruct synthesis ───────────────────────────────────────
        gr.Markdown("### Self-Instruct Synthesis")
        gr.Markdown(
            "Generate synthetic instruction-response pairs via self-instruct: "
            "the teacher generates new instructions from seed examples, then "
            "generates responses. Includes perplexity + quality filtering. "
            "Output is an HF dataset in `output_dir/synthetic_data/`."
        )
        with gr.Row():
            synth_teacher = gr.Dropdown(
                choices=teachers,
                value="Qwen/Qwen2-1.5B-Instruct",
                label="Teacher model",
                allow_custom_value=True,
                scale=3,
            )
            synth_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
        with gr.Row():
            synth_use_open  = gr.Checkbox(value=True, label="Use open Qwen2 teacher (no HF login)")
            synth_offline   = gr.Checkbox(value=False, label="Offline (use cached model)")
        with gr.Row():
            synth_output_dir = gr.Textbox(
                value=str(PROJECT_DIR / "distilled-minillm"),
                label="Output directory",
                scale=4,
            )
        with gr.Row():
            synth_n_generate   = gr.Slider(100, 20000, value=2000, step=100,
                                           label="Target pairs to generate")
            synth_batch_size   = gr.Slider(1, 32, value=8, step=1,
                                           label="Batch size")
            synth_temperature  = gr.Slider(0.5, 1.5, value=0.9, step=0.1,
                                           label="Sampling temperature")
            synth_seed_examples = gr.Slider(2, 20, value=5, step=1,
                                            label="Seed examples per prompt")
        with gr.Row():
            synth_launch_btn = gr.Button("Synthesize", variant="primary", scale=3)
            synth_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        synth_status = gr.Textbox(value="idle", label="Status", interactive=False)

        gr.Markdown("---")

        # ── Dataset filter ────────────────────────────────────────────────
        gr.Markdown("### Dataset Filter")
        gr.Markdown(
            "Filter any alpaca-format dataset by quality: length bounds, "
            "distinct-2 coherence, refusal detection, and near-dedup (Jaccard). "
            "Output is an HF dataset loadable by the training scripts."
        )
        with gr.Row():
            filt_dataset = gr.Dropdown(
                choices=datasets,
                value="yahma/alpaca-cleaned",
                label="Dataset",
                allow_custom_value=True,
                scale=4,
            )
            filt_ds_refresh = gr.Button("Refresh", scale=1, size="sm")
        with gr.Row():
            filt_output_dir = gr.Textbox(
                value=str(PROJECT_DIR / "filtered_data"),
                label="Output directory",
                scale=4,
            )
        with gr.Row():
            filt_target    = gr.Slider(500, 50000, value=5000, step=500,
                                       label="Target top-N")
            filt_min_words = gr.Slider(5, 100, value=20, step=5,
                                       label="Min response words")
            filt_min_d2    = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                       label="Min distinct-2")
            filt_offline   = gr.Checkbox(value=False, label="Offline")
        with gr.Row():
            filt_launch_btn = gr.Button("Filter", variant="primary", scale=3)
            filt_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        filt_status = gr.Textbox(value="idle", label="Status", interactive=False)

    return {
        "data_prep_progress": data_prep_progress,
        "mag_teacher": mag_teacher,
        "mag_refresh_btn": mag_refresh_btn,
        "mag_output_dir": mag_output_dir,
        "mag_n": mag_n,
        "mag_batch_size": mag_batch_size,
        "mag_backend": mag_backend,
        "mag_filter": mag_filter,
        "mag_target": mag_target,
        "mag_offline": mag_offline,
        "mag_launch_btn": mag_launch_btn,
        "mag_stop_btn": mag_stop_btn,
        "mag_status": mag_status,
        "synth_teacher": synth_teacher,
        "synth_refresh_btn": synth_refresh_btn,
        "synth_use_open": synth_use_open,
        "synth_offline": synth_offline,
        "synth_output_dir": synth_output_dir,
        "synth_n_generate": synth_n_generate,
        "synth_batch_size": synth_batch_size,
        "synth_temperature": synth_temperature,
        "synth_seed_examples": synth_seed_examples,
        "synth_launch_btn": synth_launch_btn,
        "synth_stop_btn": synth_stop_btn,
        "synth_status": synth_status,
        "filt_dataset": filt_dataset,
        "filt_ds_refresh": filt_ds_refresh,
        "filt_output_dir": filt_output_dir,
        "filt_target": filt_target,
        "filt_min_words": filt_min_words,
        "filt_min_d2": filt_min_d2,
        "filt_offline": filt_offline,
        "filt_launch_btn": filt_launch_btn,
        "filt_stop_btn": filt_stop_btn,
        "filt_status": filt_status,
    }
