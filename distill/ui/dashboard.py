#!/usr/bin/env python3
"""
Unified Gradio dashboard: training plots + model evaluation.
Runs locally on 127.0.0.1. Air-gapped friendly.
Supports PyTorch, MLX, GGUF, and vLLM backends.
"""

import argparse
import logging

import gradio as gr
import matplotlib

logger = logging.getLogger(__name__)
matplotlib.use("Agg")

from ..backends.universal_loader import UniversalModelLoader, detect_model_format
from ..infra.artifact_detector import detect_artifacts
from ..backends.mlx_utils import is_mlx_available, load_mlx_model, mlx_generate_responses
from ..backends.cpp_utils import find_gguf
from ..infra.metrics_io import load_trainer_state
from .dashboard_plots import plot_from_state, on_run_select
from .dashboard_models import select_and_load_model, _diversity_metrics, _discover_all_models
from .dashboard_streaming import (
    _run_streaming, _parse_progress_from_log, _progress_bar_html, _is_streaming_done,
)
from .dashboard_eval_ui import build_eval_ui
from .dashboard_run_evals import build_run_evals_ui
from .dashboard_discovery import find_run_dirs, find_pipeline_dirs
from .dashboard_thermal import build_thermal_tab
from .dashboard_quality import build_quality_tab, load_quality
from .dashboard_experiments import build_experiments_tab
from ..infra.config import cfg


def parse_args():
    p = argparse.ArgumentParser(description="Distillation dashboard")
    p.add_argument("--runs_dir", type=str, default=".", help="Parent dir of training outputs")
    p.add_argument("--port", type=int, default=cfg.services.gradio_port)
    return p.parse_args()


def main():
    args = parse_args()
    model_state = [None, None]  # [UniversalModelLoader, backend]
    run_dirs = find_run_dirs(args.runs_dir)
    pipeline_dirs = find_pipeline_dirs(args.runs_dir)

    with gr.Blocks(title="Distillation Dashboard") as app:
        gr.Markdown("# Distillation Dashboard")
        gr.Markdown("Training curves and model evaluation. Runs locally only.")
        with gr.Tabs():
            with gr.Tab("Plots"):
                gr.Markdown("### Training curves")
                with gr.Row():
                    run_dropdown = gr.Dropdown(
                        choices=pipeline_dirs,
                        value=pipeline_dirs[0] if pipeline_dirs else None,
                        label="Run directory",
                        scale=4,
                    )
                    plots_refresh_btn = gr.Button("Refresh", scale=1)
                plot_output = gr.Plot(
                    label="Loss & learning rate",
                    value=on_run_select(pipeline_dirs[0]) if pipeline_dirs else None,
                )
                run_dropdown.change(on_run_select, run_dropdown, plot_output)
                plots_refresh_btn.click(on_run_select, run_dropdown, plot_output)

            with gr.Tab("Pipeline"):
                gr.Markdown("### End-to-end pipeline summary")
                pipeline_run_dd = gr.Dropdown(
                    choices=pipeline_dirs,
                    value=pipeline_dirs[0] if pipeline_dirs else None,
                    label="Run directory",
                )
                pipeline_plot = gr.Plot(label="Pipeline summary")
                refresh_btn = gr.Button("Refresh")

                def on_pipeline_select(run_path):
                    if not run_path:
                        return None
                    from .plot_gguf_pipeline import plot_pipeline
                    return plot_pipeline(run_path)

                pipeline_run_dd.change(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                refresh_btn.click(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                if pipeline_dirs:
                    pipeline_plot.value = on_pipeline_select(pipeline_dirs[0])

            with gr.Tab("Thermal"):
                build_thermal_tab(args.runs_dir)

            with gr.Tab("Evaluate"):
                build_eval_ui(args.runs_dir, model_state)

            with gr.Tab("▶ Run Evals"):
                build_run_evals_ui(args.runs_dir, pipeline_dirs)

            with gr.Tab("Quality"):
                build_quality_tab(pipeline_dirs)

            with gr.Tab("Experiments"):
                build_experiments_tab(args.runs_dir, pipeline_dirs)

    app.queue()  # required for generator-based streaming in Run Evals tab
    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
