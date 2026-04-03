"""Live Logs tab widget layout."""
from __future__ import annotations

import gradio as gr
import pandas as pd


def build_tab_logs():
    """Build the Live Logs tab.

    Returns
    -------
    dict of all live-logs widgets required for event wiring.
    """
    with gr.Tab("Live Logs"):
        training_progress = gr.HTML(value="")
        with gr.Row():
            loss_plot = gr.LinePlot(
                value=pd.DataFrame({"step": [0], "loss": [0.0]}),
                x="step",
                y="loss",
                title="Training Loss",
                x_title="Step",
                y_title="Loss",
                height=220,
                min_width=300,
            )
            grad_plot = gr.LinePlot(
                value=pd.DataFrame({"step": [0], "grad_norm": [0.0]}),
                x="step",
                y="grad_norm",
                title="Gradient Norm",
                x_title="Step",
                y_title="Grad Norm",
                height=220,
                min_width=300,
            )
        log_box = gr.Textbox(
            value="",
            label="Output",
            lines=25,
            max_lines=25,
            interactive=False,
            autoscroll=True,
        )
        log_status = gr.Textbox(value="idle", label="Run status", interactive=False)
        clear_btn  = gr.Button("Clear log")

    return {
        "training_progress": training_progress,
        "loss_plot": loss_plot,
        "grad_plot": grad_plot,
        "log_box": log_box,
        "log_status": log_status,
        "clear_btn": clear_btn,
    }
