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

from ..backends.universal_loader import UniversalModelLoader, detect_model_format, ModelFormat
from ..infra.artifact_detector import detect_artifacts, format_artifact_summary
from .gradio_ui_css import CUSTOM_CSS
from .gradio_ui_tabs import (
    build_model_info_tab,
    build_generate_tab,
    build_batch_eval_tab,
    build_algorithms_tab,
    build_help_tab,
)
from .gradio_ui_handlers import make_load_model_fn, make_generate_fn, get_artifact_info


def parse_args():
    from ..infra.model_path_helper import resolve_model_path, get_model_base_path

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
    import gradio as gr

    args = parse_args()
    from ..infra.model_path_helper import resolve_model_path
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
        if artifacts_info["artifacts"]:
            print(f"\n{format_artifact_summary(artifacts_info['artifacts'])}")

    # Detect model format
    detected_format = detect_model_format(path)
    backend = args.backend if args.backend else detected_format.value

    print(f"\nBackend: {backend}")
    print(f"Starting Gradio UI on http://127.0.0.1:{args.port}")

    # Create universal loader and mutable state dict
    loader = UniversalModelLoader()
    model_loaded = {"loaded": False, "message": "No model loaded"}

    # Build handler closures
    load_model_fn = make_load_model_fn(loader, model_loaded, path, backend)
    generate_fn = make_generate_fn(loader, model_loaded)

    # Pre-compute artifact summary for the model info tab
    artifact_markdown = get_artifact_info(artifacts_info, path)

    # Build Gradio interface
    with gr.Blocks(
        title="Universal Model Evaluator",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
    ) as iface:

        with gr.Column(elem_classes="app-header"):
            gr.HTML(
                "<h1>Universal Model Evaluator</h1>"
                "<p>Supports <b>PyTorch</b>, <b>MLX</b>, <b>GGUF</b> (llama.cpp), and <b>vLLM</b> backends. "
                "Auto-detects model format &nbsp;\u00b7&nbsp; See <b>Help</b> tab for reference.</p>"
            )

        with gr.Tabs():
            with gr.Tab("\U0001f4cb Model Info"):
                info_widgets = build_model_info_tab(artifact_markdown, backend, path)

            with gr.Tab("\u2728 Generate"):
                gen_widgets = build_generate_tab()

            with gr.Tab("\U0001f9ea Batch Eval"):
                build_batch_eval_tab(path)

            with gr.Tab("\U0001f4d0 Algorithms"):
                build_algorithms_tab()

            with gr.Tab("\U0001f4d6 Help"):
                build_help_tab(path)

        # Wire events
        info_widgets["load_btn"].click(
            fn=load_model_fn,
            inputs=[info_widgets["backend_selector"]],
            outputs=[info_widgets["load_status"], gen_widgets["generate_btn"]],
        )

        gen_widgets["generate_btn"].click(
            fn=generate_fn,
            inputs=[
                gen_widgets["prompt_box"],
                gen_widgets["max_tokens_slider"],
                gen_widgets["temperature_slider"],
            ],
            outputs=[gen_widgets["output_box"]],
        )

        # Auto-load on startup if backend is specified
        if args.backend or detected_format != ModelFormat.UNKNOWN:
            iface.load(
                fn=lambda: load_model_fn(backend),
                outputs=[info_widgets["load_status"], gen_widgets["generate_btn"]],
            )

    iface.launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
    )


if __name__ == "__main__":
    main()
