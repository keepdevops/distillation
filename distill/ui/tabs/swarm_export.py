"""Swarm Export tab — generate matrix swarm configs and production export packs.

Detects trained artifacts and produces ready-to-use configs for:
  GGUF + llama.cpp · MLX + CoreML · AWQ/GPTQ/EXL2 + vLLM · Safetensors + HF · ONNX
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import gradio as gr

from distill.ui.tabs.swarm_utils import (
    detect_artifacts_summary,
    generate_swarm_config,
    EXPORT_FORMATS,
)

logger = logging.getLogger(__name__)


def build_tab() -> None:
    """Render the Swarm Export tab inside the current gr.Blocks context."""
    gr.Markdown("## 🚀 Swarm Export")
    gr.Markdown(
        "Detect trained model artifacts and generate production deployment configs "
        "for any supported inference backend or swarm runtime."
    )

    # ── Artifact detection ────────────────────────────────────────────────
    gr.Markdown("### 1. Detected Artifacts")
    artifacts_md = gr.Markdown("*Click Scan to detect trained models.*")
    scan_btn = gr.Button("🔍 Scan for Artifacts", variant="primary")

    # ── Export format matrix ──────────────────────────────────────────────
    gr.Markdown("### 2. Export Format Matrix")
    with gr.Row():
        format_cb = gr.CheckboxGroup(
            choices=EXPORT_FORMATS,
            value=["GGUF (llama.cpp)", "MLX", "Safetensors + HF Hub"],
            label="Target Formats",
        )

    # ── Swarm config options ──────────────────────────────────────────────
    gr.Markdown("### 3. Swarm Config Options")
    with gr.Row():
        model_path_dd = gr.Dropdown(
            choices=[], label="Model Artifact", allow_custom_value=True
        )
        system_prompt = gr.Textbox(
            label="System Prompt",
            placeholder="You are a helpful assistant specialized in...",
            lines=2,
        )
    with gr.Row():
        quant_method = gr.Dropdown(
            choices=["q4_k_m", "q5_k_m", "q8_0", "awq-4bit", "gptq-4bit", "exl2-4.0"],
            value="q4_k_m",
            label="Quantization Method",
        )
        output_dir = gr.Textbox(
            label="Output Directory",
            value="/Users/Shared/llama/models",
        )

    # ── Generate ──────────────────────────────────────────────────────────
    gr.Markdown("### 4. Generate & Download")
    with gr.Row():
        generate_btn = gr.Button("⚡ Generate Configs", variant="primary")
        prod_pack_btn = gr.Button("📦 Production Pack (ZIP)", variant="secondary")

    config_output = gr.Code(language="json", label="Generated Swarm Config", lines=20)
    download_file = gr.File(label="Download Config ZIP", visible=False)
    status_md = gr.Markdown("")

    # ── CLI mirror ────────────────────────────────────────────────────────
    from distill.ui.components.cli_mirror import CliMirror
    mirror = CliMirror("Equivalent Export CLI")
    mirror.render()

    # ── Event wiring ──────────────────────────────────────────────────────
    def do_scan():
        summary, paths = detect_artifacts_summary()
        return gr.update(value=summary), gr.update(choices=paths, value=paths[0] if paths else "")

    def do_generate(model_path, formats, system_p, quant, out_dir):
        try:
            cfg = generate_swarm_config(
                model_path=model_path,
                formats=formats,
                system_prompt=system_p,
                quant_method=quant,
                output_dir=out_dir,
            )
            cmd = (
                f"python -m distill.export.export_matrix \\\n"
                f"    --model {model_path} \\\n"
                f"    --formats {','.join(formats)} \\\n"
                f"    --quant {quant} \\\n"
                f"    --output-dir {out_dir}"
            )
            return (
                json.dumps(cfg, indent=2),
                gr.update(value=f"✅ Config generated for {len(formats)} format(s)."),
                gr.update(value=cmd),
            )
        except Exception as exc:
            logger.error("generate_swarm_config failed: %s", exc)
            return "{}", gr.update(value=f"❌ Error: {exc}"), gr.update()

    scan_btn.click(fn=do_scan, outputs=[artifacts_md, model_path_dd])
    generate_btn.click(
        fn=do_generate,
        inputs=[model_path_dd, format_cb, system_prompt, quant_method, output_dir],
        outputs=[config_output, status_md, mirror.box],
    )
