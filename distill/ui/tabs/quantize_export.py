"""Enhanced Quantize & Export tab — full format matrix with per-format benchmarks.

Replaces the basic export tab with a production-grade export UI covering
all 8 formats: GGUF, MLX, CoreML, Safetensors, AWQ, GPTQ, EXL2, ONNX.
"""
from __future__ import annotations

import logging

import gradio as gr

from distill.ui.tabs.swarm_utils import detect_artifacts_summary, EXPORT_FORMATS

logger = logging.getLogger(__name__)


def build_tab() -> None:
    """Render the enhanced Export tab inside the current gr.Blocks context."""
    gr.Markdown("## 📤 Quantize & Export")
    gr.Markdown(
        "Export your distilled model to any combination of deployment formats. "
        "Hardware-incompatible formats (e.g., AWQ on Apple Silicon) are flagged automatically."
    )

    # ── C++ acceleration status ───────────────────────────────────────────
    try:
        from distill.ui.components.cpp_stats import render_cpp_stats
        render_cpp_stats()
    except Exception:
        pass

    # ── Source model selection ────────────────────────────────────────────
    gr.Markdown("### 1. Source Model")
    with gr.Row():
        scan_btn = gr.Button("🔍 Scan Artifacts", variant="secondary", scale=0)
        model_dd = gr.Dropdown(choices=[], label="Model Checkpoint",
                               allow_custom_value=True, scale=3)
        merge_cb = gr.Checkbox(label="Merge LoRA before export", value=False, scale=1)

    base_model_box = gr.Textbox(
        label="Base Model ID (required if merging LoRA)",
        placeholder="Qwen/Qwen2-1.5B-Instruct",
        visible=False,
    )
    merge_cb.change(fn=lambda v: gr.update(visible=v), inputs=merge_cb, outputs=base_model_box)

    # ── Format selection matrix ───────────────────────────────────────────
    gr.Markdown("### 2. Export Formats")
    format_cb = gr.CheckboxGroup(
        choices=EXPORT_FORMATS,
        value=["GGUF (llama.cpp)", "MLX", "Safetensors + HF Hub"],
        label="Target Formats",
    )
    _render_format_compatibility_table()

    # ── Quantization settings ─────────────────────────────────────────────
    gr.Markdown("### 3. Quantization Settings")
    with gr.Row():
        quant_dd = gr.Dropdown(
            choices=["q4_k_m", "q5_k_m", "q8_0", "awq-4bit", "gptq-4bit", "exl2-4.0"],
            value="q4_k_m", label="Method",
        )
        output_dir = gr.Textbox(label="Output Directory",
                                value="/Users/Shared/llama/models/exports")

    # ── HF Hub options ────────────────────────────────────────────────────
    with gr.Accordion("HF Hub Push (Safetensors only)", open=False):
        hub_repo = gr.Textbox(label="Repo ID", placeholder="keepdevops/my-distilled-model")
        hub_token = gr.Textbox(label="HF Token (or set HF_TOKEN env var)",
                               type="password", placeholder="hf_...")

    # ── Export controls ───────────────────────────────────────────────────
    gr.Markdown("### 4. Export")
    with gr.Row():
        export_btn    = gr.Button("⚡ Export Selected Formats", variant="primary")
        prod_pack_btn = gr.Button("📦 Production Pack (ZIP all)", variant="secondary")

    progress_bar = gr.HTML("")
    status_md    = gr.Markdown("")
    results_json = gr.JSON(label="Export Results", visible=False)
    download     = gr.File(label="Download Production Pack", visible=False)

    # ── CLI mirror ────────────────────────────────────────────────────────
    from distill.ui.components.cli_mirror import CliMirror
    mirror = CliMirror("Equivalent Export CLI")
    cli_box = mirror.render()

    # ── Events ────────────────────────────────────────────────────────────
    def do_scan():
        _, paths = detect_artifacts_summary()
        return gr.update(choices=paths, value=paths[0] if paths else "")

    def do_export(model, formats, quant, out_dir, merge, base, repo, token):
        if not model:
            return "❌ No model selected.", {}, gr.update(visible=False), gr.update()
        try:
            from distill.export.export_matrix import run_export_matrix
            results = run_export_matrix(
                model_path=model, formats=formats, output_dir=out_dir,
                quant_method=quant, merge_lora=merge,
                base_model_id=base or None,
                hub_repo_id=repo, hub_token=token or None,
            )
            n_ok  = sum(1 for r in results.values() if not r.get("error"))
            n_err = len(results) - n_ok
            summary = f"✅ {n_ok} succeeded · ❌ {n_err} failed"
            cmd = (
                f"python -m distill.export.export_matrix \\\n"
                f"    --model {model} \\\n"
                f"    --formats {','.join(formats)} \\\n"
                f"    --quant {quant} \\\n"
                f"    --output-dir {out_dir}"
            )
            return summary, results, gr.update(visible=True), gr.update(value=cmd)
        except Exception as exc:
            logger.error("export failed: %s", exc)
            return f"❌ {exc}", {}, gr.update(visible=False), gr.update()

    def do_pack(out_dir, results):
        if not results:
            return gr.update(visible=False)
        try:
            from distill.export.export_matrix import zip_export_results
            zip_path = zip_export_results(results, out_dir)
            if zip_path:
                return gr.update(value=zip_path, visible=True)
        except Exception as exc:
            logger.error("pack failed: %s", exc)
        return gr.update(visible=False)

    scan_btn.click(fn=do_scan, outputs=model_dd)
    export_btn.click(
        fn=do_export,
        inputs=[model_dd, format_cb, quant_dd, output_dir,
                merge_cb, base_model_box, hub_repo, hub_token],
        outputs=[status_md, results_json, results_json, cli_box],
    )
    prod_pack_btn.click(fn=do_pack, inputs=[output_dir, results_json], outputs=download)


def _render_format_compatibility_table() -> None:
    """Show a static compatibility table for all formats."""
    gr.Markdown(
        "| Format | CPU | Apple Silicon | NVIDIA GPU | Edge |\n"
        "|---|---|---|---|---|\n"
        "| GGUF (llama.cpp) | ✅ | ✅ | ✅ | ✅ |\n"
        "| MLX | ❌ | ✅ | ❌ | ❌ |\n"
        "| CoreML | ❌ | ✅ | ❌ | iOS/macOS |\n"
        "| Safetensors + HF | ✅ | ✅ | ✅ | — |\n"
        "| AWQ | ❌ | ❌ | ✅ | — |\n"
        "| GPTQ | ❌ | ❌ | ✅ | — |\n"
        "| EXL2 | ❌ | ❌ | ✅ | — |\n"
        "| ONNX | ✅ | ✅ | ✅ | ✅ |"
    )
