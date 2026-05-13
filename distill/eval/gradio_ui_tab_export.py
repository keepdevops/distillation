"""
Export tab for the Universal Gradio UI.

Provides GGUF (via llama.cpp) and CoreML (.mlpackage) export controls.
"""
from __future__ import annotations

import gradio as gr


def build_export_tab(path: str) -> dict:
    """Render the 'Export' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys: ``model_dir``,
        ``gguf_llama_dir``, ``gguf_run_btn``, ``gguf_log``,
        ``coreml_quantize``, ``coreml_seq_len``, ``coreml_output_dir``,
        ``coreml_run_btn``, ``coreml_log``.
    """
    gr.Markdown(
        "### \U0001f4e6 Export\n"
        "Convert a distilled model to GGUF (llama.cpp) or CoreML (.mlpackage)."
    )

    model_dir = gr.Textbox(
        label="Model directory",
        value=path,
        placeholder="/path/to/distilled-model",
    )

    # ── GGUF Export ───────────────────────────────────────────────────────────
    with gr.Accordion("\U0001f916 GGUF Export (llama.cpp)", open=True):
        gr.Markdown(
            "Runs `scripts/export_student_gguf.sh <model_dir> [llama_cpp_dir]`. "
            "Requires llama.cpp cloned locally. Output goes to `<model_dir>/`."
        )
        gguf_llama_dir = gr.Textbox(
            label="llama.cpp directory",
            value="/Users/Shared/llama",
            placeholder="/path/to/llama.cpp",
        )
        gguf_run_btn = gr.Button(
            "\U0001f4e4 Export to GGUF", variant="primary", min_width=180
        )
        gguf_log = gr.Textbox(
            label="GGUF export log", lines=10, interactive=False,
            elem_classes="output-box"
        )

    # ── CoreML Export ─────────────────────────────────────────────────────────
    with gr.Accordion("\U0001f34f CoreML Export (Apple Neural Engine)", open=False):
        gr.Markdown(
            "Runs `python -m distill.export.coreml --model_dir <path>`. "
            "Requires `coremltools>=8.0`. Targets CPU + GPU + ANE."
        )
        with gr.Row():
            coreml_quantize = gr.Dropdown(
                choices=["none", "int4", "int8", "float16"],
                value="int4",
                label="Quantization",
                scale=1,
            )
            coreml_seq_len = gr.Slider(
                64, 512, value=128, step=64, label="Sequence length", scale=2
            )
        coreml_output_dir = gr.Textbox(
            label="Output directory (leave blank = alongside model)",
            value="",
            placeholder="./coreml_out",
        )
        coreml_run_btn = gr.Button(
            "\U0001f4e4 Export to CoreML", variant="primary", min_width=180
        )
        coreml_log = gr.Textbox(
            label="CoreML export log", lines=10, interactive=False,
            elem_classes="output-box"
        )

    return {
        "model_dir": model_dir,
        "gguf_llama_dir": gguf_llama_dir,
        "gguf_run_btn": gguf_run_btn,
        "gguf_log": gguf_log,
        "coreml_quantize": coreml_quantize,
        "coreml_seq_len": coreml_seq_len,
        "coreml_output_dir": coreml_output_dir,
        "coreml_run_btn": coreml_run_btn,
        "coreml_log": coreml_log,
    }
