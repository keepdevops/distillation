"""
Tab builders for the Universal Gradio UI.

Each ``build_*_tab`` function renders its tab contents using ``gr.*``
components inside a caller-managed ``gr.Tab`` context.  Functions that
need to return widget references return them as named dicts.
"""

import base64
import logging
from pathlib import Path

import gradio as gr

from .gradio_ui_help_text import HELP_MD_PART1, HELP_MD_PART2

logger = logging.getLogger(__name__)


# ── Tab builders ──────────────────────────────────────────────────────────────

def build_model_info_tab(artifact_markdown: str, backend: str, path: str) -> dict:
    """Render the 'Model Info' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys ``backend_selector``, ``load_btn``, ``load_status``.
    """
    gr.Markdown(artifact_markdown)

    with gr.Row(equal_height=True):
        backend_selector = gr.Dropdown(
            choices=["pytorch", "mlx", "gguf", "vllm"],
            value=backend,
            label="Backend",
            info="Override auto-detected backend",
            scale=2,
        )
        load_btn = gr.Button(
            "\U0001f504 Load Model",
            variant="primary",
            elem_classes="btn-primary",
            scale=1,
            min_width=140,
        )

    load_status = gr.Textbox(
        label="Status",
        value=f"Ready to load: {Path(path).name}",
        interactive=False,
        lines=3,
    )

    return {"backend_selector": backend_selector, "load_btn": load_btn, "load_status": load_status}


def build_generate_tab() -> dict:
    """Render the 'Generate' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys ``prompt_box``, ``max_tokens_slider``,
        ``temperature_slider``, ``generate_btn``, ``output_box``.
    """
    prompt_box = gr.Textbox(
        label="Prompt",
        placeholder="Enter your prompt here\u2026",
        lines=6,
    )

    with gr.Row(elem_classes="generate-row"):
        max_tokens_slider = gr.Slider(32, 2048, value=256, step=32, label="Max tokens", scale=3)
        temperature_slider = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature", scale=3)
        generate_btn = gr.Button(
            "\U0001f680 Generate",
            variant="primary",
            elem_classes="btn-primary",
            interactive=False,
            scale=1,
            min_width=140,
        )

    output_box = gr.Textbox(
        label="Generated text",
        lines=14,
        interactive=False,
        elem_classes="output-box",
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

    return {
        "prompt_box": prompt_box,
        "max_tokens_slider": max_tokens_slider,
        "temperature_slider": temperature_slider,
        "generate_btn": generate_btn,
        "output_box": output_box,
    }


def build_batch_eval_tab(path: str) -> None:
    """Stub kept for import compatibility. Interactive eval tab is in gradio_ui_tab_eval.py."""
    pass


def build_algorithms_tab() -> None:
    """Render the 'Algorithms' tab.  No widgets to return.

    Must be called inside a ``gr.Tab`` context.
    """
    try:
        from ..ui.show_algorithms import ALGORITHMS, build_html as _build_html
        _algo_html = _build_html(ALGORITHMS)
        _b64 = base64.b64encode(_algo_html.encode("utf-8")).decode("ascii")
        gr.HTML(
            f'<div class="algo-frame">'
            f'<iframe src="data:text/html;base64,{_b64}" '
            f'style="width:100%;height:84vh;border:none;" '
            f'sandbox="allow-scripts"></iframe>'
            f'</div>'
        )
    except Exception as exc:
        logger.error("Could not load algorithms tab: %s", exc, exc_info=True)
        gr.Markdown(
            f"\u26a0\ufe0f Could not load algorithms: `{exc}`\n\n"
            "Run `python -m distill.show_algorithms` directly."
        )


def build_magpie_tab() -> dict:
    """Render the 'Magpie Synth' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys: ``teacher``, ``domain``, ``n_pairs``, ``output_dir``,
        ``backend``, ``batch_size``, ``inst_temp``, ``resp_temp``,
        ``filter_chk``, ``target_n``, ``run_btn``, ``log_box``.
    """
    gr.Markdown(
        "### \U0001f9f2 Magpie Self-Synthesis\n"
        "Auto-generate `(instruction, response)` pairs from a teacher model by "
        "conditioning on its chat-template user-turn prefix.\n\n"
        "_Reference: Xu et al., 2024_"
    )

    with gr.Row():
        teacher = gr.Textbox(
            label="Teacher model",
            value="Qwen/Qwen2-1.5B-Instruct",
            placeholder="HF model id or local path",
            scale=3,
        )
        domain = gr.Dropdown(
            choices=["general", "code", "math", "creative", "science", "reasoning"],
            value="general",
            label="Domain",
            scale=1,
        )

    with gr.Row():
        n_pairs = gr.Slider(100, 50000, value=1000, step=100, label="Pairs to generate", scale=3)
        backend = gr.Dropdown(
            choices=["auto", "mlx", "mps", "cuda", "cpu"],
            value="auto",
            label="Backend",
            scale=1,
        )

    with gr.Row():
        inst_temp = gr.Slider(0.0, 2.0, value=0.9, step=0.05, label="Instruction temp", scale=2)
        resp_temp = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Response temp", scale=2)
        batch_size = gr.Slider(4, 128, value=32, step=4, label="Batch size", scale=2)

    with gr.Row():
        output_dir = gr.Textbox(
            label="Output directory",
            value="./magpie_data",
            scale=3,
        )
        filter_chk = gr.Checkbox(label="Run deep filter after generation", value=False, scale=1)

    with gr.Row():
        target_n = gr.Number(
            label="Filter target (0 = keep all)",
            value=0,
            minimum=0,
            precision=0,
            scale=1,
        )
        run_btn = gr.Button(
            "\U0001f680 Run Magpie",
            variant="primary",
            elem_classes="btn-primary",
            scale=1,
            min_width=160,
        )

    log_box = gr.Textbox(
        label="Output log",
        lines=18,
        interactive=False,
        elem_classes="output-box",
    )

    return {
        "teacher": teacher,
        "domain": domain,
        "n_pairs": n_pairs,
        "output_dir": output_dir,
        "backend": backend,
        "batch_size": batch_size,
        "inst_temp": inst_temp,
        "resp_temp": resp_temp,
        "filter_chk": filter_chk,
        "target_n": target_n,
        "run_btn": run_btn,
        "log_box": log_box,
    }


def build_help_tab(path: str) -> None:
    """Render the 'Help' tab.  No widgets to return.

    Must be called inside a ``gr.Tab`` context.
    """
    gr.Markdown(HELP_MD_PART1)
    gr.Markdown(HELP_MD_PART2)
    gr.Markdown("---\n### Algorithm Reference")
    try:
        from ..ui.show_algorithms import ALGORITHMS, build_html as _build_html_help
        _help_html = _build_html_help(ALGORITHMS)
        _help_b64 = base64.b64encode(_help_html.encode("utf-8")).decode("ascii")
        gr.HTML(
            f'<div class="algo-frame">'
            f'<iframe src="data:text/html;base64,{_help_b64}" '
            f'style="width:100%;height:80vh;border:none;" '
            f'sandbox="allow-scripts"></iframe>'
            f'</div>'
        )
    except Exception as exc:
        logger.error("Could not load algorithms in help tab: %s", exc, exc_info=True)
        gr.Markdown(f"\u26a0\ufe0f Could not load algorithms: `{exc}`")
