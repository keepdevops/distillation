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
        show_copy_button=True,
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
    """Render the 'Batch Eval' tab.  No widgets to return.

    Must be called inside a ``gr.Tab`` context.
    """
    gr.Markdown(f"""
### Batch Quality Evaluation

Run `eval_quality.py` on **{Path(path).name}** for comprehensive quality metrics:

```bash
# Basic quality eval (diversity + quality gates)
python -m distill.eval.quality {path}

# With LLM-as-judge scoring
python -m distill.eval.quality {path} --judge --teacher Qwen/Qwen2-1.5B-Instruct

# With teacher perplexity on student outputs (per-sample, not batch-average)
python -m distill.eval.quality {path} --judge-teacher-ppl

# All metrics together
python -m distill.eval.quality {path} --judge --judge-teacher-ppl
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
