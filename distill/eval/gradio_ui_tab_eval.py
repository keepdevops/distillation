"""
Interactive Batch Eval tab for the Universal Gradio UI.

Provides run buttons for quality eval, perplexity eval, and
WikiText-2 benchmark — each with a streaming log box.
"""
from __future__ import annotations

import gradio as gr


def build_batch_eval_tab(path: str) -> dict:
    """Render the interactive 'Batch Eval' tab and return widget references.

    Must be called inside a ``gr.Tab`` context.

    Returns:
        Dict with keys: ``model_dir``,
        ``qual_n_samples``, ``qual_judge``, ``qual_teacher``, ``qual_backend``,
        ``qual_run_btn``, ``qual_log``,
        ``ppl_backend``, ``ppl_run_btn``, ``ppl_log``,
        ``bench_n_seq``, ``bench_baseline``, ``bench_backend``,
        ``bench_run_btn``, ``bench_log``.
    """
    gr.Markdown(
        "### \U0001f9ea Batch Evaluation\n"
        "Run quality, perplexity, and WikiText-2 benchmark evaluations on any model dir."
    )

    model_dir = gr.Textbox(
        label="Model directory",
        value=path,
        placeholder="/path/to/distilled-model",
    )

    # ── Quality Eval ──────────────────────────────────────────────────────────
    with gr.Accordion("\U0001f4ca Quality Eval (diversity + judge)", open=True):
        gr.Markdown(
            "Samples prompts from validation split, generates responses, "
            "computes distinct-1/distinct-2/3-gram entropy and optional LLM-as-judge score."
        )
        with gr.Row():
            qual_n_samples = gr.Slider(
                10, 500, value=100, step=10, label="N samples", scale=2
            )
            qual_backend = gr.Dropdown(
                choices=["auto", "pytorch", "mlx", "gguf"],
                value="auto",
                label="Backend",
                scale=1,
            )
        with gr.Row():
            qual_judge = gr.Checkbox(label="LLM-as-judge scoring", value=False, scale=1)
            qual_teacher = gr.Textbox(
                label="Judge / teacher model",
                value="Qwen/Qwen2-1.5B-Instruct",
                scale=3,
            )
        qual_run_btn = gr.Button(
            "\U0001f680 Run Quality Eval", variant="primary", min_width=180
        )
        qual_log = gr.Textbox(
            label="Quality eval log", lines=10, interactive=False,
            elem_classes="output-box"
        )
        qual_load_btn = gr.Button("📂 Load Results (quality_metrics.json)", size="sm")
        qual_results = gr.JSON(label="Quality metrics", visible=False)
        umap_load_btn = gr.Button("🗺 Load UMAP (embedding_viz.json)", size="sm")
        umap_plot = gr.Plot(label="Embedding UMAP", visible=False)

    # ── Perplexity Eval ───────────────────────────────────────────────────────
    with gr.Accordion("\U0001f4c9 Perplexity Eval (validation loss)", open=False):
        gr.Markdown(
            "Computes cross-entropy loss and perplexity on the validation split."
        )
        with gr.Row():
            ppl_backend = gr.Dropdown(
                choices=["auto", "pytorch", "mlx", "gguf"],
                value="auto",
                label="Backend",
                scale=1,
            )
            ppl_run_btn = gr.Button(
                "\U0001f680 Run Perplexity Eval", variant="primary", min_width=180, scale=2
            )
        ppl_log = gr.Textbox(
            label="Perplexity eval log", lines=8, interactive=False,
            elem_classes="output-box"
        )
        ppl_load_btn = gr.Button("📂 Load metrics.jsonl", size="sm")
        ppl_results = gr.JSON(label="Perplexity history (metrics.jsonl)", visible=False)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    with gr.Accordion("\U0001f3c6 WikiText-2 Benchmarks", open=False):
        gr.Markdown(
            "Evaluates on WikiText-2-raw-v1 test split and optionally compares "
            "against a baseline run."
        )
        with gr.Row():
            bench_n_seq = gr.Slider(
                50, 1000, value=500, step=50, label="N sequences", scale=2
            )
            bench_backend = gr.Dropdown(
                choices=["auto", "pytorch", "mlx", "gguf"],
                value="auto",
                label="Backend",
                scale=1,
            )
        bench_baseline = gr.Textbox(
            label="Baseline dir (optional, for regression detection)",
            value="",
            placeholder="./reference-model",
        )
        bench_run_btn = gr.Button(
            "\U0001f680 Run Benchmarks", variant="primary", min_width=180
        )
        bench_log = gr.Textbox(
            label="Benchmark log", lines=8, interactive=False,
            elem_classes="output-box"
        )
        bench_load_btn = gr.Button("📂 Load benchmark_results.json", size="sm")
        bench_results = gr.JSON(label="Benchmark results", visible=False)

    return {
        "model_dir": model_dir,
        "qual_n_samples": qual_n_samples,
        "qual_judge": qual_judge,
        "qual_teacher": qual_teacher,
        "qual_backend": qual_backend,
        "qual_run_btn": qual_run_btn,
        "qual_log": qual_log,
        "qual_load_btn": qual_load_btn,
        "qual_results": qual_results,
        "umap_load_btn": umap_load_btn,
        "umap_plot": umap_plot,
        "ppl_backend": ppl_backend,
        "ppl_run_btn": ppl_run_btn,
        "ppl_log": ppl_log,
        "ppl_load_btn": ppl_load_btn,
        "ppl_results": ppl_results,
        "bench_n_seq": bench_n_seq,
        "bench_baseline": bench_baseline,
        "bench_backend": bench_backend,
        "bench_run_btn": bench_run_btn,
        "bench_log": bench_log,
        "bench_load_btn": bench_load_btn,
        "bench_results": bench_results,
    }
