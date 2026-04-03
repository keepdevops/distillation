"""Gradio run-evaluations UI for the distillation dashboard."""
from __future__ import annotations

import sys

import gradio as gr

from .dashboard_streaming import _run_streaming, _parse_progress_from_log, _progress_bar_html, _is_streaming_done

def build_run_evals_ui(runs_dir: str, pipeline_dirs: list[str]):
    """
    Gradio UI for running run_eval.py, run_benchmarks.py, and eval_quality.py
    directly from the dashboard with live streaming output.
    """
    model_choices = pipeline_dirs if pipeline_dirs else []

    gr.Markdown(
        "Select a model directory and backend, then run any eval script. "
        "Output streams live. All results are also written to the model directory."
    )

    # ── Shared controls at top ────────────────────────────────────────────────
    with gr.Row():
        shared_model_dd = gr.Dropdown(
            choices=model_choices,
            value=model_choices[0] if model_choices else None,
            label="Model directory",
            scale=4,
        )
        shared_model_custom = gr.Textbox(
            label="Or enter path manually",
            placeholder="./distilled-minillm  or  /abs/path/to/model",
            scale=3,
        )
        shared_backend = gr.Dropdown(
            choices=["auto", "gguf", "mlx", "pytorch"],
            value="auto",
            label="Backend",
            scale=1,
            info="gguf = fastest (Metal C++)",
        )

    def _model_path(dd_val, custom_val):
        return (custom_val or "").strip() or (dd_val or "")

    # ── Perplexity eval ───────────────────────────────────────────────────────
    with gr.Accordion("📊  Perplexity Eval  —  run_eval.py", open=True):
        gr.Markdown(
            "Computes cross-entropy loss and perplexity on a validation split. "
            "Appends `eval_loss` + `eval_perplexity` to `metrics.jsonl`."
        )
        with gr.Row():
            ppl_max_samples = gr.Slider(500, 10000, value=2000, step=500, label="Max dataset samples")
            ppl_max_val = gr.Slider(50, 500, value=200, step=50, label="Max val samples")
            ppl_batch = gr.Slider(4, 32, value=8, step=4, label="Batch size")
        with gr.Row():
            ppl_compare_teacher = gr.Checkbox(label="Compare teacher perplexity", value=False)
            ppl_teacher = gr.Textbox(
                label="Teacher model (HF id or path)",
                value="Qwen/Qwen2-1.5B-Instruct",
                visible=False,
            )
        ppl_compare_teacher.change(
            fn=lambda v: gr.update(visible=v),
            inputs=ppl_compare_teacher,
            outputs=ppl_teacher,
        )
        ppl_run_btn = gr.Button("▶  Run Perplexity Eval", variant="primary")
        ppl_progress = gr.HTML(value="")
        ppl_out = gr.Textbox(label="Output", lines=14, interactive=False, max_lines=200)

        def run_ppl(dd, custom, backend, max_s, max_v, batch, cmp_teacher, teacher):
            path = _model_path(dd, custom)
            if not path:
                yield "⚠ No model path specified.", ""
                return
            cmd = [
                sys.executable, "-m", "distill.eval.perplexity",
                path,
                "--backend", backend,
                "--max_samples", str(int(max_s)),
                "--max_val_samples", str(int(max_v)),
                "--batch_size", str(int(batch)),
            ]
            if cmp_teacher and teacher.strip():
                cmd += ["--compare_teacher", "--teacher", teacher.strip()]
            for text in _run_streaming(cmd):
                done = _is_streaming_done(text)
                frac, label = _parse_progress_from_log(text)
                yield text, _progress_bar_html(frac, label, running=not done)

        ppl_run_btn.click(
            fn=run_ppl,
            inputs=[shared_model_dd, shared_model_custom, shared_backend,
                    ppl_max_samples, ppl_max_val, ppl_batch,
                    ppl_compare_teacher, ppl_teacher],
            outputs=[ppl_out, ppl_progress],
        )

    # ── WikiText-2 benchmark ──────────────────────────────────────────────────
    with gr.Accordion("📈  WikiText-2 Benchmark  —  run_benchmarks.py", open=False):
        gr.Markdown(
            "Evaluates on WikiText-2-raw-v1. Detects regression vs a baseline. "
            "Saves `benchmark_results.json` and appends `wikitext2_perplexity` to `metrics.jsonl`."
        )
        with gr.Row():
            wt2_n_seq = gr.Slider(100, 2000, value=500, step=100, label="N sequences")
            wt2_max_len = gr.Slider(128, 1024, value=512, step=128, label="Max token length")
            wt2_batch = gr.Slider(4, 32, value=8, step=4, label="Batch size")
        with gr.Row():
            wt2_baseline = gr.Textbox(
                label="Baseline dir (optional — enables regression detection)",
                placeholder="./previous-run",
                scale=3,
            )
            wt2_threshold = gr.Slider(5, 30, value=15, step=5, label="Regression threshold %", scale=1)
        wt2_run_btn = gr.Button("▶  Run WikiText-2 Benchmark", variant="primary")
        wt2_progress = gr.HTML(value="")
        wt2_out = gr.Textbox(label="Output", lines=14, interactive=False, max_lines=200)

        def run_wt2(dd, custom, backend, n_seq, max_len, batch, baseline, threshold):
            path = _model_path(dd, custom)
            if not path:
                yield "⚠ No model path specified.", ""
                return
            cmd = [
                sys.executable, "-m", "distill.eval.benchmarks",
                path,
                "--backend", backend,
                "--n_sequences", str(int(n_seq)),
                "--max_length", str(int(max_len)),
                "--batch_size", str(int(batch)),
                "--threshold", str(float(threshold)),
            ]
            if (baseline or "").strip():
                cmd += ["--baseline_dir", baseline.strip()]
            for text in _run_streaming(cmd):
                done = _is_streaming_done(text)
                frac, label = _parse_progress_from_log(text)
                yield text, _progress_bar_html(frac, label, running=not done)

        wt2_run_btn.click(
            fn=run_wt2,
            inputs=[shared_model_dd, shared_model_custom, shared_backend,
                    wt2_n_seq, wt2_max_len, wt2_batch, wt2_baseline, wt2_threshold],
            outputs=[wt2_out, wt2_progress],
        )

    # ── Quality metrics ───────────────────────────────────────────────────────
    with gr.Accordion("🧪  Quality Metrics  —  eval_quality.py", open=False):
        gr.Markdown(
            "Generates responses, measures diversity (distinct-1/2, entropy), applies quality gates, "
            "and optionally runs LLM-as-judge scoring. Saves `quality_metrics.json`."
        )
        with gr.Row():
            qual_n = gr.Slider(10, 200, value=50, step=10, label="N samples")
            qual_tokens = gr.Slider(64, 1024, value=256, step=64, label="Max new tokens")
            qual_batch = gr.Slider(2, 16, value=4, step=2, label="Batch / parallel slots")
        with gr.Row():
            qual_judge = gr.Checkbox(label="LLM-as-judge scoring", value=False)
            qual_tppl = gr.Checkbox(label="Teacher perplexity on outputs", value=False)
            qual_teacher = gr.Textbox(
                label="Teacher for judge / PPL",
                value="Qwen/Qwen2-1.5B-Instruct",
                scale=3,
            )
        qual_run_btn = gr.Button("▶  Run Quality Eval", variant="primary")
        qual_progress = gr.HTML(value="")
        qual_out = gr.Textbox(label="Output", lines=18, interactive=False, max_lines=300)

        def run_qual(dd, custom, backend, n, tokens, batch, judge, tppl, teacher):
            path = _model_path(dd, custom)
            if not path:
                yield "⚠ No model path specified.", ""
                return
            cmd = [
                sys.executable, "-m", "distill.eval.quality",
                path,
                "--backend", backend,
                "--n_samples", str(int(n)),
                "--max_new_tokens", str(int(tokens)),
                "--batch_size", str(int(batch)),
            ]
            if judge:
                cmd.append("--judge")
            if tppl:
                cmd.append("--judge-teacher-ppl")
            if (judge or tppl) and teacher.strip():
                cmd += ["--teacher", teacher.strip()]
            for text in _run_streaming(cmd):
                done = _is_streaming_done(text)
                frac, label = _parse_progress_from_log(text)
                yield text, _progress_bar_html(frac, label, running=not done)

        qual_run_btn.click(
            fn=run_qual,
            inputs=[shared_model_dd, shared_model_custom, shared_backend,
                    qual_n, qual_tokens, qual_batch,
                    qual_judge, qual_tppl, qual_teacher],
            outputs=[qual_out, qual_progress],
        )


