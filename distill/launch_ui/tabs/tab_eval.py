"""Eval tab widget layout."""
from __future__ import annotations

import gradio as gr


def build_tab_eval(teachers, students, datasets, out_dirs,
                   default_teacher, default_student, default_dataset, default_out):
    """Build the Eval tab.

    Parameters
    ----------
    teachers:        list of teacher model choices
    students:        list of student model choices
    datasets:        list of dataset choices
    out_dirs:        list of output directory choices
    default_teacher: default teacher model value
    default_student: default student model value
    default_dataset: default dataset value
    default_out:     default output directory value

    Returns
    -------
    dict of all eval tab widgets required for event wiring.
    """
    with gr.Tab("Eval"):
        gr.Markdown(
            "Post-training evaluation. All evals write results to the model's output dir. "
            "Leave Checkpoint blank to eval the final merged model."
        )

        # Shared eval inputs
        with gr.Row():
            eval_output_dir = gr.Dropdown(
                choices=out_dirs,
                value=default_out,
                label="Model output dir",
                allow_custom_value=True,
                scale=3,
            )
            eval_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
        with gr.Row():
            eval_checkpoint = gr.Textbox(
                value="",
                label="Checkpoint (optional, e.g. distilled-minillm/checkpoint-80)",
                placeholder="Leave blank to eval final model",
                scale=3,
            )
        with gr.Row():
            eval_student = gr.Dropdown(
                choices=students,
                value=default_student,
                label="Base model (fallback tokenizer)",
                allow_custom_value=True,
                scale=3,
            )
            eval_dataset = gr.Dropdown(
                choices=datasets,
                value=default_dataset,
                label="Dataset",
                allow_custom_value=True,
                scale=3,
            )
        with gr.Row():
            eval_offline = gr.Checkbox(value=False, label="Offline")
        eval_progress = gr.HTML(value="")

        gr.Markdown("---")

        # Perplexity eval
        gr.Markdown("### Perplexity Eval  (`run_eval.py`)")
        gr.Markdown(
            "Computes cross-entropy loss and perplexity on the validation split. "
            "Appends `eval_loss` and `perplexity` to `metrics.jsonl`."
        )
        with gr.Row():
            ppl_max_val = gr.Slider(10, 1000, value=200, step=10,
                                    label="Max validation samples")
            ppl_batch   = gr.Slider(1, 32, value=8, step=1,
                                    label="Batch size")
        with gr.Row():
            ppl_compare_teacher = gr.Checkbox(value=False,
                                              label="Also eval teacher (log perplexity gap)")
            ppl_teacher = gr.Dropdown(
                choices=teachers,
                value=default_teacher,
                label="Teacher model (for comparison)",
                allow_custom_value=True,
                scale=3,
            )
        with gr.Row():
            ppl_launch_btn = gr.Button("Run Perplexity Eval", variant="primary", scale=3)
            ppl_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        ppl_status = gr.Textbox(value="idle", label="Status", interactive=False)

        gr.Markdown("---")

        # Quality eval
        gr.Markdown("### Quality Eval  (`eval_quality.py`)")
        gr.Markdown(
            "Samples prompts, generates student responses, computes distinct-1/2 diversity "
            "and max-repetition. Optionally runs LLM-as-judge scoring. "
            "Output: `quality_metrics.json`."
        )
        with gr.Row():
            qual_n_samples  = gr.Slider(10, 500, value=50, step=10,
                                        label="Samples to generate")
            qual_judge      = gr.Checkbox(value=False, label="LLM-as-judge scoring")
            qual_judge_teacher = gr.Dropdown(
                choices=teachers,
                value=default_teacher,
                label="Judge model",
                allow_custom_value=True,
                scale=3,
            )
        with gr.Row():
            qual_launch_btn = gr.Button("Run Quality Eval", variant="primary", scale=3)
            qual_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        qual_status = gr.Textbox(value="idle", label="Status", interactive=False)

        gr.Markdown("---")

        # WikiText-2 benchmark
        gr.Markdown("### WikiText-2 Benchmark  (`run_benchmarks.py`)")
        gr.Markdown(
            "Evaluates perplexity on WikiText-2-raw-v1 test split. "
            "Optionally compares against a previous baseline for regression detection. "
            "Saves `benchmark_results.json` and appends to `metrics.jsonl`."
        )
        with gr.Row():
            bench_n_seq     = gr.Slider(50, 2000, value=500, step=50,
                                        label="Sequences to evaluate")
            bench_batch     = gr.Slider(1, 32, value=8, step=1,
                                        label="Batch size")
            bench_threshold = gr.Slider(5.0, 50.0, value=15.0, step=1.0,
                                        label="Max regression % vs baseline")
        with gr.Row():
            bench_baseline = gr.Textbox(
                value="",
                label="Baseline dir (optional, for regression detection)",
                placeholder="e.g. ./previous-run",
                scale=4,
            )
        with gr.Row():
            bench_launch_btn = gr.Button("Run WikiText-2 Benchmark", variant="primary", scale=3)
            bench_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
        bench_status = gr.Textbox(value="idle", label="Status", interactive=False)

    return {
        "eval_output_dir": eval_output_dir,
        "eval_refresh_btn": eval_refresh_btn,
        "eval_checkpoint": eval_checkpoint,
        "eval_student": eval_student,
        "eval_dataset": eval_dataset,
        "eval_offline": eval_offline,
        "eval_progress": eval_progress,
        "ppl_max_val": ppl_max_val,
        "ppl_batch": ppl_batch,
        "ppl_compare_teacher": ppl_compare_teacher,
        "ppl_teacher": ppl_teacher,
        "ppl_launch_btn": ppl_launch_btn,
        "ppl_stop_btn": ppl_stop_btn,
        "ppl_status": ppl_status,
        "qual_n_samples": qual_n_samples,
        "qual_judge": qual_judge,
        "qual_judge_teacher": qual_judge_teacher,
        "qual_launch_btn": qual_launch_btn,
        "qual_stop_btn": qual_stop_btn,
        "qual_status": qual_status,
        "bench_n_seq": bench_n_seq,
        "bench_batch": bench_batch,
        "bench_threshold": bench_threshold,
        "bench_baseline": bench_baseline,
        "bench_launch_btn": bench_launch_btn,
        "bench_stop_btn": bench_stop_btn,
        "bench_status": bench_status,
    }
