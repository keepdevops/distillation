"""
Dashboard quality tab: quality metrics from eval_quality.py reports.
"""
import json
import logging
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)


def load_quality(run_path):
    """Load quality_metrics.json from a run directory and return (summary, rows)."""
    if not run_path:
        return "No run selected.", []
    qfile = Path(run_path) / "quality_metrics.json"
    if not qfile.exists():
        return f"quality_metrics.json not found in {Path(run_path).name}", []
    try:
        with open(qfile) as f:
            data = json.load(f)
    except Exception as exc:
        logger.error("load_quality failed reading %s: %s", qfile, exc)
        return f"Error reading file: {exc}", []

    div = data.get("diversity", {})
    judge = data.get("judge", {})
    gates = data.get("quality_gates", {})
    tppl = data.get("teacher_perplexity", {})
    n_gen = data.get("n_samples_generated", data.get("n_samples", "?"))
    n_pass = data.get("n_samples_passed", "?")
    lines = [
        f"Model:       {Path(data.get('model_dir', run_path)).name}",
        f"Timestamp:   {data.get('timestamp', 'n/a')}",
        f"Generated:   {n_gen}  |  Passed: {n_pass}"
        + (f"  ({gates.get('pass_rate_pct', '?')}%)" if gates.get("pass_rate_pct") else ""),
        "",
        "── Quality Gates ───────────────────────",
        f"  pass rate:      {gates.get('pass_rate_pct', 'n/a')}%",
        f"  refusal rate:   {gates.get('refusal_rate_pct', 'n/a')}%",
        "",
        "── Diversity ───────────────────────────",
        f"  avg distinct-1:   {div.get('avg_distinct_1', 'n/a')}",
        f"  avg distinct-2:   {div.get('avg_distinct_2', 'n/a')}",
        f"  avg max-rep:      {div.get('avg_max_rep', 'n/a')}",
        f"  3-gram entropy:   {div.get('ngram_entropy_3', 'n/a')} bits",
        f"  median length:    {div.get('median_response_tokens', 'n/a')} tokens",
    ]
    if tppl.get("enabled"):
        lines += [
            "",
            "── Teacher PPL on student outputs ──────",
            f"  avg teacher ppl:  {tppl.get('avg_teacher_ppl', 'n/a')}",
        ]
    if judge.get("enabled"):
        lines += [
            "",
            "── LLM-as-judge ────────────────────────",
            f"  teacher:    {judge.get('teacher', 'n/a')}",
            f"  avg score:  {judge.get('avg_score', 'n/a')} / 10",
            f"  scored:     {judge.get('n_scored', 0)} / {n_gen}",
        ]
    else:
        lines.append("\n(Judge not run — rerun with --judge to enable)")

    rows = []
    for s in data.get("samples", []):
        rows.append([
            (s.get("instruction") or s.get("prompt", ""))[:120],
            s.get("response", "")[:200],
            s.get("distinct_1", ""),
            s.get("distinct_2", ""),
            s.get("max_rep", ""),
            s.get("judge_score", ""),
        ])
    return "\n".join(lines), rows


def build_quality_tab(pipeline_dirs):
    """
    Build and wire the Quality tab. Must be called inside a gr.Tab context.
    Returns (quality_run_dd, quality_refresh_btn, quality_summary, quality_samples).
    """
    gr.Markdown("### Quality metrics (from eval_quality.py)")
    gr.Markdown(
        "Run `python -m distill.eval_quality <model_dir> --judge` to generate this report."
    )
    with gr.Row():
        quality_run_dd = gr.Dropdown(
            choices=pipeline_dirs,
            value=pipeline_dirs[0] if pipeline_dirs else None,
            label="Run directory",
            scale=4,
        )
        quality_refresh_btn = gr.Button("Load / Refresh", scale=1)
    quality_summary = gr.Textbox(label="Summary", interactive=False, lines=8)
    quality_samples = gr.Dataframe(
        headers=["prompt", "response", "distinct_1", "distinct_2", "max_rep", "judge_score"],
        label="Samples",
        wrap=True,
    )

    quality_run_dd.change(load_quality, quality_run_dd, [quality_summary, quality_samples])
    quality_refresh_btn.click(load_quality, quality_run_dd, [quality_summary, quality_samples])

    if pipeline_dirs:
        _qs, _qr = load_quality(pipeline_dirs[0])
        quality_summary.value = _qs

    return quality_run_dd, quality_refresh_btn, quality_summary, quality_samples
