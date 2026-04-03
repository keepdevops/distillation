"""
Dashboard experiments tab: experiment history from experiment_log.jsonl.
"""
import json
import logging
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def load_experiments(log_path):
    """Load experiment_log.jsonl and return (rows, trend_fig)."""
    p = Path(log_path)
    if not p.exists():
        return [], None
    records = []
    try:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        logger.error("load_experiments JSON decode error in %s: %s", log_path, exc)
                        continue
    except OSError as exc:
        logger.error("load_experiments OSError reading %s: %s", log_path, exc)
        return [], None

    def _s(v, fmt=".2f"):
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return ""

    rows = []
    for r in records:
        cfg = r.get("config", {})
        m = r.get("metrics", {})
        rows.append([
            r.get("run_id", "?")[:30],
            r.get("timestamp", "")[:10],
            cfg.get("backend", "?"),
            cfg.get("epochs", "?"),
            _s(m.get("eval_perplexity")),
            _s(m.get("ppl_gap_pct"), ".1f"),
            _s(m.get("judge_avg_score"), ".1f"),
            _s(m.get("wikitext2_perplexity")),
            r.get("outcome", "?"),
        ])

    fig = None
    if len(records) >= 2:
        fig = _build_trend_plot(records)

    return rows, fig


def _build_trend_plot(records):
    """Build a matplotlib trend figure from experiment records."""
    timestamps = [r.get("timestamp", "")[:10] for r in records]
    ppls = [r.get("metrics", {}).get("eval_perplexity") for r in records]
    judges = [r.get("metrics", {}).get("judge_avg_score") for r in records]
    wt2 = [r.get("metrics", {}).get("wikitext2_perplexity") for r in records]

    has_judge = any(j is not None for j in judges)
    has_wt2 = any(w is not None for w in wt2)
    n_panels = 1 + (1 if has_judge else 0) + (1 if has_wt2 else 0)

    fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), squeeze=False)
    fig.suptitle("Experiment trends", fontsize=11)
    ax_idx = 0
    x = list(range(len(records)))

    ax = axes[ax_idx][0]
    ax.plot(x, [p or float("nan") for p in ppls], "b-o", markersize=4,
            label="eval_perplexity")
    ax.set_xticks(x)
    ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Eval perplexity")
    ax.grid(True, alpha=0.3)
    ax_idx += 1

    if has_wt2:
        ax = axes[ax_idx][0]
        ax.plot(x, [w or float("nan") for w in wt2], "g-o", markersize=4,
                label="wikitext2_ppl")
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("WikiText-2 PPL")
        ax.grid(True, alpha=0.3)
        ax_idx += 1

    if has_judge:
        ax = axes[ax_idx][0]
        ax.plot(x, [j or float("nan") for j in judges], "m-o", markersize=4,
                label="judge_avg_score")
        ax.set_xticks(x)
        ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Judge score / 10")
        ax.set_ylim(0, 10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def build_experiments_tab(runs_dir, pipeline_dirs):
    """
    Build and wire the Experiments tab. Must be called inside a gr.Tab context.
    Returns (exp_log_path, exp_refresh_btn, exp_table, exp_trend_plot).
    """
    gr.Markdown("### Experiment history (from experiment_log.jsonl)")
    gr.Markdown(
        "Populated automatically by `run_distillation_agent.py --log_experiment`. "
        "Run `python -m distill.experiment_log --show 20` in a terminal for a quick view."
    )
    with gr.Row():
        exp_log_path = gr.Textbox(
            label="experiment_log.jsonl path",
            value=str(Path(runs_dir).parent / "experiment_log.jsonl"),
            scale=4,
        )
        exp_refresh_btn = gr.Button("Load / Refresh", scale=1)
    exp_table = gr.Dataframe(
        headers=["run_id", "date", "backend", "epochs",
                 "eval_ppl", "ppl_gap%", "judge", "wt2_ppl", "outcome"],
        label="Runs",
        wrap=False,
    )
    exp_trend_plot = gr.Plot(label="Metric trends over runs")

    exp_refresh_btn.click(load_experiments, exp_log_path, [exp_table, exp_trend_plot])

    _default_exp = str(Path(runs_dir).parent / "experiment_log.jsonl")
    if Path(_default_exp).exists():
        _er, _ef = load_experiments(_default_exp)
        if _er:
            exp_table.value = _er
        if _ef:
            exp_trend_plot.value = _ef

    return exp_log_path, exp_refresh_btn, exp_table, exp_trend_plot
