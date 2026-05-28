"""HTML/PDF run report generator with embedded Plotly charts.

Produces a self-contained, print-ready HTML report for a completed
distillation run including: config table, metrics summary, training
curves, eval scores, and export artefacts.
"""
from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CSS = """
<style>
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;
     max-width:960px;margin:2rem auto;padding:0 1rem;line-height:1.6}
h1{color:#a5b4fc;border-bottom:1px solid #1e293b;padding-bottom:.5rem}
h2{color:#c7d2fe;margin-top:2rem;font-size:1.1rem}
table{width:100%;border-collapse:collapse;margin:.75rem 0;font-size:.85rem}
th{background:#1e293b;color:#94a3b8;padding:.4rem .6rem;text-align:left;
   font-size:.75rem;text-transform:uppercase;letter-spacing:.05em}
td{padding:.35rem .6rem;border-bottom:1px solid #1e293b}
tr:hover td{background:#1e293b}
.badge{display:inline-block;padding:.15rem .5rem;border-radius:999px;
       font-size:.75rem;font-weight:600}
.ok{background:rgba(34,197,94,.15);color:#22c55e}
.warn{background:rgba(245,158,11,.15);color:#f59e0b}
.fail{background:rgba(239,68,68,.15);color:#ef4444}
.metric-row{display:flex;flex-wrap:wrap;gap:.75rem;margin:.75rem 0}
.metric{background:#1e293b;border-radius:.5rem;padding:.6rem 1rem;min-width:140px;text-align:center}
.mv{font-size:1.5rem;font-weight:800;color:#6366f1}
.ml{font-size:.7rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em}
pre{background:#1e1e2e;padding:1rem;border-radius:.5rem;font-size:.78rem;
    overflow-x:auto;color:#a5f3fc;white-space:pre-wrap}
.chart-container{margin:1rem 0;border:1px solid #1e293b;border-radius:.5rem;overflow:hidden}
</style>
"""


def _metric_card(label: str, value: Any, unit: str = "") -> str:
    return (
        f'<div class="metric">'
        f'<div class="mv">{value}{unit}</div>'
        f'<div class="ml">{label}</div>'
        f'</div>'
    )


def _table(rows: list[tuple[str, Any]], title: str = "") -> str:
    header = f"<h2>{title}</h2>" if title else ""
    trs = "".join(f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows)
    return f"{header}<table><tr><th>Key</th><th>Value</th></tr>{trs}</table>"


def _embed_loss_chart(log_history: list[dict]) -> str:
    """Return a self-contained Plotly HTML div for training loss."""
    try:
        import plotly.graph_objects as go
        import plotly.io as pio

        steps = [e.get("step", i) for i, e in enumerate(log_history) if "loss" in e]
        loss  = [e["loss"] for e in log_history if "loss" in e]
        if not steps:
            return ""

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=loss, name="Train Loss",
                                  line=dict(color="#6366f1", width=2)))
        fig.update_layout(
            title="Training Loss", xaxis_title="Step", yaxis_title="Loss",
            template="plotly_dark", paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            font=dict(color="#e2e8f0"), height=260,
            margin=dict(l=50, r=20, t=45, b=40),
        )
        div = pio.to_html(fig, include_plotlyjs="cdn", full_html=False)
        return f'<div class="chart-container">{div}</div>'
    except Exception as exc:
        logger.warning("chart embed failed: %s", exc)
        return ""


def _load_trainer_state(output_dir: str) -> list[dict]:
    """Read log_history from trainer_state.json if present."""
    for subdir in ["", "sft_checkpoint", "distilled-minillm"]:
        p = Path(output_dir) / subdir / "trainer_state.json"
        if p.exists():
            try:
                return json.loads(p.read_text()).get("log_history", [])
            except Exception:
                pass
    return []


def generate_report(
    run: dict[str, Any],
    output_dir: str | None = None,
) -> str:
    """Build a full HTML report for one run dict.

    Args:
        run:        Run record (from ExperimentLog or arbitrary dict).
        output_dir: If provided, load training charts from trainer_state.json.

    Returns:
        Path to the written HTML file.
    """
    cfg     = run.get("config", {})
    metrics = run.get("metrics", {})
    run_id  = run.get("run_id", f"run_{id(run)}")
    ts      = run.get("timestamp", "")

    # Key metric cards
    metric_cards = ""
    for key, label, fmt in [
        ("perplexity",    "Perplexity",  ".2f"),
        ("quality_score", "Quality",     ".3f"),
        ("tokens_per_sec","Speed (tps)", ".0f"),
        ("param_count",   "Params",      None),
    ]:
        v = metrics.get(key)
        if v is not None:
            disp = f"{v:{fmt}}" if fmt else f"{int(v/1e6)}M" if v > 1e5 else str(v)
            metric_cards += _metric_card(label, disp)

    # Training chart
    log_history = _load_trainer_state(output_dir or cfg.get("output_dir", ""))
    chart_html  = _embed_loss_chart(log_history)

    # Config table
    config_table = _table(sorted(cfg.items()), "Configuration")

    # Metrics table
    metrics_table = _table(
        [(k, f"{v:.4f}" if isinstance(v, float) else str(v)) for k, v in sorted(metrics.items())],
        "Metrics",
    )

    # Log tail
    log_tail = run.get("log_tail", [])
    log_section = ""
    if log_tail:
        lines = "\n".join(str(l) for l in log_tail[-40:])
        log_section = f"<h2>Log Tail</h2><pre>{lines}</pre>"

    # Regression summary
    regression_html = _regression_section(metrics)

    body = (
        f'<h1>🌭 Distillation Report — {run_id}</h1>'
        f'<p style="color:#94a3b8">{ts}</p>'
        f'<div class="metric-row">{metric_cards}</div>'
        + regression_html
        + chart_html
        + config_table
        + metrics_table
        + log_section
    )

    html = f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{_CSS}</head><body>{body}</body></html>"

    tmp = tempfile.NamedTemporaryFile(
        suffix=f"_{run_id}.html", prefix="distill_report_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    logger.info("Report: %s", tmp.name)
    return tmp.name


def _regression_section(metrics: dict[str, Any]) -> str:
    try:
        from distill.ui.components.regression_tracker import load_best_metrics, check_regression, format_regression_markdown
        best = load_best_metrics()
        if not best or not metrics:
            return ""
        current = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        prior   = {k: float(v) for k, v in best.items()    if isinstance(v, (int, float))}
        results = check_regression(current, prior)
        if not results:
            return ""
        md = format_regression_markdown(results)
        return f"<h2>Regression vs Previous Best</h2><pre>{md}</pre>"
    except Exception:
        return ""


def generate_all_runs_report(experiment_log_path: str | None = None) -> str:
    """Generate a multi-run HTML report from experiment_log.jsonl."""
    try:
        from distill.ui.experiment_log import ExperimentLog
        runs = ExperimentLog(experiment_log_path or "experiment_log.jsonl").load_all()
    except Exception as exc:
        logger.error("load runs: %s", exc)
        return ""

    if not runs:
        return ""

    sections = "\n<hr style='border-color:#1e293b;margin:2rem 0'>\n".join(
        _run_section(r) for r in runs[-30:]
    )
    html = (
        f"<!DOCTYPE html><html><head><meta charset='UTF-8'>{_CSS}</head>"
        f"<body><h1>🌭 All Runs Report ({len(runs)} total)</h1>{sections}</body></html>"
    )
    tmp = tempfile.NamedTemporaryFile(
        suffix="_all_runs.html", prefix="distill_report_",
        delete=False, mode="w", encoding="utf-8",
    )
    tmp.write(html)
    tmp.close()
    return tmp.name


def _run_section(run: dict) -> str:
    cfg = run.get("config", {})
    m   = run.get("metrics", {})
    rid = run.get("run_id", "?")
    ts  = run.get("timestamp", "")[:10]
    ppl = m.get("perplexity", "–")
    q   = m.get("quality_score", "–")
    be  = cfg.get("backend", "–")
    return (
        f'<h2>Run: {rid} <small style="color:#475569">{ts} · {be}</small></h2>'
        f'<p>Perplexity: <b>{ppl}</b> &nbsp; Quality: <b>{q}</b></p>'
    )
