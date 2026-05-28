"""Training Live tab — real-time loss charts, gauges, and log streaming.

Combines the launcher's existing config controls with live Plotly charts
and a scrollable log view. Talks to LiveMonitor for metric series.
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def _empty_loss_fig():
    from distill.ui.components.loss_charts import loss_figure
    return loss_figure([], [], title="Training Loss — waiting for run...")


def _empty_lr_fig():
    from distill.ui.components.loss_charts import lr_figure
    return lr_figure([], [])


def _refresh_charts():
    from distill.ui.monitoring.live_monitor import get_monitor
    from distill.ui.components.loss_charts import loss_figure, lr_figure, grad_norm_figure
    from distill.ui.components.log_parser import smooth

    series = get_monitor().get_series()
    steps = series["steps"]
    loss  = series["loss"]
    lr    = series["lr"]
    grad  = series["grad"]
    sm    = series.get("smoothed", smooth(loss))

    return (
        loss_figure(steps, loss, smoothed=sm),
        lr_figure(steps, lr),
        grad_norm_figure(steps, grad),
    )


def _refresh_log():
    from distill.ui.state_manager import get_job
    job = get_job()
    lines = job.log_tail[-80:]
    return "\n".join(lines) if lines else "Waiting for training output..."


def _status_html():
    from distill.ui.state_manager import get_job
    from distill.ui.components.hardware_gauges import build_gauges_html
    job = get_job()

    status_color = {
        "idle":      "#94a3b8",
        "running":   "#22c55e",
        "paused":    "#f59e0b",
        "completed": "#6366f1",
        "failed":    "#ef4444",
    }.get(job.status, "#94a3b8")

    progress_bar = (
        f'<div style="background:#1e293b;border-radius:4px;height:8px;margin:6px 0">'
        f'  <div style="width:{job.progress_pct():.0f}%;background:#6366f1;'
        f'height:8px;border-radius:4px;transition:width 0.3s"></div>'
        f'</div>'
    )

    info = (
        f'<div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;'
        f'margin-bottom:0.5rem">'
        f'  <span class="pill" style="background:rgba(0,0,0,0.3);color:{status_color}">'
        f'    ● {job.status.upper()}'
        f'  </span>'
        f'  <span style="color:#94a3b8;font-size:0.8rem">Phase: {job.phase or "—"}</span>'
        f'  <span style="color:#94a3b8;font-size:0.8rem">'
        f'    Step {job.step}/{job.total_steps}</span>'
        f'  <span style="color:#94a3b8;font-size:0.8rem">Loss: '
        f'    <b style="color:#e2e8f0">{job.loss:.4f}</b>'
        f'    (best {job.best_loss:.4f})</span>'
        f'  <span style="color:#94a3b8;font-size:0.8rem">⏱ {job.elapsed_str()}</span>'
        f'</div>'
        + progress_bar
    )

    return info + build_gauges_html()


def build_tab() -> None:
    """Render the Training Live tab inside the current gr.Blocks context."""
    gr.Markdown("## ▶ Training Live")
    gr.Markdown(
        "Real-time training metrics. Start a run from the SFT or Distillation tab — "
        "charts update automatically every few seconds."
    )

    # ── Status bar + hardware gauges ──────────────────────────────────────
    status_html = gr.HTML(value=_status_html, every=3)

    # ── Loss / LR / grad charts ───────────────────────────────────────────
    with gr.Row():
        loss_plot = gr.Plot(value=_empty_loss_fig, every=4, label="Loss")
    with gr.Row():
        lr_plot   = gr.Plot(value=_empty_lr_fig,   every=4, label="LR")
        grad_plot = gr.Plot(value=_empty_lr_fig,   every=4, label="Grad Norm")

    # ── Log stream ────────────────────────────────────────────────────────
    gr.Markdown("### Training Log")
    log_box = gr.Textbox(
        value=_refresh_log,
        every=3,
        label="",
        lines=15,
        max_lines=15,
        interactive=False,
        elem_classes=["cli-mirror"],
    )

    # ── Controls ──────────────────────────────────────────────────────────
    with gr.Row():
        refresh_btn = gr.Button("↻ Force Refresh", variant="secondary", scale=0)
        clear_btn   = gr.Button("🗑 Clear Charts", variant="secondary", scale=0)

    def force_refresh():
        return _status_html(), *_refresh_charts(), _refresh_log()

    def clear_charts():
        from distill.ui.monitoring.live_monitor import get_monitor
        mon = get_monitor()
        with mon._lock:
            mon._steps.clear(); mon._loss.clear()
            mon._lr.clear(); mon._grad.clear()
        return _empty_loss_fig(), _empty_lr_fig(), _empty_lr_fig()

    refresh_btn.click(
        fn=force_refresh,
        outputs=[status_html, loss_plot, lr_plot, grad_plot, log_box],
    )
    clear_btn.click(fn=clear_charts, outputs=[loss_plot, lr_plot, grad_plot])

    # Wire periodic auto-refresh for charts
    loss_plot.change(fn=lambda: _refresh_charts()[0], outputs=loss_plot)
