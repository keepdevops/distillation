"""A/B experiment component — run two configs in parallel, compare side-by-side.

Launches two subprocesses (Config A and Config B) and renders their metrics
in parallel columns. Uses LiveMonitor instances for each job.
"""
from __future__ import annotations

import logging
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gradio as gr

from distill.ui.components.log_parser import parse_line, smooth

logger = logging.getLogger(__name__)


@dataclass
class ABRunState:
    label: str
    cmd: str = ""
    pid: int = 0
    status: str = "idle"
    steps: list[int] = None       # type: ignore[assignment]
    loss: list[float] = None      # type: ignore[assignment]
    log_tail: list[str] = None    # type: ignore[assignment]

    def __post_init__(self):
        if self.steps is None:   self.steps    = []
        if self.loss is None:    self.loss     = []
        if self.log_tail is None: self.log_tail = []


_run_a = ABRunState(label="Config A")
_run_b = ABRunState(label="Config B")
_lock  = threading.Lock()


def _watch_process(state: ABRunState, proc: subprocess.Popen) -> None:
    """Thread: stream stdout into state."""
    state.pid    = proc.pid
    state.status = "running"
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            with _lock:
                state.log_tail.append(line)
                if len(state.log_tail) > 100:
                    state.log_tail = state.log_tail[-100:]
                rec = parse_line(line)
                if rec and rec.get("step") and rec.get("loss"):
                    state.steps.append(int(rec["step"]))
                    state.loss.append(float(rec["loss"]))
        state.status = "completed"
    except Exception as exc:
        state.status = "failed"
        logger.error("ABRun[%s] watcher error: %s", state.label, exc)


def start_ab_run(cmd_a: str, cmd_b: str) -> str:
    """Launch both configs as subprocesses."""
    global _run_a, _run_b
    with _lock:
        _run_a = ABRunState(label="Config A", cmd=cmd_a)
        _run_b = ABRunState(label="Config B", cmd=cmd_b)

    for state, cmd in [(_run_a, cmd_a), (_run_b, cmd_b)]:
        if not cmd.strip():
            state.status = "skipped"
            continue
        try:
            proc = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1,
            )
            t = threading.Thread(target=_watch_process, args=(state, proc), daemon=True)
            t.start()
        except Exception as exc:
            state.status = "failed"
            logger.error("Failed to start %s: %s", state.label, exc)

    return f"Started: A (cmd={'set' if cmd_a else 'empty'}), B (cmd={'set' if cmd_b else 'empty'})"


def get_ab_status() -> tuple[str, str, Any, Any]:
    """Return (log_a, log_b, fig_a, fig_b) for display."""
    from distill.ui.components.loss_charts import loss_figure

    with _lock:
        log_a  = "\n".join(_run_a.log_tail[-30:]) or f"Status: {_run_a.status}"
        log_b  = "\n".join(_run_b.log_tail[-30:]) or f"Status: {_run_b.status}"
        sm_a   = smooth(_run_a.loss)
        sm_b   = smooth(_run_b.loss)
        steps_a, loss_a = list(_run_a.steps), list(_run_a.loss)
        steps_b, loss_b = list(_run_b.steps), list(_run_b.loss)

    fig_a = loss_figure(steps_a, loss_a, smoothed=sm_a, title="Config A — Loss")
    fig_b = loss_figure(steps_b, loss_b, smoothed=sm_b, title="Config B — Loss")
    return log_a, log_b, fig_a, fig_b


def render_ab_tab() -> None:
    """Render the A/B experiment panel inside the current gr.Blocks context."""
    gr.Markdown("## ⚗ A/B Experiment Mode")
    gr.Markdown(
        "Run two configurations in parallel and compare their training curves "
        "side-by-side. Both jobs share the same dataset."
    )

    with gr.Row():
        cmd_a = gr.Textbox(
            label="Config A — CLI command",
            placeholder="python -m distill.orchestration.agent --backend mlx --epochs 3 ...",
            lines=2,
        )
        cmd_b = gr.Textbox(
            label="Config B — CLI command",
            placeholder="python -m distill.orchestration.agent --backend sft --epochs 3 ...",
            lines=2,
        )

    with gr.Row():
        start_btn = gr.Button("▶ Start A/B Run", variant="primary")
        status_md = gr.Markdown("*Idle*")

    with gr.Row():
        loss_a = gr.Plot(label="Config A — Loss", every=4)
        loss_b = gr.Plot(label="Config B — Loss", every=4)

    with gr.Row():
        log_a = gr.Textbox(label="Config A Log", lines=8, interactive=False,
                           elem_classes=["cli-mirror"])
        log_b = gr.Textbox(label="Config B Log", lines=8, interactive=False,
                           elem_classes=["cli-mirror"])

    def on_start(ca, cb):
        msg = start_ab_run(ca, cb)
        return msg

    def on_refresh():
        la, lb, fa, fb = get_ab_status()
        return la, lb, fa, fb

    start_btn.click(fn=on_start, inputs=[cmd_a, cmd_b], outputs=status_md)
    loss_a.change(fn=on_refresh, outputs=[log_a, log_b, loss_a, loss_b])
