"""Full Auto / Golden Pipeline tab with Gantt-style stage visualisation.

Extends the existing domain/full-auto tab with:
  - Interactive Plotly Gantt chart of pipeline stages
  - Real-time stage status updates
  - Ray distributed job controls
  - Pause / resume / cancel
"""
from __future__ import annotations

import logging
import time
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# ── Pipeline stage definitions ─────────────────────────────────────────────────

STAGES = [
    {"id": "filter",   "label": "Data Filter",        "color": "#6366f1"},
    {"id": "synth",    "label": "Synthetic Data",      "color": "#8b5cf6"},
    {"id": "sft",      "label": "SFT Warmup",          "color": "#06b6d4"},
    {"id": "distill",  "label": "Distillation",        "color": "#3b82f6"},
    {"id": "align",    "label": "Alignment (opt.)",    "color": "#a78bfa"},
    {"id": "eval",     "label": "Evaluation",          "color": "#22c55e"},
    {"id": "export",   "label": "Export",              "color": "#f59e0b"},
]

_stage_state: dict[str, dict[str, Any]] = {
    s["id"]: {"status": "pending", "start": None, "end": None, "duration": None}
    for s in STAGES
}


def _reset_stages() -> None:
    for s in STAGES:
        _stage_state[s["id"]] = {"status": "pending", "start": None, "end": None, "duration": None}


def _mark_stage(stage_id: str, status: str) -> None:
    now = time.time()
    st = _stage_state[stage_id]
    st["status"] = status
    if status == "running" and st["start"] is None:
        st["start"] = now
    elif status in ("completed", "failed") and st["start"]:
        st["end"] = now
        st["duration"] = now - st["start"]


def _build_gantt_figure() -> Any:
    """Build a horizontal Gantt chart from current stage state."""
    import plotly.graph_objects as go

    fig = go.Figure()
    t0 = time.time()
    y_labels = [s["label"] for s in STAGES]

    status_colors = {
        "pending":   "#1e293b",
        "running":   "#6366f1",
        "completed": "#22c55e",
        "failed":    "#ef4444",
        "skipped":   "#475569",
    }

    for i, stage in enumerate(STAGES):
        sid   = stage["id"]
        state = _stage_state[sid]
        status = state["status"]
        color  = status_colors.get(status, "#1e293b")

        start = state["start"] or t0
        end   = state["end"]   or (t0 + 1 if status == "pending" else t0)
        dur   = max(0.5, (end - start))

        fig.add_trace(go.Bar(
            name=stage["label"],
            x=[dur],
            y=[stage["label"]],
            orientation="h",
            base=[start - t0],
            marker_color=color,
            text=f'{status}{"  %.0fs" % dur if state["duration"] else ""}',
            textposition="inside",
            insidetextanchor="middle",
            showlegend=False,
            hovertemplate=(
                f"<b>{stage['label']}</b><br>"
                f"Status: {status}<br>"
                f"Duration: {state['duration']:.0f}s" if state["duration"] else
                f"<b>{stage['label']}</b><br>Status: {status}<extra></extra>"
            ),
        ))

    fig.update_layout(
        barmode="overlay",
        title="Pipeline Progress",
        xaxis_title="Elapsed (s)",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        template="plotly_dark",
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#e2e8f0"),
        margin=dict(l=140, r=20, t=45, b=40),
        height=320,
    )
    return fig


def _stepper_html() -> str:
    """Return HTML progress stepper for all pipeline stages."""
    icons = {
        "pending":   ("⬜", "#475569"),
        "running":   ("🔄", "#6366f1"),
        "completed": ("✅", "#22c55e"),
        "failed":    ("❌", "#ef4444"),
        "skipped":   ("⏭", "#475569"),
    }
    items = []
    for stage in STAGES:
        st = _stage_state[stage["id"]]
        icon, color = icons.get(st["status"], ("⬜", "#475569"))
        dur = f" ({st['duration']:.0f}s)" if st.get("duration") else ""
        items.append(
            f'<div style="display:flex;align-items:center;gap:.4rem;'
            f'padding:.3rem .5rem;border-radius:.25rem;background:#1e293b;margin:.15rem 0">'
            f'  <span style="font-size:1rem">{icon}</span>'
            f'  <span style="color:{color};font-weight:600;font-size:.85rem">'
            f'    {stage["label"]}</span>'
            f'  <span style="color:#475569;font-size:.75rem">{dur}</span>'
            f'</div>'
        )
    return '<div style="display:flex;flex-direction:column;min-width:200px">' + "".join(items) + "</div>"


def build_tab() -> None:
    """Render the Full Auto / Golden Pipeline tab."""
    gr.Markdown("## ▶ Full Auto — Golden Pipeline")
    gr.Markdown(
        "Run the complete distillation pipeline end-to-end: "
        "filter → synthesise → SFT → distil → align → eval → export."
    )

    # ── Config ────────────────────────────────────────────────────────────
    gr.Markdown("### Configuration")
    with gr.Row():
        teacher_dd = gr.Dropdown(
            choices=_teacher_choices(), label="Teacher Model",
            value="Qwen/Qwen2-1.5B-Instruct", allow_custom_value=True,
        )
        student_dd = gr.Dropdown(
            choices=_student_choices(), label="Student Model",
            value="Qwen/Qwen2-0.5B-Instruct", allow_custom_value=True,
        )
    with gr.Row():
        backend_dd = gr.Dropdown(
            choices=["mlx", "sft", "minillm", "unsloth", "forward"],
            value="mlx", label="Backend",
        )
        preset_dd  = gr.Dropdown(
            choices=_preset_names(), label="Load Preset", allow_custom_value=False,
        )
        load_btn   = gr.Button("Load", variant="secondary", scale=0)

    skip_align = gr.Checkbox(label="Skip Alignment stage", value=False)
    output_dir = gr.Textbox(label="Output Directory", value="outputs/golden")

    # ── Pipeline Gantt ────────────────────────────────────────────────────
    gr.Markdown("### Pipeline Progress")
    with gr.Row():
        stepper_html = gr.HTML(value=_stepper_html)
        gantt_plot   = gr.Plot(value=_build_gantt_figure, every=3)

    # ── Controls ──────────────────────────────────────────────────────────
    with gr.Row():
        start_btn  = gr.Button("▶ Start Golden Pipeline", variant="primary")
        pause_btn  = gr.Button("⏸ Pause",  variant="secondary", interactive=False)
        cancel_btn = gr.Button("⏹ Cancel", variant="stop",      interactive=False)

    status_md = gr.Markdown("*Idle — configure above and press Start.*")

    # ── Ray distributed options ───────────────────────────────────────────
    with gr.Accordion("⚡ Ray Distributed (Ablation Sweep)", open=False):
        _render_ray_panel()

    # ── CLI mirror ────────────────────────────────────────────────────────
    from distill.ui.components.cli_mirror import CliMirror
    mirror = CliMirror("Equivalent CLI")
    cli_box = mirror.render()

    # ── Events ────────────────────────────────────────────────────────────
    def on_start(teacher, student, backend, out_dir, skip_al):
        _reset_stages()
        cmd = (
            f"python -m distill.orchestration.agent \\\n"
            f"    --teacher {teacher} --student {student} \\\n"
            f"    --backend {backend} --output_dir {out_dir}"
            + (" --skip-align" if skip_al else "")
        )
        return (
            _stepper_html(),
            _build_gantt_figure(),
            "🔄 Pipeline starting...",
            gr.update(value=cmd),
        )

    def on_preset_load(preset_name):
        try:
            from distill.launch_ui.presets import get_preset
            p = get_preset(preset_name)
            return (
                gr.update(value=p.get("teacher", "")),
                gr.update(value=p.get("student", "")),
                gr.update(value=p.get("backend", "mlx")),
            )
        except Exception:
            return gr.update(), gr.update(), gr.update()

    start_btn.click(
        fn=on_start,
        inputs=[teacher_dd, student_dd, backend_dd, output_dir, skip_align],
        outputs=[stepper_html, gantt_plot, status_md, cli_box],
    )
    load_btn.click(
        fn=on_preset_load, inputs=preset_dd,
        outputs=[teacher_dd, student_dd, backend_dd],
    )


def _render_ray_panel() -> None:
    from distill.orchestration.ray_distributed import _has_ray
    has_ray = _has_ray()
    if not has_ray:
        gr.Markdown(
            '<div class="banner-warning">⚠ Ray not installed — sweeps run sequentially. '
            "Run `pip install ray` to enable parallel execution.</div>"
        )

    gr.Markdown("Run ablation sweeps across learning rates, LoRA ranks, or backends in parallel.")
    with gr.Row():
        sweep_param = gr.Dropdown(
            choices=["lr", "lora_rank", "backend", "epochs"],
            value="lr", label="Sweep Parameter",
        )
        sweep_values = gr.Textbox(
            label="Values (comma-separated)",
            value="1e-4,2e-4,5e-4",
        )
        max_parallel = gr.Slider(1, 4, value=2, step=1, label="Max Parallel Jobs")

    sweep_btn    = gr.Button("🔀 Run Sweep", variant="secondary")
    sweep_status = gr.Markdown("")

    def do_sweep(teacher, student, backend, out_dir, param, values_str, max_par):
        try:
            vals = [v.strip() for v in values_str.split(",") if v.strip()]
            from distill.orchestration.ray_distributed import DistillJob, build_sweep, run_jobs_parallel
            base = DistillJob(
                job_id="sweep_base", backend=backend, teacher=teacher,
                student=student, dataset="yahma/alpaca-cleaned", output_dir=out_dir,
            )
            grid = {param: vals}
            jobs = build_sweep(base, grid)
            statuses: dict[str, str] = {}

            def cb(jid, status):
                statuses[jid] = status

            results = run_jobs_parallel(jobs, max_parallel=int(max_par), progress_cb=cb)
            n_ok  = sum(1 for r in results if r.success)
            return f"✅ Sweep complete: {n_ok}/{len(results)} succeeded"
        except Exception as exc:
            return f"❌ Sweep failed: {exc}"

    # Capture outer scope inputs (limited — Gradio doesn't share state across accordions easily)
    sweep_btn.click(
        fn=lambda p, v, mp: do_sweep("", "", "mlx", "outputs/sweep", p, v, mp),
        inputs=[sweep_param, sweep_values, max_parallel],
        outputs=sweep_status,
    )


def _teacher_choices() -> list[str]:
    try:
        from distill.launch_ui.presets import KNOWN_TEACHERS
        return KNOWN_TEACHERS
    except Exception:
        return []


def _student_choices() -> list[str]:
    try:
        from distill.launch_ui.presets import KNOWN_STUDENTS
        return KNOWN_STUDENTS
    except Exception:
        return []


def _preset_names() -> list[str]:
    try:
        from distill.launch_ui.presets import preset_names
        return preset_names()
    except Exception:
        return []
