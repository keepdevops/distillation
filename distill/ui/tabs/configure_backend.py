"""Configure Backend tab — combined backend picker + LoRA panel + export matrix.

The single place where users set up the full training + export configuration
before launching a run. Integrates:
  - Hardware-aware backend recommendation (from hybrid_selector)
  - LoRA/QLoRA panel with live VRAM estimation
  - Export format matrix
  - CLI mirror showing the equivalent command
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)


def build_tab() -> None:
    """Render the Configure Backend tab inside the current gr.Blocks context."""
    gr.Markdown("## 🎛 Configure Backend & LoRA")
    gr.Markdown(
        "Hardware-aware configuration for training backend, LoRA adapters, "
        "and export targets. All settings update the CLI mirror in real-time."
    )

    # ── Hardware recommendation ───────────────────────────────────────────
    _render_hw_recommendation()

    # ── Backend + model selection ─────────────────────────────────────────
    gr.Markdown("### 1. Backend & Models")
    with gr.Row():
        backend_dd = gr.Dropdown(
            choices=_backend_choices(),
            value="mlx",
            label="Training Backend",
        )
        teacher_dd = gr.Dropdown(
            choices=_teacher_choices(), label="Teacher Model",
            allow_custom_value=True, value="Qwen/Qwen2-1.5B-Instruct",
        )
        student_dd = gr.Dropdown(
            choices=_student_choices(), label="Student Model",
            allow_custom_value=True, value="Qwen/Qwen2-0.5B-Instruct",
        )

    # Backend info card
    backend_info = gr.HTML(value=lambda: _backend_info_html("mlx"))
    backend_dd.change(fn=_backend_info_html, inputs=backend_dd, outputs=backend_info)

    # ── Training hyperparams ──────────────────────────────────────────────
    gr.Markdown("### 2. Training Hyperparameters")
    with gr.Row():
        epochs     = gr.Slider(1, 20,    value=3,    step=1,    label="Epochs")
        lr         = gr.Number(value=2e-4,            label="Learning Rate")
        batch_size = gr.Slider(1, 32,    value=4,    step=1,    label="Batch Size")
        grad_accum = gr.Slider(1, 32,    value=4,    step=1,    label="Grad Accum")
        seq_len    = gr.Slider(64, 4096, value=512,  step=64,   label="Max Length")

    # ── LoRA panel ────────────────────────────────────────────────────────
    gr.Markdown("### 3. LoRA / QLoRA")
    from distill.ui.components.lora_config import LoRAPanel
    lora_panel = LoRAPanel()
    lora_widgets = lora_panel.render(backend="mlx")

    # Update VRAM estimate when backend or student changes
    backend_dd.change(
        fn=lambda b, s: lora_panel.set_context(backend=b, student=s) or
                        lora_panel._widgets and gr.update(),
        inputs=[backend_dd, student_dd], outputs=[],
    )

    # ── Export matrix ─────────────────────────────────────────────────────
    gr.Markdown("### 4. Export Formats")
    from distill.ui.components.export_matrix_ui import ExportMatrixUI
    export_matrix = ExportMatrixUI()
    fmt_checkbox, compat_html = export_matrix.render(compact=False)

    # ── Output ────────────────────────────────────────────────────────────
    output_dir = gr.Textbox(
        label="Output Directory", value="outputs/distilled",
    )

    # ── CLI mirror ────────────────────────────────────────────────────────
    gr.Markdown("### 5. Generated CLI Command")
    cli_box = gr.Code(label="", language="shell", interactive=False,
                      elem_classes=["cli-mirror"])

    apply_btn = gr.Button("✅ Apply & Save Config", variant="primary")
    status_md = gr.Markdown("")

    # ── Event wiring ──────────────────────────────────────────────────────
    _cli_inputs = [backend_dd, teacher_dd, student_dd, epochs, lr, batch_size,
                   grad_accum, seq_len, lora_widgets.rank, lora_widgets.alpha,
                   lora_widgets.use_qlora, output_dir]

    def update_cli(*args):
        (backend, teacher, student, ep, lr_val, bs, ga, sl,
         rank, alpha, qlora, out_dir) = args
        return (
            f"python -m distill.orchestration.agent \\\n"
            f"    --backend {backend} \\\n"
            f"    --teacher {teacher} \\\n"
            f"    --student {student} \\\n"
            f"    --output_dir {out_dir} \\\n"
            f"    --epochs {int(ep)} --lr {lr_val} \\\n"
            f"    --batch_size {int(bs)} --grad_accum {int(ga)} \\\n"
            f"    --max_length {int(sl)} \\\n"
            f"    --lora_r {int(rank)} --lora_alpha {int(alpha)}"
            + (" --qlora" if qlora else "")
        )

    for w in _cli_inputs:
        w.change(fn=update_cli, inputs=_cli_inputs, outputs=cli_box)

    def on_apply(*args):
        try:
            from distill.ui.state_manager import update_config
            from distill.ui.core.event_bus import bus, Topic
            (backend, teacher, student, ep, lr_val, bs, ga, sl,
             rank, alpha, qlora, out_dir) = args
            update_config(
                backend=backend, teacher=teacher, student=student,
                epochs=int(ep), lr=float(lr_val), batch_size=int(bs),
                output_dir=out_dir,
            )
            bus.emit(Topic.CONFIG_LOADED, {
                "backend": backend, "teacher": teacher, "lora_rank": int(rank)
            }, source="configure_backend")
            return "✅ Config saved to active session"
        except Exception as exc:
            return f"❌ {exc}"

    apply_btn.click(fn=on_apply, inputs=_cli_inputs, outputs=status_md)


def _render_hw_recommendation() -> None:
    try:
        from distill.training.backends.hybrid_selector import select_backend
        from distill.ui.components.hardware_gauges import build_gauges_html
        rec = select_backend()
        backend = rec["backend"]
        rationale = rec["rationale"]
        warnings = rec.get("warnings", [])
        warn_html = "".join(
            f'<div class="banner-warning">⚠ {w}</div>' for w in warnings
        )
        html = (
            f'<div style="margin-bottom:.75rem">'
            f'  <span class="pill pill-blue">⚡ Recommended: {backend.upper()}</span>'
            f'  <span style="color:#94a3b8;font-size:.8rem;margin-left:.5rem">'
            f'    {rationale}</span>'
            f'</div>{warn_html}'
        )
        gr.HTML(value=html)
        gr.HTML(value=build_gauges_html)
    except Exception as exc:
        logger.debug("hw recommendation failed: %s", exc)


def _backend_info_html(backend: str) -> str:
    from distill.ui.core.registry import registry
    desc = registry.backend_descriptor(backend)
    if not desc:
        return ""
    lora = "✅ LoRA" if desc.lora_support else "❌ LoRA"
    qlora = "✅ QLoRA" if desc.qlora_support else "–"
    platforms = " · ".join(desc.platforms) if desc.platforms else "all"
    return (
        f'<div style="font-size:.8rem;color:#94a3b8;padding:.3rem 0">'
        f'  {desc.description} &nbsp;|&nbsp; '
        f'  {lora} &nbsp; {qlora} &nbsp;|&nbsp; Platforms: {platforms}'
        f'</div>'
    )


def _backend_choices() -> list[str]:
    from distill.ui.core.registry import registry
    return registry.backend_choices()


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
