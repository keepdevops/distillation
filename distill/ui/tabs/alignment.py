"""Alignment tab — DPO / ORPO / SimPO post-distillation alignment.

Full implementation wiring the DPO and ORPO backends, flywheel dataset,
data safety preview, and checkpoint resume detection.
"""
from __future__ import annotations

import logging

import gradio as gr

logger = logging.getLogger(__name__)

_METHOD_HELP = {
    "DPO":   "Direct Preference Optimization — trains with a reference model. "
             "Best quality, highest memory (2× model in memory).",
    "ORPO":  "Odds Ratio Preference Optimization — no reference model needed. "
             "Faster, lower memory than DPO, comparable quality.",
    "SimPO": "Simple Preference Optimization — length-normalised reward, "
             "no reference model. Fastest; good for short-context tasks.",
}


def build_tab() -> None:
    """Render the full Alignment tab inside the current gr.Blocks context."""
    gr.Markdown("## ⚖ Alignment")
    gr.Markdown(
        "Post-distillation preference alignment. Train on (chosen / rejected) pairs "
        "to improve instruction-following, reduce hallucination, and match tone."
    )

    # ── Method selector ───────────────────────────────────────────────────
    gr.Markdown("### 1. Alignment Method")
    with gr.Row():
        method_dd = gr.Dropdown(
            choices=["DPO", "ORPO", "SimPO"],
            value="ORPO",
            label="Method",
            scale=1,
        )
        method_info = gr.Markdown(_METHOD_HELP["ORPO"])
    method_dd.change(
        fn=lambda m: _METHOD_HELP.get(m, ""),
        inputs=method_dd, outputs=method_info,
    )

    # ── Model selection ───────────────────────────────────────────────────
    gr.Markdown("### 2. Source Model")
    with gr.Row():
        model_path = gr.Textbox(
            label="SFT Checkpoint Path",
            placeholder="outputs/distilled/sft_checkpoint",
            scale=3,
        )
        scan_btn = gr.Button("🔍 Scan", variant="secondary", scale=0)

    checkpoint_md = gr.Markdown("*Click Scan to detect available checkpoints.*")
    scan_btn.click(fn=_scan_checkpoints, inputs=model_path, outputs=checkpoint_md)

    # ── Preference dataset ────────────────────────────────────────────────
    gr.Markdown("### 3. Preference Dataset")
    with gr.Row():
        dataset_src = gr.Radio(
            choices=["Flywheel (captured errors)", "Upload JSONL", "HF Dataset ID"],
            value="Flywheel (captured errors)",
            label="Source",
        )
    flywheel_md = gr.Markdown(_flywheel_stats())
    dataset_path = gr.Textbox(
        label="Dataset path or HF ID",
        value="__flywheel__",
        visible=False,
    )

    def on_source_change(src):
        if src == "Flywheel (captured errors)":
            return "__flywheel__", gr.update(visible=False)
        elif src == "Upload JSONL":
            return "", gr.update(visible=True, label="JSONL file path")
        else:
            return "", gr.update(visible=True, label="HF Dataset ID (e.g. argilla/dpo-mix-7k)")

    dataset_src.change(fn=on_source_change, inputs=dataset_src,
                       outputs=[dataset_path, dataset_path])

    # ── Safety preview ────────────────────────────────────────────────────
    with gr.Accordion("🛡 Safety Filter Preview", open=False):
        safety_btn = gr.Button("Run Safety Check", variant="secondary")
        safety_md  = gr.Markdown("*Click to preview safety filter results.*")
        safety_btn.click(fn=lambda dp: _safety_preview(dp),
                         inputs=dataset_path, outputs=safety_md)

    # ── Hyperparameters ───────────────────────────────────────────────────
    gr.Markdown("### 4. Hyperparameters")
    with gr.Row():
        beta    = gr.Slider(0.01, 0.5, value=0.1, step=0.01, label="β (KL coeff / lambda)")
        lr      = gr.Number(value=8e-6, label="Learning Rate")
        epochs  = gr.Slider(1, 5, value=1, step=1, label="Epochs")
    with gr.Row():
        batch   = gr.Slider(1, 8, value=2, step=1, label="Batch Size")
        accum   = gr.Slider(1, 16, value=4, step=1, label="Grad Accum")
        lora_r  = gr.Slider(8, 64, value=16, step=8, label="LoRA Rank")

    output_dir = gr.Textbox(
        label="Output Directory", value="outputs/aligned",
    )

    # ── Launch controls ───────────────────────────────────────────────────
    gr.Markdown("### 5. Train")
    with gr.Row():
        start_btn  = gr.Button("▶ Start Alignment", variant="primary")
        cancel_btn = gr.Button("⏹ Cancel", variant="stop", interactive=False)

    progress_md = gr.Markdown("")
    result_json = gr.JSON(label="Results", visible=False)

    # ── CLI mirror ────────────────────────────────────────────────────────
    from distill.ui.components.cli_mirror import CliMirror
    mirror = CliMirror("Equivalent CLI")
    cli_box = mirror.render()

    # ── Events ────────────────────────────────────────────────────────────
    def do_train(method, model, dataset, out_dir, b, l, e, bs, ga, lr_val):
        cmd = (
            f"python -m distill.training.backends.{'dpo' if method == 'DPO' else 'orpo'} \\\n"
            f"    --method {method.lower()} \\\n"
            f"    --model {model} \\\n"
            f"    --dataset {dataset} \\\n"
            f"    --output-dir {out_dir} \\\n"
            f"    --epochs {int(e)} --lr {l} --lora-rank {int(lr_val)}"
        )
        try:
            from distill.training.oom_recovery import with_oom_recovery
            if method == "DPO":
                from distill.training.backends.dpo import run_dpo as _fn
            else:
                from distill.training.backends.orpo import run_orpo as _fn

            kwargs = dict(
                model_path=model, dataset_path=dataset, output_dir=out_dir,
                beta=float(b), epochs=int(e), lr=float(l),
                batch_size=int(bs), grad_accum=int(ga), lora_rank=int(lr_val),
            )
            result = with_oom_recovery(_fn, kwargs)
            status = "✅ Complete" if not result.get("error") else f"❌ {result['error']}"
            return status, result, gr.update(value=cmd), gr.update(visible=True)
        except Exception as exc:
            logger.error("Alignment failed: %s", exc)
            return f"❌ {exc}", {}, gr.update(value=cmd), gr.update(visible=False)

    start_btn.click(
        fn=do_train,
        inputs=[method_dd, model_path, dataset_path, output_dir,
                beta, lr, epochs, batch, accum, lora_r],
        outputs=[progress_md, result_json, cli_box, result_json],
    )


def _scan_checkpoints(path: str) -> str:
    if not path:
        return "*Enter an output directory path above.*"
    try:
        from distill.orchestration.checkpoint_resume import checkpoint_status_markdown
        return checkpoint_status_markdown(path)
    except Exception as exc:
        return f"*Scan error: {exc}*"


def _flywheel_stats() -> str:
    try:
        from distill.data.flywheel import flywheel_stats
        stats = flywheel_stats()
        if stats["total"] == 0:
            return (
                "*Flywheel is empty.* Run inference with the distilled model and "
                "log errors via `distill.data.flywheel.log_error()` to build preference data."
            )
        return (
            f"**Flywheel:** {stats['total']} examples "
            f"| Sources: {stats['sources']} "
            f"| Path: `{stats['path']}`"
        )
    except Exception as exc:
        return f"*Flywheel unavailable: {exc}*"


def _safety_preview(dataset_path: str) -> str:
    try:
        from distill.data.flywheel import FlywheelLog
        from distill.data.safety_filter import filter_dataset
        log = FlywheelLog()
        entries = log.load_all()
        if not entries:
            return "*No flywheel data to preview.*"
        texts = [e.prompt + " " + e.chosen for e in entries[:50]]
        _, stats = filter_dataset(texts, block_pii=True)
        return (
            f"**Safety check ({len(texts)} samples):**\n"
            f"- Safe: {stats['n_out']} / {stats['n_in']}\n"
            f"- Blocked (PII): {stats['blocked_pii']}\n"
            f"- Blocked (toxic): {stats['blocked_toxic']}\n"
            f"- Method: {stats['tox_method']}"
        )
    except Exception as exc:
        return f"*Safety preview error: {exc}*"
