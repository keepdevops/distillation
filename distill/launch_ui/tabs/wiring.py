"""Event wiring — connects all tab widgets to their callback functions."""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import gradio as gr

from ..discovery import (
    discover_datasets,
    discover_output_dirs,
    discover_students,
    discover_teachers,
)
from ..runner import (
    _build_cmd,
    _ep_start_proc,
    clear_logs,
    launch_eval_benchmark,
    launch_eval_perplexity,
    launch_eval_quality,
    launch_filter,
    launch_magpie,
    launch_run,
    launch_synth,
    poll_logs,
    save_custom_domain,
    stop_run,
)
from ...infra.paths import project_dir

logger = logging.getLogger(__name__)

PROJECT_DIR = project_dir()
PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Stage / backend toggle callbacks
# ---------------------------------------------------------------------------

def on_stage_change(s):
    is_sft = s == "SFT"
    return (
        gr.update(visible=is_sft),
        gr.update(visible=not is_sft),
        gr.update(visible=False),  # mlx_group always hidden when stage flips
    )


def on_backend_change(b):
    is_mlx = b == "MLX"
    if is_mlx:
        return (
            gr.update(visible=False),  # minillm_group
            gr.update(visible=True),   # mlx_group
            gr.update(value=2),        # batch_size
            gr.update(value=8),        # grad_acc
            gr.update(value=16),       # lora_r
        )
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=8),
        gr.update(value=8),
        gr.update(value=16),
    )


# ---------------------------------------------------------------------------
# Expert Pipeline inline callbacks
# ---------------------------------------------------------------------------

def _ep_inspect(dataset_id: str):
    """Run expert_pipeline.py --mode inspect and parse JSON result."""
    import subprocess as _sp
    ds = dataset_id.strip()
    if not ds or ds == "Enter a dataset ID first.":
        return ("Enter a dataset ID first.", [], [], ["(none)"])
    try:
        result = _sp.run(
            [PYTHON, "-m", "distill.data.expert",
             "--mode", "inspect", "--dataset", ds],
            capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=120,
        )
    except Exception as exc:
        logger.error("_ep_inspect subprocess error: %s", exc)
        return (f"Inspect failed: {exc}", [], [], ["(none)"])
    raw = result.stdout.strip()
    if not raw:
        err = result.stderr[-500:] if result.stderr else "No output"
        return (f"Inspect failed:\n{err}", [], [], ["(none)"])
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("_ep_inspect JSON decode error: %s — raw: %s", exc, raw[:200])
        return (f"Could not parse output:\n{raw[:300]}", [], [], ["(none)"])
    cols = data.get("columns", [])
    guessed = data.get("guessed", {})
    info = (f"{data.get('n_rows', '?')} rows  |  columns: {', '.join(cols)}\n"
            f"Auto-detected → instruction: {guessed.get('instruction')}  "
            f"output: {guessed.get('output')}  input: {guessed.get('input')}")
    col_choices   = cols
    inst_default  = guessed.get("instruction") or (cols[0] if cols else "")
    out_default   = guessed.get("output") or (cols[1] if len(cols) > 1 else "")
    input_choices = ["(none)"] + cols
    input_default = guessed.get("input") or "(none)"
    return (
        info,
        gr.update(choices=col_choices, value=inst_default),
        gr.update(choices=col_choices, value=out_default),
        gr.update(choices=input_choices, value=input_default),
    )


def _ep_remap(dataset, instruction_col, output_col, input_col,
              max_samples, output_dir):
    params = {
        "mode": "remap",
        "dataset": dataset,
        "instruction_col": instruction_col,
        "output_col": output_col,
        "output_dir": output_dir,
    }
    if input_col and input_col != "(none)":
        params["input_col"] = input_col
    if max_samples:
        params["max_samples"] = int(max_samples)
    cmd = _build_cmd("expert_pipeline.py", params)
    return _ep_start_proc(cmd, label="ep_remap")


def _ep_load_system_prompt(domain: str):
    try:
        from ...data.expert import DEFAULT_SYSTEM_PROMPT, DOMAIN_SYSTEM_PROMPTS
        prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)
    except Exception as exc:
        logger.error("_ep_load_system_prompt import error: %s", exc)
        prompt = ""
    return gr.update(placeholder=prompt, value="")


def _ep_teacher_refresh():
    try:
        choices = sorted(str(p) for p in Path("/Users/Shared/llama/models").glob("*.gguf")) \
            if Path("/Users/Shared/llama/models").exists() else []
    except Exception as exc:
        logger.error("_ep_teacher_refresh error: %s", exc)
        choices = []
    return gr.update(choices=choices)


def _ep_cot(dataset, teacher, domain, system_prompt, n_samples, temperature,
            max_tokens, ctx_size, n_parallel, batch_size, output_dir):
    params = {
        "mode": "cot",
        "dataset": dataset,
        "teacher": teacher,
        "domain": domain,
        "n_samples": int(n_samples),
        "temperature": temperature,
        "max_tokens": int(max_tokens),
        "ctx_size": int(ctx_size),
        "n_parallel": int(n_parallel),
        "batch_size": int(batch_size),
        "output_dir": output_dir,
    }
    if system_prompt.strip():
        params["system_prompt"] = system_prompt.strip()
    cmd = _build_cmd("expert_pipeline.py", params)
    return _ep_start_proc(cmd, label=f"ep_cot_{domain}")


def _ep_distill(cot_dir, distill_dataset, output_dir, backend,
                epochs, lora_r, max_samples, open_flag, offline_flag):
    dataset = distill_dataset.strip() or f"{cot_dir}/hf_dataset"
    params = {
        "mode": "distill",
        "dataset": dataset,
        "output_dir": output_dir,
        "backend": backend,
        "epochs": int(epochs),
        "lora_r": int(lora_r),
        "max_samples": int(max_samples),
        "open": open_flag,
        "offline": offline_flag,
    }
    cmd = _build_cmd("expert_pipeline.py", params)
    return _ep_start_proc(cmd, label="ep_distill")


def _save_and_launch(domain_id, label, desc, prompts_text,
                     min_words, max_words, min_d2, req_code, req_nums,
                     teacher, n, batch, use_filter, target, outdir,
                     offline, backend="auto"):
    """Save custom domain config then launch Magpie synthesis for it."""
    did = (domain_id or "").strip().lower().replace(" ", "_") or "custom"
    try:
        save_msg = save_custom_domain(domain_id, label, desc, prompts_text,
                                     min_words, max_words, min_d2, req_code, req_nums)
    except Exception as exc:
        logger.error("_save_and_launch save_custom_domain error: %s", exc)
        return f"Error saving domain: {exc}", "idle"
    if save_msg.startswith("Error"):
        return save_msg, "idle"
    log_msg, status = launch_magpie(teacher, outdir, n, batch, use_filter,
                                    target, offline, domain=did, backend=backend)
    return log_msg, status


# ---------------------------------------------------------------------------
# Main wiring entry point
# ---------------------------------------------------------------------------

def wire_events(demo, widgets):
    """Attach all .click() / .change() / gr.Timer callbacks to widgets.

    Parameters
    ----------
    demo:    the gr.Blocks instance (used for gr.Timer)
    widgets: flat dict merging all dicts returned by build_tab_* functions
    """
    w = widgets  # shorthand

    # ── Stage / backend toggles ──────────────────────────────────────────
    w["stage"].change(
        on_stage_change, w["stage"],
        [w["sft_group"], w["minillm_group"], w["mlx_group"]],
    )
    w["backend"].change(
        on_backend_change, w["backend"],
        [w["minillm_group"], w["mlx_group"],
         w["batch_size"], w["grad_acc"], w["lora_r"]],
    )

    # ── Configure & Launch refresh dropdowns ────────────────────────────
    w["refresh_teacher_btn"].click(
        fn=lambda: gr.update(choices=discover_teachers()),
        outputs=w["teacher"],
    )
    w["refresh_student_btn"].click(
        fn=lambda: gr.update(choices=discover_students()),
        outputs=w["student"],
    )
    w["refresh_dataset_btn"].click(
        fn=lambda: gr.update(choices=discover_datasets()),
        outputs=w["dataset"],
    )
    w["refresh_outdir_btn"].click(
        fn=lambda: gr.update(choices=discover_output_dirs()),
        outputs=w["output_dir"],
    )

    # ── Launch / Stop ────────────────────────────────────────────────────
    all_inputs = [
        w["stage"], w["backend"], w["use_open"],
        w["teacher"], w["student"],
        w["dataset"], w["output_dir"],
        w["epochs"], w["batch_size"], w["grad_acc"], w["lora_r"], w["max_samples"],
        w["sft_lr"], w["max_new_tokens_sft"], w["max_length"],
        w["minillm_temp"], w["minillm_lr"], w["num_generations"],
        w["max_completion_length"], w["eval_steps"],
        w["mlx_kd_temp"], w["mlx_lr"], w["mlx_eval_steps"],
        w["mlx_ce_alpha"], w["mlx_topk"], w["mlx_q_bits"], w["mlx_resume"],
        w["watchdog"],
    ]
    w["launch_btn"].click(fn=launch_run, inputs=all_inputs,
                          outputs=[w["log_box"], w["run_status"]])
    w["stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["run_status"]])
    w["clear_btn"].click(
        fn=clear_logs,
        outputs=[w["log_box"], w["log_status"], w["loss_plot"], w["grad_plot"],
                 w["ep_log_box"], w["ep_loss_plot"], w["ep_grad_plot"]],
    )

    # ── Domain synthesis wiring ──────────────────────────────────────────
    w["dom_refresh_btn"].click(
        fn=lambda: gr.update(choices=discover_teachers()), outputs=w["dom_teacher"],
    )
    for _domain, _btn, _n_slider, _target_slider, _outdir_box in [
        ("medical", w["med_btn"],   w["med_n"],   w["med_target"],   w["med_outdir"]),
        ("math",    w["math_btn"],  w["math_n"],  w["math_target"],  w["math_outdir"]),
        ("legal",   w["legal_btn"], w["legal_n"], w["legal_target"], w["legal_outdir"]),
        ("tax",     w["tax_btn"],   w["tax_n"],   w["tax_target"],   w["tax_outdir"]),
        ("coding",  w["code_btn"],  w["code_n"],  w["code_target"],  w["code_outdir"]),
        ("finance", w["fin_btn"],   w["fin_n"],   w["fin_target"],   w["fin_outdir"]),
    ]:
        def _make_handler(d):
            def _handler(teacher, n, batch, use_filter, target, outdir, offline, backend):
                return launch_magpie(teacher, outdir, n, batch, use_filter,
                                     target, offline, domain=d, backend=backend)
            return _handler

        _btn.click(
            fn=_make_handler(_domain),
            inputs=[w["dom_teacher"], _n_slider, w["dom_batch"], w["dom_filter"],
                    _target_slider, _outdir_box, w["dom_offline"], w["dom_backend"]],
            outputs=[w["log_box"], w["dom_status"]],
        )

    # ── Custom domain wiring ─────────────────────────────────────────────
    _custom_filter_inputs = [
        w["custom_id"], w["custom_label"], w["custom_desc"], w["custom_prompts"],
        w["custom_min_words"], w["custom_max_words"], w["custom_min_d2"],
        w["custom_req_code"], w["custom_req_nums"],
    ]
    w["custom_save_btn"].click(
        fn=save_custom_domain,
        inputs=_custom_filter_inputs,
        outputs=w["custom_save_status"],
    )
    w["custom_launch_btn"].click(
        fn=_save_and_launch,
        inputs=_custom_filter_inputs + [
            w["dom_teacher"], w["custom_n"], w["dom_batch"], w["dom_filter"],
            w["custom_target"], w["custom_outdir"], w["dom_offline"], w["dom_backend"],
        ],
        outputs=[w["log_box"], w["dom_status"]],
    )

    # ── Data Prep wiring ─────────────────────────────────────────────────
    w["mag_refresh_btn"].click(
        fn=lambda: gr.update(choices=discover_teachers()), outputs=w["mag_teacher"],
    )
    w["filt_ds_refresh"].click(
        fn=lambda: gr.update(choices=discover_datasets()), outputs=w["filt_dataset"],
    )
    w["synth_refresh_btn"].click(
        fn=lambda: gr.update(choices=discover_teachers()), outputs=w["synth_teacher"],
    )
    w["mag_launch_btn"].click(
        fn=launch_magpie,
        inputs=[w["mag_teacher"], w["mag_output_dir"], w["mag_n"], w["mag_batch_size"],
                w["mag_filter"], w["mag_target"], w["mag_offline"], w["mag_backend"]],
        outputs=[w["log_box"], w["mag_status"]],
    )
    w["mag_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["mag_status"]])
    w["synth_launch_btn"].click(
        fn=launch_synth,
        inputs=[w["synth_teacher"], w["synth_use_open"], w["synth_output_dir"],
                w["synth_n_generate"], w["synth_batch_size"], w["synth_temperature"],
                w["synth_seed_examples"], w["synth_offline"]],
        outputs=[w["log_box"], w["synth_status"]],
    )
    w["synth_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["synth_status"]])
    w["filt_launch_btn"].click(
        fn=launch_filter,
        inputs=[w["filt_dataset"], w["filt_output_dir"], w["filt_target"],
                w["filt_min_words"], w["filt_min_d2"], w["filt_offline"]],
        outputs=[w["log_box"], w["filt_status"]],
    )
    w["filt_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["filt_status"]])

    # ── Eval tab wiring ──────────────────────────────────────────────────
    w["eval_refresh_btn"].click(
        fn=lambda: gr.update(choices=discover_output_dirs()), outputs=w["eval_output_dir"],
    )
    w["ppl_launch_btn"].click(
        fn=launch_eval_perplexity,
        inputs=[w["eval_output_dir"], w["eval_checkpoint"], w["eval_student"],
                w["eval_dataset"], w["ppl_max_val"], w["ppl_batch"],
                w["ppl_compare_teacher"], w["ppl_teacher"], w["eval_offline"]],
        outputs=[w["log_box"], w["ppl_status"]],
    )
    w["ppl_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["ppl_status"]])
    w["qual_launch_btn"].click(
        fn=launch_eval_quality,
        inputs=[w["eval_output_dir"], w["eval_checkpoint"], w["eval_student"],
                w["eval_dataset"], w["qual_n_samples"], w["qual_judge"],
                w["qual_judge_teacher"], w["eval_offline"]],
        outputs=[w["log_box"], w["qual_status"]],
    )
    w["qual_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["qual_status"]])
    w["bench_launch_btn"].click(
        fn=launch_eval_benchmark,
        inputs=[w["eval_output_dir"], w["eval_checkpoint"], w["eval_student"],
                w["bench_n_seq"], w["bench_batch"], w["bench_baseline"],
                w["bench_threshold"], w["eval_offline"]],
        outputs=[w["log_box"], w["bench_status"]],
    )
    w["bench_stop_btn"].click(fn=stop_run, outputs=[w["log_box"], w["bench_status"]])

    # ── Expert Pipeline wiring ───────────────────────────────────────────
    _ep_ds_choices_fn = w.get("_ep_dataset_choices")
    if _ep_ds_choices_fn:
        w["ep_dataset_refresh"].click(
            fn=lambda: gr.update(choices=_ep_ds_choices_fn()),
            outputs=w["ep_dataset"],
        )
    w["ep_inspect_btn"].click(
        fn=_ep_inspect,
        inputs=w["ep_dataset"],
        outputs=[w["ep_inspect_status"], w["ep_instruction_col"],
                 w["ep_output_col"], w["ep_input_col"]],
    )
    w["ep_remap_btn"].click(
        fn=_ep_remap,
        inputs=[w["ep_dataset"], w["ep_instruction_col"], w["ep_output_col"],
                w["ep_input_col"], w["ep_max_samples_remap"], w["ep_remap_output"]],
        outputs=[w["log_box"], w["ep_status"]],
    )
    w["ep_domain"].change(
        fn=_ep_load_system_prompt,
        inputs=w["ep_domain"],
        outputs=w["ep_system_prompt"],
    )
    w["ep_cot_btn"].click(
        fn=_ep_cot,
        inputs=[w["ep_remap_output"], w["ep_teacher"], w["ep_domain"],
                w["ep_system_prompt"], w["ep_n_cot"], w["ep_cot_temp"],
                w["ep_max_tokens"], w["ep_ctx_size"], w["ep_n_parallel"],
                w["ep_batch_size_cot"], w["ep_cot_output"]],
        outputs=[w["log_box"], w["ep_status"]],
    )
    w["ep_cot_stop"].click(fn=stop_run, outputs=[w["log_box"], w["ep_status"]])
    w["ep_teacher_refresh"].click(fn=_ep_teacher_refresh, outputs=w["ep_teacher"])
    w["ep_distill_btn"].click(
        fn=_ep_distill,
        inputs=[w["ep_cot_output"], w["ep_distill_dataset"], w["ep_distill_output"],
                w["ep_distill_backend"], w["ep_epochs"], w["ep_lora_r"],
                w["ep_max_samp"], w["ep_open_chk"], w["ep_offline_chk"]],
        outputs=[w["log_box"], w["ep_status"]],
    )
    w["ep_distill_stop"].click(fn=stop_run, outputs=[w["log_box"], w["ep_status"]])

    # ── Poll logs every 2 s ──────────────────────────────────────────────
    gr.Timer(value=2).tick(
        fn=poll_logs,
        inputs=w["log_box"],
        outputs=[
            w["log_box"], w["log_status"], w["training_progress"],  # Live Logs tab
            w["launch_progress"],    # Configure & Launch tab
            w["data_prep_progress"], # Data Prep tab
            w["domain_progress"],    # Domain Synthesis tab
            w["eval_progress"],      # Eval tab
            w["dom_status"],         # Domain Synthesis status
            w["loss_plot"],          # Training loss chart
            w["grad_plot"],          # Gradient norm chart
            w["ep_progress"],        # Expert Pipeline tab progress
            w["ep_log_box"],         # Expert Pipeline embedded log
            w["ep_loss_plot"],       # Expert Pipeline loss chart
            w["ep_grad_plot"],       # Expert Pipeline grad norm chart
        ],
    )
