"""Subprocess launches, log streaming, and training progress."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

from .discovery import (
    discover_datasets,
    discover_output_dirs,
    discover_students,
    discover_teachers,
)
from ..infra.paths import project_dir, scripts_dir
from .runner_metrics import (
    parse_line_to_json as _parse_line_to_json,
    parse_progress_from_log as _parse_progress_from_log,
    extract_metrics_from_lines as _extract_metrics_from_lines_fn,
    loss_plot_data as _loss_plot_data_fn,
    progress_bar_html as _progress_bar_html,
)
from .runner_launch import (
    _build_cmd_launch as _build_cmd,  # re-exported for ui.py
    build_launch_run_cmd,
    save_custom_domain,
    build_magpie_cmd,
    build_filter_cmd,
    build_synth_cmd,
    build_eval_perplexity_cmd,
    build_eval_quality_cmd,
    build_eval_benchmark_cmd,
)

PROJECT_DIR = project_dir()
SCRIPTS_DIR = scripts_dir()
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------
_proc: subprocess.Popen | None = None
_log_queue: queue.Queue = queue.Queue()
_loss_history: list[dict] = []
_loss_step_counter: int = 0
_run_log_fh   = None
_run_json_fh  = None


def _stream_output(proc: subprocess.Popen) -> None:
    for line in proc.stdout:
        _log_queue.put(line)
        if _run_log_fh:
            try:
                _run_log_fh.write(line)
                _run_log_fh.flush()
            except Exception:
                pass
        if _run_json_fh:
            entry = _parse_line_to_json(line.rstrip())
            if entry:
                try:
                    _run_json_fh.write(json.dumps(entry) + "\n")
                    _run_json_fh.flush()
                except Exception:
                    pass
    proc.wait()
    exit_line = f"\n[Process exited with code {proc.returncode}]\n"
    _log_queue.put(exit_line)
    if _run_log_fh:
        try:
            _run_log_fh.write(exit_line)
            _run_log_fh.flush()
            _run_log_fh.close()
        except Exception:
            pass
    if _run_json_fh:
        try:
            _run_json_fh.close()
        except Exception:
            pass


def _seed_loss_history_from_dir(output_dir: str | Path) -> list[dict]:
    """Pre-populate loss history from a metrics.jsonl that already exists."""
    from ..infra.metrics_io import load_metrics
    try:
        rows = load_metrics(output_dir)
    except Exception:
        return []
    result = []
    for row in rows:
        if "loss" not in row and "eval_loss" not in row:
            continue
        entry: dict = {"step": row.get("step", 0)}
        if "loss" in row:
            entry["loss"] = row["loss"]
        elif "eval_loss" in row:
            entry["loss"] = row["eval_loss"]
        if "grad_norm" in row:
            entry["grad_norm"] = row["grad_norm"]
        result.append(entry)
    return result


def _start_proc(cmd: list[str]) -> tuple[str, str]:
    global _proc, _log_queue, _loss_history, _loss_step_counter, _run_log_fh, _run_json_fh
    if _proc is not None and _proc.poll() is None:
        return "A run is already in progress. Stop it first.", _run_status()
    _log_queue = queue.Queue()
    # Seed from existing metrics if output_dir already has training data
    _loss_history = []
    if "--output_dir" in cmd:
        idx = cmd.index("--output_dir")
        if idx + 1 < len(cmd):
            _loss_history = _seed_loss_history_from_dir(cmd[idx + 1])
    _loss_step_counter = len(_loss_history)
    for _fh in (_run_log_fh, _run_json_fh):
        if _fh:
            try:
                _fh.close()
            except Exception:
                pass
    _run_log_fh = _run_json_fh = None
    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    _proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, cwd=str(PROJECT_DIR), env=env,
    )
    threading.Thread(target=_stream_output, args=(_proc,), daemon=True).start()
    return f"Launched: {' '.join(cmd)}\n", _run_status()


def _ep_start_proc(cmd: list[str], label: str = "expert") -> tuple[str, str]:
    """Like _start_proc but also opens timestamped .log and .jsonl files under runs/."""
    global _run_log_fh, _run_json_fh
    import time as _t
    ts = _t.strftime("%Y%m%d-%H%M%S")
    log_dir = PROJECT_DIR / "runs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path  = log_dir / f"{label}_{ts}.log"
    json_path = log_dir / f"{label}_{ts}.jsonl"
    try:
        _run_log_fh  = open(log_path,  "w", buffering=1)
        _run_json_fh = open(json_path, "w", buffering=1)
    except OSError as e:
        _run_log_fh = _run_json_fh = None
        return f"Could not open log files: {e}", _run_status()
    msg, status = _start_proc(cmd)
    return msg + f"Logs → {log_path}\nJSON → {json_path}\n", status


def launch_run(
    stage, backend, use_open, teacher, student,
    dataset, output_dir, epochs, batch_size, grad_acc, lora_r, max_samples,
    sft_lr, max_new_tokens_sft, max_length,
    minillm_temp, minillm_lr, num_generations, max_completion_length, eval_steps,
    mlx_kd_temp, mlx_lr, mlx_eval_steps, mlx_ce_alpha, mlx_topk, mlx_q_bits, mlx_resume,
    watchdog,
):
    cmd = build_launch_run_cmd(
        stage, backend, use_open, teacher, student,
        dataset, output_dir, epochs, batch_size, grad_acc, lora_r, max_samples,
        sft_lr, max_new_tokens_sft, max_length,
        minillm_temp, minillm_lr, num_generations, max_completion_length, eval_steps,
        mlx_kd_temp, mlx_lr, mlx_eval_steps, mlx_ce_alpha, mlx_topk, mlx_q_bits, mlx_resume,
        watchdog,
    )
    return _start_proc(cmd)


def stop_run():
    global _proc
    if _proc is None or _proc.poll() is not None:
        return "No active run.", _run_status()
    _proc.kill()
    return "Sent SIGKILL.\n", _run_status()


def _run_status() -> str:
    if _proc is None:
        return "idle"
    if _proc.poll() is None:
        return f"running  (pid {_proc.pid})"
    return f"finished  (exit {_proc.returncode})"


def poll_logs(current_log: str):
    global _loss_history, _loss_step_counter
    lines = []
    try:
        while True:
            lines.append(_log_queue.get_nowait())
    except queue.Empty:
        pass
    new_metrics, _loss_step_counter = _extract_metrics_from_lines_fn(lines, _loss_step_counter)
    if new_metrics:
        _loss_history.extend(new_metrics)
    new_log = current_log + "".join(lines)
    running = _proc is not None and _proc.poll() is None
    frac, label = _parse_progress_from_log(new_log)
    html = _progress_bar_html(frac, label, running=running)
    status = _run_status()
    loss_df, grad_df = _loss_plot_data_fn(_loss_history)
    return new_log, status, html, html, html, html, html, status, loss_df, grad_df, html, new_log, loss_df, grad_df


def clear_logs():
    global _loss_history, _loss_step_counter
    _loss_history = []
    _loss_step_counter = 0
    seed_loss = pd.DataFrame({"step": [0], "loss": [0.0]})
    seed_grad = pd.DataFrame({"step": [0], "grad_norm": [0.0]})
    return "", _run_status(), seed_loss, seed_grad, "", seed_loss, seed_grad


def launch_magpie(teacher, output_dir, n, batch_size, use_filter, target, offline,
                  domain=None, backend="auto"):
    cmd = build_magpie_cmd(teacher, output_dir, n, batch_size, use_filter, target, offline,
                           domain=domain, backend=backend)
    return _start_proc(cmd)


def launch_filter(dataset, output_dir, target, min_response_words, min_distinct2, offline):
    cmd = build_filter_cmd(dataset, output_dir, target, min_response_words, min_distinct2, offline)
    return _start_proc(cmd)


def launch_synth(teacher, use_open, output_dir, n_generate, batch_size, temperature,
                 seed_examples, offline):
    cmd = build_synth_cmd(teacher, use_open, output_dir, n_generate, batch_size, temperature,
                          seed_examples, offline)
    return _start_proc(cmd)


def launch_eval_perplexity(output_dir, checkpoint, student, dataset, max_val_samples,
                           batch_size, compare_teacher, teacher, offline):
    cmd = build_eval_perplexity_cmd(output_dir, checkpoint, student, dataset, max_val_samples,
                                    batch_size, compare_teacher, teacher, offline)
    return _start_proc(cmd)


def launch_eval_quality(output_dir, checkpoint, student, dataset, n_samples,
                        judge, judge_teacher, offline):
    cmd = build_eval_quality_cmd(output_dir, checkpoint, student, dataset, n_samples,
                                 judge, judge_teacher, offline)
    return _start_proc(cmd)


def launch_eval_benchmark(output_dir, checkpoint, student, n_sequences, batch_size,
                          baseline_dir, threshold, offline):
    cmd = build_eval_benchmark_cmd(output_dir, checkpoint, student, n_sequences, batch_size,
                                   baseline_dir, threshold, offline)
    return _start_proc(cmd)
