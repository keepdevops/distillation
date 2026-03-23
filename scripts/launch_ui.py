#!/usr/bin/env python3
"""
Distillation launcher UI — parameter form with live dropdowns.

Usage:
    python3 scripts/launch_ui.py          # auto-relaunches via pixi if needed
    pixi run python scripts/launch_ui.py --port 7861
"""

import argparse
import base64
import json
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd

# Auto-relaunch through pixi if gradio is not importable
try:
    import gradio as gr
except ModuleNotFoundError:
    pixi = Path(__file__).parent.parent / ".pixi" / "envs" / "default" / "bin" / "python"
    if pixi.exists():
        print(f"Re-launching with pixi python: {pixi}")
        raise SystemExit(subprocess.call([str(pixi)] + sys.argv))
    else:
        sys.exit("gradio not found. Run: cd /Users/caribou/distill && pixi run python scripts/launch_ui.py")

SCRIPTS_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPTS_DIR.parent
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Known good presets (shown even if not yet cached)
# ---------------------------------------------------------------------------
KNOWN_TEACHERS = [
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-7B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.2-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "ibm-granite/granite-3.1-8b-instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
]
KNOWN_STUDENTS = [
    "Qwen/Qwen2-0.5B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
    "HuggingFaceTB/SmolLM2-360M-Instruct",
]
KNOWN_DATASETS = [
    "tatsu-lab/alpaca",
    "yahma/alpaca-cleaned",
    "mlabonne/guanaco-llama2-1k",
    "HuggingFaceH4/no_robots",
    "teknium/OpenHermes-2.5",
    "Magpie-Align/Magpie-Pro-300K-Filtered",
    "argilla/distilabel-capybara-dpo-7k-binarized",
    "bigcode/self-oss-instruct-sc2-exec-filter-50k",
]

# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def _is_hf_model_dir(d: Path) -> bool:
    if not (d / "config.json").exists():
        return False
    return bool(
        list(d.glob("*.safetensors"))
        or list(d.glob("model*.bin"))
        or (d / "pytorch_model.bin").exists()
    )


def _scan_hf_hub_cache(hub_root: Path) -> list[str]:
    """Return HF model IDs found in a hub cache directory."""
    results = []
    if not hub_root.exists():
        return results
    try:
        for entry in sorted(hub_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            label = entry.name[len("models--"):].replace("--", "/", 1)
            snaps = entry / "snapshots"
            if not snaps.exists():
                continue
            try:
                for snap in sorted(snaps.iterdir(),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if snap.is_dir() and _is_hf_model_dir(snap):
                        results.append(label)
                        break
            except PermissionError:
                continue
    except PermissionError:
        pass
    return results


def _scan_datasets_cache(cache_root: Path) -> list[str]:
    """Return dataset IDs found in an HF datasets cache directory."""
    results = []
    if not cache_root.exists():
        return results
    try:
        for entry in sorted(cache_root.iterdir()):
            if not entry.is_dir() or not entry.name.startswith("datasets--"):
                continue
            label = entry.name[len("datasets--"):].replace("--", "/", 1)
            results.append(label)
    except PermissionError:
        pass
    return results


def _scan_local_checkpoints(search_root: Path, _max_depth: int = 5) -> list[str]:
    """Return local directories that look like HF model checkpoints.

    Bounded to _max_depth levels below search_root to avoid unbounded rglob
    over the entire filesystem when search_root is the project directory.
    """
    results = []
    if not search_root.exists():
        return results
    try:
        candidates = sorted(
            (
                p for p in search_root.rglob("config.json")
                if len(p.relative_to(search_root).parts) <= _max_depth
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for d in candidates:
            parent = d.parent
            if _is_hf_model_dir(parent):
                results.append(str(parent))
    except (PermissionError, OSError):
        pass
    return results


def discover_teachers() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    # Local checkpoints (trained outputs)
    for p in _scan_local_checkpoints(PROJECT_DIR):
        add(p)

    # HF hub caches
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for m in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        add(m)
    for m in _scan_hf_hub_cache(SCRIPTS_DIR / "hf_cache"):
        add(m)

    # Known presets (fill gaps)
    for m in KNOWN_TEACHERS:
        add(m)
    return result


def discover_students() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    # SFT checkpoint first
    sft = PROJECT_DIR / "distilled-minillm" / "sft_checkpoint"
    if _is_hf_model_dir(sft):
        add(str(sft))

    # All local checkpoints
    for p in _scan_local_checkpoints(PROJECT_DIR):
        add(p)

    # HF hub caches
    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for m in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        add(m)
    for m in _scan_hf_hub_cache(SCRIPTS_DIR / "hf_cache"):
        add(m)

    # Known presets
    for m in KNOWN_STUDENTS:
        add(m)
    return result


def discover_datasets() -> list[str]:
    seen, result = set(), []
    def add(v):
        if v not in seen:
            seen.add(v); result.append(v)

    hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
    for ds in _scan_datasets_cache(Path(hf_home) / "datasets"):
        add(ds)
    for ds in _scan_datasets_cache(SCRIPTS_DIR / "datasets_cache"):
        add(ds)

    for ds in KNOWN_DATASETS:
        add(ds)
    return result


def discover_output_dirs() -> list[str]:
    result = []
    try:
        for d in sorted(PROJECT_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if d.is_dir() and not d.name.startswith(".") and d.name != "scripts":
                result.append(str(d))
    except OSError:
        pass
    default = str(PROJECT_DIR / "distilled-minillm")
    if default not in result:
        result.insert(0, default)
    return result


# ---------------------------------------------------------------------------
# Process state
# ---------------------------------------------------------------------------
_proc: subprocess.Popen | None = None
_log_queue: queue.Queue = queue.Queue()
_loss_history: list[dict] = []
_loss_step_counter: int = 0
_run_log_fh   = None   # open text file handle for current run (or None)
_run_json_fh  = None   # open jsonl file handle for current run (or None)

_LOG_STEP_RE   = re.compile(r"(?<!\w)step=(\d+)")
_LOG_EPOCH_RE  = re.compile(r"(?<!\w)epoch=([\d.]+)")
_LOG_LOSS_RE   = re.compile(r"(?<!\w)loss=([\d.eE+\-]+)")
_LOG_ELOSS_RE  = re.compile(r"(?<!\w)eval_loss=([\d.eE+\-]+)")
_LOG_GRAD_RE   = re.compile(r"['\"]grad_norm['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_LR_RE     = re.compile(r"['\"]learning_rate['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_PT_LOSS   = re.compile(r"['\"]loss['\"]\s*:\s*['\"]?([\d.eE+\-]+)")
_LOG_PT_EPOCH  = re.compile(r"['\"]epoch['\"]\s*:\s*['\"]?([\d.]+)")


def _parse_line_to_json(line: str) -> dict | None:
    """Extract structured metrics from a log line. Returns None if no metrics found."""
    import time as _t
    entry: dict = {}
    m = _LOG_STEP_RE.search(line)
    if m:
        entry["step"] = int(m.group(1))
    m = _LOG_EPOCH_RE.search(line)
    if m:
        entry["epoch"] = float(m.group(1))
    m = _LOG_ELOSS_RE.search(line)
    if m:
        entry["eval_loss"] = float(m.group(1))
    elif (m := _LOG_LOSS_RE.search(line)):
        entry["loss"] = float(m.group(1))
    if "loss" not in entry and "eval_loss" not in entry:
        m = _LOG_PT_LOSS.search(line)
        if m:
            entry["loss"] = float(m.group(1))
        m = _LOG_PT_EPOCH.search(line)
        if m:
            entry["epoch"] = float(m.group(1))
    if not entry:
        return None
    m = _LOG_GRAD_RE.search(line)
    if m:
        entry["grad_norm"] = float(m.group(1))
    m = _LOG_LR_RE.search(line)
    if m:
        entry["lr"] = float(m.group(1))
    entry["ts"] = _t.strftime("%Y-%m-%dT%H:%M:%S")
    entry["msg"] = line.strip()
    return entry


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


# ---------------------------------------------------------------------------
# Launch / stop
# ---------------------------------------------------------------------------

def _build_cmd(script: str, params: dict) -> list[str]:
    cmd = [PYTHON, str(SCRIPTS_DIR / script)]
    for flag, val in params.items():
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{flag}")
        else:
            cmd += [f"--{flag}", str(val)]
    return cmd


def _start_proc(cmd: list[str]) -> tuple[str, str]:
    global _proc, _log_queue, _loss_history, _loss_step_counter, _run_log_fh, _run_json_fh
    if _proc is not None and _proc.poll() is None:
        return "A run is already in progress. Stop it first.", _run_status()
    _log_queue = queue.Queue()
    _loss_history = []
    _loss_step_counter = 0
    # Close any leftover file handles from a previous run
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
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(PROJECT_DIR),
        env=env,
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
    stage, backend, use_open,
    teacher, student,
    dataset, output_dir,
    epochs, batch_size, grad_acc, lora_r, max_samples,
    # SFT
    sft_lr, max_new_tokens_sft, max_length,
    # MiniLLM/PyTorch
    minillm_temp, minillm_lr, num_generations, max_completion_length, eval_steps,
    # MLX
    mlx_kd_temp, mlx_lr, mlx_eval_steps, mlx_ce_alpha, mlx_topk, mlx_q_bits, mlx_resume,
    watchdog,
):
    params: dict = {
        "output_dir": output_dir,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_acc": int(grad_acc),
        "lora_r": int(lora_r),
        "max_samples": int(max_samples),
        "watchdog": watchdog,
    }

    if use_open:
        params["open"] = True
    else:
        if teacher:
            params["teacher"] = teacher
        if student:
            params["student"] = student

    if dataset and dataset != "tatsu-lab/alpaca":
        params["dataset"] = dataset

    if stage == "SFT":
        script = "distill_sft.py"
        params["learning_rate"] = float(sft_lr)
        params["max_new_tokens"] = int(max_new_tokens_sft)
        params["max_length"] = int(max_length)
    elif backend == "MLX":
        script = "distill_mlx.py"
        params["kd_temp"] = float(mlx_kd_temp)
        params["learning_rate"] = float(mlx_lr)
        params["eval_steps"] = int(mlx_eval_steps)
        params["ce_alpha"] = float(mlx_ce_alpha)
        params["topk_logits"] = int(mlx_topk)
        params["q_bits"] = int(mlx_q_bits)
        if mlx_resume:
            params["resume"] = True
    else:
        script = "distill_minillm.py"
        params["minillm_temp"] = float(minillm_temp)
        params["learning_rate"] = float(minillm_lr)
        params["num_generations"] = int(num_generations)
        params["max_new_tokens"] = int(max_completion_length)
        params["eval_steps"] = int(eval_steps)

    cmd = _build_cmd(script, params)
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
    global _loss_history
    lines = []
    try:
        while True:
            lines.append(_log_queue.get_nowait())
    except queue.Empty:
        pass
    new_metrics = _extract_metrics_from_lines(lines)
    if new_metrics:
        _loss_history.extend(new_metrics)
    new_log = current_log + "".join(lines)
    running = _proc is not None and _proc.poll() is None
    frac, label = _parse_progress_from_log(new_log)
    html = _progress_bar_html(frac, label, running=running)
    status = _run_status()
    loss_df, grad_df = _loss_plot_data()
    return new_log, status, html, html, html, html, html, status, loss_df, grad_df, html, new_log, loss_df, grad_df


def clear_logs():
    global _loss_history, _loss_step_counter
    _loss_history = []
    _loss_step_counter = 0
    seed_loss = pd.DataFrame({"step": [0], "loss": [0.0]})
    seed_grad = pd.DataFrame({"step": [0], "grad_norm": [0.0]})
    return "", _run_status(), seed_loss, seed_grad, "", seed_loss, seed_grad


def _parse_progress_from_log(log_text: str) -> tuple[float, str]:
    """Return (fraction 0–1, label) from the last progress indicator in log_text."""
    # tqdm-style: "75%|████████|"
    m = list(re.finditer(r'\b(\d{1,3})%\|', log_text))
    if m:
        pct = int(m[-1].group(1))
        return pct / 100, f"{pct}%"
    # "Step X/Y" or "step X/Y"
    m = list(re.finditer(r'[Ss]tep\s+(\d+)\s*/\s*(\d+)', log_text))
    if m:
        x, y = int(m[-1].group(1)), int(m[-1].group(2))
        return (x / y if y > 0 else 0.0), f"Step {x}/{y}"
    # "Epoch X/Y"
    m = list(re.finditer(r'[Ee]poch[:\s]+(\d+)\s*/\s*(\d+)', log_text))
    if m:
        x, y = int(m[-1].group(1)), int(m[-1].group(2))
        return (x / y if y > 0 else 0.0), f"Epoch {x}/{y}"
    # Bare "Progress: 45%"
    m = list(re.finditer(r'[Pp]rogress[:\s]+(\d+)%', log_text))
    if m:
        pct = int(m[-1].group(1))
        return pct / 100, f"{pct}%"
    return 0.0, ""


def _extract_metrics_from_lines(lines: list[str]) -> list[dict]:
    """Parse loss/grad_norm from trainer log lines into plot-ready dicts.

    Handles two formats:
      MLX:     step=320  epoch=1.70  loss=5.2211  0.11 steps/s
      PyTorch: {'loss': '1.715', 'grad_norm': '0.0989', 'epoch': '0.32'}
    """
    global _loss_step_counter
    results = []
    for line in lines:
        # Skip eval_loss lines (separate metric, not train loss)
        if "eval_loss" in line and "loss=" not in line.replace("eval_loss", ""):
            continue

        # MLX format: loss=5.2211
        loss_m = re.search(r"(?<!['\"\w])loss=([\d.eE+\-]+)", line)
        # PyTorch format: 'loss': '1.715'  or  "loss": 1.715
        if not loss_m:
            loss_m = re.search(r"['\"]loss['\"]\s*:\s*['\"]?([\d.eE+\-]+)['\"]?", line)
        if not loss_m:
            continue
        try:
            loss = float(loss_m.group(1))
        except ValueError:
            continue

        _loss_step_counter += 1
        # step= (MLX) or 'step': N (PyTorch)
        step_m = re.search(r"(?<!['\"\w])step=(\d+)", line) or \
                 re.search(r"['\"]step['\"]\s*:\s*(\d+)", line)
        epoch_m = re.search(r"(?<!['\"\w])epoch=([\d.]+)", line) or \
                  re.search(r"['\"]epoch['\"]\s*:\s*['\"]?([\d.]+)['\"]?", line)
        grad_m = re.search(r"['\"]grad_norm['\"]\s*:\s*['\"]?([\d.eE+\-]+)['\"]?", line)

        step = int(step_m.group(1)) if step_m else _loss_step_counter
        entry: dict = {"step": step, "loss": loss}
        if epoch_m:
            entry["epoch"] = float(epoch_m.group(1))
        if grad_m:
            try:
                entry["grad_norm"] = float(grad_m.group(1))
            except ValueError:
                pass
        results.append(entry)
    return results


def _loss_plot_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _loss_history:
        return pd.DataFrame({"step": [0], "loss": [0.0]}), pd.DataFrame({"step": [0], "grad_norm": [0.0]})
    loss_rows = [{"step": e["step"], "loss": e["loss"]} for e in _loss_history]
    grad_rows = [{"step": e["step"], "grad_norm": e["grad_norm"]}
                 for e in _loss_history if "grad_norm" in e]
    return pd.DataFrame(loss_rows), pd.DataFrame(grad_rows) if grad_rows else pd.DataFrame({"step": [], "grad_norm": []})


def _progress_bar_html(fraction: float, label: str, running: bool = True) -> str:
    """Return an HTML snippet for a compact progress bar, or '' when idle."""
    if not running and not label:
        return ""
    pct = max(0, min(100, int(fraction * 100)))
    color = "#2563eb" if running else "#16a34a"
    status = label or ("Running…" if running else "Done")
    return (
        '<div style="margin:4px 0 6px;">'
        '<div style="display:flex;justify-content:space-between;font-size:11px;'
        'color:#6b7280;margin-bottom:2px;">'
        f'<span>{status}</span><span>{pct}%</span></div>'
        '<div style="background:#e5e7eb;border-radius:3px;height:6px;overflow:hidden;">'
        f'<div style="width:{pct}%;background:{color};height:6px;'
        'border-radius:3px;transition:width 0.3s ease;"></div>'
        '</div></div>'
    )


def launch_magpie(teacher, output_dir, n, batch_size, use_filter, target, offline,
                  domain=None, backend="auto"):
    """Launch magpie_synth.py. Pass domain= for specialized synthesis."""
    params: dict = {
        "output_dir": output_dir,
        "n": int(n),
        "batch_size": int(batch_size),
        "filter": use_filter,
        "backend": backend,
    }
    if domain:
        params["domain"] = domain
    if teacher and teacher != "Qwen/Qwen2-1.5B-Instruct":
        params["teacher"] = teacher
    if use_filter and target and int(target) > 0:
        params["target"] = int(target)
    if offline:
        params["offline"] = True
    cmd = _build_cmd("magpie_synth.py", params)
    return _start_proc(cmd)


_DOMAINS_FILE = PROJECT_DIR / "configs" / "domain_prompts.json"


def save_custom_domain(domain_id, label, description, system_prompts_text,
                       min_resp_words, max_resp_words, min_d2,
                       require_code, require_numbers):
    """Write a new domain entry to configs/domain_prompts.json."""
    domain_id = (domain_id or "").strip().lower().replace(" ", "_")
    if not domain_id:
        return "Error: Domain ID is required."

    prompts = [line.strip() for line in system_prompts_text.splitlines() if line.strip()]
    if not prompts:
        return "Error: At least one system prompt is required."

    try:
        registry = {}
        if _DOMAINS_FILE.exists():
            with open(_DOMAINS_FILE) as f:
                registry = json.load(f)

        registry[domain_id] = {
            "label":          (label or domain_id).strip(),
            "description":    (description or "").strip(),
            "system_prompts": prompts,
            "filter": {
                "min_resp_words":  int(min_resp_words),
                "max_resp_words":  int(max_resp_words),
                "min_d2":          round(float(min_d2), 3),
                "require_code":    bool(require_code),
                "require_numbers": bool(require_numbers),
            },
        }

        _DOMAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DOMAINS_FILE, "w") as f:
            json.dump(registry, f, indent=2)

        return f"Saved domain '{domain_id}' to configs/domain_prompts.json  ({len(prompts)} system prompts)."
    except Exception as e:
        return f"Error saving domain: {e}"


def launch_filter(dataset, output_dir, target, min_response_words, min_distinct2, offline):
    params: dict = {
        "dataset": dataset,
        "output_dir": output_dir,
        "target": int(target),
        "min_response_words": int(min_response_words),
        "min_distinct2": float(min_distinct2),
    }
    if offline:
        params["offline"] = True
    cmd = _build_cmd("filter_dataset.py", params)
    return _start_proc(cmd)


def launch_synth(teacher, use_open, output_dir, n_generate, batch_size, temperature,
                 seed_examples, offline):
    params: dict = {
        "output_dir": output_dir,
        "n_generate": int(n_generate),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "seed_examples": int(seed_examples),
    }
    if use_open:
        params["open"] = True
    elif teacher:
        params["teacher"] = teacher
    if offline:
        params["offline"] = True
    cmd = _build_cmd("generate_synthetic_data.py", params)
    return _start_proc(cmd)


def launch_eval_perplexity(output_dir, checkpoint, student, dataset, max_val_samples,
                           batch_size, compare_teacher, teacher, offline):
    cmd = [PYTHON, str(SCRIPTS_DIR / "run_eval.py"), output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    if dataset and dataset != "tatsu-lab/alpaca":
        cmd += ["--dataset", dataset]
    cmd += ["--max_val_samples", str(int(max_val_samples))]
    cmd += ["--batch_size", str(int(batch_size))]
    if compare_teacher:
        cmd.append("--compare_teacher")
        if teacher:
            cmd += ["--teacher", teacher]
    if offline:
        cmd.append("--offline")
    return _start_proc(cmd)


def launch_eval_quality(output_dir, checkpoint, student, dataset, n_samples,
                        judge, judge_teacher, offline):
    cmd = [PYTHON, str(SCRIPTS_DIR / "eval_quality.py"), output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    if dataset and dataset != "tatsu-lab/alpaca":
        cmd += ["--dataset", dataset]
    cmd += ["--n_samples", str(int(n_samples))]
    if judge:
        cmd.append("--judge")
        if judge_teacher:
            cmd += ["--teacher", judge_teacher]
    if offline:
        cmd.append("--offline")
    return _start_proc(cmd)


def launch_eval_benchmark(output_dir, checkpoint, student, n_sequences, batch_size,
                          baseline_dir, threshold, offline):
    cmd = [PYTHON, str(SCRIPTS_DIR / "run_benchmarks.py"), output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    cmd += ["--n_sequences", str(int(n_sequences))]
    cmd += ["--batch_size", str(int(batch_size))]
    if baseline_dir:
        cmd += ["--baseline_dir", baseline_dir]
    cmd += ["--threshold", str(float(threshold))]
    if offline:
        cmd.append("--offline")
    return _start_proc(cmd)


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

def build_ui():
    # Discover once — reused by all tabs to avoid repeated filesystem scans.
    teachers = discover_teachers()
    students = discover_students()
    datasets = discover_datasets()
    out_dirs = discover_output_dirs()

    default_teacher = "Qwen/Qwen2-1.5B-Instruct"
    default_student = students[0] if students else "Qwen/Qwen2-0.5B-Instruct"
    default_dataset = "yahma/alpaca-cleaned"
    default_out     = str(PROJECT_DIR / "distilled-minillm")

    with gr.Blocks(title="Distillation Launcher") as demo:
        gr.Markdown("# Distillation Launcher")

        with gr.Tabs():

            # ── Tab 1: Configure & Launch ────────────────────────────────────
            with gr.Tab("Configure & Launch"):

                # Stage + Backend
                with gr.Row():
                    stage = gr.Radio(
                        ["SFT", "MiniLLM"], value="MiniLLM",
                        label="Stage",
                        info="SFT = teacher-label warmup.  MiniLLM = reverse-KL distillation.",
                    )
                    backend = gr.Radio(
                        ["PyTorch", "MLX"], value="MLX",
                        label="Backend",
                        info="PyTorch/MPS = stable, full-featured.  MLX = Apple-native, 2-5× faster on M3.",
                    )
                    use_open = gr.Checkbox(
                        value=True,
                        label="Use open Qwen2 models (1.5B→0.5B, no HF login required)",
                    )

                # Models
                gr.Markdown("### Models")
                with gr.Row():
                    teacher = gr.Dropdown(
                        choices=teachers,
                        value=default_teacher,
                        label="Teacher model",
                        allow_custom_value=True,
                        scale=3,
                        info="Select from cache or type a HuggingFace model ID",
                    )
                    refresh_teacher_btn = gr.Button("Refresh", scale=1, size="sm")

                with gr.Row():
                    student = gr.Dropdown(
                        choices=students,
                        value=default_student,
                        label="Student model / checkpoint",
                        allow_custom_value=True,
                        scale=3,
                        info="Select local checkpoint or HF model ID. For MiniLLM, point to your SFT checkpoint.",
                    )
                    refresh_student_btn = gr.Button("Refresh", scale=1, size="sm")

                # Dataset & Output
                gr.Markdown("### Dataset & Output")
                with gr.Row():
                    dataset = gr.Dropdown(
                        choices=datasets,
                        value=default_dataset,
                        label="Dataset",
                        allow_custom_value=True,
                        scale=3,
                        info="Select from local cache or type a HuggingFace dataset ID",
                    )
                    refresh_dataset_btn = gr.Button("Refresh", scale=1, size="sm")

                with gr.Row():
                    output_dir = gr.Dropdown(
                        choices=out_dirs,
                        value=default_out,
                        label="Output directory",
                        allow_custom_value=True,
                        scale=3,
                    )
                    refresh_outdir_btn = gr.Button("Refresh", scale=1, size="sm")

                # Common training params
                gr.Markdown("### Training")
                with gr.Row():
                    epochs       = gr.Slider(1, 10, value=2, step=1, label="Epochs")
                    max_samples  = gr.Slider(50, 10000, value=2000, step=50, label="Max samples")
                with gr.Row():
                    batch_size   = gr.Slider(1, 32, value=8,  step=1, label="Batch size")
                    grad_acc     = gr.Slider(1, 32, value=8,  step=1, label="Gradient accumulation")
                    lora_r       = gr.Slider(4, 128, value=16, step=4, label="LoRA rank")

                # SFT-specific
                with gr.Group(visible=False) as sft_group:
                    gr.Markdown("### SFT options")
                    with gr.Row():
                        sft_lr            = gr.Number(value=2e-4, label="Learning rate")
                        max_new_tokens_sft = gr.Slider(64, 512, value=128, step=32,
                                                        label="Teacher max new tokens")
                        max_length        = gr.Slider(128, 1024, value=384, step=64,
                                                       label="Max sequence length")

                # MiniLLM/PyTorch-specific
                with gr.Group(visible=False) as minillm_group:
                    gr.Markdown("### MiniLLM options  (PyTorch / MPS)")
                    with gr.Row():
                        minillm_temp          = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                                           label="KD temperature")
                        minillm_lr            = gr.Number(value=2e-5, label="Learning rate",
                                                           precision=6)
                        num_generations       = gr.Slider(2, 16, value=4, step=1,
                                                           label="Generations per prompt",
                                                           info="4+ gives GRPO enough reward variance for non-zero advantage")
                        max_completion_length = gr.Slider(64, 512, value=256, step=32,
                                                           label="Max completion length (tokens)",
                                                           info="256 gives completions room to terminate naturally; update _MAX_NATURAL_CHARS if changed")
                        eval_steps            = gr.Slider(1, 50, value=20, step=1,
                                                           label="Eval every N steps")

                # MLX-specific
                with gr.Group(visible=True) as mlx_group:
                    gr.Markdown("### MLX options  (Apple-native, 2-5× faster on M3)")
                    gr.Markdown(
                        "_MLX uses Apple-optimised defaults (batch=2, grad_acc=8, lora_r=16). "
                        "Teacher logits are precomputed once then freed — both models never in memory together._"
                    )
                    with gr.Row():
                        mlx_kd_temp   = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                                   label="KD temperature")
                        mlx_lr        = gr.Number(value=2e-4, label="Learning rate", precision=6)
                        mlx_eval_steps = gr.Slider(1, 100, value=50, step=1,
                                                    label="Eval every N steps")
                    with gr.Row():
                        mlx_ce_alpha  = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                                   label="CE alpha (0=pure KD, 1=pure CE)",
                                                   info="0.2 mixes CE for stability without losing KD signal")
                        mlx_topk      = gr.Slider(10, 200, value=50, step=10,
                                                   label="Top-K teacher logits",
                                                   info="50 captures >99% of teacher probability mass (~300 MB vs ~300 GB full vocab)")
                        mlx_q_bits    = gr.Radio([4, 8], value=4, label="Export quantization bits")
                        mlx_resume    = gr.Checkbox(value=False,
                                                    label="Resume from last checkpoint")

                with gr.Row():
                    watchdog = gr.Checkbox(value=False, label="Enable watchdog (pause.flag callback)")

                with gr.Row():
                    launch_btn = gr.Button("Launch", variant="primary", scale=3)
                    stop_btn   = gr.Button("Stop",   variant="stop",    scale=1)

                run_status = gr.Textbox(value="idle", label="Run status", interactive=False)
                launch_progress = gr.HTML(value="")

            # ── Tab 2: Data Prep ─────────────────────────────────────────────
            with gr.Tab("Data Prep"):
                data_prep_progress = gr.HTML(value="")

                # ── Magpie synthesis ─────────────────────────────────────────
                gr.Markdown("### Magpie Synthesis")
                gr.Markdown(
                    "Generate instruction-response pairs from the teacher by conditioning "
                    "on its chat template. Produces an HF dataset in `output_dir/hf_dataset/` "
                    "that can be used directly as a distillation dataset."
                )
                with gr.Row():
                    mag_teacher = gr.Dropdown(
                        choices=teachers,
                        value="Qwen/Qwen2-1.5B-Instruct",
                        label="Teacher model",
                        allow_custom_value=True,
                        scale=4,
                    )
                    mag_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
                with gr.Row():
                    mag_output_dir = gr.Textbox(
                        value=str(PROJECT_DIR / "magpie_data"),
                        label="Output directory",
                        scale=4,
                    )
                with gr.Row():
                    mag_n          = gr.Slider(500, 50000, value=5000, step=500,
                                               label="Pairs to generate (before filtering)")
                    mag_batch_size = gr.Slider(1, 64, value=32, step=1,
                                               label="Generation batch size",
                                               info="For MLX this is loop chunk size (generation is sequential but fast)")
                with gr.Row():
                    mag_backend = gr.Radio(
                        ["auto", "mlx", "mps"], value="auto",
                        label="Backend",
                        info="auto picks MLX on Apple Silicon (2-4× faster than MPS)",
                    )
                    mag_filter  = gr.Checkbox(value=True, label="Filter output (dedup + quality)")
                    mag_target  = gr.Slider(500, 20000, value=2000, step=500,
                                            label="Target keep (top-N after filter)")
                    mag_offline = gr.Checkbox(value=False, label="Offline (use cached model)")
                with gr.Row():
                    mag_launch_btn = gr.Button("Generate", variant="primary", scale=3)
                    mag_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                mag_status = gr.Textbox(value="idle", label="Status", interactive=False)

                gr.Markdown("---")

                # ── Self-Instruct synthesis ───────────────────────────────────
                gr.Markdown("### Self-Instruct Synthesis")
                gr.Markdown(
                    "Generate synthetic instruction-response pairs via self-instruct: "
                    "the teacher generates new instructions from seed examples, then "
                    "generates responses. Includes perplexity + quality filtering. "
                    "Output is an HF dataset in `output_dir/synthetic_data/`."
                )
                with gr.Row():
                    synth_teacher = gr.Dropdown(
                        choices=teachers,
                        value="Qwen/Qwen2-1.5B-Instruct",
                        label="Teacher model",
                        allow_custom_value=True,
                        scale=3,
                    )
                    synth_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
                with gr.Row():
                    synth_use_open  = gr.Checkbox(value=True, label="Use open Qwen2 teacher (no HF login)")
                    synth_offline   = gr.Checkbox(value=False, label="Offline (use cached model)")
                with gr.Row():
                    synth_output_dir = gr.Textbox(
                        value=str(PROJECT_DIR / "distilled-minillm"),
                        label="Output directory",
                        scale=4,
                    )
                with gr.Row():
                    synth_n_generate   = gr.Slider(100, 20000, value=2000, step=100,
                                                    label="Target pairs to generate")
                    synth_batch_size   = gr.Slider(1, 32, value=8, step=1,
                                                    label="Batch size")
                    synth_temperature  = gr.Slider(0.5, 1.5, value=0.9, step=0.1,
                                                    label="Sampling temperature")
                    synth_seed_examples = gr.Slider(2, 20, value=5, step=1,
                                                     label="Seed examples per prompt")
                with gr.Row():
                    synth_launch_btn = gr.Button("Synthesize", variant="primary", scale=3)
                    synth_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                synth_status = gr.Textbox(value="idle", label="Status", interactive=False)

                gr.Markdown("---")

                # ── Dataset filter ────────────────────────────────────────────
                gr.Markdown("### Dataset Filter")
                gr.Markdown(
                    "Filter any alpaca-format dataset by quality: length bounds, "
                    "distinct-2 coherence, refusal detection, and near-dedup (Jaccard). "
                    "Output is an HF dataset loadable by the training scripts."
                )
                with gr.Row():
                    filt_dataset = gr.Dropdown(
                        choices=datasets,
                        value="yahma/alpaca-cleaned",
                        label="Dataset",
                        allow_custom_value=True,
                        scale=4,
                    )
                    filt_ds_refresh = gr.Button("Refresh", scale=1, size="sm")
                with gr.Row():
                    filt_output_dir = gr.Textbox(
                        value=str(PROJECT_DIR / "filtered_data"),
                        label="Output directory",
                        scale=4,
                    )
                with gr.Row():
                    filt_target      = gr.Slider(500, 50000, value=5000, step=500,
                                                 label="Target top-N")
                    filt_min_words   = gr.Slider(5, 100, value=20, step=5,
                                                 label="Min response words")
                    filt_min_d2      = gr.Slider(0.1, 0.9, value=0.35, step=0.05,
                                                 label="Min distinct-2")
                    filt_offline     = gr.Checkbox(value=False, label="Offline")
                with gr.Row():
                    filt_launch_btn = gr.Button("Filter", variant="primary", scale=3)
                    filt_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                filt_status = gr.Textbox(value="idle", label="Status", interactive=False)

            # ── Tab 3: Specialized Domain Synthesis ──────────────────────────
            with gr.Tab("Domain Synthesis"):
                gr.Markdown("# Specialized Domain Synthesis")
                gr.Markdown(
                    "Generate high-quality domain-focused instruction-response pairs using "
                    "Magpie-style self-synthesis. Each domain uses curated system prompts and "
                    "domain-appropriate quality filters (e.g. coding requires code blocks, "
                    "math/tax require numbers). Output is an HF dataset ready for distillation."
                )

                # ── Shared domain controls ────────────────────────────────────
                with gr.Row():
                    dom_teacher = gr.Dropdown(
                        choices=teachers,
                        value="Qwen/Qwen2-1.5B-Instruct",
                        label="Teacher model",
                        allow_custom_value=True,
                        scale=4,
                        info="Larger teacher = richer domain knowledge. Qwen2-1.5B works well for all domains.",
                    )
                    dom_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
                with gr.Row():
                    dom_backend = gr.Radio(
                        ["auto", "mlx", "mps"], value="auto",
                        label="Backend",
                        info="auto picks MLX on Apple Silicon (2-4× faster than MPS)",
                    )
                    dom_offline = gr.Checkbox(value=False, label="Offline (use cached model)")
                    dom_filter  = gr.Checkbox(value=True,  label="Deep filter output (dedup + quality)")
                    dom_batch   = gr.Slider(1, 64, value=32, step=1, label="Batch size",
                                            info="For MLX: loop chunk size only (sequential gen). For MPS: reduce to 4-8 if OOM.")

                dom_status = gr.Textbox(value="idle", label="Status", interactive=False)
                domain_progress = gr.HTML(value="")

                gr.Markdown("---")

                # ── Medical ───────────────────────────────────────────────────
                with gr.Accordion("🏥  Medical", open=False):
                    gr.Markdown(
                        "Generates clinical education Q&A: symptoms, diagnosis reasoning, pharmacology, "
                        "pathophysiology, anatomy, public health, and medical ethics. "
                        "**Filter:** responses ≥40 words, distinct-2 ≥0.30. "
                        "Output tagged `domain=medical` for easy mixing with other datasets."
                    )
                    with gr.Row():
                        med_n      = gr.Slider(500, 20000, value=3000, step=500,
                                               label="Pairs to generate (before filter)")
                        med_target = gr.Slider(200, 10000, value=1500, step=200,
                                               label="Target keep after filter")
                        med_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "medical"),
                            label="Output directory", scale=3,
                        )
                    med_btn = gr.Button("Generate Medical Dataset", variant="primary")

                # ── Math ──────────────────────────────────────────────────────
                with gr.Accordion("📐  Mathematics", open=False):
                    gr.Markdown(
                        "Generates math problem-solution pairs: calculus, linear algebra, statistics, "
                        "discrete math, number theory, and competition math. "
                        "**Filter:** response must contain at least one number/equation, distinct-2 ≥0.25."
                    )
                    with gr.Row():
                        math_n      = gr.Slider(500, 20000, value=3000, step=500,
                                                label="Pairs to generate (before filter)")
                        math_target = gr.Slider(200, 10000, value=1500, step=200,
                                                label="Target keep after filter")
                        math_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "math"),
                            label="Output directory", scale=3,
                        )
                    math_btn = gr.Button("Generate Math Dataset", variant="primary")

                # ── Legal ─────────────────────────────────────────────────────
                with gr.Accordion("⚖️  Legal / Law", open=False):
                    gr.Markdown(
                        "Generates legal education Q&A: contracts, torts, constitutional law, criminal law, "
                        "property, corporate, IP, civil procedure, and legal reasoning. "
                        "**Filter:** responses ≥50 words (legal explanations are inherently longer), distinct-2 ≥0.30. "
                        "All outputs include an educational disclaimer."
                    )
                    with gr.Row():
                        legal_n      = gr.Slider(500, 20000, value=3000, step=500,
                                                 label="Pairs to generate (before filter)")
                        legal_target = gr.Slider(200, 10000, value=1500, step=200,
                                                 label="Target keep after filter")
                        legal_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "legal"),
                            label="Output directory", scale=3,
                        )
                    legal_btn = gr.Button("Generate Legal Dataset", variant="primary")

                # ── Tax ───────────────────────────────────────────────────────
                with gr.Accordion("🧾  Tax / IRS", open=False):
                    gr.Markdown(
                        "Generates U.S. tax education Q&A: individual filing, business taxes, capital gains, "
                        "retirement accounts, self-employment, SALT, estate/gift tax, and audits. "
                        "**Filter:** responses ≥40 words, must contain numbers (tax rules involve figures), "
                        "distinct-2 ≥0.28."
                    )
                    with gr.Row():
                        tax_n      = gr.Slider(500, 20000, value=3000, step=500,
                                               label="Pairs to generate (before filter)")
                        tax_target = gr.Slider(200, 10000, value=1500, step=200,
                                               label="Target keep after filter")
                        tax_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "tax"),
                            label="Output directory", scale=3,
                        )
                    tax_btn = gr.Button("Generate Tax Dataset", variant="primary")

                # ── Coding ────────────────────────────────────────────────────
                with gr.Accordion("💻  Coding / Programming", open=False):
                    gr.Markdown(
                        "Generates programming Q&A with working code: Python, JavaScript, C/C++, Rust, Go, "
                        "SQL, shell scripting, algorithms, ML engineering, DevOps, and API design. "
                        "**Filter:** response must contain at least one code fence (``` block or inline `code`), "
                        "distinct-2 ≥0.20 (code is naturally repetitive)."
                    )
                    with gr.Row():
                        code_n      = gr.Slider(500, 20000, value=4000, step=500,
                                                label="Pairs to generate (before filter)",
                                                info="Generate more — code filter is stricter (~30-40% pass rate)")
                        code_target = gr.Slider(200, 10000, value=2000, step=200,
                                                label="Target keep after filter")
                        code_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "coding"),
                            label="Output directory", scale=3,
                        )
                    code_btn = gr.Button("Generate Coding Dataset", variant="primary")

                # ── Finance ───────────────────────────────────────────────────
                with gr.Accordion("💰  Finance / Investing", open=False):
                    gr.Markdown(
                        "Generates finance education Q&A: personal finance, investing, corporate finance, "
                        "derivatives, fixed income, macroeconomics, risk management, and financial modeling. "
                        "**Filter:** responses ≥30 words, must contain numbers (finance involves figures), "
                        "distinct-2 ≥0.28."
                    )
                    with gr.Row():
                        fin_n      = gr.Slider(500, 20000, value=3000, step=500,
                                               label="Pairs to generate (before filter)")
                        fin_target = gr.Slider(200, 10000, value=1500, step=200,
                                               label="Target keep after filter")
                        fin_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "finance"),
                            label="Output directory", scale=3,
                        )
                    fin_btn = gr.Button("Generate Finance Dataset", variant="primary")

                # ── Custom Domain Builder ─────────────────────────────────
                with gr.Accordion("✏️  Custom Domain Builder", open=False):
                    gr.Markdown(
                        "Define a new domain with custom system prompts and quality filters. "
                        "**Save** writes it to `configs/domain_prompts.json` so it persists across sessions "
                        "and is immediately available to all domain synthesis tools."
                    )
                    with gr.Row():
                        custom_id    = gr.Textbox(label="Domain ID",
                                                   placeholder="e.g. chemistry  (lowercase, no spaces)",
                                                   scale=2)
                        custom_label = gr.Textbox(label="Display label",
                                                   placeholder="e.g. Chemistry / Science",
                                                   scale=3)
                    custom_desc = gr.Textbox(
                        label="Description (shown in UI, optional)",
                        placeholder="Chemistry Q&A: reactions, periodic table, organic/inorganic, lab safety.",
                        lines=2,
                    )
                    custom_prompts = gr.Textbox(
                        label="System prompts — one per line (at least 1 required)",
                        placeholder=(
                            "You are a chemistry educator explaining reactions clearly.\n"
                            "You are an organic chemistry tutor covering nomenclature and mechanisms.\n"
                            "You are a physical chemistry instructor discussing thermodynamics and kinetics."
                        ),
                        lines=6,
                    )
                    gr.Markdown("**Quality filter settings**")
                    with gr.Row():
                        custom_min_words  = gr.Slider(5, 100, value=20, step=5,
                                                      label="Min response words")
                        custom_max_words  = gr.Slider(100, 2000, value=600, step=50,
                                                      label="Max response words")
                        custom_min_d2     = gr.Slider(0.10, 0.60, value=0.30, step=0.05,
                                                      label="Min distinct-2")
                    with gr.Row():
                        custom_req_code   = gr.Checkbox(value=False,
                                                        label="Require code block (``` or inline `code`)")
                        custom_req_nums   = gr.Checkbox(value=False,
                                                        label="Require numbers in response")
                    custom_save_status = gr.Textbox(value="", label="Save status", interactive=False)
                    with gr.Row():
                        custom_save_btn = gr.Button("Save Domain", variant="secondary", scale=2)

                    gr.Markdown("**Generate with custom domain** (saves first if domain is new)")
                    with gr.Row():
                        custom_n      = gr.Slider(500, 20000, value=3000, step=500,
                                                  label="Pairs to generate")
                        custom_target = gr.Slider(200, 10000, value=1500, step=200,
                                                  label="Target keep after filter")
                        custom_outdir = gr.Textbox(
                            value=str(PROJECT_DIR / "domain_data" / "custom"),
                            label="Output directory", scale=3,
                        )
                    custom_launch_btn = gr.Button("Save & Generate", variant="primary")

                gr.Markdown("---")
                gr.Markdown(
                    "### After generation\n"
                    "Point the **Dataset** field in **Configure & Launch** at the output directory "
                    "(e.g. `domain_data/coding/hf_dataset`) and launch distillation normally. "
                    "You can also merge multiple domain datasets using `datasets.concatenate_datasets()` "
                    "before distillation for a multi-domain specialist model."
                )

            # ── Tab 4: Eval ──────────────────────────────────────────────────
            with gr.Tab("Eval"):
                gr.Markdown(
                    "Post-training evaluation. All evals write results to the model's output dir. "
                    "Leave Checkpoint blank to eval the final merged model."
                )

                # Shared eval inputs
                with gr.Row():
                    eval_output_dir = gr.Dropdown(
                        choices=out_dirs,
                        value=default_out,
                        label="Model output dir",
                        allow_custom_value=True,
                        scale=3,
                    )
                    eval_refresh_btn = gr.Button("Refresh", scale=1, size="sm")
                with gr.Row():
                    eval_checkpoint = gr.Textbox(
                        value="",
                        label="Checkpoint (optional, e.g. distilled-minillm/checkpoint-80)",
                        placeholder="Leave blank to eval final model",
                        scale=3,
                    )
                with gr.Row():
                    eval_student = gr.Dropdown(
                        choices=students,
                        value=default_student,
                        label="Base model (fallback tokenizer)",
                        allow_custom_value=True,
                        scale=3,
                    )
                    eval_dataset = gr.Dropdown(
                        choices=datasets,
                        value=default_dataset,
                        label="Dataset",
                        allow_custom_value=True,
                        scale=3,
                    )
                with gr.Row():
                    eval_offline = gr.Checkbox(value=False, label="Offline")
                eval_progress = gr.HTML(value="")

                gr.Markdown("---")

                # Perplexity eval
                gr.Markdown("### Perplexity Eval  (`run_eval.py`)")
                gr.Markdown(
                    "Computes cross-entropy loss and perplexity on the validation split. "
                    "Appends `eval_loss` and `perplexity` to `metrics.jsonl`."
                )
                with gr.Row():
                    ppl_max_val = gr.Slider(10, 1000, value=200, step=10,
                                            label="Max validation samples")
                    ppl_batch   = gr.Slider(1, 32, value=8, step=1,
                                            label="Batch size")
                with gr.Row():
                    ppl_compare_teacher = gr.Checkbox(value=False,
                                                      label="Also eval teacher (log perplexity gap)")
                    ppl_teacher = gr.Dropdown(
                        choices=teachers,
                        value=default_teacher,
                        label="Teacher model (for comparison)",
                        allow_custom_value=True,
                        scale=3,
                    )
                with gr.Row():
                    ppl_launch_btn = gr.Button("Run Perplexity Eval", variant="primary", scale=3)
                    ppl_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                ppl_status = gr.Textbox(value="idle", label="Status", interactive=False)

                gr.Markdown("---")

                # Quality eval
                gr.Markdown("### Quality Eval  (`eval_quality.py`)")
                gr.Markdown(
                    "Samples prompts, generates student responses, computes distinct-1/2 diversity "
                    "and max-repetition. Optionally runs LLM-as-judge scoring. "
                    "Output: `quality_metrics.json`."
                )
                with gr.Row():
                    qual_n_samples  = gr.Slider(10, 500, value=50, step=10,
                                                label="Samples to generate")
                    qual_judge      = gr.Checkbox(value=False, label="LLM-as-judge scoring")
                    qual_judge_teacher = gr.Dropdown(
                        choices=teachers,
                        value=default_teacher,
                        label="Judge model",
                        allow_custom_value=True,
                        scale=3,
                    )
                with gr.Row():
                    qual_launch_btn = gr.Button("Run Quality Eval", variant="primary", scale=3)
                    qual_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                qual_status = gr.Textbox(value="idle", label="Status", interactive=False)

                gr.Markdown("---")

                # WikiText-2 benchmark
                gr.Markdown("### WikiText-2 Benchmark  (`run_benchmarks.py`)")
                gr.Markdown(
                    "Evaluates perplexity on WikiText-2-raw-v1 test split. "
                    "Optionally compares against a previous baseline for regression detection. "
                    "Saves `benchmark_results.json` and appends to `metrics.jsonl`."
                )
                with gr.Row():
                    bench_n_seq    = gr.Slider(50, 2000, value=500, step=50,
                                               label="Sequences to evaluate")
                    bench_batch    = gr.Slider(1, 32, value=8, step=1,
                                               label="Batch size")
                    bench_threshold = gr.Slider(5.0, 50.0, value=15.0, step=1.0,
                                                label="Max regression % vs baseline")
                with gr.Row():
                    bench_baseline = gr.Textbox(
                        value="",
                        label="Baseline dir (optional, for regression detection)",
                        placeholder="e.g. ./previous-run",
                        scale=4,
                    )
                with gr.Row():
                    bench_launch_btn = gr.Button("Run WikiText-2 Benchmark", variant="primary", scale=3)
                    bench_stop_btn   = gr.Button("Stop", variant="stop", scale=1)
                bench_status = gr.Textbox(value="idle", label="Status", interactive=False)

            # ── Tab 4b: Expert Pipeline ──────────────────────────────────────
            with gr.Tab("Expert Pipeline"):
                gr.Markdown(
                    "**Domain-expert distillation pipeline** — loads any HF dataset, "
                    "remaps columns, generates Chain-of-Thought rationales via a GGUF "
                    "teacher (Metal-accelerated), then launches distillation.\n\n"
                    "Run steps in order: **1 → 2 → 3 → 4**."
                )
                ep_progress = gr.HTML(value="")

                # ── Step 1: Dataset & Column Mapping ─────────────────────────
                _EP_HF_DATASETS = [
                    # Legal
                    "nguha/legalbench",
                    "nelson-liu/legalbench",
                    # Tax
                    "Atome-LLM/Tax-Policy-Analysis",
                    # Medical
                    "medalpaca/medical_meadow_medical_flashcards",
                    "medalpaca/medical_meadow_wikidoc",
                    "pubmed_qa",
                    # Finance
                    "gbharti/finance-alpaca",
                    "FinGPT/fingpt-sentiment-train",
                    # Coding
                    "iamtarun/python_code_instructions_18k_alpaca",
                    "sahil2801/CodeAlpaca-20k",
                    # General
                    "yahma/alpaca-cleaned",
                    "tatsu-lab/alpaca",
                    "HuggingFaceH4/ultrachat_200k",
                ]
                _EP_LOCAL_CANDIDATES = [
                    "./domain_data/expert_remapped",
                    "./domain_data/tax",
                    "./domain_data/legal",
                    "./domain_data/medical",
                    "./domain_data/coding",
                ]

                def _ep_dataset_choices():
                    local = [p for p in _EP_LOCAL_CANDIDATES
                             if (PROJECT_DIR / p.lstrip("./")).exists()]
                    return _EP_HF_DATASETS + local

                with gr.Accordion("Step 1 — Dataset & Column Mapping", open=True):
                    with gr.Row():
                        ep_dataset = gr.Dropdown(
                            choices=_ep_dataset_choices(),
                            value="",
                            label="HF Dataset ID or local path",
                            allow_custom_value=True,
                            scale=4,
                        )
                        ep_dataset_refresh = gr.Button("⟳", scale=0, size="sm", min_width=40)
                        ep_inspect_btn = gr.Button("Inspect", scale=1, size="sm")
                    ep_inspect_status = gr.Textbox(
                        value="", label="Dataset info", interactive=False, lines=2,
                    )
                    with gr.Row():
                        ep_instruction_col = gr.Dropdown(
                            choices=[], allow_custom_value=True,
                            label="Instruction column", scale=1,
                        )
                        ep_output_col = gr.Dropdown(
                            choices=[], allow_custom_value=True,
                            label="Output column", scale=1,
                        )
                        ep_input_col = gr.Dropdown(
                            choices=["(none)"], value="(none)", allow_custom_value=True,
                            label="Input / context column (optional)", scale=1,
                        )
                    with gr.Row():
                        ep_max_samples_remap = gr.Slider(
                            100, 50000, value=5000, step=100,
                            label="Max samples to load",
                        )
                        ep_remap_output = gr.Textbox(
                            value="./domain_data/expert_remapped",
                            label="Save remapped dataset to",
                        )
                    ep_remap_btn = gr.Button("Remap & Save Dataset", variant="primary")

                # ── Step 2: GGUF Teacher ──────────────────────────────────────
                with gr.Accordion("Step 2 — GGUF Teacher", open=False):
                    _gguf_choices = sorted(
                        [str(p) for p in Path("/Users/Shared/llama/models").glob("*.gguf")]
                    ) if Path("/Users/Shared/llama/models").exists() else []
                    with gr.Row():
                        ep_teacher = gr.Dropdown(
                            choices=_gguf_choices,
                            value=_gguf_choices[-1] if _gguf_choices else "",
                            allow_custom_value=True,
                            label="GGUF teacher model",
                            scale=3,
                        )
                        ep_teacher_refresh = gr.Button("Refresh", scale=1, size="sm")
                    with gr.Row():
                        ep_ctx_size   = gr.Slider(1024, 32768, value=8192, step=1024,
                                                  label="Context size (tokens)")
                        ep_n_parallel = gr.Slider(1, 8, value=4, step=1,
                                                  label="Parallel server slots")
                        ep_cot_temp   = gr.Slider(0.0, 1.5, value=0.3, step=0.05,
                                                  label="Teacher temperature")
                        ep_max_tokens = gr.Slider(256, 4096, value=1024, step=128,
                                                  label="Max tokens per response")

                # ── Step 3: CoT Rationale Generation ─────────────────────────
                with gr.Accordion("Step 3 — CoT Rationale Generation", open=False):
                    with gr.Row():
                        ep_domain = gr.Dropdown(
                            choices=["tax", "legal", "medical", "finance", "coding", "general"],
                            value="general",
                            label="Domain",
                            scale=1,
                        )
                        ep_n_cot = gr.Slider(50, 10000, value=1000, step=50,
                                             label="Samples to generate", scale=2)
                    ep_system_prompt = gr.Textbox(
                        value="",
                        label="System prompt (leave blank to use domain default)",
                        lines=6,
                        placeholder="Leave blank to auto-fill from domain selection…",
                    )
                    ep_cot_output = gr.Textbox(
                        value="./domain_data/expert_cot",
                        label="CoT output directory",
                    )
                    ep_batch_size_cot = gr.Slider(4, 32, value=16, step=4,
                                                  label="Batch size")
                    with gr.Row():
                        ep_cot_btn  = gr.Button("Generate CoT Rationales", variant="primary", scale=3)
                        ep_cot_stop = gr.Button("Stop", variant="stop", scale=1)

                # ── Step 4: Distillation ──────────────────────────────────────
                with gr.Accordion("Step 4 — Distillation", open=False):
                    with gr.Row():
                        ep_student = gr.Dropdown(
                            choices=KNOWN_STUDENTS,
                            value="Qwen/Qwen2-0.5B-Instruct",
                            allow_custom_value=True,
                            label="Student model",
                            scale=2,
                        )
                        ep_distill_backend = gr.Dropdown(
                            choices=["mlx", "pytorch", "unsloth"],
                            value="mlx",
                            label="Backend",
                            scale=1,
                        )
                    with gr.Row():
                        ep_distill_dataset = gr.Textbox(
                            value="",
                            label="Dataset path (defaults to CoT output dir above)",
                            placeholder="Leave blank to use CoT output dir",
                        )
                        ep_distill_output = gr.Textbox(
                            value="./runs/expert-distilled",
                            label="Output directory",
                        )
                    with gr.Row():
                        ep_epochs   = gr.Slider(1, 5, value=3, step=1, label="Epochs")
                        ep_lora_r   = gr.Slider(8, 64, value=32, step=8, label="LoRA rank")
                        ep_max_samp = gr.Slider(500, 10000, value=3000, step=500,
                                                label="Max training samples")
                    with gr.Row():
                        ep_open_chk    = gr.Checkbox(value=True, label="Open models (no HF login)")
                        ep_offline_chk = gr.Checkbox(value=False, label="Offline mode")
                    with gr.Row():
                        ep_distill_btn  = gr.Button("Launch Distillation", variant="primary", scale=3)
                        ep_distill_stop = gr.Button("Stop", variant="stop", scale=1)

                ep_status = gr.Textbox(value="idle", label="Status", interactive=False)

                # ── Embedded live output ──────────────────────────────────────
                with gr.Row():
                    ep_loss_plot = gr.LinePlot(
                        value=pd.DataFrame({"step": [0], "loss": [0.0]}),
                        x="step", y="loss",
                        title="Training Loss",
                        x_title="Step", y_title="Loss",
                        height=220, min_width=300,
                    )
                    ep_grad_plot = gr.LinePlot(
                        value=pd.DataFrame({"step": [0], "grad_norm": [0.0]}),
                        x="step", y="grad_norm",
                        title="Gradient Norm",
                        x_title="Step", y_title="Grad Norm",
                        height=220, min_width=300,
                    )
                ep_log_box = gr.Textbox(
                    value="",
                    label="Live output",
                    lines=20,
                    max_lines=20,
                    interactive=False,
                    autoscroll=True,
                )

            # ── Tab 4: Live Logs ─────────────────────────────────────────────
            with gr.Tab("Live Logs"):
                training_progress = gr.HTML(value="")
                with gr.Row():
                    loss_plot = gr.LinePlot(
                        value=pd.DataFrame({"step": [0], "loss": [0.0]}),
                        x="step",
                        y="loss",
                        title="Training Loss",
                        x_title="Step",
                        y_title="Loss",
                        height=220,
                        min_width=300,
                    )
                    grad_plot = gr.LinePlot(
                        value=pd.DataFrame({"step": [0], "grad_norm": [0.0]}),
                        x="step",
                        y="grad_norm",
                        title="Gradient Norm",
                        x_title="Step",
                        y_title="Grad Norm",
                        height=220,
                        min_width=300,
                    )
                log_box = gr.Textbox(
                    value="",
                    label="Output",
                    lines=25,
                    max_lines=25,
                    interactive=False,
                    autoscroll=True,
                )
                log_status = gr.Textbox(value="idle", label="Run status", interactive=False)
                clear_btn = gr.Button("Clear log")

            # ── Tab 5: Help ──────────────────────────────────────────────────
            with gr.Tab("Help"):
                gr.Markdown("# Distillation Launcher — Operation Guide")
                gr.Markdown(
                    "**Seven tabs:** Configure & Launch · Data Prep · Domain Synthesis · "
                    "Eval · **Expert Pipeline** · Live Logs · Help.  "
                    "A progress bar appears on every tab — no need to switch to Live Logs to monitor a run. "
                    "Click a section below to expand it."
                )

                # ── Golden pipeline ──────────────────────────────────────────
                with gr.Accordion("🏆  Golden Pipeline — Recommended End-to-End Sequence", open=True):
                    gr.Markdown("""
```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        GOLDEN PIPELINE  (best quality)                         ║
╠════╦═══════════════════════╦══════════════════════════════════════════╦═════════╣
║ 1  ║ Data Prep  (optional) ║ Filter dataset or run Magpie synthesis   ║  5–30 m ║
║ 2  ║ Domain Synth (opt.)   ║ Generate domain pairs: code/math/legal/… ║ 30–90 m ║
║ 3  ║ Configure & Launch    ║ Stage: SFT · 1 epoch → sft_checkpoint    ║  ~30 m  ║
║ 4  ║ Configure & Launch    ║ Stage: MiniLLM · Student=sft_checkpoint  ║   ~2 h  ║
║ 5  ║ Eval                  ║ Perplexity → Quality → WikiText-2        ║  ~15 m  ║
║ 6  ║ Auto (agent)          ║ ./run_golden.sh → full pipeline headless ║  ~3 h   ║
╚════╩═══════════════════════╩══════════════════════════════════════════╩═════════╝
```

**Steps 1–2 are optional** but significantly improve output quality. Step 3 (SFT) is strongly
recommended before MiniLLM — it gives the student a warm start so GRPO rewards don't
open negative and stay there.

### One-command golden run (terminal)
```bash
./run_golden.sh                                    # foreground
./run_golden.sh > runs/golden_pipeline.log 2>&1 &  # background with log
```
Config is at `configs/golden_pipeline.json` — edit it to change dataset, epochs, LoRA rank, etc.

---

### Quickstart (no HF login · ~2 hr)
1. **Configure & Launch** → Stage: **MiniLLM** · Backend: **MLX** · ☑ **Use open Qwen2 models** → **Launch**
2. Watch progress bar on same tab. Switch to **Live Logs** for full stream + loss/grad charts.
3. **Eval** → Run Perplexity Eval once training completes.

---

### When to add SFT warmup
Run SFT first when any of these appear in Live Logs during MiniLLM:
- `reward` stuck at −1.0 for more than 50 steps
- `frac_reward_zero_std` > 0.5 after step 30
- You are using a new domain dataset the student hasn't seen before

After SFT finishes, set the MiniLLM **Student** field to `distilled-minillm/sft_checkpoint`.

---

### Decision guide
| Goal | Recommended path |
|------|-----------------|
| Fastest first result | Quickstart (skip steps 1–3) |
| Best general quality | `./run_golden.sh` |
| Domain specialist (tax / legal / medical / finance / coding) | **Expert Pipeline** tab |
| Fastest training on M-series Mac | Backend: **MLX**, batch=2, grad_acc=8 |
| Largest pair that fits 36 GB unified memory | Teacher ≤ 3B + student ≤ 1B |
| Resume an interrupted run | MLX: tick **Resume**; PyTorch: relaunch from latest checkpoint |

---

### Outputs written to disk
| File | Contents |
|------|----------|
| `output_dir/*.safetensors` | Merged final model weights (PyTorch backend) |
| `output_dir/mlx_student_weights.npz` | Trained weights (MLX backend, LoRA + base fused) |
| `output_dir/metrics.jsonl` | Per-step loss, reward, eval metrics (JSON lines) |
| `output_dir/train_log.jsonl` | Structured JSON log from agent runs (step, loss, epoch, ts) |
| `output_dir/sft_labels.jsonl` | Cached teacher labels (SFT reuses on re-run) |
| `output_dir/quality_metrics.json` | Generation diversity + judge scores |
| `output_dir/benchmark_results.json` | WikiText-2 perplexity result |
| `runs/ep_<stage>_<timestamp>.log` | Plain-text log from Expert Pipeline runs |
| `runs/ep_<stage>_<timestamp>.jsonl` | Structured JSON log from Expert Pipeline runs |
| `/Users/Shared/llama/models/*.gguf` | GGUF exports for llama.cpp / Ollama |
""")

                # ── Tab 1 help ───────────────────────────────────────────────
                with gr.Accordion("Configure & Launch — Parameter Reference", open=False):
                    gr.Markdown("""
### Stage
| Stage | Script | When to use |
|-------|--------|-------------|
| **SFT** | `distill_sft.py` | First-pass warmup. Teacher generates response labels; student trains on them with standard cross-entropy. Use before MiniLLM for best results. |
| **MiniLLM** | `distill_minillm.py` / `distill_mlx.py` | Main distillation stage. Student generates completions; GRPO advantage signal pushes it toward teacher distribution (reverse-KL). |

### Backend
| Backend | Speed | Memory | When to use |
|---------|-------|--------|-------------|
| **PyTorch / MPS** | Baseline | ~8–12 GB unified | Stable, supports all features. Use when debugging or running SFT. |
| **MLX** | 2–5× faster | ~4–8 GB unified | Apple-native lazy evaluation. Best for long MiniLLM runs. Uses lower batch defaults (2/4/8). |

> **Switching backends** auto-updates Batch size / Grad acc / LoRA rank to the recommended defaults for that backend. You can still adjust them manually.

### Models
- **Use open Qwen2 models** — Ticks `--open` flag. Forces teacher = `Qwen/Qwen2-1.5B-Instruct`, student = `Qwen/Qwen2-0.5B-Instruct`. No HuggingFace account or license needed.
- **Teacher** — Larger model that provides soft targets or hard labels. Must fit in unified memory alongside the student. Rule of thumb: teacher ≤ 3× student parameters.
- **Student** — Smaller model being trained. For MiniLLM, point this at your **SFT checkpoint** (`distilled-minillm/sft_checkpoint`) for best results.
- Click **Refresh** after a model finishes downloading to see it in the dropdown.

### Training (common to all stages)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Epochs** | 2 | SFT: 1 is usually enough. MiniLLM: 2–3. More can overfit on small datasets. |
| **Max samples** | 2000 | Samples drawn from the dataset. 2000 trains in ~1–2 hours. Use 500 for a smoke test. |
| **Batch size** | 8 (PyTorch), 2 (MLX) | Physical samples per device step. Increase until Activity Monitor shows ~80% GPU pressure. |
| **Gradient accumulation** | 8 | Multiply by batch for effective batch (8×8=64 PyTorch, 2×8=16 MLX). Larger effective batch = smoother gradients. |
| **LoRA rank** | 16 | Higher = more trainable params = slower but higher capacity. 16 is a good balance for both backends. 64 for SFT. |

### SFT options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Learning rate** | 2e-4 | Standard SFT rate. Reduce to 1e-4 if loss oscillates. |
| **Teacher max new tokens** | 128 | Tokens the teacher generates per prompt for the label cache. 128 covers most alpaca responses. |
| **Max sequence length** | 384 | Total tokens (prompt + response) kept for training. Sequences longer than this are truncated. |

### MiniLLM options (PyTorch)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Softens the teacher distribution. Higher (1.5–2.0) = smoother targets, can stabilize early training. |
| **Learning rate** | 2e-5 | Intentionally 10× lower than SFT. Increase to 5e-5 only if rewards plateau after 100 steps. |
| **Generations per prompt** | 4 | GRPO samples per prompt. **At least 4** to get reward variance within each group (fewer → frac_reward_zero_std stays high → no gradient). 8 gives richer signal but 2× slower. |
| **Max completion length** | 256 | Hard cutoff for student generations. **Critical:** too small → model hits limit before EOS → 80%+ clipped_ratio → reward collapses. 256 tokens ≈ 800 characters is the calibrated default. |
| **Eval every N steps** | 20 | Lower = more detail in metrics.jsonl, higher = faster overall run. 20 is a good balance. |

### MLX options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Same as PyTorch. |
| **Learning rate** | 2e-4 | MLX uses forward-KL + CE, not GRPO, so it tolerates a higher LR. |
| **CE alpha** | 0.2 | Weight of cross-entropy (hard label) loss. 0 = pure KD, 1 = pure CE. 0.2 stabilises early training without losing KD signal (increased from 0.1 for better convergence). |
| **Top-K teacher logits** | 50 | Keeps only top-50 teacher token probabilities. Captures >99% of probability mass while reducing logit memory from ~300 GB to ~300 MB per dataset. Teacher is freed from memory immediately after precompute. |
| **Export quantization bits** | 4 | Bits for the MLX quantized export. 4-bit is standard for llama.cpp-compatible GGUF. |
| **Resume** | off | Continue from the last epoch checkpoint in output_dir if a previous MLX run was interrupted. |

### Watchdog
Creates a `pause.flag` file callback. While training is running, you can pause it by creating `output_dir/pause.flag` from the terminal (`touch distilled-minillm/pause.flag`) and resume by deleting it. Useful for thermal management without losing progress.

### Stop button
Sends **SIGKILL** (immediate termination). The last saved checkpoint is preserved. Use it any time — the run can be resumed via **Resume** (MLX) or by relaunching from the latest checkpoint (PyTorch).
""")

                # ── Tab 2 help ───────────────────────────────────────────────
                with gr.Accordion("Data Prep & Domain Synthesis — When and How to Use Each Tool", open=False):
                    gr.Markdown("""
### When to use Data Prep
Out-of-the-box datasets (Alpaca, Guanaco) work fine for quickstart runs. Use Data Prep when:
- You want domain-specific data not in the standard datasets
- Existing dataset quality is low (short responses, repetition, refusals)
- You want to maximize distillation quality with teacher-generated data

### Domain Synthesis (separate tab)
For domain-specialist models (coding, math, medical, legal, finance, tax) use the
**Domain Synthesis** tab instead of — or in addition to — the tools below. It runs
Magpie synthesis with curated system prompts and domain-specific quality filters.
Output: `domain_data/<domain>/hf_dataset/` — point **Dataset** here in Configure & Launch.

### Magpie Synthesis
**What it does:** Loads the teacher model and repeatedly samples from it, conditioning on just the chat-template user-turn prefix. The model "auto-completes" realistic user instructions, then generates responses. No seed dataset needed.

**Output:** `output_dir/hf_dataset/` — an HF dataset you can point directly at the **Dataset** field in Configure & Launch.

**When to use:** Best quality synthetic data. Requires the teacher to be available locally (large download first time). Use when you want data tailored to the teacher's style.

| Parameter | Guidance |
|-----------|----------|
| **Pairs to generate** | Generate 2–3× your target keep count to account for filtering attrition. |
| **Filter output** | Always keep this on — removes duplicates and low-quality pairs. |
| **Target keep** | Final dataset size after filtering. 2000–5000 pairs is enough for a 2-epoch run. |

### Self-Instruct Synthesis
**What it does:** Shows the teacher `seed_examples` instructions from a base dataset, then asks it to generate a new, diverse instruction. The teacher also generates the response. Filtered by perplexity bounds and distinct-2 score.

**Output:** `output_dir/synthetic_data/` — an HF dataset directory.

**When to use:** More diverse than Magpie (prompts are anchored to varied seed examples). Slower per-pair than Magpie. Good for covering topic diversity.

| Parameter | Guidance |
|-----------|----------|
| **Temperature** | 0.9 is a good balance — high enough for diversity, low enough for coherence. Don't go above 1.2. |
| **Seed examples** | 5 is standard. Higher seeds = more context for the teacher but diminishing returns. |

### Dataset Filter
**What it does:** Takes any alpaca-format dataset (HF hub ID or local path) and removes:
- Responses shorter than `min_response_words`
- Low-coherence responses (distinct-2 bigram diversity < `min_distinct2`)
- Near-duplicate pairs (Jaccard similarity above threshold)
- Common refusal patterns ("I cannot...", "As an AI...")

**Output:** `output_dir/` — an HF dataset directory. Use this path directly in Configure & Launch.

**When to use:** Always run this before training on any public dataset. `yahma/alpaca-cleaned` is already filtered but running it again with stricter thresholds is harmless.

| Parameter | Guidance |
|-----------|----------|
| **Target top-N** | How many pairs to keep after scoring. Keeps the highest-quality pairs. |
| **Min distinct-2** | 0.35 is standard. Increase to 0.45 to keep only highly diverse responses. |
| **Min response words** | 20 words eliminates one-liners. Increase to 30–40 for richer training signal. |

### Combining tools (recommended data workflow)
```
General quality:
  1. Filter        yahma/alpaca-cleaned  →  filtered_data/              (2–3 min)
  2. Self-Instruct Generate 3000 pairs   →  synthetic_data/             (30–60 min)
  3. Configure     Dataset = filtered_data/

Domain specialist:
  1. Domain Synth  coding / math / medical …  →  domain_data/<x>/      (30–90 min)
  2. (Optional) mix with filtered general data using datasets.concatenate_datasets()
  3. Configure     Dataset = domain_data/<x>/hf_dataset/
```
""")

                # ── Tab 3 help ───────────────────────────────────────────────
                with gr.Accordion("Eval — Interpreting Results", open=False):
                    gr.Markdown("""
### Shared inputs
- **Model output dir** — The same directory you used as Output directory during training. This is where metrics.jsonl and result files are written.
- **Checkpoint** — Leave blank to eval the final merged model. Enter a path like `distilled-minillm/checkpoint-80` to eval a mid-training checkpoint.
- **Base model** — Only used as a fallback if the checkpoint directory has no tokenizer. Set to the student model you trained.

---

### Perplexity Eval (`run_eval.py`)
Loads the student model and computes **cross-entropy loss** on a held-out validation split of your training dataset.

**Output appended to:** `metrics.jsonl` as `{"step": N, "eval_loss": X, "perplexity": Y}`

**How to interpret:**
| Perplexity | Interpretation |
|------------|----------------|
| < 5 | Excellent — model fits the data distribution well |
| 5–15 | Good — typical range for a well-distilled small model |
| 15–30 | Fair — may need more epochs or a better dataset |
| > 30 | Poor — check for training bugs, wrong checkpoint, or data mismatch |

- **Compare teacher** — Also runs the teacher through the same eval and logs the gap. A good distilled student should get within 1.5–2× the teacher's perplexity.
- Run this after every training run as a quick sanity check.

---

### Quality Eval (`eval_quality.py`)
Generates responses from the student on `n_samples` prompts and measures **generation diversity**.

**Output:** `quality_metrics.json` in the model output dir.

**Metrics explained:**
| Metric | Good range | What it means |
|--------|-----------|----------------|
| `distinct_1` | > 0.15 | Fraction of unique unigrams. Low = repetitive vocabulary. |
| `distinct_2` | > 0.40 | Fraction of unique bigrams. Low = repetitive phrasing. |
| `max_repetition` | < 0.30 | Highest n-gram repetition rate in any single response. High = mode collapse. |
| `avg_length_tokens` | 50–200 | Average response length. Very short may indicate mode collapse. |
| `judge_score_mean` | 6–8 | LLM-as-judge score (1–10). Enable **LLM-as-judge** for this. |

- **LLM-as-judge** — Uses the teacher model to score each response 1–10. Adds ~5–10 minutes. Worth running once before deploying.
- Run this after perplexity eval confirms a reasonable loss.

---

### WikiText-2 Benchmark (`run_benchmarks.py`)
Evaluates the student on the **WikiText-2-raw-v1** test split — a standard open-domain NLP benchmark independent of your training data.

**Output:** `benchmark_results.json` + `metrics.jsonl` entry `{"wikitext2_perplexity": X}`

**Reference numbers (lower is better):**
| Model | WikiText-2 PPL |
|-------|---------------|
| Qwen2-0.5B-Instruct (base, no distillation) | ~18–22 |
| Well-distilled student (from 1.5B teacher) | ~14–18 |
| Qwen2-1.5B-Instruct (teacher) | ~10–13 |

- **Baseline dir** — Point at a previous run's output dir to get a regression comparison. If the new model is more than `threshold`% worse than the baseline, the benchmark prints a warning.
- Run this as a final check before exporting to GGUF.

---

### Recommended eval sequence
```
1. Perplexity Eval   → quick sanity check, ~2 min
2. Quality Eval      → verify generation diversity, ~5 min
3. WikiText-2        → standardized benchmark for comparison, ~5 min
4. Quality Eval with judge enabled  → final quality gate, ~15 min
```
""")

                # ── Expert Pipeline help ─────────────────────────────────────
                with gr.Accordion("Expert Pipeline — Domain-Specific Distillation with CoT", open=False):
                    gr.Markdown("""
The **Expert Pipeline** tab implements a 4-step workflow for building domain-specialist models
(tax, legal, medical, finance, coding) using any HuggingFace dataset and a GGUF teacher model.

---

### Step 1 — Dataset & Column Mapping
Load any HF dataset and remap its columns to the standard `instruction / input / output` format.

1. Enter the HF dataset ID (e.g. `nelson-liu/legalbench`) or a local path
2. Click **Inspect** — columns are auto-detected and dropdowns populated
3. Adjust column assignments if needed
4. Click **Remap & Save Dataset** → saves to the path in *Save remapped dataset to*

The remapped dataset is saved as an HF dataset on disk **and** as a `remapped.jsonl` for inspection.

**Supported datasets (tested):**

| Dataset | Domain | instruction_col | output_col |
|---------|--------|-----------------|------------|
| `Atome-LLM/Tax-Policy-Analysis` | Tax | `question` | `answer` |
| `nelson-liu/legalbench` | Legal | `instruction` | `output` |
| `medalpaca/medical_meadow_medical_flashcards` | Medical | `input` | `output` |
| `yahma/alpaca-cleaned` | General | `instruction` | `output` |
| `tatsu-lab/alpaca` | General | `instruction` | `output` |

---

### Step 2 — GGUF Teacher
Select a Metal-accelerated GGUF model from `/Users/Shared/llama/models/` to act as the teacher
for CoT generation. The model is served via `llama-server` during generation.

| Setting | Guidance |
|---------|----------|
| **Context size** | 8192 is safe for most models. Reduce to 4096 if the server OOMs. |
| **Parallel slots** | 4 keeps the GPU saturated. Reduce to 2 if getting timeouts. |
| **Temperature** | 0.3 for precise domains (tax, legal, medical). 0.7 for creative/coding. |
| **Max tokens** | 1024 covers most CoT responses. Increase to 2048 for complex reasoning. |

**Recommended GGUF teachers by domain:**

| Domain | Model | Quant | Why |
|--------|-------|-------|-----|
| Tax / Legal (best) | `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf` | Q4_K_M | Strongest reasoning at 70B; fits 36 GB unified |
| Legal (fast) | `law-chat.Q4_K_M.gguf` | Q4_K_M | Fine-tuned on statutory interpretation |
| Legal (alt) | `Llama-3-8B-Instruct-Legal-Q8_0.gguf` | Q8_0 | Highest-accuracy 8B legal model |
| Medical | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Q4_K_M | Best medical reasoning at 8B |
| Coding | `granite-3.1-8b-instruct-Q4_K_M.gguf` | Q4_K_M | IBM Granite excels at code |
| General | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Q4_K_M | Fast, good general quality |

> **Quantization guide:** Q4_K_M = best balance of size and quality. Q8_0 = highest accuracy, ~2× larger.
> All GGUF files go to `/Users/Shared/llama/models/` — click **Refresh** in the teacher dropdown to pick them up.

**Downloading GGUF models** — save all files to `/Users/Shared/llama/models/`:
```bash
cd /Users/Shared/llama/models

# 70B Llama-3 teacher (bartowski) — best for tax/legal CoT, needs ~42 GB RAM
curl -L -o Meta-Llama-3-70B-Instruct-Q4_K_M.gguf \\
  "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"

# Law-Chat 7B — statutory interpretation specialist
curl -L -o law-chat.Q4_K_M.gguf \\
  "https://huggingface.co/ricdomoliver/law-chat-GGUF/resolve/main/law-chat.Q4_K_M.gguf"

# Llama-3-8B Legal fine-tune (bartowski) — faster alternative
curl -L -o Llama-3-8B-Instruct-Legal-Q8_0.gguf \\
  "https://huggingface.co/bartowski/Llama-3-8B-Instruct-Legal-GGUF/resolve/main/Llama-3-8B-Instruct-Legal-Q8_0.gguf"
```
> **Note:** Verify the exact repo paths at huggingface.co before downloading — quantizer repos occasionally rename files.

**Downloading datasets:**
```bash
# LegalBench (Nguha, ~100 legal reasoning tasks) — use dataset ID in Step 1:
#   nguha/legalbench

# Tax-Policy-Analysis instruction data — use dataset ID in Step 1:
#   Atome-LLM/Tax-Policy-Analysis

# Download LegalBench as zip if you need an offline copy:
curl -L -o legalbench.zip \\
  "https://huggingface.co/datasets/nguha/legalbench/resolve/main/legalbench.zip"

# Tax parquet (if offline):
curl -L -o tax_data.parquet \\
  "https://huggingface.co/datasets/Atome-LLM/Tax-Policy-Analysis/resolve/main/data/train-00000-of-00001.parquet"
```
> **Tip:** For online runs just paste the HF dataset ID directly into the *Dataset ID* field — no download needed.

---

### Step 3 — CoT Rationale Generation
Prompts the GGUF teacher to generate **Chain-of-Thought reasoning traces** for each sample.

Each output is formatted as:
```
<reasoning>
[step-by-step domain analysis]
</reasoning>
<answer>
[final answer]
</answer>
```

The domain selector auto-fills a domain-specific system prompt (tax cites IRC sections,
legal applies statutory tests, medical follows clinical guidelines, etc.). You can override it.

**Output:** `cot_output_dir/cot_data.jsonl` + `cot_output_dir/hf_dataset/`

**Logs saved automatically to:**
- `runs/ep_cot_<domain>_<timestamp>.log` — plain text
- `runs/ep_cot_<domain>_<timestamp>.jsonl` — structured JSON (step, loss, ts, msg)

Both the **Training Loss** and **Gradient Norm** charts update live during this step,
and the full output streams in the **Live output** box below the accordions.

---

### Step 4 — Distillation
Launches `run_distillation_agent.py` on the CoT dataset. Leave *Dataset path* blank to
automatically use the `hf_dataset/` produced by Step 3.

**Key settings for domain expert models:**

| Setting | Recommended | Why |
|---------|-------------|-----|
| **LoRA rank** | 32 | Higher than default (16) — domain terminology requires more capacity |
| **Epochs** | 3 | CoT data is dense; 3 epochs ensures thorough absorption |
| **Backend** | MLX | Fastest on M3 Max; CoT outputs are long so MLX memory efficiency matters |

Distillation logs also save to `runs/ep_distill_<timestamp>.log/.jsonl`.

---

### Full expert pipeline (CLI)
```bash
# 1. Remap
python scripts/expert_pipeline.py --mode remap \\
    --dataset Atome-LLM/Tax-Policy-Analysis \\
    --instruction_col question --output_col answer \\
    --output_dir ./domain_data/tax_expert --max_samples 5000

# 2. Generate CoT (phi-4 teacher, tax domain)
python scripts/expert_pipeline.py --mode cot \\
    --dataset ./domain_data/tax_expert \\
    --teacher /Users/Shared/llama/models/phi-4-Q5_K_M.gguf \\
    --domain tax --n_samples 2000 \\
    --output_dir ./domain_data/tax_cot

# 3. Distill
python scripts/expert_pipeline.py --mode distill \\
    --dataset ./domain_data/tax_cot/hf_dataset \\
    --output_dir ./runs/tax-expert \\
    --backend mlx --open --lora_r 32 --epochs 3
```
""")

                # ── Tab 4 help ───────────────────────────────────────────────
                with gr.Accordion("Live Logs & Progress Bars — Reading Training Output", open=False):
                    gr.Markdown("""
### Progress bars
A compact progress bar appears **on every tab** — updates every 2 seconds by parsing log output
for `Step X/Y`, `Epoch X/Y`, tqdm `75%|` patterns, or `Progress: N%`. Turns green on completion.

### Loss & Gradient charts
The **Live Logs** tab and the **Expert Pipeline** tab both show live **Training Loss** and
**Gradient Norm** line charts, updated every 2 seconds. Both MLX (`step=N  loss=X`) and
PyTorch (`{'loss': 'X', 'grad_norm': 'Y'}`) log formats are parsed automatically.

### Log format
Full training output streams in real time (polled every 2 seconds). stdout and stderr merged.

### Key metrics to watch (MiniLLM / GRPO)

| Metric | What it is | Healthy range | Action if outside range |
|--------|-----------|---------------|------------------------|
| `loss` | Training loss (reverse-KL) | Decreasing over first 50 steps | If rising after step 30, LR may be too high |
| `eval_loss` | Validation cross-entropy (logged every eval_steps) | Should decrease and track training loss | Large gap = overfitting |
| `reward` / `eval_reward` | Mean reward across completions in a batch | Should trend from negative toward positive | Stuck at -1.0 = clipping or mode collapse |
| `clipped_ratio` | Fraction of completions that hit max_completion_length | < 30% | If > 60%, lower Max completion length or increase it if responses should be long |
| `frac_reward_zero_std` | Fraction of prompt groups where all completions have identical reward (no GRPO gradient) | < 20% after step 50 | If > 50%, reduce Generations per prompt or check reward function |
| `kl` | KL divergence between student and teacher | Should be finite and decreasing | NaN or exploding = training diverged, stop and reduce LR |

### Key metrics (SFT)
| Metric | Healthy | Notes |
|--------|---------|-------|
| `loss` | Decreasing from ~3–5 to ~1–2 over 1 epoch | Fast decrease early = good. Plateau at >2 = check data quality |
| `grad_norm` | 0.5–2.0 | Spike to >10 = LR too high or data issue |

### Key metrics (MLX)
| Metric | Notes |
|--------|-------|
| `kd_loss` | Forward-KL distillation loss, should decrease |
| `ce_loss` | Cross-entropy loss component (scaled by ce_alpha) |
| `total_loss` | Weighted sum: `(1-ce_alpha)*kd_loss + ce_alpha*ce_loss` |
| `eval_ppl` | Validation perplexity, logged every eval_steps |

### Warning signs
- **`[Process exited with code -9]`** — You clicked Stop (SIGKILL). Normal.
- **`[Process exited with code 1]`** — Script crashed. Scroll up in logs for the Python traceback.
- **`OutOfMemoryError` / `MPS backend out of memory`** — Reduce Batch size or Max completion length. On M3 Max, batch=4 and completion=64 always fits.
- **`RuntimeError: Expected all tensors on same device`** — MPS + bfloat16 edge case. Usually resolves after a restart.
- **`ImportError: trl`** — TRL not installed. Run `pixi run pip install trl`.
- **`nan` in loss after step 1** — Learning rate too high. Reduce by 10×.
- **Reward stuck at -0.5 from step 1** — All completions are clipping. Increase Max completion length to 256+ so responses have room to terminate before the hard limit.
- **`frac_reward_zero_std` > 0.6 after step 20** — All completions in a group get identical reward so GRPO advantage is zero. Increase Generations per prompt to 4–8.

### Thermal note
On M3 Max, GPU temperature under MPS load typically stays at 50–60°C. If you see the machine throttling (iterations getting slower over time), training will continue correctly — the pause.flag watchdog can be used to cool it down without losing progress.

### After training completes
The log ends with:
```
Distilled model saved to ./distilled-minillm
[Process exited with code 0]
```
The merged weights are in the output directory. Go to **Eval** to validate, or run the export scripts (`scripts/export_student_gguf.sh`) to produce a GGUF for llama.cpp.
""")

                # ── Troubleshooting ──────────────────────────────────────────
                with gr.Accordion("Troubleshooting & FAQ", open=False):
                    gr.Markdown("""
### Q: Training is very slow (>300s per iteration)
- Reduce **Max completion length** to 64–96. Generation is the bottleneck — shorter completions = dramatically faster.
- Reduce **Generations per prompt** to 2 (minimum).
- Reduce **Eval every N steps** to 50+ to spend less time on evaluation.
- Switch backend to **MLX** (2–5× faster on M3).

### Q: Reward is stuck at -1.0 (mode collapse)
- Lower **Max completion length** — the model is generating nothing meaningful before hitting the hard limit.
- Increase **KD temperature** to 1.5 — softens the teacher targets, easier for student to match.
- Use SFT warmup first — gives the student a starting distribution to build from.

### Q: clipped_ratio is >80%
- The student is hitting max_completion_length on almost every generation.
- Lower **Max completion length** to 64 or 96 tokens.
- Or: the model is in a loop/repetition mode — check distinct-2 with Quality Eval.

### Q: Loss is NaN or exploding after a few steps
- **Learning rate too high.** Lower by 10×: 2e-5 → 2e-6 for MiniLLM, 2e-4 → 2e-5 for SFT.
- Check **grad_norm** in logs — if it's >10 before loss explodes, LR is definitely too high.

### Q: "No module named 'trl'" or "No module named 'transformers'"
```bash
cd /Users/caribou/distill
pixi run pip install trl transformers peft datasets
```
Or relaunch the UI through pixi: `pixi run python scripts/launch_ui.py`

### Q: Can I run multiple jobs at once?
No — only one subprocess is managed by the UI at a time. If you need parallel runs, open a second terminal and call the scripts directly. The UI will show "A run is already in progress" if you try to launch while one is active.

### Q: Where are HuggingFace models cached?
Default: `~/.cache/huggingface/hub/`. Set `HF_HOME` environment variable to redirect. The dropdowns auto-scan this cache.

### Q: How do I use a locally downloaded GGUF model?
GGUFs are for inference only (llama.cpp / Ollama), not training. The training scripts use HF-format models. The pipeline exports to GGUF *after* training via `scripts/export_student_gguf.sh`.

### Q: The progress bar shows 0% but a run is active
The bar parses `Step X/Y`, `Epoch X/Y`, or tqdm `N%|` from the log. Scripts that don't emit
those patterns (e.g. model download phase) show 0% until the first progress line appears.
The Run status textbox ("running (pid …)") confirms the process is alive regardless.

### Q: The UI shows "idle" but I launched a run
Check **Live Logs** tab — the process may have crashed immediately. The status polling updates every 2 seconds.
Also check: if you launched from a tab other than Configure & Launch, only that tab's status textbox
updates on click; all tabs' progress bars update via the timer once the process writes output.
""")

                # ── Algorithm Reference ──────────────────────────────────────
                gr.Markdown("---\n### Algorithm Reference")
                try:
                    sys.path.insert(0, str(Path(__file__).parent))
                    from show_algorithms import ALGORITHMS, build_html as _build_html_help
                    _help_html = _build_html_help(ALGORITHMS)
                    _help_b64 = base64.b64encode(_help_html.encode("utf-8")).decode("ascii")
                    gr.HTML(
                        f'<div style="border-radius:10px;overflow:hidden;border:1px solid #2a2d3e;">'
                        f'<iframe src="data:text/html;base64,{_help_b64}" '
                        f'style="width:100%;height:80vh;border:none;" '
                        f'sandbox="allow-scripts"></iframe>'
                        f'</div>'
                    )
                except Exception as _e:
                    gr.Markdown(f"⚠️ Could not load algorithms: `{_e}`")

        # ── Stage / backend toggle ────────────────────────────────────────────
        def on_stage_change(s):
            is_sft = s == "SFT"
            return (
                gr.update(visible=is_sft),
                gr.update(visible=not is_sft),
                gr.update(visible=False),  # mlx_group always hidden when stage flips
            )

        def on_backend_change(b):
            is_mlx = b == "MLX"
            # Suggest backend-appropriate defaults for batch/grad_acc/lora_r
            if is_mlx:
                return (
                    gr.update(visible=False),  # minillm_group
                    gr.update(visible=True),   # mlx_group
                    gr.update(value=2),        # batch_size
                    gr.update(value=8),        # grad_acc (effective batch=16; higher than PyTorch default since MLX is more memory-efficient)
                    gr.update(value=16),       # lora_r (same capacity as PyTorch default)
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value=8),
                    gr.update(value=8),
                    gr.update(value=16),
                )

        stage.change(
            on_stage_change, stage,
            [sft_group, minillm_group, mlx_group],
        )
        backend.change(
            on_backend_change, backend,
            [minillm_group, mlx_group, batch_size, grad_acc, lora_r],
        )

        # ── Refresh dropdowns ────────────────────────────────────────────────
        refresh_teacher_btn.click(
            fn=lambda: gr.update(choices=discover_teachers()),
            outputs=teacher,
        )
        refresh_student_btn.click(
            fn=lambda: gr.update(choices=discover_students()),
            outputs=student,
        )
        refresh_dataset_btn.click(
            fn=lambda: gr.update(choices=discover_datasets()),
            outputs=dataset,
        )
        refresh_outdir_btn.click(
            fn=lambda: gr.update(choices=discover_output_dirs()),
            outputs=output_dir,
        )

        # ── Launch / Stop ────────────────────────────────────────────────────
        all_inputs = [
            stage, backend, use_open,
            teacher, student,
            dataset, output_dir,
            epochs, batch_size, grad_acc, lora_r, max_samples,
            sft_lr, max_new_tokens_sft, max_length,
            minillm_temp, minillm_lr, num_generations, max_completion_length, eval_steps,
            mlx_kd_temp, mlx_lr, mlx_eval_steps, mlx_ce_alpha, mlx_topk, mlx_q_bits, mlx_resume,
            watchdog,
        ]
        launch_btn.click(fn=launch_run, inputs=all_inputs, outputs=[log_box, run_status])
        stop_btn.click(fn=stop_run, outputs=[log_box, run_status])
        clear_btn.click(fn=clear_logs, outputs=[log_box, log_status, loss_plot, grad_plot, ep_log_box, ep_loss_plot, ep_grad_plot])

        # ── Domain synthesis wiring ──────────────────────────────────────────
        dom_refresh_btn.click(
            fn=lambda: gr.update(choices=discover_teachers()), outputs=dom_teacher,
        )
        # Wire each domain button with a closure capturing the domain name
        for _domain, _btn, _n_slider, _target_slider, _outdir_box in [
            ("medical", med_btn,   med_n,   med_target,   med_outdir),
            ("math",    math_btn,  math_n,  math_target,  math_outdir),
            ("legal",   legal_btn, legal_n, legal_target, legal_outdir),
            ("tax",     tax_btn,   tax_n,   tax_target,   tax_outdir),
            ("coding",  code_btn,  code_n,  code_target,  code_outdir),
            ("finance", fin_btn,   fin_n,   fin_target,   fin_outdir),
        ]:
            def _make_handler(d):
                def _handler(teacher, n, batch, use_filter, target, outdir, offline, backend):
                    return launch_magpie(teacher, outdir, n, batch, use_filter, target, offline,
                                        domain=d, backend=backend)
                return _handler

            _btn.click(
                fn=_make_handler(_domain),
                inputs=[dom_teacher, _n_slider, dom_batch, dom_filter, _target_slider,
                        _outdir_box, dom_offline, dom_backend],
                outputs=[log_box, dom_status],
            )

        # ── Custom domain wiring ─────────────────────────────────────────────
        _custom_filter_inputs = [
            custom_id, custom_label, custom_desc, custom_prompts,
            custom_min_words, custom_max_words, custom_min_d2,
            custom_req_code, custom_req_nums,
        ]
        custom_save_btn.click(
            fn=save_custom_domain,
            inputs=_custom_filter_inputs,
            outputs=custom_save_status,
        )

        def _save_and_launch(domain_id, label, desc, prompts_text,
                             min_words, max_words, min_d2, req_code, req_nums,
                             teacher, n, batch, use_filter, target, outdir, offline, backend="auto"):
            # Derive the sanitized domain ID the same way save_custom_domain does
            did = (domain_id or "").strip().lower().replace(" ", "_") or "custom"
            save_msg = save_custom_domain(domain_id, label, desc, prompts_text,
                                         min_words, max_words, min_d2, req_code, req_nums)
            if save_msg.startswith("Error"):
                return save_msg, "idle"
            log_msg, status = launch_magpie(teacher, outdir, n, batch, use_filter, target, offline,
                                            domain=did, backend=backend)
            return log_msg, status

        custom_launch_btn.click(
            fn=_save_and_launch,
            inputs=_custom_filter_inputs + [dom_teacher, custom_n, dom_batch, dom_filter,
                                            custom_target, custom_outdir, dom_offline, dom_backend],
            outputs=[log_box, dom_status],
        )

        # ── Data Prep wiring ─────────────────────────────────────────────────
        mag_refresh_btn.click(
            fn=lambda: gr.update(choices=discover_teachers()), outputs=mag_teacher,
        )
        filt_ds_refresh.click(
            fn=lambda: gr.update(choices=discover_datasets()), outputs=filt_dataset,
        )
        synth_refresh_btn.click(
            fn=lambda: gr.update(choices=discover_teachers()), outputs=synth_teacher,
        )
        mag_launch_btn.click(
            fn=launch_magpie,
            inputs=[mag_teacher, mag_output_dir, mag_n, mag_batch_size,
                    mag_filter, mag_target, mag_offline, mag_backend],
            outputs=[log_box, mag_status],
        )
        mag_stop_btn.click(fn=stop_run, outputs=[log_box, mag_status])
        synth_launch_btn.click(
            fn=launch_synth,
            inputs=[synth_teacher, synth_use_open, synth_output_dir, synth_n_generate,
                    synth_batch_size, synth_temperature, synth_seed_examples, synth_offline],
            outputs=[log_box, synth_status],
        )
        synth_stop_btn.click(fn=stop_run, outputs=[log_box, synth_status])
        filt_launch_btn.click(
            fn=launch_filter,
            inputs=[filt_dataset, filt_output_dir, filt_target,
                    filt_min_words, filt_min_d2, filt_offline],
            outputs=[log_box, filt_status],
        )
        filt_stop_btn.click(fn=stop_run, outputs=[log_box, filt_status])

        # ── Eval tab wiring ──────────────────────────────────────────────────
        eval_refresh_btn.click(
            fn=lambda: gr.update(choices=discover_output_dirs()), outputs=eval_output_dir,
        )
        ppl_launch_btn.click(
            fn=launch_eval_perplexity,
            inputs=[eval_output_dir, eval_checkpoint, eval_student, eval_dataset,
                    ppl_max_val, ppl_batch, ppl_compare_teacher, ppl_teacher, eval_offline],
            outputs=[log_box, ppl_status],
        )
        ppl_stop_btn.click(fn=stop_run, outputs=[log_box, ppl_status])
        qual_launch_btn.click(
            fn=launch_eval_quality,
            inputs=[eval_output_dir, eval_checkpoint, eval_student, eval_dataset,
                    qual_n_samples, qual_judge, qual_judge_teacher, eval_offline],
            outputs=[log_box, qual_status],
        )
        qual_stop_btn.click(fn=stop_run, outputs=[log_box, qual_status])
        bench_launch_btn.click(
            fn=launch_eval_benchmark,
            inputs=[eval_output_dir, eval_checkpoint, eval_student,
                    bench_n_seq, bench_batch, bench_baseline, bench_threshold, eval_offline],
            outputs=[log_box, bench_status],
        )
        bench_stop_btn.click(fn=stop_run, outputs=[log_box, bench_status])

        # ── Expert Pipeline wiring ───────────────────────────────────────────

        def _ep_inspect(dataset_id: str):
            """Run expert_pipeline.py --mode inspect and parse JSON result."""
            ds = dataset_id.strip()
            if not ds or ds == "Enter a dataset ID first.":
                return ("Enter a dataset ID first.", [], [], ["(none)"])
            import subprocess as _sp
            result = _sp.run(
                [PYTHON, str(SCRIPTS_DIR / "expert_pipeline.py"),
                 "--mode", "inspect", "--dataset", ds],
                capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=120,
            )
            raw = result.stdout.strip()
            if not raw:
                err = result.stderr[-500:] if result.stderr else "No output"
                return (f"Inspect failed:\n{err}", [], [], ["(none)"])
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return (f"Could not parse output:\n{raw[:300]}", [], [], ["(none)"])
            cols = data.get("columns", [])
            guessed = data.get("guessed", {})
            info = (f"{data.get('n_rows', '?')} rows  |  columns: {', '.join(cols)}\n"
                    f"Auto-detected → instruction: {guessed.get('instruction')}  "
                    f"output: {guessed.get('output')}  input: {guessed.get('input')}")
            col_choices = cols
            inst_default = guessed.get("instruction") or (cols[0] if cols else "")
            out_default  = guessed.get("output") or (cols[1] if len(cols) > 1 else "")
            input_choices = ["(none)"] + cols
            input_default = guessed.get("input") or "(none)"
            return (
                info,
                gr.update(choices=col_choices, value=inst_default),
                gr.update(choices=col_choices, value=out_default),
                gr.update(choices=input_choices, value=input_default),
            )

        ep_inspect_btn.click(
            fn=_ep_inspect,
            inputs=ep_dataset,
            outputs=[ep_inspect_status, ep_instruction_col, ep_output_col, ep_input_col],
        )
        ep_dataset_refresh.click(
            fn=lambda: gr.update(choices=_ep_dataset_choices()),
            outputs=ep_dataset,
        )

        def _ep_remap(dataset, instruction_col, output_col, input_col, max_samples, output_dir):
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

        ep_remap_btn.click(
            fn=_ep_remap,
            inputs=[ep_dataset, ep_instruction_col, ep_output_col, ep_input_col,
                    ep_max_samples_remap, ep_remap_output],
            outputs=[log_box, ep_status],
        )

        def _ep_load_system_prompt(domain: str):
            import sys as _sys, importlib
            _sys.path.insert(0, str(SCRIPTS_DIR))
            ep = importlib.import_module("expert_pipeline")
            prompt = ep.DOMAIN_SYSTEM_PROMPTS.get(domain, ep.DEFAULT_SYSTEM_PROMPT)
            return gr.update(placeholder=prompt, value="")

        ep_domain.change(fn=_ep_load_system_prompt, inputs=ep_domain, outputs=ep_system_prompt)

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

        ep_cot_btn.click(
            fn=_ep_cot,
            inputs=[ep_remap_output, ep_teacher, ep_domain, ep_system_prompt,
                    ep_n_cot, ep_cot_temp, ep_max_tokens, ep_ctx_size,
                    ep_n_parallel, ep_batch_size_cot, ep_cot_output],
            outputs=[log_box, ep_status],
        )
        ep_cot_stop.click(fn=stop_run, outputs=[log_box, ep_status])

        def _ep_teacher_refresh():
            choices = sorted(str(p) for p in Path("/Users/Shared/llama/models").glob("*.gguf")) \
                if Path("/Users/Shared/llama/models").exists() else []
            return gr.update(choices=choices)

        ep_teacher_refresh.click(fn=_ep_teacher_refresh, outputs=ep_teacher)

        def _ep_distill(cot_dir, distill_dataset, output_dir, backend, epochs,
                        lora_r, max_samples, open_flag, offline_flag):
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

        ep_distill_btn.click(
            fn=_ep_distill,
            inputs=[ep_cot_output, ep_distill_dataset, ep_distill_output,
                    ep_distill_backend, ep_epochs, ep_lora_r, ep_max_samp,
                    ep_open_chk, ep_offline_chk],
            outputs=[log_box, ep_status],
        )
        ep_distill_stop.click(fn=stop_run, outputs=[log_box, ep_status])

        # ── Poll logs every 2 s ──────────────────────────────────────────────
        gr.Timer(value=2).tick(
            fn=poll_logs,
            inputs=log_box,
            outputs=[
                log_box, log_status, training_progress,  # Live Logs tab
                launch_progress,    # Configure & Launch tab
                data_prep_progress, # Data Prep tab
                domain_progress,    # Domain Synthesis tab
                eval_progress,      # Eval tab
                dom_status,         # Domain Synthesis status
                loss_plot,          # Training loss chart
                grad_plot,          # Gradient norm chart
                ep_progress,        # Expert Pipeline tab progress
                ep_log_box,         # Expert Pipeline embedded log
                ep_loss_plot,       # Expert Pipeline loss chart
                ep_grad_plot,       # Expert Pipeline grad norm chart
            ],
        )

    return demo


def _free_port(port: int) -> None:
    """Kill any process listening on the given port."""
    import signal
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return  # port already free
    except OSError:
        return
    # Port is occupied — find and kill the owner
    result = subprocess.run(
        ["lsof", "-ti", f":{port}"], capture_output=True, text=True
    )
    for pid_str in result.stdout.split():
        try:
            os.kill(int(pid_str), signal.SIGTERM)
        except (ProcessLookupError, ValueError):
            pass
    import time
    time.sleep(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()
    _free_port(args.port)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=False,
                theme=gr.themes.Monochrome())


if __name__ == "__main__":
    main()
