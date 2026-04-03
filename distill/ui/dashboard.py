#!/usr/bin/env python3
"""
Unified Gradio dashboard: training plots + model evaluation.
Runs locally on 127.0.0.1. Air-gapped friendly.
Supports PyTorch, MLX, GGUF, and vLLM backends.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..backends.universal_loader import UniversalModelLoader, detect_model_format
from ..infra.artifact_detector import detect_artifacts
from ..backends.mlx_utils import is_mlx_available, load_mlx_model, mlx_generate_responses
from ..backends.cpp_utils import find_gguf
from ..infra.metrics_io import load_trainer_state
from .dashboard_plots import plot_from_state, on_run_select
from .dashboard_models import select_and_load_model, _diversity_metrics, _discover_all_models
from .dashboard_streaming import _run_streaming, _parse_progress_from_log, _progress_bar_html, _is_streaming_done
from .dashboard_eval_ui import build_eval_ui
from .dashboard_run_evals import build_run_evals_ui


def parse_args():
    p = argparse.ArgumentParser(description="Distillation dashboard")
    p.add_argument("--runs_dir", type=str, default=".", help="Parent dir of training outputs")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def find_run_dirs(runs_dir):
    """Find directories containing trainer_state.json."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    found = []
    for d in root.iterdir():
        if d.is_dir() and (d / "trainer_state.json").exists():
            found.append(str(d))
    # Also check runs_dir itself
    if (root / "trainer_state.json").exists():
        found.insert(0, str(root))
    return sorted(set(found))


def find_pipeline_dirs(runs_dir):
    """
    Find directories suitable for the pipeline view.
    A directory qualifies if it looks like a distillation output:
    - has trainer_state.json at root or inside a checkpoint subdir, OR
    - has *.gguf files AND (config.json or training_args.bin)
    Returns absolute paths.
    """
    root = Path(runs_dir).resolve()
    if not root.exists():
        return []
    found = set()
    candidates = [root] + [d for d in root.iterdir() if d.is_dir()]
    for d in candidates:
        if (d / "trainer_state.json").exists():
            found.add(str(d))
            continue
        # trainer_state.json inside a checkpoint subdir
        try:
            if any((sub / "trainer_state.json").exists()
                   for sub in d.iterdir() if sub.is_dir()):
                found.add(str(d))
                continue
        except PermissionError:
            continue
        # Live run: metrics.jsonl exists (trainer_state.json not yet written)
        if (d / "metrics.jsonl").exists():
            found.add(str(d))
            continue
        # GGUF files alongside a config.json or training_args.bin (distill output)
        if list(d.glob("*.gguf")) and (
            (d / "config.json").exists() or (d / "training_args.bin").exists()
        ):
            found.add(str(d))
    return sorted(found)


def main():
    args = parse_args()
    model_state = [None, None]  # [UniversalModelLoader, backend]
    run_dirs = find_run_dirs(args.runs_dir)
    pipeline_dirs = find_pipeline_dirs(args.runs_dir)
    with gr.Blocks(title="Distillation Dashboard") as app:
        gr.Markdown("# Distillation Dashboard")
        gr.Markdown("Training curves and model evaluation. Runs locally only.")
        with gr.Tabs():
            with gr.Tab("Plots"):
                gr.Markdown("### Training curves")
                with gr.Row():
                    run_dropdown = gr.Dropdown(
                        choices=pipeline_dirs,
                        value=pipeline_dirs[0] if pipeline_dirs else None,
                        label="Run directory",
                        scale=4,
                    )
                    plots_refresh_btn = gr.Button("Refresh", scale=1)
                plot_output = gr.Plot(
                    label="Loss & learning rate",
                    value=on_run_select(pipeline_dirs[0]) if pipeline_dirs else None,
                )
                run_dropdown.change(on_run_select, run_dropdown, plot_output)
                plots_refresh_btn.click(on_run_select, run_dropdown, plot_output)
            with gr.Tab("Pipeline"):
                gr.Markdown("### End-to-end pipeline summary")
                pipeline_run_dd = gr.Dropdown(
                    choices=pipeline_dirs,
                    value=pipeline_dirs[0] if pipeline_dirs else None,
                    label="Run directory",
                )
                pipeline_plot = gr.Plot(label="Pipeline summary")
                refresh_btn = gr.Button("Refresh")

                def on_pipeline_select(run_path):
                    if not run_path:
                        return None
                    from .plot_gguf_pipeline import plot_pipeline
                    return plot_pipeline(run_path)

                pipeline_run_dd.change(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                refresh_btn.click(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                if pipeline_dirs:
                    pipeline_plot.value = on_pipeline_select(pipeline_dirs[0])
            with gr.Tab("Thermal"):
                gr.Markdown("### CPU / GPU temperature & power over time")

                # ── Shared helpers ────────────────────────────────────────────
                _MFC_CLI = "/Applications/Macs Fan Control.app/Contents/MacOS/Macs Fan Control"

                def _parse_mactop(stdout: str) -> dict:
                    """Parse mactop JSON output. Handles both list+soc_metrics and flat formats."""
                    for line in reversed(stdout.strip().splitlines()):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            if isinstance(data, list) and data and "soc_metrics" in data[0]:
                                return data[0]["soc_metrics"]
                            if isinstance(data, dict):
                                return data
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue
                    return {}

                def _fmt_t(v):
                    try: return f"{float(v):.1f}°C"
                    except (TypeError, ValueError): return str(v)

                def _fmt_w(v):
                    try: return f"{float(v):.1f} W"
                    except (TypeError, ValueError): return str(v)

                # ── Live mactop reading ───────────────────────────────────────
                mactop_btn = gr.Button("🌡 Read Now  (mactop)", variant="secondary")
                mactop_out = gr.Textbox(label="Live reading", interactive=False, lines=3)

                def read_mactop():
                    try:
                        result = subprocess.run(
                            ["/opt/homebrew/bin/mactop",
                             "--headless", "--format", "json", "--count", "1"],
                            capture_output=True, text=True, timeout=10,
                        )
                        m = _parse_mactop(result.stdout)
                        if not m:
                            return f"Could not parse mactop output:\n{result.stdout[:200]}"
                        cpu_t = m.get("cpu_temp", m.get("cpu_temp_c", "?"))
                        gpu_t = m.get("gpu_temp", m.get("gpu_temp_c", "?"))
                        soc_t = m.get("soc_temp", m.get("soc_temp_c", "?"))
                        cpu_w = m.get("cpu_power", m.get("cpu_power_w", "?"))
                        gpu_w = m.get("gpu_power", m.get("gpu_power_w", "?"))
                        tot_w = m.get("total_power", m.get("total_power_w", "?"))
                        lines = [
                            f"CPU  {_fmt_t(cpu_t)}   GPU  {_fmt_t(gpu_t)}   SOC  {_fmt_t(soc_t)}",
                            f"CPU power  {_fmt_w(cpu_w)}   GPU power  {_fmt_w(gpu_w)}   Total  {_fmt_w(tot_w)}",
                        ]
                        for label, val in [("CPU", cpu_t), ("GPU", gpu_t)]:
                            try:
                                if float(val) >= 90:
                                    lines.append(f"⚠ {label} ≥ 90°C — watchdog pause threshold")
                            except (TypeError, ValueError):
                                pass
                        return "\n".join(lines)
                    except FileNotFoundError:
                        return "mactop not found at /opt/homebrew/bin/mactop"
                    except Exception as e:
                        return f"Error: {e}"

                mactop_btn.click(fn=read_mactop, inputs=[], outputs=mactop_out)

                gr.Markdown("---")

                # ── Fan control ───────────────────────────────────────────────
                gr.Markdown("#### Fan Control  (Macs Fan Control)")

                with gr.Row():
                    fan_status_btn = gr.Button("Check Status", variant="secondary", scale=2)
                    fan_launch_btn = gr.Button("Launch App", variant="secondary", scale=1)
                fan_feedback = gr.Textbox(label="Status", interactive=False, lines=1)

                def _mfc_status():
                    if not os.path.exists(_MFC_CLI):
                        return "Not installed — brew install --cask macs-fan-control"
                    r = subprocess.run(["pgrep", "-f", "Macs Fan Control.app"],
                                       capture_output=True, timeout=2)
                    return "Running" if r.returncode == 0 else "Not running — click Launch App before setting RPM"

                def _mfc_launch():
                    if not os.path.exists(_MFC_CLI):
                        return "Not installed — brew install --cask macs-fan-control"
                    try:
                        subprocess.Popen(["open", "-a", "Macs Fan Control"])
                        return "Launched Macs Fan Control"
                    except Exception as e:
                        return f"Failed: {e}"

                def _mfc_cmd(*args):
                    """Run a Macs Fan Control CLI command, return feedback string."""
                    if not os.path.exists(_MFC_CLI):
                        return "Macs Fan Control not installed"
                    r = subprocess.run([_MFC_CLI, *args],
                                       capture_output=True, text=True, timeout=5)
                    return r.stdout.strip() or r.stderr.strip() or "Done"

                def set_auto():
                    return _write_preset(
                        "Auto",
                        (0, "", 0, 0),
                        (0, "", 0, 0),
                    )

                def set_custom(v):
                    rpm = int(v)
                    return _write_preset(
                        f"Custom {rpm} RPM",
                        (1, "", rpm, 0),
                        (1, "", rpm, 0),
                    )

                fan_status_btn.click(fn=_mfc_status, inputs=[], outputs=fan_feedback)
                fan_launch_btn.click(fn=_mfc_launch, inputs=[], outputs=fan_feedback)

                # ── Preset helpers ────────────────────────────────────────────
                def _decode_fan(seg):
                    """Human-readable string for one fan config segment."""
                    parts = seg.split(",")
                    mode = parts[0] if parts else "?"
                    if mode == "0":
                        return "Auto"
                    if mode == "1":
                        rpm = parts[2] if len(parts) > 2 else "?"
                        return f"Constant {rpm} RPM"
                    if mode == "2":
                        sensor = parts[1].replace("_", " ").title() if len(parts) > 1 else "?"
                        mn = parts[2] if len(parts) > 2 else "?"
                        mx = parts[3] if len(parts) > 3 else "?"
                        return f"Sensor: {sensor}  {mn}–{mx} °C"
                    return seg

                def _read_active_preset():
                    import base64 as _b64
                    r = subprocess.run(
                        ["defaults", "read", "com.crystalidea.macsfancontrol", "ActivePreset"],
                        capture_output=True, text=True, timeout=5,
                    )
                    val = r.stdout.strip().strip('"')
                    b64 = val.split(":", 1)[-1] if ":" in val else val
                    try:
                        decoded = _b64.b64decode(b64).decode()
                        parts = decoded.split("|")
                        name = parts[0]
                        fans = [_decode_fan(p) for p in parts[1:]]
                        labels = ["Left ", "Right"]
                        lines = [f"Preset: {name}"] + [
                            f"  {labels[i]}: {fans[i]}" for i in range(len(fans))
                        ]
                        return "\n".join(lines)
                    except Exception as e:
                        return f"Raw: {val}\n(decode error: {e})"

                def _write_preset(name, fan0, fan1):
                    """Encode and write sensor-based preset via defaults write."""
                    import base64 as _b64
                    m0, s0, mn0, mx0 = fan0
                    m1, s1, mn1, mx1 = fan1
                    raw = f"{name}|{m0},{s0},{mn0},{mx0}|{m1},{s1},{mn1},{mx1}"
                    b64 = _b64.b64encode(raw.encode()).decode()
                    subprocess.run(
                        ["defaults", "write", "com.crystalidea.macsfancontrol",
                         "ActivePreset", f"Unsaved:{b64}"],
                        timeout=5,
                    )
                    # Activate app so it re-reads preferences
                    subprocess.run(
                        ["osascript", "-e",
                         'tell application "Macs Fan Control" to activate'],
                        capture_output=True, timeout=3,
                    )
                    return _read_active_preset()

                # ── Current preset ────────────────────────────────────────────
                with gr.Row():
                    fan_read_btn = gr.Button("📋 Read Current Preset", variant="secondary", scale=3)
                fan_current = gr.Textbox(label="Active preset", interactive=False, lines=3)
                fan_read_btn.click(fn=_read_active_preset, inputs=[], outputs=fan_current)

                # ── Sensor presets ────────────────────────────────────────────
                gr.Markdown("**Sensor-based presets**  _(both fans, GPU Clusters Average)_")
                gr.Markdown(
                    "_Left side: 1350–5349 RPM  ·  Right side: 1458–5777 RPM_"
                )
                with gr.Row():
                    btn_s_25_33 = gr.Button("GPU Sensor  25–33 °C", variant="primary")
                    btn_s_28_36 = gr.Button("GPU Sensor  28–36 °C")
                    btn_auto_preset = gr.Button("Auto  (system)")

                def apply_25_33():
                    return _write_preset(
                        "GPU Sensor 25-33",
                        (2, "gpu_clusters_average", 25, 33),
                        (2, "gpu_clusters_average", 25, 33),
                    )
                def apply_28_36():
                    return _write_preset(
                        "Matrix GPU Auto",
                        (2, "gpu_clusters_average", 28, 36),
                        (2, "gpu_clusters_average", 28, 36),
                    )

                btn_s_25_33.click(fn=apply_25_33, inputs=[], outputs=fan_feedback)
                btn_s_28_36.click(fn=apply_28_36, inputs=[], outputs=fan_feedback)
                btn_auto_preset.click(fn=set_auto, inputs=[], outputs=fan_feedback)

                # ── Constant RPM ──────────────────────────────────────────────
                gr.Markdown("**Constant RPM**  _(shared safe range for both fans)_")
                with gr.Row():
                    fan_slider = gr.Slider(1458, 5349, value=3000, step=50,
                                           label="RPM  (Left 1350–5349  ·  Right 1458–5777)",
                                           scale=4)
                    fan_apply_btn = gr.Button("Apply RPM", variant="primary", scale=1)
                fan_apply_btn.click(fn=set_custom, inputs=[fan_slider], outputs=fan_feedback)

                gr.Markdown("---")
                thermal_log_box = gr.Textbox(
                    label="Log file path",
                    value=str(Path(args.runs_dir).parent / "thermal.log"),
                    placeholder="/path/to/thermal.log",
                )
                thermal_refresh_btn = gr.Button("Load / Refresh")
                thermal_plot = gr.Plot(label="Thermal history")

                def load_thermal(log_path):
                    import csv
                    from datetime import datetime
                    p = Path(log_path)
                    if not p.exists():
                        return None
                    rows = []
                    try:
                        with open(p, newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                rows.append(row)
                    except OSError:
                        return None
                    if not rows:
                        return None
                    # Detect format by header
                    headers = list(rows[0].keys())
                    has_temp = "cpu_temp_c" in headers
                    times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w = [], [], [], [], [], [], []
                    for row in rows:
                        try:
                            t = datetime.strptime(row["time"].strip(), "%Y-%m-%d %H:%M:%S")
                        except (ValueError, KeyError):
                            continue
                        times.append(t)
                        if has_temp:
                            def _f(v):
                                try: return float(v)
                                except (TypeError, ValueError): return float("nan")
                            cpu_t.append(_f(row.get("cpu_temp_c", "")))
                            gpu_t.append(_f(row.get("gpu_temp_c", "")))
                            soc_t.append(_f(row.get("soc_temp_c", "")))
                            cpu_w.append(_f(row.get("cpu_power_w", "")))
                            gpu_w.append(_f(row.get("gpu_power_w", "")))
                            tot_w.append(_f(row.get("total_power_w", "")))
                    if not times:
                        return None
                    panels = [("temp", "Temperature (°C)"), ("power", "Power (W)")]
                    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
                    fig.suptitle(f"Thermal — {p.name}", fontsize=11)
                    ax0, ax1 = axes
                    if has_temp:
                        ax0.plot(times, cpu_t, label="CPU", color="steelblue")
                        ax0.plot(times, gpu_t, label="GPU", color="tomato")
                        ax0.plot(times, soc_t, label="SOC", color="goldenrod", linestyle="--", alpha=0.6)
                        ax0.axhline(90, color="red", linestyle=":", linewidth=1, label="pause threshold (90°C)")
                    ax0.set_ylabel("°C")
                    ax0.legend(loc="upper left", fontsize=8)
                    ax0.grid(True, alpha=0.3)
                    if has_temp:
                        ax1.plot(times, cpu_w, label="CPU", color="steelblue")
                        ax1.plot(times, gpu_w, label="GPU", color="tomato")
                        ax1.plot(times, tot_w, label="Total", color="gray", linestyle="--", alpha=0.7)
                    ax1.set_ylabel("W")
                    ax1.set_xlabel("Time")
                    ax1.legend(loc="upper left", fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    fig.autofmt_xdate(rotation=30)
                    plt.tight_layout()
                    return fig

                thermal_refresh_btn.click(load_thermal, thermal_log_box, thermal_plot)
                # Auto-load on startup
                _default_thermal = str(Path(args.runs_dir).parent / "thermal.log")

            with gr.Tab("Evaluate"):
                build_eval_ui(args.runs_dir, model_state)
            with gr.Tab("▶ Run Evals"):
                build_run_evals_ui(args.runs_dir, pipeline_dirs)
            with gr.Tab("Quality"):
                gr.Markdown("### Quality metrics (from eval_quality.py)")
                gr.Markdown(
                    "Run `python scripts/eval_quality.py <model_dir> --judge` to generate this report."
                )
                with gr.Row():
                    quality_run_dd = gr.Dropdown(
                        choices=pipeline_dirs,
                        value=pipeline_dirs[0] if pipeline_dirs else None,
                        label="Run directory",
                        scale=4,
                    )
                    quality_refresh_btn = gr.Button("Load / Refresh", scale=1)
                quality_summary = gr.Textbox(label="Summary", interactive=False, lines=8)
                quality_samples = gr.Dataframe(
                    headers=["prompt", "response", "distinct_1", "distinct_2", "max_rep", "judge_score"],
                    label="Samples",
                    wrap=True,
                )

                def load_quality(run_path):
                    if not run_path:
                        return "No run selected.", []
                    qfile = Path(run_path) / "quality_metrics.json"
                    if not qfile.exists():
                        return f"quality_metrics.json not found in {Path(run_path).name}", []
                    try:
                        with open(qfile) as f:
                            data = json.load(f)
                    except Exception as e:
                        return f"Error reading file: {e}", []

                    div = data.get("diversity", {})
                    judge = data.get("judge", {})
                    gates = data.get("quality_gates", {})
                    tppl = data.get("teacher_perplexity", {})
                    n_gen = data.get("n_samples_generated", data.get("n_samples", "?"))
                    n_pass = data.get("n_samples_passed", "?")
                    lines = [
                        f"Model:       {Path(data.get('model_dir', run_path)).name}",
                        f"Timestamp:   {data.get('timestamp', 'n/a')}",
                        f"Generated:   {n_gen}  |  Passed: {n_pass}"
                        + (f"  ({gates.get('pass_rate_pct', '?')}%)" if gates.get('pass_rate_pct') else ""),
                        "",
                        "── Quality Gates ───────────────────────",
                        f"  pass rate:      {gates.get('pass_rate_pct', 'n/a')}%",
                        f"  refusal rate:   {gates.get('refusal_rate_pct', 'n/a')}%",
                        "",
                        "── Diversity ───────────────────────────",
                        f"  avg distinct-1:   {div.get('avg_distinct_1', 'n/a')}",
                        f"  avg distinct-2:   {div.get('avg_distinct_2', 'n/a')}",
                        f"  avg max-rep:      {div.get('avg_max_rep', 'n/a')}",
                        f"  3-gram entropy:   {div.get('ngram_entropy_3', 'n/a')} bits",
                        f"  median length:    {div.get('median_response_tokens', 'n/a')} tokens",
                    ]
                    if tppl.get("enabled"):
                        lines += [
                            "",
                            "── Teacher PPL on student outputs ──────",
                            f"  avg teacher ppl:  {tppl.get('avg_teacher_ppl', 'n/a')}",
                        ]
                    if judge.get("enabled"):
                        lines += [
                            "",
                            "── LLM-as-judge ────────────────────────",
                            f"  teacher:    {judge.get('teacher', 'n/a')}",
                            f"  avg score:  {judge.get('avg_score', 'n/a')} / 10",
                            f"  scored:     {judge.get('n_scored', 0)} / {n_gen}",
                        ]
                    else:
                        lines.append("\n(Judge not run — rerun with --judge to enable)")

                    rows = []
                    for s in data.get("samples", []):
                        rows.append([
                            (s.get("instruction") or s.get("prompt", ""))[:120],
                            s.get("response", "")[:200],
                            s.get("distinct_1", ""),
                            s.get("distinct_2", ""),
                            s.get("max_rep", ""),
                            s.get("judge_score", ""),
                        ])
                    return "\n".join(lines), rows

                quality_run_dd.change(load_quality, quality_run_dd, [quality_summary, quality_samples])
                quality_refresh_btn.click(load_quality, quality_run_dd, [quality_summary, quality_samples])
                if pipeline_dirs:
                    _qs, _qr = load_quality(pipeline_dirs[0])
                    quality_summary.value = _qs

            with gr.Tab("Experiments"):
                gr.Markdown("### Experiment history (from experiment_log.jsonl)")
                gr.Markdown(
                    "Populated automatically by `run_distillation_agent.py --log_experiment`. "
                    "Run `python scripts/experiment_log.py --show 20` in a terminal for a quick view."
                )
                with gr.Row():
                    exp_log_path = gr.Textbox(
                        label="experiment_log.jsonl path",
                        value=str(Path(args.runs_dir).parent / "experiment_log.jsonl"),
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

                def load_experiments(log_path):
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
                                    except json.JSONDecodeError:
                                        continue
                    except OSError:
                        return [], None

                    rows = []
                    for r in records:
                        cfg = r.get("config", {})
                        m = r.get("metrics", {})

                        def _s(v, fmt=".2f"):
                            try:
                                return format(float(v), fmt)
                            except (TypeError, ValueError):
                                return ""

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

                    # Trend plot
                    fig = None
                    if len(records) >= 2:
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

                        ax = axes[ax_idx][0]
                        x = list(range(len(records)))
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

                    return rows, fig

                exp_refresh_btn.click(load_experiments, exp_log_path, [exp_table, exp_trend_plot])
                _default_exp = str(Path(args.runs_dir).parent / "experiment_log.jsonl")
                if Path(_default_exp).exists():
                    _er, _ef = load_experiments(_default_exp)
                    if _er:
                        exp_table.value = _er
                    if _ef:
                        exp_trend_plot.value = _ef

    app.queue()  # required for generator-based streaming in Run Evals tab
    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
