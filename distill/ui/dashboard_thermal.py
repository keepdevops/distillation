"""
Dashboard thermal tab: CPU/GPU temperature and fan control widgets.
"""
import json
import logging
import os
import subprocess
from pathlib import Path

import gradio as gr

from distill.infra.config import cfg
from .dashboard_thermal_log import load_thermal

logger = logging.getLogger(__name__)

_MFC_CLI = "/Applications/Macs Fan Control.app/Contents/MacOS/Macs Fan Control"


# ── Parsing helpers ────────────────────────────────────────────────────────

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
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            logger.error("Failed to parse mactop line: %s", exc)
            continue
    return {}


def _fmt_t(v):
    try:
        return f"{float(v):.1f}°C"
    except (TypeError, ValueError):
        return str(v)


def _fmt_w(v):
    try:
        return f"{float(v):.1f} W"
    except (TypeError, ValueError):
        return str(v)


# ── Live mactop reading ────────────────────────────────────────────────────

def read_mactop():
    _mactop_bin = cfg.paths.mactop_bin
    _pause_threshold = cfg.thermal.threshold_celsius
    try:
        result = subprocess.run(
            [_mactop_bin, "--headless", "--format", "json", "--count", "1"],
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
                if float(val) >= _pause_threshold:
                    lines.append(f"⚠ {label} >= {_pause_threshold:.0f}°C — watchdog pause threshold")
            except (TypeError, ValueError):
                logger.error("read_mactop: non-numeric temp value for %s: %r", label, val)
        return "\n".join(lines)
    except FileNotFoundError:
        logger.error("read_mactop: mactop not found at %r", _mactop_bin)
        return f"mactop not found at {_mactop_bin!r}"
    except Exception as exc:
        logger.error("read_mactop error: %s", exc)
        return f"Error: {exc}"


# ── Fan control helpers ────────────────────────────────────────────────────

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
    except Exception as exc:
        logger.error("_mfc_launch failed: %s", exc)
        return f"Failed: {exc}"


def _mfc_cmd(*args):
    """Run a Macs Fan Control CLI command, return feedback string."""
    if not os.path.exists(_MFC_CLI):
        return "Macs Fan Control not installed"
    r = subprocess.run([_MFC_CLI, *args],
                       capture_output=True, text=True, timeout=5)
    return r.stdout.strip() or r.stderr.strip() or "Done"


# ── Preset helpers ─────────────────────────────────────────────────────────

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
    except Exception as exc:
        logger.error("_read_active_preset decode error: %s", exc)
        return f"Raw: {val}\n(decode error: {exc})"


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
        ["osascript", "-e", 'tell application "Macs Fan Control" to activate'],
        capture_output=True, timeout=3,
    )
    return _read_active_preset()


def set_auto():
    return _write_preset("Auto", (0, "", 0, 0), (0, "", 0, 0))


def set_custom(v):
    rpm = int(v)
    return _write_preset(f"Custom {rpm} RPM", (1, "", rpm, 0), (1, "", rpm, 0))


# ── Tab builder ────────────────────────────────────────────────────────────

def build_thermal_tab(runs_dir):
    """Build and wire the Thermal tab. Must be called inside a gr.Tab context."""
    gr.Markdown("### CPU / GPU temperature & power over time")

    mactop_btn = gr.Button("🌡 Read Now  (mactop)", variant="secondary")
    mactop_out = gr.Textbox(label="Live reading", interactive=False, lines=3)
    mactop_btn.click(fn=read_mactop, inputs=[], outputs=mactop_out)

    gr.Markdown("---")
    gr.Markdown("#### Fan Control  (Macs Fan Control)")

    with gr.Row():
        fan_status_btn = gr.Button("Check Status", variant="secondary", scale=2)
        fan_launch_btn = gr.Button("Launch App", variant="secondary", scale=1)
    fan_feedback = gr.Textbox(label="Status", interactive=False, lines=1)

    fan_status_btn.click(fn=_mfc_status, inputs=[], outputs=fan_feedback)
    fan_launch_btn.click(fn=_mfc_launch, inputs=[], outputs=fan_feedback)

    with gr.Row():
        fan_read_btn = gr.Button("📋 Read Current Preset", variant="secondary", scale=3)
    fan_current = gr.Textbox(label="Active preset", interactive=False, lines=3)
    fan_read_btn.click(fn=_read_active_preset, inputs=[], outputs=fan_current)

    gr.Markdown("**Sensor-based presets**  _(both fans, GPU Clusters Average)_")
    gr.Markdown("_Left side: 1350–5349 RPM  ·  Right side: 1458–5777 RPM_")

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

    gr.Markdown("**Constant RPM**  _(shared safe range for both fans)_")
    with gr.Row():
        fan_slider = gr.Slider(
            1458, 5349, value=3000, step=50,
            label="RPM  (Left 1350–5349  ·  Right 1458–5777)",
            scale=4,
        )
        fan_apply_btn = gr.Button("Apply RPM", variant="primary", scale=1)
    fan_apply_btn.click(fn=set_custom, inputs=[fan_slider], outputs=fan_feedback)

    gr.Markdown("---")
    thermal_log_box = gr.Textbox(
        label="Log file path",
        value=str(Path(runs_dir).parent / "thermal.log"),
        placeholder="/path/to/thermal.log",
    )
    thermal_refresh_btn = gr.Button("Load / Refresh")
    thermal_plot = gr.Plot(label="Thermal history")
    thermal_refresh_btn.click(load_thermal, thermal_log_box, thermal_plot)
