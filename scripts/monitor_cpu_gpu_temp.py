#!/usr/bin/env python3
"""
 HEAD
Monitor CPU/GPU temperature and power on Apple Silicon M3 Max.
Uses mactop (no sudo required).

Optional automatic fan control via Macs Fan Control CLI (if installed):
  Install: brew install --cask macs-fan-control
  The app must be running for fan control to take effect.

Usage:
  python scripts/monitor_cpu_gpu_temp.py
  python scripts/monitor_cpu_gpu_temp.py --interval 3 --log thermal.log
  python scripts/monitor_cpu_gpu_temp.py --fan-threshold 75 --fan-max-temp 90
"""

import argparse
import json
import os
import shutil
import subprocess
=======
Monitor CPU and GPU power (thermal proxy) on Apple Silicon M1/M2/M3.

Apple Silicon does not expose raw temperature. Power (mW) correlates with heat.
Requires: sudo powermetrics

Usage:
  sudo python scripts/monitor_cpu_gpu_temp.py
  sudo python scripts/monitor_cpu_gpu_temp.py --interval 2 --log monitor.log
"""

import argparse
import re
import subprocess
import sys
8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
import time
from datetime import datetime

 HEAD
# Path to Macs Fan Control CLI helper (set by the app on install)
MFC_CLI = "/Applications/Macs Fan Control.app/Contents/MacOS/Macs Fan Control"

# Resolve mactop binary — conda envs often don't include /opt/homebrew/bin in PATH
def _find_mactop() -> str | None:
    found = shutil.which("mactop")
    if found:
        return found
    for candidate in [
        "/opt/homebrew/bin/mactop",
        "/usr/local/bin/mactop",
        "/opt/homebrew/Cellar/mactop/2.0.9/bin/mactop",
    ]:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    # Scan all Cellar versions
    import glob
    for p in sorted(glob.glob("/opt/homebrew/Cellar/mactop/*/bin/mactop"), reverse=True):
        if os.access(p, os.X_OK):
            return p
    return None

MACTOP_BIN = _find_mactop()

# Fan RPM curve: (temp_c, rpm) pairs — interpolated linearly between points
FAN_CURVE = [
    (65,  1200),   # below 65°C: minimum (auto floor)
    (75,  2500),   # 75°C: start ramping
    (85,  4000),   # 85°C: high
    (90,  6000),   # 90°C: full speed
]


def sample_metrics():
    """Return dict with temps, power, and memory usage. Returns None on failure."""
    if MACTOP_BIN is None:
        return None
    try:
        proc = subprocess.run(
            [MACTOP_BIN, "--headless", "--format", "json", "--count", "1"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode != 0:
            return None
        data = json.loads(proc.stdout)
        m = data[0]["soc_metrics"]
        mem = data[0].get("memory", {})
        mem_used_gb  = mem.get("used",  0) / 1e9
        mem_total_gb = mem.get("total", 0) / 1e9
        return {
            "cpu_temp":    m.get("cpu_temp",    0.0),
            "gpu_temp":    m.get("gpu_temp",    0.0),
            "soc_temp":    m.get("soc_temp",    0.0),
            "cpu_power":   m.get("cpu_power",   0.0),
            "gpu_power":   m.get("gpu_power",   0.0),
            "total_power": m.get("total_power", 0.0),
            "mem_used_gb":  round(mem_used_gb,  1),
            "mem_total_gb": round(mem_total_gb, 1),
        }
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
            json.JSONDecodeError, KeyError, IndexError):
        return None


def mfc_available():
    return os.path.exists(MFC_CLI)


def interpolate_rpm(temp, curve):
    """Linear interpolation along the fan curve."""
    if temp <= curve[0][0]:
        return curve[0][1]
    if temp >= curve[-1][0]:
        return curve[-1][1]
    for (t0, r0), (t1, r1) in zip(curve, curve[1:]):
        if t0 <= temp <= t1:
            frac = (temp - t0) / (t1 - t0)
            return int(r0 + frac * (r1 - r0))
    return curve[-1][1]


_last_rpm = None

def apply_fan_speed(temp, threshold, max_temp):
    """
    Set fan speed via Macs Fan Control CLI when temp > threshold.
    Restores auto when temp drops 5°C below threshold (hysteresis).
    """
    global _last_rpm
    if not mfc_available():
        return None

    if temp < threshold - 5:
        # Cool enough — return to auto if we were controlling
        if _last_rpm is not None:
            try:
                subprocess.run([MFC_CLI, "--set-auto"], capture_output=True, timeout=5)
            except Exception:
                pass
            _last_rpm = None
        return "auto"

    if temp >= threshold:
        curve = [(threshold, FAN_CURVE[1][1]), (max_temp, FAN_CURVE[-1][1])]
        # Build full curve anchored to threshold
        rpm = interpolate_rpm(temp, FAN_CURVE)
        if rpm != _last_rpm:
            try:
                subprocess.run(
                    [MFC_CLI, "--set-rpm", str(rpm)],
                    capture_output=True, timeout=5,
                )
                _last_rpm = rpm
            except Exception:
                return None
        return rpm
    return _last_rpm


def main():
    ap = argparse.ArgumentParser(description="Monitor CPU/GPU temperature on Apple Silicon")
    ap.add_argument("--interval", "-i", type=float, default=3.0, help="Sample interval (seconds)")
    ap.add_argument("--log", type=str, default=None, help="Append CSV samples to file")
    ap.add_argument("--fan-threshold", type=float, default=75.0,
                    help="GPU temp (°C) above which to start ramping fans (default: 75)")
    ap.add_argument("--fan-max-temp", type=float, default=90.0,
                    help="GPU temp (°C) at which fans hit maximum (default: 90)")
    ap.add_argument("--fan-control", action="store_true",
                    help="Enable automatic fan control via Macs Fan Control (requires admin auth)")
    args = ap.parse_args()

    if MACTOP_BIN is None:
        print("ERROR: mactop not found. Install with:  brew install mactop")
        print("       Then re-run this script.")
        return
    print(f"mactop: {MACTOP_BIN}")

    fan_control = args.fan_control
    if fan_control and mfc_available():
        print(f"Fan control: active (threshold={args.fan_threshold}°C, max={args.fan_max_temp}°C)")
    elif fan_control:
        print("Fan control: Macs Fan Control not found — install with:")
        print("  brew install --cask macs-fan-control")
        fan_control = False

    header = (f"{'time':<20}  {'CPU°C':>6}  {'GPU°C':>6}  {'SOC°C':>6}"
              f"  {'CPU(W)':>7}  {'GPU(W)':>7}  {'Total(W)':>9}  {'RAM':>10}")
    if fan_control:
        header += f"  {'Fan':>8}"
    print(header)
    print("-" * len(header))

    log_file = open(args.log, "a") if args.log else None
    if log_file and os.path.getsize(args.log) == 0:
        log_file.write("time,cpu_temp_c,gpu_temp_c,soc_temp_c,cpu_power_w,gpu_power_w,total_power_w,mem_used_gb,mem_total_gb\n")

    try:
        while True:
            m = sample_metrics()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if m:
                fan_str = ""
                if fan_control:
                    hot = max(m["cpu_temp"], m["gpu_temp"])
                    fan_result = apply_fan_speed(hot, args.fan_threshold, args.fan_max_temp)
                    if fan_result == "auto":
                        fan_str = f"  {'auto':>8}"
                    elif fan_result is not None:
                        fan_str = f"  {fan_result:>6} rpm"
                    else:
                        fan_str = f"  {'---':>8}"

                mem_str = f"{m['mem_used_gb']:.1f}/{m['mem_total_gb']:.0f}GB"
                row = (f"{ts:<20}  {m['cpu_temp']:>6.1f}  {m['gpu_temp']:>6.1f}  "
                       f"{m['soc_temp']:>6.1f}  {m['cpu_power']:>7.2f}  "
                       f"{m['gpu_power']:>7.2f}  {m['total_power']:>9.2f}  {mem_str:>10}{fan_str}")
                print(row)
                if log_file:
                    log_file.write(
                        f"{ts},{m['cpu_temp']:.1f},{m['gpu_temp']:.1f},"
                        f"{m['soc_temp']:.1f},{m['cpu_power']:.2f},"
                        f"{m['gpu_power']:.2f},{m['total_power']:.2f},"
                        f"{m['mem_used_gb']:.1f},{m['mem_total_gb']:.0f}\n"
                    )
                    log_file.flush()
            else:
                print(f"{ts:<20}  [failed to read metrics]")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        if fan_control and _last_rpm is not None:
            subprocess.run([MFC_CLI, "--set-auto"], capture_output=True, timeout=5)
=======
def sample_power():
    """Return (cpu_mw, gpu_mw, combined_mw) or (None, None, None).
    Calls powermetrics via sudo -n (requires NOPASSWD entry for /usr/bin/powermetrics).
    """
    try:
        proc = subprocess.run(
            ["/usr/bin/powermetrics", "-n", "1", "-i", "500",
             "--samplers", "cpu_power,gpu_power"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return None, None, None
        out = proc.stdout
        cpu = re.search(r"CPU Power:\s*(\d+)\s*mW", out)
        gpu = re.search(r"GPU Power:\s*(\d+)\s*mW", out)
        combined = re.search(r"Combined Power.*?(\d+)\s*mW", out, re.DOTALL)
        return (
            int(cpu.group(1)) if cpu else None,
            int(gpu.group(1)) if gpu else None,
            int(combined.group(1)) if combined else None,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None, None, None


def main():
    ap = argparse.ArgumentParser(description="Monitor CPU/GPU power on Apple Silicon")
    ap.add_argument("--interval", "-i", type=float, default=2.0, help="Sample interval (seconds)")
    ap.add_argument("--log", type=str, default=None, help="Append samples to file")
    ap.add_argument("--header", action="store_true", default=True, help="Print header (default: True)")
    args = ap.parse_args()

    if args.header:
        print("time                   CPU (mW)   GPU (mW)   Combined (mW)")
        print("-" * 55)

    log_file = open(args.log, "a") if args.log else None
    if log_file and args.header:
        log_file.write("time,cpu_mw,gpu_mw,combined_mw\n")

    try:
        while True:
            cpu, gpu, combined = sample_power()
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            row = f"{ts}   {cpu or '—':>8}   {gpu or '—':>8}   {combined or '—':>10}"
            print(row)

            if log_file:
                log_file.write(f"{ts},{cpu or ''},{gpu or ''},{combined or ''}\n")
                log_file.flush()

            time.sleep(args.interval)
    except KeyboardInterrupt:
8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
        print("\nStopped.")
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
