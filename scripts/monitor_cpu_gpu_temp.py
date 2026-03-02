#!/usr/bin/env python3
"""
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
import time
from datetime import datetime


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
        print("\nStopped.")
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
