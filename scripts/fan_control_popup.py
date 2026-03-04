#!/usr/bin/env python3
"""
GUI popup for manual fan speed control.

Displays current CPU/GPU temperatures and allows adjusting fan speeds
via Macs Fan Control CLI.

Usage:
    python scripts/fan_control_popup.py
    python scripts/fan_control_popup.py --threshold 75  # Show warning if temp > 75°C
"""

import argparse
import json
import os
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from pathlib import Path

# Path to Macs Fan Control CLI
MFC_CLI = "/Applications/Macs Fan Control.app/Contents/MacOS/Macs Fan Control"

# Find mactop binary
def _find_mactop():
    import shutil
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
    import glob
    for p in sorted(glob.glob("/opt/homebrew/Cellar/mactop/*/bin/mactop"), reverse=True):
        if os.access(p, os.X_OK):
            return p
    return None

MACTOP_BIN = _find_mactop()


def is_mfc_app_running():
    """Check if Macs Fan Control app is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "Macs Fan Control.app"],
            capture_output=True,
            timeout=2
        )
        return result.returncode == 0
    except Exception:
        return False


def launch_mfc_app():
    """Launch Macs Fan Control app in background."""
    try:
        subprocess.Popen(
            ["open", "-a", "Macs Fan Control"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return True
    except Exception as e:
        print(f"Failed to launch Macs Fan Control: {e}")
        return False


class FanControlGUI:
    """GUI for manual fan speed control."""

    def __init__(self, master, warning_threshold=None):
        self.master = master
        self.warning_threshold = warning_threshold
        self.master.title("Fan Control")
        self.master.geometry("400x350")
        self.master.resizable(False, False)

        # Check if Macs Fan Control is available
        self.mfc_available = os.path.exists(MFC_CLI)

        # Auto-launch Macs Fan Control if installed but not running
        if self.mfc_available and not is_mfc_app_running():
            print("Launching Macs Fan Control app...")
            if launch_mfc_app():
                import time
                time.sleep(2)  # Give app time to start
                print("Macs Fan Control launched successfully")
            else:
                print("Failed to launch Macs Fan Control - fan controls may not work")

        # Temperature display
        temp_frame = ttk.LabelFrame(master, text="Current Temperatures", padding=10)
        temp_frame.pack(fill=tk.BOTH, padx=10, pady=10)

        self.cpu_label = ttk.Label(temp_frame, text="CPU: --°C", font=("Helvetica", 12))
        self.cpu_label.pack(anchor=tk.W)

        self.gpu_label = ttk.Label(temp_frame, text="GPU: --°C", font=("Helvetica", 12))
        self.gpu_label.pack(anchor=tk.W)

        self.soc_label = ttk.Label(temp_frame, text="SOC: --°C", font=("Helvetica", 12))
        self.soc_label.pack(anchor=tk.W)

        self.power_label = ttk.Label(temp_frame, text="Power: --W", font=("Helvetica", 10))
        self.power_label.pack(anchor=tk.W, pady=(5, 0))

        # Warning label (hidden by default)
        self.warning_label = ttk.Label(
            temp_frame,
            text="⚠️ HIGH TEMPERATURE",
            foreground="red",
            font=("Helvetica", 12, "bold")
        )

        # Fan control
        if self.mfc_available:
            fan_frame = ttk.LabelFrame(master, text="Fan Speed Control", padding=10)
            fan_frame.pack(fill=tk.BOTH, padx=10, pady=10)

            # Status indicator
            mfc_running = is_mfc_app_running()
            status_text = "✓ Macs Fan Control: Running" if mfc_running else "⚠ Macs Fan Control: Not Running"
            status_color = "green" if mfc_running else "orange"
            self.status_label = ttk.Label(
                fan_frame,
                text=status_text,
                foreground=status_color,
                font=("Helvetica", 9)
            )
            self.status_label.pack(pady=(0, 5))

            # Current fan speed display
            self.fan_label = ttk.Label(fan_frame, text="Current: Auto", font=("Helvetica", 10))
            self.fan_label.pack(pady=(0, 10))

            # Slider label (create before slider to avoid AttributeError)
            self.slider_label = ttk.Label(fan_frame, text="1200 RPM", font=("Helvetica", 9))

            # Fan speed slider
            self.fan_slider = ttk.Scale(
                fan_frame,
                from_=1200,
                to=6000,
                orient=tk.HORIZONTAL,
                command=self.on_slider_change
            )
            self.fan_slider.pack(fill=tk.X, pady=5)
            self.fan_slider.set(1200)

            self.slider_label.pack()

            # Preset buttons
            button_frame = ttk.Frame(fan_frame)
            button_frame.pack(pady=10)

            ttk.Button(button_frame, text="Auto", command=self.set_auto).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Low (2000)", command=lambda: self.set_rpm(2000)).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Med (3500)", command=lambda: self.set_rpm(3500)).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="High (5000)", command=lambda: self.set_rpm(5000)).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Max (6000)", command=lambda: self.set_rpm(6000)).pack(side=tk.LEFT, padx=2)

            # Apply button
            ttk.Button(
                fan_frame,
                text="Apply Custom RPM",
                command=self.apply_custom_rpm
            ).pack(pady=5)
        else:
            error_frame = ttk.LabelFrame(master, text="Fan Control", padding=10)
            error_frame.pack(fill=tk.BOTH, padx=10, pady=10)
            ttk.Label(
                error_frame,
                text="Macs Fan Control not found\nInstall from: https://crystalidea.com/macs-fan-control",
                justify=tk.CENTER
            ).pack()

        # Close button
        ttk.Button(master, text="Close", command=master.quit).pack(pady=10)

        # Update temperatures
        self.update_temps()

    def get_temps(self):
        """Get current temperatures from mactop."""
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
            return {
                "cpu_temp": m.get("cpu_temp", 0.0),
                "gpu_temp": m.get("gpu_temp", 0.0),
                "soc_temp": m.get("soc_temp", 0.0),
                "total_power": m.get("total_power", 0.0),
            }
        except Exception:
            return None

    def update_temps(self):
        """Update temperature display."""
        temps = self.get_temps()
        if temps:
            self.cpu_label.config(text=f"CPU: {temps['cpu_temp']:.1f}°C")
            self.gpu_label.config(text=f"GPU: {temps['gpu_temp']:.1f}°C")
            self.soc_label.config(text=f"SOC: {temps['soc_temp']:.1f}°C")
            self.power_label.config(text=f"Power: {temps['total_power']:.1f}W")

            # Show warning if any temp exceeds threshold
            if self.warning_threshold:
                max_temp = max(temps['cpu_temp'], temps['gpu_temp'], temps['soc_temp'])
                if max_temp >= self.warning_threshold:
                    self.warning_label.pack(pady=(5, 0))
                else:
                    self.warning_label.pack_forget()

        # Schedule next update
        self.master.after(3000, self.update_temps)

    def on_slider_change(self, value):
        """Update slider label when slider moves."""
        rpm = int(float(value))
        self.slider_label.config(text=f"{rpm} RPM")

    def set_rpm(self, rpm):
        """Set fan speed to specific RPM."""
        if not self.mfc_available:
            return
        try:
            subprocess.run(
                [MFC_CLI, "--set-rpm", str(rpm)],
                capture_output=True,
                timeout=5,
            )
            self.fan_label.config(text=f"Current: {rpm} RPM")
            self.fan_slider.set(rpm)
        except Exception as e:
            print(f"Failed to set fan speed: {e}")

    def set_auto(self):
        """Return fan control to auto mode."""
        if not self.mfc_available:
            return
        try:
            subprocess.run(
                [MFC_CLI, "--set-auto"],
                capture_output=True,
                timeout=5,
            )
            self.fan_label.config(text="Current: Auto")
        except Exception as e:
            print(f"Failed to set auto mode: {e}")

    def apply_custom_rpm(self):
        """Apply the current slider value."""
        rpm = int(self.fan_slider.get())
        self.set_rpm(rpm)


def main():
    parser = argparse.ArgumentParser(description="Fan control GUI")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Temperature threshold for warning (°C)",
    )
    args = parser.parse_args()

    if MACTOP_BIN is None:
        print("ERROR: mactop not found. Install with: brew install mactop")
        sys.exit(1)

    root = tk.Tk()
    app = FanControlGUI(root, warning_threshold=args.threshold)
    root.mainloop()


if __name__ == "__main__":
    main()
