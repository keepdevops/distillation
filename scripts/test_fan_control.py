#!/usr/bin/env python3
"""
Test and diagnose Macs Fan Control integration.
"""

import os
import subprocess
import sys

MFC_CLI = "/Applications/Macs Fan Control.app/Contents/MacOS/Macs Fan Control"

def check_mfc_installed():
    """Check if Macs Fan Control app is installed."""
    if os.path.exists(MFC_CLI):
        print("✓ Macs Fan Control app is installed")
        return True
    else:
        print("✗ Macs Fan Control NOT found at:", MFC_CLI)
        print("  Install with: brew install --cask macs-fan-control")
        return False

def check_mfc_running():
    """Check if Macs Fan Control app is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "Macs Fan Control"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("✓ Macs Fan Control app is running")
            return True
        else:
            print("✗ Macs Fan Control app is NOT running")
            print("  Launch it from Applications folder first!")
            return False
    except Exception as e:
        print(f"✗ Could not check if app is running: {e}")
        return False

def test_cli_commands():
    """Test if CLI commands work."""
    print("\nTesting CLI commands...")

    # Test getting current fan info
    try:
        result = subprocess.run(
            [MFC_CLI, "--help"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("✓ CLI responds to --help")
        else:
            print("✗ CLI --help failed:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ CLI command failed: {e}")
        return False

    # Test setting a safe RPM (2000 is very conservative)
    print("\nAttempting to set fan to 2000 RPM (safe test)...")
    try:
        result = subprocess.run(
            [MFC_CLI, "--set-rpm", "2000"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            print("✓ Successfully set fan to 2000 RPM")
            print("  (You should hear fans spin up slightly)")

            # Restore to auto after 3 seconds
            import time
            time.sleep(3)
            result = subprocess.run(
                [MFC_CLI, "--set-auto"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                print("✓ Successfully restored to auto mode")
            return True
        else:
            print("✗ Failed to set RPM:", result.stderr)
            return False
    except Exception as e:
        print(f"✗ RPM command failed: {e}")
        return False

def show_current_temps():
    """Show current temperatures from mactop."""
    print("\nCurrent temperatures:")
    try:
        import json
        result = subprocess.run(
            ["mactop", "--headless", "--format", "json", "--count", "1"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            m = data[0]["soc_metrics"]
            print(f"  CPU: {m.get('cpu_temp', 'N/A'):.1f}°C")
            print(f"  GPU: {m.get('gpu_temp', 'N/A'):.1f}°C")
            print(f"  SOC: {m.get('soc_temp', 'N/A'):.1f}°C")
            print(f"  Total Power: {m.get('total_power', 'N/A'):.1f}W")

            max_temp = max(m.get('cpu_temp', 0), m.get('gpu_temp', 0))
            print(f"\n  Max temp: {max_temp:.1f}°C")

            if max_temp < 55:
                print("  → Temps are low, fan control won't activate yet")
                print("    (Default threshold is 60°C)")
            elif max_temp < 60:
                print("  → Close to threshold (60°C)")
            else:
                print("  → Above threshold! Fans should be ramping up")
        else:
            print("  ✗ Could not read temps from mactop")
    except Exception as e:
        print(f"  ✗ Failed to get temps: {e}")

def main():
    print("=" * 70)
    print("Macs Fan Control Diagnostic Test")
    print("=" * 70)

    # Check installation
    if not check_mfc_installed():
        print("\nFIX: Install Macs Fan Control:")
        print("  brew install --cask macs-fan-control")
        return 1

    # Check if app is running
    if not check_mfc_running():
        print("\nFIX: Launch the Macs Fan Control app from Applications")
        print("     The app must be running for CLI commands to work!")
        return 1

    # Test CLI
    if not test_cli_commands():
        print("\nFIX: Make sure Macs Fan Control app is running and try again")
        return 1

    # Show temps
    show_current_temps()

    print("\n" + "=" * 70)
    print("✓ All checks passed! Fan control should work.")
    print("=" * 70)
    print("\nTo enable fan control during monitoring, run:")
    print("  python scripts/monitor_cpu_gpu_temp.py --fan-control")
    print("\nOr with custom threshold:")
    print("  python scripts/monitor_cpu_gpu_temp.py --fan-control --fan-threshold 55")
    print("\nDefault threshold is now 60°C (fans activate when chip hits 60°C)")
    print("\nNOTE: The app must stay running while monitoring for fan control to work.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
