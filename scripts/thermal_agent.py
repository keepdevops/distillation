#!/usr/bin/env python3
"""
Autonomous thermal control agent - system-wide hardware protection.

Continuously monitors CPU/GPU/SoC temperatures and pauses ALL running jobs when
thermal limits are exceeded. Resumes jobs automatically when temps drop.

Key features:
- System-wide: Protects all jobs (training, inference, export, benchmarks)
- Multi-job: Watches multiple directories simultaneously
- Auto-resume: Clears pause.flag when temps return to safe levels
- LaunchAgent-ready: Runs as macOS background service
- Standalone: No coupling to training logic

Architecture:
- Thermal agent (this script): Hardware monitoring only
- Training watchdog (training_watchdog.py): ML-specific monitoring (plateau detection)
- PauseFlagCallback (watchdog_callbacks.py): Responds to pause.flag in training loops

Usage:
    # Watch single job
    python scripts/thermal_agent.py --watch ./distilled-minillm

    # Watch multiple jobs (system-wide protection)
    python scripts/thermal_agent.py --watch ./distilled-minillm ./distilled-mlx ./distilled-sft

    # Custom threshold (default: 85°C)
    python scripts/thermal_agent.py --watch . --threshold 70

    # Daemon mode (background process)
    python scripts/thermal_agent.py --daemon --watch . --threshold 85

    # LaunchAgent (always-on, survives reboots)
    ./scripts/install_thermal_agent.sh
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional
import threading

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ThermalAgent:
    """Autonomous thermal monitoring and control agent."""

    def __init__(
        self,
        watch_dirs: List[str],
        threshold: float = 85.0,
        metric: str = "soc_temp_c",
        hysteresis: float = 5.0,
        interval: int = 30,
        log_file: Optional[str] = None,
        show_popup: bool = False,
    ):
        """
        Initialize thermal agent.

        Args:
            watch_dirs: List of output directories to monitor
            threshold: Temperature threshold in °C (default: 85°C)
            metric: Metric to monitor (soc_temp_c, cpu_temp_c, gpu_temp_c, total_power_w)
            hysteresis: Temperature delta for auto-resume (default: 5°C)
            interval: Poll interval in seconds (default: 30s)
            log_file: Optional log file for thermal events
            show_popup: Show fan control GUI popup on high temps (default: False)
        """
        self.watch_dirs = [Path(d).resolve() for d in watch_dirs]
        self.threshold = threshold
        self.metric = metric
        self.hysteresis = hysteresis
        self.resume_threshold = threshold - hysteresis
        self.interval = interval
        self.log_file = Path(log_file) if log_file else None
        self.show_popup = show_popup
        self.paused_jobs = set()  # Track which jobs we've paused
        self.running = True
        self.popup_shown = False  # Track if popup is already open

        # Create watch directories if they don't exist
        for d in self.watch_dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Create log file if specified
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_event("agent_started", {
                "watch_dirs": [str(d) for d in self.watch_dirs],
                "threshold": self.threshold,
                "metric": self.metric,
                "hysteresis": self.hysteresis,
            })

    def _log_event(self, event_type: str, data: Dict):
        """Append thermal event to log file."""
        if not self.log_file:
            return
        try:
            event = {
                "timestamp": time.time(),
                "event": event_type,
                **data,
            }
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.warning("Failed to write thermal log: %s", e)

    def read_thermal(self) -> Optional[float]:
        """
        Read current thermal metric using mactop (Apple Silicon).

        Returns:
            Temperature in °C or power in W, or None if unavailable
        """
        key_map = {
            "soc_temp_c": "soc_temp",
            "cpu_temp_c": "cpu_temp",
            "gpu_temp_c": "gpu_temp",
            "cpu_power_w": "cpu_power",
            "gpu_power_w": "gpu_power",
            "total_power_w": "total_power",
        }
        mactop_key = key_map.get(self.metric, "soc_temp")

        try:
            proc = subprocess.run(
                ["mactop", "--headless", "--format", "json", "--count", "1"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode != 0:
                return None

            data = json.loads(proc.stdout)
            value = data[0]["soc_metrics"].get(mactop_key)
            return value
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            OSError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
        ) as e:
            logger.debug("read_thermal failed: %s", e)
            return None

    def set_pause_flag(self, job_dir: Path, temp: float):
        """Write pause.flag to job directory."""
        pause_path = job_dir / "pause.flag"
        if pause_path.exists():
            return  # Already paused

        payload = {
            "reason": "thermal",
            "value": temp,
            "metric": self.metric,
            "threshold": self.threshold,
            "agent": "thermal_agent",
        }

        try:
            pause_path.write_text(json.dumps(payload, indent=2))
            self.paused_jobs.add(job_dir)
            logger.warning(
                "🔥 THERMAL PAUSE: %s=%.1f°C >= %.1f°C — paused %s",
                self.metric,
                temp,
                self.threshold,
                job_dir.name,
            )
            self._log_event("job_paused", {
                "job_dir": str(job_dir),
                "temperature": temp,
                "threshold": self.threshold,
            })
        except OSError as e:
            logger.error("Failed to write pause.flag: %s", e)

    def clear_pause_flag(self, job_dir: Path, temp: float):
        """Remove pause.flag from job directory if we created it."""
        pause_path = job_dir / "pause.flag"
        if not pause_path.exists():
            return

        # Only clear flags we created (thermal agent)
        try:
            content = json.loads(pause_path.read_text())
            if content.get("agent") != "thermal_agent":
                # Flag created by training_watchdog (plateau) - don't touch
                return
        except (json.JSONDecodeError, OSError):
            # Can't verify ownership - be conservative, don't clear
            return

        try:
            pause_path.unlink()
            if job_dir in self.paused_jobs:
                self.paused_jobs.remove(job_dir)
            logger.info(
                "❄️  THERMAL RESUME: %s=%.1f°C < %.1f°C — resumed %s",
                self.metric,
                temp,
                self.resume_threshold,
                job_dir.name,
            )
            self._log_event("job_resumed", {
                "job_dir": str(job_dir),
                "temperature": temp,
                "resume_threshold": self.resume_threshold,
            })
        except OSError as e:
            logger.error("Failed to clear pause.flag: %s", e)

    def show_fan_control_popup(self):
        """Launch fan control GUI popup in background thread."""
        if self.popup_shown:
            return  # Popup already open

        def run_popup():
            try:
                script_dir = Path(__file__).parent
                popup_script = script_dir / "fan_control_popup.py"
                if not popup_script.exists():
                    logger.warning("fan_control_popup.py not found at: %s", popup_script)
                    return

                subprocess.run(
                    [sys.executable, str(popup_script), "--threshold", str(self.threshold)],
                    check=False,
                )
                self.popup_shown = False  # Reset when popup closes
            except Exception as e:
                logger.error("Failed to launch fan control popup: %s", e)
                self.popup_shown = False

        self.popup_shown = True
        thread = threading.Thread(target=run_popup, daemon=True)
        thread.start()

    def tick(self):
        """Run one monitoring cycle."""
        temp = self.read_thermal()

        if temp is None:
            logger.debug("Thermal reading unavailable (mactop not installed?)")
            return

        # Check if we should pause jobs
        if temp >= self.threshold:
            for job_dir in self.watch_dirs:
                self.set_pause_flag(job_dir, temp)

            # Show fan control popup if enabled
            if self.show_popup and not self.popup_shown:
                logger.info("🌡️  Launching fan control popup (temp=%.1f°C)", temp)
                self.show_fan_control_popup()

            return

        # Check if we should resume jobs
        if temp < self.resume_threshold:
            for job_dir in self.watch_dirs:
                self.clear_pause_flag(job_dir, temp)
            return

        # Normal operation - log temp periodically
        paused_count = len(self.paused_jobs)
        if paused_count > 0:
            logger.info(
                "Thermal: %.1f°C (paused: %d jobs, resume at %.1f°C)",
                temp,
                paused_count,
                self.resume_threshold,
            )
        else:
            logger.debug("Thermal: %.1f°C (ok)", temp)

    def run(self):
        """Main monitoring loop."""
        logger.info("Thermal agent started")
        logger.info("  Watching: %s", ", ".join(d.name for d in self.watch_dirs))
        logger.info("  Threshold: %.1f°C (resume at %.1f°C)", self.threshold, self.resume_threshold)
        logger.info("  Metric: %s", self.metric)
        logger.info("  Interval: %ds", self.interval)

        if self.log_file:
            logger.info("  Log file: %s", self.log_file)

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        try:
            while self.running:
                self.tick()
                time.sleep(self.interval)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt, shutting down...")
        finally:
            self.shutdown()

    def _handle_signal(self, signum, frame):
        """Handle termination signals gracefully."""
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False

    def shutdown(self):
        """Clean shutdown - clear thermal pause flags."""
        logger.info("Shutting down thermal agent")

        # Clear any thermal pause flags we created
        for job_dir in list(self.paused_jobs):
            pause_path = job_dir / "pause.flag"
            try:
                content = json.loads(pause_path.read_text())
                if content.get("agent") == "thermal_agent":
                    pause_path.unlink()
                    logger.info("Cleared thermal pause flag: %s", job_dir.name)
            except (json.JSONDecodeError, OSError, FileNotFoundError):
                pass

        if self.log_file:
            self._log_event("agent_stopped", {})

        logger.info("Thermal agent stopped")


def daemonize():
    """Fork process to run as daemon (UNIX only)."""
    try:
        pid = os.fork()
        if pid > 0:
            # Parent process - exit
            sys.exit(0)
    except OSError as e:
        logger.error("Fork failed: %s", e)
        sys.exit(1)

    # Decouple from parent environment
    os.chdir("/")
    os.setsid()
    os.umask(0)

    # Second fork
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error("Fork failed: %s", e)
        sys.exit(1)

    # Redirect standard file descriptors
    sys.stdout.flush()
    sys.stderr.flush()
    si = open(os.devnull, "r")
    so = open(os.devnull, "a+")
    se = open(os.devnull, "a+")
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())


def main():
    parser = argparse.ArgumentParser(
        description="Autonomous thermal control agent for Apple Silicon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Watch single job
  python scripts/thermal_agent.py --watch ./distilled-minillm

  # Watch multiple jobs (system-wide protection)
  python scripts/thermal_agent.py --watch ./distilled-minillm ./distilled-mlx

  # Custom threshold and interval
  python scripts/thermal_agent.py --watch . --threshold 70 --interval 15

  # Daemon mode (background process)
  python scripts/thermal_agent.py --daemon --watch . --log thermal_agent.jsonl

Metrics:
  soc_temp_c     - System-on-Chip temperature (default)
  cpu_temp_c     - CPU cores temperature
  gpu_temp_c     - GPU cores temperature
  total_power_w  - Total power consumption (Watts)
        """,
    )

    parser.add_argument(
        "--watch",
        nargs="+",
        required=True,
        help="Output directories to monitor (can specify multiple)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=85.0,
        help="Temperature threshold in °C (default: 85°C)",
    )
    parser.add_argument(
        "--metric",
        choices=["soc_temp_c", "cpu_temp_c", "gpu_temp_c", "total_power_w"],
        default="soc_temp_c",
        help="Metric to monitor (default: soc_temp_c)",
    )
    parser.add_argument(
        "--hysteresis",
        type=float,
        default=5.0,
        help="Temperature delta for auto-resume (default: 5°C)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Poll interval in seconds (default: 30)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file for thermal events (JSONL format)",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run as background daemon (UNIX only)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one check and exit (for testing)",
    )
    parser.add_argument(
        "--show-popup",
        action="store_true",
        help="Show fan control GUI popup when temperature exceeds threshold",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.threshold <= 0:
        logger.error("Threshold must be positive")
        sys.exit(1)

    if args.interval <= 0:
        logger.error("Interval must be positive")
        sys.exit(1)

    # Daemonize if requested
    if args.daemon and not args.once:
        logger.info("Starting thermal agent as daemon...")
        daemonize()

    # Create agent
    agent = ThermalAgent(
        watch_dirs=args.watch,
        threshold=args.threshold,
        metric=args.metric,
        hysteresis=args.hysteresis,
        interval=args.interval,
        log_file=args.log,
        show_popup=args.show_popup,
    )

    # Run once or continuous
    if args.once:
        agent.tick()
    else:
        agent.run()


if __name__ == "__main__":
    main()
