"""ThermalAgent class — autonomous thermal monitoring and control."""
import json
import logging
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Dict, Optional

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
        self.watch_dirs = [Path(d).resolve() for d in watch_dirs]
        self.threshold = threshold
        self.metric = metric
        self.hysteresis = hysteresis
        self.resume_threshold = threshold - hysteresis
        self.interval = interval
        self.log_file = Path(log_file) if log_file else None
        self.show_popup = show_popup
        self.paused_jobs = set()
        self.running = True
        self.popup_shown = False

        for d in self.watch_dirs:
            d.mkdir(parents=True, exist_ok=True)

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
            event = {"timestamp": time.time(), "event": event_type, **data}
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except OSError as e:
            logger.warning("Failed to write thermal log: %s", e)

    def read_thermal(self) -> Optional[float]:
        """Read current thermal metric using mactop (Apple Silicon)."""
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
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode != 0:
                return None
            data = json.loads(proc.stdout)
            return data[0]["soc_metrics"].get(mactop_key)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError,
                json.JSONDecodeError, KeyError, IndexError) as e:
            logger.debug("read_thermal failed: %s", e)
            return None

    def set_pause_flag(self, job_dir: Path, temp: float):
        """Write pause.flag to job directory."""
        pause_path = job_dir / "pause.flag"
        if pause_path.exists():
            return
        payload = {
            "reason": "thermal", "value": temp,
            "metric": self.metric, "threshold": self.threshold,
            "agent": "thermal_agent",
        }
        try:
            pause_path.write_text(json.dumps(payload, indent=2))
            self.paused_jobs.add(job_dir)
            logger.warning(
                "THERMAL PAUSE: %s=%.1f°C >= %.1f°C — paused %s",
                self.metric, temp, self.threshold, job_dir.name,
            )
            self._log_event("job_paused", {
                "job_dir": str(job_dir), "temperature": temp, "threshold": self.threshold,
            })
        except OSError as e:
            logger.error("Failed to write pause.flag: %s", e)

    def clear_pause_flag(self, job_dir: Path, temp: float):
        """Remove pause.flag from job directory if we created it."""
        pause_path = job_dir / "pause.flag"
        if not pause_path.exists():
            return
        try:
            content = json.loads(pause_path.read_text())
            if content.get("agent") != "thermal_agent":
                return
        except (json.JSONDecodeError, OSError):
            return
        try:
            pause_path.unlink()
            if job_dir in self.paused_jobs:
                self.paused_jobs.remove(job_dir)
            logger.info(
                "THERMAL RESUME: %s=%.1f°C < %.1f°C — resumed %s",
                self.metric, temp, self.resume_threshold, job_dir.name,
            )
            self._log_event("job_resumed", {
                "job_dir": str(job_dir), "temperature": temp,
                "resume_threshold": self.resume_threshold,
            })
        except OSError as e:
            logger.error("Failed to clear pause.flag: %s", e)

    def show_fan_control_popup(self):
        """Launch fan control GUI popup in background thread."""
        if self.popup_shown:
            return

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
                self.popup_shown = False
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

        if temp >= self.threshold:
            for job_dir in self.watch_dirs:
                self.set_pause_flag(job_dir, temp)
            if self.show_popup and not self.popup_shown:
                logger.info("Launching fan control popup (temp=%.1f°C)", temp)
                self.show_fan_control_popup()
            return

        if temp < self.resume_threshold:
            for job_dir in self.watch_dirs:
                self.clear_pause_flag(job_dir, temp)
            return

        paused_count = len(self.paused_jobs)
        if paused_count > 0:
            logger.info(
                "Thermal: %.1f°C (paused: %d jobs, resume at %.1f°C)",
                temp, paused_count, self.resume_threshold,
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
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False

    def shutdown(self):
        """Clean shutdown — clear thermal pause flags we created."""
        logger.info("Shutting down thermal agent")
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
