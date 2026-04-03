#!/usr/bin/env python3
"""
Autonomous thermal control agent - system-wide hardware protection.

Continuously monitors CPU/GPU/SoC temperatures and pauses ALL running jobs when
thermal limits are exceeded. Resumes jobs automatically when temps drop.

Usage:
    python -m distill.thermal_agent --watch ./distilled-minillm
    python -m distill.thermal_agent --watch ./distilled-minillm ./distilled-mlx ./distilled-sft
    python -m distill.thermal_agent --watch . --threshold 70
    python -m distill.thermal_agent --daemon --watch . --threshold 85
"""

import argparse
import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

from .thermal_agent_class import ThermalAgent  # noqa: E402


def daemonize():
    """Fork process to run as daemon (UNIX only)."""
    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error("Fork failed: %s", e)
        sys.exit(1)

    os.chdir("/")
    os.setsid()
    os.umask(0)

    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        logger.error("Fork failed: %s", e)
        sys.exit(1)

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
  python -m distill.thermal_agent --watch ./distilled-minillm
  python -m distill.thermal_agent --watch ./distilled-minillm ./distilled-mlx
  python -m distill.thermal_agent --watch . --threshold 70 --interval 15
  python -m distill.thermal_agent --daemon --watch . --log thermal_agent.jsonl

Metrics:
  soc_temp_c     - System-on-Chip temperature (default)
  cpu_temp_c     - CPU cores temperature
  gpu_temp_c     - GPU cores temperature
  total_power_w  - Total power consumption (Watts)
        """,
    )
    parser.add_argument("--watch", nargs="+", required=True,
                        help="Output directories to monitor (can specify multiple)")
    parser.add_argument("--threshold", type=float, default=85.0,
                        help="Temperature threshold in °C (default: 85°C)")
    parser.add_argument("--metric",
                        choices=["soc_temp_c", "cpu_temp_c", "gpu_temp_c", "total_power_w"],
                        default="soc_temp_c", help="Metric to monitor (default: soc_temp_c)")
    parser.add_argument("--hysteresis", type=float, default=5.0,
                        help="Temperature delta for auto-resume (default: 5°C)")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds (default: 30)")
    parser.add_argument("--log", type=str, default=None,
                        help="Log file for thermal events (JSONL format)")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as background daemon (UNIX only)")
    parser.add_argument("--once", action="store_true",
                        help="Run one check and exit (for testing)")
    parser.add_argument("--show-popup", action="store_true",
                        help="Show fan control GUI popup when temperature exceeds threshold")

    args = parser.parse_args()

    if args.threshold <= 0:
        logger.error("Threshold must be positive")
        sys.exit(1)
    if args.interval <= 0:
        logger.error("Interval must be positive")
        sys.exit(1)

    if args.daemon and not args.once:
        logger.info("Starting thermal agent as daemon...")
        daemonize()

    agent = ThermalAgent(
        watch_dirs=args.watch,
        threshold=args.threshold,
        metric=args.metric,
        hysteresis=args.hysteresis,
        interval=args.interval,
        log_file=args.log,
        show_popup=args.show_popup,
    )

    if args.once:
        agent.tick()
    else:
        agent.run()


if __name__ == "__main__":
    main()
