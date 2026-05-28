"""Unified entry point for the Wow Sausage Maker control center.

Usage:
    python -m distill.ui.entry              # default port 7860
    python -m distill.ui.entry --port 7861
    python -m distill.ui.entry --host 0.0.0.0
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _ensure_gradio() -> None:
    """Auto-relaunch via pixi python if gradio is missing."""
    try:
        import gradio  # noqa: F401
    except ModuleNotFoundError:
        repo = Path(__file__).resolve().parent.parent.parent
        pixi_py = repo / ".pixi" / "envs" / "default" / "bin" / "python"
        if pixi_py.exists():
            logger.info("Re-launching with pixi python: %s", pixi_py)
            raise SystemExit(subprocess.call([str(pixi_py)] + sys.argv))
        sys.exit("gradio not found — run: pip install gradio  or  pixi install")


def _free_port(port: int) -> None:
    """SIGTERM any process occupying the given port."""
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return
    except OSError:
        return

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


def main() -> None:
    _ensure_gradio()

    # Import here so the gradio guard above runs first
    from distill.ui.app import build_app  # noqa: PLC0415
    from distill.infra.config import cfg  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="Wow Sausage Maker — LLM Distillation UI")
    parser.add_argument("--port", type=int, default=cfg.services.gradio_port,
                        help="Port to serve the UI on (default from config)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    args = parser.parse_args()

    _free_port(args.port)
    logger.info("Starting Wow Sausage Maker on http://%s:%d", args.host, args.port)

    demo = build_app()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
