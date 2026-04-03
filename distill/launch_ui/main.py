"""CLI entry for the distillation launcher UI."""
from __future__ import annotations

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Auto-relaunch through pixi if gradio is not importable
try:
    import gradio as gr  # noqa: F401
except ModuleNotFoundError:
    _repo = Path(__file__).resolve().parent.parent.parent
    pixi = _repo / ".pixi" / "envs" / "default" / "bin" / "python"
    if pixi.exists():
        print(f"Re-launching with pixi python: {pixi}")
        raise SystemExit(subprocess.call([str(pixi)] + sys.argv))
    else:
        sys.exit("gradio not found. Run: pixi run python -m distill.launch_ui")

import gradio as gr

from .ui import build_ui

def _free_port(port: int) -> None:
    """Kill any process listening on the given port."""
    import signal
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return  # port already free
    except OSError:
        return
    # Port is occupied — find and kill the owner
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--host", type=str, default="127.0.0.1")
    args = p.parse_args()
    _free_port(args.port)
    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=False,
                theme=gr.themes.Monochrome())


if __name__ == "__main__":
    main()
