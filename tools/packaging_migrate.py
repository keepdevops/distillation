#!/usr/bin/env python3
"""One-off: relative imports + strip sys.path.insert from distill/*.py"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "distill"

MAPPINGS = [
    ("from data_pipeline import", "from .data_pipeline import"),
    ("from metrics_io import", "from .metrics_io import"),
    ("from train_utils import", "from .train_utils import"),
    ("from run_eval import", "from .run_eval import"),
    ("from universal_model_loader import", "from .universal_model_loader import"),
    ("from artifact_detector import", "from .artifact_detector import"),
    ("from mlx_eval_utils import", "from .mlx_eval_utils import"),
    ("from cpp_eval_utils import", "from .cpp_eval_utils import"),
    ("from model_path_helper import", "from .model_path_helper import"),
    ("from experiment_log import", "from .experiment_log import"),
    ("from watchdog_callbacks import", "from .watchdog_callbacks import"),
    ("from early_stopping_callback import", "from .early_stopping_callback import"),
]


def strip_sys_path_insert(text: str) -> str:
    lines = text.splitlines(keepends=True)
    out = []
    for line in lines:
        s = line.strip()
        if s.startswith("sys.path.insert"):
            continue
        out.append(line)
    return "".join(out)


def main() -> None:
    for path in sorted(ROOT.glob("*.py")):
        if path.name == "launch_ui.py":
            continue
        text = path.read_text()
        for a, b in MAPPINGS:
            text = text.replace(a, b)
        text = strip_sys_path_insert(text)
        path.write_text(text)
        print(path.name)


if __name__ == "__main__":
    main()
