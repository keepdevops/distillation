#!/usr/bin/env python3
"""Split monolithic distill/launch_ui.py into distill/launch_ui/ package."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
src_path = ROOT / "distill" / "launch_ui.py"
lines = src_path.read_text().splitlines(keepends=True)


def L(a: int, b: int) -> str:
    return "".join(lines[a - 1 : b])


def main() -> None:
    pkg = ROOT / "distill" / "launch_ui"
    pkg.mkdir(exist_ok=True)

    (pkg / "presets.py").write_text(
        '"""Preset model/dataset IDs shown in the launcher."""\n\n' + L(41, 65)
    )

    discovery = (
        '"""Discover teachers, students, and datasets from local caches."""\n'
        "from __future__ import annotations\n\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "from .presets import KNOWN_DATASETS, KNOWN_STUDENTS, KNOWN_TEACHERS\n"
        "from ..paths import project_dir, scripts_dir\n\n"
        "PROJECT_DIR = project_dir()\n"
        "SCRIPTS_DIR = scripts_dir()\n\n"
    ) + L(71, 229)
    (pkg / "discovery.py").write_text(discovery)

    runner = L(232, 745)
    runner = runner.replace(
        "def _build_cmd(script: str, params: dict) -> list[str]:\n"
        "    cmd = [PYTHON, str(SCRIPTS_DIR / script)]",
        "def _build_cmd(script: str, params: dict) -> list[str]:\n"
        '    mod = "distill." + script.replace(".py", "")\n'
        '    cmd = [PYTHON, "-m", mod]',
    )
    runner = runner.replace(
        'cmd = [PYTHON, str(SCRIPTS_DIR / "run_eval.py"), output_dir]',
        'cmd = [PYTHON, "-m", "distill.run_eval", output_dir]',
    )
    runner = runner.replace(
        'cmd = [PYTHON, str(SCRIPTS_DIR / "eval_quality.py"), output_dir]',
        'cmd = [PYTHON, "-m", "distill.eval_quality", output_dir]',
    )
    runner = runner.replace(
        'cmd = [PYTHON, str(SCRIPTS_DIR / "run_benchmarks.py"), output_dir]',
        'cmd = [PYTHON, "-m", "distill.run_benchmarks", output_dir]',
    )
    runner_header = (
        '"""Subprocess launches, log streaming, and training progress."""\n'
        "from __future__ import annotations\n\n"
        "import json\n"
        "import os\n"
        "import queue\n"
        "import re\n"
        "import subprocess\n"
        "import sys\n"
        "import threading\n"
        "from pathlib import Path\n\n"
        "import pandas as pd\n\n"
        "from .discovery import (\n"
        "    discover_datasets,\n"
        "    discover_output_dirs,\n"
        "    discover_students,\n"
        "    discover_teachers,\n"
        ")\n"
        "from ..paths import project_dir, scripts_dir\n\n"
        "PROJECT_DIR = project_dir()\n"
        "SCRIPTS_DIR = scripts_dir()\n"
        "PYTHON = sys.executable\n\n"
    )
    (pkg / "runner.py").write_text(runner_header + runner)

    ui_body = L(751, 2535)
    ui_body = ui_body.replace(
        "                try:\n"
        "                    sys.path.insert(0, str(Path(__file__).parent))\n"
        "                    from show_algorithms import ALGORITHMS, build_html as _build_html_help\n",
        "                try:\n"
        "                    from ..show_algorithms import ALGORITHMS, build_html as _build_html_help\n",
    )
    ui_body = ui_body.replace(
        "[PYTHON, str(SCRIPTS_DIR / \"expert_pipeline.py\"),\n",
        '[PYTHON, "-m", "distill.expert_pipeline",\n',
    )
    ui_body = ui_body.replace(
        "            import sys as _sys, importlib\n"
        '            _sys.path.insert(0, str(SCRIPTS_DIR))\n'
        '            ep = importlib.import_module("expert_pipeline")\n'
        "            prompt = ep.DOMAIN_SYSTEM_PROMPTS.get(domain, ep.DEFAULT_SYSTEM_PROMPT)\n",
        "            from ..expert_pipeline import DEFAULT_SYSTEM_PROMPT, DOMAIN_SYSTEM_PROMPTS\n"
        "            prompt = DOMAIN_SYSTEM_PROMPTS.get(domain, DEFAULT_SYSTEM_PROMPT)\n",
    )
    ui_header = (
        '"""Gradio layout for the distillation launcher."""\n'
        "from __future__ import annotations\n\n"
        "import base64\n"
        "import json\n"
        "import os\n"
        "import re\n"
        "import subprocess\n"
        "import sys\n"
        "from pathlib import Path\n\n"
        "import gradio as gr\n"
        "import pandas as pd\n\n"
        "from .discovery import (\n"
        "    discover_datasets,\n"
        "    discover_output_dirs,\n"
        "    discover_students,\n"
        "    discover_teachers,\n"
        ")\n"
        "from .runner import (\n"
        "    _build_cmd,\n"
        "    _ep_start_proc,\n"
        "    clear_logs,\n"
        "    launch_eval_benchmark,\n"
        "    launch_eval_perplexity,\n"
        "    launch_eval_quality,\n"
        "    launch_filter,\n"
        "    launch_magpie,\n"
        "    launch_run,\n"
        "    launch_synth,\n"
        "    poll_logs,\n"
        "    save_custom_domain,\n"
        "    stop_run,\n"
        ")\n"
        "from ..paths import project_dir\n\n"
        "PROJECT_DIR = project_dir()\n\n"
    )
    (pkg / "ui.py").write_text(ui_header + ui_body)

    main_header = (
        '"""CLI entry for the distillation launcher UI."""\n'
        "from __future__ import annotations\n\n"
        "import argparse\n"
        "import os\n"
        "import signal\n"
        "import socket\n"
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        "from pathlib import Path\n\n"
        "# Auto-relaunch through pixi if gradio is not importable\n"
        "try:\n"
        "    import gradio as gr  # noqa: F401\n"
        "except ModuleNotFoundError:\n"
        "    _repo = Path(__file__).resolve().parent.parent.parent\n"
        '    pixi = _repo / ".pixi" / "envs" / "default" / "bin" / "python"\n'
        "    if pixi.exists():\n"
        '        print(f"Re-launching with pixi python: {pixi}")\n'
        "        raise SystemExit(subprocess.call([str(pixi)] + sys.argv))\n"
        "    else:\n"
        '        sys.exit("gradio not found. Run: pixi run python -m distill.launch_ui")\n\n'
        "import gradio as gr\n\n"
        "from .ui import build_ui\n\n"
    )
    # Original main.py ends with duplicate imports — strip _free_port/main from source
    # and use our header; append _free_port + main from file lines 2538–2574
    tail = L(2538, 2574)
    (pkg / "main.py").write_text(main_header + tail)

    (pkg / "__init__.py").write_text(
        '"""Gradio distillation launcher."""\n\nfrom .main import main\n\n__all__ = ["main"]\n'
    )

    src_path.unlink()
    print("Wrote", pkg, "and removed", src_path)


if __name__ == "__main__":
    main()
