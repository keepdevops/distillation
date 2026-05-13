"""
Subprocess-based handler factories for synthesis tabs (Magpie, Golden Pipeline).

Kept separate from gradio_ui_handlers.py to satisfy the 300-LOC rule.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def make_magpie_fn():
    """Factory that returns a streaming ``magpie_fn`` closure.

    Runs ``python -m distill.data.magpie`` as a subprocess and yields
    log lines incrementally so the Gradio log box updates in real time.

    Returns:
        Generator callable suitable for wiring to a Gradio ``Button.click``
        with ``outputs=[log_box]``.
    """
    import subprocess
    import sys

    def magpie_fn(
        teacher: str,
        domain: str,
        n_pairs: int,
        output_dir: str,
        backend: str,
        batch_size: int,
        inst_temp: float,
        resp_temp: float,
        filter_chk: bool,
        target_n: int,
    ):
        cmd = [
            sys.executable, "-m", "distill.data.magpie",
            "--teacher", teacher.strip(),
            "--domain", domain,
            "--n", str(int(n_pairs)),
            "--output_dir", output_dir.strip(),
            "--backend", backend,
            "--batch_size", str(int(batch_size)),
            "--inst_temp", str(round(inst_temp, 4)),
            "--resp_temp", str(round(resp_temp, 4)),
        ]
        if filter_chk:
            cmd.append("--filter")
        if filter_chk and int(target_n) > 0:
            cmd += ["--target", str(int(target_n))]

        logger.info("Launching Magpie: %s", " ".join(cmd))
        accumulated = ""
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                accumulated += line
                yield accumulated
            proc.wait()
            if proc.returncode == 0:
                accumulated += "\n\u2705 Magpie synthesis complete."
            else:
                accumulated += f"\n\u274c Process exited with code {proc.returncode}."
            yield accumulated
        except Exception as exc:
            logger.error("Magpie subprocess error: %s", exc, exc_info=True)
            yield accumulated + f"\n\u274c Error: {exc}"

    return magpie_fn


def make_golden_fns():
    """Factory returning ``(run_fn, stop_fn)`` for the Golden Pipeline tab.

    ``run_fn`` builds a temporary config JSON from widget values, launches
    ``distill.orchestration.agent`` as a subprocess, and streams log lines.
    ``stop_fn`` terminates the running process if one exists.

    Returns:
        Tuple ``(run_fn, stop_fn)``.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    _state: dict = {"proc": None}

    def run_fn(
        dataset: str,
        max_samples: int,
        epochs: int,
        batch_size: int,
        grad_acc: int,
        lr: float,
        lora_r: int,
        temperature: float,
        ce_alpha: float,
        export: str,
        output_dir: str,
        filter_chk: bool,
        filter_target: int,
        watchdog_chk: bool,
        benchmarks_chk: bool,
        skip_eval: bool,
        skip_judge: bool,
    ):
        cfg = {
            "backend": "mlx",
            "dataset": dataset.strip(),
            "max_samples": int(max_samples),
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "grad_acc": int(grad_acc),
            "learning_rate": float(lr),
            "lora_r": int(lora_r),
            "temperature": round(float(temperature), 4),
            "ce_alpha": round(float(ce_alpha), 4),
            "export": export if export != "none" else "",
            "output_dir": output_dir.strip(),
            "filter": bool(filter_chk),
            "filter_target": int(filter_target),
            "watchdog": bool(watchdog_chk),
            "benchmarks": bool(benchmarks_chk),
            "skip_eval": bool(skip_eval),
            "skip_judge": bool(skip_judge),
            "open": True,
            "seed": 42,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="golden_run_"
        ) as tf:
            json.dump(cfg, tf, indent=2)
            tmp_path = tf.name

        cmd = [sys.executable, "-m", "distill.orchestration.agent", "--config", tmp_path]
        logger.info("Launching Golden Pipeline: %s", " ".join(cmd))

        accumulated = f"Config written to: {tmp_path}\n\n"
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )
            _state["proc"] = proc
            for line in proc.stdout:
                accumulated += line
                yield accumulated
            proc.wait()
            _state["proc"] = None
            Path(tmp_path).unlink(missing_ok=True)
            if proc.returncode == 0:
                accumulated += "\n\u2705 Golden pipeline complete."
            else:
                accumulated += f"\n\u274c Process exited with code {proc.returncode}."
            yield accumulated
        except Exception as exc:
            logger.error("Golden pipeline subprocess error: %s", exc, exc_info=True)
            _state["proc"] = None
            yield accumulated + f"\n\u274c Error: {exc}"

    def stop_fn():
        proc = _state.get("proc")
        if proc and proc.poll() is None:
            proc.terminate()
            logger.info("Golden pipeline process terminated by user.")
            return "\u23f9 Process terminated."
        return "No running process."

    return run_fn, stop_fn
