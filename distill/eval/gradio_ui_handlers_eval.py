"""
Subprocess-based handler factories for eval and export tabs.

Kept separate from gradio_ui_handlers.py and gradio_ui_handlers_synth.py
to respect the 300-LOC modular constraint.
"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import gradio as gr

logger = logging.getLogger(__name__)

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)


def _stream_subprocess(cmd: list[str], cwd: str = _PROJECT_ROOT):
    """Run *cmd* as a subprocess and yield accumulated stdout+stderr lines."""
    logger.info("Running: %s", " ".join(cmd))
    accumulated = ""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
        for line in proc.stdout:
            accumulated += line
            yield accumulated
        proc.wait()
        if proc.returncode == 0:
            accumulated += "\n\u2705 Done."
        else:
            accumulated += f"\n\u274c Process exited with code {proc.returncode}."
        yield accumulated
    except Exception as exc:
        logger.error("Subprocess error: %s", exc, exc_info=True)
        yield accumulated + f"\n\u274c Error: {exc}"


# ── Quality Eval ──────────────────────────────────────────────────────────────

def make_quality_fn():
    """Return a streaming generator for ``distill.eval.quality``."""

    def quality_fn(
        model_dir: str,
        n_samples: int,
        judge: bool,
        teacher: str,
        backend: str,
    ):
        cmd = [
            sys.executable, "-m", "distill.eval.quality",
            model_dir.strip(),
            "--n_samples", str(int(n_samples)),
            "--backend", backend,
        ]
        if judge:
            cmd.append("--judge")
            if teacher.strip():
                cmd += ["--teacher", teacher.strip()]
        yield from _stream_subprocess(cmd)

    return quality_fn


# ── Perplexity Eval ───────────────────────────────────────────────────────────

def make_perplexity_fn():
    """Return a streaming generator for ``distill.eval.perplexity``."""

    def perplexity_fn(model_dir: str, backend: str):
        cmd = [
            sys.executable, "-m", "distill.eval.perplexity",
            model_dir.strip(),
            "--backend", backend,
        ]
        yield from _stream_subprocess(cmd)

    return perplexity_fn


# ── WikiText-2 Benchmarks ─────────────────────────────────────────────────────

def make_benchmarks_fn():
    """Return a streaming generator for ``distill.eval.benchmarks``."""

    def benchmarks_fn(
        model_dir: str,
        n_sequences: int,
        baseline_dir: str,
        backend: str,
    ):
        cmd = [
            sys.executable, "-m", "distill.eval.benchmarks",
            model_dir.strip(),
            "--n_sequences", str(int(n_sequences)),
            "--backend", backend,
        ]
        if baseline_dir.strip():
            cmd += ["--baseline_dir", baseline_dir.strip()]
        yield from _stream_subprocess(cmd)

    return benchmarks_fn


# ── Quality Results Loader ────────────────────────────────────────────────────

def make_load_quality_results_fn():
    """Return a function that reads quality_metrics.json from model_dir."""
    import json as _json

    def load_quality_results(model_dir: str):
        p = Path(model_dir.strip()) / "quality_metrics.json"
        if not p.exists():
            return gr.update(value={"error": f"Not found: {p}"}, visible=True)
        try:
            data = _json.loads(p.read_text())
            return gr.update(value=data, visible=True)
        except Exception as exc:
            logger.error("Failed to load quality_metrics.json: %s", exc)
            return gr.update(value={"error": str(exc)}, visible=True)

    return load_quality_results


# ── UMAP Visualization Loader ─────────────────────────────────────────────────

def make_load_umap_fn():
    """Return a function that reads embedding_viz.json and returns a scatter plot."""
    import json as _json

    def load_umap(model_dir: str):
        p = Path(model_dir.strip()) / "embedding_viz.json"
        if not p.exists():
            return gr.update(visible=False)
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            data = _json.loads(p.read_text())
            points = data.get("points", [])
            if not points:
                return gr.update(visible=False)

            categories = sorted({pt["category"] for pt in points})
            cmap = plt.cm.get_cmap("tab10", len(categories))
            cat_idx = {c: i for i, c in enumerate(categories)}

            fig, ax = plt.subplots(figsize=(7, 5))
            for cat in categories:
                xs = [pt["x"] for pt in points if pt["category"] == cat]
                ys = [pt["y"] for pt in points if pt["category"] == cat]
                ax.scatter(xs, ys, s=12, alpha=0.7, color=cmap(cat_idx[cat]), label=cat)
            ax.legend(fontsize=7, markerscale=1.5, loc="best")
            ax.set_title("Embedding UMAP")
            ax.set_xlabel("UMAP-1")
            ax.set_ylabel("UMAP-2")
            fig.tight_layout()
            return gr.update(value=fig, visible=True)
        except Exception as exc:
            logger.error("Failed to render UMAP: %s", exc)
            return gr.update(visible=False)

    return load_umap


# ── Perplexity Results Loader ─────────────────────────────────────────────────

def make_load_ppl_results_fn():
    """Return a function that reads metrics.jsonl from model_dir."""
    import json as _json

    def load_ppl_results(model_dir: str):
        p = Path(model_dir.strip()) / "metrics.jsonl"
        if not p.exists():
            return gr.update(value={"error": f"Not found: {p}"}, visible=True)
        try:
            rows = [_json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            return gr.update(value=rows, visible=True)
        except Exception as exc:
            logger.error("Failed to load metrics.jsonl: %s", exc)
            return gr.update(value={"error": str(exc)}, visible=True)

    return load_ppl_results


# ── Benchmark Results Loader ──────────────────────────────────────────────────

def make_load_bench_results_fn():
    """Return a function that reads benchmark_results.json from model_dir."""
    import json as _json

    def load_bench_results(model_dir: str):
        p = Path(model_dir.strip()) / "benchmark_results.json"
        if not p.exists():
            return gr.update(value={"error": f"Not found: {p}"}, visible=True)
        try:
            data = _json.loads(p.read_text())
            return gr.update(value=data, visible=True)
        except Exception as exc:
            logger.error("Failed to load benchmark_results.json: %s", exc)
            return gr.update(value={"error": str(exc)}, visible=True)

    return load_bench_results


# ── GGUF Export ───────────────────────────────────────────────────────────────

def make_gguf_export_fn():
    """Return a streaming generator for ``scripts/export_student_gguf.sh``."""

    def gguf_export_fn(model_dir: str, llama_dir: str):
        script = Path(_PROJECT_ROOT) / "scripts" / "export_student_gguf.sh"
        if not script.exists():
            yield f"\u274c Script not found: {script}"
            return
        cmd = ["bash", str(script), model_dir.strip(), llama_dir.strip()]
        yield from _stream_subprocess(cmd)

    return gguf_export_fn


# ── CoreML Export ─────────────────────────────────────────────────────────────

def make_coreml_export_fn():
    """Return a streaming generator for ``distill.export.coreml``."""

    def coreml_export_fn(
        model_dir: str,
        quantize: str,
        seq_len: int,
        output_dir: str,
    ):
        cmd = [
            sys.executable, "-m", "distill.export.coreml",
            "--model_dir", model_dir.strip(),
            "--seq_len", str(int(seq_len)),
        ]
        if quantize and quantize != "none":
            cmd += ["--quantize", quantize]
        if output_dir.strip():
            cmd += ["--output_dir", output_dir.strip()]
        yield from _stream_subprocess(cmd)

    return coreml_export_fn
