#!/usr/bin/env python3
"""
End-to-end pipeline summary figure: training → eval → GGUF file stats → inference speed.
CLI: python scripts/plot_gguf_pipeline.py ./distilled-minillm
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_metrics(output_dir):
    """
    Load unified metrics from trainer_state.json and metrics.jsonl.
    Returns a list of dicts sorted by step.
    """
    output_dir = Path(output_dir)
    rows_by_step = {}

    state_path = output_dir / "trainer_state.json"
    if state_path.exists():
        try:
            with open(state_path) as f:
                state = json.load(f)
            for entry in state.get("log_history", []):
                step = entry.get("step")
                if step is None:
                    continue
                rows_by_step.setdefault(step, {}).update(entry)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Could not read trainer_state.json: %s", e)

    jsonl_path = output_dir / "metrics.jsonl"
    if jsonl_path.exists():
        try:
            with open(jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    step = entry.get("step")
                    if step is None:
                        continue
                    rows_by_step.setdefault(step, {}).update(entry)
        except OSError as e:
            logger.warning("Could not read metrics.jsonl: %s", e)

    return [rows_by_step[s] for s in sorted(rows_by_step)]


def find_gguf_files(output_dir):
    """
    Scan output_dir (and one level up) for .gguf files.
    Returns list of (name, size_gb).
    """
    root = Path(output_dir)
    candidates = list(root.glob("*.gguf")) + list(root.parent.glob("*.gguf"))
    # Also check scripts/ dir relative to project root
    scripts_dir = root.parent / "scripts"
    if scripts_dir.exists():
        candidates += list(scripts_dir.glob("*.gguf"))

    seen = set()
    results = []
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        size_gb = p.stat().st_size / (1024 ** 3)
        results.append((p.name, size_gb, str(p)))
    return sorted(results, key=lambda x: x[0])


def benchmark_gguf(gguf_path, n_tokens=50):
    """
    Run a short inference benchmark using llama_cpp.
    Returns tokens/sec, or None if llama_cpp is not installed or inference fails.
    """
    try:
        from llama_cpp import Llama
        llm = Llama(model_path=str(gguf_path), n_ctx=128, n_threads=4, verbose=False)
        t0 = time.perf_counter()
        out = llm("The capital of France is", max_tokens=n_tokens, echo=False)
        elapsed = time.perf_counter() - t0
        completion_tokens = out["usage"]["completion_tokens"]
        if elapsed > 0 and completion_tokens > 0:
            return completion_tokens / elapsed
        return None
    except ImportError:
        return None
    except Exception as e:
        logger.warning("Benchmark failed for %s: %s", gguf_path, e)
        return None


def _no_data_text(ax, label="No data yet"):
    ax.text(0.5, 0.5, label, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_pipeline(output_dir, out_file=None):
    """
    Build a 2×2 summary figure and return the matplotlib Figure.
    Also saves to out_file (or {output_dir}/pipeline_summary.png).
    """
    output_dir = Path(output_dir)
    rows = load_metrics(output_dir)
    gguf_files = find_gguf_files(output_dir)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Pipeline Summary: {output_dir.name}", fontsize=13, fontweight="bold")

    # ── [0,0] Training loss (Reverse KL Loss) ────────────────────────────────
    ax00 = axes[0, 0]
    ax00.set_title("Training Loss (Reverse KL)", fontsize=10)
    ax00.set_xlabel("Step")
    ax00.set_ylabel("Loss")
    ax00.grid(True, alpha=0.3)

    train_steps, train_loss = [], []
    for e in rows:
        step = e.get("step")
        if step is not None and "loss" in e:
            train_steps.append(step)
            train_loss.append(e["loss"])

    if train_steps:
        ax00.plot(train_steps, train_loss, "b-", alpha=0.8)
    else:
        _no_data_text(ax00)

    # ── [0,1] Perplexity over eval steps ─────────────────────────────────────
    ax01 = axes[0, 1]
    ax01.set_title("Eval Perplexity", fontsize=10)
    ax01.set_xlabel("Step")
    ax01.set_ylabel("Perplexity")
    ax01.grid(True, alpha=0.3)

    eval_steps, perp_vals = [], []
    for e in rows:
        step = e.get("step")
        if step is not None and "eval_loss" in e:
            eval_steps.append(step)
            perp_vals.append(math.exp(min(e["eval_loss"], 20)))

    if eval_steps:
        ax01.plot(eval_steps, perp_vals, "m-o", markersize=5)
    else:
        _no_data_text(ax01, "No eval data")

    # ── [1,0] GGUF artifact sizes ─────────────────────────────────────────────
    ax10 = axes[1, 0]
    ax10.set_title("GGUF Artifacts", fontsize=10)
    ax10.set_xlabel("Size (GB)")
    ax10.grid(True, alpha=0.3, axis="x")

    if gguf_files:
        names = [f[0] for f in gguf_files]
        sizes = [f[1] for f in gguf_files]
        y_pos = range(len(names))
        bars = ax10.barh(y_pos, sizes, color="steelblue", alpha=0.8)
        ax10.set_yticks(list(y_pos))
        ax10.set_yticklabels(names, fontsize=8)
        for bar, size in zip(bars, sizes):
            ax10.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{size:.2f} GB", va="center", fontsize=8)
    else:
        _no_data_text(ax10, "No .gguf files found")

    # ── [1,1] Inference speed ─────────────────────────────────────────────────
    ax11 = axes[1, 1]
    ax11.set_title("Inference Speed", fontsize=10)
    ax11.set_xlabel("Tokens / second")
    ax11.grid(True, alpha=0.3, axis="x")

    # Check llama_cpp availability
    try:
        import llama_cpp  # noqa: F401
        llama_cpp_available = True
    except ImportError:
        llama_cpp_available = False

    if not llama_cpp_available:
        _no_data_text(ax11, "pip install llama-cpp-python\nto enable benchmarks")
    elif not gguf_files:
        _no_data_text(ax11, "No .gguf files to benchmark")
    else:
        speed_names, speed_vals = [], []
        for name, _size, path in gguf_files:
            tps = benchmark_gguf(path)
            if tps is not None:
                speed_names.append(name)
                speed_vals.append(tps)

        if speed_vals:
            y_pos = range(len(speed_names))
            ax11.barh(list(y_pos), speed_vals, color="darkorange", alpha=0.8)
            ax11.set_yticks(list(y_pos))
            ax11.set_yticklabels(speed_names, fontsize=8)
            for i, v in enumerate(speed_vals):
                ax11.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
        else:
            _no_data_text(ax11, "Benchmark failed\n(check model paths)")

    plt.tight_layout()

    out_path = out_file or output_dir / "pipeline_summary.png"
    try:
        plt.savefig(out_path, dpi=120)
        logger.info("Saved pipeline summary to %s", out_path)
    except OSError as e:
        logger.error("Failed to save pipeline plot: %s", e)
    return fig


def main():
    p = argparse.ArgumentParser(description="End-to-end pipeline summary figure")
    p.add_argument("output_dir", type=str, nargs="?", default="./distilled-minillm",
                   help="Training output directory")
    p.add_argument("--out", "-o", type=str, default=None, help="Output image path")
    args = p.parse_args()
    plot_pipeline(args.output_dir, args.out)


if __name__ == "__main__":
    main()
