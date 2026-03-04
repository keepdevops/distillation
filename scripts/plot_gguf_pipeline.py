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


def find_model_artifacts(output_dir):
    """
    Scan output_dir for distillation artifacts: .gguf, .mlpackage, MLX .npz weights.
    Returns list of (label, size_gb, path, kind) sorted by name.
    kind is one of: 'gguf', 'coreml', 'mlx'.
    """
    root = Path(output_dir)
    results = []

    for p in sorted(root.glob("*.gguf")):
        size_gb = p.stat().st_size / (1024 ** 3)
        results.append((p.name, size_gb, str(p), "gguf"))

    for p in sorted(root.glob("*.mlpackage")):
        try:
            size_gb = sum(
                f.stat().st_size for f in p.rglob("*") if f.is_file()
            ) / (1024 ** 3)
        except OSError:
            size_gb = 0.0
        results.append((p.name, size_gb, str(p), "coreml"))

    # MLX weights: root-level *.npz files (raw training output)
    for p in sorted(root.glob("*.npz")):
        size_gb = p.stat().st_size / (1024 ** 3)
        results.append((p.name, size_gb, str(p), "mlx"))

    # MLX quantized subdirs (from mlx_lm.convert) contain multiple *.npz shards
    for subdir in (sorted(root.iterdir()) if root.exists() else []):
        if not subdir.is_dir():
            continue
        npz_files = list(subdir.glob("*.npz"))
        if npz_files:
            size_gb = sum(f.stat().st_size for f in npz_files) / (1024 ** 3)
            label = f"{subdir.name}/ (MLX quant)"
            results.append((label, size_gb, str(subdir), "mlx"))

    return results


def find_gguf_files(output_dir):
    """Legacy helper: returns only .gguf artifacts (used for benchmark panel)."""
    return [
        (name, size_gb, path)
        for name, size_gb, path, kind in find_model_artifacts(output_dir)
        if kind == "gguf"
    ]


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
    all_artifacts = find_model_artifacts(output_dir)
    gguf_files = [(n, s, p) for n, s, p, k in all_artifacts if k == "gguf"]

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

    # ── [1,0] Model artifact sizes (GGUF + CoreML + MLX) ─────────────────────
    ax10 = axes[1, 0]
    ax10.set_title("Model Artifacts", fontsize=10)
    ax10.set_xlabel("Size (GB)")
    ax10.grid(True, alpha=0.3, axis="x")

    kind_colors = {"gguf": "steelblue", "coreml": "darkorchid", "mlx": "seagreen"}

    if all_artifacts:
        names = [a[0] for a in all_artifacts]
        sizes = [a[1] for a in all_artifacts]
        kinds = [a[3] for a in all_artifacts]
        colors = [kind_colors.get(k, "gray") for k in kinds]
        y_pos = range(len(names))
        bars = ax10.barh(list(y_pos), sizes, color=colors, alpha=0.8)
        ax10.set_yticks(list(y_pos))
        ax10.set_yticklabels(names, fontsize=8)
        for bar, size in zip(bars, sizes):
            ax10.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                      f"{size:.2f} GB", va="center", fontsize=8)
        # Legend
        from matplotlib.patches import Patch
        legend_els = [Patch(facecolor=c, label=k.upper()) for k, c in kind_colors.items()
                      if k in kinds]
        if legend_els:
            ax10.legend(handles=legend_els, fontsize=7, loc="lower right")
    else:
        _no_data_text(ax10, "No artifacts found\n(.gguf / .mlpackage / MLX)")

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
