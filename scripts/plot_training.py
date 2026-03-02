#!/usr/bin/env python3
"""
Plot training curves from HuggingFace Trainer output.
Reads trainer_state.json and metrics.jsonl from output_dir.
Generates a dynamic 2–4-panel figure: loss, perplexity, grad norm, LR.
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_metrics(output_dir):
    """
    Load unified metrics list by merging trainer_state.json log_history with
    metrics.jsonl (if present). Returns a list of dicts keyed by step.
    """
    output_dir = Path(output_dir)
    rows_by_step = {}

    # Primary: trainer_state.json
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

    # Secondary: metrics.jsonl (real-time streaming, may have extra keys)
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


def extract_series(rows):
    """
    Extract named series from merged metric rows.
    Returns a dict of series name → (x_list, y_list).
    """
    train_steps, train_loss = [], []
    eval_steps, eval_loss = [], []
    grad_steps, grad_norm = [], []
    lr_by_step = {}

    for entry in rows:
        step = entry.get("step")
        if step is None:
            continue
        if "loss" in entry:
            train_steps.append(step)
            train_loss.append(entry["loss"])
        if "eval_loss" in entry:
            eval_steps.append(step)
            eval_loss.append(entry["eval_loss"])
        if "grad_norm" in entry:
            grad_steps.append(step)
            grad_norm.append(entry["grad_norm"])
        if "learning_rate" in entry:
            lr_by_step[step] = entry["learning_rate"]

    lr_steps = sorted(lr_by_step)
    lr_vals = [lr_by_step[s] for s in lr_steps]

    # Perplexity derived from eval_loss
    perp_steps = eval_steps[:]
    perp_vals = [math.exp(min(v, 20)) for v in eval_loss]

    return {
        "train_loss": (train_steps, train_loss),
        "eval_loss": (eval_steps, eval_loss),
        "perplexity": (perp_steps, perp_vals),
        "grad_norm": (grad_steps, grad_norm),
        "lr": (lr_steps, lr_vals),
    }


def _no_data_text(ax, label="No data yet"):
    ax.text(0.5, 0.5, label, ha="center", va="center",
            transform=ax.transAxes, color="gray", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_training(output_dir, out_file=None, title=None):
    rows = load_metrics(output_dir)
    if not rows:
        raise ValueError(f"No metrics found in {output_dir}")

    series = extract_series(rows)
    output_dir = Path(output_dir)
    run_name = output_dir.name or "training"

    # Build panel list dynamically
    panels = []

    # Panel A: Train loss + eval loss (always included)
    panels.append(("loss", "Loss"))

    # Panel B: Perplexity (only if eval_loss data exists)
    if series["perplexity"][0]:
        panels.append(("perplexity", "Perplexity"))

    # Panel C: Gradient norm (only if data exists)
    if series["grad_norm"][0]:
        panels.append(("grad_norm", "Grad Norm"))

    # Panel D: Learning rate (always included)
    panels.append(("lr", "Learning Rate"))

    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title or f"Training: {run_name}", fontsize=12)

    for ax, (panel_id, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

        if panel_id == "loss":
            t_steps, t_loss = series["train_loss"]
            e_steps, e_loss = series["eval_loss"]
            if t_steps:
                ax.plot(t_steps, t_loss, "b-", alpha=0.7, label="train loss")
            if e_steps:
                ax.plot(e_steps, e_loss, "g-o", markersize=4, label="eval loss")
            if t_steps or e_steps:
                ax.legend(loc="upper right")
            else:
                _no_data_text(ax)

        elif panel_id == "perplexity":
            p_steps, p_vals = series["perplexity"]
            if p_steps:
                ax.plot(p_steps, p_vals, "m-o", markersize=4)
            else:
                _no_data_text(ax)

        elif panel_id == "grad_norm":
            g_steps, g_vals = series["grad_norm"]
            if g_steps:
                ax.plot(g_steps, g_vals, "r-", alpha=0.7)
            else:
                _no_data_text(ax)

        elif panel_id == "lr":
            lr_steps, lr_vals = series["lr"]
            if lr_steps:
                ax.plot(lr_steps, lr_vals, color="orange", alpha=0.8)
            else:
                _no_data_text(ax)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()

    out_path = out_file or output_dir / "training_curves.png"
    try:
        plt.savefig(out_path, dpi=120)
    except OSError as e:
        logger.error("Failed to save plot to %s: %s", out_path, e)
        raise
    plt.close()
    return str(out_path)


def main():
    p = argparse.ArgumentParser(description="Plot training curves from distill output")
    p.add_argument("output_dir", type=str, nargs="?", default="./distilled-minillm",
                   help="Path to training output (contains trainer_state.json)")
    p.add_argument("--out", "-o", type=str, default=None, help="Output image path")
    p.add_argument("--title", type=str, default=None)
    args = p.parse_args()
    try:
        out_path = plot_training(args.output_dir, args.out, args.title)
        logger.info("Saved plot to %s", out_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Validation failed: %s", e)
        raise SystemExit(1)
    except (json.JSONDecodeError, OSError) as e:
        logger.error("I/O error: %s", e)
        raise SystemExit(1)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
