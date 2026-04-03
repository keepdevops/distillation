"""Training curve plotting utilities for the distillation dashboard."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_from_state(state, title="Training"):
    if not state or not state.get("log_history"):
        return None
    log_history = state["log_history"]
    steps, loss_vals, eval_steps, eval_loss_vals = [], [], [], []
    lr_data = []
    for e in log_history:
        step = e.get("step")
        if step is None:
            continue
        if "loss" in e:
            steps.append(step)
            loss_vals.append(e["loss"])
        if "learning_rate" in e:
            lr_data.append((step, e["learning_rate"]))
        if "eval_loss" in e:
            eval_steps.append(step)
            eval_loss_vals.append(e["eval_loss"])
    lr_data.sort(key=lambda x: x[0])
    lr_steps = [x[0] for x in lr_data]
    lr_vals = [x[1] for x in lr_data]

    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    fig.suptitle(title, fontsize=11)
    axes[0].plot(steps, loss_vals, "b-", alpha=0.7, label="train loss")
    if eval_steps:
        axes[0].plot(eval_steps, eval_loss_vals, "g-o", markersize=3, label="eval loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(True, alpha=0.3)
    if lr_steps:
        axes[1].plot(lr_steps, lr_vals, color="orange", alpha=0.8)
    axes[1].set_ylabel("Learning rate")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def on_run_select(run_path):
    if not run_path:
        return None
    from .plot_training import extract_series, load_metrics
    import math
    rows = load_metrics(run_path)
    if not rows:
        return None
    series = extract_series(rows)
    title = Path(run_path).name
    panels = [("loss", "Loss"), ("lr", "Learning Rate")]
    if series["perplexity"][0]:
        panels.insert(1, ("perplexity", "Perplexity"))
    if series["grad_norm"][0]:
        panels.insert(-1, ("grad_norm", "Grad Norm"))
    n = len(panels)
    fig, axes = plt.subplots(n, 1, figsize=(8, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=11)
    for ax, (panel_id, ylabel) in zip(axes, panels):
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if panel_id == "loss":
            t_steps, t_loss = series["train_loss"]
            e_steps, e_loss = series["eval_loss"]
            if t_steps:
                ax.plot(t_steps, t_loss, "b-", alpha=0.7, label="train loss")
            if e_steps:
                ax.plot(e_steps, e_loss, "g-o", markersize=3, label="eval loss")
            if t_steps or e_steps:
                ax.legend(loc="upper right", fontsize=8)
        elif panel_id == "perplexity":
            p_steps, p_vals = series["perplexity"]
            ax.plot(p_steps, p_vals, "m-o", markersize=3)
        elif panel_id == "grad_norm":
            g_steps, g_vals = series["grad_norm"]
            ax.plot(g_steps, g_vals, "r-", alpha=0.7)
        elif panel_id == "lr":
            lr_steps, lr_vals = series["lr"]
            if lr_steps:
                ax.plot(lr_steps, lr_vals, color="orange", alpha=0.8)
    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


