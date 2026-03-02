#!/usr/bin/env python3
"""
Unified Gradio dashboard: training plots + model evaluation.
Runs locally on 127.0.0.1. Air-gapped friendly.
"""

import argparse
import json
import os
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Distillation dashboard")
    p.add_argument("--runs_dir", type=str, default=".", help="Parent dir of training outputs")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def find_run_dirs(runs_dir):
    """Find directories containing trainer_state.json."""
    root = Path(runs_dir)
    if not root.exists():
        return []
    found = []
    for d in root.iterdir():
        if d.is_dir() and (d / "trainer_state.json").exists():
            found.append(str(d))
    # Also check runs_dir itself
    if (root / "trainer_state.json").exists():
        found.insert(0, str(root))
    return sorted(set(found))


def find_pipeline_dirs(runs_dir):
    """
    Find directories suitable for the pipeline view.
    A directory qualifies if it looks like a distillation output:
    - has trainer_state.json at root or inside a checkpoint subdir, OR
    - has *.gguf files AND (config.json or training_args.bin)
    Returns absolute paths.
    """
    root = Path(runs_dir).resolve()
    if not root.exists():
        return []
    found = set()
    candidates = [root] + [d for d in root.iterdir() if d.is_dir()]
    for d in candidates:
        if (d / "trainer_state.json").exists():
            found.add(str(d))
            continue
        # trainer_state.json inside a checkpoint subdir
        try:
            if any((sub / "trainer_state.json").exists()
                   for sub in d.iterdir() if sub.is_dir()):
                found.add(str(d))
                continue
        except PermissionError:
            continue
        # GGUF files alongside a config.json or training_args.bin (distill output)
        if list(d.glob("*.gguf")) and (
            (d / "config.json").exists() or (d / "training_args.bin").exists()
        ):
            found.add(str(d))
    return sorted(found)


def load_trainer_state(output_dir):
    path = Path(output_dir) / "trainer_state.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


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
    state = load_trainer_state(run_path)
    return plot_from_state(state, title=Path(run_path).name)


def select_and_load_model(path, model_state):
    """Load model into model_state; return status string."""
    import logging
    log = logging.getLogger(__name__)
    if not path or "(no " in path or not os.path.isdir(path):
        return "Select a model directory"
    path = os.path.abspath(path)
    config_path = Path(path) / "config.json"
    if not config_path.exists():
        log.warning("Model dir missing config.json: %s", path)
        return f"Invalid: no config.json in {Path(path).name}"
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, local_files_only=True)
        if torch.backends.mps.is_available():
            model = model.to("mps")
        elif torch.cuda.is_available():
            model = model.to("cuda")
        model_state[0], model_state[1] = model, tok
        log.info("Loaded model: %s", Path(path).name)
        return f"Loaded: {Path(path).name}"
    except Exception as e:
        log.warning("Model load failed %s: %s", path, e)
        return f"Failed: {e}"


def build_eval_ui(runs_dir, model_state):
    run_dirs = find_run_dirs(runs_dir)
    model_choices = [d for d in run_dirs if (Path(d) / "config.json").exists()]
    if not model_choices:
        model_choices = ["(no distilled models found)"]

    def generate(prompt, max_tokens, temperature):
        model, tok = model_state[0], model_state[1]
        if model is None or tok is None:
            return "Load a model first."
        if not (prompt or "").strip():
            return ""
        inputs = tok(prompt, return_tensors="pt")
        if model.device.type in ("mps", "cuda"):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        out = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens or 128),
            do_sample=True,
            temperature=float(temperature or 0.7),
            pad_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def on_model_select(path):
        return select_and_load_model(path, model_state)

    gr.Markdown("### Evaluate distilled model")
    model_dropdown = gr.Dropdown(
        choices=model_choices,
        value=model_choices[0] if model_choices else None,
        label="Model",
    )
    initial_status = ""
    if model_choices and "(no " not in model_choices[0]:
        initial_status = on_model_select(model_choices[0])
    load_status = gr.Textbox(label="Status", interactive=False, value=initial_status)
    model_dropdown.change(on_model_select, model_dropdown, load_status)
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", placeholder="Enter your prompt...", lines=3)
    with gr.Row():
        max_tok = gr.Slider(32, 512, value=128, step=32, label="Max tokens")
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
    gen_btn = gr.Button("Generate")
    output_box = gr.Textbox(label="Generated", lines=6)
    gen_btn.click(generate, [prompt_in, max_tok, temp], output_box)


def main():
    args = parse_args()
    model_state = [None, None]  # [model, tokenizer]
    run_dirs = find_run_dirs(args.runs_dir)
    pipeline_dirs = find_pipeline_dirs(args.runs_dir)

    with gr.Blocks(title="Distillation Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Distillation Dashboard")
        gr.Markdown("Training curves and model evaluation. Runs locally only.")
        with gr.Tabs():
            with gr.Tab("Plots"):
                gr.Markdown("### Training curves")
                run_dropdown = gr.Dropdown(
                    choices=run_dirs,
                    value=run_dirs[0] if run_dirs else None,
                    label="Run directory",
                )
                plot_output = gr.Plot(
                    label="Loss & learning rate",
                    value=on_run_select(run_dirs[0]) if run_dirs else None,
                )
                run_dropdown.change(on_run_select, run_dropdown, plot_output)
            with gr.Tab("Pipeline"):
                gr.Markdown("### End-to-end pipeline summary")
                pipeline_run_dd = gr.Dropdown(
                    choices=pipeline_dirs,
                    value=pipeline_dirs[0] if pipeline_dirs else None,
                    label="Run directory",
                )
                pipeline_plot = gr.Plot(label="Pipeline summary")
                refresh_btn = gr.Button("Refresh")

                def on_pipeline_select(run_path):
                    if not run_path:
                        return None
                    import sys
                    import os
                    scripts_dir = os.path.dirname(os.path.abspath(__file__))
                    if scripts_dir not in sys.path:
                        sys.path.insert(0, scripts_dir)
                    from plot_gguf_pipeline import plot_pipeline
                    return plot_pipeline(run_path)

                pipeline_run_dd.change(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                refresh_btn.click(on_pipeline_select, pipeline_run_dd, pipeline_plot)
                if pipeline_dirs:
                    pipeline_plot.value = on_pipeline_select(pipeline_dirs[0])

            with gr.Tab("Evaluate"):
                build_eval_ui(args.runs_dir, model_state)
    app.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
