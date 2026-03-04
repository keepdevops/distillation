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
 HEAD
        # Live run: metrics.jsonl exists (trainer_state.json not yet written)
        if (d / "metrics.jsonl").exists():
            found.add(str(d))
            continue
=======
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
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
    import sys
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from plot_training import load_metrics, extract_series
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


def select_and_load_model(path, model_state):
 HEAD
    """Load model into model_state; return status string.

    path can be a local directory or a HuggingFace model ID.
    """
    import logging
    log = logging.getLogger(__name__)
    if not path or "(no " in path:
        return "Select a model above."

    path = path.strip()

    # Determine if it's a local path or HF id
    is_local = os.path.isdir(path)
    if is_local:
        path = os.path.abspath(path)
        config_path = Path(path) / "config.json"
        if not config_path.exists():
            log.warning("Model dir missing config.json: %s", path)
            return f"Invalid: no config.json in {Path(path).name}"
        label = Path(path).name
    else:
        # Treat as HuggingFace model id
        label = path

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        local_only = is_local  # only force local_files_only for local paths
        tok = AutoTokenizer.from_pretrained(path, local_files_only=local_only)
        model = AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16,
            device_map="auto" if not torch.backends.mps.is_available() else None,
            local_files_only=local_only,
        )
        if torch.backends.mps.is_available():
            model = model.to("mps")
        model_state[0], model_state[1] = model, tok
        log.info("Loaded model: %s", label)
        return f"Loaded: {label}"
    except Exception as e:
        log.warning("Model load failed %s: %s", path, e)
        return f"Failed to load '{label}': {e}"


def _diversity_metrics(text):
    """Return (distinct_1, distinct_2, max_rep) for a generated text."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0, 0.0, 0
    d1 = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    max_run = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        max_run = max(max_run, run)
    return d1, d2, max_run if len(tokens) > 1 else 0


def _is_hf_model_dir(d: Path) -> bool:
    """Return True if d looks like a complete HuggingFace model directory."""
    if not (d / "config.json").exists():
        return False
    return bool(
        list(d.glob("*.safetensors"))
        or list(d.glob("model*.bin"))
        or (d / "pytorch_model.bin").exists()
    )


def _scan_hf_hub_cache(hub_root: Path) -> list[tuple[str, str]]:
    """Yield (display_label, abs_path) for every model snapshot in an HF hub cache dir."""
    results = []
    if not hub_root.exists():
        return results
    try:
        for entry in hub_root.iterdir():
            if not entry.is_dir() or not entry.name.startswith("models--"):
                continue
            # models--Org--Name  →  Org/Name
            label = entry.name[len("models--"):].replace("--", "/", 1)
            snaps = entry / "snapshots"
            if not snaps.exists():
                continue
            try:
                for snap in sorted(snaps.iterdir(),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                    if snap.is_dir() and _is_hf_model_dir(snap):
                        results.append((label, str(snap)))
                        break  # only latest snapshot per model
            except PermissionError:
                continue
    except PermissionError:
        pass
    return results


def _discover_all_models(runs_dir: str) -> list[tuple[str, str]]:
    """Return (display_label, abs_path) for every usable model, deduplicated.

    Sources (in priority order — earlier sources win on label conflicts):
      1. Local trained outputs under runs_dir (up to 3 levels deep)
      2. System HF hub cache  (~/.cache/huggingface/hub  or $HF_HOME/hub)
      3. Project-local hf_cache/  next to runs_dir
    """
    seen_paths: set[str] = set()
    seen_labels: set[str] = set()
    results: list[tuple[str, str]] = []

    def _add(label: str, path: str) -> None:
        """Add (label, path) deduplicating by both path and label."""
        if path in seen_paths or not Path(path).exists():
            return
        # For HF cache models: deduplicate by label (same model, different cache copy)
        if label in seen_labels:
            return
        seen_paths.add(path)
        seen_labels.add(label)
        results.append((label, path))

    # ── 1. Local trained outputs ─────────────────────────────────────────────
    root = Path(runs_dir).resolve()
    candidates: list[Path] = []
    try:
        candidates = [root] + [
            d for d in root.rglob("*")
            if d.is_dir() and len(d.relative_to(root).parts) <= 3
        ]
    except PermissionError:
        pass
    # Sort newest-first so most recent trained models appear first
    candidates.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    for d in candidates:
        if _is_hf_model_dir(d):
            try:
                rel = d.relative_to(root)
                label = str(rel) if str(rel) != "." else d.name
            except ValueError:
                label = d.name
            _add(label, str(d))

    # ── 2. System HF hub cache ────────────────────────────────────────────────
    hf_home = os.environ.get("HF_HOME") or str(
        Path.home() / ".cache" / "huggingface"
    )
    for label, path in _scan_hf_hub_cache(Path(hf_home) / "hub"):
        _add(label, path)

    # ── 3. Project-local hf_cache dirs ───────────────────────────────────────
    for local_cache in [
        root / "hf_cache",
        root.parent / "hf_cache",
        root / "scripts" / "hf_cache",
    ]:
        for label, path in _scan_hf_hub_cache(local_cache / "hub"):
            _add(label, path)
        for label, path in _scan_hf_hub_cache(local_cache):
            _add(label, path)

    return results


def build_eval_ui(runs_dir, model_state):
    _discovered = _discover_all_models(runs_dir)
    # gr.Dropdown accepts (label, value) tuples
    model_choices = [(label, path) for label, path in _discovered] if _discovered else [("(no models found)", "")]

    judge_state = [None, None]  # [judge_model, judge_tokenizer]
=======
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
    model_choices = [d for d in find_pipeline_dirs(runs_dir) if (Path(d) / "config.json").exists()]
    if not model_choices:
        model_choices = ["(no distilled models found)"]
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d

    def generate(prompt, max_tokens, temperature):
        model, tok = model_state[0], model_state[1]
        if model is None or tok is None:
 HEAD
            return "Load a model first.", ""
        if not (prompt or "").strip():
            return "", ""
=======
            return "Load a model first."
        if not (prompt or "").strip():
            return ""
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
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
 HEAD
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        d1, d2, max_rep = _diversity_metrics(text)
        flag = ""
        if d1 < 0.5:
            flag += "  ⚠ low distinct-1 (possible mode collapse)"
        if max_rep > 5:
            flag += f"  ⚠ max-rep={max_rep} (repetition loop)"
        metrics_str = (
            f"distinct-1: {d1:.3f}  |  distinct-2: {d2:.3f}  |  max-rep: {max_rep}{flag}"
        )
        return text, metrics_str

    def load_judge(judge_path):
        import logging
        log = logging.getLogger(__name__)
        if not (judge_path or "").strip():
            return "Enter a teacher model path or HF id"
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            jm = AutoModelForCausalLM.from_pretrained(
                judge_path.strip(), dtype=torch.bfloat16,
                device_map="auto", local_files_only=False,
            )
            jt = AutoTokenizer.from_pretrained(judge_path.strip())
            jt.pad_token = jt.eos_token
            if torch.backends.mps.is_available():
                jm = jm.to("mps")
            elif torch.cuda.is_available():
                jm = jm.to("cuda")
            judge_state[0], judge_state[1] = jm, jt
            log.info("Loaded judge: %s", judge_path.strip())
            return f"Judge loaded: {judge_path.strip()}"
        except Exception as e:
            log.warning("Judge load failed: %s", e)
            return f"Failed: {e}"

    def run_judge(prompt, response):
        import re
        jm, jt = judge_state[0], judge_state[1]
        if jm is None:
            return "Load a judge model first."
        if not (response or "").strip():
            return "Generate a response first."
        judge_prompt = (
            "You are evaluating an AI assistant's response.\n\n"
            f"Instruction: {(prompt or '').strip()}\n"
            f"Response: {response.strip()}\n\n"
            "Rate the response 1-10 for instruction-following and overall quality. "
            "Reply with the score first, then a one-sentence reason. Example: '8 - Clear and direct.'"
        )
        inputs = jt(judge_prompt, return_tensors="pt", truncation=True, max_length=768)
        if jm.device.type in ("mps", "cuda"):
            inputs = {k: v.to(jm.device) for k, v in inputs.items()}
        out = jm.generate(**inputs, max_new_tokens=60, do_sample=False,
                          pad_token_id=jt.eos_token_id)
        raw = jt.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        m = re.search(r"\b([1-9]|10)\b", raw)
        score = int(m.group(1)) if m else None
        prefix = f"Score: {score}/10  —  " if score is not None else ""
        return prefix + raw

    def on_model_select(path):
        if not path or not path.strip():
            return "No model selected."
        return select_and_load_model(path, model_state)

    def refresh_models(custom_path):
        """Re-scan all model sources and prepend any manually entered path."""
        new_discovered = _discover_all_models(runs_dir)
        new_choices = [(label, path) for label, path in new_discovered]
        # Prepend custom path if it's a valid dir or HF id
        if custom_path and custom_path.strip():
            cp = custom_path.strip()
            existing_paths = {path for _, path in new_choices}
            abs_cp = os.path.abspath(cp) if os.path.isdir(cp) else cp
            if abs_cp not in existing_paths:
                new_choices.insert(0, (cp, abs_cp if os.path.isdir(cp) else cp))
        if not new_choices:
            new_choices = [("(no models found)", "")]
        first_path = new_choices[0][1]
        status = (select_and_load_model(first_path, model_state)
                  if first_path else "No models found. Run a distillation first.")
        return gr.update(choices=new_choices, value=first_path), status

    def load_custom_path(custom_path):
        """Load a model directly from a manually entered path or HF id."""
        if not (custom_path or "").strip():
            return "Enter a path above."
        return select_and_load_model(custom_path.strip(), model_state)

    first_path = model_choices[0][1] if model_choices else ""

    gr.Markdown("### Evaluate distilled model")
    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=model_choices,
            value=first_path if first_path else None,
            label="Auto-detected models",
            scale=4,
        )
        refresh_btn = gr.Button("Refresh", scale=1)
    with gr.Row():
        custom_path_in = gr.Textbox(
            label="Or enter model path / HF id manually",
            placeholder="e.g. ./distilled-minillm  or  Qwen/Qwen2-0.5B-Instruct",
            scale=4,
        )
        load_path_btn = gr.Button("Load", scale=1)
    _status_default = f"Auto-detected {len(model_choices)} model(s). Select one to load." if model_choices and model_choices[0][1] else "No models found. Run a distillation first."
    load_status = gr.Textbox(label="Status", interactive=False, value=_status_default)
    model_dropdown.change(on_model_select, model_dropdown, load_status)
    refresh_btn.click(refresh_models, custom_path_in, [model_dropdown, load_status])
    load_path_btn.click(load_custom_path, custom_path_in, load_status)

=======
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
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", placeholder="Enter your prompt...", lines=3)
    with gr.Row():
        max_tok = gr.Slider(32, 512, value=128, step=32, label="Max tokens")
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
    gen_btn = gr.Button("Generate")
    output_box = gr.Textbox(label="Generated", lines=6)
 HEAD
    diversity_box = gr.Textbox(label="Diversity metrics", interactive=False)
    gen_btn.click(generate, [prompt_in, max_tok, temp], [output_box, diversity_box])

    gr.Markdown("#### LLM-as-judge")
    gr.Markdown("Load a teacher model to rate the last generated response.")
    with gr.Row():
        judge_path_in = gr.Textbox(
            label="Judge model (HF id or local path)",
            value="Qwen/Qwen2-1.5B-Instruct",
            scale=4,
        )
        load_judge_btn = gr.Button("Load judge", scale=1)
    judge_status = gr.Textbox(label="Judge status", interactive=False)
    load_judge_btn.click(load_judge, judge_path_in, judge_status)
    judge_btn = gr.Button("Judge last response")
    judge_output = gr.Textbox(label="Judge verdict", interactive=False, lines=3)
    judge_btn.click(run_judge, [prompt_in, output_box], judge_output)
=======
    gen_btn.click(generate, [prompt_in, max_tok, temp], output_box)
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d


def main():
    args = parse_args()
    model_state = [None, None]  # [model, tokenizer]
    run_dirs = find_run_dirs(args.runs_dir)
    pipeline_dirs = find_pipeline_dirs(args.runs_dir)
 HEAD
    with gr.Blocks(title="Distillation Dashboard") as app:
=======
    with gr.Blocks(title="Distillation Dashboard", theme=gr.themes.Soft()) as app:
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
        gr.Markdown("# Distillation Dashboard")
        gr.Markdown("Training curves and model evaluation. Runs locally only.")
        with gr.Tabs():
            with gr.Tab("Plots"):
                gr.Markdown("### Training curves")
 HEAD
                with gr.Row():
                    run_dropdown = gr.Dropdown(
                        choices=pipeline_dirs,
                        value=pipeline_dirs[0] if pipeline_dirs else None,
                        label="Run directory",
                        scale=4,
                    )
                    plots_refresh_btn = gr.Button("Refresh", scale=1)
=======
                run_dropdown = gr.Dropdown(
                    choices=pipeline_dirs,
                    value=pipeline_dirs[0] if pipeline_dirs else None,
                    label="Run directory",
                )
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
                plot_output = gr.Plot(
                    label="Loss & learning rate",
                    value=on_run_select(pipeline_dirs[0]) if pipeline_dirs else None,
                )
                run_dropdown.change(on_run_select, run_dropdown, plot_output)
 HEAD
                plots_refresh_btn.click(on_run_select, run_dropdown, plot_output)
=======
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d
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
 HEAD
            with gr.Tab("Thermal"):
                gr.Markdown("### CPU / GPU temperature & power over time")
                thermal_log_box = gr.Textbox(
                    label="Log file path",
                    value=str(Path(args.runs_dir).parent / "thermal.log"),
                    placeholder="/path/to/thermal.log",
                )
                thermal_refresh_btn = gr.Button("Load / Refresh")
                thermal_plot = gr.Plot(label="Thermal history")

                def load_thermal(log_path):
                    import csv
                    from datetime import datetime
                    p = Path(log_path)
                    if not p.exists():
                        return None
                    rows = []
                    try:
                        with open(p, newline="") as f:
                            reader = csv.DictReader(f)
                            for row in reader:
                                rows.append(row)
                    except OSError:
                        return None
                    if not rows:
                        return None
                    # Detect format by header
                    headers = list(rows[0].keys())
                    has_temp = "cpu_temp_c" in headers
                    times, cpu_t, gpu_t, soc_t, cpu_w, gpu_w, tot_w = [], [], [], [], [], [], []
                    for row in rows:
                        try:
                            t = datetime.strptime(row["time"].strip(), "%Y-%m-%d %H:%M:%S")
                        except (ValueError, KeyError):
                            continue
                        times.append(t)
                        if has_temp:
                            def _f(v):
                                try: return float(v)
                                except: return float("nan")
                            cpu_t.append(_f(row.get("cpu_temp_c", "")))
                            gpu_t.append(_f(row.get("gpu_temp_c", "")))
                            soc_t.append(_f(row.get("soc_temp_c", "")))
                            cpu_w.append(_f(row.get("cpu_power_w", "")))
                            gpu_w.append(_f(row.get("gpu_power_w", "")))
                            tot_w.append(_f(row.get("total_power_w", "")))
                    if not times:
                        return None
                    panels = [("temp", "Temperature (°C)"), ("power", "Power (W)")]
                    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
                    fig.suptitle(f"Thermal — {p.name}", fontsize=11)
                    ax0, ax1 = axes
                    if has_temp:
                        ax0.plot(times, cpu_t, label="CPU", color="steelblue")
                        ax0.plot(times, gpu_t, label="GPU", color="tomato")
                        ax0.plot(times, soc_t, label="SOC", color="goldenrod", linestyle="--", alpha=0.6)
                        ax0.axhline(90, color="red", linestyle=":", linewidth=1, label="pause threshold (90°C)")
                    ax0.set_ylabel("°C")
                    ax0.legend(loc="upper left", fontsize=8)
                    ax0.grid(True, alpha=0.3)
                    if has_temp:
                        ax1.plot(times, cpu_w, label="CPU", color="steelblue")
                        ax1.plot(times, gpu_w, label="GPU", color="tomato")
                        ax1.plot(times, tot_w, label="Total", color="gray", linestyle="--", alpha=0.7)
                    ax1.set_ylabel("W")
                    ax1.set_xlabel("Time")
                    ax1.legend(loc="upper left", fontsize=8)
                    ax1.grid(True, alpha=0.3)
                    fig.autofmt_xdate(rotation=30)
                    plt.tight_layout()
                    return fig

                thermal_refresh_btn.click(load_thermal, thermal_log_box, thermal_plot)
                # Auto-load on startup
                _default_thermal = str(Path(args.runs_dir).parent / "thermal.log")

            with gr.Tab("Evaluate"):
                build_eval_ui(args.runs_dir, model_state)
            with gr.Tab("Quality"):
                gr.Markdown("### Quality metrics (from eval_quality.py)")
                gr.Markdown(
                    "Run `python scripts/eval_quality.py <model_dir> --judge` to generate this report."
                )
                with gr.Row():
                    quality_run_dd = gr.Dropdown(
                        choices=pipeline_dirs,
                        value=pipeline_dirs[0] if pipeline_dirs else None,
                        label="Run directory",
                        scale=4,
                    )
                    quality_refresh_btn = gr.Button("Load / Refresh", scale=1)
                quality_summary = gr.Textbox(label="Summary", interactive=False, lines=8)
                quality_samples = gr.Dataframe(
                    headers=["prompt", "response", "distinct_1", "distinct_2", "max_rep", "judge_score"],
                    label="Samples",
                    wrap=True,
                )

                def load_quality(run_path):
                    if not run_path:
                        return "No run selected.", []
                    qfile = Path(run_path) / "quality_metrics.json"
                    if not qfile.exists():
                        return f"quality_metrics.json not found in {Path(run_path).name}", []
                    try:
                        with open(qfile) as f:
                            data = json.load(f)
                    except Exception as e:
                        return f"Error reading file: {e}", []

                    div = data.get("diversity", {})
                    judge = data.get("judge", {})
                    lines = [
                        f"Model:       {Path(data.get('model_dir', run_path)).name}",
                        f"Timestamp:   {data.get('timestamp', 'n/a')}",
                        f"N samples:   {data.get('n_samples', '?')}",
                        "",
                        "── Diversity ──────────────────────────",
                        f"  avg distinct-1:   {div.get('avg_distinct_1', 'n/a')}",
                        f"  avg distinct-2:   {div.get('avg_distinct_2', 'n/a')}",
                        f"  avg max-rep:      {div.get('avg_max_rep', 'n/a')}",
                        f"  median length:    {div.get('median_response_tokens', 'n/a')} tokens",
                    ]
                    if judge.get("enabled"):
                        lines += [
                            "",
                            "── LLM-as-judge ────────────────────────",
                            f"  teacher:    {judge.get('teacher', 'n/a')}",
                            f"  avg score:  {judge.get('avg_score', 'n/a')} / 10",
                            f"  scored:     {judge.get('n_scored', 0)} / {data.get('n_samples', '?')}",
                        ]
                    else:
                        lines.append("\n(Judge not run — rerun with --judge to enable)")

                    rows = []
                    for s in data.get("samples", []):
                        rows.append([
                            (s.get("instruction") or s.get("prompt", ""))[:120],
                            s.get("response", "")[:200],
                            s.get("distinct_1", ""),
                            s.get("distinct_2", ""),
                            s.get("max_rep", ""),
                            s.get("judge_score", ""),
                        ])
                    return "\n".join(lines), rows

                quality_run_dd.change(load_quality, quality_run_dd, [quality_summary, quality_samples])
                quality_refresh_btn.click(load_quality, quality_run_dd, [quality_summary, quality_samples])
                if pipeline_dirs:
                    _qs, _qr = load_quality(pipeline_dirs[0])
                    quality_summary.value = _qs

            with gr.Tab("Experiments"):
                gr.Markdown("### Experiment history (from experiment_log.jsonl)")
                gr.Markdown(
                    "Populated automatically by `run_distillation_agent.py --log_experiment`. "
                    "Run `python scripts/experiment_log.py --show 20` in a terminal for a quick view."
                )
                with gr.Row():
                    exp_log_path = gr.Textbox(
                        label="experiment_log.jsonl path",
                        value=str(Path(args.runs_dir).parent / "experiment_log.jsonl"),
                        scale=4,
                    )
                    exp_refresh_btn = gr.Button("Load / Refresh", scale=1)
                exp_table = gr.Dataframe(
                    headers=["run_id", "date", "backend", "epochs",
                              "eval_ppl", "ppl_gap%", "judge", "wt2_ppl", "outcome"],
                    label="Runs",
                    wrap=False,
                )
                exp_trend_plot = gr.Plot(label="Metric trends over runs")

                def load_experiments(log_path):
                    p = Path(log_path)
                    if not p.exists():
                        return [], None
                    records = []
                    try:
                        with open(p) as f:
                            for line in f:
                                line = line.strip()
                                if line:
                                    try:
                                        records.append(json.loads(line))
                                    except json.JSONDecodeError:
                                        continue
                    except OSError:
                        return [], None

                    rows = []
                    for r in records:
                        cfg = r.get("config", {})
                        m = r.get("metrics", {})

                        def _s(v, fmt=".2f"):
                            try:
                                return format(float(v), fmt)
                            except (TypeError, ValueError):
                                return ""

                        rows.append([
                            r.get("run_id", "?")[:30],
                            r.get("timestamp", "")[:10],
                            cfg.get("backend", "?"),
                            cfg.get("epochs", "?"),
                            _s(m.get("eval_perplexity")),
                            _s(m.get("ppl_gap_pct"), ".1f"),
                            _s(m.get("judge_avg_score"), ".1f"),
                            _s(m.get("wikitext2_perplexity")),
                            r.get("outcome", "?"),
                        ])

                    # Trend plot
                    fig = None
                    if len(records) >= 2:
                        timestamps = [r.get("timestamp", "")[:10] for r in records]
                        ppls = [r.get("metrics", {}).get("eval_perplexity") for r in records]
                        judges = [r.get("metrics", {}).get("judge_avg_score") for r in records]
                        wt2 = [r.get("metrics", {}).get("wikitext2_perplexity") for r in records]

                        has_judge = any(j is not None for j in judges)
                        has_wt2 = any(w is not None for w in wt2)
                        n_panels = 1 + (1 if has_judge else 0) + (1 if has_wt2 else 0)

                        fig, axes = plt.subplots(n_panels, 1, figsize=(9, 3 * n_panels), squeeze=False)
                        fig.suptitle("Experiment trends", fontsize=11)
                        ax_idx = 0

                        ax = axes[ax_idx][0]
                        x = list(range(len(records)))
                        ax.plot(x, [p or float("nan") for p in ppls], "b-o", markersize=4,
                                label="eval_perplexity")
                        ax.set_xticks(x)
                        ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
                        ax.set_ylabel("Eval perplexity")
                        ax.grid(True, alpha=0.3)
                        ax_idx += 1

                        if has_wt2:
                            ax = axes[ax_idx][0]
                            ax.plot(x, [w or float("nan") for w in wt2], "g-o", markersize=4,
                                    label="wikitext2_ppl")
                            ax.set_xticks(x)
                            ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
                            ax.set_ylabel("WikiText-2 PPL")
                            ax.grid(True, alpha=0.3)
                            ax_idx += 1

                        if has_judge:
                            ax = axes[ax_idx][0]
                            ax.plot(x, [j or float("nan") for j in judges], "m-o", markersize=4,
                                    label="judge_avg_score")
                            ax.set_xticks(x)
                            ax.set_xticklabels(timestamps, rotation=30, ha="right", fontsize=7)
                            ax.set_ylabel("Judge score / 10")
                            ax.set_ylim(0, 10)
                            ax.grid(True, alpha=0.3)

                        plt.tight_layout()

                    return rows, fig

                exp_refresh_btn.click(load_experiments, exp_log_path, [exp_table, exp_trend_plot])
                _default_exp = str(Path(args.runs_dir).parent / "experiment_log.jsonl")
                if Path(_default_exp).exists():
                    _er, _ef = load_experiments(_default_exp)
                    if _er:
                        exp_table.value = _er
                    if _ef:
                        exp_trend_plot.value = _ef

    app.launch(server_name="127.0.0.1", server_port=args.port, theme=gr.themes.Soft())
=======
            with gr.Tab("Evaluate"):
                build_eval_ui(args.runs_dir, model_state)
    app.launch(server_name="127.0.0.1", server_port=args.port)
 8b1ec5e8f369b5d44422b10b10c3a14a59bad90d


if __name__ == "__main__":
    main()
