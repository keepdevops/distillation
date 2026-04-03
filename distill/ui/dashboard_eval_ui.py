"""Gradio eval accordion UI for the distillation dashboard."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr

from .dashboard_models import select_and_load_model, _diversity_metrics, _discover_all_models

def build_eval_ui(runs_dir, model_state):
    _discovered = _discover_all_models(runs_dir)
    # gr.Dropdown accepts (label, value) tuples
    model_choices = [(label, path) for label, path in _discovered] if _discovered else [("(no models found)", "")]

    judge_state = [None, None, None]  # [judge_model, judge_tokenizer, backend]

    def generate(prompt, max_tokens, temperature):
        loader = model_state[0]
        if loader is None or not isinstance(loader, UniversalModelLoader):
            return "Load a model first.", ""
        if loader.model is None:
            return "Load a model first.", ""
        if not (prompt or "").strip():
            return "", ""

        try:
            text = loader.generate(
                prompt,
                max_tokens=int(max_tokens or 128),
                temperature=float(temperature or 0.7)
            )

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
        except Exception as e:
            return f"Generation error: {str(e)}", ""

    def load_judge(judge_path):
        import logging
        log = logging.getLogger(__name__)
        if not (judge_path or "").strip():
            return "Enter a teacher model path or HF id"
        path = judge_path.strip()
        try:
            # Prefer GGUF if path is a .gguf file or directory containing one
            gguf_path = find_gguf(path)
            if gguf_path:
                from llama_cpp import Llama
                jm = Llama(model_path=gguf_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)
                judge_state[0], judge_state[1], judge_state[2] = jm, None, "gguf"
                log.info("Loaded judge (GGUF/Metal): %s", gguf_path)
                return f"Judge loaded (GGUF/Metal): {Path(gguf_path).name}"
            elif is_mlx_available() and os.path.isdir(path):
                jm, jt = load_mlx_model(path)
                judge_state[0], judge_state[1], judge_state[2] = jm, jt, "mlx"
                log.info("Loaded judge (MLX): %s", path)
                return f"Judge loaded (MLX): {path}"
            else:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                jm = AutoModelForCausalLM.from_pretrained(
                    path, dtype=torch.bfloat16,
                    device_map="auto", local_files_only=False,
                )
                jt = AutoTokenizer.from_pretrained(path)
                jt.pad_token = jt.eos_token
                if torch.backends.mps.is_available():
                    jm = jm.to("mps")
                elif torch.cuda.is_available():
                    jm = jm.to("cuda")
                judge_state[0], judge_state[1], judge_state[2] = jm, jt, "pytorch"
                log.info("Loaded judge (PyTorch): %s", path)
                return f"Judge loaded (PyTorch): {path}"
        except Exception as e:
            log.warning("Judge load failed: %s", e)
            return f"Failed: {e}"

    def run_judge(prompt, response):
        import re
        jm, jt, j_backend = judge_state[0], judge_state[1], judge_state[2]
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
        if j_backend == "gguf":
            out = jm(judge_prompt, max_tokens=60, temperature=0.01, echo=False)
            raw = out["choices"][0]["text"].strip()
        elif j_backend == "mlx":
            results = mlx_generate_responses(jm, jt, [judge_prompt], max_new_tokens=60, temperature=0.0)
            raw = results[0].strip()
        else:
            import torch
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

    with gr.Row():
        prompt_in = gr.Textbox(label="Prompt", placeholder="Enter your prompt...", lines=3)
    with gr.Row():
        max_tok = gr.Slider(32, 512, value=128, step=32, label="Max tokens")
        temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature")
    gen_btn = gr.Button("Generate")
    output_box = gr.Textbox(label="Generated", lines=6)
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


