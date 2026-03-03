#!/usr/bin/env python3
"""
Gradio UI for evaluating distilled models.
Runs locally on 127.0.0.1 only (no public share).
"""

import argparse
import os

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str, default="./distilled-minillm")
    p.add_argument("--port", type=int, default=7860)
    return p.parse_args()


def load_model(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Resolve to absolute path so HuggingFace treats it as local, not a repo ID
    path = os.path.abspath(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, local_files_only=True)
    if torch.backends.mps.is_available():
        model = model.to("mps")
    elif torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=128):
    inputs = tokenizer(prompt, return_tensors="pt")
    if model.device.type in ("mps", "cuda"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.7)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    args = parse_args()
    path = os.path.abspath(args.model_path)
    if not os.path.isdir(path):
        print(f"Error: Model directory not found: {path}")
        print("Run distill_minillm.py first to create a distilled model, or pass an existing path with --model_path")
        raise SystemExit(1)
    print(f"Loading model from {path}")
    model, tokenizer = load_model(path)

    import gradio as gr

    def fn(prompt, max_tokens):
        if not prompt.strip():
            return ""
        return generate(model, tokenizer, prompt, max_new_tokens=int(max_tokens or 128))

    iface = gr.Interface(
        fn=fn,
        inputs=[
            gr.Textbox(label="Prompt", placeholder="Enter your prompt..."),
            gr.Slider(32, 512, value=128, step=32, label="Max tokens"),
        ],
        outputs=gr.Textbox(label="Generated"),
        title="Distilled Model - Local Eval",
        description="Evaluate your distilled model. Runs locally only.",
    )
    iface.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
