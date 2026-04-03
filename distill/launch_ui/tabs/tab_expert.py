"""Expert Pipeline tab widget layout."""
from __future__ import annotations

import gradio as gr
import pandas as pd
from pathlib import Path

from ...infra.paths import project_dir
from ..presets import KNOWN_STUDENTS

PROJECT_DIR = project_dir()

_EP_HF_DATASETS = [
    # Legal
    "nguha/legalbench",
    "nelson-liu/legalbench",
    # Tax
    "Atome-LLM/Tax-Policy-Analysis",
    # Medical
    "medalpaca/medical_meadow_medical_flashcards",
    "medalpaca/medical_meadow_wikidoc",
    "pubmed_qa",
    # Finance
    "gbharti/finance-alpaca",
    "FinGPT/fingpt-sentiment-train",
    # Coding
    "iamtarun/python_code_instructions_18k_alpaca",
    "sahil2801/CodeAlpaca-20k",
    # General
    "yahma/alpaca-cleaned",
    "tatsu-lab/alpaca",
    "HuggingFaceH4/ultrachat_200k",
]

_EP_LOCAL_CANDIDATES = [
    "./domain_data/expert_remapped",
    "./domain_data/tax",
    "./domain_data/legal",
    "./domain_data/medical",
    "./domain_data/coding",
]


def _ep_dataset_choices():
    local = [p for p in _EP_LOCAL_CANDIDATES
             if (PROJECT_DIR / p.lstrip("./")).exists()]
    return _EP_HF_DATASETS + local


def build_tab_expert(students):
    """Build the Expert Pipeline tab.

    Parameters
    ----------
    students:  list of student model choices (used as fallback; KNOWN_STUDENTS
               is the primary list for the distillation step dropdown)

    Returns
    -------
    dict of all expert pipeline widgets required for event wiring, plus the
    helper ``_ep_dataset_choices`` callable needed by wiring refresh handlers.
    """
    with gr.Tab("Expert Pipeline"):
        gr.Markdown(
            "**Domain-expert distillation pipeline** — loads any HF dataset, "
            "remaps columns, generates Chain-of-Thought rationales via a GGUF "
            "teacher (Metal-accelerated), then launches distillation.\n\n"
            "Run steps in order: **1 → 2 → 3 → 4**."
        )
        ep_progress = gr.HTML(value="")

        # ── Step 1: Dataset & Column Mapping ─────────────────────────────
        with gr.Accordion("Step 1 — Dataset & Column Mapping", open=True):
            with gr.Row():
                ep_dataset = gr.Dropdown(
                    choices=_ep_dataset_choices(),
                    value="",
                    label="HF Dataset ID or local path",
                    allow_custom_value=True,
                    scale=4,
                )
                ep_dataset_refresh = gr.Button("⟳", scale=0, size="sm", min_width=40)
                ep_inspect_btn = gr.Button("Inspect", scale=1, size="sm")
            ep_inspect_status = gr.Textbox(
                value="", label="Dataset info", interactive=False, lines=2,
            )
            with gr.Row():
                ep_instruction_col = gr.Dropdown(
                    choices=[], allow_custom_value=True,
                    label="Instruction column", scale=1,
                )
                ep_output_col = gr.Dropdown(
                    choices=[], allow_custom_value=True,
                    label="Output column", scale=1,
                )
                ep_input_col = gr.Dropdown(
                    choices=["(none)"], value="(none)", allow_custom_value=True,
                    label="Input / context column (optional)", scale=1,
                )
            with gr.Row():
                ep_max_samples_remap = gr.Slider(
                    100, 50000, value=5000, step=100,
                    label="Max samples to load",
                )
                ep_remap_output = gr.Textbox(
                    value="./domain_data/expert_remapped",
                    label="Save remapped dataset to",
                )
            ep_remap_btn = gr.Button("Remap & Save Dataset", variant="primary")

        # ── Step 2: GGUF Teacher ──────────────────────────────────────────
        with gr.Accordion("Step 2 — GGUF Teacher", open=False):
            _gguf_choices = sorted(
                [str(p) for p in Path("/Users/Shared/llama/models").glob("*.gguf")]
            ) if Path("/Users/Shared/llama/models").exists() else []
            with gr.Row():
                ep_teacher = gr.Dropdown(
                    choices=_gguf_choices,
                    value=_gguf_choices[-1] if _gguf_choices else "",
                    allow_custom_value=True,
                    label="GGUF teacher model",
                    scale=3,
                )
                ep_teacher_refresh = gr.Button("Refresh", scale=1, size="sm")
            with gr.Row():
                ep_ctx_size   = gr.Slider(1024, 32768, value=8192, step=1024,
                                          label="Context size (tokens)")
                ep_n_parallel = gr.Slider(1, 8, value=4, step=1,
                                          label="Parallel server slots")
                ep_cot_temp   = gr.Slider(0.0, 1.5, value=0.3, step=0.05,
                                          label="Teacher temperature")
                ep_max_tokens = gr.Slider(256, 4096, value=1024, step=128,
                                          label="Max tokens per response")

        # ── Step 3: CoT Rationale Generation ─────────────────────────────
        with gr.Accordion("Step 3 — CoT Rationale Generation", open=False):
            with gr.Row():
                ep_domain = gr.Dropdown(
                    choices=["tax", "legal", "medical", "finance", "coding", "general"],
                    value="general",
                    label="Domain",
                    scale=1,
                )
                ep_n_cot = gr.Slider(50, 10000, value=1000, step=50,
                                     label="Samples to generate", scale=2)
            ep_system_prompt = gr.Textbox(
                value="",
                label="System prompt (leave blank to use domain default)",
                lines=6,
                placeholder="Leave blank to auto-fill from domain selection…",
            )
            ep_cot_output = gr.Textbox(
                value="./domain_data/expert_cot",
                label="CoT output directory",
            )
            ep_batch_size_cot = gr.Slider(4, 32, value=16, step=4,
                                          label="Batch size")
            with gr.Row():
                ep_cot_btn  = gr.Button("Generate CoT Rationales", variant="primary", scale=3)
                ep_cot_stop = gr.Button("Stop", variant="stop", scale=1)

        # ── Step 4: Distillation ──────────────────────────────────────────
        with gr.Accordion("Step 4 — Distillation", open=False):
            with gr.Row():
                ep_student = gr.Dropdown(
                    choices=KNOWN_STUDENTS,
                    value="Qwen/Qwen2-0.5B-Instruct",
                    allow_custom_value=True,
                    label="Student model",
                    scale=2,
                )
                ep_distill_backend = gr.Dropdown(
                    choices=["mlx", "pytorch", "unsloth"],
                    value="mlx",
                    label="Backend",
                    scale=1,
                )
            with gr.Row():
                ep_distill_dataset = gr.Textbox(
                    value="",
                    label="Dataset path (defaults to CoT output dir above)",
                    placeholder="Leave blank to use CoT output dir",
                )
                ep_distill_output = gr.Textbox(
                    value="./runs/expert-distilled",
                    label="Output directory",
                )
            with gr.Row():
                ep_epochs   = gr.Slider(1, 5, value=3, step=1, label="Epochs")
                ep_lora_r   = gr.Slider(8, 64, value=32, step=8, label="LoRA rank")
                ep_max_samp = gr.Slider(500, 10000, value=3000, step=500,
                                        label="Max training samples")
            with gr.Row():
                ep_open_chk    = gr.Checkbox(value=True, label="Open models (no HF login)")
                ep_offline_chk = gr.Checkbox(value=False, label="Offline mode")
            with gr.Row():
                ep_distill_btn  = gr.Button("Launch Distillation", variant="primary", scale=3)
                ep_distill_stop = gr.Button("Stop", variant="stop", scale=1)

        ep_status = gr.Textbox(value="idle", label="Status", interactive=False)

        # ── Embedded live output ──────────────────────────────────────────
        with gr.Row():
            ep_loss_plot = gr.LinePlot(
                value=pd.DataFrame({"step": [0], "loss": [0.0]}),
                x="step", y="loss",
                title="Training Loss",
                x_title="Step", y_title="Loss",
                height=220, min_width=300,
            )
            ep_grad_plot = gr.LinePlot(
                value=pd.DataFrame({"step": [0], "grad_norm": [0.0]}),
                x="step", y="grad_norm",
                title="Gradient Norm",
                x_title="Step", y_title="Grad Norm",
                height=220, min_width=300,
            )
        ep_log_box = gr.Textbox(
            value="",
            label="Live output",
            lines=20,
            max_lines=20,
            interactive=False,
            autoscroll=True,
        )

    return {
        "ep_progress": ep_progress,
        "ep_dataset": ep_dataset,
        "ep_dataset_refresh": ep_dataset_refresh,
        "ep_inspect_btn": ep_inspect_btn,
        "ep_inspect_status": ep_inspect_status,
        "ep_instruction_col": ep_instruction_col,
        "ep_output_col": ep_output_col,
        "ep_input_col": ep_input_col,
        "ep_max_samples_remap": ep_max_samples_remap,
        "ep_remap_output": ep_remap_output,
        "ep_remap_btn": ep_remap_btn,
        "ep_teacher": ep_teacher,
        "ep_teacher_refresh": ep_teacher_refresh,
        "ep_ctx_size": ep_ctx_size,
        "ep_n_parallel": ep_n_parallel,
        "ep_cot_temp": ep_cot_temp,
        "ep_max_tokens": ep_max_tokens,
        "ep_domain": ep_domain,
        "ep_n_cot": ep_n_cot,
        "ep_system_prompt": ep_system_prompt,
        "ep_cot_output": ep_cot_output,
        "ep_batch_size_cot": ep_batch_size_cot,
        "ep_cot_btn": ep_cot_btn,
        "ep_cot_stop": ep_cot_stop,
        "ep_student": ep_student,
        "ep_distill_backend": ep_distill_backend,
        "ep_distill_dataset": ep_distill_dataset,
        "ep_distill_output": ep_distill_output,
        "ep_epochs": ep_epochs,
        "ep_lora_r": ep_lora_r,
        "ep_max_samp": ep_max_samp,
        "ep_open_chk": ep_open_chk,
        "ep_offline_chk": ep_offline_chk,
        "ep_distill_btn": ep_distill_btn,
        "ep_distill_stop": ep_distill_stop,
        "ep_status": ep_status,
        "ep_loss_plot": ep_loss_plot,
        "ep_grad_plot": ep_grad_plot,
        "ep_log_box": ep_log_box,
        # expose helper so wiring can use it for the refresh button
        "_ep_dataset_choices": _ep_dataset_choices,
    }
