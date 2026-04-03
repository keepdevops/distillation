"""Configure & Launch tab widget layout."""
from __future__ import annotations

import gradio as gr
from pathlib import Path


def build_tab_configure(teachers, students, datasets, out_dirs, defaults):
    """Build the Configure & Launch tab.

    Parameters
    ----------
    teachers:  list of teacher model choices
    students:  list of student model choices
    datasets:  list of dataset choices
    out_dirs:  list of output directory choices
    defaults:  dict with keys default_teacher, default_student,
               default_dataset, default_out

    Returns
    -------
    dict of all widgets required for event wiring.
    """
    default_teacher = defaults["default_teacher"]
    default_student = defaults["default_student"]
    default_dataset = defaults["default_dataset"]
    default_out     = defaults["default_out"]

    with gr.Tab("Configure & Launch"):

        # Stage + Backend
        with gr.Row():
            stage = gr.Radio(
                ["SFT", "MiniLLM"], value="MiniLLM",
                label="Stage",
                info="SFT = teacher-label warmup.  MiniLLM = reverse-KL distillation.",
            )
            backend = gr.Radio(
                ["PyTorch", "MLX"], value="MLX",
                label="Backend",
                info="PyTorch/MPS = stable, full-featured.  MLX = Apple-native, 2-5× faster on M3.",
            )
            use_open = gr.Checkbox(
                value=True,
                label="Use open Qwen2 models (1.5B→0.5B, no HF login required)",
            )

        # Models
        gr.Markdown("### Models")
        with gr.Row():
            teacher = gr.Dropdown(
                choices=teachers,
                value=default_teacher,
                label="Teacher model",
                allow_custom_value=True,
                scale=3,
                info="Select from cache or type a HuggingFace model ID",
            )
            refresh_teacher_btn = gr.Button("Refresh", scale=1, size="sm")

        with gr.Row():
            student = gr.Dropdown(
                choices=students,
                value=default_student,
                label="Student model / checkpoint",
                allow_custom_value=True,
                scale=3,
                info="Select local checkpoint or HF model ID. For MiniLLM, point to your SFT checkpoint.",
            )
            refresh_student_btn = gr.Button("Refresh", scale=1, size="sm")

        # Dataset & Output
        gr.Markdown("### Dataset & Output")
        with gr.Row():
            dataset = gr.Dropdown(
                choices=datasets,
                value=default_dataset,
                label="Dataset",
                allow_custom_value=True,
                scale=3,
                info="Select from local cache or type a HuggingFace dataset ID",
            )
            refresh_dataset_btn = gr.Button("Refresh", scale=1, size="sm")

        with gr.Row():
            output_dir = gr.Dropdown(
                choices=out_dirs,
                value=default_out,
                label="Output directory",
                allow_custom_value=True,
                scale=3,
            )
            refresh_outdir_btn = gr.Button("Refresh", scale=1, size="sm")

        # Common training params
        gr.Markdown("### Training")
        with gr.Row():
            epochs      = gr.Slider(1, 10, value=2, step=1, label="Epochs")
            max_samples = gr.Slider(50, 10000, value=2000, step=50, label="Max samples")
        with gr.Row():
            batch_size  = gr.Slider(1, 32, value=8,  step=1, label="Batch size")
            grad_acc    = gr.Slider(1, 32, value=8,  step=1, label="Gradient accumulation")
            lora_r      = gr.Slider(4, 128, value=16, step=4, label="LoRA rank")

        # SFT-specific
        with gr.Group(visible=False) as sft_group:
            gr.Markdown("### SFT options")
            with gr.Row():
                sft_lr             = gr.Number(value=2e-4, label="Learning rate")
                max_new_tokens_sft = gr.Slider(64, 512, value=128, step=32,
                                               label="Teacher max new tokens")
                max_length         = gr.Slider(128, 1024, value=384, step=64,
                                               label="Max sequence length")

        # MiniLLM/PyTorch-specific
        with gr.Group(visible=False) as minillm_group:
            gr.Markdown("### MiniLLM options  (PyTorch / MPS)")
            with gr.Row():
                minillm_temp          = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                                  label="KD temperature")
                minillm_lr            = gr.Number(value=2e-5, label="Learning rate",
                                                  precision=6)
                num_generations       = gr.Slider(2, 16, value=4, step=1,
                                                  label="Generations per prompt",
                                                  info="4+ gives GRPO enough reward variance for non-zero advantage")
                max_completion_length = gr.Slider(64, 512, value=256, step=32,
                                                  label="Max completion length (tokens)",
                                                  info="256 gives completions room to terminate naturally; update _MAX_NATURAL_CHARS if changed")
                eval_steps            = gr.Slider(1, 50, value=20, step=1,
                                                  label="Eval every N steps")

        # MLX-specific
        with gr.Group(visible=True) as mlx_group:
            gr.Markdown("### MLX options  (Apple-native, 2-5× faster on M3)")
            gr.Markdown(
                "_MLX uses Apple-optimised defaults (batch=2, grad_acc=8, lora_r=16). "
                "Teacher logits are precomputed once then freed — both models never in memory together._"
            )
            with gr.Row():
                mlx_kd_temp   = gr.Slider(0.5, 2.0, value=1.0, step=0.1,
                                          label="KD temperature")
                mlx_lr        = gr.Number(value=2e-4, label="Learning rate", precision=6)
                mlx_eval_steps = gr.Slider(1, 100, value=50, step=1,
                                           label="Eval every N steps")
            with gr.Row():
                mlx_ce_alpha  = gr.Slider(0.0, 1.0, value=0.2, step=0.05,
                                          label="CE alpha (0=pure KD, 1=pure CE)",
                                          info="0.2 mixes CE for stability without losing KD signal")
                mlx_topk      = gr.Slider(10, 200, value=50, step=10,
                                          label="Top-K teacher logits",
                                          info="50 captures >99% of teacher probability mass (~300 MB vs ~300 GB full vocab)")
                mlx_q_bits    = gr.Radio([4, 8], value=4, label="Export quantization bits")
                mlx_resume    = gr.Checkbox(value=False, label="Resume from last checkpoint")

        with gr.Row():
            watchdog = gr.Checkbox(value=False, label="Enable watchdog (pause.flag callback)")

        with gr.Row():
            launch_btn = gr.Button("Launch", variant="primary", scale=3)
            stop_btn   = gr.Button("Stop",   variant="stop",    scale=1)

        run_status      = gr.Textbox(value="idle", label="Run status", interactive=False)
        launch_progress = gr.HTML(value="")

    return {
        "stage": stage,
        "backend": backend,
        "use_open": use_open,
        "teacher": teacher,
        "student": student,
        "dataset": dataset,
        "output_dir": output_dir,
        "epochs": epochs,
        "max_samples": max_samples,
        "batch_size": batch_size,
        "grad_acc": grad_acc,
        "lora_r": lora_r,
        "sft_lr": sft_lr,
        "max_new_tokens_sft": max_new_tokens_sft,
        "max_length": max_length,
        "minillm_temp": minillm_temp,
        "minillm_lr": minillm_lr,
        "num_generations": num_generations,
        "max_completion_length": max_completion_length,
        "eval_steps": eval_steps,
        "mlx_kd_temp": mlx_kd_temp,
        "mlx_lr": mlx_lr,
        "mlx_eval_steps": mlx_eval_steps,
        "mlx_ce_alpha": mlx_ce_alpha,
        "mlx_topk": mlx_topk,
        "mlx_q_bits": mlx_q_bits,
        "mlx_resume": mlx_resume,
        "watchdog": watchdog,
        "sft_group": sft_group,
        "minillm_group": minillm_group,
        "mlx_group": mlx_group,
        "launch_btn": launch_btn,
        "stop_btn": stop_btn,
        "run_status": run_status,
        "launch_progress": launch_progress,
        "refresh_teacher_btn": refresh_teacher_btn,
        "refresh_student_btn": refresh_student_btn,
        "refresh_dataset_btn": refresh_dataset_btn,
        "refresh_outdir_btn": refresh_outdir_btn,
    }
