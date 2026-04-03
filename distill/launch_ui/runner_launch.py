"""High-level launch helpers for the Gradio launch UI."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..infra.paths import project_dir

PROJECT_DIR = project_dir()
PYTHON = sys.executable
_DOMAINS_FILE = PROJECT_DIR / "configs" / "domain_prompts.json"


def _build_cmd_launch(script: str, params: dict) -> list[str]:
    mod = "distill." + script.replace(".py", "")
    cmd = [PYTHON, "-m", mod]
    for flag, val in params.items():
        if val is None or val == "":
            continue
        if isinstance(val, bool):
            if val:
                cmd.append(f"--{flag}")
        else:
            cmd += [f"--{flag}", str(val)]
    return cmd


def build_launch_run_cmd(
    stage, backend, use_open,
    teacher, student,
    dataset, output_dir,
    epochs, batch_size, grad_acc, lora_r, max_samples,
    sft_lr, max_new_tokens_sft, max_length,
    minillm_temp, minillm_lr, num_generations, max_completion_length, eval_steps,
    mlx_kd_temp, mlx_lr, mlx_eval_steps, mlx_ce_alpha, mlx_topk, mlx_q_bits, mlx_resume,
    watchdog,
) -> list[str]:
    """Build the subprocess command for a training run."""
    params: dict = {
        "output_dir": output_dir,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "grad_acc": int(grad_acc),
        "lora_r": int(lora_r),
        "max_samples": int(max_samples),
        "watchdog": watchdog,
    }

    if use_open:
        params["open"] = True
    else:
        if teacher:
            params["teacher"] = teacher
        if student:
            params["student"] = student

    if dataset and dataset != "tatsu-lab/alpaca":
        params["dataset"] = dataset

    if stage == "SFT":
        script = "distill_sft.py"
        params["learning_rate"] = float(sft_lr)
        params["max_new_tokens"] = int(max_new_tokens_sft)
        params["max_length"] = int(max_length)
    elif backend == "MLX":
        script = "distill_mlx.py"
        params["kd_temp"] = float(mlx_kd_temp)
        params["learning_rate"] = float(mlx_lr)
        params["eval_steps"] = int(mlx_eval_steps)
        params["ce_alpha"] = float(mlx_ce_alpha)
        params["topk_logits"] = int(mlx_topk)
        params["q_bits"] = int(mlx_q_bits)
        if mlx_resume:
            params["resume"] = True
    else:
        script = "distill_minillm.py"
        params["minillm_temp"] = float(minillm_temp)
        params["learning_rate"] = float(minillm_lr)
        params["num_generations"] = int(num_generations)
        params["max_new_tokens"] = int(max_completion_length)
        params["eval_steps"] = int(eval_steps)

    return _build_cmd_launch(script, params)


def save_custom_domain(domain_id, label, description, system_prompts_text,
                       min_resp_words, max_resp_words, min_d2,
                       require_code, require_numbers) -> str:
    """Write a new domain entry to configs/domain_prompts.json."""
    domain_id = (domain_id or "").strip().lower().replace(" ", "_")
    if not domain_id:
        return "Error: Domain ID is required."

    prompts = [line.strip() for line in system_prompts_text.splitlines() if line.strip()]
    if not prompts:
        return "Error: At least one system prompt is required."

    try:
        registry = {}
        if _DOMAINS_FILE.exists():
            with open(_DOMAINS_FILE) as f:
                registry = json.load(f)

        registry[domain_id] = {
            "label":          (label or domain_id).strip(),
            "description":    (description or "").strip(),
            "system_prompts": prompts,
            "filter": {
                "min_resp_words":  int(min_resp_words),
                "max_resp_words":  int(max_resp_words),
                "min_d2":          round(float(min_d2), 3),
                "require_code":    bool(require_code),
                "require_numbers": bool(require_numbers),
            },
        }

        _DOMAINS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(_DOMAINS_FILE, "w") as f:
            json.dump(registry, f, indent=2)

        return f"Saved domain '{domain_id}' to configs/domain_prompts.json  ({len(prompts)} system prompts)."
    except Exception as e:
        return f"Error saving domain: {e}"


def build_magpie_cmd(teacher, output_dir, n, batch_size, use_filter, target, offline,
                     domain=None, backend="auto") -> list[str]:
    params: dict = {
        "output_dir": output_dir,
        "n": int(n),
        "batch_size": int(batch_size),
        "filter": use_filter,
        "backend": backend,
    }
    if domain:
        params["domain"] = domain
    if teacher and teacher != "Qwen/Qwen2-1.5B-Instruct":
        params["teacher"] = teacher
    if use_filter and target and int(target) > 0:
        params["target"] = int(target)
    if offline:
        params["offline"] = True
    return _build_cmd_launch("magpie_synth.py", params)


def build_filter_cmd(dataset, output_dir, target, min_response_words, min_distinct2, offline) -> list[str]:
    params: dict = {
        "dataset": dataset,
        "output_dir": output_dir,
        "target": int(target),
        "min_response_words": int(min_response_words),
        "min_distinct2": float(min_distinct2),
    }
    if offline:
        params["offline"] = True
    return _build_cmd_launch("filter_dataset.py", params)


def build_synth_cmd(teacher, use_open, output_dir, n_generate, batch_size, temperature,
                    seed_examples, offline) -> list[str]:
    params: dict = {
        "output_dir": output_dir,
        "n_generate": int(n_generate),
        "batch_size": int(batch_size),
        "temperature": float(temperature),
        "seed_examples": int(seed_examples),
    }
    if use_open:
        params["open"] = True
    elif teacher:
        params["teacher"] = teacher
    if offline:
        params["offline"] = True
    return _build_cmd_launch("generate_synthetic_data.py", params)


def build_eval_perplexity_cmd(output_dir, checkpoint, student, dataset, max_val_samples,
                              batch_size, compare_teacher, teacher, offline) -> list[str]:
    cmd = [PYTHON, "-m", "distill.eval.perplexity", output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    if dataset and dataset != "tatsu-lab/alpaca":
        cmd += ["--dataset", dataset]
    cmd += ["--max_val_samples", str(int(max_val_samples))]
    cmd += ["--batch_size", str(int(batch_size))]
    if compare_teacher:
        cmd.append("--compare_teacher")
        if teacher:
            cmd += ["--teacher", teacher]
    if offline:
        cmd.append("--offline")
    return cmd


def build_eval_quality_cmd(output_dir, checkpoint, student, dataset, n_samples,
                           judge, judge_teacher, offline) -> list[str]:
    cmd = [PYTHON, "-m", "distill.eval.quality", output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    if dataset and dataset != "tatsu-lab/alpaca":
        cmd += ["--dataset", dataset]
    cmd += ["--n_samples", str(int(n_samples))]
    if judge:
        cmd.append("--judge")
        if judge_teacher:
            cmd += ["--teacher", judge_teacher]
    if offline:
        cmd.append("--offline")
    return cmd


def build_eval_benchmark_cmd(output_dir, checkpoint, student, n_sequences, batch_size,
                             baseline_dir, threshold, offline) -> list[str]:
    cmd = [PYTHON, "-m", "distill.eval.benchmarks", output_dir]
    if checkpoint:
        cmd += ["--checkpoint", checkpoint]
    if student and student != "Qwen/Qwen2-0.5B-Instruct":
        cmd += ["--student", student]
    cmd += ["--n_sequences", str(int(n_sequences))]
    cmd += ["--batch_size", str(int(batch_size))]
    if baseline_dir:
        cmd += ["--baseline_dir", baseline_dir]
    cmd += ["--threshold", str(float(threshold))]
    if offline:
        cmd.append("--offline")
    return cmd
