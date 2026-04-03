#!/usr/bin/env python3
"""
Autonomous distillation agent: runs the full pipeline (distill → export) end-to-end.

Use for headless runs, cron, or LaunchAgent. Supports --watchdog for plateau detection
and --offline for air-gapped.

Usage:
  python -m distill.run_distillation_agent --open
  python -m distill.run_distillation_agent --open --watchdog --export gguf
  python -m distill.run_distillation_agent --backend mlx --export all --open
  python -m distill.run_distillation_agent --backend unsloth --export coreml --open
  python -m distill.run_distillation_agent --config configs/agent_config.json
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

from .subprocess_runner import run_cmd
from .export_utils import export_gguf, export_coreml, export_mlx_quant
from .arg_parser import build_arg_parser, setup_logging
from .cmd_builder import _build_distill_cmd
from .pipeline_steps import (
    run_filter_step,
    run_synthetic_step,
    run_sft_warmup,
    run_eval_step,
    run_quality_step,
    run_benchmark_step,
)
from .trial_loop import run_multi_trial

LOG = logging.getLogger(__name__)


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_logging(args.verbose)
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root)

    # ── Config overrides ──────────────────────────────────────────────────────
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
                LOG.info("Config override: %s=%s", k, v)

    # Legacy --export-gguf → --export gguf
    if args.export_gguf and args.export != "gguf":
        args.export = "gguf"

    # Auto-name output_dir from config hash + timestamp
    if args.auto_name:
        import hashlib
        from datetime import datetime as _dt
        cfg_str = (f"{args.backend}-e{args.epochs}-r{args.lora_r}"
                   f"-t{args.temperature}-s{args.max_samples}")
        h = hashlib.md5(cfg_str.encode()).hexdigest()[:6]
        ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        args.output_dir = f"./runs/{h}-{ts}"
        LOG.info("Auto-named output_dir: %s", args.output_dir)

    _start_time = time.time()
    _steps_completed: list[str] = []

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Backend: %s  Export: %s", args.backend, args.export)

    # ── Step 0: Experiment history ────────────────────────────────────────────
    from ..ui.experiment_log import ExperimentLog
    from ..ui.experiment_collect import collect_metrics as _collect_metrics_fn

    _exp_log = ExperimentLog(args.experiment_log)
    if args.log_experiment or args.n_trials > 1:
        _summary = _exp_log.summarize(10)
        if "(no experiment history" not in _summary:
            LOG.info("Recent experiment history:\n%s", _summary)
        else:
            LOG.info("No previous experiment history found.")

    # ── Step 0a: Memory check ─────────────────────────────────────────────────
    if args.memory_check:
        _run_memory_check(args)

    # ── Step 0b: Dataset filtering ────────────────────────────────────────────
    _dataset_override: str | None = None
    if args.filter:
        _dataset_override = run_filter_step(args, output_dir, project_root)
        _steps_completed.append("filter")

    # ── Step 0c: Synthetic data generation ───────────────────────────────────
    if args.synthetic_data:
        _dataset_override = run_synthetic_step(args, output_dir, project_root)
        _steps_completed.append("synthetic_data")

    # ── Step 0d: SFT curriculum warmup (PyTorch backend only) ────────────────
    _sft_checkpoint: str | None = None
    if args.curriculum:
        if args.backend != "pytorch":
            LOG.warning(
                "--curriculum is only supported with --backend pytorch. "
                "Skipping SFT warmup for backend '%s'.", args.backend,
            )
        else:
            _sft_checkpoint = run_sft_warmup(
                args, output_dir, project_root, _dataset_override
            )
            _steps_completed.append("sft_warmup")

    # ── 1. Distillation ───────────────────────────────────────────────────────
    if args.n_trials > 1:
        output_dir, trial_steps = run_multi_trial(
            args, output_dir, project_root,
            _sft_checkpoint, _dataset_override,
            _exp_log, _collect_metrics_fn,
        )
        _steps_completed.extend(trial_steps)
    else:
        distill_cmd = _build_distill_cmd(
            args, output_dir, _sft_checkpoint, _dataset_override, args.seed
        )
        run_cmd(distill_cmd, project_root, json_log=output_dir / "train_log.jsonl")
        _steps_completed.append("distill")

    # ── 2. Export ─────────────────────────────────────────────────────────────
    export_targets: set[str] = set()
    if args.export == "all":
        export_targets = {"gguf", "coreml", "mlx"}
    elif args.export != "none":
        export_targets = {args.export}

    if "gguf" in export_targets:
        LOG.info("Exporting GGUF...")
        export_gguf(output_dir, project_root, args.outtype)

    if "coreml" in export_targets:
        LOG.info("Exporting CoreML...")
        export_coreml(output_dir, project_root, args.coreml_quantize)

    if "mlx" in export_targets and args.backend != "mlx":
        LOG.info("Exporting MLX quantization...")
        export_mlx_quant(output_dir, project_root, args.q_bits)

    # ── 3. Perplexity eval ────────────────────────────────────────────────────
    if not args.skip_eval:
        run_eval_step(args, output_dir, project_root)

    # ── 4. Quality eval ───────────────────────────────────────────────────────
    if not args.skip_eval and not args.skip_judge:
        run_quality_step(args, output_dir, project_root)
        _steps_completed.append("quality_eval")

    # ── 5. WikiText-2 benchmark ───────────────────────────────────────────────
    if args.benchmarks:
        run_benchmark_step(args, output_dir, project_root)
        _steps_completed.append("benchmarks")

    # ── 5b. Diagnose (single-trial only) ─────────────────────────────────────
    if args.n_trials <= 1 and not args.skip_eval:
        _final_metrics = _collect_metrics_fn(str(output_dir))
        for _d in _exp_log.diagnose(_final_metrics):
            LOG.info("Diagnosis: %s", _d)

    # ── 6. Log experiment ─────────────────────────────────────────────────────
    if args.log_experiment:
        _log_experiment(args, output_dir, _dataset_override, _steps_completed,
                        _start_time, _exp_log, _collect_metrics_fn)

    LOG.info("Agent finished. Output: %s  Steps: %s", output_dir, _steps_completed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_memory_check(args) -> None:
    """Estimate memory usage and emit warnings if near the 36 GB limit."""
    import json as _json

    def _estimate_gb(model_id: str, cache_dir) -> float:
        """Rough estimate: 2 bytes/param (bf16) × 4× for training overhead."""
        cfg_path = None
        for search in [
            Path(cache_dir or "") / "models--" / model_id.replace("/", "--"),
            Path.home() / ".cache" / "huggingface" / "hub" /
            ("models--" + model_id.replace("/", "--")),
        ]:
            for cfg in search.rglob("config.json"):
                cfg_path = cfg
                break
            if cfg_path:
                break
        if not cfg_path:
            return 0.0
        with open(cfg_path) as f:
            cfg = _json.load(f)
        params = cfg.get("num_parameters") or (
            cfg.get("hidden_size", 0) * cfg.get("num_hidden_layers", 0) * 12
        )
        return params * 2 * 4 / 1e9  # bf16 × 4× training overhead

    cache_dir = os.environ.get("HF_HOME") or args.__dict__.get("cache_dir")
    teacher_id = ("Qwen/Qwen2-1.5B-Instruct" if args.open
                  else "meta-llama/Llama-3.2-8B-Instruct")
    student_id = ("Qwen/Qwen2-0.5B-Instruct" if args.open
                  else "meta-llama/Llama-3.2-1B-Instruct")
    est = _estimate_gb(teacher_id, cache_dir) + _estimate_gb(student_id, cache_dir)
    if est > 0:
        if args.backend == "mlx":
            LOG.info(
                "Estimated memory (MLX): ~%.1f GB student+optimizer "
                "(teacher freed after precompute)", _estimate_gb(student_id, cache_dir)
            )
            vocab_gb = args.batch_size * 512 * 152000 * 4 / 1e9
            LOG.info(
                "Peak logit tensor per micro-batch: ~%.2f GB "
                "(batch_size=%d × seq_len=512 × vocab=152k × float32)",
                vocab_gb, args.batch_size,
            )
            if vocab_gb > 2.0:
                LOG.warning(
                    "Logit tensor %.2f GB > 2 GB — likely OOM cause. "
                    "Reduce --batch_size (currently %d) to 1 or 2.",
                    vocab_gb, args.batch_size,
                )
        else:
            LOG.info("Estimated memory: ~%.1f GB (teacher + student training)", est)
            if est > 30:
                LOG.warning(
                    "Memory estimate (%.1f GB) may approach 36GB M3 Max limit. "
                    "Consider --batch_size 2 or --grad_acc 32 if OOM.", est,
                )


def _log_experiment(
    args,
    output_dir: Path,
    dataset_override: str | None,
    steps_completed: list[str],
    start_time: float,
    exp_log,
    collect_fn,
) -> None:
    """Build and append an experiment log entry."""
    try:
        from ..ui.experiment_collect import build_entry, make_run_id
        import torch as _torch

        hardware = (
            "mps" if _torch.backends.mps.is_available() else
            "cuda" if _torch.cuda.is_available() else "cpu"
        )
        teacher_id = ("Qwen/Qwen2-1.5B-Instruct" if args.open
                      else "meta-llama/Llama-3.2-8B-Instruct")
        student_id = ("Qwen/Qwen2-0.5B-Instruct" if args.open
                      else "meta-llama/Llama-3.2-1B-Instruct")
        run_id = make_run_id(str(output_dir))
        config = {
            "backend": args.backend,
            "teacher": teacher_id,
            "student": student_id,
            "dataset": str(dataset_override or args.dataset),
            "epochs": args.epochs,
            "max_samples": args.max_samples,
            "temperature": args.temperature,
            "lora_r": args.lora_r,
            "curriculum": args.curriculum,
            "sft_epochs": args.sft_epochs if args.curriculum else 0,
            "synthetic_data": args.synthetic_data,
            "n_synthetic": args.n_synthetic if args.synthetic_data else 0,
            "export": args.export,
            "seed": args.seed,
            "n_trials": args.n_trials,
        }
        metrics = collect_fn(str(output_dir))
        entry = build_entry(
            run_id=run_id,
            output_dir=str(output_dir),
            config=config,
            metrics=metrics,
            outcome="success",
            steps_completed=steps_completed,
            start_time=start_time,
            hardware=hardware,
        )
        exp_log.append(entry)
        LOG.info("Experiment logged to %s (run_id: %s)", args.experiment_log, run_id)
    except Exception as exc:
        LOG.error("Failed to log experiment entry: %s", exc)


if __name__ == "__main__":
    main()
