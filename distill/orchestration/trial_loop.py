"""
Multi-trial distillation loop: runs distill+eval N times, returns the winner.
"""
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Callable

from .subprocess_runner import run_cmd
from .cmd_builder import _build_distill_cmd

LOG = logging.getLogger(__name__)


def run_multi_trial(
    args,
    output_dir: Path,
    project_root: Path,
    sft_checkpoint: str | None,
    dataset_override: str | None,
    exp_log,
    collect_fn: Callable[[str], dict],
) -> tuple[Path, list[str]]:
    """Run N independent distillation trials; return (winning_dir, steps_completed).

    Args:
        args: Parsed argument namespace (n_trials, seed, temperature, lora_r, epochs, …).
        output_dir: Base output directory; each trial writes to output_dir/trial_NN/.
        project_root: Repository root passed to run_cmd as cwd.
        sft_checkpoint: Optional SFT warmup checkpoint path.
        dataset_override: Optional filtered/synthetic dataset path.
        exp_log: ExperimentLog instance used to propose configs and diagnose results.
        collect_fn: Callable(str) → dict that reads metrics from a trial directory.

    Returns:
        A tuple of (best_trial_dir, steps_completed) where steps_completed contains
        "distill" when at least one trial finished.
    """
    steps_completed: list[str] = []
    LOG.info("Multi-trial mode: %d trials", args.n_trials)

    base_config = {
        "temperature": args.temperature,
        "lora_r": args.lora_r,
        "epochs": args.epochs,
    }
    trial_results: list[tuple[float, int, Path, dict]] = []

    for t in range(args.n_trials):
        # Propose hyperparameter config for this trial
        if t == 0:
            tc = dict(base_config)
        else:
            tc = exp_log.propose_next(base_config)

        t_temperature = tc.get("temperature", args.temperature)
        t_lora_r = int(tc.get("lora_r", args.lora_r))
        t_epochs = int(tc.get("epochs", args.epochs))
        trial_dir = output_dir / f"trial_{t:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        trial_seed = args.seed + t

        LOG.info(
            "Trial %d/%d — dir=%s  temp=%.2f  lora_r=%d  epochs=%d  seed=%d",
            t + 1, args.n_trials, trial_dir,
            t_temperature, t_lora_r, t_epochs, trial_seed,
        )

        trial_args = copy.copy(args)
        trial_args.temperature = t_temperature
        trial_args.lora_r = t_lora_r
        trial_args.epochs = t_epochs

        distill_cmd = _build_distill_cmd(
            trial_args, trial_dir, sft_checkpoint, dataset_override, trial_seed
        )
        try:
            run_cmd(distill_cmd, project_root)
        except RuntimeError as exc:
            LOG.error("Trial %d distillation failed: %s — skipping", t, exc)
            trial_results.append((float("inf"), t, trial_dir, tc))
            continue

        # Per-trial perplexity eval (no export, no quality eval yet)
        eval_cmd = [
            __import__("sys").executable,
            "-m", "distill.eval.perplexity",
            str(trial_dir),
        ]
        if args.open:
            eval_cmd += ["--student", "Qwen/Qwen2-0.5B-Instruct"]
        if args.offline:
            eval_cmd += ["--offline"]
        try:
            run_cmd(eval_cmd, project_root)
        except RuntimeError as exc:
            LOG.error("Trial %d eval failed: %s", t, exc)

        t_metrics = collect_fn(str(trial_dir))
        t_ppl = t_metrics.get("eval_perplexity", float("inf"))
        trial_results.append((t_ppl, t, trial_dir, tc))
        LOG.info("Trial %d result: eval_perplexity=%.3f", t, t_ppl)

    # Select winner by lowest perplexity
    trial_results.sort(key=lambda x: x[0])
    best_ppl, best_idx, best_dir, best_config = trial_results[0]
    LOG.info(
        "Winner: trial_%02d  eval_perplexity=%.3f  config=%s",
        best_idx, best_ppl, best_config,
    )

    # Diagnose winner
    winner_metrics = collect_fn(str(best_dir))
    diagnoses = exp_log.diagnose(winner_metrics)
    for diag in diagnoses:
        LOG.info("Diagnosis: %s", diag)

    steps_completed.append("distill")
    return best_dir, steps_completed
