"""ORPO (Odds Ratio Preference Optimization) and SimPO alignment backends.

ORPO trains without a reference model — lower memory, faster than DPO.
SimPO uses a simple length-normalised reward without a reference model.
Both use TRL's ORPOTrainer.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_orpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    lambda_: float = 0.1,
    epochs: int = 1,
    lr: float = 8e-6,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 1024,
    lora_rank: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Run ORPO fine-tuning.

    ORPO integrates SFT and preference alignment into a single pass — no
    separate reference model needed, halving GPU memory vs DPO.

    Args:
        lambda_: Relative weight of the odds-ratio loss (0.1 typical).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    try:
        from trl import ORPOTrainer, ORPOConfig  # type: ignore[import]
    except ImportError:
        return {"output_path": "", "error": "trl>=0.15 required for ORPOTrainer"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.backends.mps.is_available():   device = "mps"
        elif torch.cuda.is_available():          device = "cuda"
        else:                                    device = "cpu"

    try:
        logger.info("Loading model for ORPO: %s", model_path)
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

        from distill.training.backends.dpo import _load_preference_dataset, _validate_preference_dataset
        dataset = _load_preference_dataset(dataset_path)
        _validate_preference_dataset(dataset)

        training_args = ORPOConfig(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lambda_=lambda_,
            max_length=max_length,
            logging_steps=10,
            save_steps=100,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = ORPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        logger.info("Starting ORPO (lambda=%.3f, epochs=%d)...", lambda_, epochs)
        result = trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(out)

        metrics = {"orpo_loss": result.training_loss, "train_steps": result.global_step}
        logger.info("ORPO complete: loss=%.4f", metrics["orpo_loss"])
        return {"output_path": str(out), "metrics": metrics, "error": ""}

    except Exception as exc:
        logger.error("ORPO failed: %s", exc, exc_info=True)
        return {"output_path": "", "metrics": {}, "error": str(exc)}


def run_simpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    gamma: float = 0.5,
    beta: float = 2.5,
    epochs: int = 1,
    lr: float = 8e-6,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 1024,
    lora_rank: int = 16,
    device: str = "auto",
) -> dict[str, Any]:
    """Run SimPO fine-tuning.

    SimPO uses length-normalised sequence likelihood without a ref model.
    gamma controls the target reward margin; beta is the inverse temperature.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    try:
        from trl import CPOTrainer, CPOConfig  # SimPO is available via CPO + SimPO option
    except ImportError:
        return {"output_path": "", "error": "trl>=0.10 required for SimPO (via CPOTrainer)"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if device == "auto":
        if torch.backends.mps.is_available():   device = "mps"
        elif torch.cuda.is_available():          device = "cuda"
        else:                                    device = "cpu"

    try:
        dtype = torch.bfloat16 if device != "cpu" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank, lora_alpha=lora_rank * 2,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

        from distill.training.backends.dpo import _load_preference_dataset, _validate_preference_dataset
        dataset = _load_preference_dataset(dataset_path)
        _validate_preference_dataset(dataset)

        training_args = CPOConfig(
            output_dir=str(out),
            loss_type="simpo",
            simpo_gamma=gamma,
            beta=beta,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            max_length=max_length,
            logging_steps=10,
            save_steps=100,
            report_to="none",
        )

        trainer = CPOTrainer(
            model=model, args=training_args,
            train_dataset=dataset, tokenizer=tokenizer,
        )

        logger.info("Starting SimPO (gamma=%.2f, beta=%.2f)...", gamma, beta)
        result = trainer.train()
        trainer.save_model(str(out))
        tokenizer.save_pretrained(out)

        metrics = {"simpo_loss": result.training_loss, "train_steps": result.global_step}
        return {"output_path": str(out), "metrics": metrics, "error": ""}

    except Exception as exc:
        logger.error("SimPO failed: %s", exc, exc_info=True)
        return {"output_path": "", "metrics": {}, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ORPO/SimPO alignment backend")
    p.add_argument("--method", choices=["orpo", "simpo"], default="orpo")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/orpo")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=8e-6)
    p.add_argument("--lora-rank", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fn = run_orpo if args.method == "orpo" else run_simpo
    result = fn(
        model_path=args.model, dataset_path=args.dataset,
        output_dir=args.output_dir, epochs=args.epochs,
        lr=args.lr, lora_rank=args.lora_rank,
    )
    if result["error"]:
        logger.error("%s failed: %s", args.method.upper(), result["error"])


if __name__ == "__main__":
    main()
