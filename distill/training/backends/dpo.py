"""DPO (Direct Preference Optimization) alignment backend.

Trains the student on preference pairs (chosen / rejected) after SFT.
Uses TRL's DPOTrainer with LoRA for memory efficiency.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_preference_dataset(dataset_path: str) -> Any:
    """Load preference data from a JSONL file or HF dataset ID."""
    from datasets import load_dataset, Dataset
    p = Path(dataset_path)
    if p.exists() and p.suffix == ".jsonl":
        import json
        rows = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return Dataset.from_list(rows)
    # Flywheel dataset
    if dataset_path == "__flywheel__":
        from distill.data.flywheel import get_flywheel_dataset
        return get_flywheel_dataset()
    # HF Hub dataset ID
    ds = load_dataset(dataset_path, split="train")
    return ds


def _validate_preference_dataset(ds: Any) -> None:
    """Raise ValueError if required columns are missing."""
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(ds.column_names)
    if missing:
        raise ValueError(
            f"Preference dataset missing columns: {missing}. "
            f"Got: {ds.column_names}"
        )


def run_dpo(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    beta: float = 0.1,
    epochs: int = 1,
    lr: float = 5e-5,
    batch_size: int = 2,
    grad_accum: int = 4,
    max_length: int = 1024,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    device: str = "auto",
) -> dict[str, Any]:
    """Run DPO fine-tuning on preference data.

    Args:
        model_path:   Path to the SFT-trained student model.
        dataset_path: Path to JSONL preference data or HF dataset ID.
        output_dir:   Where to save the aligned checkpoint.
        beta:         KL regularisation coefficient (0.1 typical).
        epochs:       Training epochs.
        lr:           Learning rate.
        batch_size:   Per-device train batch size.
        grad_accum:   Gradient accumulation steps.
        max_length:   Maximum sequence length.
        lora_rank:    LoRA rank for the DPO adapter.
        lora_alpha:   LoRA alpha scaling.
        device:       "auto", "mps", "cuda", or "cpu".

    Returns:
        dict with output_path, metrics, error.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
    from peft import LoraConfig, get_peft_model, TaskType
    try:
        from trl import DPOTrainer, DPOConfig  # type: ignore[import]
    except ImportError:
        return {"output_path": "", "error": "trl not installed. Run: pip install trl>=0.10"}

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Resolve device
    if device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    try:
        logger.info("Loading model for DPO: %s", model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj"],
        )
        model = get_peft_model(model, lora_cfg)

        dataset = _load_preference_dataset(dataset_path)
        _validate_preference_dataset(dataset)

        training_args = DPOConfig(
            output_dir=str(out),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            beta=beta,
            max_length=max_length,
            logging_steps=10,
            save_steps=100,
            report_to="none",
            remove_unused_columns=False,
        )

        trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer,
        )

        logger.info("Starting DPO training (beta=%.2f, epochs=%d)...", beta, epochs)
        train_result = trainer.train()

        trainer.save_model(str(out))
        tokenizer.save_pretrained(out)

        metrics = {
            "dpo_loss":    train_result.training_loss,
            "train_steps": train_result.global_step,
        }
        logger.info("DPO complete: %s | loss=%.4f", out, metrics["dpo_loss"])
        return {"output_path": str(out), "metrics": metrics, "error": ""}

    except Exception as exc:
        logger.error("DPO training failed: %s", exc, exc_info=True)
        return {"output_path": "", "metrics": {}, "error": str(exc)}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPO alignment backend")
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output-dir", default="outputs/dpo")
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lora-rank", type=int, default=16)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dpo(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        beta=args.beta,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        lora_rank=args.lora_rank,
    )
    if result["error"]:
        logger.error("DPO failed: %s", result["error"])
    else:
        logger.info("DPO output: %s", result["output_path"])


if __name__ == "__main__":
    main()
