#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) warmup stage for curriculum distillation.

The teacher generates responses for all training prompts, then the student
is trained with standard cross-entropy loss (no KL divergence).

This gives the student a good distribution-matching starting point before
the full reverse-KL distillation stage in distill_minillm.py.

Output: {output_dir}/sft_checkpoint/   (merged LoRA weights, HF format)
        {output_dir}/sft_labels.jsonl  (teacher-generated labels cache)

Usage:
    python -m distill.distill_sft --open --output_dir ./distilled-minillm
    python -m distill.distill_sft --open --epochs 1 --max_samples 2000 \
        --output_dir ./distilled-minillm --watchdog
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Add scripts directory to path for local imports

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model

from ...data.pipeline import load_dataset_split, format_prompt_only, validate_dataset_schema, DATASET_HELP
from ...infra.train_utils import get_device
from .sft_dataset import SFTDataset, generate_labels  # noqa: F401 — re-exported for callers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"
OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="SFT warmup for curriculum distillation")
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-8B-Instruct")
    p.add_argument("--student", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--open", action="store_true",
                   help="Use open Qwen2 models (no HF login)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-minillm")
    p.add_argument("--epochs", type=int, default=1,
                   help="SFT warmup epochs (default: 1 — just one pass)")
    p.add_argument("--batch_size", type=int, default=8, help="Physical batch size (default: 8, optimized for M3 Max)")
    p.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch = 64)")
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--max_new_tokens", type=int, default=128,
                   help="Max tokens teacher generates per prompt")
    p.add_argument("--max_length", type=int, default=384,
                   help="Max token length for training sequences")
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--watchdog", action="store_true",
                   help="Enable pause.flag callback for watchdog")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return p.parse_args()





def main():
    args = parse_args()
    if args.open:
        args.teacher = OPEN_TEACHER
        args.student = OPEN_STUDENT
        logger.info("Using open models: teacher=%s  student=%s", args.teacher, args.student)

    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    sft_dir = output_dir / "sft_checkpoint"
    labels_path = output_dir / "sft_labels.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    device = get_device()
    logger.info("Device: %s", device)

    # Load shared tokenizer (teacher and student share tokenizer family for Qwen2)
    tokenizer = AutoTokenizer.from_pretrained(
        args.student, cache_dir=cache_dir, local_files_only=offline,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only: left-pad for correct batch generation

    # Load training prompts
    logger.info("Loading dataset: %s", args.dataset)
    train_ds = load_dataset_split(args.dataset, args.max_samples, ds_cache, offline)
    validate_dataset_schema(train_ds, args.dataset, logger=logger)
    prompts = [format_prompt_only(row) for row in train_ds]
    logger.info("Training prompts: %d", len(prompts))

    # Generate teacher labels (or load from cache)
    if labels_path.exists():
        logger.info("Loading cached teacher labels from %s", labels_path)
        labeled = []
        with open(labels_path) as f:
            for line in f:
                labeled.append(json.loads(line))
        if len(labeled) >= len(prompts):
            responses = [r["response"] for r in labeled[: len(prompts)]]
        else:
            logger.info("Cache incomplete (%d < %d), regenerating", len(labeled), len(prompts))
            responses = None
    else:
        responses = None

    if responses is None:
        logger.info("Loading teacher: %s", args.teacher)
        teacher = AutoModelForCausalLM.from_pretrained(
            args.teacher, torch_dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
        teacher.to(device)
        teacher.eval()

        responses = generate_labels(teacher, tokenizer, prompts, args.max_new_tokens, device, batch_size=args.batch_size)

        # Cache labels
        with open(labels_path, "w") as f:
            for p, r in zip(prompts, responses):
                f.write(json.dumps({"prompt": p, "response": r}) + "\n")
        logger.info("Teacher labels cached to %s", labels_path)

        # Free teacher memory before loading student
        del teacher
        if device.type == "mps":
            torch.mps.empty_cache()

    # Load student with LoRA
    logger.info("Loading student: %s", args.student)
    student = AutoModelForCausalLM.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, device_map="auto",
        cache_dir=cache_dir, local_files_only=offline,
    )
    peft_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,  # scale=2.0 regardless of r (was hardcoded 128)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, peft_cfg)
    student.print_trainable_parameters()

    # Dataset — prompts and responses passed separately so SFTDataset can mask
    # prompt tokens in labels (-100), ensuring only response tokens contribute to loss.
    sft_dataset = SFTDataset(prompts, responses, tokenizer, args.max_length)

    # Warmup steps (warmup_ratio deprecated in transformers 5.2)
    _train_size = len(sft_dataset)
    _steps_per_epoch = max(1, _train_size // (args.batch_size * args.grad_acc))
    _total_steps = _steps_per_epoch * args.epochs
    _warmup_steps = max(1, round(0.03 * _total_steps))

    # Training
    training_args = TrainingArguments(
        output_dir=str(sft_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=False,  # 36GB M3 Max has plenty of RAM; checkpointing adds ~25% overhead
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=500,
        save_total_limit=1,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        report_to="none",
        dataloader_pin_memory=False,  # MPS does not support pin_memory
        dataloader_num_workers=0,     # MPS: forking workers causes instability; data is pre-tokenized so overhead is minimal
    )

    callbacks = []
    if args.watchdog:
        from distill.infra.watchdog_callbacks import PauseFlagCallback
        callbacks.append(PauseFlagCallback(str(sft_dir)))

    trainer = Trainer(
        model=student,
        args=training_args,
        train_dataset=sft_dataset,
        # No data_collator: SFTDataset pre-tokenizes with explicit labels (-100 masking).
        # default_data_collator (used when None) just stacks the pre-built tensors.
        callbacks=callbacks,
    )

    logger.info("Starting SFT warmup (%d epoch(s))...", args.epochs)
    trainer.train()

    # Merge LoRA and save
    logger.info("Merging LoRA weights...")
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(str(sft_dir))
    tokenizer.save_pretrained(str(sft_dir))
    logger.info("SFT checkpoint saved to %s", sft_dir)


if __name__ == "__main__":
    subprocess.Popen(['caffeinate', '-i', 'sleep', '3600'])
    main()
