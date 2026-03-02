#!/usr/bin/env python3
"""
MiniLLM-style knowledge distillation (reverse KL) for LLMs.
Bare-metal, air-gapped compatible. Optimized for Apple M3 (MPS).
"""

import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# Open models (no HuggingFace login / Meta license)
OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"
OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="MiniLLM distillation on M3")
    p.add_argument("--teacher", type=str, default="meta-llama/Llama-3.2-8B-Instruct")
    p.add_argument("--student", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--open", action="store_true",
                   help="Use open models (Qwen2 1.5B→0.5B) — no HF login or Meta license")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--output_dir", type=str, default="./distilled-minillm")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_acc", type=int, default=16)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--use_4bit_teacher", action="store_true")
    p.add_argument("--minillm_temp", type=float, default=1.0, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000, help="Max train samples (for quick runs)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true",
                   help="Air-gapped: use local cache only, no network (HF_HOME, HF_DATASETS_CACHE)")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag callback for watchdog")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_example(example):
    """Format as prompt (MiniLLM uses prompts for on-policy generation)."""
    prompt = example.get("instruction", example.get("prompt", ""))
    if "input" in example and example["input"]:
        prompt += "\n\nInput: " + example["input"]
    prompt += "\n\n### Response:"
    return {"prompt": prompt}


def main():
    args = parse_args()
    if args.open:
        args.teacher = OPEN_TEACHER
        args.student = OPEN_STUDENT
        print("Using open models (no login):", args.teacher)

    device = get_device()
    print(f"Device: {device}")

    # Air-gapped: no network
    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Tokenizer
    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    tokenizer = AutoTokenizer.from_pretrained(
        args.student, cache_dir=cache_dir, local_files_only=offline
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Teacher (optionally quantized)
    quant_config = None
    if args.use_4bit_teacher:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )
    teacher.eval()

    # Student
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, peft_config)
    student.print_trainable_parameters()

    # Dataset
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    if Path(args.dataset).exists():
        data = load_from_disk(args.dataset)
        dataset = data["train"] if isinstance(data, dict) and "train" in data else data
    else:
        dataset = load_dataset(args.dataset, split="train", cache_dir=ds_cache)
        if args.max_samples and args.max_samples < len(dataset):
            dataset = dataset.select(range(args.max_samples))

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.train_test_split(test_size=0.02, seed=42)

    # MiniLLM config (TRL)
    import logging
    log = logging.getLogger(__name__)
    try:
        from trl import MiniLLMTrainer, MiniLLMConfig
        log.debug("Using trl.MiniLLMTrainer")
    except ImportError as e:
        log.debug("trl main import failed: %s", e)
        try:
            from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig
            log.debug("Using trl.experimental.minillm")
        except ImportError as e2:
            log.error("TRL not found: %s; %s", e, e2)
            raise ImportError("Install trl: pip install trl") from e2

    # MiniLLMConfig = GRPOConfig = TrainingArguments (single args object)
    # num_generations must divide train batch; num_generations_eval must divide eval batch
    batch = args.batch_size
    config = MiniLLMConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=2e-5,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=500,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        num_generations=4,
        num_generations_eval=4,
        rkl_advantage=True,
        length_normalization=True,
        kd_temperature=args.minillm_temp,
        single_step_decomposition=False,
        gamma=0.0,
    )

    callbacks = []
    from watchdog_callbacks import MetricsCallback
    callbacks.append(MetricsCallback(args.output_dir))
    if args.watchdog:
        from watchdog_callbacks import PauseFlagCallback
        callbacks.append(PauseFlagCallback(args.output_dir))

    trainer = MiniLLMTrainer(
        model=student,
        teacher_model=teacher,
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()
    # Merge LoRA for full-model export (needed for GGUF/llama.cpp)
    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(args.output_dir)
    print(f"Distilled model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
