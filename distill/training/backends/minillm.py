#!/usr/bin/env python3
"""
MiniLLM-style knowledge distillation (reverse KL) for LLMs.
Bare-metal, air-gapped compatible. Optimized for Apple M3 (MPS).
"""

import argparse
import logging
import os
import subprocess
import sys
import warnings
from pathlib import Path

# Add scripts directory to path for local imports

import torch
from peft import LoraConfig, get_peft_model

from ...data.pipeline import load_dataset_split, format_prompt_only, validate_dataset_schema, DATASET_HELP
from ...infra.config import cfg
from ...infra.train_utils import get_device
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# ── Silence known deprecation warnings from TRL / transformers 5.x ──────────
# TRL 0.29 passes disable_compile as a kwarg alongside generation_config;
# transformers 5.2 wants it inside the GenerationConfig — TRL's bug to fix.
warnings.filterwarnings(
    "ignore",
    message="Passing `generation_config` together with generation-related arguments",
)
# The same message is also emitted via logging (not warnings), so filter that too.
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
# Silence TRLExperimentalWarning for the minillm import path
os.environ.setdefault("TRL_EXPERIMENTAL_SILENCE", "1")


LOG = logging.getLogger(__name__)

from .minillm_utils import (  # noqa: F401 — re-exported for callers
    response_quality_reward, detect_attn_impl, _MAX_NATURAL_CHARS,
)


def parse_args():
    p = argparse.ArgumentParser(description="MiniLLM distillation on M3")
    p.add_argument("--teacher", type=str, default=cfg.models.default_teacher)
    p.add_argument("--student", type=str, default=cfg.models.default_student)
    p.add_argument("--open", action="store_true",
                   help="Use open models (Qwen2 1.5B→0.5B) — no HF login or Meta license")
    p.add_argument("--dataset", type=str, default=cfg.models.default_dataset, help=DATASET_HELP)
    p.add_argument("--output_dir", type=str, default="./distilled-minillm")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=cfg.training.batch_size, help="Physical batch size (default: 8, optimized for M3 Max)")
    p.add_argument("--grad_acc", type=int, default=8, help="Gradient accumulation steps (default: 8, effective batch = 64)")
    p.add_argument("--lora_r", type=int, default=cfg.training.lora_r)
    p.add_argument("--use_4bit_teacher", action="store_true")
    p.add_argument("--minillm_temp", type=float, default=cfg.training.kd_temperature, help="KD temperature")
    p.add_argument("--max_samples", type=int, default=2000, help="Max train samples (for quick runs)")
    p.add_argument("--eval_steps", type=int, default=20, help="Run eval every N steps (default: 20; lower gives more detail, higher is faster)")
    p.add_argument("--num_generations", type=int, default=4, help="Generations per prompt for GRPO/MiniLLM (default: 4; more gives better advantage variance at cost of speed)")
    p.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens to generate (default: 256; update _MAX_NATURAL_CHARS if changed significantly)")
    p.add_argument("--learning_rate", type=float, default=cfg.training.learning_rate, help="Learning rate (default: 2e-4)")
    p.add_argument("--eval_split", type=float, default=0.02, help="Eval dataset size as fraction of train (default: 0.02 = 2%%)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true",
                   help="Air-gapped: use local cache only, no network (HF_HOME, HF_DATASETS_CACHE)")
    p.add_argument("--watchdog", action="store_true", help="Enable pause.flag callback for watchdog")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    return p.parse_args()




def main():
    args = parse_args()
    if args.open:
        args.teacher = cfg.models.open_teacher
        args.student = cfg.models.open_student
        print("Using open models (no login):", args.teacher)
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)

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
    tokenizer.padding_side = "left"

    # Teacher (optionally quantized)
    quant_config = None
    if args.use_4bit_teacher:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    _attn_impl = detect_attn_impl()

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        quantization_config=quant_config,
        dtype=torch.bfloat16,
        attn_implementation=_attn_impl,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )
    teacher.eval()

    # Clear cache after teacher load (prevent memory fragmentation)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Student
    student = AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=torch.bfloat16,
        attn_implementation=_attn_impl,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,  # scale=2.0 regardless of r (was hardcoded 128, broke at r<64)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,  # no dropout — faster and no benefit for distillation
        bias="none",
        task_type="CAUSAL_LM",
    )
    student = get_peft_model(student, peft_config)
    student.print_trainable_parameters()
    # Transformers 5.x: passing generation kwargs alongside a saved generation_config
    # is deprecated. Reset to a clean config so TRL can set its own kwargs freely.
    from transformers import GenerationConfig
    student.generation_config = GenerationConfig()

    # torch.compile() optimization (20-40% speedup, PyTorch 2.0+)
    # NOTE: Disabled on MPS due to compilation issues with some operations
    if hasattr(torch, "compile") and torch.__version__ >= "2.0" and device.type != "mps":
        print("✓ Compiling student model with torch.compile() (20-40% speedup)")
        print("  First run has ~1-2 min compilation overhead, subsequent runs benefit fully")
        try:
            # Use reduce-overhead mode for best training performance
            student = torch.compile(student, mode="reduce-overhead")
            # Also compile teacher for generation speedup
            teacher = torch.compile(teacher, mode="reduce-overhead")
        except Exception as e:
            print(f"⚠ torch.compile() failed: {e}")
            print("  Continuing without compilation (still works, just slower)")
    elif device.type == "mps":
        print("torch.compile() skipped on MPS (Apple Silicon) due to compatibility issues")
        print("  Still get 2-3x speedup from other optimizations (Flash Attention, etc.)")
    else:
        print("torch.compile() not available (requires PyTorch 2.0+)")

    # Dataset
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    dataset = load_dataset_split(args.dataset, args.max_samples, ds_cache, offline)
    validate_dataset_schema(dataset, args.dataset, logger=LOG)
    dataset = dataset.map(
        lambda ex: {"prompt": format_prompt_only(ex)},
        remove_columns=dataset.column_names,
    )
    dataset = dataset.train_test_split(test_size=args.eval_split, seed=42)

    # MiniLLM config (TRL)
    try:
        from trl import MiniLLMTrainer, MiniLLMConfig
        LOG.debug("Using trl.MiniLLMTrainer")
    except ImportError as e:
        LOG.debug("trl main import failed: %s", e)
        try:
            from trl.experimental.minillm import MiniLLMTrainer, MiniLLMConfig
            LOG.debug("Using trl.experimental.minillm")
        except ImportError as e2:
            LOG.error("TRL not found: %s; %s", e, e2)
            raise ImportError("Install trl: pip install trl") from e2

    # MiniLLMConfig = GRPOConfig = TrainingArguments (single args object)
    # num_generations must divide train batch; num_generations_eval must divide eval batch
    batch = args.batch_size
    num_generations = args.num_generations
    # Approximate total steps so we can set a sensible warmup_steps
    # (warmup_ratio is deprecated in transformers 5.2, use warmup_steps instead)
    _train_size = max(1, int(len(dataset["train"])))
    _steps_per_epoch = max(1, _train_size // (batch * args.grad_acc))
    _total_steps = _steps_per_epoch * args.epochs * num_generations
    _warmup_steps = max(1, round(0.03 * _total_steps))

    config = MiniLLMConfig(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        num_train_epochs=args.epochs,
        max_steps=_total_steps,  # Required for on-policy dataloader
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=batch,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.learning_rate,
        bf16=True,
        gradient_checkpointing=False,  # 36GB M3 Max has plenty of RAM; checkpointing adds ~25% overhead
        optim="adamw_torch_fused",
        logging_steps=10,
        save_steps=500,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=_warmup_steps,
        dataloader_pin_memory=False,  # MPS does not support pin_memory
        dataloader_num_workers=0,     # on-policy data is generated in-process; workers add fork overhead
        num_generations=num_generations,
        num_generations_eval=num_generations,
        max_completion_length=args.max_new_tokens,
        rkl_advantage=True,
        length_normalization=True,
        kd_temperature=args.minillm_temp,
        single_step_decomposition=False,
        gamma=0.0,
    )

    callbacks = []
    from distill.infra.watchdog_callbacks import MetricsCallback
    callbacks.append(MetricsCallback(args.output_dir))
    if args.watchdog:
        from distill.infra.watchdog_callbacks import PauseFlagCallback
        callbacks.append(PauseFlagCallback(args.output_dir))

    # Early stopping for diverging trials (saves time on bad configs)
    from distill.infra.early_stopping_callback import EarlyStoppingCallback
    callbacks.append(EarlyStoppingCallback(
        check_step=20,              # Check after 20 steps (~2-3 min)
        divergence_threshold=1.5,   # Stop if loss > baseline × 1.5
    ))

    trainer = MiniLLMTrainer(
        model=student,
        teacher_model=teacher,
        reward_funcs=[response_quality_reward],
        args=config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # Clear cache after training (free memory before merge/export)
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # Merge LoRA for full-model export (needed for GGUF/llama.cpp)
    trainer.model = trainer.model.merge_and_unload()
    trainer.save_model(args.output_dir)
    print(f"Distilled model saved to {args.output_dir}")


if __name__ == "__main__":
    subprocess.Popen(['caffeinate', '-i', 'sleep', '3600'])
    main()
