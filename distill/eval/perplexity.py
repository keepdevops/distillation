#!/usr/bin/env python3
"""
Standalone eval: compute cross-entropy loss on the validation split and
append eval_loss + perplexity to {output_dir}/metrics.jsonl.

Usage:
    python -m distill.run_eval ./distilled-minillm
    python -m distill.run_eval ./distilled-minillm --checkpoint ./distilled-minillm/checkpoint-80
    python -m distill.run_eval ./distilled-minillm --step 90
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch

from ..infra.cli_common import add_cache_and_offline
from ..data.pipeline import load_dataset_split, format_prompt_full
from ..infra.train_utils import get_device, load_student_model
from ..backends.mlx_utils import is_mlx_available, load_mlx_model, compute_mlx_perplexity
from ..backends.cpp_inference import compute_gguf_perplexity
from ..backends.cpp_utils import is_cpp_available, find_gguf
from .perplexity_utils import (  # noqa: F401 — re-exported for callers
    detect_step, last_step_in_jsonl, eval_loss, eval_model_at_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"


def parse_args():
    p = argparse.ArgumentParser(description="Standalone eval for distilled model")
    p.add_argument("output_dir", type=str, help="Training output dir (metrics.jsonl written here)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint dir to eval (default: output_dir itself)")
    p.add_argument("--student", type=str, default=OPEN_STUDENT,
                   help="Base model id (for tokenizer if checkpoint has none)")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--max_val_samples", type=int, default=200,
                   help="Cap validation samples to keep eval fast")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation (default: 8)")
    p.add_argument("--step", type=int, default=None,
                   help="Step number to record in metrics.jsonl (default: auto-detect from checkpoint)")
    add_cache_and_offline(p)
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "pytorch", "mlx", "gguf"],
                   help="Inference backend: gguf (llama.cpp/Metal) > mlx > pytorch for speed (default: auto)")
    p.add_argument("--compare_teacher", action="store_true",
                   help="Also eval teacher model and log perplexity gap")
    p.add_argument("--teacher", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                   help="Teacher model id or path (used with --compare_teacher)")
    p.add_argument("--quant_dir", type=str, default=None,
                   help="Path to quantized HF-format model dir to compare against student")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else output_dir
    jsonl_path = output_dir / "metrics.jsonl"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir

    # Determine step
    step = args.step
    if step is None:
        step = detect_step(checkpoint_dir)
    if step is None:
        step = last_step_in_jsonl(jsonl_path)
    logger.info("Recording eval at step %d", step)

    # Resolve backend: gguf (C++/Metal) > mlx > pytorch
    use_gguf = False
    use_mlx = False

    if args.backend == "gguf":
        if not is_cpp_available():
            raise SystemExit("--backend gguf requested but llama.cpp binaries not found")
        use_gguf = True
    elif args.backend == "mlx":
        if not is_mlx_available():
            raise SystemExit("--backend mlx requested but mlx/mlx-lm not installed")
        use_mlx = True
    elif args.backend == "auto":
        gguf_candidate = find_gguf(str(checkpoint_dir))
        if gguf_candidate and is_cpp_available():
            use_gguf = True
            logger.info("Backend: GGUF/llama.cpp (auto-detected: %s)", Path(gguf_candidate).name)
        elif is_mlx_available():
            use_mlx = True
            logger.info("Backend: MLX (auto-detected)")

    if not use_gguf and not use_mlx:
        logger.info("Backend: PyTorch")

    # Load dataset (needed regardless of backend for tokenizer fallback path)
    logger.info("Loading dataset %s", args.dataset)
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    dataset = load_dataset_split(args.dataset, args.max_samples, ds_cache, offline)
    dataset = dataset.map(
        lambda ex: {"text": format_prompt_full(ex)},
        remove_columns=dataset.column_names,
    )
    split = dataset.train_test_split(test_size=args.val_size, seed=42)
    val_ds = split["test"]

    if args.max_val_samples and len(val_ds) > args.max_val_samples:
        val_ds = val_ds.select(range(args.max_val_samples))
    logger.info("Validation samples: %d", len(val_ds))
    texts = list(val_ds["text"])

    if use_gguf:
        gguf_path = find_gguf(str(checkpoint_dir))
        if gguf_path is None:
            raise SystemExit(f"No .gguf file found in {checkpoint_dir}")
        logger.info("GGUF model: %s", gguf_path)
        loss = compute_gguf_perplexity(gguf_path, texts, ctx_size=args.max_length)
    elif use_mlx:
        # MLX path: batched forward pass, no autoregressive decode
        model, tokenizer = load_mlx_model(str(checkpoint_dir))
        loss = compute_mlx_perplexity(model, tokenizer, texts, args.max_length, args.batch_size)
        del model
        import mlx.core as mx
        mx.clear_cache()
    else:
        # PyTorch path
        logger.info("Loading model from %s", checkpoint_dir)
        device = get_device()
        model, tokenizer = load_student_model(checkpoint_dir, args.student, cache_dir, offline, device)
        loss = eval_loss(model, tokenizer, texts, args.max_length, args.batch_size, device)
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    if loss is None:
        logger.error("Could not compute eval loss (no tokens)")
        raise SystemExit(1)

    perplexity = math.exp(min(loss, 20))
    logger.info("step=%d  student: eval_loss=%.4f  perplexity=%.2f", step, loss, perplexity)

    row = {"step": step, "eval_loss": loss, "eval_perplexity": perplexity}

    if args.compare_teacher:
        logger.info("Evaluating teacher: %s", args.teacher)
        try:
            if use_gguf:
                t_gguf = find_gguf(args.teacher)
                if t_gguf is None:
                    raise FileNotFoundError(f"No .gguf found for teacher: {args.teacher}")
                t_loss = compute_gguf_perplexity(t_gguf, texts, ctx_size=args.max_length)
            elif use_mlx:
                t_model, t_tok = load_mlx_model(args.teacher)
                t_loss = compute_mlx_perplexity(t_model, t_tok, texts, args.max_length, args.batch_size)
                del t_model
                import mlx.core as mx; mx.clear_cache()
            else:
                t_loss = eval_model_at_path(
                    args.teacher, tokenizer, texts, args.max_length, args.batch_size, device,
                    cache_dir=cache_dir, offline=offline,
                )
            t_ppl = math.exp(min(t_loss, 20))
            gap_pct = (perplexity - t_ppl) / t_ppl * 100
            logger.info("teacher: eval_loss=%.4f  perplexity=%.2f  ppl_gap=+%.1f%%",
                        t_loss, t_ppl, gap_pct)
            row.update({
                "teacher_eval_loss": t_loss,
                "teacher_eval_perplexity": t_ppl,
                "ppl_gap_pct": gap_pct,
            })
        except Exception:
            raise

    if args.quant_dir:
        qpath = Path(args.quant_dir)
        if qpath.exists():
            logger.info("Evaluating quantized model: %s", qpath)
            try:
                if use_gguf or find_gguf(str(qpath)):
                    q_gguf = find_gguf(str(qpath))
                    if q_gguf is None:
                        raise FileNotFoundError(f"No .gguf found in: {qpath}")
                    q_loss = compute_gguf_perplexity(q_gguf, texts, ctx_size=args.max_length)
                elif use_mlx:
                    q_model, q_tok = load_mlx_model(str(qpath))
                    q_loss = compute_mlx_perplexity(q_model, q_tok, texts, args.max_length, args.batch_size)
                    del q_model
                    import mlx.core as mx; mx.clear_cache()
                else:
                    q_loss = eval_model_at_path(
                        str(qpath), tokenizer, texts, args.max_length, args.batch_size, device,
                        cache_dir=None, offline=True,
                    )
                q_ppl = math.exp(min(q_loss, 20))
                q_gap_pct = (q_ppl - perplexity) / perplexity * 100
                logger.info("quant:  eval_loss=%.4f  perplexity=%.2f  gap_vs_fp16=+%.1f%%",
                            q_loss, q_ppl, q_gap_pct)
                row.update({
                    "quant_eval_loss": q_loss,
                    "quant_eval_perplexity": q_ppl,
                    "quant_ppl_gap_pct": q_gap_pct,
                })
            except Exception:
                raise
        else:
            logger.warning("--quant_dir not found: %s", args.quant_dir)

    # Append to metrics.jsonl
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Appended to %s", jsonl_path)


if __name__ == "__main__":
    main()
