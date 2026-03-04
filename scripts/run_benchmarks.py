#!/usr/bin/env python3
"""
WikiText-2 perplexity benchmark + optional regression detection.

Evaluates a distilled model on the WikiText-2-raw-v1 test split and
optionally compares against a previous baseline's benchmark_results.json.

Appends {"step": ..., "wikitext2_perplexity": ...} to metrics.jsonl.
Saves benchmark_results.json to model dir.

Usage:
    python scripts/run_benchmarks.py ./distilled-minillm
    python scripts/run_benchmarks.py ./distilled-minillm --baseline_dir ./previous-run
    python scripts/run_benchmarks.py ./distilled-minillm --n_sequences 100 --offline
"""

import argparse
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Reuse eval_loss from run_eval to avoid duplication
sys.path.insert(0, str(Path(__file__).parent))
from run_eval import eval_loss, detect_step, last_step_in_jsonl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"
WIKITEXT_DATASET = "wikitext"
WIKITEXT_CONFIG = "wikitext-2-raw-v1"


def parse_args():
    p = argparse.ArgumentParser(description="WikiText-2 benchmark for distilled models")
    p.add_argument("output_dir", type=str, help="Model output dir (results written here)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Checkpoint dir to eval (default: output_dir itself)")
    p.add_argument("--student", type=str, default=OPEN_STUDENT,
                   help="Base model id (fallback for tokenizer)")
    p.add_argument("--n_sequences", type=int, default=500,
                   help="Number of sequences to evaluate (default: 500)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max token length per sequence (default: 512)")
    p.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation (default: 8)")
    p.add_argument("--baseline_dir", type=str, default=None,
                   help="Previous run dir to compare against (regression detection)")
    p.add_argument("--threshold", type=float, default=15.0,
                   help="Max allowed perplexity regression %% vs baseline (default: 15.0)")
    p.add_argument("--step", type=int, default=None,
                   help="Step number to record (default: auto-detect)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wikitext2(n_sequences: int, cache_dir: str | None, offline: bool) -> list[str]:
    """Load WikiText-2 test split, return non-empty text chunks."""
    logger.info("Loading WikiText-2 test split (%d sequences)...", n_sequences)
    ds = load_dataset(WIKITEXT_DATASET, WIKITEXT_CONFIG, split="test",
                      cache_dir=cache_dir)
    texts = [row["text"].strip() for row in ds if row["text"].strip()]
    if n_sequences and n_sequences < len(texts):
        texts = texts[:n_sequences]
    return texts


def load_student_model(checkpoint_dir: Path, student_id: str, cache_dir, offline, device):
    """Load model + tokenizer, handling LoRA adapter checkpoints."""
    tok_dir = str(checkpoint_dir) if (checkpoint_dir / "tokenizer_config.json").exists() else student_id
    tokenizer = AutoTokenizer.from_pretrained(tok_dir, cache_dir=cache_dir, local_files_only=offline)
    tokenizer.pad_token = tokenizer.eos_token

    is_adapter = (checkpoint_dir / "adapter_config.json").exists()
    if is_adapter:
        with open(checkpoint_dir / "adapter_config.json") as f:
            adapter_cfg = json.load(f)
        base_id = adapter_cfg.get("base_model_name_or_path", student_id)
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_id, dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        model = model.merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir), dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
    model.to(device)
    return model, tokenizer


def load_baseline_result(baseline_dir: str) -> dict | None:
    """Load benchmark_results.json from a previous run."""
    path = Path(baseline_dir) / "benchmark_results.json"
    if not path.exists():
        logger.warning("No benchmark_results.json found in baseline dir: %s", baseline_dir)
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load baseline results: %s", e)
        return None


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else output_dir
    jsonl_path = output_dir / "metrics.jsonl"
    results_path = output_dir / "benchmark_results.json"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    device = get_device()
    logger.info("Device: %s", device)

    # Determine step
    step = args.step
    if step is None:
        step = detect_step(checkpoint_dir)
    if step is None:
        step = last_step_in_jsonl(jsonl_path)
    logger.info("Recording benchmark at step %d", step)

    # Load model
    logger.info("Loading model from %s", checkpoint_dir)
    model, tokenizer = load_student_model(checkpoint_dir, args.student, cache_dir, offline, device)

    # Load WikiText-2
    texts = load_wikitext2(args.n_sequences, ds_cache, offline)
    logger.info("Evaluating on %d sequences (max_length=%d)...", len(texts), args.max_length)

    loss = eval_loss(model, tokenizer, texts, args.max_length, args.batch_size, device)
    if loss is None:
        logger.error("Could not compute loss (no tokens)")
        raise SystemExit(1)

    perplexity = math.exp(min(loss, 20))
    logger.info("WikiText-2: loss=%.4f  perplexity=%.2f", loss, perplexity)

    # Free model memory
    del model
    if device.type == "mps":
        torch.mps.empty_cache()

    # Regression detection
    regression_status = "not_checked"
    regression_delta_pct = None
    if args.baseline_dir:
        baseline = load_baseline_result(args.baseline_dir)
        if baseline and "wikitext2_perplexity" in baseline:
            baseline_ppl = baseline["wikitext2_perplexity"]
            delta_pct = (perplexity - baseline_ppl) / baseline_ppl * 100
            regression_delta_pct = round(delta_pct, 2)
            if delta_pct > args.threshold:
                regression_status = "REGRESSION"
                logger.warning(
                    "REGRESSION DETECTED: WikiText-2 perplexity %.2f vs baseline %.2f "
                    "(+%.1f%% > threshold %.1f%%)",
                    perplexity, baseline_ppl, delta_pct, args.threshold,
                )
            else:
                regression_status = "pass"
                logger.info(
                    "Regression check: PASS (%.2f vs baseline %.2f, delta=+%.1f%%)",
                    perplexity, baseline_ppl, delta_pct,
                )

    # Save benchmark_results.json
    result = {
        "step": step,
        "wikitext2_perplexity": round(perplexity, 4),
        "wikitext2_loss": round(loss, 6),
        "n_sequences": len(texts),
        "max_length": args.max_length,
        "regression_status": regression_status,
        "regression_delta_pct": regression_delta_pct,
        "baseline_dir": args.baseline_dir,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Benchmark results saved to %s", results_path)

    # Append to metrics.jsonl
    row = {"step": step, "wikitext2_perplexity": round(perplexity, 4)}
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    logger.info("Appended wikitext2_perplexity to %s", jsonl_path)

    if regression_status == "REGRESSION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
