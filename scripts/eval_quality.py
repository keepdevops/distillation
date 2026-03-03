#!/usr/bin/env python3
"""
Generation-based quality eval: diversity metrics + optional LLM-as-judge.

Samples N prompts from the validation split, generates student responses,
computes distinct-1/distinct-2/max-repetition, and optionally scores each
response with a teacher model (LLM-as-judge).

Output: {output_dir}/quality_metrics.json

Usage:
    python scripts/eval_quality.py ./distilled-minillm
    python scripts/eval_quality.py ./distilled-minillm --judge
    python scripts/eval_quality.py ./distilled-minillm --judge --teacher Qwen/Qwen2-1.5B-Instruct
    python scripts/eval_quality.py ./distilled-minillm --checkpoint ./distilled-minillm/checkpoint-80
"""

import argparse
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_STUDENT = "Qwen/Qwen2-0.5B-Instruct"
OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"

JUDGE_PROMPT = (
    "You are evaluating an AI assistant's response.\n\n"
    "Instruction: {instruction}\n"
    "Response: {response}\n\n"
    "Rate the response 1-10 for instruction-following and overall quality. "
    "Reply with the score first, then a one-sentence reason. Example: '8 - Clear and direct.'"
)


def parse_args():
    p = argparse.ArgumentParser(description="Generation-based quality eval")
    p.add_argument("output_dir", type=str, help="Training output dir (quality_metrics.json written here)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Model checkpoint dir to eval (default: output_dir itself)")
    p.add_argument("--student", type=str, default=OPEN_STUDENT,
                   help="Base model id (fallback for tokenizer)")
    p.add_argument("--teacher", type=str, default=OPEN_TEACHER,
                   help="Teacher model id or path (used with --judge)")
    p.add_argument("--judge", action="store_true",
                   help="Run LLM-as-judge scoring using --teacher")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--n_samples", type=int, default=50,
                   help="Number of prompts to generate and score (default: 50)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def format_example(example):
    prompt = example.get("instruction", example.get("prompt", ""))
    if "input" in example and example["input"]:
        prompt += "\n\nInput: " + example["input"]
    prompt += "\n\n### Response:"
    return {"prompt": prompt, "instruction": example.get("instruction", prompt)}


def load_student(checkpoint_dir, student_id, cache_dir, offline, device):
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
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_response(model, tokenizer, prompt, max_new_tokens, temperature, device):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def diversity_metrics(text):
    """Return distinct-1, distinct-2, max consecutive repeated word run."""
    tokens = text.lower().split()
    if not tokens:
        return 0.0, 0.0, 0
    d1 = len(set(tokens)) / len(tokens)
    bigrams = list(zip(tokens, tokens[1:]))
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0
    max_run = run = 1
    for i in range(1, len(tokens)):
        run = run + 1 if tokens[i] == tokens[i - 1] else 1
        max_run = max(max_run, run)
    return d1, d2, max_run if len(tokens) > 1 else 0


def parse_judge_score(judge_text):
    """Extract the first integer 1-10 from judge response."""
    m = re.search(r"\b([1-9]|10)\b", judge_text)
    return int(m.group(1)) if m else None


@torch.no_grad()
def judge_response(judge_model, judge_tok, instruction, response, device):
    prompt = JUDGE_PROMPT.format(instruction=instruction, response=response)
    inputs = judge_tok(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = judge_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=False,
        pad_token_id=judge_tok.eos_token_id,
    )
    return judge_tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint) if args.checkpoint else output_dir
    out_path = output_dir / "quality_metrics.json"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    device = get_device()
    logger.info("Device: %s", device)

    # Load student
    logger.info("Loading student from %s", checkpoint_dir)
    student, tokenizer = load_student(checkpoint_dir, args.student, cache_dir, offline, device)

    # Load dataset
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    logger.info("Loading dataset: %s", args.dataset)
    if Path(args.dataset).exists():
        data = load_from_disk(args.dataset)
        dataset = data["train"] if isinstance(data, dict) and "train" in data else data
    else:
        dataset = load_dataset(args.dataset, split="train", cache_dir=ds_cache)

    if args.max_samples and args.max_samples < len(dataset):
        dataset = dataset.select(range(args.max_samples))

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    split = dataset.train_test_split(test_size=args.val_size, seed=42)
    val_ds = split["test"]

    n = min(args.n_samples, len(val_ds))
    val_ds = val_ds.select(range(n))
    logger.info("Generating %d responses...", n)

    # Generate + diversity
    samples = []
    d1_sum = d2_sum = 0.0
    max_rep_sum = 0
    lengths = []

    for i, ex in enumerate(val_ds):
        prompt = ex["prompt"]
        instruction = ex.get("instruction", prompt)
        response = generate_response(student, tokenizer, prompt, args.max_new_tokens,
                                     args.temperature, device)
        d1, d2, max_rep = diversity_metrics(response)
        d1_sum += d1
        d2_sum += d2
        max_rep_sum += max_rep
        lengths.append(len(response.split()))
        samples.append({
            "prompt": prompt,
            "instruction": instruction,
            "response": response,
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "max_rep": max_rep,
        })
        if (i + 1) % 10 == 0:
            logger.info("  %d/%d  avg_d1=%.3f  avg_d2=%.3f", i + 1, n, d1_sum / (i + 1), d2_sum / (i + 1))

    avg_d1 = d1_sum / n
    avg_d2 = d2_sum / n
    avg_max_rep = max_rep_sum / n
    median_len = sorted(lengths)[n // 2]

    logger.info("Diversity summary: distinct-1=%.3f  distinct-2=%.3f  avg_max_rep=%.2f",
                avg_d1, avg_d2, avg_max_rep)

    if avg_d1 < 0.5:
        logger.warning("Low distinct-1 (%.3f) — possible mode collapse", avg_d1)
    if avg_max_rep > 3:
        logger.warning("High avg max repetition (%.1f) — check for repetition loops", avg_max_rep)

    # LLM-as-judge
    judge_result = {"enabled": False}
    if args.judge:
        logger.info("Loading teacher judge: %s", args.teacher)
        del student  # free memory before loading teacher
        if device.type == "mps":
            torch.mps.empty_cache()

        judge_model = AutoModelForCausalLM.from_pretrained(
            args.teacher, dtype=torch.bfloat16, device_map="auto",
            cache_dir=cache_dir, local_files_only=offline,
        )
        judge_tok = AutoTokenizer.from_pretrained(
            args.teacher, cache_dir=cache_dir, local_files_only=offline,
        )
        judge_tok.pad_token = judge_tok.eos_token
        judge_model.to(device)
        judge_model.eval()

        scores = []
        logger.info("Judging %d responses...", n)
        for i, s in enumerate(samples):
            raw = judge_response(judge_model, judge_tok, s["instruction"], s["response"], device)
            score = parse_judge_score(raw)
            s["judge_raw"] = raw
            s["judge_score"] = score
            if score is not None:
                scores.append(score)
            if (i + 1) % 10 == 0:
                avg_so_far = sum(scores) / len(scores) if scores else float("nan")
                logger.info("  %d/%d  avg_score=%.2f", i + 1, n, avg_so_far)

        valid_scores = [sc for sc in scores if sc is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
        logger.info("Judge avg score: %.2f / 10 (%d/%d parseable)", avg_score or 0, len(valid_scores), n)

        if avg_score is not None and avg_score < 5:
            logger.warning("Low judge avg score (%.2f) — check instruction corruption or mode collapse",
                           avg_score)

        judge_result = {
            "enabled": True,
            "teacher": args.teacher,
            "avg_score": round(avg_score, 2) if avg_score is not None else None,
            "n_scored": len(valid_scores),
            "scores": scores,
        }

        del judge_model
        if device.type == "mps":
            torch.mps.empty_cache()

    result = {
        "model_dir": str(output_dir),
        "checkpoint": str(checkpoint_dir),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": n,
        "diversity": {
            "avg_distinct_1": round(avg_d1, 4),
            "avg_distinct_2": round(avg_d2, 4),
            "avg_max_rep": round(avg_max_rep, 2),
            "median_response_tokens": median_len,
        },
        "judge": judge_result,
        "samples": samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Quality metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
