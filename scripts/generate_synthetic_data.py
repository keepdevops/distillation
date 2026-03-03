#!/usr/bin/env python3
"""
Synthetic instruction-response data generation using the teacher model.

Self-instruct style pipeline:
  1. Sample seed examples from a base dataset (Alpaca format)
  2. Teacher generates diverse new instructions (deduped by n-gram Jaccard)
  3. Teacher generates responses for each surviving instruction
  4. Filter by: perplexity range, response length, distinct-2 coherence check
  5. Save as HF dataset on disk + synthetic_stats.json

Usage:
    python scripts/generate_synthetic_data.py --open --n_generate 2000 \
        --output_dir ./distilled-minillm

    # Standalone, output to separate dir
    python scripts/generate_synthetic_data.py --open --n_generate 5000 \
        --output_dir ./my_synthetic_data
"""

import argparse
import json
import logging
import math
import os
from pathlib import Path

import torch
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OPEN_TEACHER = "Qwen/Qwen2-1.5B-Instruct"

INSTRUCTION_PROMPT = """\
Here are {n} example instructions for an AI assistant:
{examples}

Write one NEW instruction that:
- Is DIFFERENT in topic and style from all examples above
- Is a clear, specific request a human might ask an AI assistant
- Is between 1 and 3 sentences

New instruction:"""

RESPONSE_PROMPT = """\
You are a helpful AI assistant. Answer the following instruction clearly and concisely.

Instruction: {instruction}

Response:"""


def parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic instruction-response data")
    p.add_argument("--teacher", type=str, default=OPEN_TEACHER,
                   help="Teacher model id or path")
    p.add_argument("--open", action="store_true",
                   help="Use open Qwen2 teacher (Qwen/Qwen2-1.5B-Instruct)")
    p.add_argument("--base_dataset", type=str, default="tatsu-lab/alpaca",
                   help="Seed dataset in Alpaca format")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save synthetic_data/ and synthetic_stats.json")
    p.add_argument("--n_generate", type=int, default=2000,
                   help="Target number of (instruction, response) pairs to generate (default: 2000)")
    p.add_argument("--seed_examples", type=int, default=5,
                   help="Seed examples shown in each instruction generation prompt (default: 5)")
    p.add_argument("--ppl_low", type=float, default=1.2,
                   help="Discard responses with teacher perplexity below this (degenerate/trivial, default: 1.2)")
    p.add_argument("--ppl_high", type=float, default=100.0,
                   help="Discard responses with teacher perplexity above this (incoherent, default: 100.0)")
    p.add_argument("--min_response_tokens", type=int, default=10)
    p.add_argument("--max_response_tokens", type=int, default=400)
    p.add_argument("--min_distinct2", type=float, default=0.3,
                   help="Discard responses with distinct-2 below this (incoherent/repetitive)")
    p.add_argument("--jaccard_threshold", type=float, default=0.7,
                   help="Discard instructions with Jaccard similarity above this vs existing")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Generation batch size (default: 1 — MPS is most stable at 1)")
    p.add_argument("--max_new_tokens_instruction", type=int, default=80)
    p.add_argument("--max_new_tokens_response", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.9,
                   help="Sampling temperature for instruction and response generation")
    p.add_argument("--max_seed_samples", type=int, default=5000,
                   help="Max seed examples to load from base dataset")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ngrams(tokens: list[str], n: int) -> set[str]:
    return {" ".join(tokens[i: i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str, n: int = 3) -> float:
    ta, tb = a.lower().split(), b.lower().split()
    na, nb = ngrams(ta, n), ngrams(tb, n)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def distinct2(text: str) -> float:
    tokens = text.lower().split()
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int,
                  temperature: float, device) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=768)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


@torch.no_grad()
def compute_perplexity(model, tokenizer, prompt: str, response: str,
                       max_length: int, device) -> float:
    """Compute teacher perplexity on (prompt, response) — loss over response tokens only."""
    full_text = prompt + " " + response
    enc = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    prompt_len = prompt_enc["input_ids"].shape[1]

    input_ids = enc["input_ids"].to(device)
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens
    if (labels != -100).sum() == 0:
        return float("inf")

    out = model(input_ids=input_ids, labels=labels)
    return math.exp(min(out.loss.item(), 20))


def load_seeds(base_dataset: str, max_samples: int, cache_dir: str | None,
               offline: bool) -> list[dict]:
    """Load seed examples from a dataset, extracting instruction + response."""
    if Path(base_dataset).exists():
        data = load_from_disk(base_dataset)
        ds = data["train"] if hasattr(data, "__getitem__") and "train" in data else data
    else:
        ds = load_dataset(base_dataset, split="train", cache_dir=cache_dir)

    if max_samples and max_samples < len(ds):
        ds = ds.select(range(max_samples))

    seeds = []
    for row in ds:
        instr = row.get("instruction", row.get("prompt", "")).strip()
        inp = row.get("input", "").strip()
        out = row.get("output", row.get("response", "")).strip()
        if instr and out:
            full_instr = instr + (f"\n\nInput: {inp}" if inp else "")
            seeds.append({"instruction": full_instr, "response": out})
    return seeds


def format_seed_examples(seeds: list[dict], n: int) -> str:
    """Format N random seeds as numbered list for the generation prompt."""
    import random
    chosen = random.sample(seeds, min(n, len(seeds)))
    lines = []
    for i, s in enumerate(chosen, 1):
        instr = s["instruction"].replace("\n", " ").strip()[:200]
        lines.append(f"{i}. {instr}")
    return "\n".join(lines)


def main():
    args = parse_args()
    if args.open:
        args.teacher = OPEN_TEACHER
        logger.info("Using open teacher: %s", args.teacher)

    output_dir = Path(args.output_dir)
    synthetic_dir = output_dir / "synthetic_data"
    stats_path = output_dir / "synthetic_stats.json"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir
    device = get_device()
    logger.info("Device: %s", device)

    # Load teacher
    logger.info("Loading teacher: %s", args.teacher)
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, cache_dir=cache_dir, local_files_only=offline,
    )
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher, dtype=torch.bfloat16, device_map="auto",
        cache_dir=cache_dir, local_files_only=offline,
    )
    model.to(device)
    model.eval()

    # Load seed examples
    logger.info("Loading seeds from %s...", args.base_dataset)
    seeds = load_seeds(args.base_dataset, args.max_seed_samples, ds_cache, offline)
    logger.info("Loaded %d seed examples", len(seeds))

    # Generation loop
    accepted = []
    existing_instructions = [s["instruction"] for s in seeds[:500]]  # dedup pool

    stats = {
        "target": args.n_generate,
        "generated_instructions": 0,
        "filtered_duplicate": 0,
        "filtered_ppl_low": 0,
        "filtered_ppl_high": 0,
        "filtered_length": 0,
        "filtered_distinct2": 0,
        "accepted": 0,
    }

    logger.info("Generating %d synthetic pairs...", args.n_generate)
    attempts = 0
    max_attempts = args.n_generate * 5  # safety cap

    while len(accepted) < args.n_generate and attempts < max_attempts:
        attempts += 1

        # 1. Generate instruction
        seed_block = format_seed_examples(seeds, args.seed_examples)
        instr_prompt = INSTRUCTION_PROMPT.format(
            n=args.seed_examples, examples=seed_block,
        )
        instruction = generate_text(
            model, tokenizer, instr_prompt,
            args.max_new_tokens_instruction, args.temperature, device,
        )
        # Clean up: take first line, strip leading bullets/numbers
        instruction = instruction.split("\n")[0].strip().lstrip("0123456789.-) ")
        # Truncate at first response-like continuation (teacher starting to answer)
        import re as _re
        # Try to extract just the first complete sentence
        m = _re.match(r'^([^.!?]{15,}[.!?])', instruction)
        if m:
            instruction = m.group(1).strip()
        if not instruction or len(instruction) < 10:
            continue
        stats["generated_instructions"] += 1

        # 2. Dedup check
        too_similar = any(
            jaccard(instruction, ex, n=3) > args.jaccard_threshold
            for ex in existing_instructions[-200:]  # rolling window for speed
        )
        if too_similar:
            stats["filtered_duplicate"] += 1
            continue

        # 3. Generate response
        resp_prompt = RESPONSE_PROMPT.format(instruction=instruction)
        response = generate_text(
            model, tokenizer, resp_prompt,
            args.max_new_tokens_response, args.temperature, device,
        )
        if not response:
            continue

        # 4. Length filter
        resp_tokens = len(tokenizer.encode(response, add_special_tokens=False))
        if resp_tokens < args.min_response_tokens or resp_tokens > args.max_response_tokens:
            stats["filtered_length"] += 1
            continue

        # 5. Distinct-2 filter
        d2 = distinct2(response)
        if d2 < args.min_distinct2:
            stats["filtered_distinct2"] += 1
            continue

        # 6. Perplexity filter
        ppl = compute_perplexity(
            model, tokenizer, resp_prompt, response, 768, device,
        )
        if ppl < args.ppl_low:
            stats["filtered_ppl_low"] += 1
            continue
        if ppl > args.ppl_high:
            stats["filtered_ppl_high"] += 1
            continue

        # Accept
        accepted.append({
            "instruction": instruction,
            "input": "",
            "output": response,
            "source": "synthetic",
            "teacher_ppl": round(ppl, 2),
        })
        existing_instructions.append(instruction)
        stats["accepted"] += 1

        if len(accepted) % 100 == 0:
            logger.info(
                "  %d/%d accepted  (dup=%d, ppl_low=%d, ppl_high=%d, len=%d, d2=%d)",
                len(accepted), args.n_generate,
                stats["filtered_duplicate"], stats["filtered_ppl_low"],
                stats["filtered_ppl_high"], stats["filtered_length"],
                stats["filtered_distinct2"],
            )

    logger.info(
        "Generation complete: %d accepted from %d attempts",
        len(accepted), attempts,
    )
    logger.info(
        "Filtered: dup=%d  ppl_low=%d  ppl_high=%d  length=%d  distinct2=%d",
        stats["filtered_duplicate"], stats["filtered_ppl_low"],
        stats["filtered_ppl_high"], stats["filtered_length"],
        stats["filtered_distinct2"],
    )

    if not accepted:
        logger.error("No samples accepted — check filters or teacher model")
        raise SystemExit(1)

    # Save as HF dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(accepted)
    ds.save_to_disk(str(synthetic_dir))
    logger.info("Synthetic dataset saved to %s (%d samples)", synthetic_dir, len(ds))

    # Save stats
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
