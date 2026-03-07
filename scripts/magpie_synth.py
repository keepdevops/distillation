#!/usr/bin/env python3
"""
Magpie-style self-synthesis: auto-generate (instruction, response) pairs
from a teacher model by conditioning on the chat-template user-turn prefix.

Reference: "Magpie: Alignment Data Synthesis from Scratch by Conditioning
LLMs on Their Instruction-Following Demonstrations" (Xu et al., 2024)

How it works:
  1. Feed the model just the opening of a user turn:
       <|im_start|>system\\n{sys_prompt}<|im_end|>\\n<|im_start|>user\\n
  2. The model generates what it "thinks" a user would ask  →  instruction
  3. Append <|im_end|>\\n<|im_start|>assistant\\n and generate  →  response
  4. Filter inline (length, refusal, distinct-2); save as alpaca JSONL
  5. --filter pipes the JSONL through filter_dataset.py (dedup, teacher NLL)

Output is alpaca-schema JSONL + an HF dataset directory, both loadable by
load_dataset_split() for direct use as distillation training data.

Usage:
    # Quick bootstrap: generate 10 k pairs, keep best ~5 k
    python scripts/magpie_synth.py --n 10000 --output_dir ./magpie_data

    # With deep filter (dedup + teacher NLL re-ranking)
    python scripts/magpie_synth.py --n 20000 --output_dir ./magpie_data --filter --target 8000

    # Offline (uses cached teacher)
    python scripts/magpie_synth.py --n 5000 --output_dir ./magpie_data --offline
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOG = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

# ── Diverse system prompts to encourage instruction variety ──────────────────
SYSTEM_PROMPTS = [
    "You are a helpful assistant.",
    "You are an expert software engineer. Help with programming and debugging.",
    "You are a mathematics tutor. Help solve problems step by step.",
    "You are a creative writing coach. Help with stories, poetry, and prose.",
    "You are a science educator. Explain concepts clearly with examples.",
    "You are a data analysis expert. Help with statistics and data interpretation.",
    "You are a language learning tutor. Help with grammar, vocabulary, and usage.",
    "You are a logic and reasoning coach. Help solve puzzles and structured problems.",
    "You are a writing editor. Help improve clarity, style, and argumentation.",
    "You are a curious generalist. Answer factual questions thoughtfully.",
    "You are a research assistant. Summarize and synthesize information.",
    "You are an expert in algorithms and computer science fundamentals.",
    "You are a philosophy discussion partner. Explore ideas rigorously.",
    "You are a practical life-skills advisor. Give concrete, actionable advice.",
    "You are a history and social science educator.",
    "You are a brainstorming partner. Generate and expand ideas creatively.",
]

# ── Inline quality filters (fast, no external deps) ─────────────────────────
MIN_INSTRUCTION_WORDS = 5
MAX_INSTRUCTION_WORDS = 180
MIN_RESPONSE_WORDS    = 20
MAX_RESPONSE_WORDS    = 600
MIN_DISTINCT2         = 0.30

REFUSAL_PATTERNS = [
    r"(?i)I(?:'m| am) sorry,?\s+(?:but\s+)?I (?:can'?t|cannot|am unable to)",
    r"(?i)I (?:can'?t|cannot) (?:help|assist|provide|do)",
    r"(?i)As an AI(?: language model| assistant)?,?\s+I (?:can'?t|cannot|don't)",
    r"(?i)I'?m not (?:able|allowed|programmed) to",
]
NOISE_PATTERNS = [
    r"(?i)^\s*N/?A\s*$",
    r"(?i)^\s*none\.?\s*$",
    r"(?i)^\s*I don'?t know\.?\s*$",
    r"(?i)^\s*\[.*?\]\s*$",        # bare bracketed placeholders
]


def _distinct2(text: str) -> float:
    toks = text.lower().split()
    bg = list(zip(toks, toks[1:]))
    return len(set(bg)) / len(bg) if bg else 0.0


def _passes_filter(instruction: str, response: str) -> bool:
    if not instruction or not response:
        return False
    iw = len(instruction.split())
    rw = len(response.split())
    if not (MIN_INSTRUCTION_WORDS <= iw <= MAX_INSTRUCTION_WORDS):
        return False
    if not (MIN_RESPONSE_WORDS <= rw <= MAX_RESPONSE_WORDS):
        return False
    for p in REFUSAL_PATTERNS:
        if re.search(p, response):
            return False
    for p in NOISE_PATTERNS:
        if re.search(p, response) or re.search(p, instruction):
            return False
    if _distinct2(response) < MIN_DISTINCT2:
        return False
    return True


# ── Generation ───────────────────────────────────────────────────────────────

def _build_instruction_prefix(system_prompt: str) -> str:
    """Prefix that ends just before the user's message — model generates the instruction."""
    return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n"


def _build_response_prefix(system_prompt: str, instruction: str) -> str:
    """Prefix that ends just before the assistant's response."""
    return (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _decode_new_tokens(tokenizer, output_ids, prefix_len: int, im_end_id: int) -> str:
    """Decode tokens generated after prefix_len; truncate at <|im_end|>."""
    new_ids = output_ids[prefix_len:]
    # Find first im_end in new tokens
    for i, tok in enumerate(new_ids):
        if tok == im_end_id:
            new_ids = new_ids[:i]
            break
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int, temperature: float,
                   device, im_end_id: int) -> list[str]:
    """Tokenize prompts (left-padded), generate, decode new tokens only."""
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=768,
    ).to(device)

    # Use the full padded input length as the cutoff — all sequences share the
    # same padded length so this correctly skips ALL input tokens (including pads).
    total_input_len = enc["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=[tokenizer.eos_token_id, im_end_id],
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    return [
        _decode_new_tokens(tokenizer, out[i].tolist(), total_input_len, im_end_id)
        for i in range(len(prompts))
    ]


# ── Convert JSONL → HF dataset ───────────────────────────────────────────────

def jsonl_to_hf_dataset(jsonl_path: Path, hf_dir: Path) -> int:
    from datasets import Dataset
    items = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    if not items:
        return 0
    ds = Dataset.from_list(items)
    hf_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(hf_dir))
    return len(items)


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Magpie self-synthesis for distillation")
    p.add_argument("--teacher", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                   help="Teacher model ID for generation (default: Qwen/Qwen2-1.5B-Instruct)")
    p.add_argument("--n", type=int, default=10000,
                   help="Number of pairs to GENERATE before filtering (default: 10000). "
                        "Expect ~40-60%% to survive inline filters.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to write magpie_raw.jsonl + hf_dataset/")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Generation batch size (default: 8, reduce if OOM)")
    p.add_argument("--max_instruction_tokens", type=int, default=150,
                   help="Max new tokens for instruction generation (default: 150)")
    p.add_argument("--max_response_tokens", type=int, default=512,
                   help="Max new tokens for response generation (default: 512)")
    p.add_argument("--inst_temp", type=float, default=0.9,
                   help="Sampling temperature for instruction generation (default: 0.9)")
    p.add_argument("--resp_temp", type=float, default=0.7,
                   help="Sampling temperature for response generation (default: 0.7)")
    p.add_argument("--filter", action="store_true",
                   help="Run filter_dataset.py on the output (dedup + optional teacher NLL)")
    p.add_argument("--target", type=int, default=None,
                   help="Target size for --filter top-N selection (default: all that pass)")
    p.add_argument("--teacher_score", action="store_true",
                   help="Use teacher NLL scoring in filter step (slower, higher quality)")
    p.add_argument("--save_every", type=int, default=200,
                   help="Flush to disk every N kept examples (default: 200)")
    p.add_argument("--resume", action="store_true",
                   help="Resume from existing magpie_raw.jsonl (skip already-generated count)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true",
                   help="Air-gapped: use local HF cache only")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "magpie_raw.jsonl"
    hf_dir     = output_dir / "hf_dataset"

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    # ── Detect device ────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    LOG.info("Device: %s", device)

    # ── Load teacher ─────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cache_dir = os.environ.get("HF_HOME") or args.cache_dir

    LOG.info("Loading teacher: %s", args.teacher)
    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher, cache_dir=cache_dir, local_files_only=offline,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=cache_dir,
        local_files_only=offline,
    )
    model.eval()

    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id == tokenizer.unk_token_id:
        # Fallback for non-ChatML models: use newline + EOS as stop
        im_end_id = tokenizer.eos_token_id
        LOG.warning("Model does not use <|im_end|> — stop token set to eos_token_id")

    # ── Resume: count already-generated examples ─────────────────────────────
    n_already = 0
    if args.resume and jsonl_path.exists():
        with open(jsonl_path) as f:
            n_already = sum(1 for line in f if line.strip())
        LOG.info("Resuming: found %d existing examples, generating %d more",
                 n_already, max(0, args.n - n_already))
    n_to_generate = max(0, args.n - n_already)

    if n_to_generate == 0:
        LOG.info("Already have %d examples (--n %d). Skipping generation.", n_already, args.n)
    else:
        # ── Generation loop ──────────────────────────────────────────────────
        n_generated = 0
        n_kept      = n_already
        stats = {"empty_instr": 0, "empty_resp": 0, "filtered": 0}

        with open(jsonl_path, "a", buffering=1) as fout:
            sys_cycle = 0
            while n_generated < n_to_generate:
                batch = min(args.batch_size, n_to_generate - n_generated)

                # Pick system prompts (cycle through for diversity)
                sys_prompts = [
                    SYSTEM_PROMPTS[(sys_cycle + i) % len(SYSTEM_PROMPTS)]
                    for i in range(batch)
                ]
                sys_cycle = (sys_cycle + batch) % len(SYSTEM_PROMPTS)

                # ── Phase 1: generate instructions ───────────────────────────
                inst_prefixes = [_build_instruction_prefix(sp) for sp in sys_prompts]
                try:
                    instructions = generate_batch(
                        model, tokenizer, inst_prefixes,
                        max_new_tokens=args.max_instruction_tokens,
                        temperature=args.inst_temp,
                        device=device,
                        im_end_id=im_end_id,
                    )
                except Exception as e:
                    LOG.warning("Instruction batch failed: %s", e)
                    n_generated += batch
                    continue

                # Drop empty instructions
                valid = [(sp, instr) for sp, instr in zip(sys_prompts, instructions) if instr]
                stats["empty_instr"] += batch - len(valid)

                if not valid:
                    n_generated += batch
                    continue

                valid_sys, valid_instr = zip(*valid)

                # ── Phase 2: generate responses ──────────────────────────────
                resp_prefixes = [
                    _build_response_prefix(sp, instr)
                    for sp, instr in zip(valid_sys, valid_instr)
                ]
                try:
                    responses = generate_batch(
                        model, tokenizer, resp_prefixes,
                        max_new_tokens=args.max_response_tokens,
                        temperature=args.resp_temp,
                        device=device,
                        im_end_id=im_end_id,
                    )
                except Exception as e:
                    LOG.warning("Response batch failed: %s", e)
                    n_generated += batch
                    continue

                # ── Inline filter + save ─────────────────────────────────────
                for instr, resp in zip(valid_instr, responses):
                    if not resp:
                        stats["empty_resp"] += 1
                        continue
                    if not _passes_filter(instr, resp):
                        stats["filtered"] += 1
                        continue
                    fout.write(json.dumps({"instruction": instr, "input": "", "output": resp}))
                    fout.write("\n")
                    n_kept += 1

                n_generated += batch

                if n_generated % args.save_every == 0 or n_generated >= n_to_generate:
                    pass_rate = (n_kept - n_already) / max(n_generated, 1) * 100
                    LOG.info(
                        "Progress: generated=%d/%d  kept=%d  pass_rate=%.1f%%  "
                        "(empty_instr=%d empty_resp=%d filtered=%d)",
                        n_generated, n_to_generate, n_kept, pass_rate,
                        stats["empty_instr"], stats["empty_resp"], stats["filtered"],
                    )
                    # Rebuild HF dataset incrementally so it's always usable on interrupt
                    fout.flush()
                    jsonl_to_hf_dataset(jsonl_path, hf_dir)

        LOG.info("Generation complete: %d generated, %d kept (%.1f%% pass rate)",
                 n_generated, n_kept - n_already,
                 (n_kept - n_already) / max(n_generated, 1) * 100)

    # ── Build HF dataset from JSONL ──────────────────────────────────────────
    n_total = jsonl_to_hf_dataset(jsonl_path, hf_dir)
    LOG.info("HF dataset saved: %s  (%d examples)", hf_dir, n_total)

    # ── Optional: deep filter ────────────────────────────────────────────────
    if args.filter:
        filtered_dir = output_dir / "filtered"
        filter_cmd = [
            sys.executable, "scripts/filter_dataset.py",
            "--dataset", str(hf_dir),
            "--output_dir", str(filtered_dir),
            "--minhash",
        ]
        if args.target:
            filter_cmd += ["--target", str(args.target)]
        if args.teacher_score:
            filter_cmd += ["--teacher_score", "--teacher", args.teacher]
        if args.offline:
            filter_cmd += ["--offline"]

        LOG.info("Running filter_dataset.py → %s", filtered_dir)
        import subprocess
        result = subprocess.run(filter_cmd, cwd=Path(__file__).parent.parent)
        if result.returncode == 0:
            LOG.info("Filtered dataset saved: %s", filtered_dir)
        else:
            LOG.warning("filter_dataset.py exited with code %d", result.returncode)

    LOG.info(
        "Done. Use with distillation:\n"
        "  python scripts/run_distillation_agent.py --dataset %s --backend mlx --open",
        hf_dir if not args.filter else output_dir / "filtered",
    )


if __name__ == "__main__":
    main()
