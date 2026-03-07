#!/usr/bin/env python3
"""
Aggressive quality filter for instruction-tuning datasets.

Filters tatsu-lab/alpaca (or any alpaca-schema dataset) down to
8k–20k high-quality pairs using:
  - Length bounds (instruction + response)
  - Refusal detection
  - Distinct-2 coherence check
  - Near-dedup via 3-gram Jaccard
  - Optional: response-length percentile cutoff for top-K selection

Output: HF dataset saved to disk, loadable by load_dataset_split().

Usage:
    # Filter full alpaca → ~10k high-quality pairs
    python scripts/filter_dataset.py --output_dir ./filtered_alpaca_v2

    # Stricter: top 8k by response length + diversity
    python scripts/filter_dataset.py --output_dir ./filtered_alpaca_v2 --target 8000

    # From cached/offline dataset
    python scripts/filter_dataset.py --output_dir ./filtered_alpaca_v2 --offline
"""

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
from data_pipeline import load_dataset_split, _extract_pair

# ── Filter thresholds ────────────────────────────────────────────────────────
MIN_INSTRUCTION_WORDS = 5
MAX_INSTRUCTION_WORDS = 200

MIN_RESPONSE_WORDS = 20
MAX_RESPONSE_WORDS = 600

MIN_DISTINCT2 = 0.35       # distinct-2 on response (filters repetitive/degenerate)
JACCARD_DEDUP_THRESHOLD = 0.6   # n-gram Jaccard; instructions above this are near-dups

REFUSAL_PATTERNS = [
    r"(?i)I(?:'m| am) sorry,?\s+(?:but\s+)?I (?:can'?t|cannot|am unable to)",
    r"(?i)I (?:can'?t|cannot) (?:help|assist|provide|do)",
    r"(?i)As an AI(?: language model| assistant)?,?\s+I (?:can'?t|cannot|don't|do not)",
    r"(?i)I'?m not (?:able|allowed|programmed) to",
    r"(?i)I don'?t have (?:the ability|access|permission)",
    r"(?i)I cannot (and will not|fulfil)",
]

NOISE_PATTERNS = [
    r"(?i)^\s*N/?A\s*$",           # bare "N/A" responses
    r"(?i)^\s*none\.?\s*$",        # bare "None"
    r"(?i)^\s*I don'?t know\.?\s*$",
]

TASK_VERBS = {
    "explain", "describe", "compare", "analyze", "list", "write", "create",
    "summarize", "calculate", "solve", "translate", "classify", "define",
    "evaluate", "discuss", "identify", "generate", "convert", "implement",
    "design", "suggest", "outline", "provide", "demonstrate",
}
QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which"}


def distinct2(text: str) -> float:
    tokens = text.lower().split()
    bigrams = list(zip(tokens, tokens[1:]))
    if not bigrams:
        return 0.0
    return len(set(bigrams)) / len(bigrams)


def ngrams(tokens: list, n: int) -> set:
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def jaccard(a: str, b: str, n: int = 3) -> float:
    ta, tb = a.lower().split(), b.lower().split()
    na, nb = ngrams(ta, n), ngrams(tb, n)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return len(na & nb) / len(na | nb)


def is_refusal(text: str) -> bool:
    for p in REFUSAL_PATTERNS:
        if re.search(p, text):
            return True
    return False


def is_noise(text: str) -> bool:
    for p in NOISE_PATTERNS:
        if re.search(p, text):
            return True
    return False


def score_example(instruction: str, response: str) -> float:
    """
    Composite quality score (higher = better).
    Combines: bigram diversity, sentence-level variety, instruction complexity,
    specificity signals (numbers, structure), task-verb and question presence.
    """
    r_toks = response.split()
    i_toks = instruction.lower().split()
    d2 = distinct2(response)

    # Sentence-level diversity (unique sentences / total sentences)
    sentences = [s.strip() for s in re.split(r"[.!?]+", response) if len(s.strip()) > 8]
    sent_div = len({s.lower() for s in sentences}) / max(len(sentences), 1)

    # Specificity: numbers, bullet lists, code blocks → richer responses
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", response))
    has_structure = bool(re.search(r"(?:\n[-•*]|\n\d+\.|\`\`\`)", response))
    specificity = (0.05 if has_numbers else 0.0) + (0.05 if has_structure else 0.0)

    # Instruction complexity
    task_bonus = 0.10 if any(w in TASK_VERBS for w in i_toks) else 0.0
    question_bonus = 0.05 if any(w in QUESTION_WORDS for w in i_toks) else 0.0
    i_len_score = min(len(i_toks), 50) / 50 * 0.10

    # Core response quality (bigram diversity × length saturation)
    r_len_score = d2 * min(len(r_toks), 300) / 300 * 0.50

    return r_len_score + sent_div * 0.10 + i_len_score + task_bonus + question_bonus + specificity


def minhash_dedup(items: list, threshold: float, num_perm: int = 128) -> tuple[list, int] | None:
    """Global near-dedup via MinHash LSH. Returns None if datasketch not installed."""
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        return None
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    kept = []
    removed = 0
    for i, item in enumerate(items):
        m = MinHash(num_perm=num_perm)
        for tok in item["instruction"].lower().split():
            m.update(tok.encode())
        key = f"i{i}"
        if lsh.query(m):
            removed += 1
        else:
            lsh.insert(key, m)
            kept.append(item)
    return kept, removed


def score_with_teacher(items: list, teacher_id: str, batch_size: int = 1) -> list[float]:
    """
    Score each item by teacher NLL over the response given the prompt.
    Lower NLL = teacher assigns higher probability = better quality.
    Loads teacher on MPS (M3 Max), unloads and clears cache when done.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info("Loading teacher '%s' for scoring on %s ...", teacher_id, device)

    tok = AutoTokenizer.from_pretrained(teacher_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        teacher_id, torch_dtype=torch.float16
    ).to(device)
    model.eval()

    nlls = []
    for idx, item in enumerate(items):
        prompt_text = item["instruction"] + "\n\n### Response:\n"
        full_text = prompt_text + item["output"]
        p_ids = tok(prompt_text, return_tensors="pt", truncation=True, max_length=384)["input_ids"]
        f_ids = tok(full_text, return_tensors="pt", truncation=True, max_length=512)["input_ids"]
        p_len, f_len = p_ids.shape[1], f_ids.shape[1]
        if f_len <= p_len + 2:
            nlls.append(float("inf"))
            continue
        with torch.no_grad():
            logits = model(f_ids.to(device)).logits[0, p_len - 1: f_len - 1]
        targets = f_ids[0, p_len:f_len].to(device)
        nll = torch.nn.functional.cross_entropy(logits, targets).item()
        nlls.append(nll)
        if (idx + 1) % 200 == 0:
            logger.info("Teacher scoring: %d / %d (last NLL=%.3f)", idx + 1, len(items), nll)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return nlls


def parse_args():
    p = argparse.ArgumentParser(description="Filter dataset to high-quality subset")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca",
                   help="HF dataset ID or local path (default: tatsu-lab/alpaca)")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save filtered dataset")
    p.add_argument("--target", type=int, default=None,
                   help="If set, keep top-N by quality score after filtering (default: keep all passing)")
    p.add_argument("--max_load", type=int, default=None,
                   help="Max samples to load from source (default: all)")
    p.add_argument("--min_response_words", type=int, default=MIN_RESPONSE_WORDS)
    p.add_argument("--max_response_words", type=int, default=MAX_RESPONSE_WORDS)
    p.add_argument("--min_instruction_words", type=int, default=MIN_INSTRUCTION_WORDS)
    p.add_argument("--min_distinct2", type=float, default=MIN_DISTINCT2)
    p.add_argument("--jaccard_threshold", type=float, default=JACCARD_DEDUP_THRESHOLD)
    p.add_argument("--no_dedup", action="store_true", help="Skip near-dedup step (faster)")
    p.add_argument("--minhash", action="store_true",
                   help="Use MinHash LSH for global near-dedup (requires: pip install datasketch)")
    p.add_argument("--teacher_score", action="store_true",
                   help="Re-rank top-N candidates by teacher log-probability (requires transformers)")
    p.add_argument("--teacher", type=str, default="Qwen/Qwen2-1.5B-Instruct",
                   help="Teacher model ID for --teacher_score (default: Qwen/Qwen2-1.5B-Instruct)")
    p.add_argument("--teacher_batch_size", type=int, default=1,
                   help="Batch size for teacher scoring forward passes (default: 1)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)

    offline = args.offline or os.environ.get("HF_HUB_OFFLINE") == "1"
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    cache_dir = os.environ.get("HF_HOME") or args.cache_dir
    ds_cache = os.environ.get("HF_DATASETS_CACHE") or args.cache_dir

    logger.info("Loading dataset: %s", args.dataset)
    ds = load_dataset_split(args.dataset, args.max_load, ds_cache, offline)
    logger.info("Loaded %d samples", len(ds))

    stats = {
        "source": args.dataset,
        "n_loaded": len(ds),
        "filtered_empty": 0,
        "filtered_instruction_length": 0,
        "filtered_response_length": 0,
        "filtered_refusal": 0,
        "filtered_noise": 0,
        "filtered_distinct2": 0,
        "filtered_dedup": 0,
        "n_passed": 0,
        "n_final": 0,
    }

    # ── Pass 1: per-example quality filters ──────────────────────────────────
    passed = []
    for ex in ds:
        instruction, response = _extract_pair(ex)

        # Empty fields
        if not instruction or not response:
            stats["filtered_empty"] += 1
            continue

        i_words = len(instruction.split())
        r_words = len(response.split())

        # Instruction length
        if i_words < args.min_instruction_words or i_words > MAX_INSTRUCTION_WORDS:
            stats["filtered_instruction_length"] += 1
            continue

        # Response length
        if r_words < args.min_response_words or r_words > args.max_response_words:
            stats["filtered_response_length"] += 1
            continue

        # Refusal
        if is_refusal(response):
            stats["filtered_refusal"] += 1
            continue

        # Noise
        if is_noise(response):
            stats["filtered_noise"] += 1
            continue

        # Distinct-2
        if distinct2(response) < args.min_distinct2:
            stats["filtered_distinct2"] += 1
            continue

        passed.append({
            "instruction": instruction,
            "input": ex.get("input", "").strip() if hasattr(ex, "get") else "",
            "output": response,
            "_score": score_example(instruction, response),
        })

    logger.info(
        "Pass 1 complete: %d / %d passed  "
        "(empty=%d, instr_len=%d, resp_len=%d, refusal=%d, noise=%d, d2=%d)",
        len(passed), len(ds),
        stats["filtered_empty"],
        stats["filtered_instruction_length"],
        stats["filtered_response_length"],
        stats["filtered_refusal"],
        stats["filtered_noise"],
        stats["filtered_distinct2"],
    )
    stats["n_passed"] = len(passed)

    # ── Pass 2: near-dedup ────────────────────────────────────────────────────
    if not args.no_dedup:
        deduped = None
        if args.minhash:
            logger.info("Pass 2: MinHash LSH dedup (threshold=%.2f)...", args.jaccard_threshold)
            result = minhash_dedup(passed, threshold=args.jaccard_threshold)
            if result is not None:
                deduped, n_removed = result
                stats["filtered_dedup"] = n_removed
                logger.info(
                    "Pass 2 complete: %d / %d after MinHash dedup (removed %d near-dups)",
                    len(deduped), len(passed), n_removed,
                )
            else:
                logger.warning(
                    "datasketch not installed — falling back to sliding-window dedup. "
                    "Install with: pip install datasketch"
                )

        if deduped is None:  # sliding-window fallback
            logger.info("Pass 2: sliding-window Jaccard dedup (threshold=%.2f, window=500)...",
                        args.jaccard_threshold)
            deduped = []
            accepted_instrs: list[str] = []
            for item in passed:
                instr = item["instruction"]
                window = accepted_instrs[-500:]
                if any(jaccard(instr, ex) > args.jaccard_threshold for ex in window):
                    stats["filtered_dedup"] += 1
                    continue
                deduped.append(item)
                accepted_instrs.append(instr)
            logger.info(
                "Pass 2 complete: %d / %d after dedup (removed %d near-dups)",
                len(deduped), len(passed), stats["filtered_dedup"],
            )
    else:
        deduped = passed

    # ── Optional: teacher log-prob re-ranking ────────────────────────────────
    _teacher_ranked = False
    if args.teacher_score and args.target and len(deduped) > args.target:
        logger.info(
            "Teacher scoring %d candidates → selecting top-%d by NLL...",
            len(deduped), args.target,
        )
        try:
            nlls = score_with_teacher(deduped, args.teacher, args.teacher_batch_size)
            for item, nll in zip(deduped, nlls):
                item["_teacher_nll"] = nll
            deduped.sort(key=lambda x: x.get("_teacher_nll", float("inf")))
            valid_nlls = [n for n in nlls if n != float("inf")]
            if valid_nlls:
                logger.info(
                    "Teacher NLL range: best=%.3f  worst=%.3f  median=%.3f",
                    min(valid_nlls), max(valid_nlls),
                    sorted(valid_nlls)[len(valid_nlls) // 2],
                )
            deduped = deduped[: args.target]
            _teacher_ranked = True
            logger.info("Teacher-ranked: kept top-%d by NLL", args.target)
        except Exception as e:
            logger.warning("Teacher scoring failed (non-fatal): %s", e)

    # ── Optional: top-N by heuristic quality score ────────────────────────────
    if not _teacher_ranked and args.target and len(deduped) > args.target:
        deduped.sort(key=lambda x: x["_score"], reverse=True)
        deduped = deduped[: args.target]
        logger.info("Trimmed to top-%d by heuristic quality score", args.target)

    # Clean internal score fields
    for item in deduped:
        item.pop("_score", None)
        item.pop("_teacher_nll", None)

    stats["n_final"] = len(deduped)
    logger.info("Final dataset: %d samples", len(deduped))

    if len(deduped) < 1000:
        logger.warning(
            "Only %d samples survived — consider relaxing filters or using a larger source dataset.",
            len(deduped),
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    from datasets import Dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    hf_ds = Dataset.from_list(deduped)
    hf_ds.save_to_disk(str(output_dir))
    logger.info("Saved to %s", output_dir)

    stats_path = output_dir / "filter_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats: %s", json.dumps({k: v for k, v in stats.items() if k != "source"}, separators=(", ", "=")))
    logger.info("Filter stats saved to %s", stats_path)


if __name__ == "__main__":
    main()
