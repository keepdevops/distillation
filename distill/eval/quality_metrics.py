"""Pure computation metrics for quality evaluation (no model loading required)."""
from __future__ import annotations

import math
import re

from ..data.pipeline import is_refusal

MIN_RESPONSE_TOKENS = 10
MAX_RESPONSE_TOKENS = 2000
TARGET_MIN_TOKENS = 200

JUDGE_PROMPT = (
    "You are evaluating an AI assistant's response.\n\n"
    "Instruction: {instruction}\n"
    "Response: {response}\n\n"
    "Rate the response 1-10 for instruction-following and overall quality. "
    "Reply with the score first, then a one-sentence reason. Example: '8 - Clear and direct.'"
)


def check_length_valid(
    text: str,
    min_tokens: int = MIN_RESPONSE_TOKENS,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> bool:
    """Check if response length is within acceptable range."""
    length = len(text.split())
    return min_tokens <= length <= max_tokens


def check_quality_gates(response: str, instruction: str = "") -> tuple[bool, str, dict]:
    """
    Check if response passes production quality gates.
    Returns (passed: bool, reason: str, flags: dict).
    """
    tokens = response.split()
    length = len(tokens)
    flags: dict = {}

    if length < MIN_RESPONSE_TOKENS:
        return False, f"too_short ({length} < {MIN_RESPONSE_TOKENS})", {"too_short": True}
    if length > MAX_RESPONSE_TOKENS:
        return False, f"too_long ({length} > {MAX_RESPONSE_TOKENS})", {"too_long": True}

    if is_refusal(response):
        return False, "refusal_detected", {"refusal": True}

    if length < TARGET_MIN_TOKENS:
        flags["below_target"] = True

    return True, "passed", flags


def detect_category(instruction: str, response: str = "") -> str:
    """Classify instruction into: math, code, creative, reasoning, qa, other."""
    text = (instruction + " " + response).lower()

    if any(kw in text for kw in [
        "calculate", "compute", "solve", "equation", "number", "sum", "multiply",
        "divide", "percentage", "average", "math", "arithmetic", "algebra",
    ]):
        return "math"

    if any(kw in text for kw in [
        "code", "program", "function", "script", "debug", "implement", "python",
        "javascript", "java", "algorithm", "api", "compile", "syntax",
    ]):
        return "code"

    if any(kw in text for kw in [
        "write a story", "poem", "creative", "imagine", "describe", "narrative",
        "fiction", "character", "plot", "write an essay",
    ]):
        return "creative"

    if any(kw in text for kw in [
        "why", "explain", "reason", "because", "analyze", "compare", "evaluate",
        "logic", "argument", "conclusion", "therefore",
    ]):
        return "reasoning"

    if any(kw in text for kw in [
        "what is", "who is", "when", "where", "how many", "list", "name", "define",
    ]):
        return "qa"

    return "other"


def diversity_metrics(text: str) -> tuple[float, float, int]:
    """Return (distinct-1, distinct-2, max consecutive repeated word run)."""
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


def compute_ngram_entropy(texts: list[str], n: int = 3) -> float:
    """Compute entropy of n-gram distribution across all texts."""
    from collections import Counter
    ngrams = []
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i : i + n]))
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def parse_judge_score(judge_text: str) -> int | None:
    """Extract the first integer 1-10 from judge response."""
    m = re.search(r"\b([1-9]|10)\b", judge_text)
    return int(m.group(1)) if m else None
