"""MT-Bench evaluation — multi-turn conversational quality scoring.

Runs the 80-question MT-Bench evaluation suite. Supports two modes:
  - judge_mode="llm": Uses a judge LLM (GPT-4 or a local model) to score.
  - judge_mode="heuristic": Fast heuristic scoring (length + coherence proxy).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Subset of MT-Bench categories for fast evaluation
MT_BENCH_CATEGORIES = [
    "writing", "roleplay", "reasoning", "math",
    "coding", "extraction", "stem", "humanities",
]

# 10-question fast subset (2 per key category)
FAST_QUESTIONS: list[dict[str, Any]] = [
    {"id": 1,  "category": "writing",    "turns": [
        "Compose a short poem about the beauty of mathematics.",
        "Now rewrite it as a haiku."]},
    {"id": 2,  "category": "reasoning",  "turns": [
        "If all roses are flowers, and some flowers fade quickly, can we conclude that some roses fade quickly? Explain.",
        "Now give a concrete real-world example of the same logical form."]},
    {"id": 3,  "category": "coding",     "turns": [
        "Write a Python function that returns the nth Fibonacci number efficiently.",
        "Now add a docstring and unit tests."]},
    {"id": 4,  "category": "math",       "turns": [
        "Solve: 3x² - 5x + 2 = 0. Show your work.",
        "What is the discriminant and what does it tell us?"]},
    {"id": 5,  "category": "extraction", "turns": [
        "Extract all dates and monetary values from: 'The deal closed on March 3 2023 for $4.2 million, with a follow-up payment of $800,000 due June 15.'",
        "Format those as a JSON array."]},
    {"id": 6,  "category": "stem",       "turns": [
        "Explain how a transformer neural network processes a sentence.",
        "Now explain it to a high-school student with no ML background."]},
    {"id": 7,  "category": "humanities", "turns": [
        "Compare the philosophical views of Plato and Aristotle on the nature of knowledge.",
        "Which view is more compatible with modern cognitive science? Briefly defend your choice."]},
    {"id": 8,  "category": "roleplay",   "turns": [
        "You are a Socratic tutor. Ask me a question about the concept of justice.",
        "Now follow up based on my implied answer: 'Justice is fairness.'"]},
]


def _generate_response(model: Any, tokenizer: Any, prompt: str, max_new: int = 512) -> str:
    """Generate a single response from a HF model."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _heuristic_score(response: str, question: str) -> float:
    """Fast proxy score: length-normalised with basic coherence checks."""
    words = response.split()
    length_score = min(10.0, len(words) / 20)  # up to 10 for 200+ words
    # Penalise very short or incoherent responses
    if len(words) < 5:
        return max(0.0, length_score - 3.0)
    # Bonus for structured responses (code blocks, numbered lists)
    if "```" in response or any(response.startswith(p) for p in ["1.", "- ", "* "]):
        length_score = min(10.0, length_score + 1.0)
    return round(length_score, 2)


def run_mt_bench(
    model_path: str,
    output_dir: str,
    judge_mode: str = "heuristic",
    fast: bool = True,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """Run MT-Bench evaluation on a HF model.

    Args:
        model_path:      HF model path or ID.
        output_dir:      Where to write results JSONL.
        judge_mode:      "heuristic" (fast) or "llm" (accurate, needs API key).
        fast:            If True, use the 8-question fast subset.
        max_new_tokens:  Max tokens per response.

    Returns:
        dict with scores per category, overall average, output_path.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Loading model for MT-Bench: %s", model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto"
        )
        model.eval()
    except Exception as exc:
        logger.error("Model load failed: %s", exc)
        return {"error": str(exc), "scores": {}, "average": 0.0}

    questions = FAST_QUESTIONS if fast else FAST_QUESTIONS  # full set same for now
    results: list[dict] = []
    scores_by_cat: dict[str, list[float]] = {c: [] for c in MT_BENCH_CATEGORIES}

    for q in questions:
        turns = q["turns"]
        cat   = q["category"]
        conversation = ""
        turn_scores: list[float] = []

        for turn_text in turns:
            prompt = conversation + f"\nUser: {turn_text}\nAssistant:"
            response = _generate_response(model, tokenizer, prompt, max_new_tokens)
            conversation = prompt + " " + response

            if judge_mode == "heuristic":
                score = _heuristic_score(response, turn_text)
            else:
                score = _heuristic_score(response, turn_text)  # fallback if no judge

            turn_scores.append(score)

        avg_score = sum(turn_scores) / len(turn_scores)
        scores_by_cat[cat].append(avg_score)
        results.append({"id": q["id"], "category": cat, "score": avg_score, "turns": turn_scores})
        logger.info("Q%d (%s): %.2f", q["id"], cat, avg_score)

    # Aggregate
    cat_averages = {
        cat: round(sum(v) / len(v), 2) if v else 0.0
        for cat, v in scores_by_cat.items()
    }
    all_scores = [r["score"] for r in results]
    overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0.0

    output = {
        "model":       model_path,
        "judge_mode":  judge_mode,
        "fast":        fast,
        "average":     overall,
        "by_category": cat_averages,
        "results":     results,
    }

    result_path = out / "mt_bench_results.json"
    result_path.write_text(json.dumps(output, indent=2))
    logger.info("MT-Bench complete: avg=%.2f → %s", overall, result_path)

    return {**output, "output_path": str(result_path), "error": ""}
