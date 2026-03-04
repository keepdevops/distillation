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
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Production quality gates
MIN_RESPONSE_TOKENS = 10    # Minimum viable response length
MAX_RESPONSE_TOKENS = 2000  # Maximum to prevent excessive generation
TARGET_MIN_TOKENS = 200     # Target minimum for quality samples
REFUSAL_PATTERNS = [
    r"(?i)I(?:'m| am) sorry,?\s+(?:but\s+)?I (?:can'?t|cannot|am unable to)",
    r"(?i)I (?:can'?t|cannot) (?:help|assist|provide|do)",
    r"(?i)As an AI(?: language model| assistant)?,?\s+I (?:can'?t|cannot|don't|do not)",
    r"(?i)I'?m not (?:able|allowed|programmed) to",
    r"(?i)I don'?t have (?:the ability|access|permission)",
]

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
    p.add_argument("--judge-teacher-ppl", action="store_true",
                   help="Compute teacher perplexity on student generations")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca")
    p.add_argument("--max_samples", type=int, default=2000)
    p.add_argument("--val_size", type=float, default=0.02)
    p.add_argument("--n_samples", type=int, default=50,
                   help="Number of prompts to generate and score (default: 50)")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens to generate (default: 512, allows 200-2000 range)")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for generation and judging (default: 8)")
    p.add_argument("--cache_dir", type=str, default=None)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "mlx"],
                   help="Backend for inference (mlx for Apple Silicon optimization)")
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


def load_mlx_model(model_path):
    """Load model using MLX for Apple Silicon optimization."""
    try:
        from mlx_lm import load
    except ImportError:
        raise ImportError("MLX not installed. Install with: pip install mlx-lm")

    logger.info("Loading model with MLX backend...")
    model, tokenizer = load(str(model_path))
    return model, tokenizer


def mlx_batch_generate(model, tokenizer, prompts, max_new_tokens, temperature):
    """Generate responses using MLX backend (optimized for Apple Silicon)."""
    from mlx_lm import generate

    responses = []
    for prompt in prompts:
        # MLX generate is fast enough that batching is less critical
        response = generate(
            model, tokenizer, prompt=prompt,
            max_tokens=max_new_tokens, temp=temperature
        )
        # Extract only generated text (after prompt)
        if response.startswith(prompt):
            response = response[len(prompt):].strip()
        responses.append(response)

    return responses


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


@torch.no_grad()
def batch_generate_responses(model, tokenizer, prompts, max_new_tokens, temperature, device, batch_size=8):
    """Generate responses in batches for 10-100x speedup over sequential."""
    all_responses = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", truncation=True,
                          max_length=512, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only generated tokens (skip input)
        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            response = tokenizer.decode(out[input_len:], skip_special_tokens=True).strip()
            all_responses.append(response)

    return all_responses


def detect_refusal(text):
    """Check if response contains refusal patterns. Returns True if refusal detected."""
    for pattern in REFUSAL_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def check_length_valid(text, min_tokens=MIN_RESPONSE_TOKENS, max_tokens=MAX_RESPONSE_TOKENS):
    """Check if response length is within acceptable range."""
    tokens = text.split()
    length = len(tokens)
    return min_tokens <= length <= max_tokens


def check_quality_gates(response, instruction=""):
    """
    Check if response passes production quality gates.
    Returns (passed: bool, reason: str, flags: dict)
    """
    tokens = response.split()
    length = len(tokens)
    flags = {}

    # Length check
    if length < MIN_RESPONSE_TOKENS:
        return False, f"too_short ({length} < {MIN_RESPONSE_TOKENS})", {"too_short": True}
    if length > MAX_RESPONSE_TOKENS:
        return False, f"too_long ({length} > {MAX_RESPONSE_TOKENS})", {"too_long": True}

    # Refusal check
    if detect_refusal(response):
        return False, "refusal_detected", {"refusal": True}

    # Warn if below target minimum but don't reject
    if length < TARGET_MIN_TOKENS:
        flags["below_target"] = True

    return True, "passed", flags


def detect_category(instruction, response=""):
    """
    Classify instruction into category: math, code, creative, reasoning, qa, other.
    Uses keyword-based heuristics for speed.
    """
    text = (instruction + " " + response).lower()

    # Math keywords
    math_kw = ["calculate", "compute", "solve", "equation", "number", "sum", "multiply",
               "divide", "percentage", "average", "math", "arithmetic", "algebra"]
    if any(kw in text for kw in math_kw):
        return "math"

    # Code keywords
    code_kw = ["code", "program", "function", "script", "debug", "implement", "python",
               "javascript", "java", "algorithm", "api", "compile", "syntax"]
    if any(kw in text for kw in code_kw):
        return "code"

    # Creative keywords
    creative_kw = ["write a story", "poem", "creative", "imagine", "describe", "narrative",
                   "fiction", "character", "plot", "write an essay"]
    if any(kw in text for kw in creative_kw):
        return "creative"

    # Reasoning keywords
    reasoning_kw = ["why", "explain", "reason", "because", "analyze", "compare", "evaluate",
                    "logic", "argument", "conclusion", "therefore"]
    if any(kw in text for kw in reasoning_kw):
        return "reasoning"

    # QA keywords (factual questions)
    qa_kw = ["what is", "who is", "when", "where", "how many", "list", "name", "define"]
    if any(kw in text for kw in qa_kw):
        return "qa"

    return "other"


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


def compute_ngram_entropy(texts, n=3):
    """Compute entropy of n-gram distribution across all texts."""
    from collections import Counter
    ngrams = []
    for text in texts:
        tokens = text.lower().split()
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))

    if not ngrams:
        return 0.0

    counts = Counter(ngrams)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)

    return entropy


@torch.no_grad()
def compute_embedding_diversity(model, tokenizer, texts, device, batch_size=16):
    """
    Compute semantic diversity metrics using model embeddings.
    Returns: mean pairwise distance, coverage radius (approximation of spread)
    """
    embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True,
                          max_length=256, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Use last hidden state mean as embedding
        outputs = model(**inputs, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]  # Last layer
        # Mean pool across sequence length
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1)
        embeddings.append(pooled.cpu().numpy())

    embeddings = np.vstack(embeddings)

    # Compute pairwise distances (sample for efficiency if large)
    n = len(embeddings)
    if n > 100:
        # Sample pairs for efficiency
        indices = np.random.choice(n, size=min(100, n), replace=False)
        sample_emb = embeddings[indices]
    else:
        sample_emb = embeddings

    # Pairwise L2 distances
    from scipy.spatial.distance import pdist
    distances = pdist(sample_emb, metric="euclidean")
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))

    # Coverage: radius containing 95% of points from centroid
    centroid = embeddings.mean(axis=0)
    radii = np.linalg.norm(embeddings - centroid, axis=1)
    coverage_radius = float(np.percentile(radii, 95))

    return {
        "mean_pairwise_distance": round(mean_dist, 4),
        "std_pairwise_distance": round(std_dist, 4),
        "coverage_radius_95": round(coverage_radius, 4),
        "embeddings": embeddings,  # For UMAP visualization
    }


def parse_judge_score(judge_text):
    """Extract the first integer 1-10 from judge response."""
    m = re.search(r"\b([1-9]|10)\b", judge_text)
    return int(m.group(1)) if m else None


def create_umap_visualization(embeddings, categories, output_path):
    """
    Create UMAP 2D projection of embeddings colored by category.
    Saves to JSON for dashboard visualization.
    """
    try:
        from umap import UMAP
    except ImportError:
        logger.warning("UMAP not installed, skipping visualization. Install with: pip install umap-learn")
        return None

    # UMAP projection
    reducer = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embedding_2d = reducer.fit_transform(embeddings)

    # Prepare data for visualization
    viz_data = {
        "points": [
            {
                "x": float(embedding_2d[i, 0]),
                "y": float(embedding_2d[i, 1]),
                "category": categories[i],
            }
            for i in range(len(embedding_2d))
        ],
        "category_counts": {cat: categories.count(cat) for cat in set(categories)},
    }

    with open(output_path, "w") as f:
        json.dump(viz_data, f, indent=2)

    logger.info("UMAP visualization saved to %s", output_path)
    return viz_data


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


@torch.no_grad()
def batch_judge_responses(judge_model, judge_tok, instructions, responses, device, batch_size=8):
    """Judge multiple responses in batches for significant speedup."""
    all_judgments = []
    for i in range(0, len(instructions), batch_size):
        batch_inst = instructions[i:i+batch_size]
        batch_resp = responses[i:i+batch_size]

        prompts = [JUDGE_PROMPT.format(instruction=inst, response=resp)
                   for inst, resp in zip(batch_inst, batch_resp)]

        inputs = judge_tok(prompts, return_tensors="pt", truncation=True,
                          max_length=768, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = judge_model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=judge_tok.eos_token_id,
        )

        for j, out in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            judgment = judge_tok.decode(out[input_len:], skip_special_tokens=True).strip()
            all_judgments.append(judgment)

    return all_judgments


@torch.no_grad()
def compute_teacher_perplexity_on_responses(teacher_model, teacher_tok, prompts, responses, device, batch_size=4):
    """
    Compute teacher's perplexity on student-generated responses.
    Returns list of per-sample perplexities.
    """
    perplexities = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        batch_responses = responses[i:i+batch_size]

        # Combine prompt + response for teacher scoring
        full_texts = [p + r for p, r in zip(batch_prompts, batch_responses)]

        inputs = teacher_tok(full_texts, return_tensors="pt", truncation=True,
                            max_length=1024, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = teacher_model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss.item()
        ppl = math.exp(loss) if loss < 10 else float("inf")

        # Approximate per-sample (simplified - full version would need individual losses)
        for _ in range(len(batch_prompts)):
            perplexities.append(ppl)

    return perplexities


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

    # Backend selection
    use_mlx = args.backend == "mlx"
    if use_mlx:
        import platform
        if platform.processor() != "arm":
            logger.warning("MLX backend requested but not on Apple Silicon, falling back to PyTorch")
            use_mlx = False
        else:
            logger.info("Backend: MLX (Apple Silicon optimized)")
    else:
        logger.info("Backend: PyTorch, Device: %s", device)

    # Load student
    logger.info("Loading student from %s", checkpoint_dir)
    if use_mlx:
        student, tokenizer = load_mlx_model(checkpoint_dir)
    else:
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
    logger.info("Generating %d responses with batch_size=%d...", n, args.batch_size)

    # Phase 1: Batch generation
    prompts = [ex["prompt"] for ex in val_ds]
    instructions = [ex.get("instruction", ex["prompt"]) for ex in val_ds]

    if use_mlx:
        responses = mlx_batch_generate(
            student, tokenizer, prompts, args.max_new_tokens, args.temperature
        )
    else:
        responses = batch_generate_responses(
            student, tokenizer, prompts, args.max_new_tokens,
            args.temperature, device, args.batch_size
        )
    logger.info("Generated %d responses", len(responses))

    # Phase 2: Quality gate filtering
    samples = []
    rejected = {"too_short": 0, "too_long": 0, "refusal": 0, "below_target": 0}
    d1_sum = d2_sum = 0.0
    max_rep_sum = 0
    lengths = []
    categories = []

    for i, (prompt, instruction, response) in enumerate(zip(prompts, instructions, responses)):
        # Quality gates
        passed, reason, flags = check_quality_gates(response, instruction)

        # Track all rejections
        for flag_key in flags:
            if flag_key in rejected:
                rejected[flag_key] += 1

        # Diversity metrics
        d1, d2, max_rep = diversity_metrics(response)
        d1_sum += d1
        d2_sum += d2
        max_rep_sum += max_rep
        length = len(response.split())
        lengths.append(length)

        # Category detection
        category = detect_category(instruction, response)
        categories.append(category)

        sample = {
            "prompt": prompt,
            "instruction": instruction,
            "response": response,
            "distinct_1": round(d1, 4),
            "distinct_2": round(d2, 4),
            "max_rep": max_rep,
            "length_tokens": length,
            "category": category,
            "quality_gate_passed": passed,
            "quality_gate_reason": reason,
        }

        # Only include in final samples if passed quality gates
        if passed:
            samples.append(sample)

        if (i + 1) % 20 == 0:
            logger.info("  Processed %d/%d  avg_d1=%.3f  passed=%d  rejected=%d",
                       i + 1, n, d1_sum / (i + 1), len(samples), i + 1 - len(samples))

    # Quality gate summary
    n_passed = len(samples)
    n_rejected = n - n_passed
    refusal_rate = rejected["refusal"] / n * 100
    logger.info("")
    logger.info("Quality Gate Summary:")
    logger.info("  Passed: %d/%d (%.1f%%)", n_passed, n, n_passed/n*100)
    logger.info("  Rejected: %d/%d (%.1f%%)", n_rejected, n, n_rejected/n*100)
    logger.info("    - Too short (<%d tok): %d", MIN_RESPONSE_TOKENS, rejected["too_short"])
    logger.info("    - Too long (>%d tok): %d", MAX_RESPONSE_TOKENS, rejected["too_long"])
    logger.info("    - Refusals: %d (%.1f%%)", rejected["refusal"], refusal_rate)
    logger.info("    - Below target (<%d tok): %d", TARGET_MIN_TOKENS, rejected["below_target"])

    if refusal_rate > 5.0:
        logger.warning("Refusal rate %.1f%% exceeds 5%% threshold!", refusal_rate)

    # Category balance
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    category_pcts = {cat: count/len(categories)*100 for cat, count in category_counts.items()}
    logger.info("")
    logger.info("Category Distribution:")
    for cat in sorted(category_counts.keys()):
        logger.info("  %s: %d (%.1f%%)", cat, category_counts[cat], category_pcts[cat])

    # Diversity summary
    avg_d1 = d1_sum / n
    avg_d2 = d2_sum / n
    avg_max_rep = max_rep_sum / n
    median_len = sorted(lengths)[n // 2]
    ngram_entropy = compute_ngram_entropy([s["response"] for s in samples], n=3)

    logger.info("")
    logger.info("Diversity Summary:")
    logger.info("  distinct-1: %.3f", avg_d1)
    logger.info("  distinct-2: %.3f", avg_d2)
    logger.info("  avg_max_rep: %.2f", avg_max_rep)
    logger.info("  3-gram entropy: %.2f bits", ngram_entropy)
    logger.info("  median_length: %d tokens", median_len)

    if avg_d1 < 0.5:
        logger.warning("Low distinct-1 (%.3f) — possible mode collapse", avg_d1)
    if avg_max_rep > 3:
        logger.warning("High avg max repetition (%.1f) — check for repetition loops", avg_max_rep)

    # Phase 3: Embedding diversity (compute before loading teacher)
    embedding_result = {"enabled": False}
    if not use_mlx:
        logger.info("")
        logger.info("Computing embedding diversity...")
        try:
            emb_div = compute_embedding_diversity(
                student, tokenizer,
                [s["response"] for s in samples],
                device, batch_size=args.batch_size
            )
            embedding_result = {
                "enabled": True,
                "mean_pairwise_distance": emb_div["mean_pairwise_distance"],
                "std_pairwise_distance": emb_div["std_pairwise_distance"],
                "coverage_radius_95": emb_div["coverage_radius_95"],
            }
            logger.info("  Mean pairwise distance: %.4f", emb_div["mean_pairwise_distance"])
            logger.info("  Coverage radius (95%%): %.4f", emb_div["coverage_radius_95"])

            # UMAP visualization
            umap_path = output_dir / "embedding_viz.json"
            create_umap_visualization(
                emb_div["embeddings"],
                [s["category"] for s in samples],
                umap_path
            )
        except Exception as e:
            logger.warning("Failed to compute embedding diversity: %s", e)
    else:
        logger.info("Skipping embedding diversity (not supported with MLX backend)")

    # Phase 4: Teacher evaluation (judge + perplexity)
    del student  # Free memory before loading teacher
    if device.type == "mps":
        torch.mps.empty_cache()

    judge_result = {"enabled": False}
    teacher_ppl_result = {"enabled": False}

    if args.judge or args.judge_teacher_ppl:
        logger.info("")
        logger.info("Loading teacher: %s", args.teacher)

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

        # Teacher perplexity on student outputs
        if args.judge_teacher_ppl:
            logger.info("Computing teacher perplexity on student generations...")
            sample_prompts = [s["prompt"] for s in samples]
            sample_responses = [s["response"] for s in samples]

            teacher_ppls = compute_teacher_perplexity_on_responses(
                judge_model, judge_tok, sample_prompts, sample_responses,
                device, batch_size=4
            )

            avg_teacher_ppl = sum(teacher_ppls) / len(teacher_ppls) if teacher_ppls else None
            for i, ppl in enumerate(teacher_ppls):
                samples[i]["teacher_ppl"] = round(ppl, 2)

            teacher_ppl_result = {
                "enabled": True,
                "avg_teacher_ppl": round(avg_teacher_ppl, 2) if avg_teacher_ppl else None,
            }
            logger.info("  Avg teacher perplexity: %.2f", avg_teacher_ppl or 0)

            if avg_teacher_ppl and avg_teacher_ppl > 100:
                logger.warning("High teacher perplexity (%.2f) on student outputs — check quality", avg_teacher_ppl)

        # LLM-as-judge scoring
        if args.judge:
            logger.info("Batch judging %d responses (batch_size=%d)...", len(samples), args.batch_size)

            sample_instructions = [s["instruction"] for s in samples]
            sample_responses = [s["response"] for s in samples]

            judgments = batch_judge_responses(
                judge_model, judge_tok, sample_instructions, sample_responses,
                device, args.batch_size
            )

            scores = []
            for i, (s, raw) in enumerate(zip(samples, judgments)):
                score = parse_judge_score(raw)
                s["judge_raw"] = raw
                s["judge_score"] = score
                if score is not None:
                    scores.append(score)

                if (i + 1) % 20 == 0:
                    avg_so_far = sum(scores) / len(scores) if scores else float("nan")
                    logger.info("  Judged %d/%d  avg_score=%.2f", i + 1, len(samples), avg_so_far)

            valid_scores = [sc for sc in scores if sc is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else None
            logger.info("Judge avg score: %.2f / 10 (%d/%d parseable)", avg_score or 0, len(valid_scores), len(samples))

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
        "n_samples_generated": n,
        "n_samples_passed": len(samples),
        "quality_gates": {
            "passed": n_passed,
            "rejected": n_rejected,
            "pass_rate_pct": round(n_passed/n*100, 1),
            "refusal_rate_pct": round(refusal_rate, 2),
            "rejection_reasons": rejected,
        },
        "category_distribution": {
            "counts": category_counts,
            "percentages": {cat: round(pct, 1) for cat, pct in category_pcts.items()},
        },
        "diversity": {
            "avg_distinct_1": round(avg_d1, 4),
            "avg_distinct_2": round(avg_d2, 4),
            "avg_max_rep": round(avg_max_rep, 2),
            "ngram_entropy_3": round(ngram_entropy, 2),
            "median_response_tokens": median_len,
        },
        "embedding_diversity": embedding_result,
        "teacher_perplexity": teacher_ppl_result,
        "judge": judge_result,
        "samples": samples,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Quality metrics saved to %s", out_path)


if __name__ == "__main__":
    main()
