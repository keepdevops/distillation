"""Standalone helpers: build experiment log entries and collect metrics from output dirs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def make_run_id(output_dir: str) -> str:
    """Generate a run_id from output_dir name + timestamp."""
    name = Path(output_dir).name
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{name}-{ts}"


def build_entry(
    run_id: str,
    output_dir: str,
    config: dict,
    metrics: dict,
    outcome: str,
    steps_completed: list[str],
    start_time: float,
    hardware: str = "mps",
) -> dict:
    """Build a complete experiment log entry."""
    import time
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "output_dir": str(output_dir),
        "config": config,
        "metrics": metrics,
        "outcome": outcome,
        "steps_completed": steps_completed,
        "duration_seconds": round(time.time() - start_time),
        "hardware": hardware,
    }


def collect_metrics(output_dir: str) -> dict:
    """
    Read the latest metrics from output_dir:
    - Last eval_perplexity, teacher_eval_perplexity, ppl_gap_pct,
      quant_ppl_gap_pct from metrics.jsonl
    - judge_avg_score from quality_metrics.json
    - wikitext2_perplexity from benchmark_results.json
    """
    d = Path(output_dir)
    metrics: dict = {}

    mf = d / "metrics.jsonl"
    if mf.exists():
        with open(mf) as f:
            for line in f:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in ("eval_loss", "eval_perplexity", "teacher_eval_perplexity",
                            "ppl_gap_pct", "quant_ppl_gap_pct"):
                    if key in row:
                        metrics[key] = row[key]

    qf = d / "quality_metrics.json"
    if qf.exists():
        try:
            with open(qf) as f:
                q = json.load(f)

            judge = q.get("judge", {})
            if judge.get("enabled") and judge.get("avg_score") is not None:
                metrics["judge_avg_score"] = judge["avg_score"]

            div = q.get("diversity", {})
            for k in ("avg_distinct_1", "avg_distinct_2", "avg_max_rep", "ngram_entropy_3"):
                if k in div:
                    metrics[k] = div[k]

            gates = q.get("quality_gates", {})
            for k in ("pass_rate_pct", "refusal_rate_pct"):
                if k in gates:
                    metrics[k] = gates[k]

            teacher_ppl = q.get("teacher_perplexity", {})
            if teacher_ppl.get("enabled") and teacher_ppl.get("avg_teacher_ppl") is not None:
                metrics["avg_teacher_ppl"] = teacher_ppl["avg_teacher_ppl"]

            emb_div = q.get("embedding_diversity", {})
            if emb_div.get("enabled"):
                for k in ("mean_pairwise_distance", "coverage_radius_95"):
                    if k in emb_div:
                        metrics[k] = emb_div[k]

            cat_dist = q.get("category_distribution", {})
            if "percentages" in cat_dist:
                metrics["category_distribution"] = cat_dist["percentages"]
        except Exception:
            pass

    bf = d / "benchmark_results.json"
    if bf.exists():
        try:
            with open(bf) as f:
                b = json.load(f)
            if "wikitext2_perplexity" in b:
                metrics["wikitext2_perplexity"] = b["wikitext2_perplexity"]
        except Exception:
            pass

    return metrics
