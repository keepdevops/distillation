#!/usr/bin/env python3
"""
Structured experiment memory: read/write a JSONL log of all distillation runs.

Used by run_distillation_agent.py to:
  - Print relevant history before proposing next trial
  - Log full config + metrics after each completed run

Usage (standalone CLI):
    python scripts/experiment_log.py --show 10
    python scripts/experiment_log.py --show 5 --log ./experiment_log.jsonl
"""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("experiment_log.jsonl")

# Hyperparameter search space for propose_next()
_SEARCH_SPACE = {
    "temperature": [0.5, 0.7, 1.0, 1.3, 1.5, 2.0],
    "lora_r": [16, 32, 64, 128],
    "epochs": [1, 2, 3, 4],
}


class ExperimentLog:
    """Read/write a JSONL file of distillation run records."""

    def __init__(self, log_path: str | Path = DEFAULT_LOG):
        self.log_path = Path(log_path)

    def append(self, entry: dict) -> None:
        """Append one run record to the log."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def load_all(self) -> list[dict]:
        """Return all records, oldest first."""
        if not self.log_path.exists():
            return []
        records = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def recent(self, n: int = 10) -> list[dict]:
        """Return the N most recent records."""
        return self.load_all()[-n:]

    def find_similar(self, teacher: str, student: str, dataset: str) -> list[dict]:
        """Return records sharing teacher, student, and dataset."""
        results = []
        for rec in self.load_all():
            cfg = rec.get("config", {})
            if (cfg.get("teacher", "") == teacher and
                    cfg.get("student", "") == student and
                    cfg.get("dataset", "") == dataset):
                results.append(rec)
        return results

    def summarize(self, n: int = 10) -> str:
        """Return a formatted table of the N most recent runs."""
        records = self.recent(n)
        if not records:
            return "(no experiment history found)"

        header = (
            f"{'run_id':<32}  {'date':<10}  {'backend':<9}  "
            f"{'ep':>3}  {'ppl':>7}  {'gap%':>6}  "
            f"{'judge':>5}  {'pass%':>5}  {'ref%':>5}  {'wt2':>7}  {'outcome':<8}"
        )
        sep = "-" * len(header)
        lines = [header, sep]

        for r in records:
            cfg = r.get("config", {})
            m = r.get("metrics", {})
            ts = r.get("timestamp", "")[:10]
            run_id = r.get("run_id", "?")[:31]
            backend = cfg.get("backend", "?")[:9]
            epochs = cfg.get("epochs", "?")
            ppl = m.get("eval_perplexity", "")
            gap = m.get("ppl_gap_pct", "")
            judge = m.get("judge_avg_score", "")
            pass_rate = m.get("pass_rate_pct", "")
            refusal_rate = m.get("refusal_rate_pct", "")
            wt2 = m.get("wikitext2_perplexity", "")
            outcome = r.get("outcome", "?")[:8]

            def _fmt(v, fmt):
                try:
                    return format(float(v), fmt)
                except (TypeError, ValueError):
                    import re as _re
                    m = _re.match(r"(\d+)", fmt)
                    w = int(m.group(1)) if m else 4
                    return format("--", f">{w}")

            lines.append(
                f"{run_id:<32}  {ts:<10}  {backend:<9}  "
                f"{str(epochs):>3}  {_fmt(ppl, '7.2f')}  {_fmt(gap, '6.1f')}  "
                f"{_fmt(judge, '5.1f')}  {_fmt(pass_rate, '5.1f')}  {_fmt(refusal_rate, '5.2f')}  "
                f"{_fmt(wt2, '7.2f')}  {outcome:<8}"
            )
        return "\n".join(lines)

    def diagnose(self, metrics: dict) -> list[str]:
        """Return [OK/WARN/ERROR] diagnostic strings for a set of metrics."""
        msgs = []

        ppl_gap = metrics.get("ppl_gap_pct")
        d1 = metrics.get("avg_distinct_1")
        judge = metrics.get("judge_avg_score")
        quant_gap = metrics.get("quant_ppl_gap_pct")
        wt2 = metrics.get("wikitext2_perplexity")
        ppl = metrics.get("eval_perplexity")
        refusal_rate = metrics.get("refusal_rate_pct")
        pass_rate = metrics.get("pass_rate_pct")
        teacher_ppl = metrics.get("avg_teacher_ppl")
        ngram_entropy = metrics.get("ngram_entropy_3")

        # Perplexity gap
        if ppl_gap is not None:
            if ppl_gap > 50:
                msgs.append(
                    f"[ERROR] Large perplexity gap ({ppl_gap:.1f}%). "
                    "Try: --temperature 1.5 --lora_r 128 --epochs 3"
                )
            elif ppl_gap > 30:
                msgs.append(
                    f"[WARN]  Moderate perplexity gap ({ppl_gap:.1f}%). "
                    "Try: +1 epoch or --curriculum"
                )
            else:
                msgs.append(f"[OK]    Perplexity gap {ppl_gap:.1f}% (acceptable)")

        # Diversity metrics
        if d1 is not None:
            if d1 < 0.4:
                msgs.append(
                    f"[ERROR] Mode collapse: distinct-1={d1:.2f}. "
                    "Try: raise --temperature, add --synthetic_data"
                )
            elif d1 < 0.55:
                msgs.append(f"[WARN]  Low diversity: distinct-1={d1:.2f}")
            else:
                msgs.append(f"[OK]    Diversity distinct-1={d1:.2f}")

        if ngram_entropy is not None:
            if ngram_entropy < 6.0:
                msgs.append(
                    f"[WARN]  Low 3-gram entropy ({ngram_entropy:.1f} bits). "
                    "Try: increase --temperature or dataset diversity"
                )

        # Judge scoring
        if judge is not None:
            if judge < 4:
                msgs.append(
                    f"[ERROR] Poor instruction following: judge={judge:.1f}/10. "
                    "Try: --curriculum, larger dataset"
                )
            elif judge < 6:
                msgs.append(f"[WARN]  Mediocre judge score: {judge:.1f}/10")
            else:
                msgs.append(f"[OK]    Judge score {judge:.1f}/10")

        # Quality gates
        if refusal_rate is not None:
            if refusal_rate > 5.0:
                msgs.append(
                    f"[ERROR] High refusal rate ({refusal_rate:.1f}% > 5%). "
                    "Try: --curriculum or filter refusals from training data"
                )
            elif refusal_rate > 2.0:
                msgs.append(f"[WARN]  Elevated refusal rate: {refusal_rate:.1f}%")
            else:
                msgs.append(f"[OK]    Refusal rate {refusal_rate:.1f}% (acceptable)")

        if pass_rate is not None and pass_rate < 80:
            msgs.append(
                f"[WARN]  Low quality gate pass rate ({pass_rate:.1f}%). "
                "Many samples rejected for length/quality issues"
            )

        # Teacher perplexity on student outputs
        if teacher_ppl is not None:
            if teacher_ppl > 100:
                msgs.append(
                    f"[ERROR] Very high teacher perplexity on student outputs ({teacher_ppl:.1f}). "
                    "Student generating low-quality text"
                )
            elif teacher_ppl > 50:
                msgs.append(
                    f"[WARN]  High teacher perplexity on student outputs ({teacher_ppl:.1f})"
                )

        # Quantization
        if quant_gap is not None and quant_gap > 15:
            msgs.append(
                f"[WARN]  Quantization collapse: quant_ppl_gap={quant_gap:.1f}%. "
                "Try: smaller --lora_r"
            )

        # Catastrophic forgetting
        if wt2 is not None and ppl is not None and ppl > 0:
            ratio = wt2 / ppl
            if ratio > 2.5:
                msgs.append(
                    f"[WARN]  Possible catastrophic forgetting: "
                    f"wikitext2_ppl/eval_ppl={ratio:.1f}x. Try: lower --learning_rate"
                )

        # Volume guidance (LIMA/Orca findings)
        msgs.append("")
        msgs.append("[INFO]  Volume Guidance (LIMA/Orca findings):")
        if pass_rate and pass_rate >= 80:
            msgs.append("  • High quality samples detected (pass rate ≥80%)")
            msgs.append("  • Target: 10k-100k high-quality samples > 1M noisy samples")
            msgs.append("  • Consider expanding to 50k-100k samples at current quality")
        else:
            msgs.append("  • Low pass rate suggests noisy data")
            msgs.append("  • Focus on quality over quantity: filter aggressively")
            msgs.append("  • 10k clean samples > 100k noisy samples")

        if not msgs:
            msgs.append("[OK]    No issues detected in metrics")

        return msgs

    def propose_next(self, base_config: dict) -> dict:
        """Propose the next hyperparameter config based on run history.

        With < 3 successful runs: randomly perturb base_config within each
        parameter's search space.  With >= 3 runs: hill-climb — compare the
        best third vs. worst third of runs by eval_perplexity and move each
        hyperparameter value toward the range that produced lower perplexity.
        """
        import random

        valid = [
            r for r in self.load_all()
            if r.get("metrics", {}).get("eval_perplexity") is not None
        ]

        config = dict(base_config)

        if len(valid) < 3:
            # Not enough history — random exploration
            for key, space in _SEARCH_SPACE.items():
                if key in config:
                    config[key] = random.choice(space)
            return config

        # Sort ascending by perplexity (lower = better)
        valid.sort(key=lambda r: r["metrics"]["eval_perplexity"])
        k = max(1, len(valid) // 3)
        best_runs = valid[:k]
        worst_runs = valid[-k:]

        for key, space in _SEARCH_SPACE.items():
            if key not in config:
                continue
            best_vals = [r.get("config", {}).get(key) for r in best_runs]
            worst_vals = [r.get("config", {}).get(key) for r in worst_runs]
            best_vals = [v for v in best_vals if v is not None]
            worst_vals = [v for v in worst_vals if v is not None]
            if not best_vals:
                continue

            best_mean = sum(best_vals) / len(best_vals)
            worst_mean = sum(worst_vals) / len(worst_vals) if worst_vals else best_mean
            span = abs(best_mean - worst_mean)
            # Perturb toward best_mean with 20% random noise
            target = best_mean + random.uniform(-0.2, 0.2) * max(span, 0.1)
            config[key] = min(space, key=lambda x: abs(x - target))

        return config


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

    # metrics.jsonl — take last row that has each key
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

    # quality_metrics.json
    qf = d / "quality_metrics.json"
    if qf.exists():
        try:
            with open(qf) as f:
                q = json.load(f)

            # Judge
            judge = q.get("judge", {})
            if judge.get("enabled") and judge.get("avg_score") is not None:
                metrics["judge_avg_score"] = judge["avg_score"]

            # Diversity
            div = q.get("diversity", {})
            for k in ("avg_distinct_1", "avg_distinct_2", "avg_max_rep", "ngram_entropy_3"):
                if k in div:
                    metrics[k] = div[k]

            # Quality gates
            gates = q.get("quality_gates", {})
            for k in ("pass_rate_pct", "refusal_rate_pct"):
                if k in gates:
                    metrics[k] = gates[k]

            # Teacher perplexity on student outputs
            teacher_ppl = q.get("teacher_perplexity", {})
            if teacher_ppl.get("enabled") and teacher_ppl.get("avg_teacher_ppl") is not None:
                metrics["avg_teacher_ppl"] = teacher_ppl["avg_teacher_ppl"]

            # Embedding diversity
            emb_div = q.get("embedding_diversity", {})
            if emb_div.get("enabled"):
                for k in ("mean_pairwise_distance", "coverage_radius_95"):
                    if k in emb_div:
                        metrics[k] = emb_div[k]

            # Category balance
            cat_dist = q.get("category_distribution", {})
            if "percentages" in cat_dist:
                metrics["category_distribution"] = cat_dist["percentages"]

        except Exception:
            pass

    # benchmark_results.json
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


def parse_args():
    p = argparse.ArgumentParser(description="View experiment history")
    p.add_argument("--show", type=int, default=10, help="Show N most recent runs")
    p.add_argument("--log", type=str, default=str(DEFAULT_LOG),
                   help="Path to experiment_log.jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    log = ExperimentLog(args.log)
    print(log.summarize(args.show))


if __name__ == "__main__":
    main()
