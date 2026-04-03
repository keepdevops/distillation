# Quality Gates Quick Start

## Installation

```bash
# Required (scipy already installed ✓)
# No additional packages needed for basic functionality

# Optional: UMAP visualization
pip install umap-learn

# Optional: MLX backend (Apple Silicon only)
pip install mlx-lm
```

---

## Quick Commands

### Minimal (Fast Check)

```bash
# 10 samples, no judge, ~5 seconds
python scripts/eval_quality.py ./distilled-minillm \
    --n_samples 10 \
    --offline
```

### Standard (Production)

```bash
# 50 samples + judge, ~60 seconds
python scripts/eval_quality.py ./distilled-minillm \
    --judge \
    --n_samples 50 \
    --offline
```

### Full (All Features)

```bash
# 100 samples + all metrics, ~2 minutes
python scripts/eval_quality.py ./distilled-minillm \
    --judge \
    --judge-teacher-ppl \
    --n_samples 100 \
    --batch_size 8 \
    --backend mlx \
    --offline
```

---

## What to Look For

### ✅ Good Model

```
Pass rate: ≥80%
Refusal rate: <5%
distinct-1: ≥0.65
Judge score: ≥7/10
Teacher ppl: <50
```

### ⚠️ Needs Work

```
Pass rate: 60-80%
Refusal rate: 5-10%
distinct-1: 0.5-0.65
Judge score: 5-7/10
Teacher ppl: 50-100
```

### ❌ Problems

```
Pass rate: <60%
Refusal rate: >10%
distinct-1: <0.5
Judge score: <5/10
Teacher ppl: >100
```

---

## Common Fixes

| Problem | Fix |
|---------|-----|
| High refusal rate | `--curriculum` or filter training data |
| Low pass rate | Increase `--max_new_tokens 1024` |
| Low diversity | Raise `--temperature 1.0` |
| High teacher ppl | Add `--curriculum` or more epochs |

---

## Output Files

```
./distilled-minillm/
├── quality_metrics.json       # Main output with all metrics
└── embedding_viz.json         # UMAP visualization data (optional)
```

---

## Integration with Agent

No changes needed! The agent already calls eval_quality.py.

To enable judge scoring:
```bash
python scripts/run_distillation_agent.py \
    --open --offline --epochs 2 \
    --export gguf \
    --log_experiment \
    --judge                      # ← Add this flag
```

---

## Performance

| Configuration | Time (50 samples) |
|---------------|-------------------|
| Old (sequential) | 250s (~4 min) |
| New (batched) | 40s (~40 sec) |
| Speedup | **6-10x faster** |

---

## Support

See full documentation:
- `PRODUCTION_QUALITY_GATES.md` - Complete feature guide
- `CHANGELOG_QUALITY_GATES.md` - Technical changes
