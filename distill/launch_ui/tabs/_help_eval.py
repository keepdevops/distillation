"""Help accordion: Eval — Interpreting Results section."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Eval — Interpreting Results", open=False):
        gr.Markdown("""
### Shared inputs
- **Model output dir** — The same directory you used as Output directory during training. This is where metrics.jsonl and result files are written.
- **Checkpoint** — Leave blank to eval the final merged model. Enter a path like `distilled-minillm/checkpoint-80` to eval a mid-training checkpoint.
- **Base model** — Only used as a fallback if the checkpoint directory has no tokenizer. Set to the student model you trained.

---

### Perplexity Eval (`run_eval.py`)
Loads the student model and computes **cross-entropy loss** on a held-out validation split of your training dataset.

**Output appended to:** `metrics.jsonl` as `{"step": N, "eval_loss": X, "perplexity": Y}`

**How to interpret:**
| Perplexity | Interpretation |
|------------|----------------|
| < 5 | Excellent — model fits the data distribution well |
| 5–15 | Good — typical range for a well-distilled small model |
| 15–30 | Fair — may need more epochs or a better dataset |
| > 30 | Poor — check for training bugs, wrong checkpoint, or data mismatch |

- **Compare teacher** — Also runs the teacher through the same eval and logs the gap. A good distilled student should get within 1.5–2× the teacher's perplexity.
- Run this after every training run as a quick sanity check.

---

### Quality Eval (`eval_quality.py`)
Generates responses from the student on `n_samples` prompts and measures **generation diversity**.

**Output:** `quality_metrics.json` in the model output dir.

**Metrics explained:**
| Metric | Good range | What it means |
|--------|-----------|----------------|
| `distinct_1` | > 0.15 | Fraction of unique unigrams. Low = repetitive vocabulary. |
| `distinct_2` | > 0.40 | Fraction of unique bigrams. Low = repetitive phrasing. |
| `max_repetition` | < 0.30 | Highest n-gram repetition rate in any single response. High = mode collapse. |
| `avg_length_tokens` | 50–200 | Average response length. Very short may indicate mode collapse. |
| `judge_score_mean` | 6–8 | LLM-as-judge score (1–10). Enable **LLM-as-judge** for this. |

- **LLM-as-judge** — Uses the teacher model to score each response 1–10. Adds ~5–10 minutes. Worth running once before deploying.
- Run this after perplexity eval confirms a reasonable loss.

---

### WikiText-2 Benchmark (`run_benchmarks.py`)
Evaluates the student on the **WikiText-2-raw-v1** test split — a standard open-domain NLP benchmark independent of your training data.

**Output:** `benchmark_results.json` + `metrics.jsonl` entry `{"wikitext2_perplexity": X}`

**Reference numbers (lower is better):**
| Model | WikiText-2 PPL |
|-------|---------------|
| Qwen2-0.5B-Instruct (base, no distillation) | ~18–22 |
| Well-distilled student (from 1.5B teacher) | ~14–18 |
| Qwen2-1.5B-Instruct (teacher) | ~10–13 |

- **Baseline dir** — Point at a previous run's output dir to get a regression comparison. If the new model is more than `threshold`% worse than the baseline, the benchmark prints a warning.
- Run this as a final check before exporting to GGUF.

---

### Recommended eval sequence
```
1. Perplexity Eval   → quick sanity check, ~2 min
2. Quality Eval      → verify generation diversity, ~5 min
3. WikiText-2        → standardized benchmark for comparison, ~5 min
4. Quality Eval with judge enabled  → final quality gate, ~15 min
```
""")
