"""Help accordion: Data Prep & Domain Synthesis section."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Data Prep & Domain Synthesis — When and How to Use Each Tool", open=False):
        gr.Markdown("""
### When to use Data Prep
Out-of-the-box datasets (Alpaca, Guanaco) work fine for quickstart runs. Use Data Prep when:
- You want domain-specific data not in the standard datasets
- Existing dataset quality is low (short responses, repetition, refusals)
- You want to maximize distillation quality with teacher-generated data

### Domain Synthesis (separate tab)
For domain-specialist models (coding, math, medical, legal, finance, tax) use the
**Domain Synthesis** tab instead of — or in addition to — the tools below. It runs
Magpie synthesis with curated system prompts and domain-specific quality filters.
Output: `domain_data/<domain>/hf_dataset/` — point **Dataset** here in Configure & Launch.

### Magpie Synthesis
**What it does:** Loads the teacher model and repeatedly samples from it, conditioning on just the chat-template user-turn prefix. The model "auto-completes" realistic user instructions, then generates responses. No seed dataset needed.

**Output:** `output_dir/hf_dataset/` — an HF dataset you can point directly at the **Dataset** field in Configure & Launch.

**When to use:** Best quality synthetic data. Requires the teacher to be available locally (large download first time). Use when you want data tailored to the teacher's style.

| Parameter | Guidance |
|-----------|----------|
| **Pairs to generate** | Generate 2–3× your target keep count to account for filtering attrition. |
| **Filter output** | Always keep this on — removes duplicates and low-quality pairs. |
| **Target keep** | Final dataset size after filtering. 2000–5000 pairs is enough for a 2-epoch run. |

### Self-Instruct Synthesis
**What it does:** Shows the teacher `seed_examples` instructions from a base dataset, then asks it to generate a new, diverse instruction. The teacher also generates the response. Filtered by perplexity bounds and distinct-2 score.

**Output:** `output_dir/synthetic_data/` — an HF dataset directory.

**When to use:** More diverse than Magpie (prompts are anchored to varied seed examples). Slower per-pair than Magpie. Good for covering topic diversity.

| Parameter | Guidance |
|-----------|----------|
| **Temperature** | 0.9 is a good balance — high enough for diversity, low enough for coherence. Don't go above 1.2. |
| **Seed examples** | 5 is standard. Higher seeds = more context for the teacher but diminishing returns. |

### Dataset Filter
**What it does:** Takes any alpaca-format dataset (HF hub ID or local path) and removes:
- Responses shorter than `min_response_words`
- Low-coherence responses (distinct-2 bigram diversity < `min_distinct2`)
- Near-duplicate pairs (Jaccard similarity above threshold)
- Common refusal patterns ("I cannot...", "As an AI...")

**Output:** `output_dir/` — an HF dataset directory. Use this path directly in Configure & Launch.

**When to use:** Always run this before training on any public dataset. `yahma/alpaca-cleaned` is already filtered but running it again with stricter thresholds is harmless.

| Parameter | Guidance |
|-----------|----------|
| **Target top-N** | How many pairs to keep after scoring. Keeps the highest-quality pairs. |
| **Min distinct-2** | 0.35 is standard. Increase to 0.45 to keep only highly diverse responses. |
| **Min response words** | 20 words eliminates one-liners. Increase to 30–40 for richer training signal. |

### Combining tools (recommended data workflow)
```
General quality:
  1. Filter        yahma/alpaca-cleaned  →  filtered_data/              (2–3 min)
  2. Self-Instruct Generate 3000 pairs   →  synthetic_data/             (30–60 min)
  3. Configure     Dataset = filtered_data/

Domain specialist:
  1. Domain Synth  coding / math / medical …  →  domain_data/<x>/      (30–90 min)
  2. (Optional) mix with filtered general data using datasets.concatenate_datasets()
  3. Configure     Dataset = domain_data/<x>/hf_dataset/
```
""")
