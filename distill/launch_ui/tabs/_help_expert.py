"""Help accordion: Expert Pipeline — Domain-Specific Distillation with CoT."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Expert Pipeline — Domain-Specific Distillation with CoT", open=False):
        gr.Markdown("""
The **Expert Pipeline** tab implements a 4-step workflow for building domain-specialist models
(tax, legal, medical, finance, coding) using any HuggingFace dataset and a GGUF teacher model.

---

### Step 1 — Dataset & Column Mapping
Load any HF dataset and remap its columns to the standard `instruction / input / output` format.

1. Enter the HF dataset ID (e.g. `nelson-liu/legalbench`) or a local path
2. Click **Inspect** — columns are auto-detected and dropdowns populated
3. Adjust column assignments if needed
4. Click **Remap & Save Dataset** → saves to the path in *Save remapped dataset to*

The remapped dataset is saved as an HF dataset on disk **and** as a `remapped.jsonl` for inspection.

**Supported datasets (tested):**

| Dataset | Domain | instruction_col | output_col |
|---------|--------|-----------------|------------|
| `Atome-LLM/Tax-Policy-Analysis` | Tax | `question` | `answer` |
| `nelson-liu/legalbench` | Legal | `instruction` | `output` |
| `medalpaca/medical_meadow_medical_flashcards` | Medical | `input` | `output` |
| `yahma/alpaca-cleaned` | General | `instruction` | `output` |
| `tatsu-lab/alpaca` | General | `instruction` | `output` |

---

### Step 2 — GGUF Teacher
Select a Metal-accelerated GGUF model from `/Users/Shared/llama/models/` to act as the teacher
for CoT generation. The model is served via `llama-server` during generation.

| Setting | Guidance |
|---------|----------|
| **Context size** | 8192 is safe for most models. Reduce to 4096 if the server OOMs. |
| **Parallel slots** | 4 keeps the GPU saturated. Reduce to 2 if getting timeouts. |
| **Temperature** | 0.3 for precise domains (tax, legal, medical). 0.7 for creative/coding. |
| **Max tokens** | 1024 covers most CoT responses. Increase to 2048 for complex reasoning. |

**Recommended GGUF teachers by domain:**

| Domain | Model | Quant | Why |
|--------|-------|-------|-----|
| Tax / Legal (best) | `Meta-Llama-3-70B-Instruct-Q4_K_M.gguf` | Q4_K_M | Strongest reasoning at 70B; fits 36 GB unified |
| Legal (fast) | `law-chat.Q4_K_M.gguf` | Q4_K_M | Fine-tuned on statutory interpretation |
| Legal (alt) | `Llama-3-8B-Instruct-Legal-Q8_0.gguf` | Q8_0 | Highest-accuracy 8B legal model |
| Medical | `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf` | Q4_K_M | Best medical reasoning at 8B |
| Coding | `granite-3.1-8b-instruct-Q4_K_M.gguf` | Q4_K_M | IBM Granite excels at code |
| General | `Llama-3.2-3B-Instruct-Q4_K_M.gguf` | Q4_K_M | Fast, good general quality |

> **Quantization guide:** Q4_K_M = best balance of size and quality. Q8_0 = highest accuracy, ~2× larger.
> All GGUF files go to `/Users/Shared/llama/models/` — click **Refresh** in the teacher dropdown to pick them up.

**Downloading GGUF models** — save all files to `/Users/Shared/llama/models/`:
```bash
cd /Users/Shared/llama/models

# 70B Llama-3 teacher (bartowski) — best for tax/legal CoT, needs ~42 GB RAM
curl -L -o Meta-Llama-3-70B-Instruct-Q4_K_M.gguf \\
  "https://huggingface.co/bartowski/Meta-Llama-3-70B-Instruct-GGUF/resolve/main/Meta-Llama-3-70B-Instruct-Q4_K_M.gguf"

# Law-Chat 7B — statutory interpretation specialist
curl -L -o law-chat.Q4_K_M.gguf \\
  "https://huggingface.co/ricdomoliver/law-chat-GGUF/resolve/main/law-chat.Q4_K_M.gguf"

# Llama-3-8B Legal fine-tune (bartowski) — faster alternative
curl -L -o Llama-3-8B-Instruct-Legal-Q8_0.gguf \\
  "https://huggingface.co/bartowski/Llama-3-8B-Instruct-Legal-GGUF/resolve/main/Llama-3-8B-Instruct-Legal-Q8_0.gguf"
```
> **Note:** Verify the exact repo paths at huggingface.co before downloading — quantizer repos occasionally rename files.

**Downloading datasets:**
```bash
# LegalBench (Nguha, ~100 legal reasoning tasks) — use dataset ID in Step 1:
#   nguha/legalbench

# Tax-Policy-Analysis instruction data — use dataset ID in Step 1:
#   Atome-LLM/Tax-Policy-Analysis

# Download LegalBench as zip if you need an offline copy:
curl -L -o legalbench.zip \\
  "https://huggingface.co/datasets/nguha/legalbench/resolve/main/legalbench.zip"

# Tax parquet (if offline):
curl -L -o tax_data.parquet \\
  "https://huggingface.co/datasets/Atome-LLM/Tax-Policy-Analysis/resolve/main/data/train-00000-of-00001.parquet"
```
> **Tip:** For online runs just paste the HF dataset ID directly into the *Dataset ID* field — no download needed.

---

### Step 3 — CoT Rationale Generation
Prompts the GGUF teacher to generate **Chain-of-Thought reasoning traces** for each sample.

Each output is formatted as:
```
<reasoning>
[step-by-step domain analysis]
</reasoning>
<answer>
[final answer]
</answer>
```

The domain selector auto-fills a domain-specific system prompt (tax cites IRC sections,
legal applies statutory tests, medical follows clinical guidelines, etc.). You can override it.

**Output:** `cot_output_dir/cot_data.jsonl` + `cot_output_dir/hf_dataset/`

**Logs saved automatically to:**
- `runs/ep_cot_<domain>_<timestamp>.log` — plain text
- `runs/ep_cot_<domain>_<timestamp>.jsonl` — structured JSON (step, loss, ts, msg)

Both the **Training Loss** and **Gradient Norm** charts update live during this step,
and the full output streams in the **Live output** box below the accordions.

---

### Step 4 — Distillation
Launches `run_distillation_agent.py` on the CoT dataset. Leave *Dataset path* blank to
automatically use the `hf_dataset/` produced by Step 3.

**Key settings for domain expert models:**

| Setting | Recommended | Why |
|---------|-------------|-----|
| **LoRA rank** | 32 | Higher than default (16) — domain terminology requires more capacity |
| **Epochs** | 3 | CoT data is dense; 3 epochs ensures thorough absorption |
| **Backend** | MLX | Fastest on M3 Max; CoT outputs are long so MLX memory efficiency matters |

Distillation logs also save to `runs/ep_distill_<timestamp>.log/.jsonl`.

---

### Full expert pipeline (CLI)
```bash
# 1. Remap
python -m distill.expert_pipeline --mode remap \\
    --dataset Atome-LLM/Tax-Policy-Analysis \\
    --instruction_col question --output_col answer \\
    --output_dir ./domain_data/tax_expert --max_samples 5000

# 2. Generate CoT (phi-4 teacher, tax domain)
python -m distill.expert_pipeline --mode cot \\
    --dataset ./domain_data/tax_expert \\
    --teacher /Users/Shared/llama/models/phi-4-Q5_K_M.gguf \\
    --domain tax --n_samples 2000 \\
    --output_dir ./domain_data/tax_cot

# 3. Distill
python -m distill.expert_pipeline --mode distill \\
    --dataset ./domain_data/tax_cot/hf_dataset \\
    --output_dir ./runs/tax-expert \\
    --backend mlx --open --lora_r 32 --epochs 3
```
""")
