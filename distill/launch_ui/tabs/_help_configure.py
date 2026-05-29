"""Help accordion: Configure & Launch parameter reference."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Configure & Launch — Parameter Reference", open=False):
        gr.Markdown("""
### Stage
| Stage | Script | When to use |
|-------|--------|-------------|
| **SFT** | `distill_sft.py` | First-pass warmup. Teacher generates response labels; student trains on them with standard cross-entropy. Use before MiniLLM for best results. |
| **MiniLLM** | `distill_minillm.py` / `distill_mlx.py` | Main distillation stage. Student generates completions; GRPO advantage signal pushes it toward teacher distribution (reverse-KL). |

### Backend
| Backend | Speed | Memory | When to use |
|---------|-------|--------|-------------|
| **PyTorch / MPS** | Baseline | ~8–12 GB unified | Stable, supports all features. Use when debugging or running SFT. |
| **MLX** | 2–5× faster | ~4–8 GB unified | Apple-native lazy evaluation. Best for long MiniLLM runs. Uses lower batch defaults (2/4/8). |

> **Switching backends** auto-updates Batch size / Grad acc / LoRA rank to the recommended defaults for that backend. You can still adjust them manually.

### Models
- **Use open Qwen2 models** — Ticks `--open` flag. Forces teacher = `Qwen/Qwen2-1.5B-Instruct`, student = `Qwen/Qwen2-0.5B-Instruct`. No HuggingFace account or license needed.
- **Teacher** — Larger model that provides soft targets or hard labels. Must fit in unified memory alongside the student. Rule of thumb: teacher ≤ 3× student parameters.
- **Student** — Smaller model being trained. For MiniLLM, point this at your **SFT checkpoint** (`distilled-minillm/sft_checkpoint`) for best results.
- Click **Refresh** after a model finishes downloading to see it in the dropdown.

### Training (common to all stages)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Epochs** | 2 | SFT: 1 is usually enough. MiniLLM: 2–3. More can overfit on small datasets. |
| **Max samples** | 2000 | Samples drawn from the dataset. 2000 trains in ~1–2 hours. Use 500 for a smoke test. |
| **Batch size** | 8 (PyTorch), 2 (MLX) | Physical samples per device step. Increase until Activity Monitor shows ~80% GPU pressure. |
| **Gradient accumulation** | 8 | Multiply by batch for effective batch (8×8=64 PyTorch, 2×8=16 MLX). Larger effective batch = smoother gradients. |
| **LoRA rank** | 16 | Higher = more trainable params = slower but higher capacity. 16 is a good balance for both backends. 64 for SFT. |

### SFT options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **Learning rate** | 2e-4 | Standard SFT rate. Reduce to 1e-4 if loss oscillates. |
| **Teacher max new tokens** | 128 | Tokens the teacher generates per prompt for the label cache. 128 covers most alpaca responses. |
| **Max sequence length** | 384 | Total tokens (prompt + response) kept for training. Sequences longer than this are truncated. |

### MiniLLM options (PyTorch)
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Softens the teacher distribution. Higher (1.5–2.0) = smoother targets, can stabilize early training. |
| **Learning rate** | 2e-5 | Intentionally 10× lower than SFT. Increase to 5e-5 only if rewards plateau after 100 steps. |
| **Generations per prompt** | 4 | GRPO samples per prompt. **At least 4** to get reward variance within each group (fewer → frac_reward_zero_std stays high → no gradient). 8 gives richer signal but 2× slower. |
| **Max completion length** | 256 | Hard cutoff for student generations. **Critical:** too small → model hits limit before EOS → 80%+ clipped_ratio → reward collapses. 256 tokens ≈ 800 characters is the calibrated default. |
| **Eval every N steps** | 20 | Lower = more detail in metrics.jsonl, higher = faster overall run. 20 is a good balance. |

### MLX options
| Parameter | Default | Guidance |
|-----------|---------|----------|
| **KD temperature** | 1.0 | Same as PyTorch. |
| **Learning rate** | 2e-4 | MLX uses forward-KL + CE, not GRPO, so it tolerates a higher LR. |
| **CE alpha** | 0.2 | Weight of cross-entropy (hard label) loss. 0 = pure KD, 1 = pure CE. 0.2 stabilises early training without losing KD signal (increased from 0.1 for better convergence). |
| **Top-K teacher logits** | 50 | Keeps only top-50 teacher token probabilities. Captures >99% of probability mass while reducing logit memory from ~300 GB to ~300 MB per dataset. Teacher is freed from memory immediately after precompute. |
| **Export quantization bits** | 4 | Bits for the MLX quantized export. 4-bit is standard for llama.cpp-compatible GGUF. |
| **Resume** | off | Continue from the last epoch checkpoint in output_dir if a previous MLX run was interrupted. |

### Watchdog
Creates a `pause.flag` file callback. While training is running, you can pause it by creating `output_dir/pause.flag` from the terminal (`touch distilled-minillm/pause.flag`) and resume by deleting it. Useful for thermal management without losing progress.

### Stop button
Sends **SIGKILL** (immediate termination). The last saved checkpoint is preserved. Use it any time — the run can be resumed via **Resume** (MLX) or by relaunching from the latest checkpoint (PyTorch).
""")
