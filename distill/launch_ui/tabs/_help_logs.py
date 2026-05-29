"""Help accordion: Live Logs & Progress Bars section."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Live Logs & Progress Bars — Reading Training Output", open=False):
        gr.Markdown("""
### Progress bars
A compact progress bar appears **on every tab** — updates every 2 seconds by parsing log output
for `Step X/Y`, `Epoch X/Y`, tqdm `75%|` patterns, or `Progress: N%`. Turns green on completion.

### Loss & Gradient charts
The **Live Logs** tab and the **Expert Pipeline** tab both show live **Training Loss** and
**Gradient Norm** line charts, updated every 2 seconds. Both MLX (`step=N  loss=X`) and
PyTorch (`{'loss': 'X', 'grad_norm': 'Y'}`) log formats are parsed automatically.

### Log format
Full training output streams in real time (polled every 2 seconds). stdout and stderr merged.

### Key metrics to watch (MiniLLM / GRPO)

| Metric | What it is | Healthy range | Action if outside range |
|--------|-----------|---------------|------------------------|
| `loss` | Training loss (reverse-KL) | Decreasing over first 50 steps | If rising after step 30, LR may be too high |
| `eval_loss` | Validation cross-entropy (logged every eval_steps) | Should decrease and track training loss | Large gap = overfitting |
| `reward` / `eval_reward` | Mean reward across completions in a batch | Should trend from negative toward positive | Stuck at -1.0 = clipping or mode collapse |
| `clipped_ratio` | Fraction of completions that hit max_completion_length | < 30% | If > 60%, lower Max completion length or increase it if responses should be long |
| `frac_reward_zero_std` | Fraction of prompt groups where all completions have identical reward (no GRPO gradient) | < 20% after step 50 | If > 50%, reduce Generations per prompt or check reward function |
| `kl` | KL divergence between student and teacher | Should be finite and decreasing | NaN or exploding = training diverged, stop and reduce LR |

### Key metrics (SFT)
| Metric | Healthy | Notes |
|--------|---------|-------|
| `loss` | Decreasing from ~3–5 to ~1–2 over 1 epoch | Fast decrease early = good. Plateau at >2 = check data quality |
| `grad_norm` | 0.5–2.0 | Spike to >10 = LR too high or data issue |

### Key metrics (MLX)
| Metric | Notes |
|--------|-------|
| `kd_loss` | Forward-KL distillation loss, should decrease |
| `ce_loss` | Cross-entropy loss component (scaled by ce_alpha) |
| `total_loss` | Weighted sum: `(1-ce_alpha)*kd_loss + ce_alpha*ce_loss` |
| `eval_ppl` | Validation perplexity, logged every eval_steps |

### Warning signs
- **`[Process exited with code -9]`** — You clicked Stop (SIGKILL). Normal.
- **`[Process exited with code 1]`** — Script crashed. Scroll up in logs for the Python traceback.
- **`OutOfMemoryError` / `MPS backend out of memory`** — Reduce Batch size or Max completion length. On M3 Max, batch=4 and completion=64 always fits.
- **`RuntimeError: Expected all tensors on same device`** — MPS + bfloat16 edge case. Usually resolves after a restart.
- **`ImportError: trl`** — TRL not installed. Run `pixi run pip install trl`.
- **`nan` in loss after step 1** — Learning rate too high. Reduce by 10×.
- **Reward stuck at -0.5 from step 1** — All completions are clipping. Increase Max completion length to 256+ so responses have room to terminate before the hard limit.
- **`frac_reward_zero_std` > 0.6 after step 20** — All completions in a group get identical reward so GRPO advantage is zero. Increase Generations per prompt to 4–8.

### Thermal note
On M3 Max, GPU temperature under MPS load typically stays at 50–60°C. If you see the machine throttling (iterations getting slower over time), training will continue correctly — the pause.flag watchdog can be used to cool it down without losing progress.

### After training completes
The log ends with:
```
Distilled model saved to ./distilled-minillm
[Process exited with code 0]
```
The merged weights are in the output directory. Go to **Eval** to validate, or run the export scripts (`scripts/export_student_gguf.sh`) to produce a GGUF for llama.cpp.
""")
