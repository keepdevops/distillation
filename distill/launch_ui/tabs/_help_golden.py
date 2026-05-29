"""Help accordion: Golden Pipeline section."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("🏆  Golden Pipeline — Recommended End-to-End Sequence", open=True):
        gr.Markdown("""
```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                        GOLDEN PIPELINE  (best quality)                         ║
╠════╦═══════════════════════╦══════════════════════════════════════════╦═════════╣
║ 1  ║ Data Prep  (optional) ║ Filter dataset or run Magpie synthesis   ║  5–30 m ║
║ 2  ║ Domain Synth (opt.)   ║ Generate domain pairs: code/math/legal/… ║ 30–90 m ║
║ 3  ║ Configure & Launch    ║ Stage: SFT · 1 epoch → sft_checkpoint    ║  ~30 m  ║
║ 4  ║ Configure & Launch    ║ Stage: MiniLLM · Student=sft_checkpoint  ║   ~2 h  ║
║ 5  ║ Eval                  ║ Perplexity → Quality → WikiText-2        ║  ~15 m  ║
║ 6  ║ Auto (agent)          ║ scripts/run_golden.sh → full pipeline headless ║  ~3 h   ║
╚════╩═══════════════════════╩══════════════════════════════════════════╩═════════╝
```

**Steps 1–2 are optional** but significantly improve output quality. Step 3 (SFT) is strongly
recommended before MiniLLM — it gives the student a warm start so GRPO rewards don't
open negative and stay there.

### One-command golden run (terminal)
```bash
./scripts/run_golden.sh                                    # foreground
./scripts/run_golden.sh > runs/golden_pipeline.log 2>&1 &  # background with log
```
Config is at `configs/golden_pipeline.json` — edit it to change dataset, epochs, LoRA rank, etc.

---

### Quickstart (no HF login · ~2 hr)
1. **Configure & Launch** → Stage: **MiniLLM** · Backend: **MLX** · ☑ **Use open Qwen2 models** → **Launch**
2. Watch progress bar on same tab. Switch to **Live Logs** for full stream + loss/grad charts.
3. **Eval** → Run Perplexity Eval once training completes.

---

### When to add SFT warmup
Run SFT first when any of these appear in Live Logs during MiniLLM:
- `reward` stuck at −1.0 for more than 50 steps
- `frac_reward_zero_std` > 0.5 after step 30
- You are using a new domain dataset the student hasn't seen before

After SFT finishes, set the MiniLLM **Student** field to `distilled-minillm/sft_checkpoint`.

---

### Decision guide
| Goal | Recommended path |
|------|-----------------|
| Fastest first result | Quickstart (skip steps 1–3) |
| Best general quality | `./scripts/run_golden.sh` |
| Domain specialist (tax / legal / medical / finance / coding) | **Expert Pipeline** tab |
| Fastest training on M-series Mac | Backend: **MLX**, batch=2, grad_acc=8 |
| Largest pair that fits 36 GB unified memory | Teacher ≤ 3B + student ≤ 1B |
| Resume an interrupted run | MLX: tick **Resume**; PyTorch: relaunch from latest checkpoint |

---

### Outputs written to disk
| File | Contents |
|------|----------|
| `output_dir/*.safetensors` | Merged final model weights (PyTorch backend) |
| `output_dir/mlx_student_weights.npz` | Trained weights (MLX backend, LoRA + base fused) |
| `output_dir/metrics.jsonl` | Per-step loss, reward, eval metrics (JSON lines) |
| `output_dir/train_log.jsonl` | Structured JSON log from agent runs (step, loss, epoch, ts) |
| `output_dir/sft_labels.jsonl` | Cached teacher labels (SFT reuses on re-run) |
| `output_dir/quality_metrics.json` | Generation diversity + judge scores |
| `output_dir/benchmark_results.json` | WikiText-2 perplexity result |
| `runs/ep_<stage>_<timestamp>.log` | Plain-text log from Expert Pipeline runs |
| `runs/ep_<stage>_<timestamp>.jsonl` | Structured JSON log from Expert Pipeline runs |
| `/Users/Shared/llama/models/*.gguf` | GGUF exports for llama.cpp / Ollama |
""")
