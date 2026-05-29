"""Help accordion: Troubleshooting & FAQ section."""
from __future__ import annotations

import gradio as gr


def build_section():
    with gr.Accordion("Troubleshooting & FAQ", open=False):
        gr.Markdown("""
### Q: Training is very slow (>300s per iteration)
- Reduce **Max completion length** to 64–96. Generation is the bottleneck — shorter completions = dramatically faster.
- Reduce **Generations per prompt** to 2 (minimum).
- Reduce **Eval every N steps** to 50+ to spend less time on evaluation.
- Switch backend to **MLX** (2–5× faster on M3).

### Q: Reward is stuck at -1.0 (mode collapse)
- Lower **Max completion length** — the model is generating nothing meaningful before hitting the hard limit.
- Increase **KD temperature** to 1.5 — softens the teacher targets, easier for student to match.
- Use SFT warmup first — gives the student a starting distribution to build from.

### Q: clipped_ratio is >80%
- The student is hitting max_completion_length on almost every generation.
- Lower **Max completion length** to 64 or 96 tokens.
- Or: the model is in a loop/repetition mode — check distinct-2 with Quality Eval.

### Q: Loss is NaN or exploding after a few steps
- **Learning rate too high.** Lower by 10×: 2e-5 → 2e-6 for MiniLLM, 2e-4 → 2e-5 for SFT.
- Check **grad_norm** in logs — if it's >10 before loss explodes, LR is definitely too high.

### Q: "No module named 'trl'" or "No module named 'transformers'"
```bash
cd /Users/caribou/distill
pixi run pip install trl transformers peft datasets
```
Or relaunch the UI through pixi: `pixi run python -m distill.launch_ui`

### Q: Can I run multiple jobs at once?
No — only one subprocess is managed by the UI at a time. If you need parallel runs, open a second terminal and call the scripts directly. The UI will show "A run is already in progress" if you try to launch while one is active.

### Q: Where are HuggingFace models cached?
Default: `~/.cache/huggingface/hub/`. Set `HF_HOME` environment variable to redirect. The dropdowns auto-scan this cache.

### Q: How do I use a locally downloaded GGUF model?
GGUFs are for inference only (llama.cpp / Ollama), not training. The training scripts use HF-format models. The pipeline exports to GGUF *after* training via `scripts/export_student_gguf.sh`.

### Q: The progress bar shows 0% but a run is active
The bar parses `Step X/Y`, `Epoch X/Y`, or tqdm `N%|` from the log. Scripts that don't emit
those patterns (e.g. model download phase) show 0% until the first progress line appears.
The Run status textbox ("running (pid …)") confirms the process is alive regardless.

### Q: The UI shows "idle" but I launched a run
Check **Live Logs** tab — the process may have crashed immediately. The status polling updates every 2 seconds.
Also check: if you launched from a tab other than Configure & Launch, only that tab's status textbox
updates on click; all tabs' progress bars update via the timer once the process writes output.

### Q: `FileNotFoundError: Directory domain_data/<domain> is neither a Dataset directory nor a DatasetDict directory`
The domain's `magpie_raw.jsonl` is empty — synthesis was never run for that domain.
Generate it first from the **Domain Synthesis** tab, or via terminal:
```bash
pixi run python -m distill.data.magpie \\
  --domain coding --n 500 --output_dir domain_data/coding \\
  --backend mps --batch_size 8 --resume
```
Replace `coding` with `finance`, `medical`, or `tax` as needed. The `hf_dataset/` directory
is created automatically when generation finishes.

### Q: `ValueError: Unknown split "train". Should be one of ['train_sft', 'test_sft', …]`
The dataset uses non-standard HuggingFace split names. The loader now auto-selects
`train` → `train_sft` → `train_gen` → first available split. If you see this on an older
install, update to the latest code.

### Q: Domain synthesis falls back to 'general' prompts even after selecting a domain
Symptom: `Domain registry not found … using built-in general prompts.`
The domain registry path was incorrect in older builds. Verify the fix is in place:
```bash
python -c "from distill.data.magpie_filter import _DEFAULT_DOMAINS_FILE; \\
           from pathlib import Path; print(Path(_DEFAULT_DOMAINS_FILE).exists())"
# Should print: True
```
If it prints `False`, the registry points at the wrong path — pull the latest code.

### Q: Training watchdog not running / `zsh: command not found: affeinate`
The `watchdog` tmux window may have a typo (`affeinate` vs `caffeinate`). Restart it:
```bash
caffeinate -dims pixi run python -m distill.training_watchdog \\
  distilled-minillm --config configs/watchdog_rules.json 2>&1 | tee -a watchdog.log
```
""")
