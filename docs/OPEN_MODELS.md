# Running Without Gated Model Access

**meta-llama** models require:
1. `huggingface-cli login`
2. Accepting the Meta Llama license at huggingface.co

If you don't have access, use **open models** instead.

---

## Quick: Use `--open` flag

**Recommended — MLX backend (2–5× faster on M3):**
```bash
python scripts/distill_mlx.py --open --output_dir ./distilled-mlx
```

**PyTorch backend:**
```bash
python scripts/distill_minillm.py --open --output_dir ./distilled-minillm
```

Both use **Qwen2-1.5B-Instruct** → **Qwen2-0.5B-Instruct** (no login required).

---

## Recommended Open Pairs

| Teacher | Student | Notes |
|---------|---------|-------|
| `Qwen/Qwen2-1.5B-Instruct` | `Qwen/Qwen2-0.5B-Instruct` | Default `--open` |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | same | Quick test (self-distill) |
| `HuggingFaceH4/smollm2-1.7B-Instruct` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Different archs |

Example with custom open pair:
```bash
python scripts/distill_minillm.py \
  --teacher Qwen/Qwen2-1.5B-Instruct \
  --student Qwen/Qwen2-0.5B-Instruct \
  --output_dir ./distilled-qwen
```

---

## Then: Convert to GGUF

```bash
# Agent handles this automatically with --export gguf
python scripts/run_distillation_agent.py --open --backend mlx --export gguf

# Or manually (llama.cpp at /Users/Shared/llama):
./scripts/export_student_gguf.sh ./distilled-mlx
```

---

## With Meta Llama (gated)

```bash
huggingface-cli login
# Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
python scripts/distill_minillm.py --output_dir ./distilled-minillm
```
