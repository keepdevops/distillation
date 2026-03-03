# Running Without Gated Model Access

**meta-llama** models require:
1. `huggingface-cli login`
2. Accepting the Meta Llama license at huggingface.co

If you don't have access, use **open models** instead.

---

## Quick: Use `--open` flag

```bash
python scripts/distill_minillm.py --open --output_dir ./distilled-minillm
```

Uses **Qwen2-1.5B-Instruct** → **Qwen2-0.5B-Instruct** (no login).

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
./scripts/export_student_gguf.sh ./distilled-minillm ../llama.cpp
```

(or `./distilled-qwen` if you used Qwen)

---

## With Meta Llama (gated)

```bash
huggingface-cli login
# Accept license: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
python scripts/distill_minillm.py --output_dir ./distilled-minillm
```
