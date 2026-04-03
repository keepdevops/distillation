# llama.cpp + Distilled Student Model

Run your distilled student with **llama.cpp** for fast, lightweight inference on M3.

**llama.cpp location on this machine:** `/Users/Shared/llama`
**GGUF model output directory:** `/Users/Shared/llama/models/`
**llama-server binary:** `/Users/Shared/llama/llama-server`

## Workflow

```
distill_mlx.py or distill_minillm.py
        ↓
run_distillation_agent.py (--export gguf or --export all)
        ↓  [agent handles conversion automatically]
convert_hf_to_gguf.py → student.gguf
        ↓
llama-server -m student.gguf
```

The agent (`run_distillation_agent.py --export gguf`) handles conversion automatically. Manual steps below are for custom workflows.

---

## 1. Distill (produces student model)

**Recommended — MLX backend:**
```bash
python -m distill.distill_mlx --open --output_dir ./distilled-mlx
```

**Or PyTorch backend:**
```bash
python -m distill.distill_minillm --open --output_dir ./distilled-minillm
```

**Or run the full agent (distill + export in one command):**
```bash
python -m distill.run_distillation_agent --open --backend mlx --export gguf
# GGUF saved to ./distilled-mlx/student-f16.gguf
```

---

## 2. Convert student to GGUF (manual)

llama.cpp is pre-installed at `/Users/Shared/llama`. No cloning needed.

**Prerequisite:** `pip install sentencepiece` (required for Qwen/Llama tokenizers).

**Helper script** (auto-detects llama.cpp location):

```bash
./scripts/export_student_gguf.sh ./distilled-mlx
# Runs in tmux session 'distill-export'; attach with: tmux attach -t distill-export
```

**Manual** (from project root):

```bash
python /Users/Shared/llama/convert_hf_to_gguf.py ./distilled-mlx \
  --outfile /Users/Shared/llama/models/student-f16.gguf \
  --outtype f16
```

**Quantization options** (smaller = faster, lower quality):

| `--outtype` | Size (0.5B model) | Size (1B model) | Use case |
|-------------|-------------------|-----------------|----------|
| `f16` | ~1.0 GB | ~2.5 GB | Best quality, reference |
| `q8_0` | ~0.5 GB | ~1.3 GB | High quality |
| `q4_K_M` | ~0.3 GB | ~700 MB | Balanced (default production) |
| `q4_0` | ~0.25 GB | ~600 MB | Compact |

---

## 3. Run with llama.cpp

**Server (OpenAI-compatible API):**
```bash
/Users/Shared/llama/llama-server \
  -m /Users/Shared/llama/models/student-f16.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  -c 2048
```

Then `curl http://127.0.0.1:8080/completion` or use any OpenAI-compatible client.

**Interactive CLI:**
```bash
/Users/Shared/llama/llama-cli \
  -m /Users/Shared/llama/models/student-f16.gguf \
  -p "Hello" -n 128
```

**Build from source** (if binary unavailable):
```bash
cd /Users/Shared/llama
cmake -B build          # Metal enabled by default on macOS
cmake --build build --config Release
```

---

## 4. Chat template

For instruction-tuned models (Qwen, Llama), pass the chat template:

```bash
/Users/Shared/llama/llama-server \
  -m /Users/Shared/llama/models/student-f16.gguf \
  --chat-template qwen
  # or: --chat-template llama3
```

Or use `--jinja` with a custom template file.

---

## Air-Gapped Deployment

1. **Staging:** Run distill + convert; produce GGUF at `/Users/Shared/llama/models/`
2. **Transfer:** Copy GGUF + `llama-server` binary to USB
3. **Target:** `./llama-server -m student-f16.gguf`

No Python or HuggingFace required on target — just the GGUF and the llama-server binary.
