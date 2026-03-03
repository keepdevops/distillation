# llama.cpp + Distilled Student Model

Run your distilled student with **llama.cpp** for fast, lightweight inference on M3.

## Workflow

```
distill_minillm.py  →  convert_hf_to_gguf.py  →  llama-server / main
(HuggingFace)             (GGUF)                    (llama.cpp)
```

---

## 1. Distill (produces student in HuggingFace format)

**Open models (no login):**
```bash
python scripts/distill_minillm.py --open --output_dir ./distilled-minillm
```

**Meta Llama (requires login + license):**
```bash
huggingface-cli login   # + accept Meta license
python scripts/distill_minillm.py --output_dir ./distilled-minillm
```

Output: `./distilled-minillm/` with `config.json`, `model.safetensors`, tokenizer files.

---

## 2. Convert student to GGUF

Clone **llama.cpp** (inside project or sibling):

```bash
git clone https://github.com/ggerganov/llama.cpp.git
# Inside project: distill/llama.cpp
# Or sibling: ../llama.cpp
```

**Prerequisite:** `pip install sentencepiece` (required for Qwen/Llama tokenizers in convert_hf_to_gguf).

**Helper script** (auto-detects `./llama.cpp` or `../llama.cpp`):

```bash
./scripts/export_student_gguf.sh ./distilled-minillm
# Or: ./scripts/export_student_gguf.sh ./distilled-minillm ./llama.cpp
```

**Manual** (from llama.cpp dir):

```bash
cd llama.cpp
pip install -r requirements.txt
python convert_hf_to_gguf.py /path/to/distill/distilled-minillm \
  --outfile distilled-student.gguf \
  --outtype f16
```

**Quantization options** (smaller = faster, lower quality):

| `--outtype` | Size (1B) | Use |
|-------------|-----------|-----|
| `f16` | ~2.5 GB | Best quality |
| `q8_0` | ~1.3 GB | High quality |
| `q4_K_M` | ~700 MB | Balanced |
| `q4_0` | ~600 MB | Compact |

---

## 3. Run with llama.cpp

**Build llama.cpp** (ARM64 / M3):

```bash
cd llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build
```

**Interactive (CLI):**
```bash
./build/bin/main -m distilled-student.gguf -p "Hello" -n 128
```

**Server (API):**
```bash
./build/bin/llama-server -m distilled-student.gguf --host 127.0.0.1 -c 2048
```

Then `curl http://127.0.0.1:8080/completion` or use OpenAI-compatible clients.

---

## Air-gapped

1. **Staging:** Run distill + convert; produce `distilled-student.gguf`
2. **Transfer:** Copy GGUF + `llama.cpp/build/bin` to USB
3. **Target:** Run `./llama-server -m distilled-student.gguf`

No Python or HuggingFace required on target—just the GGUF and llama.cpp binary.

---

## Chat template

Llama-3.2 uses a specific chat format. If using `llama-server`, pass the chat template:

```bash
./build/bin/llama-server -m distilled-student.gguf --chat-template llama
```

Or use `--jinja` with a custom template file for instruction-tuning.
