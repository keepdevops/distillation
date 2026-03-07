# Universal Gradio UI

The Distill project now includes a **universal Gradio interface** that can load and evaluate models from multiple backends:

- **PyTorch/HuggingFace** - Standard transformers models
- **MLX** - Apple Silicon optimized models
- **GGUF** - llama.cpp quantized models
- **vLLM** - High-performance inference

## Features

✅ **Auto-detection** - Automatically detects model format
✅ **Backend selection** - Override auto-detection if needed
✅ **Artifact detection** - Shows all available export formats
✅ **Training metrics** - Displays training method and checkpoints
✅ **Multi-format support** - Switch between PyTorch, MLX, GGUF, vLLM

## Quick Start

### Standalone Evaluation UI

```bash
# PyTorch model (auto-detected)
python scripts/eval_gradio.py --model_path ./distilled-minillm

# MLX model
python scripts/eval_gradio.py --model_path ./distilled-mlx

# GGUF file
python scripts/eval_gradio.py --model_path ./distilled-minillm/model-q4_0.gguf

# Force specific backend
python scripts/eval_gradio.py --model_path ./distilled-unsloth --backend pytorch

# Custom port
python scripts/eval_gradio.py --model_path ./my-model --port 8080
```

### Full Dashboard

```bash
# Launch dashboard with training plots + eval
python scripts/dashboard.py --runs_dir .

# Custom port
python scripts/dashboard.py --runs_dir . --port 7861
```

## Usage Guide

### 1. Model Info Tab

Shows detected information about your model:

- **Path**: Model directory or file name
- **Training method**: Detected training approach (MiniLLM, SFT, MLX, etc.)
- **Available formats**: All export formats found (pytorch, mlx, gguf, coreml)
- **Metrics**: Whether training metrics are available
- **Checkpoints**: Training checkpoints if available
- **Artifacts**: List of all export files with sizes

### 2. Generate Tab

Test your model with interactive generation:

1. Enter a prompt
2. Adjust max tokens (32–512)
3. Adjust temperature (0.1–2.0)
4. Click "Generate"

**Temperature guide:**
- 0.1–0.3 — Focused, near-deterministic
- 0.7 — Default, balanced
- 1.0–1.5 — More creative / varied

### 3. Batch Eval Tab

Instructions for running `eval_quality.py` with full quality metrics:

- **Distinct-1 / Distinct-2** — lexical diversity
- **3-gram entropy** — generation variety across all outputs
- **Refusal rate** — quality gate (alert if > 5%)
- **Teacher PPL (per-sample)** — accurate per-sample teacher perplexity
- **Judge score (1–10)** — LLM-as-judge quality scoring

```bash
python scripts/eval_quality.py ./your-model --judge --judge-teacher-ppl
```

Results saved to `quality_metrics.json` in the model directory.

### 4. Help Tab

Quick-reference guide covering:
- Backend selection decision table
- Model path examples for each format
- Temperature guide
- Agent output directory structure
- Running the full pipeline
- Watchdog and thermal protection commands

### 5. Backend Selection

The UI auto-detects the model format, but you can override it:

**PyTorch**: Standard HuggingFace models with `config.json` and `.safetensors` or `.bin` weights
**MLX**: Models with `.npz` weights or MLX quantized subdirectories
**GGUF**: Single `.gguf` files or directories containing them
**vLLM**: Uses HuggingFace format but with optimized inference engine

## Supported Backends

### PyTorch

**Requirements:**
```bash
pip install torch transformers peft
```

**Supported formats:**
- Standard HuggingFace models
- LoRA adapters (automatically merged)
- SafeTensors and pickle formats

**Devices:**
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU

### MLX

**Requirements:**
```bash
pip install mlx mlx-lm
```

**Supported formats:**
- MLX models with `config.json`
- MLX quantized models (4-bit, 8-bit)

**Devices:**
- Apple Silicon only (M1, M2, M3, M4)

### GGUF (llama.cpp)

**Requirements:**
```bash
pip install llama-cpp-python
```

**Supported formats:**
- All GGUF quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q8_0, etc.)

**Acceleration:**
- Metal (Apple Silicon)
- CUDA (NVIDIA GPUs)
- CPU

### vLLM

**Requirements:**
```bash
pip install vllm
```

**Supported formats:**
- HuggingFace models (uses vLLM's optimized inference)

**Features:**
- Continuous batching
- PagedAttention
- Faster inference than standard PyTorch

**Devices:**
- CUDA (NVIDIA GPUs) only

## Format Detection

The system automatically detects model format based on files:

| Format | Detection Criteria |
|--------|-------------------|
| PyTorch | `config.json` + `*.safetensors` or `model*.bin` |
| MLX | `*.npz` files or MLX quantized subdirs |
| GGUF | `*.gguf` files |
| LoRA | `adapter_config.json` + `adapter_model.bin` |

## Model Loading

### PyTorch Models

Supports:
- Full models
- LoRA adapters (auto-merged)
- BFloat16 precision
- Device auto-placement

### MLX Models

Requires:
- `config.json` for full models
- Uses lazy evaluation for efficiency
- Unified memory on Apple Silicon

### GGUF Models

Features:
- Context window: 2048 tokens (configurable)
- GPU acceleration via Metal
- All quantization formats supported

### vLLM Models

Features:
- Optimized for throughput
- Efficient memory usage
- Batch processing

## Troubleshooting

### "No config.json found"

**Issue**: PyTorch models require `config.json`

**Solution**: Make sure you're pointing to a complete model directory, not just weights files.

### "MLX not installed"

**Issue**: MLX backend requires `mlx` and `mlx-lm`

**Solution**:
```bash
pip install mlx mlx-lm
```

**Note**: MLX only works on Apple Silicon Macs.

### "llama-cpp-python not installed"

**Issue**: GGUF backend requires llama-cpp-python

**Solution**:
```bash
# CPU only
pip install llama-cpp-python

# With Metal (Apple Silicon)
CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python

# With CUDA
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
```

### "vLLM not installed"

**Issue**: vLLM backend requires vLLM package

**Solution**:
```bash
pip install vllm
```

**Note**: vLLM requires NVIDIA GPU with CUDA.

### GGUF loads but generation is slow

**Issue**: Not using GPU acceleration

**Solution**: Reinstall llama-cpp-python with Metal or CUDA support (see above).

### Raw .npz weights without config.json

**Issue**: MLX training outputs raw weights that need config

**Solution**:
1. Use the full MLX export with config (created by agent)
2. Or use MLX quantized subdirectories which include config

## Architecture

### Components

1. **`universal_model_loader.py`** - Backend-agnostic model loading
2. **`artifact_detector.py`** - Detects available formats and metrics
3. **`eval_gradio.py`** - Standalone evaluation UI (Model Info, Generate, Batch Eval, Help)
4. **`dashboard.py`** - Full dashboard with plots + eval
5. **`eval_quality.py`** - Batch quality evaluation (diversity, teacher PPL, judge scoring)
6. **`data_pipeline.py`** - Shared dataset loading and prompt formatting used by all training scripts

### Model State

The universal loader maintains:
- `model`: Backend-specific model instance
- `tokenizer`: Tokenizer (or None for GGUF)
- `backend`: Current backend type
- `model_path`: Path to loaded model

### Generation API

All backends use a unified generation interface:

```python
loader.generate(
    prompt="Your prompt here",
    max_tokens=128,
    temperature=0.7
)
```

## Examples

### Example 1: Evaluate PyTorch Model

```bash
python scripts/eval_gradio.py --model_path ./distilled-minillm
```

Opens UI at `http://127.0.0.1:7860` with:
- Auto-detected PyTorch backend
- Model info showing training method and artifacts
- Generation interface

### Example 2: Compare GGUF vs PyTorch

```bash
# Terminal 1: PyTorch
python scripts/eval_gradio.py --model_path ./distilled-minillm --port 7860

# Terminal 2: GGUF
python scripts/eval_gradio.py --model_path ./distilled-minillm/model-q4_0.gguf --port 7861
```

Compare speed and quality between formats!

### Example 3: MLX Model

```bash
python scripts/eval_gradio.py --model_path ./distilled-mlx --backend mlx
```

Uses MLX backend for Apple Silicon optimized inference.

### Example 4: Full Dashboard

```bash
python scripts/dashboard.py --runs_dir .
```

Shows:
- Training curves from all runs
- Artifact detection and sizes
- Model evaluation with any backend
- LLM-as-judge quality scoring

## Performance Tips

### PyTorch
- Use `--backend pytorch` with bfloat16 precision
- Enable Flash Attention for 2-3x speedup (when available)

### MLX
- Best performance on Apple Silicon (M1/M2/M3/M4)
- Uses unified memory efficiently
- Try quantized models (4-bit, 8-bit) for speed

### GGUF
- Use Q4_K_M or Q5_K_M for best speed/quality tradeoff
- Q4_0 for maximum speed
- Q8_0 for maximum quality
- Enable Metal acceleration on Mac

### vLLM
- Best for high-throughput batch inference
- Requires NVIDIA GPU
- Efficient memory usage with PagedAttention

## Advanced Usage

### Custom Backend Selection

```python
from universal_model_loader import UniversalModelLoader

loader = UniversalModelLoader()
success, message = loader.load("./my-model", backend="mlx")

if success:
    response = loader.generate("Hello!", max_tokens=50, temperature=0.7)
    print(response)
```

### Artifact Detection

```python
from artifact_detector import detect_artifacts

info = detect_artifacts("./distilled-minillm")
print(f"Formats: {info['formats']}")
print(f"Training method: {info['training_method']}")
print(f"Artifacts: {len(info['artifacts'])}")
```

### Load All Metrics

```python
from artifact_detector import load_all_metrics

metrics = load_all_metrics("./distilled-minillm")
for m in metrics:
    print(f"Step {m['step']}: loss={m.get('loss', 'N/A')}")
```

## Integration

The universal Gradio UI integrates seamlessly with the distillation pipeline:

1. **Run distillation** (any backend: PyTorch, MLX, Unsloth)
2. **Export formats** (GGUF, CoreML, MLX quant)
3. **Launch Gradio** - automatically detects all formats
4. **Switch backends** - compare PyTorch vs GGUF vs MLX
5. **View metrics** - training curves and artifacts

## Security

- Runs on `127.0.0.1` only (localhost)
- No public sharing enabled
- Air-gapped friendly (local files only)
- No data logging or telemetry

## Support

For issues or questions:
1. Check this guide
2. See main [USER_MANUAL.md](USER_MANUAL.md)
3. Report bugs at [GitHub Issues](https://github.com/anthropics/claude-code/issues)
