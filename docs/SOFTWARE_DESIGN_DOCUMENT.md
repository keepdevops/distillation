# Software Design Document: Knowledge Distillation System
## Bare-Metal, Air-Gapped Environment on Apple M3

**Version:** 1.0  
**Date:** February 2026  
**Target:** Apple M3 (ARM64), Conda, Offline-First

---

## 1. Introduction

### 1.1 Purpose

This SDD specifies the design and implementation of a **knowledge distillation** system for machine learning models, optimized for:

- **Apple M3** (ARM64, unified memory, MPS/Neural Engine)
- **Bare-metal** execution (no Docker/VM; native macOS)
- **Air-gapped** deployment (no internet post-setup; physical media transfer)
- **Conda** for reproducible dependency management

Knowledge distillation transfers knowledge from a large "teacher" model to a smaller "student" model—reducing size, latency, and power while preserving quality. This is ideal for edge deployment, secure/classified environments, and resource-constrained hardware.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|--------------|
| NLP & CV distillation (classification, generation) | Real-time streaming distillation |
| MiniLLM (reverse KL) + vanilla forward KL | Adversarial distillation without extensions |
| Offline Conda env, Hugging Face models (pre-cached) | Online model/data fetches |
| Python primary; optional C++ inference | Chemical/process distillation |

### 1.3 Do I Need an Inference Engine?

**For distillation itself:** No. PyTorch, MLX, or LibTorch handle training and basic inference natively on M3 via MPS.

**For post-distillation deployment:** Recommended.

| Use Case | Recommendation |
|----------|----------------|
| Latency &lt;10 ms | **Core ML** (coremltools) — 2–5× speedup on M3 Neural Engine |
| Cross-platform | **ONNX Runtime** (ARM build) |
| MLX-native | **MLX** doubles as inference engine (lazy eval, unified memory) |
| C++ deployment | **LibTorch** or **llama.cpp** (GGUF) for lightweight engine |

**Bottom line:** Not required for distillation; strongly recommended for production inference.

---

## 2. Requirements

### 2.1 Functional Requirements

| ID | Requirement |
|----|-------------|
| FR1 | Set up Conda env with ML deps compatible with M3 (ARM64) |
| FR2 | Implement distillation pipeline: load teacher/student, prepare data, train via KL + CE loss |
| FR3 | Support bare-metal; optional Docker for portability |
| FR4 | Provide UX: web UI (Gradio) + CLI fallback |
| FR5 | Evaluate on accuracy, size, inference speed |
| FR6 | Optional C++ path for inference optimization |
| FR7 | Full air-gapped workflow via `conda-pack` + pre-cached models/datasets |

### 2.2 Non-Functional Requirements

- **Performance:** ≥80% teacher accuracy on student; &lt;50 ms inference/batch on M3
- **Scalability:** Datasets up to 100k samples; gradient checkpointing for large models
- **Security:** SHA-256 verification on transfers; no runtime network calls
- **Usability:** Intuitive UI; error handling for ARM/MPS mismatches |
- **Maintainability:** Modular code; unit tests for pipeline stages

### 2.3 Constraints

- M3 ARM64 → ARM-native packages only
- Air-gapped → No `conda install` or `pip install` at runtime
- Conda for env management (no pip-only)
- Open-source tooling only

---

## 3. Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STAGING MACHINE (Online)                          │
│  conda pack │ HF cache (models) │ datasets cache │ SHA-256 │ USB/SSD     │
└─────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼ (physical transfer)
┌─────────────────────────────────────────────────────────────────────────┐
│                     TARGET M3 (Air-Gapped Bare-Metal)                     │
│  Conda env │ distill.py │ Gradio UI │ Optional C++ inference             │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Diagram

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  Data Layer  │──▶│ Training     │──▶│ Evaluation   │
│  (HF Datasets│   │ (Distiller,  │   │ (Metrics,    │
│   offline)   │   │  MiniLLM)    │   │  ROUGE, etc) │
└──────────────┘   └──────────────┘   └──────────────┘
        │                    │                  │
        └────────────────────┼──────────────────┘
                             │
                    ┌────────┴────────┐
                    │  UX Layer       │
                    │  Gradio / CLI   │
                    └────────────────┘
```

### 3.3 Technology Stack

| Layer | Primary | Alternative |
|-------|---------|-------------|
| Runtime | Python 3.11 | — |
| Framework | PyTorch 2.x (MPS) | MLX (Apple-native, faster) |
| Models | Hugging Face Transformers | — |
| Distillation | TRL MiniLLMTrainer (reverse KL) | Custom forward KL |
| Inference (optional) | LibTorch (C++) / Core ML | ONNX Runtime |
| UX | Gradio | Streamlit, CLI |
| Env | Conda / Miniforge | — |

---

## 4. Detailed Design

### 4.1 Environment Setup (Conda on M3)

#### environment.yml

```yaml
name: distillation_m3
channels:
  - conda-forge
  - pytorch
  - huggingface
dependencies:
  - python=3.11
  - pytorch::pytorch
  - pytorch::torchvision
  - pytorch::torchaudio
  - transformers
  - datasets
  - accelerate
  - peft
  - evaluate
  - gradio
  - matplotlib
  - seaborn
  - pip
  - pip:
    - bitsandbytes  # 4/8-bit quantization
    - trl           # MiniLLMTrainer
```

#### M3-Specific Notes

- Use `CONDA_SUBDIR=osx-arm64` for ARM packages
- Verify MPS: `torch.backends.mps.is_available()`
- Small batch sizes (4–16) for 18–36 GB RAM
- Gradient checkpointing for large teachers

### 4.2 Distillation Implementation

#### 4.2.1 Python (Bare-Metal or Docker)

**Vanilla Forward KL (classification):**
```
L = α × CE(student, labels) + (1-α) × T² × KL(soft_student || soft_teacher)
```

**MiniLLM (Reverse KL, generative):**
- On-policy sampling from student
- Advantage-weighted policy gradient
- Mode-seeking: student focuses on teacher’s high-confidence outputs

See `scripts/distill_minillm.py` and `scripts/distill_forward.py`.

#### 4.2.2 C++ (Bare-Metal Inference / Simple KD)

Use **LibTorch** (PyTorch C++) for:
- Fast inference post-distillation
- Simple vanilla KD training (classification)

**Build:** Download LibTorch ARM64; CMake; link against Torch.

**Flow:** Python exports TorchScript → C++ loads `.pt` → train/infer.

See `cpp/` directory.

#### 4.2.3 Docker (Optional)

For portability (e.g., CI/CD, x86):

```dockerfile
FROM --platform=linux/arm64 mambaorg/micromamba:1.5.1
COPY environment.yml /tmp/
RUN micromamba install -y -n base -f /tmp/environment.yml
WORKDIR /app
COPY scripts/ /app/scripts/
CMD ["python", "scripts/distill_minillm.py", "--help"]
```

**Note:** MPS passthrough on Mac Docker is experimental; bare-metal preferred for M3 perf.

### 4.3 UX Design

| Mode | Description |
|------|-------------|
| **CLI** | `python scripts/distill_minillm.py --teacher X --student Y --dataset Z` |
| **Gradio** | Local web UI: model selection, hyperparams, progress, plots |
| **Logging** | JSON/CSV metrics, matplotlib plots to disk (no WandB online) |

Gradio runs on `127.0.0.1` only (no public share).

---

## 5. Air-Gapped Workflow

### 5.1 Staging Machine (Online)

1. Install Miniforge (ARM64)
2. Create env: `conda env create -f environment.yml`
3. Cache models: `python scripts/cache_models.py`
4. Cache datasets: `python scripts/cache_datasets.py`
5. Package: `conda pack -n distillation_m3 -o distill-offline.tar.gz`
6. Compute SHA-256 of all artifacts
7. Transfer via USB/SSD

### 5.2 Target M3 (Air-Gapped)

1. Install Miniforge offline
2. `conda unpack distill-offline.tar.gz -d ~/envs/distill-offline`
3. `source ~/envs/distill-offline/bin/activate`
4. `export HF_HOME=/path/to/hf_cache`
5. `export HF_DATASETS_CACHE=/path/to/ds_cache`
6. Run `python scripts/distill_minillm.py ...`

---

## 6. Quantization After Distillation

**Recommended order:** Prune → Distill → Quantize (P-KD-Q)

- Distillation first yields a compressible student
- Quantize last (GPTQ, AWQ) for 2–4× further compression
- On M3: use MLX or coremltools for INT4/INT8

See `docs/QUANTIZATION.md` for details.

---

## 7. Testing and Validation

- **Unit:** pytest for loss functions, data loading
- **Integration:** End-to-end on toy data (e.g., Alpaca subset)
- **Performance:** Benchmark inference before/after distillation on M3
- **Edge cases:** Low RAM, dataset corruption, MPS fallback to CPU

---

## 8. Deployment and Maintenance

- **Deployment:** Bare-metal for dev; Docker for CI if needed
- **Monitoring:** Offline logs (TensorBoard export, JSON)
- **Updates:** Rebuild Conda env for new PyTorch; re-verify SHA-256
- **Risks:** ARM compatibility breaks; mitigate with version pinning

---

## 9. References

- [Hugging Face TRL / MiniLLM](https://huggingface.co/docs/trl/main/en/minillm)
- [Apple MLX](https://github.com/ml-explore/mlx)
- [PyTorch MPS](https://pytorch.org/docs/stable/notes/mps.html)
- [Conda Air-Gap](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-offline-package)
