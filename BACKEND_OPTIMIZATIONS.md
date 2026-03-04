# Backend-Specific Optimizations

**Important:** Different backends support different optimizations. This guide clarifies which optimizations work with each backend.

---

## Optimization Compatibility Matrix

| Optimization | PyTorch (CUDA) | PyTorch (MPS) | MLX | Unsloth |
|--------------|----------------|---------------|-----|---------|
| **Flash Attention 2** | ✅ Yes (2-3x) | ❌ No | ❌ No (has own kernels) | ✅ Yes |
| **torch.compile()** | ✅ Yes (20-40%) | ❌ No (InductorError) | ❌ No (PyTorch-only) | ✅ Yes |
| **DataLoader (4 workers)** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Batch tuning (8/8)** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Memory cache clearing** | ✅ Yes | ✅ Yes | ✅ Yes (mlx.clear_cache) | ✅ Yes |
| **Early stopping** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Quality gates** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |

---

## Backend-Specific Notes

### PyTorch (CUDA) — Best Performance

**Full optimization support:**
- ✅ Flash Attention 2: `pip install flash-attn --no-build-isolation`
- ✅ torch.compile(): Auto-enabled (PyTorch 2.0+)
- ✅ All Phase 1 + 1.5 optimizations

**Expected speedup:** 55-60% (3.0-3.5 hours for 5 trials)

**Hardware:** NVIDIA GPUs (RTX 3090, A100, H100, etc.)

---

### PyTorch (MPS) — Apple Silicon

**Limited PyTorch optimization support:**
- ❌ Flash Attention 2: Not compatible with MPS backend
- ❌ torch.compile(): Disabled due to InductorError (will be fixed in future PyTorch versions)
- ✅ DataLoader optimization (4 workers, prefetch=2)
- ✅ Batch tuning (8/8)
- ✅ Memory cache clearing (torch.mps.empty_cache)
- ✅ Early stopping callback
- ✅ Quality gates

**Expected speedup:** 45-50% (3.5-4.0 hours for 5 trials)

**Hardware:** M1/M2/M3 Mac (MacBook Pro, Mac Studio, etc.)

**Why limitations exist:**
- MPS backend is newer and still catching up to CUDA feature parity
- torch.compile() has compatibility issues with some MPS operations
- Flash Attention requires CUDA-specific kernels

**Recommendation:** Use MLX backend instead for best Apple Silicon performance (see below)

---

### MLX — Apple Silicon Native (Fastest for Mac)

**MLX has its own optimizations:**
- ❌ Flash Attention: Not needed (MLX has optimized kernels built-in)
- ❌ torch.compile(): Not applicable (MLX is not PyTorch)
- ✅ Unified memory architecture (no CPU↔GPU transfers)
- ✅ Lazy evaluation (efficient memory usage)
- ✅ Metal-optimized kernels (faster than MPS)
- ✅ DataLoader equivalent (batch processing)
- ✅ Batch tuning (8)
- ✅ Memory cache clearing (mlx.clear_cache)
- ✅ Early stopping callback
- ✅ Quality gates

**Expected speedup:** 2-5× faster than PyTorch/MPS (native Apple Silicon)

**Hardware:** M1/M2/M3 Mac only

**When to use:**
- Running on Apple Silicon: MLX is 2-5× faster than PyTorch/MPS
- Air-gapped: MLX has smaller dependencies
- Production: Native optimization, no PyTorch overhead

**Installation:**
```bash
pip install mlx mlx-lm
python scripts/distill_mlx.py --open --offline
```

---

### Unsloth — Optimized PyTorch Wrapper

**Optimizations via Unsloth + PyTorch:**
- ✅ Flash Attention 2: Integrated in Unsloth
- ✅ torch.compile(): Supported (if CUDA)
- ✅ Unsloth-specific optimizations (fused kernels)
- ✅ All Phase 1 + 1.5 optimizations

**Expected speedup:** 2-5× faster than base PyTorch (Unsloth's own optimizations)

**Hardware:** NVIDIA GPUs (CUDA required)

**When to use:**
- NVIDIA GPU available
- Want maximum speed with minimal code changes
- Fine-tuning focus (Unsloth specializes in PEFT)

**Installation:**
```bash
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"
python scripts/distill_unsloth.py --open
```

---

## Choosing the Right Backend

### Decision Tree:

**Do you have an NVIDIA GPU?**
- ✅ Yes → Use **Unsloth** (fastest) or **PyTorch (CUDA)** (most flexible)
  - Install Flash Attention for 2-3× speedup
  - torch.compile() auto-enabled

**Do you have Apple Silicon (M1/M2/M3)?**
- ✅ Yes → Use **MLX** (2-5× faster than PyTorch/MPS)
  - Native optimization, no Flash Attention needed
  - Unified memory, lazy evaluation
  - Alternative: PyTorch (MPS) if you need HuggingFace ecosystem features

**CPU only?**
- Use **PyTorch (CPU)** (slow, not recommended for production)
  - No GPU optimizations available
  - Consider cloud GPU (RunPod, Lambda Labs, Modal)

---

## Performance Comparison (5-Trial Run)

| Backend | Hardware | Flash Attn | torch.compile | Time | vs Original |
|---------|----------|------------|---------------|------|-------------|
| **PyTorch (CUDA)** | RTX 4090 | ✅ Yes | ✅ Yes | 3.0 hrs | 58% faster |
| **Unsloth (CUDA)** | RTX 4090 | ✅ Yes | ✅ Yes | 2.5 hrs | 65% faster |
| **MLX (Metal)** | M3 Max | N/A | N/A | 3.0 hrs | 58% faster |
| **PyTorch (MPS)** | M3 Max | ❌ No | ❌ No | 3.5 hrs | 51% faster |
| **PyTorch (CPU)** | i9-13900K | ❌ No | ❌ No | 18 hrs | 0% faster |

*(Times are estimates for Qwen2-1.5B→0.5B distillation with 2000 samples, 2 epochs)*

---

## Common Misconceptions

### ❌ "Flash Attention works everywhere"
**Reality:** Flash Attention is CUDA-only (NVIDIA GPUs). It does NOT work with:
- MPS (Apple Silicon PyTorch backend)
- MLX (uses its own optimized kernels instead)
- CPU (no GPU operations)

### ❌ "torch.compile() works on MPS"
**Reality:** torch.compile() has InductorError on MPS backend (as of PyTorch 2.5). Disabled in our scripts. May work in future PyTorch versions.

### ❌ "MLX needs Flash Attention"
**Reality:** MLX has its own Metal-optimized kernels that are faster than Flash Attention for Apple Silicon. Flash Attention is PyTorch-specific and not needed.

### ❌ "MPS is faster than MLX"
**Reality:** MLX is 2-5× faster than PyTorch/MPS on Apple Silicon due to:
- Unified memory (no transfers)
- Lazy evaluation (efficient memory)
- Native Metal kernels (no PyTorch overhead)

---

## Current Implementation Status

All scripts automatically detect backend and enable/disable optimizations:

**✅ scripts/distill_minillm.py** (PyTorch)
- Detects MPS and disables torch.compile() (compatibility)
- Detects Flash Attention and enables if available
- All Phase 1.5 optimizations enabled

**✅ scripts/distill_mlx.py** (MLX)
- Uses MLX-native optimizations (no Flash Attention needed)
- Batch tuning, early stopping, quality gates

**✅ scripts/distill_unsloth.py** (Unsloth)
- Flash Attention integrated via Unsloth
- All PyTorch optimizations available

**✅ scripts/distill_sft.py** (PyTorch SFT warmup)
- Same backend detection as distill_minillm.py
- Phase 1.5 optimizations for curriculum learning

---

## Troubleshooting

### "Flash Attention import error on Mac"
**Expected.** Flash Attention doesn't work on MPS/MLX. Use MLX backend instead:
```bash
python scripts/distill_mlx.py --open
```

### "torch.compile() InductorError on MPS"
**Fixed.** Our scripts automatically skip torch.compile() on MPS. Update to latest:
```bash
git pull origin main
```

### "MLX slower than expected"
**Check:**
1. Are you using quantization? MLX is fastest with `--q_bits 4`
2. Batch size: MLX benefits from larger batches (try `--batch_size 8`)
3. Unified memory enabled? (automatic on M1/M2/M3)

### "Want fastest speed on Apple Silicon"
**Use MLX backend:**
```bash
pip install mlx mlx-lm
python scripts/distill_mlx.py --open --q_bits 4 --batch_size 8
```

---

## References

- **Flash Attention 2:** https://github.com/Dao-AILab/flash-attention (CUDA only)
- **torch.compile():** https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- **MLX:** https://github.com/ml-explore/mlx (Apple Silicon native)
- **Unsloth:** https://github.com/unslothai/unsloth (optimized PyTorch)
- **MPS Backend:** https://pytorch.org/docs/stable/notes/mps.html (PyTorch on Metal)

---

**Last updated:** 2026-03-03 (Phase 1.5 release)
