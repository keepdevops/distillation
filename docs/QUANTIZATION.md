# Quantization After Distillation

## Ordering

**Best:** Prune → Distill → Quantize (P-KD-Q)  
**Avoid:** Quantize → Distill (noisy teacher, poor student)

Distillation first yields a robust, compressible student. Quantizing afterward typically adds 2–4× compression with &lt;5% quality loss when using modern PTQ.

## Recommended PTQ (2025–2026)

| Method | Notes |
|--------|-------|
| **GPTQ** | Layer-wise Hessian optimization; strong perplexity; slower quant |
| **AWQ** | Activation-aware; faster quant; often better on instruction-tuned models |
| **SliM-LLM** | Mixed-precision; good for post-distillation |

## GPTQ vs AWQ

| Aspect | GPTQ | AWQ |
|--------|------|-----|
| Idea | Second-order error compensation | Protect salient channels via activation stats |
| Quant time | Slower | 3–10× faster |
| Accuracy | Strong | Often better on instruction/reasoning |
| MoE models | Good | Usually better (router handling) |

For new models (Llama-3.2, Qwen2.5, Gemma-2), AWQ is often the default choice.

## M3 / Air-Gapped

- Use **llama.cpp** (GGUF) or **MLX** for quantized inference on M3
- **coremltools** for Neural Engine
- Pre-quantize on staging; transfer GGUF with your bundle
