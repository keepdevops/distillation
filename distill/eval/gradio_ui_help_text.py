"""
Static Markdown strings for the Help tab of the Universal Gradio UI.

Split into two parts so each section stays readable and the importing
module remains within the 300-LOC budget.
"""

HELP_MD_PART1 = """
# Distillation System — Complete Reference

---

## 1. Pipeline Architecture

The system runs in sequential stages orchestrated by `run_distillation_agent.py`:

```
[Data] → [Stage 1: SFT Warmup] → [Stage 2: KD Training] → [Eval] → [Export]
            distill_sft.py         distill_mlx.py               run_eval.py    export_student_gguf.sh
                                   distill_minillm.py            run_benchmarks.py
```

**Choose your path:**

| Goal | Recommended command |
|------|---------------------|
| Fastest on M3 Max | `--backend mlx --export all` |
| Largest student, best quality | `--backend pytorch --export gguf` |
| Domain expert (tax/legal/medical) | `expert_pipeline.py --mode distill` |
| Multi-trial hyperparameter search | `--n_trials 3 --curriculum` |
| Production run (all features) | `scripts/run_golden.sh` |

```bash
# Standard MLX run (recommended for M3 Max)
python -m distill.run_distillation_agent \\
  --open --backend mlx --export all --curriculum --watchdog

# Full production (3 trials, synthetic data, benchmarks, judge)
bash scripts/run_golden.sh
```

---

## 2. Training Backends

### `distill_mlx.py` — MLX Forward KL (recommended for M3 Max)

Apple-native MLX training. **2–5× faster** than PyTorch MPS. Teacher logits are
pre-computed once as top-K sparse tensors (~300 MB vs 311 GB full vocab), then frozen.
Training runs the combined forward KL + CE loss with optional linear annealing.

```bash
python -m distill.training.backends.mlx \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-mlx \\
  --epochs 3 --batch_size 2 --grad_acc 8 \\
  --kd_temp 1.0 --ce_alpha 0.1 --topk_logits 50 \\
  --lora_r 8 --q_bits 4 --watchdog
```

Key flags: `--topk_logits 50`, `--ce_alpha 0.1`, `--kd_temp`, `--q_bits 4`,
`--multi_turn_ratio`, `--resume`.

Outputs: `mlx_student_weights.npz`, `mlx_q4/` quantized dir, `metrics.jsonl`

---

### `distill_minillm.py` — Reverse KL / GRPO (MiniLLM)

Trains via **Group Relative Policy Optimization**: sample G completions per prompt,
score with a reward function, compute group-normalized advantage, clip importance ratio.

```bash
python -m distill.training.backends.minillm \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-minillm \\
  --epochs 2 --batch_size 8 --grad_acc 8 \\
  --lora_r 64 --learning_rate 2e-5 --watchdog
```

Reward: `+0.5` clean, `-1.0` < 10 tokens, `-0.5` > 800 tokens.

---

### `distill_sft.py` — SFT Warmup (Stage 1)

Teacher generates greedy completions; student minimises CE on response tokens only.

```bash
python -m distill.training.backends.sft \\
  --teacher Qwen/Qwen2-1.5B-Instruct \\
  --student Qwen/Qwen2-0.5B \\
  --output_dir ./distilled-sft \\
  --epochs 1 --batch_size 4 --lora_r 64
```

---

### `distill_forward.py` — Hinton KD for Classification

Vanilla forward KL for **encoder classification models** (BERT, DistilBERT).

```bash
python -m distill.training.backends.forward \\
  --teacher bert-base-uncased --student distilbert-base-uncased \\
  --dataset glue --dataset_config sst2 \\
  --temperature 5.0 --alpha 0.5
```
"""

HELP_MD_PART2 = """
---

## 3. Orchestrator

```bash
python -m distill.run_distillation_agent \\
  --open --backend mlx --export all --curriculum \\
  --n_trials 3 --synthetic_data --benchmarks --watchdog \\
  --config configs/golden_pipeline.json
```

---

## 4. Data Pipeline

Supported datasets: `tatsu-lab/alpaca`, `teknium/OpenHermes-2.5`,
`HuggingFaceH4/no_robots`, any local JSONL with `instruction`/`output` or `messages`.

Quality filters: min 20 / max 600 response words, `distinct-2 > 0.35`, refusal & noise detection.

```bash
python -m distill.data.filter --dataset teknium/OpenHermes-2.5 --output_dir ./filtered --target 8000
python -m distill.data.magpie --teacher Qwen/Qwen2-1.5B-Instruct --n 5000 --target 3000 --domain math
```

---

## 5. Evaluation

| Metric | Good threshold |
|--------|----------------|
| Distinct-1 | > 0.3 |
| Distinct-2 | > 0.5 |
| 3-gram entropy | > 8.0 |
| Refusal rate | < 5% |
| Quality gate pass | > 90% |
| Judge score (1–10) | > 7.0 |

```bash
python -m distill.eval.quality ./distilled-mlx --judge --judge-teacher-ppl --n_samples 200
python -m distill.eval.perplexity ./distilled-minillm --backend auto --compare_teacher
python -m distill.eval.benchmarks ./distilled-mlx --baseline_dir ./reference-model
```

---

## 6. Inference Backends

| Backend | Speed (M3 Max) | Use case |
|---------|---------------|----------|
| **gguf** | 8–15× baseline | Production, PPL benchmarks |
| **mlx** | 3–5× baseline | Training + eval on Apple Silicon |
| **pytorch** | baseline | Full fine-tuning, LoRA merge |
| **vllm** | 5–10× baseline | High-throughput NVIDIA serving |

---

## 7. Monitoring & Protection

```bash
python -m distill.thermal_agent --watch ./distilled-mlx --threshold 85 --interval 30
python -m distill.monitor_cpu_gpu_temp --interval 3 --log thermal.log
```

---

## 8. Export

```bash
bash scripts/export_student_gguf.sh ./distilled-minillm
python -m distill.export.coreml --model_dir ./distilled-minillm --quantize int4
```

---

## 9. Session Management

```bash
bash scripts/start.sh --backend mlx --eval
tmux attach -t distill
bash scripts/stop.sh
```

---

## 10. Temperature Guide

| Value | Effect |
|-------|--------|
| 0.0 | Greedy — fully deterministic |
| 0.1–0.3 | Near-deterministic, focused |
| 0.5–0.8 | Balanced — default 0.7 |
| 1.0–1.5 | Creative, more varied outputs |
| > 1.5 | Exploratory / experimental |
"""
