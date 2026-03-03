# Autonomous Agent for Distillation

Run the distillation pipeline end-to-end without manual steps. Two approaches:

1. **Pipeline orchestrator** — Deterministic script: distill → export GGUF (optional watchdog)
2. **LLM-powered agent** — Agent that chooses models, datasets, hyperparams, runs pipeline, evaluates, iterates

---

## 1. Pipeline Orchestrator (Recommended)

Single script runs the full workflow. Use for headless runs, cron jobs, or LaunchAgent.

### Quick start

```bash
# PyTorch → GGUF (original behavior, unchanged)
python scripts/run_distillation_agent.py --open --export gguf

# MLX backend (2-5× faster on M3) → GGUF
python scripts/run_distillation_agent.py --open --backend mlx --export gguf

# MLX → all exports (GGUF + CoreML + MLX quant)
python scripts/run_distillation_agent.py --open --backend mlx --export all

# With watchdog (plateau detection, pause.flag)
python scripts/run_distillation_agent.py --open --watchdog --backend mlx --export all

# Air-gapped (offline cache only)
python scripts/run_distillation_agent.py --open --offline --backend mlx --export gguf

# Legacy flag still works
python scripts/run_distillation_agent.py --open --export-gguf
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--backend` | `pytorch` | Training backend: `pytorch`, `mlx`, `unsloth` |
| `--export` | `gguf` | Export format: `gguf`, `coreml`, `mlx`, `all`, `none` |
| `--open` | off | Use Qwen2 1.5B→0.5B (no HuggingFace login) |
| `--offline` | off | Air-gapped: local cache only |
| `--watchdog` | off | Enable plateau detection + pause.flag for trainer |
| `--outtype` | `f16` | GGUF quantization: f16, q8_0, q4_K_M |
| `--q_bits` | `4` | MLX quantization bits: 4 or 8 |
| `--coreml_quantize` | none | CoreML quantization: `int4`, `int8`, `float16` |
| `--epochs` | `2` | Training epochs |
| `--max_samples` | `2000` | Max train samples |
| `--lora_r` | `64` | LoRA rank |
| `--temperature` | `1.0` | KD softmax temperature |
| `--config` | — | JSON config file (overrides CLI) |

### Config file

```json
{
  "output_dir": "./distilled-minillm",
  "open": true,
  "offline": false,
  "watchdog": false,
  "backend": "mlx",
  "export": "all",
  "outtype": "f16",
  "q_bits": 4,
  "coreml_quantize": null,
  "epochs": 2,
  "max_samples": 2000,
  "temperature": 1.0,
  "lora_r": 64
}
```

```bash
python scripts/run_distillation_agent.py --config configs/agent_config.json
```

### LaunchAgent (macOS daemon)

To run the agent as a background service:

1. Create a plist that runs the agent (e.g. on a schedule or after login)
2. Distillation is long-running; consider `--watchdog` so the trainer can respond to plateau/thermal signals

Example plist for a one-shot run after login:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.caribou.distill-agent</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/conda/envs/distillation_m3/bin/python</string>
        <string>/path/to/distill/scripts/run_distillation_agent.py</string>
        <string>--open</string>
        <string>--export-gguf</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/distill</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>/tmp/distill-agent.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/distill-agent.err</string>
</dict>
</plist>
```

Install: `cp com.caribou.distill-agent.plist ~/Library/LaunchAgents/` and `launchctl load ~/Library/LaunchAgents/com.caribou.distill-agent.plist`.

---

## 2. Watchdog integration

When `--watchdog` is set:

- The **Python watchdog** (`scripts/training_watchdog.py`) or **C++ watchdog** (`cpp/build/watchdog`) monitors `trainer_state.json` during training
- On plateau, it writes `watchdog_suggestions.json` (e.g. `next_lr_scale`)
- On thermal/emergency, it writes `pause.flag` → trainer saves and exits
- Run the watchdog **in parallel** with distillation (separate terminal or LaunchAgent)

```bash
# Terminal 1: start distillation with watchdog support
python scripts/run_distillation_agent.py --open --watchdog

# Terminal 2: run watchdog (polls every 60s)
python scripts/training_watchdog.py ./distilled-minillm --interval 60
```

Or use the existing [LaunchAgent for the watchdog](scripts/launch_agent/README.md).

---

## 3. LLM-powered agent (optional)

For an agent that **decides what to distill** (model pairs, datasets, hyperparams) and iterates:

| Approach | Use case |
|----------|----------|
| **CrewAI** | Multi-agent: planner → runner → evaluator |
| **AutoGen** | Conversational agents with code execution |
| **Custom loop** | LLM + tool calls: `run_distill(teacher, student, dataset)` |

Example tool definition for an LLM agent:

```python
tools = [
    {
        "name": "run_distillation",
        "description": "Run knowledge distillation from teacher to student model",
        "parameters": {
            "teacher": "HuggingFace model id (e.g. Qwen/Qwen2-1.5B-Instruct)",
            "student": "HuggingFace model id (e.g. Qwen/Qwen2-0.5B-Instruct)",
            "dataset": "Dataset name or path (e.g. tatsu-lab/alpaca)",
            "epochs": 2,
        }
    }
]
```

The agent would call `run_distillation_agent.py` (or `distill_minillm.py` with custom args) via subprocess, then evaluate the output (e.g. run a few prompts, score coherence), and decide whether to retry with different params.

---

## 4. Full workflow summary

```
cache_models.py + cache_datasets.py  (optional, for air-gapped)
         ↓
run_distillation_agent.py --open --backend mlx --export all --watchdog
         ↓
distill_mlx.py  →  distilled-minillm/
         │               metrics.jsonl  (dashboard picks this up)
         │               mlx_student_weights.npz
         │               mlx_q4/  (MLX quantized)
         ↓
convert_hf_to_gguf.py  →  distilled-minillm-f16.gguf
         ↓
export_coreml.py  →  distilled-minillm.mlpackage  (Apple Neural Engine)
         ↓
llama-server -m distilled-minillm-f16.gguf
```

The orchestrator handles steps 2–4 automatically when `--export all` is set.

**Backend decision guide:**

| Situation | Use |
|-----------|-----|
| First time, just want it to work | `--backend pytorch` |
| Daily use on M3, fastest training | `--backend mlx` |
| Lowest memory (requires unsloth install) | `--backend unsloth` |
| Need GGUF for llama-server | `--export gguf` |
| Need iOS/macOS app with ANE | `--export coreml` |
| Need everything | `--export all` |
