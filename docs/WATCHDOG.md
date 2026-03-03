# Autonomous Training Watchdog

Deterministic rules for self-optimization during long distillation runs. No LLM agent—just heartbeat checks, plateau detection, and atomic writes. Production-safe on M3 Max.

## Architecture

```
┌─────────────────────┐     poll      ┌──────────────────────────┐
│  training_watchdog  │ ────────────▶  │ trainer_state.json       │
│  (LaunchAgent or    │               │ (HuggingFace Trainer)    │
│   terminal)         │               └──────────────────────────┘
└─────────┬───────────┘
          │ writes
          ▼
┌─────────────────────┐     reads     ┌──────────────────────────┐
│ watchdog_suggestions│ ◀──────────── │ PauseFlagCallback        │
│ pause.flag          │               │ (in Trainer)             │
└─────────────────────┘               └──────────────────────────┘
```

## Rules (configurable)

| Rule | Trigger | Action |
|------|---------|--------|
| Plateau | Last 3 loss deltas < 0.001 | `next_lr_scale *= 0.8` (for next run) |
| Thermal | CPU power > threshold (mW) or thermal pressure | Write `pause.flag` → Trainer saves and exits |

Validator: backs up before overwrite; clamps `lr_scale` to ≥ 0.5.

## Quick Start

### 1. Run watchdog (terminal)

**Python:**

```bash
# One-off check
python scripts/training_watchdog.py ./distilled-minillm --once

# Continuous (every 60s)
python scripts/training_watchdog.py ./distilled-minillm --interval 60
```

**C++** (no Python, lighter):

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build .
./watchdog ../../distilled-minillm --interval 60 --config ../../configs/watchdog_rules.json
```

### 2. Enable pause.flag support in training

```bash
python scripts/distill_minillm.py --output_dir ./distilled-minillm --watchdog
```

### 3. LaunchAgent (survives reboot)

```bash
# Edit paths in scripts/launch_agent/com.caribou.distill-watchdog.plist
cp scripts/launch_agent/com.caribou.distill-watchdog.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.caribou.distill-watchdog.plist
```

## Config

`configs/watchdog_rules.json`:

```json
{
  "plateau": {
    "window": 3,
    "max_delta": 0.001,
    "min_points": 5,
    "lr_scale": 0.8
  },
  "thermal": {
    "enabled": false,
    "pause_if_over": 18000,
    "metric": "cpu_power_mw"
  },
  "validator": { "backup_before_write": true, "max_lr_scale": 0.5 }
}
```

**Thermal (M1/M2/M3):** Apple Silicon does not expose raw CPU °C via standard APIs. Use:

| `metric` | `pause_if_over` | Description |
|----------|-----------------|-------------|
| `cpu_power_mw` | mW (e.g. 18000 = 18W) | CPU power — high power ≈ hot chip |
| `thermal_pressure` | 0–100 | Aggregate thermal throttling (if available) |

Requires `sudo powermetrics`. For daemon use, add NOPASSWD:

```bash
sudo visudo
# Add: your_user ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
```

Override via `--config /path/to/rules.json`.

## Outputs

- **watchdog_suggestions.json** — Proposed params for next run (e.g. `next_lr_scale`)
- **pause.flag** — Signals Trainer to save and exit (thermal/emergency)

## Adding thermal (M3)

Thermal monitoring uses `powermetrics` (CPU power or thermal pressure as proxies for temperature). Implemented in `check_thermal()`.

1. Enable in config: `"thermal": { "enabled": true, "pause_if_over": 18000, "metric": "cpu_power_mw" }`
2. Allow powermetrics without password prompt (for LaunchAgent):

   ```bash
   sudo visudo
   # Add line: your_username ALL=(ALL) NOPASSWD: /usr/bin/powermetrics
   ```

3. Adjust `pause_if_over`: 18000 mW (18W) is a reasonable default for M3 Max under heavy distillation.

## Hybrid with Unsloth / LLaMA-Factory

For faster training + watchdog:

1. Switch distillation to Unsloth or LLaMA-Factory (2–5× faster on M3)
2. Point watchdog at their output dir (or adapt parser for their log format)
3. Same LaunchAgent + rules apply
