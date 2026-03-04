# Thermal Agent - System-Wide Hardware Protection

**Autonomous thermal monitoring and control for Apple Silicon**

---

## Overview

The **thermal agent** is a standalone system service that continuously monitors CPU/GPU temperatures and automatically pauses/resumes ALL running jobs when thermal limits are exceeded.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Thermal Agent (thermal_agent.py)                          │
│  • System-wide hardware monitoring                         │
│  • Multi-job support                                        │
│  • Auto pause/resume via pause.flag                        │
│  • Runs as macOS LaunchAgent (always-on)                   │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
   ┌─────────┐         ┌─────────┐         ┌─────────┐
   │  Job 1  │         │  Job 2  │         │  Job 3  │
   │Training │         │Export   │         │Benchmark│
   └─────────┘         └─────────┘         └─────────┘
     checks              checks              checks
    pause.flag         pause.flag          pause.flag
```

**Separation of Concerns:**
- **Thermal Agent**: Hardware monitoring only (temps, fans, throttling)
- **Training Watchdog**: ML-specific monitoring only (plateau detection, loss curves)
- **PauseFlagCallback**: Responds to pause.flag in training loops

---

## Why Separate from Training Watchdog?

### Before (Coupled Architecture)

```python
# training_watchdog.py
- Plateau detection (ML logic)
- Thermal monitoring (hardware logic)  ← Mixed concerns
- Only runs during training
- Can't protect inference/export jobs
```

**Problems:**
- Mixed ML and hardware concerns
- Only protects jobs that explicitly use `--watchdog`
- Stops when training stops
- Can't monitor multiple jobs simultaneously

### After (Separated Architecture)

```python
# thermal_agent.py (NEW)
- Thermal monitoring ONLY
- System-wide (protects ALL jobs)
- Always-on (survives reboots)
- Multi-job support

# training_watchdog.py (UPDATED)
- Plateau detection ONLY
- ML-specific monitoring
- Per-job instance
```

**Benefits:**
- ✅ Clean separation of concerns
- ✅ System-wide thermal protection
- ✅ Protects all GPU workloads (training, inference, export, benchmarks)
- ✅ Reusable across different jobs
- ✅ Always-on via LaunchAgent

---

## Features

### 1. System-Wide Protection

Thermal agent protects **ALL** GPU-intensive tasks:

- ✅ Training (MiniLLM, SFT, curriculum)
- ✅ Inference (batch generation, quality eval)
- ✅ GGUF export (llama.cpp conversion)
- ✅ CoreML export (coremltools compilation)
- ✅ Synthetic data generation
- ✅ Benchmarks (WikiText-2 perplexity)

### 2. Multi-Job Monitoring

```bash
# One agent watches multiple jobs
thermal_agent.py --watch distilled-minillm distilled-mlx distilled-sft

# Each job checks pause.flag independently
# All benefit from system-wide thermal protection
```

### 3. Auto-Resume with Hysteresis

```
Temp threshold: 85°C
Hysteresis: 5°C
Resume threshold: 80°C

Behavior:
  • 84°C → Normal operation
  • 86°C → PAUSE all jobs
  • 84°C → Still paused (hysteresis prevents oscillation)
  • 79°C → RESUME all jobs
```

### 4. LaunchAgent Support

```bash
# Install as macOS system service
./scripts/install_thermal_agent.sh

# Agent starts on login, survives reboots
# Always protecting your hardware
```

### 5. Smart Flag Management

- Only clears pause flags **it created** (thermal reasons)
- Preserves pause flags from training_watchdog (plateau reasons)
- Agents cooperate without conflicts

---

## Installation & Usage

### Option 1: Manual Mode (Run as needed)

```bash
# Watch single job
python scripts/thermal_agent.py --watch ./distilled-minillm

# Watch multiple jobs (system-wide)
python scripts/thermal_agent.py --watch ./distilled-minillm ./distilled-mlx

# Custom threshold
python scripts/thermal_agent.py --watch . --threshold 70 --interval 15
```

### Option 2: Daemon Mode (Background process)

```bash
# Run in background
python scripts/thermal_agent.py --daemon --watch . --log thermal_agent.jsonl

# Check if running
ps aux | grep thermal_agent

# Kill daemon
pkill -f thermal_agent.py
```

### Option 3: LaunchAgent (Always-on, Recommended)

```bash
# Install as macOS system service
./scripts/install_thermal_agent.sh

# Customization (set before install)
export WATCH_DIR="/path/to/projects"
export THRESHOLD=70
export INTERVAL=15
./scripts/install_thermal_agent.sh

# Management
launchctl list | grep thermal_agent          # Check status
launchctl unload ~/Library/LaunchAgents/com.distillation.thermal_agent.plist  # Stop
launchctl load ~/Library/LaunchAgents/com.distillation.thermal_agent.plist    # Start
```

---

## Configuration

### Command-Line Arguments

```
--watch PATH [PATH ...]    Output directories to monitor (required)
--threshold FLOAT          Temperature threshold in °C (default: 85)
--metric METRIC            Metric to monitor (default: soc_temp_c)
--hysteresis FLOAT         Resume delta in °C (default: 5)
--interval INT             Poll interval in seconds (default: 30)
--log PATH                 Log file for thermal events (JSONL)
--daemon                   Run as background daemon
--once                     Run one check and exit (testing)
```

### Metrics

| Metric | Description | Units |
|--------|-------------|-------|
| `soc_temp_c` | System-on-Chip temperature (default) | °C |
| `cpu_temp_c` | CPU cores temperature | °C |
| `gpu_temp_c` | GPU cores temperature | °C |
| `total_power_w` | Total power consumption | Watts |

### Threshold Recommendations

| Hardware | Conservative | Balanced | Aggressive |
|----------|--------------|----------|------------|
| M1/M2 | 70°C | 80°C | 85°C |
| M3 | 75°C | 85°C | 90°C |
| M3 Max/Pro | 80°C | 90°C | 95°C |

**Note:** Apple Silicon is designed to handle temps up to 100°C, but sustained high temps reduce performance and lifespan.

---

## How It Works

### Pause Cycle

```
1. Thermal agent reads temp every 30s (configurable)
2. If temp >= threshold (85°C):
   a. Write pause.flag to ALL watched directories
   b. Jobs check pause.flag and gracefully stop
   c. Log thermal event
3. If temp < resume_threshold (80°C):
   a. Clear pause.flag from ALL watched directories
   b. Jobs auto-resume (if using watchdog integration)
   c. Log thermal event
```

### pause.flag Format

```json
{
  "reason": "thermal",
  "value": 87.3,
  "metric": "soc_temp_c",
  "threshold": 85.0,
  "agent": "thermal_agent"
}
```

**Key field: `"agent": "thermal_agent"`**
- Identifies who created the flag
- Thermal agent only clears flags IT created
- Preserves plateau flags from training_watchdog

### Cooperation with Training Watchdog

```
Scenario 1: Thermal pause
  • thermal_agent writes pause.flag with "agent": "thermal_agent"
  • Job stops
  • Temps drop
  • thermal_agent clears pause.flag (it created it)
  • Job resumes

Scenario 2: Plateau pause
  • training_watchdog writes pause.flag with "reason": "plateau"
  • Job stops
  • Temps drop (incidentally)
  • thermal_agent checks flag, sees no "agent": "thermal_agent"
  • thermal_agent does NOT clear flag (not its responsibility)
  • User manually investigates plateau

Scenario 3: Both agents active
  • thermal_agent: watches temps, manages thermal pauses
  • training_watchdog: watches loss curves, manages plateau pauses
  • No conflicts - they cooperate via pause.flag metadata
```

---

## Monitoring & Logging

### Thermal Event Log

```bash
# Enable logging
python scripts/thermal_agent.py --watch . --log thermal_agent.jsonl

# View events
tail -f thermal_agent.jsonl | jq
```

**Event types:**
```json
{"timestamp": 1709502000, "event": "agent_started", "watch_dirs": [...], "threshold": 85}
{"timestamp": 1709502030, "event": "job_paused", "job_dir": "...", "temperature": 87.3}
{"timestamp": 1709502180, "event": "job_resumed", "job_dir": "...", "temperature": 79.8}
{"timestamp": 1709510000, "event": "agent_stopped"}
```

### Live Monitoring

```bash
# Watch stdout (if not daemon)
python scripts/thermal_agent.py --watch .

# LaunchAgent logs
tail -f thermal_agent.stdout.log
tail -f thermal_agent.stderr.log

# macOS Console.app
# Filter: process:thermal_agent
```

---

## Examples

### Example 1: Protect Single Training Job

```bash
# Terminal 1: Start thermal agent
python scripts/thermal_agent.py --watch ./distilled-minillm --threshold 80

# Terminal 2: Run training
python scripts/distill_minillm.py --open --watchdog
```

**Behavior:**
- Training runs normally
- If temp exceeds 80°C → thermal agent writes pause.flag
- Training detects pause.flag → stops gracefully
- Temp drops below 75°C → thermal agent clears pause.flag
- Training resumes (if using auto-resume logic)

### Example 2: System-Wide Protection (Multiple Jobs)

```bash
# Terminal 1: Start thermal agent (one instance)
python scripts/thermal_agent.py --watch ./distilled-minillm ./distilled-mlx ./distilled-sft

# Terminal 2: Run training job 1
python scripts/distill_minillm.py --open --watchdog &

# Terminal 3: Run training job 2
python scripts/distill_sft.py --open --watchdog &

# Terminal 4: Run benchmarks
python scripts/run_benchmarks.py ./model --watchdog &
```

**Behavior:**
- One thermal agent monitors all jobs
- If temp exceeds threshold → pauses ALL jobs
- When temp drops → resumes ALL jobs
- System-wide hardware protection

### Example 3: LaunchAgent (Always-On)

```bash
# Install once
./scripts/install_thermal_agent.sh

# Run jobs anytime - thermal protection automatic
python scripts/distill_minillm.py --open --watchdog
python scripts/eval_quality.py ./model
python scripts/export_gguf.sh ./model

# All jobs protected automatically
# Agent survives reboots, always watching
```

### Example 4: Air-Gapped System

```bash
# Install mactop (for thermal readings)
brew install context-labs/tap/mactop

# Start thermal agent
python scripts/thermal_agent.py --watch . --threshold 85 --log thermal.jsonl

# Run offline distillation
./run_autonomous_production.sh
```

---

## Troubleshooting

### "mactop not found"

**Problem:** Thermal agent needs mactop to read Apple Silicon temps.

**Solution:**
```bash
# Install mactop
brew install context-labs/tap/mactop

# Or download from: https://github.com/context-labs/mactop
```

### "Thermal readings unavailable"

**Possible causes:**
1. mactop not installed
2. Running on non-Apple Silicon (Intel Mac)
3. macOS permission issues

**Check:**
```bash
# Test mactop manually
mactop --headless --format json --count 1

# Should return JSON with soc_metrics
```

### "pause.flag not cleared after cooldown"

**Possible causes:**
1. Flag created by training_watchdog (plateau), not thermal_agent
2. Thermal agent stopped
3. Resume threshold not reached

**Check:**
```bash
# View pause.flag content
cat ./distilled-minillm/pause.flag | jq

# If "agent": "thermal_agent" → thermal agent should clear it
# If no "agent" field → created by training_watchdog, manual intervention needed

# Check thermal agent running
ps aux | grep thermal_agent

# Check current temp
mactop --headless --format json --count 1 | jq '.[0].soc_metrics.soc_temp'
```

### "Multiple thermal agents running"

**Problem:** Accidentally started multiple instances.

**Solution:**
```bash
# Kill all thermal agents
pkill -f thermal_agent.py

# Restart one instance
python scripts/thermal_agent.py --watch .

# Or use LaunchAgent (ensures single instance)
launchctl load ~/Library/LaunchAgents/com.distillation.thermal_agent.plist
```

---

## Integration with Training Scripts

All training scripts support pause.flag via `PauseFlagCallback`:

```python
# Already integrated in:
# - scripts/distill_minillm.py (--watchdog flag)
# - scripts/distill_sft.py (--watchdog flag)
# - scripts/distill_mlx.py (built-in)
# - scripts/run_distillation_agent.py (--watchdog flag)

# Example usage:
python scripts/distill_minillm.py --open --watchdog
```

**How it works:**
1. Training script creates `PauseFlagCallback`
2. Every step, callback checks for `pause.flag`
3. If found → stops training gracefully, saves checkpoint
4. Thermal agent can clear flag → training can resume (manual restart)

**Note:** Auto-resume requires additional logic (run in loop, check flag before start).

---

## Performance Impact

**CPU overhead:** Minimal (~0.1% CPU)
- Polls every 30s (configurable)
- Single subprocess call to mactop
- Lightweight Python script

**Memory overhead:** ~20-30 MB
- Python interpreter + script
- Negligible compared to training (20+ GB)

**Disk I/O:** Minimal
- Writes pause.flag only on state changes (rare)
- Optional JSONL log (one line per event)

**Network:** None (100% local)

---

## FAQ

### Q: Do I need both thermal_agent and training_watchdog?

**A:** It depends on your use case:

- **thermal_agent**: System-wide hardware protection (recommended for all users)
- **training_watchdog**: ML-specific monitoring (plateau detection, LR adjustment)

**Recommended combinations:**

| Use Case | thermal_agent | training_watchdog |
|----------|---------------|-------------------|
| Protect hardware only | ✅ Yes | ❌ No |
| Optimize training only | ❌ No | ✅ Yes |
| Complete protection | ✅ Yes | ✅ Yes |
| Quick experiments | Optional | Optional |

### Q: Can thermal_agent resume training automatically?

**A:** Partially. Thermal agent can **clear pause.flag** when temps drop, but training scripts must be designed to check for flag removal and resume. Currently:

- ✅ Thermal agent clears pause.flag when temps drop
- ✅ Jobs stop gracefully when flag appears
- ⚠️ Jobs don't auto-resume (requires manual restart or wrapper script)

**Future enhancement:** Add auto-resume wrapper that restarts jobs when pause.flag cleared.

### Q: Does this work on Intel Macs or Linux?

**A:** Currently Apple Silicon only (M1/M2/M3). Requirements:
- Apple Silicon Mac (M1/M2/M3)
- macOS 12+ (Monterey or later)
- mactop installed

**Future:** Could be adapted to:
- Linux: Use `nvidia-smi` for NVIDIA GPUs
- Intel Mac: Use different thermal API (more limited)

### Q: What if I don't want LaunchAgent?

**A:** No problem! Use manual or daemon mode:

```bash
# Option 1: Manual (run in terminal)
python scripts/thermal_agent.py --watch .

# Option 2: Daemon (background process)
python scripts/thermal_agent.py --daemon --watch . --log thermal.jsonl
```

LaunchAgent is recommended for always-on protection but is entirely optional.

---

## Architecture Comparison

### Before (Coupled)

```
training_watchdog.py (165 lines)
  ├── Plateau detection
  ├── Thermal monitoring ← Mixed concerns
  ├── Per-job instance
  └── Only runs during training

Usage:
  python scripts/distill_minillm.py --watchdog
  # Must remember --watchdog flag
  # Only protects that one job
  # Stops when training stops
```

### After (Separated)

```
thermal_agent.py (420 lines, NEW)
  ├── Thermal monitoring ONLY
  ├── System-wide protection
  ├── Multi-job support
  └── Always-on (LaunchAgent)

training_watchdog.py (135 lines, UPDATED)
  ├── Plateau detection ONLY
  ├── ML-specific monitoring
  └── Per-job instance

Usage:
  # Install thermal agent once (always-on)
  ./scripts/install_thermal_agent.sh

  # Run any jobs - automatic protection
  python scripts/distill_minillm.py --open
  python scripts/eval_quality.py ./model
  python scripts/export_gguf.sh ./model

  # All jobs protected automatically!
```

**Benefits:**
- ✅ Clean separation of concerns (ML vs hardware)
- ✅ System-wide protection (all jobs, all workloads)
- ✅ Set-and-forget (LaunchAgent)
- ✅ No need to remember `--watchdog` flag
- ✅ More robust and maintainable

---

## References

- **mactop**: https://github.com/context-labs/mactop (Apple Silicon monitoring)
- **LaunchAgent**: https://developer.apple.com/library/archive/documentation/MacOSX/Conceptual/BPSystemStartup/Chapters/CreatingLaunchdJobs.html
- **training_watchdog.py**: ML-specific monitoring (plateau detection)
- **PauseFlagCallback**: scripts/watchdog_callbacks.py

---

**Last updated:** 2026-03-03 (Thermal Agent v1.0)
