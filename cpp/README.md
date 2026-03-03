# C++ Distillation & Watchdog

**Two executables:**

| Binary     | Depends   | Purpose                                             |
|------------|-----------|-----------------------------------------------------|
| `distill`  | LibTorch  | Vanilla KD training; fast inference post-distillation |
| `watchdog` | nlohmann/json (fetched) | Training monitor: plateau detection, writes suggestions |

## Prerequisites

**Watchdog only (no LibTorch):** CMake fetches nlohmann/json. Network required for first configure.

```bash
cd cpp && mkdir -p build && cd build
cmake .. && cmake --build .
./watchdog ../distilled-minillm --interval 60
```

**Distill (LibTorch):**

1. **LibTorch ARM64** — Download from [pytorch.org](https://pytorch.org/get-started/locally/):
   - LibTorch → C++ → macOS → ARM64

2. **Export models from Python** (on staging machine):

```python
import torch
from torchvision.models import resnet50, resnet18

teacher = resnet50(weights="IMAGENET1K_V1").eval()
student = resnet18(num_classes=10)
example = torch.rand(1, 3, 32, 32)

torch.jit.trace(teacher, example).save("teacher.pt")
torch.jit.trace(student, example).save("student.pt")
```

Transfer `teacher.pt` and `student.pt` with your air-gapped bundle.

## Build (Air-Gapped M3)

```bash
export LIBTORCH_PATH=/path/to/libtorch
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$LIBTORCH_PATH ..
cmake --build .
```

## Run

```bash
export DISTILL_TEACHER_PATH=/path/to/teacher.pt
export DISTILL_STUDENT_PATH=/path/to/student.pt
./distill
```

Output: `distilled_student.pt`

## Watchdog CLI

```bash
./watchdog <output_dir> [--interval SEC] [--config PATH] [--once] [--dry-run]
```

- `output_dir` — Directory containing `trainer_state.json` (HuggingFace Trainer)
- `--interval` — Poll interval seconds (default: 60)
- `--config` — JSON rules file (e.g. `../configs/watchdog_rules.json`)
- `--once` — Run one tick and exit
- `--dry-run` — Log only, do not write `watchdog_suggestions.json`

Compatible with Python trainer output. Use with LaunchAgent; see `scripts/launch_agent/`.

## MPS (M3 GPU)

LibTorch must be built with MPS support. Check `torch::mps::is_available()` at runtime. If false, falls back to CPU.
