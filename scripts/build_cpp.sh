#!/usr/bin/env bash
# Build the distill_cpp pybind11 extension module.
# Usage: bash scripts/build_cpp.sh [--clean]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CPP_SRC="$REPO_ROOT/distill/cpp"
BUILD_DIR="$REPO_ROOT/build/distill_cpp"

echo "==> distill_cpp build"
echo "    source : $CPP_SRC"
echo "    build  : $BUILD_DIR"

# ── Clean ──────────────────────────────────────────────────────────────────
if [[ "${1:-}" == "--clean" ]]; then
    echo "==> Cleaning $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

# ── Ensure pybind11 is available ───────────────────────────────────────────
python -c "import pybind11" 2>/dev/null || {
    echo "==> Installing pybind11 ..."
    pip install pybind11 --quiet
}

PYBIND11_CMAKE="$(python -c "import pybind11; print(pybind11.get_cmake_dir())")"
echo "==> pybind11 cmake dir: $PYBIND11_CMAKE"

# ── Configure ─────────────────────────────────────────────────────────────
mkdir -p "$BUILD_DIR"
cmake -S "$CPP_SRC" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -Dpybind11_DIR="$PYBIND11_CMAKE" \
    -DCMAKE_INSTALL_PREFIX="$REPO_ROOT" \
    2>&1

# ── Build ──────────────────────────────────────────────────────────────────
cmake --build "$BUILD_DIR" --config Release --parallel "$(sysctl -n hw.ncpu 2>/dev/null || echo 4)"

# ── Copy .so next to __init__.py ───────────────────────────────────────────
SO_FILE=$(find "$BUILD_DIR" -name "distill_cpp*.so" -o -name "distill_cpp*.pyd" 2>/dev/null | head -1)
if [[ -z "$SO_FILE" ]]; then
    echo "ERROR: built .so not found in $BUILD_DIR"
    exit 1
fi

DEST="$CPP_SRC/$(basename "$SO_FILE")"
cp "$SO_FILE" "$DEST"
echo "==> Installed: $DEST"

# ── Smoke test ─────────────────────────────────────────────────────────────
python - <<'EOF'
import sys
sys.path.insert(0, ".")
import distill_cpp
r = distill_cpp.ThermalReading()
r.cpu_temp = 44.5
print(f"  ThermalReading: {r}")
m = distill_cpp.ModelMetrics()
m.tokens_per_sec = 125.3
print(f"  ModelMetrics:   {m}")
h = distill_cpp.MetricsHistory()
print(f"  MetricsHistory: len={len(h)}")
print("  distill_cpp smoke test PASSED")
EOF

echo "==> Build complete."
