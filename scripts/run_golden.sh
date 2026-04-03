#!/bin/bash
set -e
cd "$(dirname "$0")/.."
mkdir -p runs
exec .pixi/envs/default/bin/python -m distill.run_distillation_agent --config configs/golden_pipeline.json "$@"
