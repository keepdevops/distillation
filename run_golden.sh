#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p runs
exec .pixi/envs/default/bin/python scripts/run_distillation_agent.py --config configs/golden_pipeline.json "$@"
