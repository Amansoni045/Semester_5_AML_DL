#!/usr/bin/env bash
# ==============================================================================
# Master EDA Pipeline Automation Script
#
# 1. Runs scripts/setup_environment.sh to ensure .venv & requirements are ready.
# 2. Runs scripts/run_eda_pipeline.py using the .venv environment to execute
#    all 6 EDA notebooks in sequence and verify outputs.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Step 1: Run Environment Setup first
bash scripts/setup_environment.sh

# Step 2: Run Python EDA Pipeline Runner
echo ""
python3 scripts/run_eda_pipeline.py
