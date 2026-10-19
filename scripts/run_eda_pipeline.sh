#!/usr/bin/env bash
# ==============================================================================
# Master EDA Pipeline Automation Script
#
# 1. Checks/creates root virtual environment (.venv)
# 2. Activates .venv
# 3. Installs/updates dependencies from requirements.txt
# 4. Executes all 6 EDA notebooks in sequence using the .venv kernel
# 5. Verifies generated outputs in data/GeneratedOutput/01.EDA/
# ==============================================================================

set -e

# Change directory to project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "           STARTING FULL EDA AUTOMATION PIPELINE                     "
echo "======================================================================"
echo "Project Root: $PROJECT_ROOT"

# Step 1: Check if .venv exists, create if missing
if [ ! -d ".venv" ]; then
    echo ">>> [.venv] not found. Creating new virtual environment..."
    python3 -m venv .venv
    echo "    Created .venv successfully."
else
    echo ">>> Found existing virtual environment at .venv"
fi

# Step 2: Activate .venv
echo ">>> Activating .venv..."
source .venv/bin/activate
echo "    Active Python: $(which python)"

# Step 3: Install / Update requirements
echo ">>> Installing / Updating dependencies from requirements.txt..."
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo "    Dependencies verified and ready."

# Step 4: Run the Python notebook runner
echo ">>> Executing all 6 EDA notebooks in sequence..."
python scripts/run_eda_notebooks.py

echo ""
echo "======================================================================"
echo "           EDA PIPELINE EXECUTION FINISHED SUCCESSFULLY               "
echo "======================================================================"
