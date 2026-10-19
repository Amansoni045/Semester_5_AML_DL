#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script
# Creates the project virtual environment (.venv), installs/updates all
# dependencies from requirements.txt, and registers the Jupyter kernel.
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "======================================================================"
echo "          SETTING UP PYTHON VIRTUAL ENVIRONMENT (.venv)               "
echo "======================================================================"
echo "Project Root: $PROJECT_ROOT"

# 1. Create .venv if missing
if [ ! -d ".venv" ]; then
    echo ">>> Creating new virtual environment at .venv..."
    python3 -m venv .venv
    echo "    Created .venv successfully."
else
    echo ">>> Found existing .venv virtual environment."
fi

# 2. Activate .venv
echo ">>> Activating virtual environment..."
source .venv/bin/activate
echo "    Active Python: $(which python)"

# 3. Upgrade pip & install requirements
echo ">>> Upgrading pip and installing requirements from requirements.txt..."
python -m pip install --upgrade pip --quiet
python -m pip install -r requirements.txt --quiet
echo "    Dependencies installed and up-to-date."

# 4. Register Jupyter / VS Code Kernel
echo ">>> Registering Jupyter kernel for VS Code and notebooks..."
python -m ipykernel install --user --name=aml-dl-venv --display-name="Python (.venv - AML/DL)"
echo "    Kernel [Python (.venv - AML/DL)] registered successfully."

echo "======================================================================"
echo "          ENVIRONMENT SETUP COMPLETED SUCCESSFULLY!                   "
echo "======================================================================"
