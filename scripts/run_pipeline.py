#!/usr/bin/env python3
"""
Full pipeline runner for the Water Quality Prediction project.
Executes all six Phase 1 notebooks sequentially in strict pipeline order:
  1. 01.EDA/01_data_understanding.ipynb
  2. 01.EDA/02_data_patterns_and_target.ipynb
  3. 02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb
  4. 03.ModelTraining/01_baseline_and_tree_models.ipynb
  5. 04.ModelEvaluation/01_ml_model_comparison.ipynb
  6. 04.ModelEvaluation/02_feature_importance.ipynb
"""

import os
import sys
import time
import nbformat
from nbclient import NotebookClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NOTEBOOKS = [
    os.path.join(PROJECT_ROOT, "notebooks", "01.EDA", "01_data_understanding.ipynb"),
    os.path.join(PROJECT_ROOT, "notebooks", "01.EDA", "02_data_patterns_and_target.ipynb"),
    os.path.join(PROJECT_ROOT, "notebooks", "02.Preprocessing&FeatureSelection", "01_data_preprocessing.ipynb"),
    os.path.join(PROJECT_ROOT, "notebooks", "03.ModelTraining", "01_baseline_and_tree_models.ipynb"),
    os.path.join(PROJECT_ROOT, "notebooks", "04.ModelEvaluation", "01_ml_model_comparison.ipynb"),
    os.path.join(PROJECT_ROOT, "notebooks", "04.ModelEvaluation", "02_feature_importance.ipynb"),
]


def execute_notebook(nb_path: str, index: int, total: int):
    rel_path = os.path.relpath(nb_path, PROJECT_ROOT)
    print(f"\n[{index}/{total}] Starting: {rel_path}")
    t0 = time.time()

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    client = NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": os.path.dirname(os.path.abspath(nb_path))}},
    )
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    elapsed = time.time() - t0
    print(f"[{index}/{total}] Completed in {elapsed:.1f}s: {rel_path}")


def main():
    total = len(NOTEBOOKS)
    start_all = time.time()
    print("=" * 70)
    print("Starting Water Quality Prediction End-to-End ML Pipeline")
    print("=" * 70)

    for i, nb_path in enumerate(NOTEBOOKS, start=1):
        if not os.path.exists(nb_path):
            print(f"Error: Notebook not found: {nb_path}")
            sys.exit(1)
        execute_notebook(nb_path, i, total)

    total_time = time.time() - start_all
    print("\n" + "=" * 70)
    print(f"Pipeline executed successfully in {total_time:.1f}s!")
    print("=" * 70)


if __name__ == "__main__":
    main()
