"""Generates and executes the initial dataset inspection audit notebook."""

import os
import sys
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_DIR = os.path.join(PROJECT_ROOT, "notebooks", "01.EDA")
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "00_initial_dataset_audit.ipynb")

def create_notebook():
    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (.venv - AML/DL)",
            "language": "python",
            "name": "aml-dl-venv",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.5",
        },
    }

    cells = []

    cells.append(nbf.v4.new_markdown_cell(
"""# Initial Dataset Audit
Initial inspection, data quality audit, and summary statistics.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
import time
import psutil
import warnings
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=FutureWarning)

current_dir = os.path.abspath(os.getcwd())
if os.path.basename(current_dir) == 'notebooks':
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
elif os.path.exists(os.path.join(current_dir, 'src', 'data_audit.py')):
    project_root = current_dir
else:
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_audit import (
    find_dataset_path,
    load_dataset,
    inspect_dataset_overview,
    audit_missingness,
    audit_duplicates,
    audit_unique_values,
    audit_numerical_statistics,
    audit_categorical_statistics,
    audit_suspicious_values_and_outliers,
    audit_target_distribution,
    generate_audit_visualizations,
    generate_audit_report,
)

sns.set_theme(style='whitegrid', font_scale=1.0)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', 1000)

print("Environment initialized successfully.")
print(f"Pandas version: {pd.__version__} | NumPy version: {np.__version__}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 1: Identify Dataset
Locate the dataset file across the data directories.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""search_paths = [
    os.path.join(project_root, 'data'),
    os.path.join(project_root, 'data', 'Dataset'),
    os.path.join(project_root, 'data', 'Dataset', 'Parquet', 'Combined Data'),
]

selected_file, discovered_candidates = find_dataset_path(search_dirs=search_paths)

candidates_df = pd.DataFrame(discovered_candidates)
print(f"Total dataset candidates discovered: {len(candidates_df)}")
display(candidates_df[['file_name', 'format', 'size_mb', 'relative_path']].head(10))

print(f"\\n--> Primary Selected Dataset: {selected_file}")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 2: Load Dataset
Load the dataset into pandas and record loading time and memory usage.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""if not os.path.exists(selected_file):
    resolved_path = os.path.join(project_root, selected_file)
else:
    resolved_path = selected_file

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)
t_start = time.time()

df = load_dataset(resolved_path, low_memory=False)

t_elapsed = time.time() - t_start
mem_after = process.memory_info().rss / (1024 * 1024)
df_mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

print(f"Dataset successfully loaded in {t_elapsed:.2f} seconds.")
print(f"RAM utilized: +{mem_after - mem_before:.1f} MB (DataFrame deep size: {df_mem_mb:.1f} MB)")
print(f"Dimensions: {df.shape[0]:,} observations × {df.shape[1]} columns")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 3: Data Profiling
Inspect dataset dimensions, column data types, missing values, duplicates, and summary statistics.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""overview = inspect_dataset_overview(df)

print(f"Rows: {overview['num_rows']:,}")
print(f"Columns: {overview['num_cols']}")
print(f"In-Memory Footprint: {overview['memory_mb']:.1f} MB")
print(f"Numerical Features ({len(overview['numerical_columns'])}): {overview['numerical_columns']}")
print(f"Categorical/Metadata Features ({len(overview['categorical_columns'])}): {overview['categorical_columns']}\\n")

display(overview['summary_table'])
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""missing_df, missing_summary = audit_missingness(df)

print("=" * 60)
print("MISSINGNESS AUDIT SUMMARY")
print("=" * 60)
print(f"Total Cells: {missing_summary['total_cells']:,}")
print(f"Missing Cells: {missing_summary['total_missing_cells']:,} ({missing_summary['overall_missing_percent']}%)")
print(f"Complete Rows (0 missing): {missing_summary['complete_rows_count']:,} ({missing_summary['complete_rows_percent']}%)")
print(f"Completely Clean Columns: {len(missing_summary['completely_clean_columns'])}")
print(f"Columns with Missingness: {len(missing_summary['columns_with_missing'])}\\n")

display(missing_df)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""dup_summary = audit_duplicates(df)
print(f"Exact Duplicate Rows: {dup_summary['num_duplicates']:,} ({dup_summary['duplicate_percentage']}%)")
if dup_summary['num_duplicates'] > 0:
    print("Sample duplicate records:")
    display(dup_summary['sample_duplicates'].head(3))

unique_df = audit_unique_values(df)
print("\\nCardinality Classification across features:")
display(unique_df)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""num_stats = audit_numerical_statistics(df)
print("Summary Statistics for Numerical Predictors & Targets:")
display(num_stats)
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""cat_stats = audit_categorical_statistics(df, top_n=5)
print("Categorical Modalities & Frequency Breakdown:")
display(cat_stats)
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 4: Anomaly Detection
Identify sensor sentinels, physically impossible values, and statistical outliers.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""outlier_df, outlier_summary = audit_suspicious_values_and_outliers(df)

print("=" * 70)
print("ANOMALY & OUTLIER AUDIT RESULTS")
print("=" * 70)
print(f"Features with Telemetry Sentinels: {outlier_summary['features_with_sentinels']}")
print(f"Features with Negative Concentrations: {outlier_summary['features_with_negatives']}")
print(f"Features with Domain Physical Violations: {outlier_summary['features_with_domain_violations']}\\n")

display(outlier_df)

print("\\nCategorical Type & Whitespace Inconsistencies:")
display(outlier_summary['type_inconsistencies_table'])
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 5: Diagnostic Plots
Generate and display diagnostic plots for missingness, distributions, and outliers.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""fig_dir = os.path.join(project_root, 'reports', 'figures')
saved_figs = generate_audit_visualizations(df, output_dir=fig_dir, sample_size=250000)

print(f"Generated and saved {len(saved_figs)} diagnostic figures to: {fig_dir}\\n")
for name, path in saved_figs.items():
    print(f"  • {name}: {os.path.relpath(path, project_root)}")
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""from IPython.display import Image, display
display(Image(filename=saved_figs['missingness_overview']))

display(Image(filename=saved_figs['missingness_correlation']))

display(Image(filename=saved_figs['numerical_distributions']))

display(Image(filename=saved_figs['outlier_boxplots']))

display(Image(filename=saved_figs['target_distributions']))

display(Image(filename=saved_figs['correlation_matrix']))
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 6: Target Distribution
Check score distributions and class balance for CCME WQI targets.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""target_audit = audit_target_distribution(df)

for t_name in target_audit.get('identified_targets', []):
    print("=" * 60)
    print(f"TARGET PROFILE: {t_name}")
    print("=" * 60)
    t_info = target_audit[t_name]
    for k, v in t_info.items():
        if k != 'class_breakdown':
            print(f"  {k}: {v}")
        else:
            print("  Class Breakdown:")
            for cls_name, cls_data in v.items():
                print(f"    - {cls_name:10s}: {cls_data['count']:,} ({cls_data['percentage']}%)")
    print()
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 7: Export Report and Figures
Save diagnostic plots and export the dataset audit summary report.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""report_path = os.path.join(project_root, 'reports', 'dataset_audit_report.md')
report_text = generate_audit_report(
    df=df,
    dataset_path=selected_file,
    output_path=report_path,
    figures_dir=fig_dir,
)

print(f"Audit report saved successfully: {report_path}")
print(f"Report length: {len(report_text.splitlines())} lines ({len(report_text):,} bytes)")
"""
    ))

    cells.append(nbf.v4.new_markdown_cell(
"""## Task 8: Summary
Summary of findings and next steps for data preprocessing.
"""
    ))

    nb.cells = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Created notebook at: {NOTEBOOK_PATH} ({len(cells)} cells)")
    return NOTEBOOK_PATH

def execute_notebook(nb_path):
    print(f"Executing notebook: {nb_path} with kernel 'aml-dl-venv'...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)

    resources = {"metadata": {"path": NOTEBOOK_DIR}}
    client = NotebookClient(nb, timeout=600, kernel_name="aml-dl-venv", resources=resources)
    client.execute()

    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    print(f"Executed and saved notebook in-place: {nb_path}")

if __name__ == "__main__":
    nb_path = create_notebook()
    execute_notebook(nb_path)
