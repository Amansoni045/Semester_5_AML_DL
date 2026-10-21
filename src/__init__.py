"""
Water Quality Prediction ML Project - Data Audit Package.
Contains reusable data inspection, quality audit, and visualization utilities.
"""

from .data_audit import (
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
    run_full_audit,
)

__all__ = [
    "find_dataset_path",
    "load_dataset",
    "inspect_dataset_overview",
    "audit_missingness",
    "audit_duplicates",
    "audit_unique_values",
    "audit_numerical_statistics",
    "audit_categorical_statistics",
    "audit_suspicious_values_and_outliers",
    "audit_target_distribution",
    "generate_audit_visualizations",
    "generate_audit_report",
    "run_full_audit",
]
