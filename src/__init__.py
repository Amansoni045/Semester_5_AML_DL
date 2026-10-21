"""
Water Quality Prediction ML Project — Source Package.

Modules
-------
data_audit      : Dataset inspection, quality auditing, and visualization utilities.
preprocessing   : Reusable data-cleaning pipeline (Role 3 — Data Preprocessing).
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

from .preprocessing import (
    WaterQualityPreprocessor,
    CleaningSummary,
    run_cleaning_pipeline,
    COLUMN_NAME_MAP,
    COLUMN_NAME_MAP_REVERSE,
    MISSING_VALUE_STRATEGY,
    SUSPICIOUS_VALUE_RULES,
    SDL_CLIP_COLUMNS,
    CCME_WQI_CATEGORIES,
    # Assertion / validation helpers
    assert_schema_preserved,
    assert_no_sentinels,
    assert_no_sdl_negatives,
    assert_no_exact_duplicates,
    assert_targets_complete,
    assert_flag_columns_present,
    assert_usa_temperatures_in_celsius_range,
    assert_raw_unmodified,
    generate_data_quality_summary,
)

__all__ = [
    # data_audit
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
    # preprocessing
    "WaterQualityPreprocessor",
    "CleaningSummary",
    "run_cleaning_pipeline",
    "COLUMN_NAME_MAP",
    "COLUMN_NAME_MAP_REVERSE",
    "MISSING_VALUE_STRATEGY",
    "SUSPICIOUS_VALUE_RULES",
    "SDL_CLIP_COLUMNS",
    "CCME_WQI_CATEGORIES",
    "assert_schema_preserved",
    "assert_no_sentinels",
    "assert_no_sdl_negatives",
    "assert_no_exact_duplicates",
    "assert_targets_complete",
    "assert_flag_columns_present",
    "assert_usa_temperatures_in_celsius_range",
    "assert_raw_unmodified",
    "generate_data_quality_summary",
]

