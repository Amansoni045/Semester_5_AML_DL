"""Unit tests for Water Quality Preprocessing and Data Integrity."""

import os
import json
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "02.Preprocessing&FeatureSelection")
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet")

EXPECTED_INPUT_FEATURES = [
    "ph",
    "temperature_c",
    "do_mg_l",
    "bod_mg_l",
    "ammonia_mg_l",
    "orthophosphate_mg_l",
    "nitrogen_mg_l",
    "nitrate_mg_l",
    "np_ratio",
    "bod_do_ratio",
    "do_deficit",
    "month",
    "quarter",
    "month_sin",
    "month_cos",
]

EXPECTED_BASE_FEATURES = [
    "ph",
    "temperature_c",
    "do_mg_l",
    "bod_mg_l",
    "ammonia_mg_l",
    "orthophosphate_mg_l",
    "nitrogen_mg_l",
    "nitrate_mg_l",
]

TARGET_COLUMNS = ["ccme_values", "ccme_wqi", "ccme_wqi_code"]


def test_raw_dataset_exists():
    assert os.path.exists(RAW_DATA_PATH), f"Raw dataset must exist at {RAW_DATA_PATH}"


def test_preprocessing_metadata_structure():
    metadata_path = os.path.join(DATA_DIR, "preprocessing_metadata.json")
    assert os.path.exists(metadata_path), "preprocessing_metadata.json must exist"

    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert meta["metadata_version"] == "3.0.0"
    assert "Complete Case Analysis" in meta["strategy"]
    assert meta["input_feature_count"] == 15
    assert meta["input_features_ordered"] == EXPECTED_INPUT_FEATURES
    assert meta["split_ratios"] == {"train": 0.7, "val": 0.15, "test": 0.15}
    assert meta["complete_case_count"] == 2817500
    assert meta["split_counts"]["train"] == 1972250
    assert meta["split_counts"]["val"] == 422625
    assert meta["split_counts"]["test"] == 422625

    # Check target quarantine
    for target in TARGET_COLUMNS:
        assert target not in meta["input_features_ordered"], f"Target {target} must not be in input features"


def test_preprocessed_partition_shapes_and_completeness():
    splits = {
        "train.parquet": 1972250,
        "val.parquet": 422625,
        "test.parquet": 422625,
    }

    for filename, expected_rows in splits.items():
        file_path = os.path.join(DATA_DIR, filename)
        assert os.path.exists(file_path), f"{filename} must exist"

        df = pd.read_parquet(file_path)
        assert len(df) == expected_rows, f"{filename} row count mismatch: {len(df)} vs {expected_rows}"

        # Base physical and chemical features must have strictly 0 missing values
        for col in EXPECTED_BASE_FEATURES:
            assert col in df.columns, f"Base feature {col} missing in {filename}"
            assert df[col].isna().sum() == 0, f"Base feature {col} has missing values in {filename}"

        # Engineered features exist with >= 99.9% completeness (only safe zero-division yield NaN)
        for col in EXPECTED_INPUT_FEATURES:
            assert col in df.columns, f"Feature {col} missing in {filename}"
            completeness = (df[col].notna().sum()) / len(df)
            assert completeness >= 0.999, f"Feature {col} completeness below 99.9% in {filename}: {completeness}"

        # Check targets exist and are 100% complete
        for target in TARGET_COLUMNS:
            assert target in df.columns, f"Target {target} missing in {filename}"
            assert df[target].isna().sum() == 0, f"Target {target} has missing values in {filename}"


def test_target_quarantine_in_partitions():
    val_path = os.path.join(DATA_DIR, "val.parquet")
    if os.path.exists(val_path):
        df = pd.read_parquet(val_path)
        feature_set = set(EXPECTED_INPUT_FEATURES)
        target_set = set(TARGET_COLUMNS)
        assert len(feature_set.intersection(target_set)) == 0, "Target leakage detected: targets found in features"
