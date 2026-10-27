"""Unit tests for Phase 1 Evaluation metrics and model comparison."""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EVAL_DIR = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "04.ModelEvaluation")
PRED_PATH = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "03.ModelTraining", "tree_val_predictions.parquet")


def test_metrics_calculation_known_values():
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred_perfect = np.array([10.0, 20.0, 30.0, 40.0])
    assert mean_absolute_error(y_true, y_pred_perfect) == 0.0
    assert np.sqrt(mean_squared_error(y_true, y_pred_perfect)) == 0.0
    assert r2_score(y_true, y_pred_perfect) == 1.0

    y_pred_off = np.array([12.0, 18.0, 32.0, 38.0])
    assert mean_absolute_error(y_true, y_pred_off) == 2.0
    assert np.isclose(np.sqrt(mean_squared_error(y_true, y_pred_off)), 2.0)
    assert r2_score(y_true, y_pred_off) > 0.90


def test_model_comparison_csv_matches_predictions():
    csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path) or not os.path.exists(PRED_PATH):
        pytest.skip("model_comparison.csv or tree_val_predictions.parquet not yet generated")

    df_csv = pd.read_csv(csv_path)
    df_preds = pd.read_parquet(PRED_PATH)

    assert set(df_csv.columns) == {"model", "MAE", "RMSE", "R2"}
    assert len(df_csv) == 3

    model_mapping = {
        "Mean baseline": "pred_baseline",
        "Random Forest": "pred_rf",
        "XGBoost": "pred_xgb",
    }

    actual = df_preds["actual"]
    for _, row in df_csv.iterrows():
        model_name = row["model"]
        assert model_name in model_mapping, f"Unexpected model name in CSV: {model_name}"
        pred_col = model_mapping[model_name]

        expected_mae = mean_absolute_error(actual, df_preds[pred_col])
        expected_rmse = np.sqrt(mean_squared_error(actual, df_preds[pred_col]))
        expected_r2 = r2_score(actual, df_preds[pred_col])

        assert np.isclose(row["MAE"], expected_mae, rtol=1e-4), f"MAE mismatch for {model_name}"
        assert np.isclose(row["RMSE"], expected_rmse, rtol=1e-4), f"RMSE mismatch for {model_name}"
        assert np.isclose(row["R2"], expected_r2, rtol=1e-4), f"R2 mismatch for {model_name}"


def test_model_performance_benchmarks():
    csv_path = os.path.join(EVAL_DIR, "model_comparison.csv")
    if not os.path.exists(csv_path):
        pytest.skip("model_comparison.csv not yet generated")

    df_csv = pd.read_csv(csv_path).set_index("model")

    # Tree models must vastly outperform mean baseline
    baseline_rmse = df_csv.loc["Mean baseline", "RMSE"]
    rf_rmse = df_csv.loc["Random Forest", "RMSE"]
    xgb_rmse = df_csv.loc["XGBoost", "RMSE"]

    assert rf_rmse < 1.0, f"Random Forest RMSE unexpectedly high: {rf_rmse}"
    assert xgb_rmse < 1.0, f"XGBoost RMSE unexpectedly high: {xgb_rmse}"
    assert rf_rmse < baseline_rmse / 10, "RF must achieve at least 10x error reduction over baseline"
    assert xgb_rmse < baseline_rmse / 10, "XGBoost must achieve at least 10x error reduction over baseline"
