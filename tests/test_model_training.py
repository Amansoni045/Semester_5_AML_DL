import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import joblib
import pytest
import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "03.ModelTraining")


def test_random_forest_model_load_and_predict():
    rf_path = os.path.join(OUTPUT_DIR, "random_forest_model.joblib")
    assert os.path.exists(rf_path), "random_forest_model.joblib must exist"
    rf = joblib.load(rf_path)
    X_dummy = np.random.randn(5, 15).astype(np.float32)
    preds = rf.predict(X_dummy)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_xgboost_model_load_and_predict():
    xgb_path = os.path.join(OUTPUT_DIR, "xgboost_model.json")
    assert os.path.exists(xgb_path), "xgboost_model.json must exist"
    model = xgb.XGBRegressor()
    model.load_model(xgb_path)
    X_dummy = np.random.randn(5, 15).astype(np.float32)
    preds = model.predict(X_dummy)
    assert preds.shape == (5,)
    assert np.all(np.isfinite(preds))


def test_validation_predictions_handoff_file():
    preds_path = os.path.join(OUTPUT_DIR, "tree_val_predictions.parquet")
    assert os.path.exists(preds_path), "tree_val_predictions.parquet must exist for evaluation handoff"
    df_preds = pd.read_parquet(preds_path)
    expected_cols = ["actual", "pred_baseline", "pred_rf", "pred_xgb"]
    for col in expected_cols:
        assert col in df_preds.columns, f"Expected column {col} in validation predictions"
    assert len(df_preds) > 0
    assert np.all(np.isfinite(df_preds["pred_baseline"].to_numpy()))
    assert np.all(np.isfinite(df_preds["pred_rf"].to_numpy()))
    assert np.all(np.isfinite(df_preds["pred_xgb"].to_numpy()))
