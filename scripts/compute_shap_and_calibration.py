#!/usr/bin/env python3
"""
Compute SHAP values for XGBoost and Probability Calibration for PyTorch Dual-Head MLP.
Generates:
1. data/GeneratedOutput/04.ModelEvaluation/shap_summary.png
2. data/GeneratedOutput/04.ModelEvaluation/dl_calibration_curve.png
3. data/GeneratedOutput/04.ModelEvaluation/calibration_and_shap_metrics.json
"""

import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from xgboost import XGBRegressor
import shap


from src.models import DualHeadMLP
from src.metrics import compute_brier_score, compute_expected_calibration_error

DATA_DIR = REPO_DIR / "data" / "GeneratedOutput"
OUT_DIR = DATA_DIR / "04.ModelEvaluation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_shap_analysis():
    print("--- 1. Computing SHAP Values for XGBoost ---")
    xgb_path = DATA_DIR / "03.ModelTraining" / "xgboost_model.json"
    meta_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "preprocessing_metadata.json"
    val_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "val_scaled.parquet"

    features = json.loads(meta_path.read_text())["input_features_ordered"]
    xgb_model = XGBRegressor()
    xgb_model.load_model(str(xgb_path))


    val_df = pd.read_parquet(val_path)
    val_sample = val_df[features].sample(n=250, random_state=42)

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer(val_sample)

    fig = plt.figure(figsize=(10, 8), dpi=150)
    shap.summary_plot(shap_values.values, val_sample, max_display=15, show=False)
    plt.title("TreeSHAP Feature Attribution: Top 15 Predictors (XGBoost Regressor)", fontsize=13, pad=15)
    plt.tight_layout()
    shap_fig_path = OUT_DIR / "shap_summary.png"
    plt.savefig(shap_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"SHAP summary figure saved to: {shap_fig_path}", flush=True)


    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:5]
    top_shap_features = {features[i]: float(mean_abs_shap[i]) for i in top_indices}
    print(f"Top 5 SHAP features: {top_shap_features}")
    return top_shap_features


def run_calibration_analysis():
    print("\n--- 2. Computing Probability Calibration for PyTorch Dual-Head MLP ---")
    val_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "val_scaled.parquet"
    meta_path = DATA_DIR / "02.Preprocessing&FeatureSelection" / "preprocessing_metadata.json"
    model_path = DATA_DIR / "03.ModelTraining" / "deep_learning_mlp_model.pt"

    features = json.loads(meta_path.read_text())["input_features_ordered"]
    val_df = pd.read_parquet(val_path)

    val_sample = val_df.sample(n=min(10000, len(val_df)), random_state=42)
    X_val = torch.tensor(val_sample[features].values, dtype=torch.float32)
    y_cls_true = val_sample["ccme_wqi_code"].values.astype(int)

    model = DualHeadMLP(input_dim=len(features), num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        _, cls_logits = model(X_val)
        probs = F.softmax(cls_logits, dim=-1).numpy()

    brier = compute_brier_score(y_cls_true, probs, num_classes=5)
    ece, bin_accs, bin_confs, bin_counts = compute_expected_calibration_error(y_cls_true, probs, n_bins=10)

    print(f"Multi-Class Brier Score          : {brier:.4f}")
    print(f"Expected Calibration Error (ECE) : {ece * 100:.2f}%")

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    valid_bins = bin_counts > 0
    ax.plot(bin_confs[valid_bins], bin_accs[valid_bins], "s-", color="#2563eb", linewidth=2, markersize=7, label=f"Dual-Head MLP (ECE = {ece*100:.2f}%)")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Empirical Accuracy", fontsize=12)
    ax.set_title("Probability Calibration Reliability Diagram (Dual-Head PyTorch MLP)", fontsize=13, pad=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.legend(frameon=True, fontsize=11, loc="upper left")
    plt.tight_layout()

    cal_fig_path = OUT_DIR / "dl_calibration_curve.png"
    plt.savefig(cal_fig_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Calibration curve saved to: {cal_fig_path}")

    return {
        "brier_score": float(brier),
        "expected_calibration_error_pct": float(ece * 100.0),
    }


def main():
    top_shap = run_shap_analysis()
    cal_metrics = run_calibration_analysis()

    summary = {
        "shap_top_features": top_shap,
        "calibration": cal_metrics,
    }

    out_json = OUT_DIR / "calibration_and_shap_metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\nAll metrics successfully saved to: {out_json}")


if __name__ == "__main__":
    main()
