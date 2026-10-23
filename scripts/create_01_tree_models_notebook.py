import os
import nbformat as nbf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks", "03.ModelTraining")
NB_PATH = os.path.join(NB_DIR, "01_baseline_and_tree_models.ipynb")

def create_notebook():
    os.makedirs(NB_DIR, exist_ok=True)
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python (.venv - AML/DL)",
            "language": "python",
            "name": "aml-dl-venv",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    }

    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell(
"""# Baseline and Tree-Based Models for Water Quality Prediction
Train a simple mean baseline, Random Forest, and XGBoost regressor to predict continuous water quality scores (`ccme_values`).
Save trained model artifacts and validation predictions for the evaluation teammate.
"""
    ))

    # Cell 1: Imports
    cells.append(nbf.v4.new_markdown_cell(
"""## 1. Setup and Imports
Import standard libraries for data handling, model training, and serialization.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

# Ensure project root is on sys.path
for candidate in ['.', '..', '../..']:
    if os.path.exists(os.path.join(candidate, 'src', 'preprocessing.py')):
        PROJECT_ROOT = os.path.abspath(candidate)
        break

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import MODEL_INPUT_FEATURES

print("Environment ready.")
print(f"Project root: {PROJECT_ROOT}")
print(f"Features count: {len(MODEL_INPUT_FEATURES)}")
"""
    ))

    # Cell 2: Feature and Target Inspection
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Feature and Target Inspection
Inspect the 15 input features and check whether any assignment-requested measurements are missing.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""print("Authoritative Input Features (15):")
for i, feat in enumerate(MODEL_INPUT_FEATURES, 1):
    print(f"  {i:>2}. {feat}")

assignment_features = ['ph', 'do_mg_l', 'temperature_c', 'turbidity', 'conductivity', 'total_dissolved_solids']
print("\\nChecking assignment requested parameters in dataset:")
for p in assignment_features:
    status = "Available" if p in MODEL_INPUT_FEATURES else "NOT AVAILABLE in dataset"
    print(f"  • {p:<25}: {status}")

print("\\nTarget variable: ccme_values (continuous score 0.0 to 100.0)")
"""
    ))

    # Cell 3: Data Loading
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Load Preprocessed Data
Load the training and validation parquet datasets prepared in Stage 02.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""train_path = os.path.join(PROJECT_ROOT, 'data', 'GeneratedOutput', '02.Preprocessing&FeatureSelection', 'train.parquet')
val_path = os.path.join(PROJECT_ROOT, 'data', 'GeneratedOutput', '02.Preprocessing&FeatureSelection', 'val.parquet')

t0 = time.time()
df_train = pd.read_parquet(train_path, columns=MODEL_INPUT_FEATURES + ['ccme_values'])
df_val = pd.read_parquet(val_path, columns=MODEL_INPUT_FEATURES + ['ccme_values'])
load_time = time.time() - t0

print(f"Loaded datasets in {load_time:.2f} seconds.")
print(f"Train shape: {df_train.shape[0]:,} rows x {df_train.shape[1]} columns")
print(f"Val shape  : {df_val.shape[0]:,} rows x {df_val.shape[1]} columns")

X_train = df_train[MODEL_INPUT_FEATURES].to_numpy(dtype=np.float32)
y_train = df_train['ccme_values'].to_numpy(dtype=np.float32)

X_val = df_val[MODEL_INPUT_FEATURES].to_numpy(dtype=np.float32)
y_val = df_val['ccme_values'].to_numpy(dtype=np.float32)

print(f"\\nX_train matrix: {X_train.shape} | y_train vector: {y_train.shape}")
print(f"X_val matrix  : {X_val.shape} | y_val vector  : {y_val.shape}")
"""
    ))

    # Cell 4: Smoke Test
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Pipeline Smoke Test
Run a quick test on a 5,000-sample subset to ensure model fitting and prediction work properly before full-scale training.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""smoke_X_tr = X_train[:5000]
smoke_y_tr = y_train[:5000]
smoke_X_val = X_val[:1000]

quick_rf = RandomForestRegressor(n_estimators=10, max_depth=6, random_state=42, n_jobs=-1)
quick_rf.fit(smoke_X_tr, smoke_y_tr)
smoke_preds = quick_rf.predict(smoke_X_val)

assert len(smoke_preds) == 1000
assert np.all(np.isfinite(smoke_preds))
print("Smoke test passed: Model fitting and prediction pipeline working properly.")
"""
    ))

    # Cell 5: Model A: Mean Baseline
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Model A: Mean Baseline Regressor
Establish a simple baseline that always predicts the mean of the training target.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""t0 = time.time()
mean_prediction = float(np.mean(y_train))
baseline_train_time = time.time() - t0

t0 = time.time()
y_val_pred_baseline = np.full_like(y_val, fill_value=mean_prediction)
baseline_pred_time = time.time() - t0

print(f"Mean Baseline fitted in {baseline_train_time:.4f} seconds.")
print(f"  Training set target mean: {mean_prediction:.4f}")
print(f"  Generated {len(y_val_pred_baseline):,} validation predictions in {baseline_pred_time:.4f} seconds.")
"""
    ))

    # Cell 6: Model B: Random Forest
    cells.append(nbf.v4.new_markdown_cell(
"""## 6. Model B: Random Forest Regressor
Train an ensemble of decision trees. We set max_samples to 100,000 per tree for fast, memory-safe training across the 3 million records.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""rf_model = RandomForestRegressor(
    n_estimators=50,
    max_depth=14,
    max_samples=100000,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest Regressor on full 3M training dataset...")
t0 = time.time()
rf_model.fit(X_train, y_train)
rf_train_time = time.time() - t0

t0 = time.time()
y_val_pred_rf = rf_model.predict(X_val)
rf_pred_time = time.time() - t0

print(f"Random Forest Training Completed in {rf_train_time:.2f} seconds.")
print(f"Generated {len(y_val_pred_rf):,} Random Forest predictions in {rf_pred_time:.2f} seconds.")
"""
    ))

    # Cell 7: Model C: XGBoost Regressor
    cells.append(nbf.v4.new_markdown_cell(
"""## 7. Model C: XGBoost Regressor
Train a gradient boosted decision tree model using histogram binning and early stopping on validation loss.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""xgb_model = xgb.XGBRegressor(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=20,
)

print("Training XGBoost Regressor on full 3M training dataset...")
t0 = time.time()
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
xgb_train_time = time.time() - t0

t0 = time.time()
y_val_pred_xgb = xgb_model.predict(X_val)
xgb_pred_time = time.time() - t0

print(f"\\nXGBoost Training Completed in {xgb_train_time:.2f} seconds.")
print(f"Best Iteration: {xgb_model.best_iteration}")
print(f"Generated {len(y_val_pred_xgb):,} XGBoost predictions in {xgb_pred_time:.2f} seconds.")
"""
    ))

    # Cell 8: Save Model Artifacts and Validation Predictions
    cells.append(nbf.v4.new_markdown_cell(
"""## 8. Save Trained Models and Validation Predictions
Save the trained model files and validation predictions for handoff to the evaluation teammate.
Model evaluation (MAE, RMSE, R² calculation, residual analysis, and model comparison) is handled separately in Stage 04.
"""
    ))
    cells.append(nbf.v4.new_code_cell(
"""out_dir = os.path.join(PROJECT_ROOT, 'data', 'GeneratedOutput', '03.ModelTraining')
os.makedirs(out_dir, exist_ok=True)

# Save Random Forest model
rf_path = os.path.join(out_dir, 'random_forest_model.joblib')
joblib.dump(rf_model, rf_path)

# Save XGBoost model
xgb_path = os.path.join(out_dir, 'xgboost_model.json')
xgb_model.save_model(xgb_path)

# Save validation predictions for evaluation teammate
val_preds_df = pd.DataFrame({
    'actual': y_val,
    'pred_baseline': y_val_pred_baseline,
    'pred_rf': y_val_pred_rf,
    'pred_xgb': y_val_pred_xgb,
})
preds_path = os.path.join(out_dir, 'tree_val_predictions.parquet')
val_preds_df.to_parquet(preds_path, index=False, compression='snappy')

print(f"Saved Phase 1 Model Training Artifacts to: {out_dir}")
print(f"  • Random Forest Model : {os.path.basename(rf_path)}")
print(f"  • XGBoost Model       : {os.path.basename(xgb_path)}")
print(f"  • Validation Handoff  : {os.path.basename(preds_path)} ({len(val_preds_df):,} rows)")
print("\\nNote: Model evaluation metrics (MAE, RMSE, R²) and comparison charts are deferred to the evaluation teammate.")
"""
    ))

    nb.cells = cells
    with open(NB_PATH, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created notebook: {NB_PATH} ({len(cells)} cells)")

if __name__ == "__main__":
    create_notebook()
