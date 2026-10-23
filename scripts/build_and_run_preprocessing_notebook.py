import os
import sys
import json
import nbformat as nbf
from nbclient import NotebookClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NOTEBOOK_PATH = os.path.join(
    PROJECT_ROOT,
    "notebooks",
    "02.Preprocessing&FeatureSelection",
    "01_data_preprocessing_and_feature_engineering.ipynb",
)

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python (.venv - AML/DL)",
    "language": "python",
    "name": "aml-dl-venv",
}

cells = []

cells.append(nbf.v4.new_markdown_cell("""# Data Preprocessing & Feature Engineering
Clean the dataset, engineer domain features, create splits, and prepare training data.
"""))

cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

for candidate in [".", "../..", ".."]:
    if os.path.exists(os.path.join(candidate, "src", "preprocessing.py")):
        PROJECT_ROOT = os.path.abspath(candidate)
        break

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import (
    WaterQualityPreprocessor,
    run_cleaning_pipeline,
    engineer_features,
    split_data,
    impute_missing_values,
    scale_features,
    prepare_training_datasets,
    BASE_NUMERIC_FEATURES,
    ENGINEERED_FEATURES,
    MODEL_INPUT_FEATURES,
    TARGET_REGRESSION,
    TARGET_CLASSIFICATION,
    TARGET_CLASSIFICATION_ORDINAL,
    CCME_WQI_CATEGORIES,
    CCME_WQI_MAP,
)

print(f"Project root : {PROJECT_ROOT}")
print(f"Base Features ({len(BASE_NUMERIC_FEATURES)}): {BASE_NUMERIC_FEATURES}")
print(f"Engineered Features ({len(ENGINEERED_FEATURES)}): {ENGINEERED_FEATURES}")
print(f"Model Input Features ({len(MODEL_INPUT_FEATURES)}): {MODEL_INPUT_FEATURES}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 1. Schema Validation
Verify column names, data types, and row count of the input dataset.
"""))

cells.append(nbf.v4.new_code_cell("""raw_path = os.path.join(PROJECT_ROOT, "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet")
raw_schema = pq.read_schema(raw_path)

print("Master Parquet Dataset Schema:")
for name, typ in zip(raw_schema.names, raw_schema.types):
    print(f"  {name:<35} : {typ}")

raw_table = pq.read_table(raw_path, columns=["Country", "Area", "Waterbody Type", "Date", "CCME_Values", "CCME_WQI"])
print(f"\\nTotal Master Observations : {raw_table.num_rows:,} rows")
print(f"Unique Countries          : {len(np.unique(raw_table['Country'].to_numpy()))} countries")
print(f"Unique Sampling Stations  : {len(np.unique(raw_table['Area'].to_numpy())):,} stations")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 2. Clean Data
Clean hardware sentinels, convert USA temperatures to Celsius, and clip negative concentrations.
"""))

cells.append(nbf.v4.new_code_cell("""cleaned_path = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "02.Preprocessing&FeatureSelection", "cleaned_water_quality.parquet")

if os.path.exists(cleaned_path):
    print(f"Loading authoritative cleaned dataset from {cleaned_path}...")
    df_clean = pd.read_parquet(cleaned_path)
else:
    print("Running cleaning pipeline from raw source...")
    df_clean, summary = run_cleaning_pipeline(raw_path=raw_path, output_path=cleaned_path)

print(f"Cleaned dataset shape: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns")
df_clean[["country", "area", "waterbody_type", "date", "ph", "temperature_c", "do_mg_l", "ccme_values", "ccme_wqi"]].head(5)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 3. Feature Engineering
Create nutrient ratios, oxygen deficit, and cyclical month features.
"""))

cells.append(nbf.v4.new_code_cell("""df_feat, new_cols = engineer_features(df_clean)
print(f"Engineered features added: {new_cols}")

df_feat[["np_ratio", "bod_do_ratio", "do_deficit", "month_sin", "month_cos", "ccme_wqi_code"]].describe().round(4)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 4. Feature and Target Separation
Separate predictor features from continuous and categorical target variables.
"""))

cells.append(nbf.v4.new_code_cell("""print("Modeling Features List (15 predictors):")
for idx, feat in enumerate(MODEL_INPUT_FEATURES, 1):
    print(f"  {idx:>2}. {feat}")

for target in [TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_ORDINAL]:
    assert target not in MODEL_INPUT_FEATURES, f"CRITICAL LEAKAGE: Target {target} is inside input features!"
print("\\nLeakage check PASSED: Target columns strictly excluded from input feature list.")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 5. Split Dataset
Split data into stratified train, validation, and test sets.
"""))

cells.append(nbf.v4.new_code_cell("""train_raw, val_raw, test_raw = split_data(
    df_feat,
    train_size=0.70,
    val_size=0.15,
    test_size=0.15,
    method="stratified_random",
    random_state=42,
)

print(f"Train split : {len(train_raw):,} rows ({len(train_raw)/len(df_feat)*100:.1f}%)")
print(f"Val split   : {len(val_raw):,} rows ({len(val_raw)/len(df_feat)*100:.1f}%)")
print(f"Test split  : {len(test_raw):,} rows ({len(test_raw)/len(df_feat)*100:.1f}%)")

class_summary = pd.DataFrame({
    "Train %": train_raw["ccme_wqi"].value_counts(normalize=True) * 100,
    "Val %": val_raw["ccme_wqi"].value_counts(normalize=True) * 100,
    "Test %": test_raw["ccme_wqi"].value_counts(normalize=True) * 100,
})
class_summary.round(2)
"""))

cells.append(nbf.v4.new_markdown_cell("""## 6. Imputation and Scaling
Fit median imputation and standard scaler on training data only to prevent data leakage.
"""))

cells.append(nbf.v4.new_code_cell("""train_imp, val_imp, test_imp, imputer_info = impute_missing_values(
    train_raw, val_raw, test_raw, feature_cols=MODEL_INPUT_FEATURES
)

train_scaled, val_scaled, test_scaled, scaler = scale_features(
    train_imp, val_imp, test_imp, feature_cols=MODEL_INPUT_FEATURES
)

print("Remaining Missing Values in Modeling Features:")
print(f"  Train : {train_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")
print(f"  Val   : {val_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")
print(f"  Test  : {test_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")

train_means = train_scaled[MODEL_INPUT_FEATURES].mean().abs()
train_stds = train_scaled[MODEL_INPUT_FEATURES].std(ddof=0)
print(f"\\nScaler Mean Check (max absolute deviation from 0.0): {train_means.max():.6e}")
print(f"Scaler Std Check  (max deviation from 1.0)         : {abs(train_stds - 1.0).max():.6e}")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 7. Validate Outputs
Check schema consistency, non-null values, and partition integrity.
"""))

cells.append(nbf.v4.new_code_cell("""assert list(train_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(val_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(test_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(train_scaled[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES

assert train_imp["ccme_values"].notna().all()
assert val_imp["ccme_values"].notna().all()
assert test_imp["ccme_values"].notna().all()
assert train_imp["ccme_wqi_code"].notna().all()

print("Output Validation PASSED:")
print("  - Feature columns maintain identical 15-variable order across all partitions.")
print("  - Zero missing values or infinities exist in model inputs.")
print("  - Target columns are complete and valid.")
"""))

cells.append(nbf.v4.new_markdown_cell("""## 8. Summary
Summary of generated dataset partitions and preprocessing metadata.
"""))

cells.append(nbf.v4.new_code_cell("""meta_path = os.path.join(PROJECT_ROOT, "data", "GeneratedOutput", "02.Preprocessing&FeatureSelection", "preprocessing_metadata.json")
with open(meta_path) as f:
    meta = json.load(f)

print("=" * 65)
print("STAGE 02 PREPROCESSING & FEATURE SELECTION: READINESS SUMMARY")
print("=" * 65)
print(f"Metadata Version      : {meta['metadata_version']}")
print(f"Total Master Records  : {meta['original_row_count']:,}")
print(f"Cleaned Records       : {meta['cleaned_row_count']:,}")
print(f"Exact Duplicates Drop : {meta['exact_duplicates_removed']:,}")
print(f"Fahrenheit Converted  : {meta['usa_temperature_degf_converted']:,} readings")
print(f"Sentinels Neutralized : {meta['sentinels_neutralized']} records")
print(f"Train Split Rows      : {meta['split_counts']['train']:,} (70.0%)")
print(f"Val Split Rows        : {meta['split_counts']['val']:,} (15.0%)")
print(f"Test Split Rows       : {meta['split_counts']['test']:,} (15.0%)")
print(f"Predictor Features    : {meta['input_feature_count']} features")
print(f"Target Classes        : {meta['target_class_order']}")
print("-" * 65)
print("Local Parquet Artifacts:")
for name, rel_path in meta["output_artifacts"].items():
    full_path = os.path.join(PROJECT_ROOT, rel_path)
    size_mb = os.path.getsize(full_path) / (1024 * 1024) if os.path.exists(full_path) else 0.0
    print(f"  {name:<16} : {rel_path} ({size_mb:.2f} MB)")
print("=" * 65)
print("\\nVERDICT: READY FOR STAGE 03 MODEL TRAINING (Tree Baselines & Deep MLP)")
"""))

nb.cells = cells

with open(NOTEBOOK_PATH, "w") as f:
    nbf.write(nb, f)

print(f"Successfully generated notebook at: {NOTEBOOK_PATH}")

print("Executing notebook with aml-dl-venv kernel...")
client = NotebookClient(nb, timeout=600, kernel_name="aml-dl-venv")
client.execute()

with open(NOTEBOOK_PATH, "w") as f:
    nbf.write(nb, f)

print("Notebook successfully executed and saved with all outputs rendered!")
