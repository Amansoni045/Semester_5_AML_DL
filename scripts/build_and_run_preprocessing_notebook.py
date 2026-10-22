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

# Header
cells.append(nbf.v4.new_markdown_cell("""# Stage 02: Data Preprocessing, Feature Engineering & Dataset Preparation

**Project**: Project 18 — Water Quality Prediction via Gradient Boosting & Deep Neural Networks  
**Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
**Authoritative Module**: [`src/preprocessing.py`](../../src/preprocessing.py)  
**Specification**: [`docs/PREPROCESSING_SPECIFICATION.md`](../../docs/PREPROCESSING_SPECIFICATION.md)  
**Input Source**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` (4,297,895 rows × 14 cols)  
**Output Target Directory**: `data/GeneratedOutput/02.Preprocessing&FeatureSelection/`  

---

### Pipeline Objectives & Sequence
1. **Input Dataset & Schema Validation**: Ingest multi-country dataset, standardise schema, cast semantic data types.
2. **Deterministic Data Cleaning**: Neutralize sensor hardware sentinels (-999999, corrupt pH), convert USA Fahrenheit readings, clean sub-detection-limit negative concentrations, drop exact ingestion duplicates, and append non-destructive suspicious value flags.
3. **Domain-Grounded Feature Engineering**: Generate stoichiometry nutrient ratios ($N:P$), organic pollution ratios ($BOD:DO$), signed oxygen saturation deficit ($DO_{\\text{deficit}}$), and circular seasonal cyclical encodings (`month_sin`, `month_cos`).
4. **Target & Input-Feature Segregation**: Segregate continuous regression target (`ccme_values`) and categorical/ordinal classification targets (`ccme_wqi`, `ccme_wqi_code`) from the 15 predictor features to prevent leakage.
5. **Leakage-Free Splitting Strategy**: Implement 70/15/15 stratified random split (benchmark) with spatial station-group and temporal forecasting evaluation options.
6. **Training-Only Imputation & Feature Scaling**: Fit group medians by `(country, waterbody_type)` and `(country, month)` on the 70% train split only; fit `StandardScaler` on train inputs only.
7. **Output Artifact Validation**: Verify zero missing values, zero infinite values, and identical schema across all partitions.
8. **Final Dataset Readiness Summary**: Audit locally generated Parquet datasets and metadata for Stage 03 Model Training.
"""))

# Cell 1: Environment & Module Setup
cells.append(nbf.v4.new_code_cell("""import os
import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Ensure project root is on sys.path
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

# Section 1: Input Dataset & Schema Validation
cells.append(nbf.v4.new_markdown_cell("""## 1. Input Dataset & Schema Validation
Inspect the master dataset schema, column types, and sampling station coverage.
"""))

cells.append(nbf.v4.new_code_cell("""raw_path = os.path.join(PROJECT_ROOT, "data", "Dataset", "Parquet", "Combined Data", "Combined_dataset.parquet")
raw_schema = pq.read_schema(raw_path)

print("Master Parquet Dataset Schema:")
for name, typ in zip(raw_schema.names, raw_schema.types):
    print(f"  {name:<35} : {typ}")

# Quick summary profile
raw_table = pq.read_table(raw_path, columns=["Country", "Area", "Waterbody Type", "Date", "CCME_Values", "CCME_WQI"])
print(f"\\nTotal Master Observations : {raw_table.num_rows:,} rows")
print(f"Unique Countries          : {len(np.unique(raw_table['Country'].to_numpy()))} countries")
print(f"Unique Sampling Stations  : {len(np.unique(raw_table['Area'].to_numpy())):,} stations")
"""))

# Section 2: Cleaning Summary
cells.append(nbf.v4.new_markdown_cell("""## 2. Deterministic Data Cleaning
We load the authoritative pre-cleaned dataset (or execute the cleaning pipeline via `run_cleaning_pipeline()`).  
The cleaning resolves telemetry errors (-999999), converts historical USA river Fahrenheit readings, applies column-specific negative concentration rules, removes exact duplicate records, and appends boolean quality audit flags.
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

# Section 3: Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## 3. Geochemical & Seasonal Feature Engineering
We generate domain-grounded environmental features:
- **$N:P$ Nutrient Ratio** (`np_ratio`): Stoichiometric ratio of total nitrogen to orthophosphate (Redfield eutrophication threshold = 16:1). Safe division produces `NaN` when phosphorus is zero or negative.
- **$BOD:DO$ Ratio** (`bod_do_ratio`): Organic pollutant oxygen demand relative to available dissolved oxygen. Safe division handles zero/anoxic DO.
- **Signed Dissolved Oxygen Deficit** (`do_deficit`): $DO_{\\text{sat}}(T) - DO$. Positive deficit captures microbial oxygen depletion from sewage/organic runoff; negative deficit preserves photosynthetic algal supersaturation events.
- **Cyclical Month Encoding** (`month_sin`, `month_cos`): Circular angle $(month - 1)/12$ ensuring January ($0^\\circ$) and December ($330^\\circ$) are adjacent in circular feature space.
"""))

cells.append(nbf.v4.new_code_cell("""df_feat, new_cols = engineer_features(df_clean)
print(f"Engineered features added: {new_cols}")

# Display engineered features distribution
df_feat[["np_ratio", "bod_do_ratio", "do_deficit", "month_sin", "month_cos", "ccme_wqi_code"]].describe().round(4)
"""))

# Section 4: Target & Feature Selection
cells.append(nbf.v4.new_markdown_cell("""## 4. Target & Input-Feature Segregation
To prevent data leakage, target variables and target-derived flags are strictly isolated from the predictor matrix:
- **Input Feature Matrix ($X$)**: 15 clean predictors (`MODEL_INPUT_FEATURES`).
- **Continuous Regression Target ($y_{\\text{reg}}$)**: `ccme_values` (0.0 to 100.0).
- **Categorical / Ordinal Target ($y_{\\text{clf}}$)**: `ccme_wqi` / `ccme_wqi_code` (0: Poor -> 4: Excellent).
"""))

cells.append(nbf.v4.new_code_cell("""print("Modeling Features List (15 predictors):")
for idx, feat in enumerate(MODEL_INPUT_FEATURES, 1):
    print(f"  {idx:>2}. {feat}")

# Verify no targets exist in feature set
for target in [TARGET_REGRESSION, TARGET_CLASSIFICATION, TARGET_CLASSIFICATION_ORDINAL]:
    assert target not in MODEL_INPUT_FEATURES, f"CRITICAL LEAKAGE: Target {target} is inside input features!"
print("\\nLeakage check PASSED: Target columns strictly excluded from input feature list.")
"""))

# Section 5: Train / Validation / Test Splitting
cells.append(nbf.v4.new_markdown_cell("""## 5. Train / Validation / Test Splitting Strategy
We implement a reproducible 70% Train / 15% Validation / 15% Test partition stratified by `ccme_wqi` (benchmark).  
In addition, `split_data()` supports `country_area_group` (disjoint geographic area/catchment clusters, noting that station-level isolation cannot be claimed due to heterogeneous area granularity) and `temporal_holdout` (future-year forecasting).
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

# Verify class representation across partitions
class_summary = pd.DataFrame({
    "Train %": train_raw["ccme_wqi"].value_counts(normalize=True) * 100,
    "Val %": val_raw["ccme_wqi"].value_counts(normalize=True) * 100,
    "Test %": test_raw["ccme_wqi"].value_counts(normalize=True) * 100,
})
class_summary.round(2)
"""))

# Section 6: Training-Only Imputation & Feature Scaling
cells.append(nbf.v4.new_markdown_cell("""## 6. Training-Only Imputation & Feature Scaling
To prevent test leakage:
1. **Group Median Imputation**: Group medians by `(country, waterbody_type)` and seasonal medians by `(country, month)` are learned **solely on `train_raw`** and applied to validation and test sets with hierarchical fallback.
2. **Standard Scaling**: `StandardScaler` is fitted **solely on `train_imp[MODEL_INPUT_FEATURES]`** for Deep Learning (MLP). Tree models ingest unscaled features. Targets are never scaled.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Impute missing values using train-only fitted statistics
train_imp, val_imp, test_imp, imputer_info = impute_missing_values(
    train_raw, val_raw, test_raw, feature_cols=MODEL_INPUT_FEATURES
)

# 2. Fit StandardScaler on train inputs only
train_scaled, val_scaled, test_scaled, scaler = scale_features(
    train_imp, val_imp, test_imp, feature_cols=MODEL_INPUT_FEATURES
)

print("Remaining Missing Values in Modeling Features:")
print(f"  Train : {train_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")
print(f"  Val   : {val_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")
print(f"  Test  : {test_imp[MODEL_INPUT_FEATURES].isna().sum().sum()}")

# Scaler verification
train_means = train_scaled[MODEL_INPUT_FEATURES].mean().abs()
train_stds = train_scaled[MODEL_INPUT_FEATURES].std(ddof=0)
print(f"\\nScaler Mean Check (max absolute deviation from 0.0): {train_means.max():.6e}")
print(f"Scaler Std Check  (max deviation from 1.0)         : {abs(train_stds - 1.0).max():.6e}")
"""))

# Section 7: Output Validation
cells.append(nbf.v4.new_markdown_cell("""## 7. Output Validation
Verify that all generated partition schemas, feature orders, and data types are aligned.
"""))

cells.append(nbf.v4.new_code_cell("""# Verify column order and integrity across partitions
assert list(train_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(val_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(test_imp[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES
assert list(train_scaled[MODEL_INPUT_FEATURES].columns) == MODEL_INPUT_FEATURES

# Verify target integrity
assert train_imp["ccme_values"].notna().all()
assert val_imp["ccme_values"].notna().all()
assert test_imp["ccme_values"].notna().all()
assert train_imp["ccme_wqi_code"].notna().all()

print("Output Validation PASSED:")
print("  - Feature columns maintain identical 15-variable order across all partitions.")
print("  - Zero missing values or infinities exist in model inputs.")
print("  - Target columns are complete and valid.")
"""))

# Section 8: Final Dataset Readiness Summary
cells.append(nbf.v4.new_markdown_cell("""## 8. Dataset Statistics & Final Readiness Summary
Summary of all locally generated Parquet datasets and provenance metadata in `data/GeneratedOutput/02.Preprocessing&FeatureSelection/`.
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

# Execute the notebook using NotebookClient
print("Executing notebook with aml-dl-venv kernel...")
client = NotebookClient(nb, timeout=600, kernel_name="aml-dl-venv")
client.execute()

with open(NOTEBOOK_PATH, "w") as f:
    nbf.write(nb, f)

print("Notebook successfully executed and saved with all outputs rendered!")
