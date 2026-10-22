# Water Quality Prediction ML Project — Data Cleaning & Preprocessing Specification

> **Module**: [`src/preprocessing.py`](../src/preprocessing.py)  
> **Test Suite**: [`tests/test_preprocessing.py`](../tests/test_preprocessing.py)  
> **Audited Dataset Source**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet`  
> **Cleaned Dataset Target**: `data/GeneratedOutput/02.Preprocessing&FeatureSelection/cleaned_water_quality.parquet`  
> **Last Updated**: 2026-10-22  

---

## 1. Overview & Purpose

This document provides a comprehensive operational and scientific record of the data cleaning pipeline implemented for **Project 18: Water Quality Prediction via Gradient Boosting and Deep Neural Networks**.

The pipeline ingests the combined multi-country water quality observations (4,297,895 rows across 7 countries) and executes a deterministic, reproducible, 9-stage sequence to resolve telemetry error codes, sensor scale anomalies, negative analytical detection limits, exact duplicates, and inconsistent units, while preserving schema integrity and preventing target/test leakage.

---

## 2. Input Files & Schema

### Raw Input Datasets
- **Primary Raw File**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` (665.5 MB uncompressed, Snappy parquet).
- **Secondary Format Support**: `Combined_dataset.csv` (compatible via CLI and preprocessor auto-discovery).

### Raw Column Specification & Target Aliases

| Raw Column Name | Inferred Raw Dtype | Standardised Alias | Semantic Data Type | Description |
|---|---|---|---|---|
| `Country` | string / object | `country` | string | Sovereign nation of sampling station (7 countries) |
| `Area` | string / object | `area` | string | Sampling station, river name, or geographic waterbody |
| `Waterbody Type` | string / object | `waterbody_type` | string | Waterbody classification (River, Lake, Stream, etc.) |
| `Date` | string (`DD-MM-YYYY`) | `date` | `datetime64[ns]` | Timestamp of sample collection |
| `Ammonia (mg/l)` | `float64` | `ammonia_mg_l` | `float64` | Ammonium / ammonia concentration |
| `Biochemical Oxygen Demand (mg/l)` | `float64` | `bod_mg_l` | `float64` | 5-day Biochemical Oxygen Demand |
| `Dissolved Oxygen (mg/l)` | `float64` | `do_mg_l` | `float64` | Dissolved oxygen concentration |
| `Orthophosphate (mg/l)` | `float64` | `orthophosphate_mg_l` | `float64` | Dissolved reactive phosphorus |
| `pH (ph units)` | `float64` | `ph` | `float64` | Negative log hydrogen ion activity |
| `Temperature (cel)` | `float64` | `temperature_c` | `float64` | Water temperature in Celsius |
| `Nitrogen (mg/l)` | `float64` | `nitrogen_mg_l` | `float64` | Total / dissolved nitrogen |
| `Nitrate (mg/l)` | `float64` | `nitrate_mg_l` | `float64` | Nitrate ion concentration |
| `CCME_Values` | `float64` | `ccme_values` | `float64` | Continuous index (0.0 – 100.0) — **Regression Target** |
| `CCME_WQI` | string / object | `ccme_wqi` | `Categorical` | Ordered grade (Poor < Marginal < Fair < Good < Excellent) — **Classification Target** |

---

## 3. Cleaning Decisions & Rationale

Every transformation addresses a specific empirical anomaly identified during the exploratory dataset audit:

### Stage 1: Column Name Standardisation
- **Decision**: Map long, unit-bracketed raw column names to concise, snake_case aliases (`ph`, `temperature_c`, `do_mg_l`, etc.).
- **Rationale**: Prevents quoting bugs across Polars, DuckDB, PyTorch, and XGBoost. The complete bi-directional mapping is permanently preserved in DataFrame `.attrs["column_mapping"]`.

### Stage 2: Semantic Data Type Casting
- **Decision**: 
  - Parse `Date` strings using `pd.to_datetime(dayfirst=True)` to `datetime64[ns]`.
  - Cast `ccme_wqi` to an ordered `pd.CategoricalDtype(categories=["Poor", "Marginal", "Fair", "Good", "Excellent"], ordered=True)`.
  - Strip whitespace from categorical strings and normalize blank or null-like markers (`"nan"`, `"null"`, `"none"`, `""`) to `np.nan`.
- **Rationale**: Preserves ordinal ranking for classification loss functions (Focal Loss / Ordinal Cross-Entropy) and allows proper seasonal feature extraction.

### Stage 3: Telemetry Hardware Sentinel Replacement
- **Decision**: Replace known hardware failure sentinels (`-999999.0`, `-596.0`, `-99.9`, `-80.0`, `-1.0` in pH) and scale-corruption anomalies (`pH > 14.0`, including `67115.0`) with `np.nan`.
- **Rationale**: Natural water pH cannot exceed 14.0 or drop below 0.0 under open atmospheric conditions. Hardware transmitters assign `-999999.0` when probes disconnect. Leaving these values would catastrophically bias gradient computations and tree split points.

### Stage 4: USA Temperature Unit Conversion
- **Decision**: Convert temperature readings for `country == 'USA'` where `temperature_c > 50.0` from Fahrenheit to Celsius using $(F - 32) \times \frac{5}{9}$.
- **Rationale**: Audit revealed 5,852 USA records with temperatures exceeding 50°C (up to 104°F). Surface water in continental USA cannot reach 50°C–100°C without boiling. Post-conversion, all values cleanly map to the plausible range of 10°C–36°C. Non-USA high temperatures (e.g., England industrial effluent discharges) were left untouched based on station site metadata.

### Stage 5: Sub-Detection-Limit (SDL) Negative Clipping
- **Decision**: Clip negative chemical concentrations in the range $[-5.0, 0.0)$ to `0.0` for `ammonia_mg_l`, `bod_mg_l`, `do_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, and `nitrate_mg_l`.
- **Rationale**: Concentrations of physical matter cannot be negative. These values are standard analytical chemistry baseline noise around sensor zero-calibration points. In contrast, `temperature_c` is permitted to retain sub-zero values for freeze conditions in Canada and Ireland.

### Stage 6: Exact Duplicate Removal
- **Decision**: Drop identical duplicate records across all 14 features using `keep='first'`.
- **Rationale**: 10,005 exact duplicate records (originating exclusively from Canadian provincial ingestion merges) were confirmed. Removing duplicates prevents test set memorization and artificial density inflation in the training set.

### Stage 7: Domain-Informed Suspicious Value Flagging (Non-Destructive)
- **Decision**: Append boolean flag columns (`{col}_flag_suspicious`) rather than deleting rows for values outside plausible environmental ranges.
- **Rationale**: Deleting outliers distorts empirical distributions and discards valid high-pollution event data. Flag columns give downstream modeling teams the choice to filter, re-weight, or evaluate model calibration separately on clean vs anomalous samples.

### Stage 8: Documented Missingness Strategy (No Global Mean/Median Imputation)
- **Decision**: Document per-column missingness mechanisms without performing naive imputation on the raw dataset.
- **Rationale**: Nutrients exhibit 34.2% Missing At Random (MAR) synchrony. Imputing globally before splitting causes severe data leakage. All imputations (KNN within Country $\times$ Waterbody Type or physics-informed DO estimation) must be fitted strictly on the 70% training split.

### Stage 9: Output Parquet with Snappy Compression
- **Decision**: Write cleaned data to Parquet using `compression='snappy'`.
- **Rationale**: Required by project guidelines for fast random-access reads and zero-copy slicing in PyArrow, DuckDB, and the VS Code Parquetly extension.

---

## 4. Before-versus-After Data Quality Summary

Execution results on the full 4.29M-row dataset:

### Overall Dataset Shape & Duplication

| Metric | Raw Dataset | Cleaned Dataset | Delta / Impact |
|---|---|---|---|
| **Total Rows** | 4,297,895 | 4,287,890 | -10,005 rows (-0.2328%) |
| **Total Columns** | 14 | 23 | +9 boolean flag columns |
| **Exact Duplicate Rows** | 10,005 | 0 | 10,005 duplicates dropped |
| **Hardware Sentinel Codes** | 84 (`-999999.0` across T, pH, DO) | 0 | 100% neutralized to NaN |
| **Scale Corrupted pH (> 14)** | 592 rows (max 67,115.0) | 0 | 100% neutralized to NaN |
| **USA Fahrenheit Readings** | 5,852 (> 50°C) | 0 | 100% converted to Celsius |
| **Sub-Zero Concentrations** | 43 rows | 0 | Clipped to 0.0 mg/L |

### Per-Column Missing Value Analysis

| Feature | Raw Missing Count | Raw Missing % | Cleaned Missing Count | Cleaned Missing % | Net Missing Delta | Root Cause of Delta |
|---|---|---|---|---|---|---|
| `country` | 0 | 0.000% | 0 | 0.000% | 0 | Maintained 100% complete |
| `area` | 0 | 0.000% | 182 | 0.004% | +182 | Unnamed blank strings normalized to NaN |
| `waterbody_type` | 0 | 0.000% | 0 | 0.000% | 0 | Maintained 100% complete |
| `date` | 0 | 0.000% | 0 | 0.000% | 0 | Successfully parsed to `datetime64` |
| `ammonia_mg_l` | 1,469,831 | 34.199% | 1,469,797 | 34.278% | -34 | 34 duplicate missing rows removed |
| `bod_mg_l` | 1,466,334 | 34.117% | 1,466,331 | 34.197% | -3 | 3 duplicate missing rows removed |
| `do_mg_l` | 1,140,980 | 26.547% | 1,140,980 | 26.609% | 0 | 1 sentinel replaced + 1 dup dropped |
| `orthophosphate_mg_l` | 1,469,831 | 34.199% | 1,469,797 | 34.278% | -34 | 34 duplicate missing rows removed |
| `ph` | 1,201,075 | 27.946% | 1,201,671 | 28.025% | +596 | +592 high pH + 4 sentinels replaced |
| `temperature_c` | 6,558 | 0.153% | 6,639 | 0.155% | +81 | +81 extreme telemetry sentinels replaced |
| `nitrogen_mg_l` | 1,469,831 | 34.199% | 1,469,797 | 34.278% | -34 | 34 duplicate missing rows removed |
| `nitrate_mg_l` | 1,466,431 | 34.120% | 1,466,428 | 34.199% | -3 | 3 duplicate missing rows removed |
| `ccme_values` | 0 | 0.000% | 0 | 0.000% | 0 | Continuous target 100% complete |
| `ccme_wqi` | 0 | 0.000% | 0 | 0.000% | 0 | Categorical target 100% complete |

---

## 5. Known Limitations & Downstream Modeling Recommendations

1. **Nutrient Missingness Synchrony**:
   - `ammonia_mg_l`, `orthophosphate_mg_l`, and `nitrogen_mg_l` share identical missingness patterns (34.28%). They represent synchronized lab test omissions rather than sensor dropouts. Tree models handle these naturally via surrogate splits; MLP architectures must use group-imputed values fitted strictly on the train split.
2. **Sparse Site Identifiers (`area`)**:
   - 182 rows contain unrecorded or blank site names. These samples have complete regional (`country`) and waterbody (`waterbody_type`) metadata, allowing them to remain valid for regional-level modeling.
3. **Imbalanced Classification Target (`ccme_wqi`)**:
   - Class ratios are strongly skewed: ~50.1% *Excellent* vs ~3.0% *Poor*. Downstream models should utilize class-weighted cross-entropy loss or focal loss to prevent under-predicting degraded water quality classes.

---

## 6. Execution Commands & Verification

### Running the Preprocessing Pipeline via CLI

```bash
# Execute cleaning pipeline on the default combined dataset
python -m src.preprocessing \
    --input "data/Dataset/Parquet/Combined Data/Combined_dataset.parquet" \
    --output "data/GeneratedOutput/02.Preprocessing&FeatureSelection/cleaned_water_quality.parquet"

# Fast trial on first 50,000 rows
python -m src.preprocessing \
    --input "data/Dataset/Parquet/Combined Data/Combined_dataset.parquet" \
    --output "data/GeneratedOutput/02.Preprocessing&FeatureSelection/cleaned_sample.parquet" \
    --nrows 50000
```

### Running Test Suite

```bash
# Run all unit, corner-case, and fast-fixture tests
pytest tests/test_preprocessing.py -v

# Run full integration test suite (including real dataset schema validation)
pytest tests/test_preprocessing.py -v -k "not test_real_dataset_full_run_schema"
```
