# Water Quality Prediction ML Project — Data Cleaning & Preprocessing Specification

> **Module**: [`notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb`](../notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb)  
> **Audited Dataset Source**: `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet`  
> **Generated Outputs**: `data/GeneratedOutput/02.Preprocessing&FeatureSelection/`  
> **Strategy**: Complete Case Analysis (Zero Synthetic Imputation Bias)  
> **Last Updated**: 2026-10-27  

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

### Stage 5: Domain-Informed Physical Bounds & Negative Concentration Rules
- **Decision**: Refined from blunt `[-5.0, 0.0)` clipping to column-specific physical and analytical detection limits:
  - `ammonia_mg_l` & `orthophosphate_mg_l`: $[-0.01, 0.0) \to 0.0$ (calibrated zero-baseline drift); values $<-0.01 \to \text{NaN}$.
  - `bod_mg_l`: $[-0.10, 0.0) \to 0.0$; values $<-0.10 \to \text{NaN}$ (neutralizing unphysical $-2.0$ mg/L anomaly to NaN).
  - `do_mg_l`: $[-0.15, 0.0) \to 0.0$ (resolving 94 records of $-0.05$ and $-0.10$ mg/L polarographic probe zero-drift in anoxic waters); values $<-0.15 \to \text{NaN}$ (hardware sentinels like $-999999.0$).
  - `temperature_c`: Valid sub-zero winter freeze readings (down to $-5.3^\circ\text{C}$ in Canada/Ireland) are preserved. Hardware dropouts ($<-10.0^\circ\text{C}$, including $-999999.0$) are converted to `np.nan`. Suspicious non-USA high temperatures ($> 45.0^\circ\text{C}$) are flagged.
  - `nitrogen_mg_l` & `nitrate_mg_l`: $[-0.05, 0.0) \to 0.0$; values $<-0.05 \to \text{NaN}$.
- **Rationale**: Physical matter concentrations cannot be negative. Applying column-specific boundaries distinguishes instrument zero-calibration noise from severe sensor corruptions or hardware dropout codes without fabricating data.

### Stage 6: Exact Duplicate Removal
- **Decision**: Drop identical duplicate records across all 14 features using `keep='first'`.
- **Rationale**: 10,005 exact duplicate records (originating exclusively from Canadian provincial ingestion merges) were confirmed. Removing duplicates prevents test set memorization and artificial density inflation in the training set.

### Stage 7: Domain-Informed Suspicious Value Flagging (Non-Destructive)
- **Decision**: Append boolean flag columns (`{col}_flag_suspicious`) rather than deleting rows for values outside plausible environmental ranges.
- **Rationale**: Deleting outliers distorts empirical distributions and discards valid high-pollution event data. Flag columns give downstream modeling teams the choice to filter, re-weight, or evaluate model calibration separately on clean vs anomalous samples.

### Stage 8: Feature Engineering, Saturation Thermodynamics & Target Integrity
- **Decision**: 
  - **Authoritative 15-Feature Input Matrix (`MODEL_INPUT_FEATURES`)**:
    - **Base Physical/Chemical (8)**: `ph`, `temperature_c`, `do_mg_l`, `bod_mg_l`, `ammonia_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l`.
    - **Engineered Geochemical/Seasonal (7)**: `np_ratio`, `bod_do_ratio`, `do_deficit`, `month`, `quarter`, `month_sin`, `month_cos`.
  - **Cyclical Month Encoding**: Generated using $\sin(2\pi(m-1)/12)$ and $\cos(2\pi(m-1)/12)$ alongside raw integer `month` and `quarter` so that January ($0^\circ$) and December ($330^\circ$) are adjacent in circular space. Missing dates yield `NaN`, subsequently imputed during train-only imputation.
  - **Safe Ratio Calculations**: `np_ratio` and `bod_do_ratio` set undefined denominators ($P \le 0$ or $DO \le 0$) to `np.nan` rather than adding arbitrary epsilon constants that distort stoichiometric chemical relationships.
  - **Signed Dissolved Oxygen Deficit & Sub-Zero Saturation**: Computed as $DO_{\text{sat}}(T) - DO$. Theoretical saturation $DO_{\text{sat}}(T)$ is calculated via the APHA Standard Methods / Elmore & Hayes polynomial and Benson & Krause (1984) USGS thermodynamic standard over the supported liquid domain [$0.0^\circ\text{C}, 45.0^\circ\text{C}$]. Valid sub-zero temperature measurements ($-10.0^\circ\text{C} \le T < 0.0^\circ\text{C}$, such as winter freeze conditions in Canada and Ireland) are evaluated at the freezing point saturation ($0.0^\circ\text{C}$, $14.652$ mg/L) representing the thermodynamic phase boundary of liquid freshwater, avoiding unphysical polynomial extrapolation. Depletion yields $D > 0$; photosynthetic supersaturation yields $D < 0$.
  - **Target Segregation**: Continuous target `ccme_values` (regression) and categorical target `ccme_wqi` / `ccme_wqi_code` (classification) are strictly separated from `MODEL_INPUT_FEATURES`. Because CCME WQI is an algebraic index of component chemical exceedances, predicting it from contemporaneous measurements is documented as surrogate index reconstruction.

### Stage 9: Complete Case Analysis, Leakage-Free Splitting, and Scaling
- **Decision**:
  - Instead of synthetic imputation (which could inject artificial values into ~34% of the dataset and bias model weights), apply **Complete Case Analysis**: filter down to records having 100% physically measured, ground-truth base features.
  - Retains **2,817,500 complete observations** across 5 nations (England: 2,127,897; USA: 405,243; Ireland: 234,491; China: 45,977; Canada: 3,892).
  - Missing values in modeling features after complete-case filtering: **0** across Train, Val, and Test.
  - Stratified 70/15/15 random split: Train (1,972,250), Val (422,625), Test (422,625).
  - `StandardScaler` fitted strictly on train input features only; targets are never scaled.

### Stage 10: Output Parquet with Snappy Compression
- **Decision**: Write partitions and repository samples to Parquet using `compression='snappy'`.
- **Rationale**: Required by project guidelines for fast random-access reads and zero-copy slicing in PyArrow, DuckDB, and Jupyter environments.

---

## 4. Before-versus-After Data Quality Summary

Execution results on the master dataset:

### Overall Dataset Shape & Complete Case Analysis

| Metric | Raw Dataset | Complete Case Partitions | Notes |
|---|---|---|---|
| **Total Rows** | 4,297,895 | 2,817,500 | 2.817M 100% complete ground-truth records |
| **Exact Duplicate Rows Dropped** | 10,005 | 0 | 100% deduplicated |
| **Hardware Sentinel Codes** | 84 (`-999999.0`) | 0 | Neutralized |
| **Scale Corrupted pH (> 14)** | 592 rows | 0 | Neutralized |
| **USA Fahrenheit Readings** | 5,852 (> 50°C) | 0 | Converted to Celsius |
| **Sub-Zero Concentrations** | 43 rows | 0 | Clipped to detection limits (0.0 mg/L) |
| **Missing Values across Features** | ~34% missing base features | **0 missing values** | Zero synthetic imputation bias |

---

## 5. Known Limitations & Modeling Recommendations

1. **Complete Case Ground Truth**:
   - Because no synthetic medians, means, or modes are injected, models train purely on real environmental observations.
2. **Imbalanced Classification Target (`ccme_wqi`)**:
   - Class ratios: ~50% *Excellent*, ~32% *Good*, ~7% *Marginal*, ~7% *Fair*, ~3% *Poor*.
3. **Phase 1 ML Performance**:
   - Tree models achieve high fidelity on continuous CCME index reconstruction ($R^2 > 0.999$, RMSE $\approx 0.52$).

---

## 6. Execution Commands & Verification

### Running the Notebook Pipeline

```bash
# Execute Preprocessing (Complete Case Analysis)
python scripts/run_notebook.py "notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb"

# Execute Baseline & Tree Models Training
python scripts/run_notebook.py "notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb"

# Execute Model Comparison & Evaluation
python scripts/run_notebook.py "notebooks/04.ModelEvaluation/01_ml_model_comparison.ipynb"
```

