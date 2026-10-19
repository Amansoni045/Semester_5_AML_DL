# AGENT CONTEXT & OPERATIONAL GUIDE

> **Project**: Project 18 — Water Quality Prediction via Gradient Boosting & Deep Neural Networks  
> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **My Role in Team**: **Role 2 — Dataset Collection & Exploratory Data Analysis (EDA)**  
> **Last Updated**: 2026-10-19  

---

## 1. Project Overview & Objective

The goal of this project is to predict surface water quality from physical and chemical measurements.  
The team is building a hybrid modeling pipeline:
1. **AML Baseline**: Tree models (Random Forest, XGBoost) on tabular water quality features.
2. **Deep Learning Model**: A regularized Multi-Layer Perceptron (MLP) to capture smooth and nonlinear chemical interactions.
3. **Comparison**: Fair evaluation using site-separated or time-separated splits reporting MAE, RMSE, $R^2$ (regression) and Macro F1 (classification).

---

## 2. My Specific Role: Dataset Collection & EDA (Role 2)

As **Role 2**, my responsibility is to:
- Collect, validate, and maintain high-quality surface water quality datasets.
- Perform comprehensive **Exploratory Data Analysis (EDA)** across structured Jupyter Notebooks.
- Inspect distributions, handle missing values, detect sensor outliers, and uncover feature correlations.
- Prepare and export clean, structured datasets in `.parquet` format into `data/GeneratedOutput/` for downstream teammates (Preprocessing, Feature Engineering, AML, DL).
- Keep simple, plain-English documentation updated in `docs/` so anyone reviewing the project understands exactly what was done and why.

---

## 3. What We Are Predicting (Targets $y$)

We use the Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**):

| Target Column | Type | Values / Categories | Evaluation Metrics |
|---|---|---|---|
| `CCME_Values` | **Continuous (Regression)** | Range: `0.0` to `100.0` | MAE, RMSE, $R^2$ |
| `CCME_WQI` | **Categorical (Classification)** | `Excellent` ($95-100$), `Good` ($80-94$), `Fair` ($65-79$), `Marginal` ($45-64$), `Poor` ($0-44$) | Macro F1, Accuracy, Confusion Matrix, Calibration |

---

## 4. Input Features ($X$) & Metadata

### Physical & Chemical Parameters ($X$)
1. `pH (ph units)`: Acidity/alkalinity standard (Standard: 8.5)
2. `Dissolved Oxygen (mg/l)`: Oxygen available for aquatic life (Standard: 14.0)
3. `Temperature (cel)`: Water temperature (Standard: 25.0)
4. `Biochemical Oxygen Demand (mg/l)`: Organic matter breakdown demand (Standard: 6.0)
5. `Ammonia (mg/l)`: Toxic breakdown product (Standard: 0.5)
6. `Orthophosphate (mg/l)`: Soluble reactive phosphorus (Standard: 0.05)
7. `Nitrogen (mg/l)`: Total nitrogen nutrient load (Standard: 20.0)
8. `Nitrate (mg/l)`: Oxidized nitrogen compound (Standard: 45.0)

### Spatio-Temporal Metadata
- `Country`: Australia, Canada, China, England, India, Ireland, USA
- `Area`: Monitoring station, river basin, or catchment identifier
- `Waterbody Type`: River, Stream, Lake, Marine, Estuary, Transitional
- `Date`: Sampling date formatted as `DD-MM-YYYY`

---

## 5. File & Directory Layout

```
Semester_5_AML_DL/
├── AGENT.md                                      # Master AI context & instructions (this file)
├── README.md                                     # Public project README
├── data/
│   ├── Dataset/
│   │   ├── Parquet/
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # Master dataset (4,297,895 rows, Snappy)
│   │   │   └── Country-Wise Data/
│   │   │       ├── Australia_dataset.parquet     # 87 rows
│   │   │       ├── Canada_dataset.parquet        # 3,995 rows
│   │   │       ├── China_dataset.parquet         # 45,997 rows
│   │   │       ├── England_dataset.parquet       # 2,129,198 rows
│   │   │       ├── India_dataset.parquet         # 3,633 rows
│   │   │       ├── Ireland_dataset.parquet       # 235,019 rows
│   │   │       └── USA_dataset.parquet           # 1,879,966 rows
│   │   └── CSV/                                  # Raw CSVs (ignored by .gitignore)
│   ├── Data Collection Sources/
│   │   ├── DataCollectionSources.parquet         # Source provenance table (153 sources)
│   │   └── DataCollectionSources.xlsx            # Source Excel reference (ignored by .gitignore)
│   └── GeneratedOutput/                          # ALL output data from notebooks goes here (.parquet)
├── nootbooks/                                    # EDA and modeling Jupyter Notebooks
└── docs/                                         # Easy English project & EDA documentation
```

---

## 6. Golden Rules for AI Operations

1. **Parquet Only**:
   - Always load from and save to `.parquet` format.
   - **Always use `compression='snappy'`** when saving `.parquet` files so they open smoothly in VS Code's **Parquetly** extension without codec errors.
   - Never output `.csv` or `.xlsx` files into Git.
2. **Data Output Location**:
   - Any data generated, cleaned, or filtered by notebooks MUST be saved into `data/GeneratedOutput/<filename>.parquet`.
3. **Notebook Organization**:
   - Place all Jupyter notebooks inside `nootbooks/`.
   - Name them with sequential numbers and descriptive titles (e.g., `01_dataset_overview_and_summary.ipynb`, `02_missing_values_and_imputation_analysis.ipynb`, etc.).
   - Each notebook must be self-contained: loads data from `data/`, executes analysis, and exports resulting data to `data/GeneratedOutput/`.
4. **Documentation in Easy English**:
   - Write documentation in `docs/` using simple, clear, conversational English.
   - Maintain a running log of what was analyzed, key insights found, and recommendations for the modeling team.
5. **Always Update `AGENT.md`**:
   - Whenever a new task is completed, a new notebook is created, or new data is produced, immediately update the **Task Log & Current Progress** section in this file.

---

## 7. The 6-Stage Pure EDA Architecture (Role 2 Scope)

Since our responsibility is strictly **Role 2: Dataset Collection & EDA**, our notebooks focus 100% on exploratory analysis, data profiling, anomaly detection, and geochemical relationships. Data splitting and preprocessing pipelines belong to **Role 3: Data Preprocessing & Feature Engineering**.

The 6 pure EDA notebooks in `nootbooks/`:

- [ ] **Notebook 1: `01_data_architecture_and_profiling.ipynb`**
  - High-dimensional profiling, memory footprints, country distributions, waterbody breakdowns, sampling frequencies.
  - Output: `data/GeneratedOutput/01_dataset_summary_profile.parquet`
- [ ] **Notebook 2: `02_sensor_sparsity_and_missingness.ipynb`**
  - Sensor missingness patterns, co-occurrence matrices, dendrogram clustering, and country-level sensor protocols.
  - Output: `data/GeneratedOutput/02_missingness_patterns.parquet`
- [ ] **Notebook 3: `03_geochemical_physics_and_anomalies.ipynb`**
  - Cleaning sensor error sentinels (`-999999.0`, `pH 67115.0`), Henry's Law DO-Temperature saturation bounds, detection limit offsets.
  - Output: `data/GeneratedOutput/03_cleaned_physically_validated.parquet`
- [ ] **Notebook 4: `04_bivariate_and_multivariate_correlations.ipynb`**
  - Pearson & Spearman correlation matrices, $N:P$ nutrient ratios, BOD-DO oxygen sag curves.
  - Output: `data/GeneratedOutput/04_feature_correlations.parquet`
- [ ] **Notebook 5: `05_temporal_seasonality_and_spatial_trends.ipynb`**
  - Multi-decade historical evolution (1940–2023), seasonal cyclic dynamics, waterbody comparisons (rivers, streams, lakes, coastal).
  - Output: `data/GeneratedOutput/05_temporal_spatial_aggregates.parquet`
- [ ] **Notebook 6: `06_target_behavior_and_leakage_audit.ipynb`**
  - Target score analysis (`CCME_Values` spike at 100), class imbalance (Excellent 50.1% vs Poor 3.0%), mathematical leakage audit.
  - Formulates **70 / 15 / 15 Splitting Guidelines** as the formal handoff to Role 3.
  - Output: `data/GeneratedOutput/06_target_audit_and_splitting_guidelines.parquet`

---

## 8. Task Log & Current Progress

| Date | Step | Description | Output Files | Status |
|---|---|---|---|---|
| 2026-10-19 | Dataset Collection | Harvested 4.29M rows across 7 nations (England, USA, Ireland, China, Canada, India, Australia) with exact CCME WQI | `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` | Completed |
| 2026-10-19 | Parquet Compatibility | Re-saved all Parquet datasets with `snappy` compression for full Parquetly viewing | All `.parquet` files in `data/` | Completed |
| 2026-10-19 | Agent Context Setup | Created `AGENT.md`, established workflow between `nootbooks/`, `docs/`, and `data/GeneratedOutput/` | `AGENT.md`, `docs/` | Completed |
| 2026-10-19 | Deep Data Profiling | Profiled all 4.3M rows; identified sentinels (-999999, pH 67115), sensor missingness, and target distributions | Terminal execution logs | Completed |
| 2026-10-19 | Comprehensive EDA Plan | Designed 7-stage modular EDA pipeline and 70/15/15 splitting strategy in `docs/01_comprehensive_eda_plan.md` | `docs/01_comprehensive_eda_plan.md` | Completed |

