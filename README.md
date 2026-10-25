# Project 18: Water Quality Prediction via Gradient Boosting & Deep Neural Networks

> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Master Dataset**: 4,297,895 records × 14 columns across 7 countries (1940–2023)  
> **Target Index**: Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**)  

---

## 1. Project Overview & The "Hybrid" Logic

This project predicts surface water quality from physical and chemical measurements using a complementary machine learning and deep learning architecture:

- **AML Baseline (Tree Models)**: Train **XGBoost** and **Random Forest** models on physical and chemical features (`pH`, `Dissolved Oxygen`, `Temperature`, `BOD`, `Ammonia`, `Orthophosphate`, `Nitrogen`, `Nitrate`). Tree models offer strong performance on tabular data and robust splitting behavior.
- **Deep Learning Model (Neural Network)**: Train a regularized **Multi-Layer Perceptron (MLP)** with dual heads (regression for continuous `CCME_Values` and classification for `CCME_WQI` classes) to capture smooth combinations and calibrated uncertainty when multiple sensors interact non-linearly.
- **Fair Evaluation Protocol**: Use site-stratified and temporal holdout splits to prevent spatial and temporal memorization leakage, reporting **MAE, RMSE, $R^2$** for regression and **Macro F1 & Calibration** for classification.

---

## 2. Quickstart & Automated Pipeline Runner

You can set up the environment and run the entire EDA pipeline with a single command:

```bash
# Option A: One-line Universal Python Runner (runs environment setup first, then runs all 6 EDA notebooks):
python3 scripts/run_eda_pipeline.py

# Option B: Run environment setup script explicitly, then run pipeline:
./scripts/setup_environment.sh
.venv/bin/python scripts/run_eda_pipeline.py
```

> [!NOTE]  
> `scripts/run_eda_pipeline.py` automatically checks if `.venv` exists. If run with system Python, it runs `scripts/setup_environment.sh` first to ensure `.venv` and all packages from `requirements.txt` are installed, registers the Jupyter kernel (`aml-dl-venv`), bootstraps execution into `.venv`, runs all 6 EDA notebooks sequentially, and audits generated Parquet files in `data/GeneratedOutput/01.EDA/`.
> For full setup details, see the [Environment Setup Guide](docs/environment_setup.md).

---

## 3. Global Dataset Architecture

Our harmonized dataset contains **4,297,895 surface water monitoring observations** across **64,665 unique stations** spanning 1940 to 2023.

### Country Breakdown

| Country | Records | Year Range | Primary Source Networks |
|---|---|---|---|
| **England** | 2,129,198 | 2000–2023 | Environment Agency UK (`data.gov.uk`) — Complete 8-sensor panels |
| **USA** | 1,879,966 | 1940–2023 | USGS NWIS Telemetry (FL, CA, TX, NY, WA, NC) + EPA Water Quality Portal |
| **Ireland** | 235,019 | 1940–2023 | EPA Ireland (`catchments.ie` / `data.europa.eu`) — Complete panels |
| **China** | 45,997 | 2001–2017 | Ministry of Ecology and Environment / `data.gov.hk` |
| **Canada** | 3,995 | 1968–2021 | Open Canada / ECCC / Lac Hertel High-Frequency Sonde Network |
| **India** | 3,633 | 2003–2023 | Central Pollution Control Board (CPCB) National River Network |
| **Australia** | 87 | 2014–2015 | Torrens River & Lake Water Quality (`data.sa.gov.au`) |
| **Total** | **4,297,895** | **1940–2023** | **Harmonized 14-Column Master Dataset** |

---

## 4. Target Schema & Feature Definitions

- **Metadata / Context**:
  - `Country`: Country of origin (string)
  - `Area`: Monitoring station location / river / catchment area (string)
  - `Waterbody Type`: River, Stream, Effluent, Lake, Estuary, Marine (string)
  - `Date`: Sampling date formatted as `DD-MM-YYYY` (string)
- **Authoritative Model Input Features ($X$, 15 Features)**:
  - *Base Physical & Chemical Parameters (8)*:
    - `ph`: Negative log hydrogen activity (pH units)
    - `temperature_c`: Water temperature (Celsius)
    - `do_mg_l`: Dissolved Oxygen concentration (mg/L)
    - `bod_mg_l`: 5-day Biochemical Oxygen Demand (mg/L)
    - `ammonia_mg_l`: Ammonium / ammonia concentration (mg/L)
    - `orthophosphate_mg_l`: Dissolved reactive phosphorus (mg/L)
    - `nitrogen_mg_l`: Total / dissolved nitrogen (mg/L)
    - `nitrate_mg_l`: Nitrate ion concentration (mg/L)
  - *Domain-Engineered Geochemical & Seasonal Predictors (7)*:
    - `np_ratio`: Stoichiometric Total Nitrogen to Orthophosphate ratio
    - `bod_do_ratio`: Organic loading relative to available dissolved oxygen
    - `do_deficit`: Signed oxygen deficit $DO_{\text{sat}}(T) - DO$ (sub-zero bounded at $0^\circ\text{C}$ freezing point saturation)
    - `month`: Integer calendar month ($1 \dots 12$)
    - `quarter`: Integer calendar quarter ($1 \dots 4$)
    - `month_sin`: Cyclical seasonal transform $\sin(2\pi(m-1)/12)$
    - `month_cos`: Cyclical seasonal transform $\cos(2\pi(m-1)/12)$
- **Target Indices ($y$, Strictly Excluded from $X$)**:
  - `ccme_values`: Canadian Council of Ministers of the Environment Water Quality Index score (`0.0` – `100.0`, continuous regression target)
  - `ccme_wqi`: Categorical classification (`Poor` < `Marginal` < `Fair` < `Good` < `Excellent`)
  - `ccme_wqi_code`: Ordinal integer classification label (`0` $\dots$ `4`)

---

## 5. Exploratory Data Analysis & Preprocessing

The exploratory analysis is compiled in:  
**[Exploratory Data Analysis Comprehensive Report](docs/EDA_REPORT.md)** and **[Preprocessing Specification](docs/PREPROCESSING_SPECIFICATION.md)**.

All notebooks are organized in **[`notebooks/`](notebooks/)**:

| Notebook | Focus & Tasks | Outputs |
|---|---|---|
| **[`notebooks/01.EDA/00_initial_dataset_audit.ipynb`](notebooks/01.EDA/00_initial_dataset_audit.ipynb)** | Initial inspection, profiling, and anomaly diagnostics | `reports/dataset_audit_report.md` |
| **[`notebooks/01.EDA/01_data_architecture_and_profiling.ipynb`](notebooks/01.EDA/01_data_architecture_and_profiling.ipynb)** | Multi-country data architecture & sampling frequency | `01_dataset_summary_profile.parquet` |
| **[`notebooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb`](notebooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb)** | Missingness mechanisms & nutrient co-occurrence ($r > 0.99$) | `02_missingness_patterns.parquet` |
| **[`notebooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb`](notebooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb)** | Sensor sentinel code detection & Henry's Law DO saturation | `03_cleaned_physically_validated.parquet` |
| **[`notebooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb`](notebooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb)** | Correlation heatmaps, N:P ratios & Streeter-Phelps curves | `04_feature_correlations.parquet` |
| **[`notebooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb`](notebooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb)** | Historical trends (1970–2023) & seasonal cycles | `05_temporal_spatial_aggregates.parquet` |
| **[`notebooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb`](notebooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb)** | Target distribution, class imbalance, and leakage audit | Target analysis summaries |
| **[`notebooks/02.Preprocessing&FeatureSelection/01_feature_analysis.ipynb`](notebooks/02.Preprocessing&FeatureSelection/01_feature_analysis.ipynb)** | Feature distributions, correlation heatmaps, VIF multicollinearity, and leakage audit | Feature analysis profiles & recommendations |
| **[`notebooks/02.Preprocessing&FeatureSelection/02_data_preprocessing_and_feature_engineering.ipynb`](notebooks/02.Preprocessing&FeatureSelection/02_data_preprocessing_and_feature_engineering.ipynb)** | Cleaning, 15-feature engineering, 70/15/15 split, imputation, scaling | `train.parquet`, `val.parquet`, `test.parquet`, scaled files |
| **[`notebooks/02.Preprocessing&FeatureSelection/03_model_ready_validation.ipynb`](notebooks/02.Preprocessing&FeatureSelection/03_model_ready_validation.ipynb)** | Model-ready validation, strict target quarantine check, spatial & temporal leakage audit | Verification sign-off & data readiness |
| **[`notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb`](notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb)** | Phase 1 ML: Mean baseline, Random Forest, XGBoost training & prediction handoff | `random_forest_model.joblib`, `xgboost_model.json`, `tree_val_predictions.parquet` |

---

## 6. Repository Layout

```
Semester_5_AML_DL/
├── AGENT.md                                      # Project overview & roadmap
├── README.md                                     # Main project guide
├── requirements.txt                              # Python package dependencies
├── scripts/
│   ├── setup_environment.sh                      # Environment setup script (.venv)
│   ├── run_eda_pipeline.py                       # Automated EDA runner
│   └── run_notebook.py                           # CLI notebook runner
├── src/
│   ├── __init__.py                               # Package exports
│   ├── preprocessing.py                          # Data cleaning, feature engineering, split & scaling
│   └── data_audit.py                             # Inspection & anomaly auditing utilities
├── tests/
│   ├── __init__.py
│   └── test_preprocessing.py                     # Unit & integration test suite (65+ tests)
├── data/
│   ├── Dataset/
│   │   ├── Parquet/                              # Tracked parquet master datasets
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # Master combined dataset (4.29M rows)
│   │   │   └── Country-Wise Data/                # Country-specific parquet files
│   │   └── CSV/                                  # Raw CSV archive (ignored by git)
│   │       ├── Combined Data/
│   │       └── Country-Wise Data/
│   ├── Data Collection Sources/
│   └── GeneratedOutput/
│       ├── 01.EDA/                               # EDA summary parquets
│       ├── 02.Preprocessing&FeatureSelection/    # Model training datasets (train, val, test, scaled)
│       ├── 03.ModelTraining/                     # Tree models & MLP checkpoints
│       └── 04.ModelEvaluation/                   # Metric tables & SHAP artifacts
├── notebooks/
│   ├── 01.EDA/                                   # EDA notebooks (00 through 06)
│   ├── 02.Preprocessing&FeatureSelection/        # Preprocessing & feature engineering notebook
│   ├── 03.ModelTraining/                         # Tree baseline & neural net training notebooks
│   └── 04.ModelEvaluation/                       # Evaluation & explainability notebooks
├── reports/
│   ├── dataset_audit_report.md
│   └── figures/
└── docs/
    ├── EDA_REPORT.md
    ├── PREPROCESSING_SPECIFICATION.md
    └── environment_setup.md
```

---

## 7. Viewing Parquet Files in VS Code

All `.parquet` files use standard **`SNAPPY`** compression for fast I/O and broad compatibility.

To inspect, filter, and plot any Parquet file directly inside VS Code:
1. Open the **Extensions** panel (`Cmd + Shift + X` on macOS / `Ctrl + Shift + X` on Windows/Linux).
2. Install **Parquetly** (**Parquetly - Parquet Viewer and Editor**).
3. Click any `.parquet` file in `data/` or right-click &rarr; **Reopen Editor With...** &rarr; **Parquetly**.
