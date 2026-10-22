# AGENT MASTER CONTEXT & OPERATIONAL GUIDE

> **Project**: Project 18 — Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Scope**: Master AI Instructions across the entire end-to-end project (All 6 Roles)  
> **Last Updated**: 2026-10-19  

---

## 1. Project Overview & The Hybrid Machine Learning / Deep Learning Logic

This project aims to predict documented surface water quality indices and categorical classes from physical and chemical measurements across freshwater and marine environments.

### The AML / DL Hybrid Logic
1. **AML Baseline (Tree Models)**:
   - Algorithms: **Random Forest** and **XGBoost**.
   - Input Features: `pH`, `Dissolved Oxygen`, `Temperature`, `Biochemical Oxygen Demand`, `Ammonia`, `Orthophosphate`, `Nitrogen`, `Nitrate`.
   - Strengths: Strong performance on tabular data and robust to non-linear step-functions.
2. **Deep Learning Model (Neural Network)**:
   - Architecture: Small, regularized **Multi-Layer Perceptron (MLP)**.
   - Purpose: Capture smooth combinations and calibrated uncertainty when multiple chemical sensors interact non-linearly.
   - Dual-Head Setup: A regression head for the continuous index (`CCME_Values`) and a classification head for the categorical grade (`CCME_WQI`).
3. **Fair Evaluation Protocol**:
   - Use site-separated or time-separated splits (avoid naive random splitting).
   - Regression Metrics: Mean Absolute Error (**MAE**), Root Mean Squared Error (**RMSE**), and Coefficient of Determination (**$R^2$**).
   - Classification Metrics: **Macro F1**, Class-wise Accuracy, Confusion Matrix, and Brier Score / Calibration.

---

## 2. Project Team Division (All 6 Roles)

This project is divided among team members across 6 collaborative areas:

| Role # | Functional Area | Primary Responsibilities | Current Status |
|---|---|---|---|
| **Role 1** | **Literature Review & Research** | Research water quality prediction literature (Breiman 2001, Goodfellow 2016), compare methods, identify research gaps. | In Progress |
| **Role 2** | **Dataset Collection & EDA** | Find multi-country datasets, handle missing values, clean sensor sentinels, perform full EDA across 6 notebooks, export summaries. | **Completed** |
| **Role 3** | **Data Preprocessing & Feature Engineering** | Clean/impute data, scale inputs for MLP, engineer features, create **70 / 15 / 15 Train/Val/Test** splits. | **Up Next** |
| **Role 4** | **Machine Learning Models** | Implement and tune Random Forest and XGBoost baselines; feature importance with SHAP. | Queued |
| **Role 5** | **Deep Learning, Theory & Evaluation** | Train regularized MLP neural net, implement regression/classification heads, evaluate calibration & error metrics. | Queued |
| **Role 6** | **Integration, GitHub & Documentation** | Maintain repository integrity, consolidate final report, presentation, and viva preparation. | Ongoing |

---

## 3. Target Variables & What We Are Predicting

We use the Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**):

| Target Column | Nature | Values / Range | Purpose / Loss Function |
|---|---|---|---|
| `CCME_Values` | **Continuous (Regression)** | `0.0` to `100.0` | Predict precise water quality score; Loss: MSE / Huber Loss. |
| `CCME_WQI` | **Categorical (Classification)** | `Excellent` ($95-100$), `Good` ($80-94$), `Fair` ($65-79$), `Marginal` ($45-64$), `Poor` ($0-44$) | Multi-class classification; Loss: Class-weighted Cross-Entropy or Focal Loss (addresses 50.1% Excellent vs 3.0% Poor imbalance). |

---

## 4. Input Features ($X$) & Metadata

### Chemical & Physical Predictors ($X$)
1. `pH (ph units)`: Standard limit: 8.5 (Physically bounded between 3.0 and 12.0)
2. `Dissolved Oxygen (mg/l)`: Standard limit: 14.0 (Henry's Law saturation checked)
3. `Temperature (cel)`: Standard limit: 25.0°C
4. `Biochemical Oxygen Demand (mg/l)`: Standard limit: 6.0 (Organic pollution load)
5. `Ammonia (mg/l)`: Standard limit: 0.5 (Toxic breakdown product)
6. `Orthophosphate (mg/l)`: Standard limit: 0.05 (Key driver of eutrophication, $r = -0.66$)
7. `Nitrogen (mg/l)`: Standard limit: 20.0 (Total nutrient load)
8. `Nitrate (mg/l)`: Standard limit: 45.0 (Oxidized nitrogen runoff)

### Metadata Attributes
- `Country`: England (2.13M), USA (1.88M), Ireland (235k), China (46k), Canada (4.0k), India (3.6k), Australia (87).
- `Area`: 64,665 unique monitoring stations / river catchments.
- `Waterbody Type`: River (43.2%), Stream (34.1%), Effluent (14.0%), Lake (3.6%), Estuary, Marine, etc.
- `Date`: Sampling date formatted as `DD-MM-YYYY` (1940 to 2023).

---

## 5. Global Repository Layout

```
Semester_5_AML_DL/
├── AGENT.md                                      # Master guide for the entire project
├── README.md                                     # Main project guide
├── requirements.txt                              # Root dependencies
├── scripts/
│   ├── setup_environment.sh                      # Environment setup script (.venv)
│   ├── run_eda_pipeline.py                       # Automated EDA runner
│   └── build_and_run_audit_notebook.py           # Audit notebook builder
├── data/
│   ├── Dataset/
│   │   ├── Parquet/
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # 4,297,895 records × 14 columns (Snappy)
│   │   │   └── Country-Wise Data/                # Country-specific parquet files
│   │   └── CSV/                                  # Raw CSV archive (ignored by git)
│   ├── Data Collection Sources/
│   └── GeneratedOutput/
│       ├── 01.EDA/                               # EDA summary parquets
│       ├── 02.Preprocessing&FeatureSelection/    # Cleaned and training-ready datasets (train, val, test, scaled)
│       ├── 03.ModelTraining/                     # Tree baselines & MLP checkpoints
│       └── 04.ModelEvaluation/                   # Metric tables & SHAP artifacts
├── notebooks/
│   ├── 01.EDA/                                   # EDA notebooks (00 through 06)
│   ├── 02.Preprocessing&FeatureSelection/        # Preprocessing & feature engineering notebook
│   ├── 03.ModelTraining/                         # Baseline tree & DL training notebooks
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

## 6. Golden Rules for Pipeline Operations

1. **Parquet with Snappy Only**:
   - Always read and write `.parquet` files using **`compression='snappy'`**.
   - Ensures universal compatibility with DuckDB, Polars, and PyArrow.
   - Never commit `.csv` or `.xlsx` files into Git.
2. **Data Leakage Safeguards**:
   - Do not allow models to predict `CCME_Values` from the exact constituent inputs without acknowledging mathematical leakage.
   - Impute missing data **strictly within the training split** (never compute global medians across test sets).
   - Scale inputs specifically for the MLP using parameters derived only from `train`.
3. **Data Splitting Architecture**:
   - Target Split: **70% Train, 15% Validation, 15% Test**.
   - Stratified across `ccme_wqi` to preserve class ratios and geographic representation.
4. **Notebooks & Output Conventions**:
   - Keep notebooks in numbered stage folders under `notebooks/`.
   - Save corresponding generated data under `data/GeneratedOutput/<stage>/`.

---

## 7. Master Task Log & Progress Tracker

| Date | Phase / Role | Key Milestone | Artifacts Produced | Status |
|---|---|---|---|---|
| 2026-10-19 | Role 2 (Collection) | Aggregated 4.29M rows across 7 countries; computed exact CCME WQI | `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Profiled 4.3M rows; discovered sentinels (-999999, pH 67115) & missingness mechanisms | Profiling logs & summary parquets | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Built & executed pure EDA Notebooks with embedded plots | `notebooks/01.EDA/*.ipynb` (00 to 06) | **Completed** |
| 2026-10-19 | Role 6 (Setup) | Created root `requirements.txt` and virtual environment setup guide | `requirements.txt`, `docs/environment_setup.md` | **Completed** |
| 2026-10-21 | Role 2 (Audit) | Automated dataset inspection, data quality audit, and visual diagnostics | `src/data_audit.py`, `notebooks/01.EDA/00_initial_dataset_audit.ipynb` | **Completed** |
| 2026-10-22 | **Role 3 (Preprocessing & Feature Engineering)** | Full data cleaning, feature engineering (N:P ratio, BOD:DO ratio, DO deficit, cyclical season), stratified 70/15/15 split, train-only group median imputation, and MLP StandardScaler | `src/preprocessing.py`, `tests/test_preprocessing.py`, `notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing_and_feature_engineering.ipynb`, `data/GeneratedOutput/02.Preprocessing&FeatureSelection/` (`train.parquet`, `val.parquet`, `test.parquet`, scaled files) | **Completed** |
| Next | **Role 4 (AML Models)** | Train and tune Random Forest and XGBoost baselines; evaluate MAE/RMSE/R2 and Macro F1 | `notebooks/03.ModelTraining/` | **Up Next** |
| Next | **Role 5 (Deep Learning)** | Train regularized dual-head MLP; evaluate calibration, uncertainty, and SHAP explainability | `notebooks/03.ModelTraining/`, `notebooks/04.ModelEvaluation/` | Queued |
| Next | **Role 6 (Integration)** | Compile final comparative benchmark report, presentation slide deck, and viva prep | `docs/FINAL_REPORT.md` | Queued |

