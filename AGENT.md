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
├── AGENT.md                                      # Master AI instructions for the entire project
├── README.md                                     # Main public README linking all resources
├── requirements.txt                              # Root dependencies (without pinned versions)
├── scripts/
│   ├── setup_environment.sh                      # Environment setup script (.venv, pip, requirements, ipykernel)
│   └── run_eda_pipeline.py                       # Python master runner (auto-bootstraps .venv, runs 6 NBs, parquet audit)
├── data/
│   ├── Dataset/
│   │   ├── Parquet/
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # 4,297,895 records × 14 columns (Snappy)
│   │   │   └── Country-Wise Data/
│   │   │       ├── Australia_dataset.parquet     # 87 rows
│   │   │       ├── Canada_dataset.parquet        # 3,995 rows
│   │   │       ├── China_dataset.parquet         # 45,997 rows
│   │   │       ├── England_dataset.parquet       # 2,129,198 rows
│   │   │       ├── India_dataset.parquet         # 3,633 rows
│   │   │       ├── Ireland_dataset.parquet       # 235,019 rows
│   │   │       └── USA_dataset.parquet           # 1,879,966 rows
│   │   └── CSV/                                  # Local raw CSVs (ignored by .gitignore)
│   ├── Data Collection Sources/
│   │   ├── DataCollectionSources.parquet         # 153 official provenance records (Snappy)
│   │   └── DataCollectionSources.xlsx            # Raw Excel reference (ignored by .gitignore)
│   └── GeneratedOutput/
│       └── 01.EDA/                               # All 6 analytical Parquet outputs from EDA
├── nootbooks/
│   └── 01.EDA/                                   # All 6 fully executed EDA Jupyter notebooks
└── docs/
    ├── EDA_REPORT.md                             # Single comprehensive EDA scientific report
    └── environment_setup.md                      # Virtual environment (.venv) setup guide
```

---

## 6. Golden Rules for AI Operations

1. **Parquet with Snappy Only**:
   - Always read and write `.parquet` files using **`compression='snappy'`**.
   - Ensures universal compatibility with the VS Code **Parquetly** extension, DuckDB, Polars, and PyArrow.
   - Never commit `.csv` or `.xlsx` files into Git.
2. **Data Leakage Safeguards**:
   - Do not allow models to predict `CCME_Values` from the exact constituent inputs without acknowledging mathematical leakage.
   - Impute missing data **strictly within the training split** (never compute global medians across test sets).
   - Scale inputs specifically for the MLP using parameters derived only from `train`.
3. **Data Splitting Architecture (Handoff from Role 2 to Role 3)**:
   - Target Split: **70% Train, 15% Validation, 15% Test**.
   - Must be stratified across `Country`, `Waterbody Type`, and `CCME_WQI` to maintain identical class ratios and prevent station-level memorization.
4. **Notebooks & Output Conventions**:
   - Keep notebooks in numbered stage folders under `nootbooks/` (e.g. `nootbooks/01.EDA/`, `nootbooks/02.Preprocessing/`, etc.).
   - Save corresponding generated data under `data/GeneratedOutput/<stage>/`.
5. **Continuous Documentation Updates**:
   - Whenever any phase or role completes a task, update `AGENT.md` and document results in `docs/`.

---

## 7. Master Task Log & Progress Tracker

| Date | Phase / Role | Key Milestone | Artifacts Produced | Status |
|---|---|---|---|---|
| 2026-10-19 | Role 2 (Collection) | Aggregated 4.29M rows across 7 countries; computed exact CCME WQI | `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` | **Completed** |
| 2026-10-19 | Role 2 (Collection) | Re-encoded all Parquet files with Snappy compression for Parquetly viewer | All Parquet files in `data/` | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Profiled 4.3M rows; discovered sentinels (-999999, pH 67115) & missingness mechanisms | Profiling execution logs | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Built & executed all 6 Pure EDA Notebooks with embedded plots | `nootbooks/01.EDA/*.ipynb` (6 notebooks) | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Exported 6 analytical summary Parquet datasets | `data/GeneratedOutput/01.EDA/*.parquet` (6 files) | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Compiled single unified EDA scientific report | `docs/EDA_REPORT.md` | **Completed** |
| 2026-10-19 | Role 6 (Setup) | Created root `requirements.txt` and virtual environment setup guide | `requirements.txt`, `docs/environment_setup.md` | **Completed** |
| Next | **Role 3 (Preprocessing)** | Build clean imputation pipeline, feature scaling, and 70/15/15 stratified train/val/test splits | `nootbooks/02.Preprocessing/`, `data/GeneratedOutput/02.Preprocessing/` | **Up Next** |
| Next | **Role 4 (AML Models)** | Train and tune Random Forest and XGBoost baselines; evaluate MAE/RMSE/R2 and Macro F1 | `nootbooks/03.AML_Models/` | Queued |
| Next | **Role 5 (Deep Learning)** | Train regularized dual-head MLP; evaluate calibration, uncertainty, and SHAP explainability | `nootbooks/04.Deep_Learning/` | Queued |
| Next | **Role 6 (Integration)** | Compile final comparative benchmark report, presentation slide deck, and viva prep | `docs/FINAL_REPORT.md` | Queued |
