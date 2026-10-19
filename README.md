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
> For full setup details, see the [Environment Setup Guide](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/docs/environment_setup.md).

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
- **Water Quality Parameters ($X$)**:
  - `Ammonia (mg/l)`: Standard limit: 0.5 mg/L
  - `Biochemical Oxygen Demand (mg/l)`: Standard limit: 6.0 mg/L
  - `Dissolved Oxygen (mg/l)`: Standard limit: 14.0 mg/L
  - `Orthophosphate (mg/l)`: Standard limit: 0.05 mg/L
  - `pH (ph units)`: Standard limit: 8.5
  - `Temperature (cel)`: Standard limit: 25.0 °C
  - `Nitrogen (mg/l)`: Standard limit: 20.0 mg/L
  - `Nitrate (mg/l)`: Standard limit: 45.0 mg/L
- **Target Indices ($y$)**:
  - `CCME_Values`: Canadian Council of Ministers of the Environment Water Quality Index score (`0.0` – `100.0`, regression target)
  - `CCME_WQI`: Categorical classification (`Excellent`, `Good`, `Fair`, `Marginal`, `Poor`)

---

## 5. Exploratory Data Analysis (EDA) Suite

The complete exploratory analysis is compiled into a single unified report:  
**[Exploratory Data Analysis Comprehensive Report](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/docs/EDA_REPORT.md)**

All 6 notebooks are fully executed with embedded high-resolution figures in **[`nootbooks/01.EDA/`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/)**:

| Notebook | Focus & Scientific Analyses | Exported Parquet (`data/GeneratedOutput/01.EDA/`) |
|---|---|---|
| **[`01_data_architecture_and_profiling.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/01_data_architecture_and_profiling.ipynb)** | 4.3M rows, 14 columns, 64,665 stations; country & waterbody breakdowns; sampling frequency over 83 years. | `01_dataset_summary_profile.parquet` |
| **[`02_sensor_sparsity_and_missingness.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb)** | Sensor missingness matrix; complete cases (2.41M in England/Ireland/China) vs telemetry (USA/India); nutrient co-occurrence ($r > 0.99$). | `02_missingness_patterns.parquet` |
| **[`03_geochemical_physics_and_anomalies.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb)** | Sensor sentinel code detection (`-999999.0`), scale bugs (`pH = 67115.0`); **Henry's Law** oxygen saturation curves; physical anomaly bounds. | `03_cleaned_physically_validated.parquet` |
| **[`04_bivariate_and_multivariate_correlations.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb)** | Pearson & Spearman heatmaps ($PO_4$ $r = -0.66$, $N$ $r = -0.61$); **eutrophication stoichiometric $N:P$ ratios**; **Streeter-Phelps oxygen sag curves**. | `04_feature_correlations.parquet` |
| **[`05_temporal_seasonality_and_spatial_trends.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb)** | Historical trajectory (1970–2023); monthly seasonal cycles (spring runoff vs summer temperature peaks); flowing rivers vs lake vs effluent contrasts. | `05_temporal_spatial_aggregates.parquet` |
| **[`06_target_behavior_and_leakage_audit.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb)** | `CCME_Values` spike at 100.0; class imbalance (50.1% Excellent vs 3.0% Poor); **AML/DL Target Leakage Warning**; **70 / 15 / 15 Splitting Handoff to Role 3**. | `06_target_audit_and_splitting_guidelines.parquet` |

---

## 6. Repository Layout

```
Semester_5_AML_DL/
├── AGENT.md                                      # Master AI context & operational instructions (All 6 Roles)
├── README.md                                     # Main project entrypoint (this file)
├── requirements.txt                              # Root dependencies (without pinned versions)
├── scripts/
│   ├── setup_environment.sh                      # Dedicated .venv setup & requirements installer
│   └── run_eda_pipeline.py                       # Python master runner (auto-boots .venv, runs 6 notebooks)
├── data/
│   ├── Dataset/
│   │   ├── Parquet/                              # Tracked in Git (Snappy compressed, fast I/O)
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # Master dataset (4,297,895 rows × 14 columns)
│   │   │   └── Country-Wise Data/
│   │   │       ├── Australia_dataset.parquet     # 87 rows
│   │   │       ├── Canada_dataset.parquet        # 3,995 rows
│   │   │       ├── China_dataset.parquet         # 45,997 rows
│   │   │       ├── England_dataset.parquet       # 2,129,198 rows
│   │   │       ├── India_dataset.parquet         # 3,633 rows
│   │   │       ├── Ireland_dataset.parquet       # 235,019 rows
│   │   │       └── USA_dataset.parquet           # 1,879,966 rows
│   │   └── CSV/                                  # Raw CSV files stored locally (ignored via .gitignore)
│   ├── Data Collection Sources/
│   │   ├── DataCollectionSources.parquet         # Provenance URLs & source metadata (153 records)
│   │   └── DataCollectionSources.xlsx            # Raw reference file (ignored via .gitignore)
│   └── GeneratedOutput/
│       └── 01.EDA/                               # All 6 analytical Parquet outputs from EDA
├── nootbooks/
│   └── 01.EDA/                                   # All 6 fully executed EDA Jupyter notebooks
└── docs/
    ├── EDA_REPORT.md                             # Single comprehensive EDA scientific report
    └── environment_setup.md                      # Virtual environment (.venv) setup guide
```

---

## 7. Viewing Parquet Files in VS Code

All `.parquet` files use standard **`SNAPPY`** compression for fast I/O and broad compatibility.

To inspect, filter, and plot any Parquet file directly inside VS Code:
1. Open the **Extensions** panel (`Cmd + Shift + X` on macOS / `Ctrl + Shift + X` on Windows/Linux).
2. Install **Parquetly** (**Parquetly - Parquet Viewer and Editor**).
3. Click any `.parquet` file in `data/` or right-click &rarr; **Reopen Editor With...** &rarr; **Parquetly**.
