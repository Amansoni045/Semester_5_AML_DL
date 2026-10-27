# Project 18: Water Quality Prediction via Gradient Boosting & Deep Neural Networks

> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Master Dataset**: 4,297,895 records × 14 columns across 7 countries (1940–2023)  
> **Target Index**: Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**)  

---

## 1. Project Overview & Architecture

This project predicts surface water quality from physical and chemical sensor measurements using a two-phase evaluation framework:

- **Phase 1: Machine Learning (Current Responsibility)**: Train **Baseline**, **Random Forest**, and **XGBoost** regressors on 15 verified features (8 physical/chemical parameters + 7 engineered geochemical and seasonal predictors). Models are serialized and predictions are exported for model evaluation.
- **Phase 2: Deep Learning (Future Phase)**: Multi-layer perceptron neural networks to capture non-linear sensor dynamics and uncertainty.
- **Model Evaluation**: Managed by another team member in `notebooks/04.ModelEvaluation/` and `data/GeneratedOutput/04.ModelEvaluation/`.

---

## 2. Quickstart & Workflow Execution

You can run individual notebooks directly or use the provided automation scripts:

```bash
# Setup environment:
./scripts/setup_environment.sh

# Run EDA pipeline:
.venv/bin/python scripts/run_eda_pipeline.py

# Or run any individual notebook directly:
.venv/bin/python scripts/run_notebook.py notebooks/01.EDA/01_data_understanding.ipynb
.venv/bin/python scripts/run_notebook.py notebooks/01.EDA/02_data_patterns_and_target.ipynb
.venv/bin/python scripts/run_notebook.py "notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb"
.venv/bin/python scripts/run_notebook.py notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb
```

---

## 3. Global Dataset Architecture

Our harmonized dataset contains **4,297,895 surface water monitoring observations** across **64,665 unique stations** spanning 1940 to 2023.

| Country | Records | Year Range | Primary Source Networks |
|---|---|---|---|
| **England** | 2,129,198 | 2000–2023 | Environment Agency UK (`data.gov.uk`) |
| **USA** | 1,879,966 | 1940–2023 | USGS NWIS Telemetry + EPA Water Quality Portal |
| **Ireland** | 235,019 | 1940–2023 | EPA Ireland (`catchments.ie` / `data.europa.eu`) |
| **China** | 45,997 | 2001–2017 | Ministry of Ecology and Environment / `data.gov.hk` |
| **Canada** | 3,995 | 1968–2021 | Open Canada / ECCC / Lac Hertel High-Frequency Sonde Network |
| **India** | 3,633 | 2003–2023 | Central Pollution Control Board (CPCB) |
| **Australia** | 87 | 2014–2015 | Torrens River & Lake Water Quality (`data.sa.gov.au`) |
| **Total** | **4,297,895** | **1940–2023** | **Harmonized 14-Column Master Dataset** |

---

## 4. Feature Definitions & Target Quarantine

- **Base Features (8)**:
  - `ph`: Negative log hydrogen activity (pH units)
  - `temperature_c`: Water temperature (Celsius)
  - `do_mg_l`: Dissolved Oxygen concentration (mg/L)
  - `bod_mg_l`: 5-day Biochemical Oxygen Demand (mg/L)
  - `ammonia_mg_l`: Ammonium / ammonia concentration (mg/L)
  - `orthophosphate_mg_l`: Dissolved reactive phosphorus (mg/L)
  - `nitrogen_mg_l`: Total / dissolved nitrogen (mg/L)
  - `nitrate_mg_l`: Nitrate ion concentration (mg/L)
- **Engineered Features (7)**:
  - `np_ratio`: Stoichiometric Total Nitrogen to Orthophosphate ratio
  - `bod_do_ratio`: Organic loading relative to available dissolved oxygen
  - `do_deficit`: Signed oxygen deficit $DO_{\text{sat}}(T) - DO$
  - `month`: Integer calendar month (1–12)
  - `quarter`: Integer calendar quarter (1–4)
  - `month_sin`: Cyclical seasonal transform $\sin(2\pi(m-1)/12)$
  - `month_cos`: Cyclical seasonal transform $\cos(2\pi(m-1)/12)$
- **Target Variables (Strictly quarantined from model inputs)**:
  - `ccme_values`: Canadian Council of Ministers of the Environment Water Quality Index score (`0.0`–`100.0`, continuous regression target)
  - `ccme_wqi`: Categorical classification (`Poor`, `Marginal`, `Fair`, `Good`, `Excellent`)
  - `ccme_wqi_code`: Ordinal integer classification label (`0`–`4`)

---

## 5. Notebooks Structure

```
notebooks/
├── 01.EDA/
│   ├── 01_data_understanding.ipynb                # Dataset loading, structure, missing values, summary stats
│   └── 02_data_patterns_and_target.ipynb          # Sentinels, physical validity, correlations, seasonality, target distributions
├── 02.Preprocessing&FeatureSelection/
│   └── 01_data_preprocessing.ipynb               # Full 7-step pipeline: cleaning, feature engineering, split, imputation, scaling
├── 03.ModelTraining/
│   └── 01_baseline_and_tree_models.ipynb          # Phase 1: Mean baseline, Random Forest, XGBoost training & artifact export
└── 04.ModelEvaluation/
    ├── 01_ml_model_comparison.ipynb               # Metric calculation (MAE, RMSE, R²), alignment check & comparison
    └── 02_feature_importance.ipynb                # Built-in Random Forest MDI & XGBoost gain feature importances
```

All notebooks follow clean Colab-style code formatting with single-line Markdown section headers and concise, readable Python cells without unnecessary boilerplate or inline comment clutter.

---

## 6. Repository Layout

```text
Semester_5_AML_DL/
├── README.md                                     # Main project documentation
├── AGENT.md                                      # Project progress & roadmap
├── requirements.txt                              # Python package dependencies
├── .gitignore                                    # Git ignore rules for virtualenvs & large binaries
├── scripts/
│   ├── setup_environment.sh                      # Virtual environment setup script
│   ├── run_eda_pipeline.py                       # Automated EDA notebook runner
│   └── run_notebook.py                           # CLI notebook executor
├── data/
│   ├── Dataset/
│   │   ├── Parquet/                              # Tracked parquet master dataset
│   │   └── CSV/                                  # Raw CSV archive (ignored by git)
│   └── GeneratedOutput/
│       ├── 01.EDA/                               # EDA summary parquets and final clean dataset
│       ├── 02.Preprocessing&FeatureSelection/    # Model-ready train, val, test splits & metadata
│       ├── 03.ModelTraining/                     # Serialized models & validation predictions
│       └── 04.ModelEvaluation/                   # Model comparison CSV, plots & feature importance charts

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

All `.parquet` files use standard **`SNAPPY`** compression.

To inspect, filter, and plot any Parquet file directly inside VS Code:
1. Open the **Extensions** panel (`Cmd + Shift + X` on macOS / `Ctrl + Shift + X` on Windows/Linux).
2. Install **Parquetly** (**Parquetly - Parquet Viewer and Editor**).
3. Click any `.parquet` file in `data/` to view its contents interactively.
