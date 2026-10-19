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

## 7. Planned Notebook Workflow (EDA & Dataset Preparation)

- [x] **Phase 0 — Dataset Collection & Harmonization**: 4.29M records collected, harmonized to 14 columns with CCME WQI, Snappy compressed.
- [ ] **Notebook 1 — Dataset Overview & Univariate Distribution**: Inspect feature ranges, statistical distributions, boxplots, histograms, and class balance across countries.
- [ ] **Notebook 2 — Missing Value & Sensor Sparsity Analysis**: Quantify missingness patterns per parameter and per country, analyze missing-at-random (MAR) vs sensor absence.
- [ ] **Notebook 3 — Outlier Detection & Physical Boundary Checks**: Flag impossible chemical values (e.g. negative pH, dissolved oxygen > physical saturation) and quantify sensor drift.
- [ ] **Notebook 4 — Bivariate & Multivariate Correlation**: Pearson and Spearman correlations among chemical parameters, correlation with `CCME_Values`.
- [ ] **Notebook 5 — Temporal & Spatial Trend Analysis**: Multi-year seasonal trends, water quality variation across river vs lake waterbodies.
- [ ] **Notebook 6 — Modeling Dataset Split & Benchmark Prep**: Generate leak-free train/val/test splits (time and site stratified) saved into `data/GeneratedOutput/` for Role 3 & Role 4.

---

## 8. Task Log & Current Progress

| Date | Step | Description | Output Files | Status |
|---|---|---|---|---|
| 2026-10-19 | Dataset Collection | Harvested 4.29M rows across 7 nations (England, USA, Ireland, China, Canada, India, Australia) with exact CCME WQI | `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` | Completed |
| 2026-10-19 | Parquet Compatibility | Re-saved all Parquet datasets with `snappy` compression for full Parquetly viewing | All `.parquet` files in `data/` | Completed |
| 2026-10-19 | Agent Context Setup | Created `AGENT.md`, established workflow between `nootbooks/`, `docs/`, and `data/GeneratedOutput/` | `AGENT.md`, `docs/` | Completed |
