# AGENT MASTER CONTEXT & OPERATIONAL GUIDE

> **Project**: Project 18 — Water Quality Prediction via Gradient Boosting and Deep Neural Networks  
> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Scope**: Master AI Instructions across the entire end-to-end project (All 6 Roles)  
> **Last Updated**: 2026-10-25  

---

## 1. Project Overview & The Hybrid Machine Learning / Deep Learning Logic

This project aims to predict documented surface water quality indices and categorical classes from physical and chemical measurements across freshwater and marine environments.

### The AML / DL Hybrid Logic
1. **AML Baseline (Tree Models)**:
   - Algorithms: **Mean Baseline**, **Random Forest**, and **XGBoost**.
   - Input Features: 15 verified features (8 physical/chemical parameters + 7 domain-engineered geochemical and seasonal predictors).
   - Strengths: Strong performance on tabular data and robust to non-linear thresholds.
2. **Deep Learning Model (Neural Network - Phase 2)**:
   - Architecture: Small, regularized **Multi-Layer Perceptron (MLP)**.
   - Purpose: Capture smooth combinations and calibrated uncertainty when multiple chemical sensors interact non-linearly.
   - Dual-Head Setup: A regression head for the continuous index (`CCME_Values`) and a classification head for the categorical grade (`CCME_WQI`).
3. **Fair Evaluation Protocol**:
   - Managed by evaluation teammate in `notebooks/04.ModelEvaluation/`.
   - Uses site-separated and stratified holdout splits (strictly preventing spatial and temporal leakage).

---

## 2. Project Team Division (All 6 Roles)

| Role # | Functional Area | Primary Responsibilities | Current Status |
|---|---|---|---|
| **Role 1** | **Literature Review & Research** | Research water quality prediction literature (Breiman 2001, Goodfellow 2016), compare methods, identify research gaps. | In Progress |
| **Role 2** | **Dataset Collection & EDA** | Find multi-country datasets, handle missing values, clean sensor sentinels, perform full EDA in clean notebooks, export summaries. | **Completed** |
| **Role 3** | **Data Preprocessing & Feature Engineering** | Clean/impute data, scale inputs, engineer 15 features, create **70 / 15 / 15 Train/Val/Test** stratified splits. | **Completed** |
| **Role 4** | **Machine Learning Models** | Implement and train Mean Baseline, Random Forest, and XGBoost; serialize models and prediction handoff. | **Completed** |
| **Role 5** | **Deep Learning, Theory & Evaluation** | Train regularized MLP neural net, implement regression/classification heads, evaluate calibration & error metrics. | Queued (Phase 2) |
| **Role 6** | **Integration, GitHub & Documentation** | Maintain repository integrity, consolidate final report, presentation, and viva preparation. | Ongoing |

---

## 3. Target Variables & What We Are Predicting

We use the Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**):

| Target Column | Nature | Values / Range | Purpose / Loss Function |
|---|---|---|---|
| `ccme_values` | **Continuous (Regression)** | `0.0` to `100.0` | Predict precise water quality score; Loss: MSE / Huber Loss. |
| `ccme_wqi` | **Categorical (Classification)** | `Excellent` ($95-100$), `Good` ($80-94$), `Fair` ($65-79$), `Marginal` ($45-64$), `Poor` ($0-44$) | Multi-class classification; Loss: Class-weighted Cross-Entropy. |
| `ccme_wqi_code`| **Ordinal Integer** | `0` to `4` | Machine-readable class labels. |

---

## 4. Input Features ($X$) & Metadata

### Chemical & Physical Predictors (8)
1. `ph`: Standard limit: 8.5 (Physically bounded between 3.0 and 12.0)
2. `do_mg_l`: Standard limit: 14.0 (Henry's Law saturation checked)
3. `temperature_c`: Standard limit: 25.0°C (Fahrenheit readings normalized)
4. `bod_mg_l`: Standard limit: 6.0 (Organic pollution load)
5. `ammonia_mg_l`: Standard limit: 0.5 (Toxic breakdown product)
6. `orthophosphate_mg_l`: Standard limit: 0.05 (Key driver of eutrophication)
7. `nitrogen_mg_l`: Standard limit: 20.0 (Total nutrient load)
8. `nitrate_mg_l`: Standard limit: 45.0 (Oxidized nitrogen runoff)

### Engineered Predictors (7)
9. `np_ratio`: Nitrogen to Orthophosphate ratio
10. `bod_do_ratio`: Biochemical oxygen demand to dissolved oxygen ratio
11. `do_deficit`: Dissolved oxygen deficit relative to temperature saturation
12. `month`: Integer calendar month (1–12)
13. `quarter`: Integer calendar quarter (1–4)
14. `month_sin`: $\sin(2\pi \cdot \text{month} / 12)$
15. `month_cos`: $\cos(2\pi \cdot \text{month} / 12)$

---

## 5. Repository Layout

```text
Semester_5_AML_DL/
├── AGENT.md                                      # Master guide for the entire project
├── README.md                                     # Main project guide
├── requirements.txt                              # Root dependencies
├── scripts/
│   ├── setup_environment.sh                      # Environment setup script (.venv)
│   ├── run_eda_pipeline.py                       # Automated EDA runner
│   └── run_notebook.py                           # CLI notebook runner
├── data/
│   ├── Dataset/
│   │   ├── Parquet/
│   │   │   ├── Combined Data/
│   │   │   │   └── Combined_dataset.parquet      # 4,297,895 records × 14 columns (Snappy)
│   │   │   └── Country-Wise Data/
│   │   └── CSV/
│   └── GeneratedOutput/
│       ├── 01.EDA/                               # EDA summary parquets & final cleaned data
│       ├── 02.Preprocessing&FeatureSelection/    # Model training datasets (train, val, test, scaled)
│       ├── 03.ModelTraining/                     # Serialized models & validation predictions
│       └── 04.ModelEvaluation/                   # Model comparison CSV, plots & feature importance charts
├── notebooks/
│   ├── 01.EDA/                                   # 01_data_understanding.ipynb, 02_data_patterns_and_target.ipynb
│   ├── 02.Preprocessing&FeatureSelection/        # 01_data_preprocessing.ipynb
│   ├── 03.ModelTraining/                         # 01_baseline_and_tree_models.ipynb
│   └── 04.ModelEvaluation/                       # 01_ml_model_comparison.ipynb, 02_feature_importance.ipynb
├── reports/
│   ├── dataset_audit_report.md
│   └── figures/
```

---

## 6. Master Task Log & Progress Tracker

| Date | Phase / Role | Key Milestone | Artifacts Produced | Status |
|---|---|---|---|---|
| 2026-10-19 | Role 2 (Collection) | Aggregated 4.29M rows across 7 countries; computed exact CCME WQI | `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` | **Completed** |
| 2026-10-19 | Role 2 (EDA) | Profiled 4.3M rows; discovered sentinels & missingness mechanisms | Profiling logs & summary parquets | **Completed** |
| 2026-10-25 | Role 2 (EDA Simplification) | Consolidated 7 EDA notebooks into 2 clean Colab notebooks | `notebooks/01.EDA/01_data_understanding.ipynb`, `02_data_patterns_and_target.ipynb` | **Completed** |
| 2026-10-25 | Role 3 (Preprocessing Simplification) | Consolidated 3 preprocessing notebooks into 1 clean Colab notebook | `notebooks/02.Preprocessing&FeatureSelection/01_data_preprocessing.ipynb` | **Completed** |
| 2026-10-25 | Role 4 (AML Models Simplification) | Formatted Phase 1 ML training notebook with single-line headers; verified saved models & predictions | `notebooks/03.ModelTraining/01_baseline_and_tree_models.ipynb`, `data/GeneratedOutput/03.ModelTraining/` | **Completed** |
| 2026-10-26 | Role 5 / Evaluation | Evaluated baseline, Random Forest, XGBoost; verified alignment, metrics & feature importance | `notebooks/04.ModelEvaluation/`, `data/GeneratedOutput/04.ModelEvaluation/` | **Completed** |
| Future | Role 5 (Deep Learning) | Train regularized dual-head MLP; evaluate calibration, uncertainty (Phase 2) | `notebooks/03.ModelTraining/` | Queued (Phase 2) |

