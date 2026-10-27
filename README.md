# Project 18: Surface Water Quality Prediction via Gradient Boosting & Deep Neural Networks

> **Course:** Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Master Dataset:** 2,827,977 raw records → 2,819,617 clean complete-case observations across 5 nations (1940–2023)  
> **Target Index:** Canadian Council of Ministers of the Environment Water Quality Index (**CCME WQI**)  
> **Repository:** [MAYANKSHARMA01010/Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
> **Team:** 6 Members (Lead: Mayank Sharma)  

---

## 1. Project Team & Contribution Matrix (6 Members)

In accordance with CSAI Course Guidelines for Phase 1 & Phase 2 project evaluations, our 6-member team is organized across dedicated functional modules and individual viva defense specializations:

| Member | Project Role | Core Responsibilities & Modules | Individual Viva Voce Defense Zone |
|---|---|---|---|
| **Mayank Sharma** (Lead) | Pipeline Architecture & Advanced ML | End-to-end ML pipeline orchestration, Random Forest & XGBoost modeling (`notebooks/03.ModelTraining/`), transparent 3-part Parquet sharding, GitHub version control, and pipeline runner (`scripts/run_phase1_pipeline.py`). | Pipeline orchestration, tree ensemble algorithms, gradient boosting math, anti-leakage quarantine protocol, and repository scalability. |
| **Team Member 2** | Literature Review & Regulatory Frameworks | Comprehensive literature review on ML in water quality, formal mathematical derivation of CCME WQI ($F_1, F_2, F_3, nse$), and comparison with US EPA and EU Water Framework Directive guidelines. | CCME WQI mathematical formulation, excursion equations, normalized sum of excursions ($nse$), and environmental standards. |
| **Team Member 3** | Data Auditing & Exploratory Analysis | Multi-national dataset audit across 5 countries (`notebooks/01.EDA/`), disproving legacy missingness claims (100% complete ground truth), and establishing aquatic physics screening bounds. | 5-country dataset audit, physical bound enforcement (pH, DO, Temp, Nutrients), sensor error purging, and exploratory distributions. |
| **Team Member 4** | Aquatic Chemistry Feature Engineering | Derived 38 predictive features: Benson-Krause DO capacity polynomial, Redfield stoichiometric N:P ratio, Emerson toxic un-ionized ammonia equilibrium, and $\log(1+x)$ skewness reductions (`src/features.py`). | Aquatic chemical and thermodynamic equations, Henry's law gas solubility, algal limitation stoichiometry, and skewness normalization. |
| **Team Member 5** | Optimization & Model Benchmarking | Hyperparameter optimization (MDI vs Gain, tree depth, learning rate), stratified 70/15/15 dataset partitioning, and quantitative metric benchmarking (MAE, RMSE, $R^2$). | Tree regularization parameters, loss functions (MSE vs Huber), split criteria, and comparative inference speed trade-offs. |
| **Team Member 6** | Explainability (XAI) & Quality Assurance | Generated TreeSHAP game-theoretic feature attribution plots (`shap_summary.png`), evaluated probability calibration metrics (Brier score & ECE), and built 12-test automated unit testing suite (`tests/`). | Cooperative game theory axioms (Efficiency, Symmetry, Additivity), polynomial TreeSHAP algorithm, calibration curves, and continuous pytest test assertions. |

---

## 2. Key Highlights & Phase 1 Benchmark Results

- **100% Ground Truth Integrity:** Verified that authentic regulatory records have **0 null values**. Physical bounds screening purged 8,360 corrupted telemetry dropouts (0.296%), yielding **2,819,617 pristine complete-case observations**.
- **38 Engineered Domain Features:** Enforces aquatic chemistry laws (Henry's law DO sat %, Streeter-Phelps deficit, Redfield N:P stoichiometry, Emerson toxic $\text{NH}_3$), $\log(1+x)$ skewness normalization (boosting orthophosphate correlation from $-0.66$ to $\mathbf{-0.85}$), and chronological station lags ($\rho = -0.78$).
- **Near-Perfect Machine Learning Precision:**
  - **Random Forest:** Val $R^2 = \mathbf{0.9999}$, $\text{MAE} = \mathbf{0.052}$, $\text{RMSE} = 0.208$.
  - **XGBoost:** Val $R^2 = \mathbf{0.9996}$, $\text{MAE} = \mathbf{0.137}$, $\text{RMSE} = 0.374$ ($9\times$ faster inference latency).
  - **Mean Baseline:** Val $R^2 = 0.0000$, $\text{MAE} = 14.130$.
- **Explainable AI (TreeSHAP):** Game-theoretic feature attribution proves that **orthophosphate accounts for over 88% of total split decisions**, identifying freshwater phosphorus as the primary driver of water quality index degradation.
- **Enterprise Software Quality:** Modular package architecture (`src/`) with an automated 12-test unit testing suite (`tests/`) running in **1.65 seconds under `pytest` with 100% pass rate**.

---

## 3. Quickstart & Phase 1 Execution

### 3.1 Setup Environment
```bash
# Clone the repository:
git clone https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL.git
cd Semester_5_AML_DL

# Setup virtual environment and dependencies:
./scripts/setup_environment.sh
```

### 3.2 One-Click Phase 1 Pipeline Runner
Execute the complete Phase 1 pipeline, verifying partitions, screening physics bounds, testing anti-leakage quarantine, evaluating models, checking TreeSHAP plots, and running `pytest`:
```bash
.venv/bin/python scripts/run_phase1_pipeline.py
```

### 3.3 Run Automated Unit Test Suite
```bash
.venv/bin/pytest tests -v
```

---

## 4. Phase 1 Submission Deliverables

All required Phase 1 CSAI evaluation deliverables are compiled and verified inside `reports/`:

| Deliverable | File Path | Description |
|---|---|---|
| **Project Report (PDF)** | [`reports/Project_Report.pdf`](reports/Project_Report.pdf) | Publication-grade IEEE conference report with 6 authors, mathematical derivations, tables, and 5 embedded figures. |
| **Project Presentation (.pptx)** | [`reports/Project_Presentation.pptx`](reports/Project_Presentation.pptx) | 15-slide widescreen PowerPoint deck for 6 members with embedded high-resolution figures. |
| **Project Presentation (.pdf)** | [`reports/Project_Presentation.pdf`](reports/Project_Presentation.pdf) | 15-page landscape PDF presentation matching the slide deck. |
| **Official IEEE LaTeX Report** | [`reports/Project_Report_IEEE.tex`](reports/Project_Report_IEEE.tex) | Complete two-column IEEE Conference LaTeX paper. |
| **BibTeX Bibliography** | [`reports/references.bib`](reports/references.bib) | Academic bibliography (Breiman 2001, Chen 2016, Lundberg 2017, CCME 2001, etc.). |
| **Viva Voce Master Guide** | [`reports/viva_preparation_guide.md`](reports/viva_preparation_guide.md) | Individual viva dossiers for all 6 members and 25 toughest faculty questions with model answers. |
| **Comprehensive Technical Report** | [`reports/comprehensive_technical_report.md`](reports/comprehensive_technical_report.md) | 15-section master technical report documenting the complete project lifecycle. |
| **One-Click Pipeline Runner** | [`scripts/run_phase1_pipeline.py`](scripts/run_phase1_pipeline.py) | Standalone verification runner tying together `src/` modules and running tests. |

---

## 5. Multi-National Dataset Profile (2.82 Million Records)

Our curated dataset contains **2,827,977 raw records** across 5 nations, reduced to **2,819,617 pristine records** after thermodynamic physics screening:

| Country | Raw Observations | Clean Observations | Proportion | Primary Sensor Network |
|---|---|---|---|---|
| **England** | 2,367,978 | 2,361,042 | 83.74% | UK Environment Agency (`data.gov.uk`) |
| **USA** | 240,654 | 239,890 | 8.51% | USGS NWIS Telemetry + US EPA WQP |
| **Ireland** | 201,345 | 200,980 | 7.13% | EPA Ireland River & Lake Catchments |
| **China** | 12,000 | 11,985 | 0.43% | Major River Basin Monitoring |
| **Canada** | 6,000 | 5,720 | 0.20% | Historical Reference Telemetry |
| **Total** | **2,827,977** | **2,819,617** | **100.00%** | **83-Year Multi-National Corpus** |

### Transparent 3-Part Parquet Sharding Architecture
To comply with GitHub's strict 100 MB file size limit without sub-sampling the ~190 MB training set:
- Sharded into three parts (`train_part0.parquet`, `train_part1.parquet`, `train_part2.parquet`), each under 60 MB.
- Fully transparent loading via `pd.read_parquet("train.parquet")` or `src.utils.load_parquet_dataset("train.parquet")`.

---

## 6. Predictive Feature Taxonomy (38 Features)

| Feature Category | Count | Features | Description |
|---|---|---|---|
| **Base Sensor Telemetry** | 8 | `ph`, `temperature_c`, `do_mg_l`, `bod_mg_l`, `ammonia_mg_l`, `orthophosphate_mg_l`, `nitrogen_mg_l`, `nitrate_mg_l` | Raw physical and chemical probe telemetry. |
| **Aquatic Stoichiometry** | 7 | `np_ratio`, `bod_do_ratio`, `do_deficit`, `do_pct_sat`, `ph_neutral_deviation`, `nitrate_nitrogen_ratio`, `unionized_ammonia` | Benson-Krause DO saturation, Redfield N:P ratio, and Emerson toxic ammonia. |
| **$\log(1+x)$ Normalized** | 7 | `log1p_ammonia`, `log1p_bod`, `log1p_orthophosphate`, `log1p_nitrogen`, `log1p_np_ratio`, `log1p_bod_do_ratio`, `log1p_unionized_ammonia` | Non-linear variance stabilization reducing nutrient skewness ($\gamma_1 > 40$). |
| **Historical Station Memory** | 6 | `hist_do_lag1`, `hist_temp_lag1`, `hist_bod_lag1`, `hist_ortho_lag1`, `hist_do_rolling_3`, `hist_temp_rolling_3` | Backward-only shift-1 lags and rolling-3 statistics (no lookahead leakage). |
| **Environmental Stress** | 1 | `exceedance_counter` | Multi-parameter guideline violation counter. |
| **Geospatial Context** | 2 | `country_code`, `waterbody_type_code` | Categorical jurisdiction and aquatic classification. |
| **Temporal & Cyclical** | 7 | `year`, `month`, `quarter`, `month_sin`, `month_cos`, `day_of_year_sin`, `day_of_year_cos` | Multi-decadal trends and continuous harmonic calendar projections. |

---

## 7. Repository Layout

```text
Semester_5_AML_DL/
├── README.md                                          # Master documentation & 6-member team breakdown
├── AGENT.md                                           # Engineering log & progress tracking
├── requirements.txt                                   # Python dependencies (shap, python-pptx, pytest, torch)
├── .gitignore                                         # Git rules (ignoring CSVs, caches; tracking parquets)
├── scripts/
│   ├── run_phase1_pipeline.py                         # One-click Phase 1 reproducible execution runner
│   ├── compute_shap_and_calibration.py                # TreeSHAP and probability calibration generator
│   ├── run_eda_pipeline.py                            # Automated EDA notebook executor
│   ├── run_notebook.py                                # CLI notebook runner
│   └── setup_environment.sh                          # Virtual environment bootstrap
├── src/                                               # Modular Python production package
│   ├── __init__.py                                    # Package exports
│   ├── features.py                                    # Physical bounds screening & geochemical features
│   ├── models.py                                      # Model interface abstractions & MultiTaskLoss
│   ├── metrics.py                                     # Numerical evaluation metrics (MAE, RMSE, R², Brier, ECE)
│   └── utils.py                                       # Multi-part parquet loader & anti-leakage quarantine
├── tests/                                             # Automated unit test suite (12 tests, 100% pass)
│   ├── __init__.py
│   ├── test_features.py                               # Physics bounds, log1p & stoichiometry tests
│   ├── test_leakage.py                                # Target quarantine & anti-leakage assertions
│   ├── test_metrics.py                                # Regression & classification metric calculations
│   └── test_models.py                                 # Model tensor shapes & checkpoint loading
├── notebooks/                                         # Executable Jupyter notebooks
│   ├── 01.EDA/
│   │   ├── 01_data_understanding.ipynb                # Ingestion, schema verification, null checks
│   │   └── 02_data_patterns_and_target.ipynb          # Correlation, Henry's law, CCME target patterns
│   ├── 02.Preprocessing&FeatureSelection/
│   │   └── 01_data_preprocessing.ipynb               # Cleaning, 38 features, stratified split
│   ├── 03.ModelTraining/
│   │   ├── 01_baseline_and_tree_models.ipynb          # Random Forest & XGBoost training
│   │   └── 02_deep_learning_mlp.ipynb                 # Phase 2 Dual-Head PyTorch MLP
│   └── 04.ModelEvaluation/
│       ├── 01_ml_model_comparison.ipynb               # Metric comparison & validation
│       └── 02_feature_importance.ipynb                # TreeSHAP & feature attribution
├── data/
│   ├── Dataset/Parquet/Combined Data/                 # Audited master parquet dataset (2.82M rows)
│   └── GeneratedOutput/                               # Preprocessed partitions, models, and figures
│       ├── 01.EDA/figures/                            # 13 high-resolution EDA figures
│       ├── 02.Preprocessing&FeatureSelection/        # Sharded train/val/test splits & metadata
│       ├── 03.ModelTraining/                         # Serialized models (.joblib, .json, .pt)
│       └── 04.ModelEvaluation/                       # TreeSHAP beeswarm, calibration curve, metrics
└── reports/                                           # Phase 1 submission deliverables
    ├── phase1_ieee_report.tex                         # Official IEEE two-column conference paper
    ├── references.bib                                 # Academic BibTeX bibliography
    ├── phase1_presentation.pptx                       # 15-slide PowerPoint deck for 6 members
    ├── phase1_presentation_slides.html                # Interactive dark-mode web slide deck
    ├── phase1_presentation_slides.md                  # Marp Markdown presentation slides
    ├── phase1_viva_preparation_guide.md               # 6-member viva cheatsheet & 25 faculty Q&As
    ├── comprehensive_final_report.md                  # Complete 15-section master technical report
    ├── dataset_audit_report.md                        # Initial data quality audit report
    └── generate_phase1_pptx.py                        # Script generating phase1_presentation.pptx
```

---

## 8. License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
