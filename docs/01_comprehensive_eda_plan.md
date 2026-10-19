# Comprehensive EDA & Data Splitting Implementation Plan

> **Project**: Project 18 — Water Quality Prediction via Gradient Boosting & Deep Neural Networks  
> **Author / Role**: Role 2 — Dataset Collection & Exploratory Data Analysis (EDA)  
> **Status**: Approved Implementation Plan  
> **Dataset**: 4,297,895 records × 14 columns across 7 countries (1940–2023)  

---

## Executive Summary

To make our water quality prediction project stand out and demonstrate research-grade rigor during academic evaluation, our Exploratory Data Analysis (EDA) goes far beyond basic summary statistics. 

We have designed an end-to-end **7-Stage EDA Pipeline** across modular Jupyter notebooks. Each stage focuses on a specific challenge in environmental data science (sensor sparsity, geochemical physics, extreme anomalies, seasonal cycles, and spatial autocorrelation). Finally, we use advanced machine learning principles to split the data into **Train, Validation, and Test sets (70% / 15% / 15%)** without data leakage.

Every notebook reads from `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet` and exports clean, processed subsets directly into `data/GeneratedOutput/` as `.parquet` files compressed with standard `snappy`.

---

## 1. What Our Data Profiling Discovered (The Ground Truth)

Before designing the plan, we analyzed all **4,297,895 rows** in detail. Here are the critical facts discovered:

| Characteristic | Key Metric / Value | Why It Matters for Modeling |
|---|---|---|
| **Total Rows** | 4,297,895 | Massive scale suitable for deep neural networks (MLP) and gradient boosted trees. |
| **Unique Stations (`Area`)** | 64,665 unique stations | Huge spatial diversity across 7 countries (England, USA, Ireland, China, Canada, India, Australia). |
| **Date Range** | 1940 to 2023 (83 years) | Long temporal horizon with heavy concentration in modern years (2014–2023: >2.2M rows). |
| **Complete Panels** | England (2.13M), Ireland (235k), China (46k) | Have **0.0% missing values** across all 8 chemical parameters. |
| **Telemetry Streams** | USA (1.88M), India (3.6k) | Focus on core in-situ continuous probes (`Temperature`, `pH`, `DO`). Lab nutrients (`Ammonia`, `BOD`, `Nitrate`) are sampled periodically. |
| **Sentinel Values Found** | `-999999.0`, `67115.0`, negative values | Faulty sensors or missing codes recorded as extreme numbers. Must be cleaned so neural network gradients do not explode! |
| **Target Distribution** | `CCME_Values` (Mean: 89.6, Min: 31.3, Max: 100.0) | Multi-modal with a strong spike at 100.0 (pristine water). |
| **Class Imbalance** | Excellent (50.1%), Good (31.8%), Marginal (7.6%), Fair (7.4%), Poor (3.0%) | Severe class imbalance requires stratified sampling, weighted loss, or Focal Loss for classification. |

---

## 2. The 6 Pure EDA Notebooks (Role 2 Scope)

Since our role is strictly **Role 2: Dataset Collection & EDA**, our notebooks focus 100% on exploratory analysis, data profiling, anomaly detection, and geochemical relationships. 

Data splitting, imputation pipelines, and model baseline setup belong to **Role 3: Data Preprocessing & Feature Engineering**. Therefore, our final EDA notebook (`06_target_behavior_and_leakage_audit.ipynb`) wraps up with clear **Handoff Recommendations** for the Role 3 teammate on how to perform a 70/15/15 leakage-free split.

```
nootbooks/
├── 01_data_architecture_and_profiling.ipynb
├── 02_sensor_sparsity_and_missingness.ipynb
├── 03_geochemical_physics_and_anomalies.ipynb
├── 04_bivariate_and_multivariate_correlations.ipynb
├── 05_temporal_seasonality_and_spatial_trends.ipynb
└── 06_target_behavior_and_leakage_audit.ipynb
```

---

### Notebook 1: Dataset Architecture & High-Dimensional Profiling
- **Objective**: Establish the global scope, volume, and integrity of the 4.3M-row dataset.
- **Key Analyses & Visualizations**:
  - Global dimensions, column data types, and memory optimization.
  - Country breakdown (England 49.5%, USA 43.7%, Ireland 5.5%, China 1.1%, Canada, India, Australia).
  - Waterbody type distribution (River, Stream, Effluent, Lake, Estuary, Bay, Marine).
  - Sampling frequency analysis (daily telemetry vs monthly grab samples).
- **Presentation Value**: Shows the evaluators the unprecedented geographic scale and multi-country diversity of our dataset.
- **Saved Output**: `data/GeneratedOutput/01_dataset_summary_profile.parquet`

---

### Notebook 2: Sensor Sparsity & Missing Data Mechanics
- **Objective**: Rigorously analyze why data is missing (Missing Completely at Random vs Missing at Random) and evaluate imputation options.
- **Key Analyses & Visualizations**:
  - Missingness matrix & correlation heatmap of missing values (reveals which sensors are deployed together).
  - Dendrogram clustering of missing parameters (groups physical continuous sensors vs laboratory nutrient panels).
  - Country-by-country sensor availability comparison.
  - Evaluation of imputation strategies for downstream teams:
    1. Complete-case benchmark (England/Ireland/China: 2.41M rows with 100% complete features).
    2. Median/iterative imputation for telemetry rows (USA/India).
- **Presentation Value**: Proves to the committee that missing data is handled scientifically rather than using blind averages.
- **Saved Output**: `data/GeneratedOutput/02_missingness_patterns.parquet`

---

### Notebook 3: Geochemical Physics & Anomaly Auditing
- **Objective**: Clean sensor errors and validate chemical laws before passing data to ML/DL models.
- **Key Analyses & Visualizations**:
  - **Sentinel Code Detection**: Remove sensor default codes like `-999999.0` and impossible values like `pH = 67115.0`.
  - **Detection Limit Analysis**: Address negative values (e.g. Ammonia `-0.005`) caused by instrument zero-calibration offsets.
  - **Henry’s Law Validation**: Check Dissolved Oxygen (DO) against water temperature. Colder water holds more oxygen; physically impossible combinations indicate malfunctioning sensors.
  - **Physiological pH Bounds**: Water pH in nature is bounded between 4.0 and 10.0; flag acidification or extreme alkaline events.
  - **Extreme Outlier Detection**: Compare Tukey's IQR vs Isolation Forest vs Z-score bounds.
- **Presentation Value**: Shows domain knowledge in aquatic chemistry and environmental science, not just naive code.
- **Saved Output**: `data/GeneratedOutput/03_cleaned_physically_validated.parquet`

---

### Notebook 4: Bivariate, Multivariate & Stoichiometric Relationships
- **Objective**: Uncover complex interactions between chemical parameters and explain how they influence water quality.
- **Key Analyses & Visualizations**:
  - Full Pearson (linear) and Spearman (rank/monotonic) correlation matrices.
  - **Eutrophication Dynamics**: Nitrogen-to-Phosphorus ($N:P$) stoichiometric ratios.
  - **Oxygen Sag Curve**: Relationship between Biochemical Oxygen Demand (BOD) and Dissolved Oxygen (DO) deficit (Streeter-Phelps model).
  - Temperature effects on nitrification (Ammonia &rarr; Nitrate oxidation rate).
  - Pairwise scatter matrix across key chemical indicators, color-coded by water quality grade (`CCME_WQI`).
- **Presentation Value**: Demonstrates the chemical feature interactions that the Deep Learning MLP will learn.
- **Saved Output**: `data/GeneratedOutput/04_feature_correlations.parquet`

---

### Notebook 5: Temporal Dynamics, Seasonality & Spatial Trends
- **Objective**: Quantify how water quality shifts over time, across seasons, and between waterbody types.
- **Key Analyses & Visualizations**:
  - Multi-decade historical evolution (1940s to 2023) showing long-term environmental recovery.
  - Seasonal decomposition (Spring agricultural runoff vs Summer algae blooms vs Winter freeze).
  - Monthly variation boxplots for temperature, dissolved oxygen, and nitrates.
  - Waterbody comparison: Rivers vs Streams vs Lakes vs Coastal Waters vs Effluent discharge.
- **Presentation Value**: Proves that water quality is not static and explains why time-aware modeling is necessary.
- **Saved Output**: `data/GeneratedOutput/05_temporal_spatial_aggregates.parquet`

---

### Notebook 6: Target Metric Behavior & Leakage Risk Audit
- **Objective**: Deep-dive into the target variables (`CCME_Values` and `CCME_WQI`) and perform an AML/DL leakage audit.
- **Key Analyses & Visualizations**:
  - Distribution analysis of `CCME_Values` (identifying the spike at 100.0 for pristine water and tail towards 31.3).
  - Class distribution analysis of `CCME_WQI` (measuring the 50.1% Excellent vs 3.0% Poor class imbalance).
  - **Target Leakage Analysis**:
    - The CCME formula is a mathematical combination of feature standard exceedances.
    - We document how a model can either learn the proxy index from subset features or predict directly from raw sensor signals without data leakage.
  - Recommendations for model training: Class-weighted Cross-Entropy, Focal Loss, and Stratified sampling.
- **Presentation Value**: Directly addresses the project requirement: *"do not derive a target index from the same components without acknowledging leakage."*
- **Saved Output**: `data/GeneratedOutput/06_target_audit.parquet`

---

### Notebook 6: Target Metric Behavior, Leakage Audit & Handoff to Role 3
- **Objective**: Deep-dive into the target variables (`CCME_Values` and `CCME_WQI`), perform an AML/DL leakage audit, and formulate the **70 / 15 / 15 Splitting Guidelines** for Role 3.
- **Key Analyses & Visualizations**:
  - Distribution analysis of `CCME_Values` (identifying the spike at 100.0 for pristine water and tail towards 31.3).
  - Class distribution analysis of `CCME_WQI` (measuring the 50.1% Excellent vs 3.0% Poor class imbalance).
  - **Target Leakage Analysis**:
    - The CCME formula is a mathematical combination of feature standard exceedances.
    - We document how a model can either learn the proxy index from subset features or predict directly from raw sensor signals without data leakage.
  - **Handoff to Role 3 (Preprocessing & Feature Engineering)**:
    - Recommendations for handling the severe class imbalance (Class-weighted Cross-Entropy, Focal Loss).
    - Guidelines for executing a **70 / 15 / 15 Train / Val / Test split** (stratified across Country, Waterbody Type, and WQI Class to prevent temporal and spatial data leakage).
- **Presentation Value**: Directly addresses the project requirement: *"do not derive a target index from the same components without acknowledging leakage"* and cleanly establishes our team handoff.
- **Saved Output**: `data/GeneratedOutput/06_target_audit_and_splitting_guidelines.parquet`

---

## 3. Summary of Deliverables & Output Files

| Notebook | Focus | Key Output File (`data/GeneratedOutput/`) |
|---|---|---|
| `01_data_architecture_and_profiling.ipynb` | Global volume & country breakdown | `01_dataset_summary_profile.parquet` |
| `02_sensor_sparsity_and_missingness.ipynb` | Missing value patterns & mechanisms | `02_missingness_patterns.parquet` |
| `03_geochemical_physics_and_anomalies.ipynb` | Sentinel code cleanup & physical bounds | `03_cleaned_physically_validated.parquet` |
| `04_bivariate_and_multivariate_correlations.ipynb` | Chemical interactions & correlations | `04_feature_correlations.parquet` |
| `05_temporal_seasonality_and_spatial_trends.ipynb` | Seasonal cycles & spatial variations | `05_temporal_spatial_aggregates.parquet` |
| `06_target_behavior_and_leakage_audit.ipynb` | CCME WQI behavior, leakage audit & Role 3 splitting handoff | `06_target_audit_and_splitting_guidelines.parquet` |

---

## 4. How to Present This Work

When presenting this project to evaluators:
1. **Show the 6-Notebook EDA Architecture**: Emphasize that our team built a comprehensive 6-stage data science investigation that systematically covers every aspect of environmental water quality.
2. **Highlight the Discovery of Sensor Anomalies**: Show before-and-after plots of discovering `-999999.0` sentinels and `pH 67115.0` outliers that would otherwise crash deep learning models.
3. **Show the Chemical Science**: Present the Henry's Law DO-Temperature curve and the $N:P$ eutrophication ratio.
4. **Demonstrate Methodological Rigor & Clean Team Handoff**: Show how our EDA directly warned against data leakage and handed off mathematically sound **70 / 15 / 15 Stratification Guidelines** to the Preprocessing & Modeling teammates.

