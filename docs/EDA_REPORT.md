# Exploratory Data Analysis (EDA) Comprehensive Report

> **Project**: Project 18 — Water Quality Prediction via Gradient Boosting & Deep Neural Networks  
> **Course**: Semester 5 Applied Machine Learning & Deep Learning (AML / DL)  
> **Role Responsible**: Role 2 — Dataset Collection & Exploratory Data Analysis  
> **Executed Notebooks**: [`notebooks/01.EDA/`](../notebooks/01.EDA/)  
> **Analytical Parquet Outputs**: [`data/GeneratedOutput/01.EDA/`](../data/GeneratedOutput/01.EDA/)  
> **Dataset**: 4,297,895 records × 14 columns across 7 countries (1940–2023)  

---

## 1. Executive Summary & Objective

This document provides a single, unified reference for the entire Exploratory Data Analysis (EDA) performed on the global surface water quality monitoring dataset. 

Our investigation analyzed **4,297,895 records** across **64,665 unique monitoring stations** over an **83-year timeline (1940–2023)**. Rather than relying on simple summary statistics, our analysis was structured into 6 modular stages covering sensor physics, missingness mechanisms, geochemical laws (Henry's Law, nutrient stoichiometry), long-term climate trajectories, and machine learning data leakage risks.

All 6 notebooks have been executed with embedded high-resolution figures in `notebooks/01.EDA/`, and every analytical table is saved as a Snappy-compressed `.parquet` file in `data/GeneratedOutput/01.EDA/`.

---

## 2. Global Dataset Profile & Architecture

| Characteristic | Key Metric / Value | Significance for AML / DL Modeling |
|---|---|---|
| **Total Observations** | 4,297,895 rows | Massive dataset scale suitable for deep neural networks (MLP) and gradient boosted trees (XGBoost). |
| **Feature Dimensions** | 14 columns | 4 metadata attributes, 8 chemical predictors ($X$), and 2 target indices ($y$). |
| **Unique Stations (`Area`)** | 64,665 monitoring points | High spatial heterogeneity; requires grouped/stratified splitting to prevent spatial memorization. |
| **Temporal Span** | 1940 to 2023 (83 years) | Covers historical benchmarks (>98% concentrated in the digital monitoring era 2000–2023). |
| **Country Distribution** | 7 nations | England: 49.5%, USA: 43.7%, Ireland: 5.5%, China: 1.1%, Canada, India, Australia: <0.3%. |
| **Waterbody Breakdown** | 13 categories | Rivers (43.2%), Streams (34.1%), Effluent (14.0%), Lakes (3.6%), Estuaries (1.2%), Marine (0.5%). |
| **Storage & I/O** | 44.9 MB (Snappy Parquet) | Optimized columnar binary format; loads in <2.5 seconds in Pandas/DuckDB and opens in VS Code Parquetly. |

*Generated Output*: [`data/GeneratedOutput/01.EDA/01_dataset_summary_profile.parquet`](../data/GeneratedOutput/01.EDA/01_dataset_summary_profile.parquet)  
*Notebook*: [`notebooks/01.EDA/01_data_architecture_and_profiling.ipynb`](../notebooks/01.EDA/01_data_architecture_and_profiling.ipynb)

---

## 3. Sensor Sparsity & Missing Data Mechanics

Our audit revealed a critical structural divergence between laboratory monitoring panels and continuous in-situ telemetry:

1. **Complete Laboratory Panels (Zero Missingness)**:
   - **England (2.13M rows), Ireland (235k rows), and China (46k rows)** have **0.0% missing values** across all 8 chemical parameters.
   - Together, these form a **2,410,214-row complete-case gold standard** benchmark.
2. **Continuous Physical Telemetry vs Grab Samples**:
   - **USA (1.88M rows)** and **India (3.6k rows)** continuously monitor physical probes (`Temperature` 99.7% present, `Dissolved Oxygen` 39.3%, `pH` 36.1%), while chemical nutrients (`Ammonia`, `BOD`, `Nitrate`, `Orthophosphate`) are sampled intermittently during discrete campaigns.
3. **Sensor Co-Occurrence Patterns**:
   - Nutrients (`Ammonia`, `Orthophosphate`, `Nitrogen`, `Nitrate`) drop together in synchronized laboratory batches with pairwise correlation $r > 0.99$.
   - This proves the missingness is **Missing at Random (MAR)** governed by operational protocols, rather than random hardware dropouts.
4. **Recommendations for Role 3 (Preprocessing)**:
   - Train baseline tree models on the 2.41M complete cases.
   - For nationwide telemetry rows, apply iterative / KNN imputation or construct a physical-probe feature branch.

*Generated Output*: [`data/GeneratedOutput/01.EDA/02_missingness_patterns.parquet`](../data/GeneratedOutput/01.EDA/02_missingness_patterns.parquet)  
*Notebook*: [`notebooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb`](../notebooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb)

---

## 4. Geochemical Physics & Anomaly Auditing

To protect machine learning models from corrupt gradient updates, we audited the physical plausibility of all measurements:

1. **Hardware Error Sentinels**:
   - Detected sensor default codes recorded as **`-999999.0`** in Temperature (81 records), pH (3 records), and Dissolved Oxygen (1 record).
   - If not removed, these extreme negative numbers distort feature scaling and cause neural network gradients to explode.
2. **Sensor Scale Corruptions**:
   - Identified sensor calibration bugs such as **`pH = 67115.0`**. Natural surface freshwater pH is strictly bounded between 3.0 and 12.0 (typically 6.5–8.5).
3. **Detection Limits & Zero Offsets**:
   - Flagged small negative concentrations (e.g. Ammonia `-0.005 mg/L`, BOD `-2.0 mg/L`) representing instrument baseline zero-calibration offsets below detection limits.
4. **Henry's Law Oxygen Saturation Validation**:
   - Validated aquatic physics: colder water holds higher dissolved oxygen.
   - Compared empirical DO against the theoretical saturation formula:
     $$\text{DO}_{\text{sat}}(T) \approx 14.652 - 0.41022 T + 0.007991 T^2 - 0.000077774 T^3$$
   - Confirmed the majority of records track the 100% saturation curve; isolated rare super-saturation events (>150%) caused by intense photosynthetic algae blooms.
5. **Clean Physically Validated Filter**:
   - Applied physical constraint boundaries that safely eliminate corrupt sentinels while preserving genuine heavy pollution spikes in industrial effluents.

*Generated Output*: [`data/GeneratedOutput/01.EDA/03_cleaned_physically_validated.parquet`](../data/GeneratedOutput/01.EDA/03_cleaned_physically_validated.parquet)  
*Notebook*: [`notebooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb`](../notebooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb)

---

## 5. Bivariate & Multivariate Chemical Relationships

We evaluated both linear (Pearson) and monotonic non-linear (Spearman) relationships among chemical variables and the target index:

1. **Correlation with Water Quality Score (`CCME_Values`)**:
   - **Orthophosphate**: $r = -0.66$ (strongest negative driver; excess phosphorus promotes harmful algae growth).
   - **Total Nitrogen**: $r = -0.61$ (excess agricultural nutrients severely degrade water quality).
   - **Ammonia**: $r = -0.39$ (toxic organic breakdown byproduct).
   - **Biochemical Oxygen Demand (BOD)**: $r = -0.32$ (microbial oxygen consumption).
   - **Dissolved Oxygen**: Positive monotonic association with cleaner water grades.
2. **Eutrophication Stoichiometry ($N:P$ Ratio)**:
   - Evaluated the classical **Redfield ratio benchmark (16:1 by mass)** across all paired nutrient samples.
   - **58.4% of waters are Phosphorus-Limited ($N:P > 16$)**, meaning phosphorus control is the primary environmental intervention needed to curb eutrophication.
3. **Streeter-Phelps Oxygen Sag Curve**:
   - As Biochemical Oxygen Demand (BOD) rises above 10 mg/L, Dissolved Oxygen drops sharply, validating classical river oxygen deficit models.

*Generated Output*: [`data/GeneratedOutput/01.EDA/04_feature_correlations.parquet`](../data/GeneratedOutput/01.EDA/04_feature_correlations.parquet)  
*Notebook*: [`notebooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb`](../notebooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb)

---

## 6. Temporal Dynamics, Seasonality & Spatial Variations

1. **Long-Term Decadal Trajectory (1970–2023)**:
   - Median annual water quality index improved consistently from the 1980s through the 2010s, reflecting the long-term impact of clean water legislation and sewage treatment upgrades.
2. **Monthly Seasonal Periodicities**:
   - **Water Temperature**: Peaks in July/August (~18°C) and reaches minimums in January/February (~6°C).
   - **Dissolved Oxygen**: Inversely mirrors temperature, peaking in winter (~11.5 mg/L) due to higher solubility and dipping in late summer (~9.0 mg/L).
   - **Nitrates**: Spike in late winter and spring due to agricultural fertilizer runoff washed into rivers by seasonal rainfall, dropping in summer as aquatic vegetation absorbs nutrients.
3. **Waterbody Differences**:
   - Industrial effluent discharge and canals exhibit significantly lower median scores (~75–80) with high variability.
   - Natural flowing rivers and protected lakes maintain higher median quality scores (~90–98).

*Generated Output*: [`data/GeneratedOutput/01.EDA/05_temporal_spatial_aggregates.parquet`](../data/GeneratedOutput/01.EDA/05_temporal_spatial_aggregates.parquet)  
*Notebook*: [`notebooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb`](../notebooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb)

---

## 7. Target Metric Behavior & Data Leakage Audit

### Target Distributions
- **Continuous Score (`CCME_Values`)**:
  - Bound between `31.3` and `100.0` (Mean: 89.6, Median: 100.0).
  - Shows a pronounced spike at 100.0 (over 50% of records exhibit zero standard exceedances).
- **Classification Categories (`CCME_WQI`)**:
  - **Excellent**: 50.1% (2,155,328 records)
  - **Good**: 31.8% (1,368,562 records)
  - **Marginal**: 7.6% (328,795 records)
  - **Fair**: 7.4% (318,228 records)
  - **Poor**: 3.0% (126,982 records)

### The AML / DL Data Leakage Warning
The CCME Water Quality Index is an algebraic formulation combining Scope ($F_1$), Frequency ($F_2$), and Amplitude ($F_3$) calculated from chemical threshold exceedances:
$$\text{WQI} = 100 - \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732}$$
> [!WARNING]
> If a machine learning model is trained on all 8 clean constituent parameters to predict CCME WQI, it will merely memorize the algebraic arithmetic rather than learning environmental chemistry. The authentic machine learning objective is predicting water quality from **sparse physical sensor proxies** (`Temperature`, `pH`, `Dissolved Oxygen`) or **forecasting future water quality** from historical lag measurements.

*Generated Output*: [`data/GeneratedOutput/01.EDA/06_target_distributions_and_leakage_audit.parquet`](../data/GeneratedOutput/01.EDA/06_target_distributions_and_leakage_audit.parquet)  
*Notebook*: [`notebooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb`](../notebooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb)

---

## 8. Directory of Generated EDA Artifacts

All outputs are saved with standard `snappy` compression:

```
data/GeneratedOutput/01.EDA/
├── 01_dataset_summary_profile.parquet                 # Global dataset architectural metrics
├── 02_missingness_patterns.parquet                    # Country sensor missingness rates
├── 03_cleaned_physically_validated.parquet            # Anomaly & sentinel removal metrics
├── 04_feature_correlations.parquet                    # Spearman rank correlation matrix
├── 05_temporal_spatial_aggregates.parquet             # Monthly seasonal parameter profiles
├── 06_target_distributions_and_leakage_audit.parquet  # Target distributions & algebraic leakage manifest
└── final_eda_cleaned_dataset.parquet                  # PRIMARY PROCESSED DATASET for Role 3 (4,290,073 clean rows)
```
