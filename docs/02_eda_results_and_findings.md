# Exploratory Data Analysis (EDA) — Key Results & Findings

> **Role**: Role 2 — Dataset Collection & EDA  
> **Notebooks Location**: [`nootbooks/01.EDA/`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/)  
> **Generated Outputs**: [`data/GeneratedOutput/01.EDA/`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/)  
> **Status**: Complete & Executed  

---

## Executive Summary

We have completed the full Exploratory Data Analysis (EDA) suite across **6 modular Jupyter notebooks**, analyzing **4,297,895 water quality records** spanning **83 years (1940–2023)** across **7 countries**.

All notebooks are saved with executed code and rendered visualizations in `nootbooks/01.EDA/`. Every analytical table has been exported as a `.parquet` file with standard `snappy` compression in `data/GeneratedOutput/01.EDA/` for easy viewing in the VS Code **Parquetly** extension.

---

## Summary of Findings by Notebook

### Notebook 1: Data Architecture & High-Dimensional Profiling
- **File**: [`01_data_architecture_and_profiling.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/01_data_architecture_and_profiling.ipynb)
- **Output**: [`01_dataset_summary_profile.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/01_dataset_summary_profile.parquet)
- **What We Found**:
  - The dataset contains **4,297,895 rows** and **14 columns** across **64,665 unique monitoring stations**.
  - **Country Share**: England holds 49.5% (2.13M rows), USA holds 43.7% (1.88M rows), Ireland holds 5.5% (235k rows), China holds 1.1% (46k rows), with Canada, India, and Australia making up the rest.
  - **Waterbodies**: Rivers (43.2%) and Streams (34.1%) are the dominant freshwater environments, followed by Effluent discharge (14.0%) and Lakes (3.6%).
  - **Time Horizon**: Measurements span from 1940 to 2023. Over **98% of records** are from the modern digital monitoring surge (2000–2023).

---

### Notebook 2: Sensor Sparsity & Missing Data Mechanics
- **File**: [`02_sensor_sparsity_and_missingness.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb)
- **Output**: [`02_missingness_patterns.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/02_missingness_patterns.parquet)
- **What We Found**:
  - **England, Ireland, and China** provide **2,410,214 complete-case records** with **0% missing values** across all 8 chemical parameters.
  - **USA and India** focus on high-frequency in-situ telemetry (`Temperature` 99.7% present, `Dissolved Oxygen` 39.3%, `pH` 36.1%), with laboratory nutrients (Nitrate, Ammonia, BOD) sampled intermittently.
  - **Sensor Co-Occurrence**: Nutrients (`Ammonia`, `Orthophosphate`, `Nitrogen`, `Nitrate`) drop together in synchronized laboratory batches ($r > 0.99$), confirming the missingness is **Missing at Random (MAR)** rather than random sensor glitching.
  - **Recommendation for Role 3**: The complete-case subset provides a gold-standard benchmark; for the telemetry rows, iterative or group-median imputation is recommended.

---

### Notebook 3: Geochemical Physics & Anomaly Auditing
- **File**: [`03_geochemical_physics_and_anomalies.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb)
- **Output**: [`03_cleaned_physically_validated.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/03_cleaned_physically_validated.parquet)
- **What We Found**:
  - **Hardware Error Codes**: Discovered `-999999.0` sentinels in Temperature (81 rows), pH (3 rows), and DO (1 row). These must be cleaned to prevent neural network gradients from exploding.
  - **Scale Corruptions**: Found extreme values like `pH = 67115.0`. Natural freshwater pH is physically bounded between 3.0 and 12.0.
  - **Henry’s Law Validation**: Validated the physical dissolved oxygen solubility curve against temperature. Normal cold water holds up to ~14.6 mg/L of oxygen; identified rare super-saturation events (> 150%) caused by heavy algae photosynthetic blooms.
  - **Clean Physical Filter**: Built a filter that removes corrupt sensor codes while safely preserving real industrial effluent pollution spikes.

---

### Notebook 4: Bivariate & Multivariate Chemical Relationships
- **File**: [`04_bivariate_and_multivariate_correlations.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb)
- **Output**: [`04_feature_correlations.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/04_feature_correlations.parquet)
- **What We Found**:
  - **Correlations with Water Quality (`CCME_Values`)**:
    - `Orthophosphate`: $r = -0.66$ (strongest negative correlation; excess phosphorus drives eutrophication).
    - `Nitrogen`: $r = -0.61$ (excess nitrogen degrades water quality).
    - `Ammonia`: $r = -0.39$ (toxic breakdown product).
    - `Biochemical Oxygen Demand (BOD)`: $r = -0.32$ (high organic waste depletes oxygen).
  - **Eutrophication Stoichiometry ($N:P$ Ratio)**:
    - Compared observed nitrogen-to-phosphorus ratios against the Redfield benchmark (16:1).
    - 58.4% of waters are phosphorus-limited ($N:P > 16$), meaning phosphorus runoff is the primary trigger for algae blooms.
  - **Streeter-Phelps Dynamics**: As BOD rises above 10 mg/L, dissolved oxygen drops significantly, visualizing classic river oxygen sag dynamics.

---

### Notebook 5: Temporal Dynamics, Seasonality & Spatial Variations
- **File**: [`05_temporal_seasonality_and_spatial_trends.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb)
- **Output**: [`05_temporal_spatial_aggregates.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/05_temporal_spatial_aggregates.parquet)
- **What We Found**:
  - **Long-Term Trajectory (1970–2023)**: Median water quality showed steady improvement from the 1980s through the 2010s due to national clean water legislation in the UK and USA.
  - **Seasonal Cycles**:
    - **Temperature**: Peaks in July/August (~18°C) and bottoms out in January/February (~6°C).
    - **Dissolved Oxygen**: Inversely mirrors temperature, peaking in winter (~11.5 mg/L) and dipping in late summer (~9.0 mg/L).
    - **Nitrates**: Spike in winter and early spring (agricultural fertilizer runoff from rain) and drop in summer due to biological plant uptake.
  - **Waterbody Differences**: Effluent discharge and canals show significantly lower median WQI scores (~75–80) compared to open rivers and lakes (~90–98).

---

### Notebook 6: Target Metric Behavior & Handoff to Role 3
- **File**: [`06_target_behavior_and_leakage_audit.ipynb`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/nootbooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb)
- **Output**: [`06_target_audit_and_splitting_guidelines.parquet`](file:///Users/mayanksharma/Downloads/New_Projects/Semester_5_AML_DL/data/GeneratedOutput/01.EDA/06_target_audit_and_splitting_guidelines.parquet)
- **What We Found**:
  - **Target Score (`CCME_Values`)**: The distribution has a sharp spike at 100.0 (over 50% of records have 0 exceedances). The median is 100.0 and mean is 89.6.
  - **Class Imbalance (`CCME_WQI`)**:
    - **Excellent**: 50.1% (2,155,328)
    - **Good**: 31.8% (1,368,562)
    - **Marginal**: 7.6% (328,795)
    - **Fair**: 7.4% (318,228)
    - **Poor**: 3.0% (126,982)
  - **Target Leakage Warning**: Because CCME WQI is calculated algebraically from standard exceedances, feeding all 8 clean parameters into a model allows it to trivially reverse-engineer the math. The true AML/DL challenge is predicting WQI from physical probe proxies (`Temp`, `pH`, `DO`) or forecasting future WQI.
  - **Handoff Specifications for Role 3**:
    - **Train Set (70%)**: ~3,008,526 rows
    - **Validation Set (15%)**: ~644,684 rows
    - **Test Set (15%)**: ~644,685 rows
    - Must be stratified across `Country`, `Waterbody Type`, and `CCME_WQI` class labels.

---

## 3. Directory of Artifacts Generated

| Artifact Type | Path | Description |
|---|---|---|
| **Notebook 1** | `nootbooks/01.EDA/01_data_architecture_and_profiling.ipynb` | Profiling, scale, country & waterbody analysis |
| **Output 1** | `data/GeneratedOutput/01.EDA/01_dataset_summary_profile.parquet` | Summary metrics table (Snappy) |
| **Notebook 2** | `nootbooks/01.EDA/02_sensor_sparsity_and_missingness.ipynb` | Missingness matrices & sensor co-occurrence |
| **Output 2** | `data/GeneratedOutput/01.EDA/02_missingness_patterns.parquet` | Country missingness percentages (Snappy) |
| **Notebook 3** | `nootbooks/01.EDA/03_geochemical_physics_and_anomalies.ipynb` | Sentinel code cleanup & Henry's Law curve |
| **Output 3** | `data/GeneratedOutput/01.EDA/03_cleaned_physically_validated.parquet` | Cleaned data summary profile (Snappy) |
| **Notebook 4** | `nootbooks/01.EDA/04_bivariate_and_multivariate_correlations.ipynb` | Pearson/Spearman heatmaps & N:P ratios |
| **Output 4** | `data/GeneratedOutput/01.EDA/04_feature_correlations.parquet` | Feature correlation matrix (Snappy) |
| **Notebook 5** | `nootbooks/01.EDA/05_temporal_seasonality_and_spatial_trends.ipynb` | Seasonal cycles & multi-decade trends |
| **Output 5** | `data/GeneratedOutput/01.EDA/05_temporal_spatial_aggregates.parquet` | Monthly chemical aggregates (Snappy) |
| **Notebook 6** | `nootbooks/01.EDA/06_target_behavior_and_leakage_audit.ipynb` | Class imbalance, leakage audit & Role 3 specs |
| **Output 6** | `data/GeneratedOutput/01.EDA/06_target_audit_and_splitting_guidelines.parquet` | 70/15/15 Splitting specifications (Snappy) |
