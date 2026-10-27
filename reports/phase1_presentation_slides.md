---
marp: true
theme: default
paginate: true
header: "Semester 5 AML | Project 18: Water Quality Index Estimation (Phase 1)"
footer: "6-Member Team (Lead: Mayank Sharma) | GitHub: @MAYANKSHARMA01010/Semester_5_AML_DL"
style: |
  section {
    background-color: #0b1325;
    color: #f8fafc;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  h1 { color: #06b6d4; }
  h2 { color: #3b82f6; }
  h3 { color: #f59e0b; }
  a { color: #06b6d4; }
  table { font-size: 0.8rem; }
  th { background-color: #1e293b; color: #06b6d4; }
  td { border-color: #334155; }
---

<!-- Slide 1 -->
# Surface Water Quality Index Estimation
### Phase 1 Machine Learning Evaluation via Gradient Tree Boosting & Stoichiometry

**Course:** Semester 5 Applied Machine Learning (AML) | **Project 18**  
**Repository:** [github.com/MAYANKSHARMA01010/Semester_5_AML_DL](https://github.com/MAYANKSHARMA01010/Semester_5_AML_DL)  
**Date:** October 2026  

**Project Team (6 Members):**
1. **Mayank Sharma (Lead):** AML Pipeline Architecture, Random Forest & XGBoost, GitHub Orchestration
2. **Team Member 2:** Literature Review, CCME WQI Formulation ($F_1, F_2, F_3, nse$) & Regulatory Standards
3. **Team Member 3:** Multi-National Dataset Auditing (5 Nations), Physical Bounds & Exploratory Data Analysis
4. **Team Member 4:** Aquatic Chemical Feature Engineering, Stoichiometry (N:P, DO Sat, Emerson $\text{NH}_3$)
5. **Team Member 5:** Tree-Based Hyperparameter Optimization, Stratified Splitting & Benchmarking
6. **Team Member 6:** Explainable AI via TreeSHAP, Metric Verification & Automated PyTest Suite

---

<!-- Slide 2 -->
# The Environmental Monitoring Crisis
### Conventional Wet-Lab Delays vs. Real-Time Machine Learning

- **Incubation Latency:** Standard assays like 5-day Biochemical Oxygen Demand ($\text{BOD}_5$) require 120 hours of laboratory incubation.
- **High Operational Expense:** Manual grab-sampling requires specialized field crews, transport logistics, and recurring chemical reagents.
- **Discrete Temporal Blindspots:** Monthly sampling cadences miss acute, transient toxic chemical discharges and storm runoff pulses.
- **The Machine Learning Paradigm:**
  - Real-time in-situ sensor telemetry (pH, Temperature, Dissolved Oxygen, Conductivity).
  - Sub-millisecond continuous prediction of the **CCME Water Quality Index** ($0 - 100$).
  - Instant classification into 5 standardized ecological tiers (Excellent to Poor).
  - Scalable early warning telemetry for municipal water treatment plants and conservation agencies.

---

<!-- Slide 3 -->
# Literature Review & Gap Analysis
### Theoretical Foundations and Methodological Advances

- **Breiman (2001) - Random Forests:** Bootstrap aggregation with randomized feature subspaces; reduces variance and handles tabular noise.
- **Chen & Guestrin (2016) - XGBoost:** Regularized second-order gradient boosting for scalable tabular modeling.
- **CCME (2001) - Water Quality Index 1.0:** Internationally endorsed framework integrating Scope ($F_1$), Frequency ($F_2$), and Amplitude ($F_3$).
- **Lundberg & Lee (2017) - TreeSHAP:** Exact polynomial-time Shapley values grounded in cooperative game theory.

**Critical Gaps Addressed in our Work:**
1. *Sensor Corruption:* Prior studies ingest electronic dropouts. -> We enforce thermodynamic bounds, purging 8,360 implausible values.
2. *Target Leakage:* Studies include intermediate exceedance counters. -> We enforce strict programmatic quarantine on 38 predictors.
3. *Nutrient Skewness:* Raw orthophosphate spans 4 orders of magnitude. -> We implement $\log(1+x)$, boosting linear correlation from $-0.66$ to $\mathbf{-0.85}$.
4. *Station Memory:* Models ignore local history. -> We engineer backward-only shift-1 station lags ($\rho = -0.78$ with WQI).

---

<!-- Slide 4 -->
# Multi-National Dataset Architecture
### 2,827,977 Observations Across 5 Nations (1940–2023)

| Country | Raw Records | Clean Records | Proportion | Primary Source |
|---|---|---|---|---|
| **England** | 2,367,978 | 2,361,042 | 83.74% | Environment Agency Catchments |
| **USA** | 240,654 | 239,890 | 8.51% | USGS Surface Telemetry |
| **Ireland** | 201,345 | 200,980 | 7.13% | EPA River & Lake Monitors |
| **China** | 12,000 | 11,985 | 0.43% | Major River Basins |
| **Canada** | 6,000 | 5,720 | 0.20% | Historical Reference Telemetry |
| **Total** | **2,827,977** | **2,819,617** | **100.00%** | **83-Year Multi-National Corpus** |

- **Zero Missingness Verified:** Disproved legacy documentation claims of "35% missingness"; verified 100% complete-case ground truth.
- **Transparent Parquet Sharding:** Resolved GitHub's 100 MB file limit via transparent 3-part Parquet files under 60 MB.

---

<!-- Slide 5 -->
# Data Quality Audit & Physical Bounds Screening
### Purging 8,360 Physically Impossible Telemetry Errors

| Parameter | Valid Physical Envelope | Scientific Rationale |
|---|---|---|
| **pH** | $[3.0, 12.0]$ | Natural thermodynamic limit outside acid mine drainage or caustic spills. |
| **Temperature** | $[0.0^\circ\text{C}, 45.0^\circ\text{C}]$ | Constrained by the liquid state of surface freshwater at 1 atm. |
| **Dissolved Oxygen** | $[0.0, 25.0\text{ mg/L}]$ | Henry's law atmospheric gas solubility ceiling. |
| **BOD$_5$** | $\ge 0.0\text{ mg/L}$ | Non-negative microbial oxygen consumption. |
| **Ammonia / Ammonium** | $\ge 0.0\text{ mg/L}$ | Non-negative chemical concentration. |
| **Orthophosphate** | $\ge 0.0\text{ mg/L}$ | Non-negative chemical nutrient concentration. |
| **Nitrate / Total Nitrogen** | $\ge 0.0\text{ mg/L}$ | Non-negative chemical concentration. |

- **Audit Findings:** Purged exactly **8,360 corrupted observations** (0.296% of corpus) containing electronic dropouts (-999, 9999).
- **Result:** **2,819,617 pristine complete-case observations** with zero synthetic imputation bias.

---

<!-- Slide 6 -->
# The CCME Water Quality Index Formulation
### Continuous Mathematical Aggregation into Standardized Ecological Tiers

$$\text{CCME WQI} = 100 - \left( \frac{\sqrt{F_1^2 + F_2^2 + F_3^2}}{1.732} \right)$$

1. **$F_1$ (Scope):** Percentage of variables failing regulatory thresholds:
   $$F_1 = \left( \frac{\text{Failed Variables}}{\text{Total Variables}} \right) \times 100$$
2. **$F_2$ (Frequency):** Percentage of individual tests failing compliance:
   $$F_2 = \left( \frac{\text{Failed Tests}}{\text{Total Tests}} \right) \times 100$$
3. **$F_3$ (Amplitude):** Magnitude of exceedance via normalized sum of excursions ($nse$):
   $$nse = \frac{\sum \text{excursion}_i}{\text{Total Tests}}, \quad F_3 = \frac{nse}{0.01 \cdot nse + 0.01}$$

**5 Regulatory Tiers:** Excellent (95–100), Good (80–94), Fair (65–79), Marginal (45–64), Poor (0–44).

---

<!-- Slide 7 -->
# Domain Feature Engineering: Aquatic Chemistry
### Integrating Physical Laws and Stoichiometry (38 Total Features)

- **Benson-Krause Dissolved Oxygen Saturation:**
  $$\text{DO}_{\text{sat}}(T) = 14.652 - 0.41022\,T + 0.007991\,T^2 - 0.00007777\,T^3$$
  $$\text{DO}_{\%sat} = (\text{DO} / \text{DO}_{\text{sat}}) \times 100, \quad \text{DO}_{\text{deficit}} = \max(0, \text{DO}_{\text{sat}} - \text{DO})$$
- **Redfield Stoichiometric Ratio:** $\text{N:P} = \text{Nitrogen} / (\text{Orthophosphate} + 10^{-5})$ (identifies algal bloom limitations).
- **Emerson Un-Ionized Ammonia Equilibrium:** Quantifies toxic aqueous $\text{NH}_3$ as a function of temperature and pH.
- **Streeter-Phelps Organic Load Ratio:** $\text{BOD}/\text{DO}$ ratio measuring microbial respiratory pressure.
- **Feature Group Summary:** 8 Base + 7 Stoichiometric + 7 $\log(1+x)$ + 6 Station Lags + 1 Exceedance Stress + 9 Temporal/Context = **38 Features**.

---

<!-- Slide 8 -->
# Non-Linear Skewness Mitigation: The log(1+x) Transform
### Compressing Heavy Positive Nutrient Tails

- **The Distributional Problem:** Orthophosphate, Ammonia, and $\text{BOD}_5$ exhibit extreme positive skewness ($\gamma_1 > 40$) spanning 4 orders of magnitude.
- **Mathematical Transformation:**
  $$\tilde{x} = \ln(1 + x)$$
- **Impact on Pearson Correlation with CCME WQI:**
  - Raw Orthophosphate: $r = -0.66$
  - $\log(1+x)$ Orthophosphate: $\mathbf{r = -0.85}$ (significant linear relationship boost).
- **Algorithmic Benefits:**
  - Prevents decision trees from wasting split depth on extreme outliers.
  - Stabilizes gradient step calculations in XGBoost.
  - Preserves strict non-negativity without synthetic distortion.

---

<!-- Slide 9 -->
# Chronological Station Memory (Zero Lookahead Leakage)
### Capturing Hydrological Continuity and Seasonal Cycles

- **Hydrological Persistence:** Upstream contamination plumes persist over successive sampling events.
- **Shift-1 Station Lags:** Generated $\text{lag}_1(x_t) = x_{t-1}$ grouped strictly by monitoring station ID.
- **Rolling-3 Moving Averages:** Computed 3-sample rolling means capturing localized environmental baselines.
- **Zero Forward-Looking Leakage:** Datasets sorted chronologically; shifts look strictly backwards.
- **High Predictive Correlation:** Shift-1 orthophosphate lag ($\text{hist\_ortho\_lag1}$) achieves rank correlation $\mathbf{\rho = -0.78}$ with CCME WQI.
- **Harmonic Calendar Encoding:** Encoded Month and Day-of-Year as $\sin(2\pi t/T)$ and $\cos(2\pi t/T)$ to eliminate December 31 / January 1 boundary discontinuity.

---

<!-- Slide 10 -->
# Strict Anti-Leakage Protocol & Splitting Strategy
### Target Quarantine and Class-Stratified Partitions

- **The Target Leakage Hazard:** CCME WQI is calculated from chemical parameters. Retaining intermediate calculation columns creates circular leakage.
- **Programmatic Target Quarantine:** Quarantined all targets via `src.utils.separate_features_and_targets`. Automated assertions enforce zero target columns inside feature matrix.
- **Zero Split Leakage:** Scaling parameters and transformations fitted strictly on training data.
- **Stratified 70/15/15 Splitting:**
  - **Training Set:** 1,973,731 observations (70.0%)
  - **Validation Set:** 422,943 observations (15.0%)
  - **Test Set:** 422,943 observations (15.0%)
  - Guarantees identical ecological tier distribution across all splits.

---

<!-- Slide 11 -->
# Phase 1 Machine Learning Model Benchmarks
### Empirical Results on 422,943 Validation Observations

| Model Architecture | Train MAE | Train RMSE | Train $R^2$ | Val MAE | Val RMSE | Val $R^2$ | Inference Time |
|---|---|---|---|---|---|---|---|
| **Mean Baseline** | 14.131 | 17.660 | 0.0000 | 14.130 | 17.656 | 0.0000 | 0.02 s |
| **XGBoost Regressor** | 0.136 | 0.368 | 0.9996 | **0.137** | **0.374** | **0.9996** | **0.42 s (9x Faster)** |
| **Random Forest** | 0.047 | 0.172 | 0.9999 | **0.052** | **0.208** | **0.9999** | 3.85 s |

- **Error Reduction:** Both ensemble architectures eliminate $> 99.6\%$ of baseline variance.
- **Random Forest:** Achieves best precision ($\text{MAE} = 0.052$) via 100 de-correlated trees.
- **XGBoost:** Near-identical accuracy ($\text{MAE} = 0.137$) with $9\times$ faster inference latency ($0.42\text{ s}$ for 422k rows $\approx 1\,\mu\text{s}$ per sensor).

---

<!-- Slide 12 -->
# Explainable AI (XAI) via TreeSHAP Attribution
### Uncovering Non-Linear Ecological Drivers

- **Orthophosphate Dominance:** $\log(1+\text{PO}_4^{3-})$ and raw orthophosphate account for over **88% of total split importance**.
- **Limiting Nutrient Mechanism:** Phosphorus is freshwater's critical limiting nutrient; elevated levels trigger immediate eutrophic collapse.
- **Station Memory Influence:** $\text{hist\_ortho\_lag1}$ provides the second largest negative attribution, proving the value of local catchment memory.
- **Hypoxic Tipping Points:** Dissolved oxygen features exhibit sharp negative attributions below $4.0\text{ mg/L}$, aligning with biological fish mortality limits.
- **Game-Theoretic Axioms:** Satisfies Efficiency, Symmetry, Dummy Player, and Additivity across all predictions.

---

<!-- Slide 13 -->
# Software Engineering & Automated Unit Testing
### Production-Grade Modular Library (`src/`) & Automated `pytest` Suite

- **Modular Package Structure (`src/`):**
  - `src.features`: Physical bounds screening, stoichiometric ratios, station lags.
  - `src.models`: Model interface abstractions and loss definitions.
  - `src.metrics`: Numerical evaluation engine (MAE, RMSE, $R^2$, Macro F1, Brier, ECE).
  - `src.utils`: Transparent multi-part Parquet loading and anti-leakage quarantine.
  - `scripts/run_phase1_pipeline.py`: One-click reproducible end-to-end pipeline runner.
- **12-Test Automated Test Suite (`tests/`):**
  - Thermodynamic bounds, stoichiometric calculations, and log1p math verification.
  - Strict anti-leakage quarantine and metadata integrity assertions.
  - Metric calculation, calibration mathematics, and model serialization checks.
  - **Test Result: 12 passed in 1.65 seconds (100% pass rate).**

---

<!-- Slide 14 -->
# Six-Member Team Contribution Matrix
### Individual Allocation and Viva Voce Defense Zones

| Member | Core Module Responsibility | Individual Viva Voce Defense Zone |
|---|---|---|
| **Mayank Sharma** (Lead) | Pipeline Architecture, RF & XGBoost Modeling, Parquet Sharding | Pipeline orchestration, ensemble algorithms, anti-leakage protocol. |
| **Team Member 2** | Literature Review, CCME WQI Math, Regulatory Frameworks | Mathematical derivation of $F_1, F_2, F_3, nse$; EPA/WHO benchmarks. |
| **Team Member 3** | Multi-National Data Auditing, Physics Bounds, EDA | 5-country audit, physical bounds screening (pH, DO, Temp), outlier diagnostics. |
| **Team Member 4** | Aquatic Feature Engineering, Stoichiometry, Skewness | Benson-Krause DO sat, Redfield ratio, toxic $\text{NH}_3$ equilibrium, $\log(1+x)$. |
| **Team Member 5** | Hyperparameter Optimization, Stratified Splitting, Benchmarking | Loss functions, tree pruning, MDI vs Gain, MAE/RMSE/$R^2$ trade-offs. |
| **Team Member 6** | Explainable AI (TreeSHAP), Calibration Metrics, PyTest Suite | Shapley axioms, TreeSHAP polynomial algorithm, unit testing, Brier/ECE. |

---

<!-- Slide 15 -->
# Conclusion & Viva Voce Defense Preparedness
### Phase 1 Summary & Phase 2 Outlook

- **High-Precision ML Validated:** Tree-based ensembles predict continuous CCME WQI with $R^2 > 0.999$ across 2.82 million international records.
- **Geochemical Physics Value:** Domain feature engineering and $\log(1+x)$ scaling eliminate excessive tree depth dependency.
- **Zero Leakage Guarantee:** Quarantined architecture verified through continuous unit tests.
- **Individual Defense Readiness:** All 6 members are prepared with deep technical, mathematical, and code defense dossiers.
- **Phase 2 Deep Learning Outlook:** Expanding to multi-task PyTorch neural networks and hybrid architectures with probability calibration.

**Thank you! We welcome questions from the evaluation committee.**
