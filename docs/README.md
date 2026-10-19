# Project Documentation: Water Quality Prediction

Welcome! This folder contains all the documentation for our **Water Quality Prediction** project. Everything is written in simple, easy-to-understand English so anyone on our team can quickly see what has been done and what comes next.

---

## 1. What Is This Project About?

Water quality is critical for human health, agriculture, and ecosystems.  
In this project, our goal is to **predict water quality** using physical and chemical sensor readings (like pH, temperature, dissolved oxygen, and nutrient levels).

Our team is building two kinds of models:
1. **Machine Learning Baseline**: Tree models (Random Forest, XGBoost) that work well with tabular data.
2. **Deep Learning Model**: A Multi-Layer Perceptron (MLP neural network) to learn complex, non-linear chemical interactions.

---

## 2. What Is Our Specific Role?

Our project is divided among team members. Our assigned role is:

### **Role 2: Dataset Collection & Exploratory Data Analysis (EDA)**
Our responsibilities are:
- **Collect & organize the data**: Find reliable water quality datasets from around the world.
- **Perform Exploratory Data Analysis (EDA)**: Understand the data, check distributions, find missing values, spot outliers, and see how different water parameters relate to each other.
- **Save clean outputs**: Export cleaned and processed datasets as fast `.parquet` files for our teammates doing Preprocessing, Feature Engineering, and Model Training.
- **Document everything**: Write down every finding clearly so our team and professors can easily understand our work.

---

## 3. What Are We Predicting?

We use the widely accepted **CCME Water Quality Index (CCME WQI)** as our target:

1. **Continuous Score (`CCME_Values`)**:
   - A score from **0 to 100**.
   - Higher is cleaner water; lower means more polluted water.
   - Models predict this using regression (evaluated with MAE, RMSE, and $R^2$).

2. **Water Quality Category (`CCME_WQI`)**:
   - Classification into 5 standard grades:
     - **Excellent** (95 – 100): Pristine, near-natural condition.
     - **Good** (80 – 94): Protected with only minor impairment.
     - **Fair** (65 – 79): Usually protected but occasionally threatened.
     - **Marginal** (45 – 64): Frequently threatened or impaired.
     - **Poor** (0 – 44): Almost always threatened or impaired.

---

## 4. What Features Are in Our Dataset?

We have 8 core physical and chemical parameters:

| Parameter | Unit | Standard Limit | What It Measures |
|---|---|---|---|
| **pH** | pH units | 8.5 | Acidity or basicity of the water |
| **Dissolved Oxygen (DO)** | mg/L | 14.0 | Amount of oxygen available for fish and aquatic life |
| **Temperature** | °C | 25.0 | Water temperature (affects oxygen solubility) |
| **Biochemical Oxygen Demand (BOD)** | mg/L | 6.0 | Organic pollution load in water |
| **Ammonia** | mg/L | 0.5 | Toxic waste from sewage and fertilizer |
| **Orthophosphate** | mg/L | 0.05 | Phosphorus nutrient that can cause algae blooms |
| **Nitrogen** | mg/L | 20.0 | Total nitrogen nutrient concentration |
| **Nitrate** | mg/L | 45.0 | Oxidized nitrogen from agricultural runoff |

We also have location and time metadata:
- `Country`: England, USA, Ireland, China, Canada, India, Australia
- `Area`: Name of the river, lake, or station
- `Waterbody Type`: River, Stream, Lake, Estuary, etc.
- `Date`: Sampling date (`DD-MM-YYYY`)

---

## 5. What Has Been Done So Far? (Progress Summary)

### **Phase 1: Dataset Collection & Harmonization (Complete)**
- We collected and harmonized **4,297,895 water quality records across 7 countries**:
  - **England**: 2,129,198 records (2000–2023)
  - **USA**: 1,879,966 records (1940–2023)
  - **Ireland**: 235,019 records (1940–2023)
  - **China**: 45,997 records (2001–2017)
  - **Canada**: 3,995 records (1968–2021)
  - **India**: 3,633 records (2003–2023)
  - **Australia**: 87 records (2014–2015)
- Standardized all datasets into the exact same 14 columns.
- Calculated the standard CCME WQI score and category for all rows (0 missing target values).
- Converted all files to `.parquet` format using standard `snappy` compression, so they open easily in the **Parquetly** extension in VS Code.

---

## 6. How the Work Is Organized

- **Notebooks (`nootbooks/`)**:
  - Each notebook handles one specific part of the analysis (e.g., summary statistics, missing values, outliers, correlations).
  - Notebooks read data from `data/Dataset/Parquet/Combined Data/Combined_dataset.parquet`.
  - When a notebook cleans or filters data, it saves the output to `data/GeneratedOutput/` as a `.parquet` file.

- **Generated Data (`data/GeneratedOutput/`)**:
  - Contains intermediate and processed datasets created by the notebooks.
  - Always saved in `.parquet` format with snappy compression.

- **Documentation (`docs/`)**:
  - Simple, clear reports explaining the results of each notebook so everyone can follow along.

---

## 7. Upcoming EDA Roadmap

1. **`01_dataset_overview_and_summary.ipynb`**: Summary statistics, record counts per country, date distributions, and target class balance.
2. **`02_missing_values_and_sparsity.ipynb`**: Finding which sensors are missing most often and deciding how to handle them.
3. **`03_outliers_and_sensor_anomalies.ipynb`**: Spotting extreme or impossible values (like negative pH or extreme temperatures).
4. **`04_feature_correlations.ipynb`**: Analyzing how parameters correlate with each other and with the water quality score.
5. **`05_temporal_and_spatial_trends.ipynb`**: Seeing how water quality changes over seasons, years, and waterbody types.
6. **`06_benchmark_splits_for_modeling.ipynb`**: Creating leak-free Train, Validation, and Test datasets for the modeling team.
