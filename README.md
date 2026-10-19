# Semester_5_AML-DL

### Surface Water Quality Monitoring Dataset (1940–2023)
A harmonized multi-country surface water quality dataset covering **4,297,895 records across 7 countries** spanning 1940 to 2023, prepared for empirical environmental modeling and deep learning / machine learning workflows.

- **Countries Covered**: Australia, Canada, China, England, India, Ireland, USA
- **Temporal Coverage**: 1940 – 2023
- **Primary Source**: [A Comprehensive Surface Water Quality Monitoring Dataset 1940-2023](https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394) + National Monitoring Networks (USGS NWIS Telemetry, CPCB India, City of Adelaide Australia, Lac Hertel Sonde Telemetry)

### Country Breakdown

| Country | Records | Year Range | Primary Sources |
|---|---|---|---|
| **England** | 2,129,198 | 2000–2023 | Environment Agency UK (`data.gov.uk`) |
| **USA** | 1,879,966 | 1940–2023 | USGS NWIS Telemetry / EPA Water Quality Portal |
| **Ireland** | 235,019 | 1940–2023 | EPA Ireland (`catchments.ie` / `data.europa.eu`) |
| **China** | 45,997 | 2001–2017 | Environmental Protection Ministry / `data.gov.hk` |
| **Canada** | 3,995 | 1968–2021 | Open Canada / ECCC / Lac Hertel Sonde Network |
| **India** | 3,633 | 2003–2023 | Central Pollution Control Board (CPCB) River Monitoring Network |
| **Australia** | 87 | 2014–2015 | Torrens River & Lake Water Quality (`data.sa.gov.au`) |
| **Total** | **4,297,895** | **1940–2023** | **Harmonized 14-Column Target Schema** |

### Target Schema & Features

- **Metadata / Context**:
  - `Country`: Country of origin (string)
  - `Area`: Monitoring station location / river / catchment area (string)
  - `Waterbody Type`: River, Lake, Marine, Transitional, Estuary (string)
  - `Date`: Sampling date formatted as `DD-MM-YYYY` (string)
- **Water Quality Parameters ($X$)**:
  - `Ammonia (mg/l)`
  - `Biochemical Oxygen Demand (mg/l)`
  - `Dissolved Oxygen (mg/l)`
  - `Orthophosphate (mg/l)`
  - `pH (ph units)`
  - `Temperature (cel)`
  - `Nitrogen (mg/l)`
  - `Nitrate (mg/l)`
- **Target Indices ($y$)**:
  - `CCME_Values`: Canadian Council of Ministers of the Environment Water Quality Index score (`0.0` – `100.0`, regression target)
  - `CCME_WQI`: Categorical classification (`Excellent`, `Good`, `Fair`, `Marginal`, `Poor`, `Unsuitable`)

### Directory Structure

```
data/
├── Dataset/
│   ├── Parquet/                      # Tracked in Git (Snappy compressed, fast I/O)
│   │   ├── Combined Data/
│   │   │   └── Combined_dataset.parquet   # 4,297,895 rows × 14 columns
│   │   └── Country-Wise Data/
│   │       ├── Australia_dataset.parquet  # 87 rows
│   │       ├── Canada_dataset.parquet     # 3,995 rows
│   │       ├── China_dataset.parquet      # 45,997 rows
│   │       ├── England_dataset.parquet    # 2,129,198 rows
│   │       ├── India_dataset.parquet      # 3,633 rows
│   │       ├── Ireland_dataset.parquet    # 235,019 rows
│   │       └── USA_dataset.parquet        # 1,879,966 rows
│   └── CSV/                          # Raw CSV files stored locally (ignored via .gitignore)
├── Data Collection Sources/
│   ├── DataCollectionSources.parquet # Provenance URLs & source metadata for all 7 countries
│   └── DataCollectionSources.xlsx    # Raw reference file (ignored via .gitignore)
└── Code/                             # Technical validation & harmonization notebooks
```

### Viewing Parquet Files in VS Code / IDE

Since `.parquet` files use a compressed binary format, standard text editors cannot view them directly.

To view, filter, and inspect the Parquet datasets inside your editor:
1. Open the **Extensions** panel (`Cmd + Shift + X` on macOS / `Ctrl + Shift + X` on Windows/Linux).
2. Search for and install **Parquetly** (**Parquetly - Parquet Viewer and Editor**).
3. Click any `.parquet` file in `data/`, or right-click the file and select **Reopen Editor With...** &rarr; **Parquetly**.
