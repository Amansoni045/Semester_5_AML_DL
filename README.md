# Semester_5_AML-DL

### Dataset Reference
- **Dataset Download Link**: [A Comprehensive Surface Water Quality Monitoring Dataset 1940-2023](https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394)
- **Directory Structure**:
  - `data/Dataset/Parquet/`: Compressed Apache Parquet format (`.parquet`, Snappy compression) tracked in Git for fast I/O, low storage, and direct model training (~82% reduction from raw CSV).
  - `data/Dataset/CSV/`: Raw CSV files stored locally and excluded from Git via `.gitignore`.
  - `data/Data Collection Sources/`: Data collection source details (`DataCollectionSources.parquet` tracked, raw `.xlsx` ignored).
  - `data/Code/`: Reference notebooks (ignored in Git; custom code can be created here).
