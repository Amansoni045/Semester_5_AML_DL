# Semester_5_AML-DL

### Dataset Reference
- **Dataset Download Link**: [A Comprehensive Surface Water Quality Monitoring Dataset 1940-2023](https://figshare.com/articles/dataset/A_Comprehensive_Surface_Water_Quality_Monitoring_Dataset_1940-2023_2_82Million_Record_Resource_for_Empirical_and_ML-Based_Research/27800394)
- **Storage Format**: Converted to Apache Parquet format (`.parquet`, Snappy compression) for faster I/O, column pruning, and reduced storage size (~82% reduction from raw CSV). Raw `.csv` files are excluded via `.gitignore`.
