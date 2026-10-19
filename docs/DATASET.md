# Dataset notes

The project fetches UCI Machine Learning Repository Heart Disease dataset ID 45 using `ucimlrepo`. Dataset variants may differ in row count, feature names, missingness and target coding. Record the exact version and retrieval date in your final report.

Run `pip install -r requirements.txt` then `python download_data.py`. The CSV is written to `data/heart_disease.csv`. The target is identified using common names (`num`, `target`, `HeartDisease`, `heart_disease`, `diagnosis`) or defaults to the final column. Target value 0 is mapped to absence and values greater than 0 to presence; verify this against the exact downloaded version. `?`, `NA`, `N/A`, and blank values are parsed as missing. Missing predictor values are imputed inside the pipeline to prevent cross-validation leakage.
