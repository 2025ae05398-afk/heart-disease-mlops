# Architecture

```mermaid
flowchart TD
  A[UCI Repository] --> B[download_data.py] --> C[CSV]
  C --> D[EDA and data checks]
  C --> E[Stratified train/test split]
  E --> F[Impute, scale, encode]
  F --> G[Logistic Regression GridSearchCV]
  F --> H[Random Forest GridSearchCV]
  G --> I[Select by training CV ROC-AUC]
  H --> I
  G --> J[MLflow]
  H --> J
  I --> K[Serialized full pipeline]
  K --> L[FastAPI /predict]
  L --> M[Logs and Prometheus metrics]
  N[GitHub Actions] --> D
  N --> E
  L --> O[Docker / Kubernetes]
```

The API returns a binary prediction, confidence, class probabilities, and an educational-use disclaimer. `/health` checks service availability and model load state; `/metrics` exposes Prometheus metrics.
