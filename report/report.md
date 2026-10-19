# End-to-End MLOps for Heart Disease Classification
## Experimental Learning Assignment

**Student:** [Enter name]  
**Course / section:** [Enter course]  
**Institution:** [Enter institution]  
**Date:** [Enter submission date]  
**Repository:** [Add GitHub URL after publishing]

> **Important scope note:** This is an educational software engineering exercise using a small historical dataset. It is not a clinically validated model and must not be used to diagnose, treat, or make decisions about patients.

---

## 1. Executive summary

This project implements a reproducible machine learning workflow for binary classification using the processed Cleveland subset of the UCI Heart Disease dataset. It combines data acquisition and exploratory data analysis, train-fitted preprocessing, model comparison and cross-validation, MLflow experiment tracking, automated tests, a GitHub Actions workflow, a FastAPI prediction service, Docker packaging, Kubernetes manifests for Minikube, request logging, and Prometheus metrics.

Two candidate estimators are compared: Logistic Regression and Random Forest. A stratified holdout set is reserved for final evaluation, while stratified five-fold cross-validation and grid search are used for model selection. The best estimator is selected using mean cross-validation ROC-AUC. The full preprocessing-plus-estimator pipeline is serialized so that inference uses the same transformations as training.

**Measured results:** Populate this section from `artifacts/training_summary.json` after executing training. Do not enter expected or invented metrics.

| Model | CV ROC-AUC mean | Test accuracy | Test precision | Test recall | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | [Run training] | [Run training] | [Run training] | [Run training] | [Run training] |
| Random Forest | [Run training] | [Run training] | [Run training] | [Run training] | [Run training] |

## 2. Problem statement and objectives

The objective is to demonstrate a full lifecycle for a binary classifier: acquire and inspect data, define a repeatable training process, compare candidates, record experiments, package the model, test the service, and deploy it locally in a container and Kubernetes cluster. The API accepts patient-feature JSON and returns a class, confidence, and positive-class probability.

The intended learning outcome is operational maturity rather than clinical utility. In particular, the project demonstrates version-controlled code, fixed random seeds, a single preprocessing pipeline, metrics and artifacts, automated quality checks, containerized serving, health checks, and basic observability.

## 3. Dataset acquisition and data quality

The source is the UCI Machine Learning Repository's Heart Disease dataset, using the classic processed Cleveland file. The processed subset is commonly described as 303 records with 13 predictors plus a target. Predictors include age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, ST depression, slope, number of major vessels, and thalassemia-related category.

The acquisition script `src/data.py` downloads the processed file, parses `?` as missing values, validates required columns, converts columns to numeric values, and maps the original target to a binary label: 0 remains 0, and positive values 1–4 become 1. This mapping is documented because it changes the original multi-level target into a binary outcome.

Run:

```bash
python -m src.data
python -m src.eda
```

The script emits a data profile with row/column counts, missing-value counts, target counts, and descriptive statistics. Figures include age distributions by target, target class balance, and a feature correlation heatmap.

**Insert actual EDA findings:** [Describe class balance, missingness, distributions, and any noteworthy correlations after inspecting generated artifacts.] Do not assume there are no missing values; the classic Cleveland file contains missing markers in some categorical-like fields.

## 4. Exploratory data analysis

EDA is intended to answer four questions: how the target is distributed, whether predictors contain missing values, how numeric predictors are distributed, and which feature pairs show linear correlation. Histograms help identify spread and skew; the class count plot helps reveal imbalance; the heatmap is a descriptive summary rather than evidence of causality.

**Figure 1 — EDA distributions:** [Insert `artifacts/eda_distributions.png` after running the EDA script.]  
**Figure 2 — Correlation heatmap:** [Insert `artifacts/correlation_heatmap.png` after running the EDA script.]

The data profile is saved as `artifacts/data_profile.json` so that the report can be completed from a generated artifact rather than manually transcribed values. Any conclusions should be tied to the actual plots and the version of the dataset used.

## 5. Feature engineering and model development

A scikit-learn `ColumnTransformer` applies numeric and categorical preprocessing within the model pipeline. Numeric predictors are median-imputed and standardized. Categorical-coded predictors are imputed using the most frequent observed value and one-hot encoded with unknown categories ignored. Although some fields are stored as numbers, one-hot encoding is appropriate for nominal category codes; numeric scaling is used for continuous measurements.

The target is excluded from the feature matrix. Data is split into training and test sets using an 80/20 stratified split and random seed 42. Stratification preserves approximate class proportions in both partitions. All learned transformations are fitted only on training folds through the pipeline, reducing leakage risk.

Candidate models:
- **Logistic Regression:** interpretable baseline, standardized input, balanced class weights, and tuning over `C`.
- **Random Forest:** nonlinear ensemble baseline with balanced class weights; tuning over maximum depth and minimum leaf size.

Each candidate uses `GridSearchCV` with stratified five-fold cross-validation and ROC-AUC scoring. The holdout test set is used for final metrics after selecting hyperparameters. Metrics include accuracy, precision, recall, F1, and ROC-AUC. ROC-AUC assesses ranking quality across thresholds; precision and recall make different error types visible. For health-related tasks, neither accuracy alone nor a high AUC is enough to establish clinical usefulness.

**Model selection result:** [Insert selected model and its actual CV ROC-AUC from `artifacts/training_summary.json`.]  
**Confusion matrix and ROC curve:** [Insert actual generated evaluation figure(s).]

## 6. Experiment tracking with MLflow

The trainer creates or reuses an MLflow experiment named `heart-disease-uci`. Each model candidate is logged as a separate run. Logged parameters include the estimator, seed, test fraction, and selected hyperparameters. Metrics include mean CV ROC-AUC and holdout accuracy, precision, recall, F1, and ROC-AUC. Artifacts include evaluation plots, a classification report, and the fitted scikit-learn pipeline.

Launch the UI with:

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

**Figure 3 — MLflow runs:** [Insert a genuine screenshot of the local MLflow UI showing both completed runs.]  
**Tracking location:** `./mlruns` by default; configurable through `MLFLOW_TRACKING_URI`.

The local file store is suitable for a coursework demonstration. A shared team or production setup should use a managed tracking server and durable artifact storage with access control and backup.

## 7. Model packaging and reproducibility

The selected full pipeline is saved as `artifacts/model.joblib`. Packaging preprocessing and estimator together prevents API code from needing to duplicate transformations. `requirements.txt` declares package version ranges, while the random seed, split settings, selected hyperparameters, model metrics, and dataset counts are recorded in the training summary.

For stronger reproducibility, a production extension should create a fully locked dependency file, record the exact source dataset checksum, pin the base container image by digest, and version the model artifact alongside the training code. Serialized Python objects should only be loaded from trusted sources and compatible package versions.

## 8. CI/CD and automated tests

The GitHub Actions workflow triggers on pushes, pull requests, and manual dispatch. It installs dependencies, runs Ruff linting, executes Pytest, downloads the dataset, generates EDA artifacts, trains candidate models, and uploads the `artifacts/` and `mlruns/` directories. The workflow provides automated quality checks and repeatable training, but it does not currently publish an image to a registry or automatically deploy to a cluster.

Tests cover target binarization and schema, missing-value parsing, model pipeline fit/predict, invalid model selection, API health behavior, model-not-loaded behavior, and Prometheus metrics availability.

**Figure 4 — Test output:** [Insert terminal screenshot showing `pytest -q` output.]  
**Figure 5 — CI workflow:** [Insert a genuine GitHub Actions run screenshot with job status and uploaded artifact.]

A future enhancement would split fast unit tests from slower integration/training jobs, cache dependencies, build and scan the container, and gate deployment on approvals.

## 9. API, containerization, and deployment

The FastAPI application exposes:
- `GET /`: service metadata
- `GET /health`: service and model-loaded status
- `GET /metrics`: Prometheus exposition format
- `POST /predict`: schema-validated prediction request

The prediction endpoint returns the predicted binary class, a label, the maximum class probability as model confidence, and probability of the positive class. The confidence value is not a calibrated clinical confidence measure.

Docker packages the Python dependencies, application, and trained model artifact. Build and run after training:

```bash
docker build -t heart-disease-mlops:local .
docker run --rm -p 8000:8000 heart-disease-mlops:local
```

Open `http://localhost:8000/docs` to inspect the generated API schema. A sample JSON payload is provided in the README.

**Figure 6 — Docker running:** [Insert a genuine screenshot of `docker ps` and API health response.]  
**Figure 7 — API request:** [Insert a genuine `/docs` or `curl` prediction request and response.]

For local Kubernetes, the repository includes a Deployment and NodePort Service. The container image is loaded into Minikube, avoiding the need for a public registry:

```bash
minikube start
minikube image load heart-disease-mlops:local
kubectl apply -f deployment/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
kubectl port-forward service/heart-disease-api 8000:8000
```

**Figure 8 — Minikube resources:** [Insert actual output showing Ready pod and Service.]  
**Endpoint verification:** [Record the result of `GET /health` and one sample `/predict` call.]

The deployment manifest contains readiness and liveness probes and resource requests/limits. This is a local educational deployment, not a high-availability production cluster. A public deployment would require registry hosting, TLS, authentication, network policy, secrets management, autoscaling, and a security review.

## 10. Monitoring, logging, and conclusions

The API emits structured application logs for request method, route, status, latency, and completed predictions. Prometheus counters track request totals by method, endpoint, and status; a histogram records request latency. The included Compose monitoring file starts the API and Prometheus, with the API scraped at `/metrics`.

**Figure 9 — Prometheus:** [Insert actual Prometheus target status and a query for `api_requests_total` or `api_request_latency_seconds`.]

These metrics demonstrate service-level observability but do not detect model drift, data quality changes, or clinical performance degradation. Production monitoring would add input schema and distribution checks, privacy-aware logging, drift alerts, model-version metadata, and a governed process for re-evaluation. Avoid logging identifiable patient data.

In conclusion, the repository demonstrates the main stages of an MLOps lifecycle: data acquisition, EDA, train-fitted preprocessing, model selection, experiment tracking, serialization, automated testing, containerization, local Kubernetes deployment, and basic monitoring. The system should be evaluated on its reproducibility and engineering workflow, not on claims of clinical validity.

### Evidence and completion record

| Evidence | Status |
|---|---|
| EDA plots | [Generate and inspect] |
| MLflow experiment runs | [Run training and capture UI] |
| Test suite | [Run pytest and record result] |
| GitHub Actions | [Push repository and capture run] |
| Docker API | [Build, run, verify] |
| Prometheus | [Run Compose monitoring and verify scrape] |
| Minikube | [Deploy, verify, capture resources] |
| Public repository URL | [Add after publishing] |
| Walkthrough video URL/file | [Record and add] |

### References

1. UCI Machine Learning Repository, Heart Disease dataset: https://archive.ics.uci.edu/dataset/45/heart+disease
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. MLflow documentation: https://mlflow.org/docs/latest/
4. FastAPI documentation: https://fastapi.tiangolo.com/
5. Docker documentation: https://docs.docker.com/
6. Kubernetes documentation: https://kubernetes.io/docs/
7. Prometheus documentation: https://prometheus.io/docs/
