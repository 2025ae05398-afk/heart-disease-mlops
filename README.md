# Heart Disease MLOps Project

Runnable reference implementation for the Machine Learning Operations (MLOps) AIMLCZG523 Assignment 01 assignment: UCI dataset acquisition, EDA, preprocessing, Logistic Regression and Random Forest, stratified cross-validation, MLflow, FastAPI, pytest, Ruff, Docker, GitHub Actions, Kubernetes and Prometheus metrics.

**Disclaimer:** Educational demonstration only. Not a medical device; never use for diagnosis or treatment decisions.

## Quick start (Python 3.11)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
python -m src.eda
python -m src.train
pytest -q
ruff check .
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs`, `GET /health`, and  `POST /predict` accepts one JSON object with every feature key shown in `artifacts/metadata.json`. Example for the common Cleveland feature schema:

```json
{"age":63,"sex":1,"cp":3,"trestbps":145,"chol":233,"fbs":1,"restecg":0,"thalach":150,"exang":0,"oldpeak":2.3,"slope":0,"ca":0,"thal":1}
```

The precise dataset version may use different feature names. Inspect the generated metadata and adapt the JSON to it.

## MLflow

```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

## Docker

Train first (the model artifact is copied into the image), then:

```bash
docker build -t heart-disease-api:local .
docker run --rm -p 8000:8000 heart-disease-api:local
```

## Kubernetes

Build the image where your cluster can access it, then:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl get pods,svc
kubectl port-forward svc/heart-disease-api 8000:80
```

For Minikube, you may need `minikube tunnel` or port-forwarding. The manifests are templates and do not constitute proof of a public cloud deployment.

## Reproducibility and design

- `download_data.py` fetches UCI dataset ID 45 using `ucimlrepo`.
- Source target 0 maps to absence; any target value greater than 0 maps to presence. Verify this mapping for the downloaded dataset version.
- Missing values are imputed inside the scikit-learn pipeline to prevent cross-validation leakage.
- Grid search uses five-fold stratified CV. Model selection uses training CV ROC-AUC; the held-out test set is for final reporting.
- Artifacts: `artifacts/model.joblib`, `metadata.json`, `metrics.json`, plots, and local `mlruns/`.
- CI runs linting, tests, data download, EDA, training and artifact upload. It does not automatically deploy to a cloud account.
