"""Train, cross-validate, compare, track with MLflow, and package a model."""
from __future__ import annotations

import json
import os
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay, RocCurveDisplay, classification_report,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from src.data import load_dataset, ROOT
from src.preprocessing import build_pipeline

RANDOM_STATE = 42
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"

def train():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    df = load_dataset().dropna(subset=["target"])
    X, y = df.drop(columns="target"), df["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", str(ROOT / "mlruns")))
    mlflow.set_experiment("heart-disease-uci")

    candidates = {
        "logistic_regression": (
            build_pipeline("logistic_regression"),
            {"model__C": [0.1, 1.0, 10.0]},
        ),
        "random_forest": (
            build_pipeline("random_forest"),
            {"model__max_depth": [None, 5, 10], "model__min_samples_leaf": [1, 3]},
        ),
    }
    summaries = []
    best_score = -1.0
    best_model = None
    best_name = None
    best_metrics = None

    for name, (pipeline, grid) in candidates.items():
        search = GridSearchCV(
            pipeline, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
            refit=True, return_train_score=False,
        )
        with mlflow.start_run(run_name=name) as run:
            search.fit(X_train, y_train)
            pred = search.predict(X_test)
            proba = search.predict_proba(X_test)[:, 1]
            metrics = {
                "cv_roc_auc_mean": float(search.best_score_),
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_precision": float(precision_score(y_test, pred, zero_division=0)),
                "test_recall": float(recall_score(y_test, pred, zero_division=0)),
                "test_f1": float(f1_score(y_test, pred, zero_division=0)),
                "test_roc_auc": float(roc_auc_score(y_test, proba)),
            }
            mlflow.log_params({"model": name, "random_state": RANDOM_STATE, "test_size": 0.2, **search.best_params_})
            mlflow.log_metrics(metrics)
            mlflow.log_dict({"best_params": search.best_params_, "classification_report": classification_report(y_test, pred, output_dict=True)}, "evaluation.json")

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            ConfusionMatrixDisplay.from_predictions(y_test, pred, ax=axes[0], colorbar=False)
            RocCurveDisplay.from_predictions(y_test, proba, ax=axes[1])
            fig.tight_layout()
            plot_path = ARTIFACT_DIR / f"{name}_evaluation.png"
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)
            mlflow.log_artifact(str(plot_path))
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
            summaries.append({"model": name, "run_id": run.info.run_id, **metrics, "best_params": search.best_params_})
            if metrics["cv_roc_auc_mean"] > best_score:
                best_score = metrics["cv_roc_auc_mean"]
                best_model = search.best_estimator_
                best_name = name
                best_metrics = metrics

    assert best_model is not None
    joblib.dump(best_model, MODEL_PATH)
    report = {
        "selected_model": best_name,
        "selection_metric": "mean 5-fold stratified cross-validation ROC-AUC",
        "random_state": RANDOM_STATE,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "models": summaries,
        "selected_test_metrics": best_metrics,
        "disclaimer": "Educational dataset and demonstration only; not a medical diagnostic device.",
    }
    (ARTIFACT_DIR / "training_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved model to {MODEL_PATH}")
    return report

if __name__ == "__main__":
    train()
