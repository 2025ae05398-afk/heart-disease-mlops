from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, roc_auc_score,
                             classification_report, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from src.data import load_dataset, prepare_xy
from src.features import build_preprocessor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"


def train():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, target = load_dataset()
    X, y = prepare_xy(frame, target)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("uci-heart-disease-classification")
    candidates = [
        ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced",
                                                    random_state=42),
         {"classifier__C": [0.1, 1.0, 10.0]}),
        ("random_forest", RandomForestClassifier(class_weight="balanced", random_state=42),
         {"classifier__n_estimators": [100, 250], "classifier__max_depth": [None, 5, 10],
          "classifier__min_samples_leaf": [1, 2]}),
    ]
    fitted = []
    for name, estimator, grid in candidates:
        pipe = Pipeline([("preprocessor", build_preprocessor(Xtr)), ("classifier", estimator)])
        search = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                              refit=True, error_score="raise")
        with mlflow.start_run(run_name=name):
            search.fit(Xtr, ytr)
            prob = search.predict_proba(Xte)[:, 1]
            pred = search.predict(Xte)
            metrics = {
                "cv_best_roc_auc": float(search.best_score_),
                "accuracy": float(accuracy_score(yte, pred)),
                "precision": float(precision_score(yte, pred, zero_division=0)),
                "recall": float(recall_score(yte, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(yte, prob)),
            }
            mlflow.log_params({"model": name, **search.best_params_, "seed": 42})
            mlflow.log_metrics(metrics)
            mlflow.log_dict(search.best_params_, "best_params.json")
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")
            fitted.append({"name": name, "model": search.best_estimator_,
                           "metrics": metrics, "params": search.best_params_})
    # Select only by cross-validation score on the training split.
    winner = max(fitted, key=lambda item: item["metrics"]["cv_best_roc_auc"])
    model = winner["model"]
    joblib.dump(model, OUT / "model.joblib")
    pred, prob = model.predict(Xte), model.predict_proba(Xte)[:, 1]
    report = {
        "selected_model": winner["name"],
        "selection_metric": "mean 5-fold training CV ROC-AUC",
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, zero_division=0)),
        "recall": float(recall_score(yte, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(yte, prob)),
        "cv_best_roc_auc": float(winner["metrics"]["cv_best_roc_auc"]),
        "classification_report": classification_report(yte, pred, output_dict=True,
                                                       zero_division=0),
        "candidate_results": [{"name": r["name"], "metrics": r["metrics"],
                               "best_params": r["params"]} for r in fitted],
    }
    (OUT / "metrics.json").write_text(json.dumps(report, indent=2))
    metadata = {"feature_names": list(X.columns), "target_source_column": target,
                "target_mapping": "0 -> absence; values > 0 -> presence",
                "selected_model": winner["name"], "positive_class": 1,
                "model_path": "artifacts/model.joblib",
                "note": "Educational only; not for clinical use."}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2))
    with mlflow.start_run(run_name="selected_model_test_evaluation"):
        mlflow.log_params({"selected_model": winner["name"], "seed": 42})
        mlflow.log_metrics({k: v for k, v in report.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(str(OUT / "metrics.json"))
        mlflow.log_artifact(str(OUT / "metadata.json"))
        ConfusionMatrixDisplay.from_predictions(yte, pred)
        plt.tight_layout()
        plt.savefig(OUT / "confusion_matrix.png", dpi=150)
        plt.close()
        RocCurveDisplay.from_predictions(yte, prob)
        plt.tight_layout()
        plt.savefig(OUT / "roc_curve.png", dpi=150)
        plt.close()
        mlflow.log_artifact(str(OUT / "confusion_matrix.png"))
        mlflow.log_artifact(str(OUT / "roc_curve.png"))
        for plot in (OUT / "eda").glob("*.png"):
            mlflow.log_artifact(str(plot), artifact_path="eda")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    train()
