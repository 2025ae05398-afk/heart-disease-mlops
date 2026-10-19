"""Reusable preprocessing and estimator pipeline construction."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data import NUMERIC_COLUMNS, CATEGORICAL_COLUMNS

def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])

def build_pipeline(model_name: str, random_state: int = 42, **params) -> Pipeline:
    if model_name == "logistic_regression":
        estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state, **params)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=random_state,
            n_jobs=-1, **params
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
