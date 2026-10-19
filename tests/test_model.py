import pandas as pd
from src.preprocessing import build_pipeline

def sample():
    return pd.DataFrame([{
        "age": 58, "sex": 1, "cp": 4, "trestbps": 130, "chol": 250,
        "fbs": 0, "restecg": 0, "thalach": 140, "exang": 0,
        "oldpeak": 2.0, "slope": 2, "ca": 0, "thal": 3,
    }, {
        "age": 44, "sex": 0, "cp": 2, "trestbps": 120, "chol": 210,
        "fbs": 0, "restecg": 1, "thalach": 170, "exang": 0,
        "oldpeak": 0.0, "slope": 1, "ca": 0, "thal": 3,
    }])

def test_logistic_pipeline_fit_predict():
    X = pd.concat([sample()] * 15, ignore_index=True)
    y = pd.Series([0, 1] * 15)
    model = build_pipeline("logistic_regression")
    model.fit(X, y)
    assert list(model.predict(X).shape) == [30]
    assert model.predict_proba(X).shape == (30, 2)

def test_unknown_model_raises():
    import pytest
    with pytest.raises(ValueError):
        build_pipeline("unsupported")
