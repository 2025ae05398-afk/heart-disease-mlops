import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.features import build_preprocessor


def test_preprocessor_handles_missing_and_categorical_values():
    X = pd.DataFrame({"age": [40, 50, None, 60, 45, 52],
                      "sex": [1, 0, 1, 0, 1, 0],
                      "category": ["a", "b", "a", None, "c", "b"]})
    y = [0, 1, 0, 1, 0, 1]
    pipe = Pipeline([("preprocessor", build_preprocessor(X)),
                     ("classifier", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    assert pipe.predict(X).shape == (6,)
