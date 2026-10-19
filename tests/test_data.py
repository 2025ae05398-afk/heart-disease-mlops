import pandas as pd
from src.data import prepare_xy


def test_binary_target_mapping():
    frame = pd.DataFrame({"age": [50, 60, 55], "num": [0, 2, 1]})
    X, y = prepare_xy(frame, "num")
    assert list(X.columns) == ["age"]
    assert y.tolist() == [0, 1, 1]


def test_missing_predictor_is_preserved_for_pipeline_imputation():
    X, y = prepare_xy(pd.DataFrame({"age": [50, None, 55], "num": [0, 1, 0]}), "num")
    assert X["age"].isna().sum() == 1
    assert y.tolist() == [0, 1, 0]
