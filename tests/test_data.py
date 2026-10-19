import pandas as pd
from src.data import COLUMNS, load_dataset

def test_dataset_schema_and_binary_target(tmp_path):
    row = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 2]
    path = tmp_path / "sample.csv"
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, index=False)
    df = load_dataset(path)
    assert list(df.columns) == COLUMNS
    assert set(df["target"].dropna().unique()) <= {0, 1}
    assert int(df["target"].iloc[0]) == 1

def test_missing_marker_is_parsed_as_nan(tmp_path):
    row = [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", 6, 0]
    path = tmp_path / "sample.csv"
    pd.DataFrame([row], columns=COLUMNS).to_csv(path, index=False)
    df = load_dataset(path)
    assert pd.isna(df.loc[0, "ca"])
