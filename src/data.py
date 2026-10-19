"""Download and validate the Cleveland Heart Disease dataset from UCI.

The canonical UCI processed Cleveland file is used because it contains 303 rows
and 14 columns (13 predictors + target). The target is converted to binary:
0 = no disease; 1-4 = disease present. This is a teaching dataset, not a
clinical-grade diagnostic dataset.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_PATH = RAW_DIR / "heart.csv"
UCI_URLS = [
    "https://raw.githubusercontent.com/plotly/datasets/master/heart.csv",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC_COLUMNS = [
    "age", "trestbps", "chol", "thalach", "oldpeak", "ca",
]
CATEGORICAL_COLUMNS = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "thal",
]

def _read_uci_processed() -> pd.DataFrame:
    response = requests.get(UCI_URLS[1], timeout=30)
    response.raise_for_status()
    from io import StringIO
    df = pd.read_csv(StringIO(response.text), header=None, names=COLUMNS,
                     na_values=["?", " ?"], skipinitialspace=True)
    return df

def download_dataset(force: bool = False) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    # Try the UCI-hosted canonical file first; fall back to a well-known mirror.
    try:
        df = _read_uci_processed()
    except Exception as uci_error:
        try:
            df = pd.read_csv(UCI_URLS[0])
            # Normalize common names used by public mirrors of this dataset.
            aliases = {
                "cp": "cp", "chest_pain": "cp", "trestbps": "trestbps",
                "resting_blood_pressure": "trestbps", "chol": "chol",
                "cholesterol": "chol", "fbs": "fbs", "fasting_blood_sugar": "fbs",
                "restecg": "restecg", "resting_ecg": "restecg", "thalach": "thalach",
                "max_heart_rate": "thalach", "exang": "exang",
                "exercise_induced_angina": "exang", "oldpeak": "oldpeak",
                "slope": "slope", "ca": "ca", "thal": "thal", "target": "target",
                "num": "target",
            }
            normalized = {}
            for column in df.columns:
                key = str(column).strip().lower().replace(" ", "_")
                normalized[column] = aliases.get(key, key)
            df = df.rename(columns=normalized)
            required = set(COLUMNS)
            if not required.issubset(df.columns):
                raise ValueError(f"Mirror schema mismatch; columns found: {list(df.columns)}")
            df = df[COLUMNS]
        except Exception as mirror_error:
            raise RuntimeError(
                "Dataset download failed. Check network access and retry. "
                f"UCI error={uci_error}; mirror error={mirror_error}"
            ) from mirror_error
    df.to_csv(DATA_PATH, index=False)
    return DATA_PATH

def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        download_dataset()
    df = pd.read_csv(path, na_values=["?", " ?"], skipinitialspace=True)
    # Some CSV mirrors use an alternate target name.
    if "num" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"num": "target"})
    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_cols)}")
    df = df[COLUMNS].copy()
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["target"] = df["target"].map(lambda x: 0 if x == 0 else (1 if pd.notna(x) else float("nan")))
    return df

def main() -> None:
    path = download_dataset()
    df = load_dataset(path)
    print(f"Dataset saved to: {path}")
    print(f"Shape: {df.shape}; missing cells: {int(df.isna().sum().sum())}")
    print(df.head().to_string(index=False))

if __name__ == "__main__":
    main()
