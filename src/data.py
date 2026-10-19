from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease.csv"
TARGET_CANDIDATES = ("num", "target", "HeartDisease", "heart_disease", "diagnosis")


def download_dataset(output_path: Path = DATA_PATH) -> Path:
    """Fetch UCI Heart Disease dataset ID 45 using the official client."""
    from ucimlrepo import fetch_ucirepo
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = fetch_ucirepo(id=45)
    features, targets = dataset.data.features.copy(), dataset.data.targets.copy()
    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, got {list(targets.columns)}")
    frame = features.copy()
    frame[targets.columns[0]] = targets.iloc[:, 0].values
    frame.to_csv(output_path, index=False)
    return output_path


def load_dataset(path: Path = DATA_PATH):
    if not path.exists():
        raise FileNotFoundError(f"Dataset missing at {path}; run `python download_data.py`.")
    frame = pd.read_csv(path, na_values=["?", "NA", "N/A", ""])
    if frame.empty:
        raise ValueError("Dataset is empty")
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    target = next((lookup[n.lower()] for n in TARGET_CANDIDATES if n.lower() in lookup),
                  frame.columns[-1])
    frame = frame.dropna(subset=[target]).copy()
    return frame, target


def prepare_xy(frame: pd.DataFrame, target: str):
    raw = pd.to_numeric(frame[target], errors="coerce")
    valid = raw.notna()
    X = frame.loc[valid].drop(columns=[target]).copy()
    y = (raw.loc[valid] > 0).astype(int)
    for col in X.columns:
        numeric = pd.to_numeric(X[col], errors="coerce")
        if numeric.notna().sum() >= max(1, int(0.5 * len(X))):
            X[col] = numeric
    if y.nunique() != 2:
        raise ValueError(f"Expected binary target after mapping, got {y.unique()}")
    return X, y
