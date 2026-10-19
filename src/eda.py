"""Generate EDA figures and a compact data profile."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_dataset, ROOT

def run_eda():
    df = load_dataset()
    out = ROOT / "artifacts"
    out.mkdir(exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=df, x="age", hue="target", multiple="stack", ax=axes[0])
    axes[0].set_title("Age distribution by target")
    sns.countplot(data=df, x="target", ax=axes[1])
    axes[1].set_title("Binary target balance")
    fig.tight_layout()
    fig.savefig(out / "eda_distributions.png", dpi=160)
    plt.close(fig)

    numeric = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature correlation heatmap")
    fig.tight_layout()
    fig.savefig(out / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    profile = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items()},
        "target_counts": {str(k): int(v) for k, v in df.target.value_counts(dropna=False).items()},
        "numeric_summary": json.loads(df.describe(include="all").to_json()),
    }
    (out / "data_profile.json").write_text(json.dumps(profile, indent=2))
    print(f"EDA artifacts written to {out}")
    print(json.dumps({k: profile[k] for k in ("rows", "columns", "missing_by_column", "target_counts")}, indent=2))

if __name__ == "__main__":
    run_eda()
