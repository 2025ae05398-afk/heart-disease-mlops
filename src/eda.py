from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from src.data import load_dataset, prepare_xy

OUT = Path(__file__).resolve().parents[1] / "artifacts" / "eda"


def run_eda():
    frame, target = load_dataset()
    X, y = prepare_xy(frame, target)
    OUT.mkdir(parents=True, exist_ok=True)
    X.describe(include="all").transpose().to_csv(OUT / "feature_summary.csv")
    y.value_counts().rename_axis("target").reset_index(name="count").to_csv(
        OUT / "class_balance.csv", index=False)
    plt.figure(figsize=(6, 4))
    sns.countplot(x=y)
    plt.title("Target class balance")
    plt.xlabel("0 = absence; 1 = presence")
    plt.tight_layout()
    plt.savefig(OUT / "class_balance.png", dpi=150)
    plt.close()
    numeric = X.select_dtypes(include="number")
    if not numeric.empty:
        numeric.hist(figsize=(14, 10), bins=20, edgecolor="black")
        plt.tight_layout()
        plt.savefig(OUT / "feature_histograms.png", dpi=150)
        plt.close()
        plt.figure(figsize=(12, 9))
        sns.heatmap(numeric.assign(target=y).corr(numeric_only=True),
                    cmap="coolwarm", center=0)
        plt.title("Feature and target correlation heatmap")
        plt.tight_layout()
        plt.savefig(OUT / "correlation_heatmap.png", dpi=150)
        plt.close()
    print(f"Rows={len(X)}, features={X.shape[1]}, target={target}, classes={y.value_counts().to_dict()}")


if __name__ == "__main__":
    run_eda()
