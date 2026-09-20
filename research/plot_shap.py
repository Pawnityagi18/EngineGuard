from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESEARCH_DIR = Path("research")


def plot_global_shap() -> None:
    data = pd.read_csv(
        RESEARCH_DIR / "shap_global_feature_importance.csv"
    ).head(15)

    data = data.sort_values("mean_abs_shap")

    plt.figure(figsize=(10, 7))
    plt.barh(data["feature"], data["mean_abs_shap"])
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Feature")
    plt.title("Global SHAP Feature Importance - NASA C-MAPSS FD001")
    plt.tight_layout()
    plt.savefig(
        RESEARCH_DIR / "shap_global_importance.png",
        dpi=160,
    )
    plt.close()


def plot_engine_1() -> None:
    data = pd.read_csv(
        RESEARCH_DIR / "shap_engine_1.csv"
    ).head(10)

    data = data.sort_values("shap_value")

    plt.figure(figsize=(10, 6))
    plt.barh(data["feature"], data["shap_value"])
    plt.axvline(0, linewidth=1)
    plt.xlabel("SHAP Value")
    plt.ylabel("Feature")
    plt.title("SHAP Explanation - Engine 1")
    plt.tight_layout()
    plt.savefig(
        RESEARCH_DIR / "shap_engine_1_explanation.png",
        dpi=160,
    )
    plt.close()


def main() -> None:
    plot_global_shap()
    plot_engine_1()
    print("SHAP plots created successfully.")


if __name__ == "__main__":
    main()