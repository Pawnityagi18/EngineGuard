from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from cmapss_maintenance.data import add_train_rul, load_fd001
from cmapss_maintenance.features import add_health_features, select_feature_columns


DATA_DIR = Path("data/raw")
ARTIFACT_DIR = Path("artifacts")
OUTPUT_DIR = Path("research")

WINDOW = 30


def main() -> None:
    bundle = joblib.load(ARTIFACT_DIR / "fd001_models.joblib")

    train, test, _ = load_fd001(DATA_DIR)

    train_with_rul = add_train_rul(train)
    base_features = select_feature_columns(train_with_rul)

    test_engineered = add_health_features(
        test,
        base_features,
        window=WINDOW,
    )

    features = bundle["feature_columns"]
    model = bundle["regression_model"]

    endpoints = (
        test_engineered
        .groupby("unit_number", sort=True)
        .tail(1)
        .sort_values("unit_number")
        .reset_index(drop=True)
    )

    X = endpoints[features]

    print(f"Model: {type(model).__name__}")
    print(f"Engines analyzed: {len(X)}")
    print(f"Features analyzed: {len(features)}")
    print("Calculating SHAP values...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X).values

    # Global SHAP importance
    global_importance = pd.DataFrame(
        {
            "feature": features,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    global_path = OUTPUT_DIR / "shap_global_feature_importance.csv"
    global_importance.to_csv(global_path, index=False)

    print("\nTop 15 SHAP features:")
    print(global_importance.head(15).to_string(index=False))

    # Save Engine 1 explanation as a case study
    engine_1_values = pd.DataFrame(
        {
            "feature": features,
            "shap_value": shap_values[0],
            "feature_value": X.iloc[0].to_numpy(),
        }
    )

    engine_1_values["abs_shap"] = engine_1_values["shap_value"].abs()

    engine_1_values = engine_1_values.sort_values(
        "abs_shap",
        ascending=False,
    )

    engine_1_path = OUTPUT_DIR / "shap_engine_1.csv"
    engine_1_values.to_csv(engine_1_path, index=False)

    print(f"\nSaved global analysis to: {global_path}")
    print(f"Saved Engine 1 explanation to: {engine_1_path}")


if __name__ == "__main__":
    main()