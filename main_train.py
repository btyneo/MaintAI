# main_train.py
from pathlib import Path
import numpy as np
import pandas as pd

from src.data import load_fd001_train, add_rul_labels, split_by_engine, select_feature_matrix
from src.modeling import make_rf, fit_regressor, predict
from src.metrics import eval_regression, metrics_table, save_metrics_csv
from src.plots import save_pred_vs_true_scatter, save_feature_importances

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def main():
    # 1) Load + label
    df = load_fd001_train("data/CMAPSS/train_FD001.txt")
    df = add_rul_labels(df)

    # 2) Split by engine (no leakage)
    train_df, test_df = split_by_engine(df, train_max_unit=80)

    # 3) Build X/y
    X_train = select_feature_matrix(train_df)
    y_train = train_df["RUL"].values
    X_test  = select_feature_matrix(test_df)
    y_test  = test_df["RUL"].values

    # 4) Train RF (seeded)
    rf = make_rf(n_estimators=400, seed=42, n_jobs=-1)
    rf = fit_regressor(rf, X_train, y_train)

    # 5) Predict + evaluate
    y_pred = predict(rf, X_test)
    # Optional: clip negative predictions
    y_pred = np.maximum(y_pred, 0)

    m = eval_regression(y_test, y_pred)
    print(f"[RandomForest] MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.4f}")

    # 6) Save artifacts
    save_pred_vs_true_scatter(y_test, y_pred, RESULTS / "pred_vs_true.png")
    feat_names = list(X_train.columns)
    save_feature_importances(rf, feat_names, RESULTS / "feature_importances.png")
    save_metrics_csv(
        metrics_table([{"Model": "RandomForest", "Features": "Ops+Sensors", **m}]),
        RESULTS / "metrics.csv"
    )

if __name__ == "__main__":
    main()
