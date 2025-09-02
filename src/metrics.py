# src/metrics.py
from __future__ import annotations
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def eval_regression(y_true, y_pred) -> dict:
    """Return MAE, RMSE, R2 with a timestamp."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(root_mean_squared_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def metrics_table(rows: list[dict]) -> pd.DataFrame:
    """Build a tidy table from one or more eval dicts."""
    return pd.DataFrame(rows)

def save_metrics_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
