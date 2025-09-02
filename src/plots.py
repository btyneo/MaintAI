# src/plots.py
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def save_pred_vs_true_scatter(y_true, y_pred, out_path: str, title: str = "Predicted vs True RUL") -> None:
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, s=16)
    lim = [0, max(np.max(y_true), np.max(y_pred)) * 1.05]
    plt.plot(lim, lim, 'k--')
    plt.xlabel("True RUL"); plt.ylabel("Predicted RUL")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_feature_importances(model, feature_names: list[str], out_path: str, top_k: int = 10, title: str = "Top Feature Importances") -> None:
    importances = np.asarray(getattr(model, "feature_importances_", None))
    if importances is None or importances.size == 0:
        raise ValueError("Model has no feature_importances_.")
    idx = np.argsort(importances)[::-1][:top_k]
    top_feats = pd.Series(importances[idx], index=[feature_names[i] for i in idx])

    plt.figure(figsize=(7, 5))
    top_feats.iloc[::-1].plot(kind="barh")
    plt.xlabel("Importance"); plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()

def save_engine_curves(df_with_true_and_pred: pd.DataFrame, unit_ids: list[int], out_path: str, title: str = "Per-engine RUL curves") -> None:
    """df must have columns: unit_number, time_in_cycles, RUL, (optional) RUL_pred."""
    plt.figure(figsize=(8, 5))
    for u in unit_ids:
        sub = df_with_true_and_pred[df_with_true_and_pred["unit_number"] == u].sort_values("time_in_cycles")
        plt.plot(sub["time_in_cycles"], sub["RUL"], label=f"true u{u}", alpha=0.9)
        if "RUL_pred" in sub.columns:
            plt.plot(sub["time_in_cycles"], sub["RUL_pred"], "--", label=f"pred u{u}", alpha=0.9)
    plt.xlabel("Cycle"); plt.ylabel("RUL")
    plt.title(title); plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close()
