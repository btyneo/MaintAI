# src/data.py
from pathlib import Path
import pandas as pd

COL_NAMES = (
    ["unit_number", "time_in_cycles",
     "operational_setting_1", "operational_setting_2", "operational_setting_3"]
    + [f"sensor_measurement_{i}" for i in range(1, 22)]
)

def load_fd001_train(path: str = "../data/CMAPSS/train_FD001.txt") -> pd.DataFrame:
    """Load FD001 training data with column names."""
    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None
    )
    df.columns = COL_NAMES
    return df

def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add Remaining Useful Life (RUL) column: max(cycle) - cycle per engine."""
    df = df.copy()
    df["RUL"] = df.groupby("unit_number")["time_in_cycles"].transform(
        lambda s: s.max() - s
    )
    return df

def split_by_engine(df: pd.DataFrame, train_max_unit: int = 80):
    """Split dataset into train/test by engine ID."""
    train_df = df[df["unit_number"] <= train_max_unit].copy()
    test_df = df[df["unit_number"] > train_max_unit].copy()
    return train_df, test_df

def select_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only operational settings + 21 sensors as features."""
    cols = (
        ["operational_setting_1", "operational_setting_2", "operational_setting_3"]
        + [f"sensor_measurement_{i}" for i in range(1, 22)]
    )
    return df[cols]
