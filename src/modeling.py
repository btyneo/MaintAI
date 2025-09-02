# src/modeling.py
from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor

def make_rf(n_estimators: int = 400, seed: int = 42, n_jobs: int = -1) -> RandomForestRegressor:
    """Create a reproducible RandomForest regressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=n_jobs
    )

def fit_regressor(model: RandomForestRegressor, X, y) -> RandomForestRegressor:
    """Fit any sklearn regressor and return it."""
    model.fit(X, y)
    return model

def predict(model: RandomForestRegressor, X):
    """Predict with a trained regressor."""
    return model.predict(X)
