"""Forecast benchmarks with immutable chronological holdout sets."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURES, chronological_split


def candidates() -> dict:
    return {
        "persistence": None,
        "recent_3_mean": None,
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=90, learning_rate=0.06, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=0.3, random_state=42,
        ),
    }


def predict(model_name: str, estimator, df: pd.DataFrame) -> np.ndarray:
    if model_name == "persistence":
        return df["lag_1"].to_numpy(dtype=float)
    if model_name == "recent_3_mean":
        return df["past_mean_3"].to_numpy(dtype=float)
    return estimator.predict(df[FEATURES])


def metrics(y: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    y_true = y.to_numpy(dtype=float)
    residual = y_true - prediction
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mape_pct": float(np.mean(np.abs(residual / y_true)) * 100),
    }


def evaluate(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Choose a model on validation only, then evaluate all on untouched test dates.

    Same athletes can occur in later splits; this estimates future measurements
    for monitored athletes, NOT generalization to athletes absent from training.
    """
    all_results = []
    all_test_predictions = []
    fitted = {}
    split_details = {}
    for test_id, subset in data.groupby("test_id"):
        train, validation, test = chronological_split(subset)
        if min(len(train), len(validation), len(test)) < 10:
            raise ValueError(f"Not enough sessions for metric {test_id}")
        split_details[test_id] = {
            "train_end": str(train.session_date.max().date()),
            "validation_start": str(validation.session_date.min().date()),
            "validation_end": str(validation.session_date.max().date()),
            "test_start": str(test.session_date.min().date()),
            "test_end": str(test.session_date.max().date()),
            "train_n": len(train), "validation_n": len(validation), "test_n": len(test),
        }
        valid_scores = {}
        trained = {}
        for name, candidate in candidates().items():
            if candidate is not None:
                candidate.fit(train[FEATURES], train["value"])
            trained[name] = candidate
            valid_scores[name] = metrics(validation["value"], predict(name, candidate, validation))["mae"]
        chosen = min(valid_scores, key=valid_scores.get)
        # Validation selected the algorithm. Refit that algorithm on pre-test rows.
        selected_model = candidates()[chosen]
        if selected_model is not None:
            earlier = pd.concat([train, validation], ignore_index=True)
            selected_model.fit(earlier[FEATURES], earlier["value"])
        fitted[test_id] = (chosen, selected_model)
        for name, candidate in trained.items():
            if name == chosen:
                candidate = selected_model
            predicted = predict(name, candidate, test)
            report = metrics(test["value"], predicted)
            all_results.append({
                "test_id": test_id, "unit": test.unit.iloc[0], "model": name,
                "chosen_on_validation": name == chosen,
                "validation_mae": round(valid_scores[name], 4),
                "n_test": len(test), **{key: round(value, 4) for key, value in report.items()},
            })
            out = test[["athlete_id", "session_date", "test_id", "value", "unit"]].copy()
            out["model"] = name
            out["prediction"] = np.round(predicted, 4)
            out["split"] = "test"
            all_test_predictions.append(out)
    return (
        pd.DataFrame(all_results),
        pd.concat(all_test_predictions, ignore_index=True),
        fitted,
        split_details,
    )
