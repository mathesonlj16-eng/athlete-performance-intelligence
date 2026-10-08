"""Validation and strictly historical one-step prediction features."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = {"athlete_id", "session_date", "test_id", "value", "unit"}
FEATURES = [
    "lag_1", "lag_2", "lag_3", "past_mean_3", "past_std_3",
    "past_change_pct", "days_since_prev", "session_number",
]


def validate_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Data set is empty")
    out = df.copy()
    for col in ["athlete_id", "test_id", "unit"]:
        if out[col].isna().any() or (out[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Blank or null values in {col}")
        out[col] = out[col].astype(str).str.strip()
    parsed_dates = pd.to_datetime(out["session_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise ValueError("Invalid or missing session_date")
    out["session_date"] = parsed_dates.dt.normalize()
    out["value"] = pd.to_numeric(out["value"], errors="raise")
    if not np.isfinite(out["value"]).all() or (out["value"] <= 0).any():
        raise ValueError("All test values must be finite and positive")
    if out.duplicated(["athlete_id", "test_id", "session_date"]).any():
        raise ValueError("Repeated athlete/test/date rows: aggregate trials before analysis")
    if (out.groupby("test_id")["unit"].nunique() > 1).any():
        raise ValueError("Multiple units for one test: convert values before analysis")
    return out.sort_values(["athlete_id", "test_id", "session_date"]).reset_index(drop=True)


def make_historical_features(data: pd.DataFrame) -> pd.DataFrame:
    """Rows at t contain only t-1 and earlier measurement features.

    Test feature rows may use earlier test-period *observations*, consistent with
    rolling one-step-ahead forecasts, never values from t or t+1.
    """
    df = validate_and_sort(data)
    group = df.groupby(["athlete_id", "test_id"], sort=False)
    df["lag_1"] = group["value"].shift(1)
    df["lag_2"] = group["value"].shift(2)
    df["lag_3"] = group["value"].shift(3)
    # Explicitly use shifted values to avoid current-row target leakage.
    group_lag = df.groupby(["athlete_id", "test_id"], sort=False)["lag_1"]
    df["past_mean_3"] = group_lag.transform(lambda x: x.rolling(3, min_periods=3).mean())
    df["past_std_3"] = group_lag.transform(lambda x: x.rolling(3, min_periods=3).std(ddof=0))
    df["past_change_pct"] = (df["lag_1"] - df["lag_2"]) / df["lag_2"]
    df["days_since_prev"] = group["session_date"].diff().dt.days
    df["session_number"] = group.cumcount()
    # Stable weekly tests are not required, but missing history is intentionally dropped.
    return df.loc[df[FEATURES].notna().all(axis=1)].copy().reset_index(drop=True)


def chronological_split(df: pd.DataFrame, *, train_fraction: float = .65,
                        validation_fraction: float = .15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = np.sort(df["session_date"].unique())
    if len(dates) < 12:
        raise ValueError("At least 12 distinct test-session dates are required")
    i = int(len(dates) * train_fraction)
    j = int(len(dates) * (train_fraction + validation_fraction))
    if not 0 < i < j < len(dates):
        raise ValueError("Insufficient dates for chronological train/validation/test split")
    train = df.loc[df.session_date < dates[i]].copy()
    validation = df.loc[(df.session_date >= dates[i]) & (df.session_date < dates[j])].copy()
    test = df.loc[df.session_date >= dates[j]].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Each split needs at least one valid sample")
    return train, validation, test
