"""Transparent unusual-performance-drop detection (not injury prediction)."""

from __future__ import annotations

import pandas as pd
import numpy as np

from .features import validate_and_sort


def find_unusual_drops(data: pd.DataFrame, *, min_drop_pct: float = 0.06,
                       z_threshold: float = 2.5) -> pd.DataFrame:
    """Flag surprising drops relative to the 5 preceding sessions for each athlete/test.

    The 5-session median/MAD uses only earlier records; a 1.5% absolute relative
    median floor prevents near-zero variability from generating spurious z scores.
    Requires four prior observations. No medical/injury interpretation is implied.
    """
    df = validate_and_sort(data)
    group_cols = ["athlete_id", "test_id"]
    histories = df.groupby(group_cols)["value"].shift(1)
    frame = df[group_cols].copy()
    frame["history"] = histories
    prior_median = frame.groupby(group_cols)["history"].transform(
        lambda v: v.rolling(5, min_periods=4).median()
    )
    # Individual historical rolling-window median absolute deviations; no current value.
    def prior_mad(v: pd.Series) -> pd.Series:
        return v.rolling(5, min_periods=4).apply(
            lambda a: float(np.median(np.abs(a - np.median(a)))), raw=True
        )
    prior_mad_values = frame.groupby(group_cols)["history"].transform(prior_mad)
    scale = np.maximum(prior_mad_values * 1.4826, prior_median * 0.015)
    df["historical_median"] = prior_median
    df["change_from_median_pct"] = (df.value / prior_median - 1) * 100
    df["historical_robust_z"] = (df.value - prior_median) / scale
    df["unusual_drop"] = (
        (df.change_from_median_pct <= -(min_drop_pct * 100)) &
        (df.historical_robust_z <= -z_threshold)
    ).fillna(False).astype(bool)
    return df
