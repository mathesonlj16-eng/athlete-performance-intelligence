"""Generate explicitly synthetic test-session data for reproducible ML experiments."""

from __future__ import annotations

import numpy as np
import pandas as pd

METRICS = {
    "countermovement_jump": ("cm", 36.0, 7.0),
    "grip_strength": ("kg", 47.0, 9.0),
}


def generate_synthetic_data(*, athletes: int = 120, weeks: int = 32, seed: int = 42,
                            scenario: str = "smooth") -> pd.DataFrame:
    """Simulate smooth, noisy, irregular or random-walk fictional trajectories.

    The default smooth generator retains the original fixed-seed data. Anomaly
    labels are for evaluation only, never prediction. No clinical claims.
    """
    if scenario not in {"smooth", "noisy", "random_walk", "irregular"}:
        raise ValueError("Unknown synthetic scenario")
    if athletes < 12 or weeks < 18:
        raise ValueError("Use at least 12 athletes and 18 weeks for useful temporal splits")
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2025-01-06")
    records: list[dict] = []
    for i in range(athletes):
        athlete_id = f"SYN{i + 1:04d}"
        athlete_trend = rng.normal(0.0008, 0.00045)
        for metric, (unit, typical, spread) in METRICS.items():
            skill = max(typical * 0.55, rng.normal(typical, spread))
            trend = athlete_trend + rng.normal(0, 0.0003)
            noise_state = 0.0
            event_week = (
                int(rng.integers(weeks // 2, weeks - 3))
                if scenario != "random_walk" and rng.random() < 0.32 else -1
            )
            for week in range(weeks):
                if scenario == "random_walk":
                    # No predictably smooth drift or injected drop; changes are
                    # independent innovations. Persistence should be hard to beat.
                    noise_state += rng.normal(0, 0.025)
                    observed = skill * np.exp(noise_state)
                    anomaly = 0
                else:
                    # Original smooth process (unchanged for scenario="smooth").
                    noise_state = 0.55 * noise_state + rng.normal(0, 0.018)
                    cyclical_training = 0.014 * np.sin((week + i % 7) / 4.5)
                    latent = skill * (1 + trend * week + cyclical_training + noise_state)
                    anomaly = int(week == event_week)
                    observed = latent * (0.88 if anomaly else 1.0)
                    observed *= (1 + rng.normal(0, 0.025 if scenario == "noisy" else 0.006))
                if scenario == "irregular" and week >= 4 and rng.random() < 0.25:
                    # Missed visits are independent of the observed value.
                    continue
                records.append({
                    "athlete_id": athlete_id,
                    "session_date": str((start + pd.Timedelta(days=week * 7)).date()),
                    "test_id": metric,
                    "value": round(max(observed, 1.0), 3),
                    "unit": unit,
                    "synthetic_event": anomaly,  # Evaluations ONLY; excluded from features.
                })
    return pd.DataFrame.from_records(records)
