"""Secondary athlete-held-out check and athlete-cluster uncertainty intervals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FEATURES
from .forecast import candidates, metrics, predict


def held_out_athlete_evaluation(data: pd.DataFrame, *, seed: int = 17) -> tuple[pd.DataFrame, dict]:
    """Learn population patterns on other athletes; never train on test-athlete IDs.

    New athlete forecasts still require at least three PRIOR sessions for that
    athlete, but do not see that athlete's data during the model fitting stage.
    This answers a DIFFERENT generalization question from chronological holdout.
    """
    athlete_ids = np.array(sorted(data.athlete_id.unique()))
    if len(athlete_ids) < 12:
        raise ValueError("Need at least 12 athletes to test athlete-held-out generalization")
    rng = np.random.default_rng(seed)
    rng.shuffle(athlete_ids)
    i, j = int(.65 * len(athlete_ids)), int(.8 * len(athlete_ids))
    train_ids = set(athlete_ids[:i])
    valid_ids = set(athlete_ids[i:j])
    test_ids = set(athlete_ids[j:])
    assert not (train_ids & valid_ids or train_ids & test_ids or valid_ids & test_ids)
    details = {"train_athletes": len(train_ids), "validation_athletes": len(valid_ids),
               "test_athletes": len(test_ids), "method": "disjoint athletes (not chronological)"}
    results = []
    for metric, subset in data.groupby("test_id"):
        train = subset.loc[subset.athlete_id.isin(train_ids)]
        validation = subset.loc[subset.athlete_id.isin(valid_ids)]
        test = subset.loc[subset.athlete_id.isin(test_ids)]
        if min(len(train), len(validation), len(test)) < 10:
            raise ValueError(f"Too few sessions in held-out athlete split for {metric}")
        scores = {}
        for name, estimator in candidates().items():
            if estimator is not None:
                estimator.fit(train[FEATURES], train.value)
            scores[name] = metrics(validation.value, predict(name, estimator, validation))["mae"]
        winner = min(scores, key=scores.get)
        selected = candidates()[winner]
        if selected is not None:
            earlier = pd.concat([train, validation], ignore_index=True)
            selected.fit(earlier[FEATURES], earlier.value)
        winner_metrics = metrics(test.value, predict(winner, selected, test))
        baseline_metrics = metrics(test.value, predict("persistence", None, test))
        results.append({
            "test_id": metric, "unit": test.unit.iloc[0], "validation_selected": winner,
            "n_test": len(test), "test_athletes": test.athlete_id.nunique(),
            "selected_mae": round(winner_metrics["mae"], 4),
            "persistence_mae": round(baseline_metrics["mae"], 4),
            "improvement_pct": round(100 * (baseline_metrics["mae"] - winner_metrics["mae"]) / baseline_metrics["mae"], 2),
        })
    return pd.DataFrame(results), details


def athlete_cluster_ci(predictions: pd.DataFrame, summary: pd.DataFrame,
                       *, draws: int = 1000, seed: int = 74) -> pd.DataFrame:
    """Paired bootstrap of percent MAE improvement versus persistence.

    Samples athletes rather than observations to avoid falsely treating an
    athlete's repeated weekly measurements as independent observations.
    """
    rng = np.random.default_rng(seed)
    output = []
    for metric, rows in summary.groupby("test_id"):
        winner = rows.loc[rows.chosen_on_validation, "model"].iloc[0]
        subset = predictions.loc[(predictions.test_id == metric) &
                                 (predictions.model.isin([winner, "persistence"]))].copy()
        subset["abs_error"] = (subset.value - subset.prediction).abs()
        if winner == "persistence":
            output.append({"test_id": metric, "chosen_model": winner,
                           "improvement_pct": 0.0, "ci_low_pct": 0.0, "ci_high_pct": 0.0,
                           "test_athletes": subset.athlete_id.nunique()})
            continue
        pivot = subset.pivot(index=["athlete_id", "session_date"],
                             columns="model", values="abs_error").reset_index()
        per = pivot.groupby("athlete_id")[["persistence", winner]].agg(["sum", "count"])
        base_sum = per[("persistence", "sum")].to_numpy()
        selected_sum = per[(winner, "sum")].to_numpy()
        count = per[("persistence", "count")].to_numpy()
        n = len(base_sum)
        choices = rng.integers(0, n, size=(draws, n))
        bases = base_sum[choices].sum(axis=1) / count[choices].sum(axis=1)
        selected = selected_sum[choices].sum(axis=1) / count[choices].sum(axis=1)
        changes = 100 * (bases - selected) / bases
        observed = 100 * (base_sum.sum() - selected_sum.sum()) / base_sum.sum()
        low, high = np.quantile(changes, [.025, .975])
        output.append({"test_id": metric, "chosen_model": winner,
                       "improvement_pct": round(float(observed), 2),
                       "ci_low_pct": round(float(low), 2), "ci_high_pct": round(float(high), 2),
                       "test_athletes": n})
    return pd.DataFrame(output)
