"""Walk-forward forecasting stress tests using explicitly synthetic data.

These are engineering validation exercises, not estimates of field performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .features import FEATURES, make_historical_features
from .forecast import candidates, metrics, predict
from .synthetic import generate_synthetic_data

SCENARIOS = ("smooth", "noisy", "random_walk", "irregular")


def expanding_window_folds(data: pd.DataFrame, *, folds: int = 3):
    """Return non-overlapping test blocks with earlier train/validation dates.

    Later folds can train on observations from previous test blocks, as they
    would become available during real one-step-ahead monitoring.
    """
    dates = np.sort(data.session_date.unique())
    if folds < 2 or len(dates) < max(18, 3 * folds + 6):
        raise ValueError("Insufficient distinct dates for expanding-window validation")
    first_test = int(len(dates) * 0.60)
    blocks = np.array_split(np.arange(first_test, len(dates)), folds)
    validation_dates = max(3, int(len(dates) * 0.15))
    for fold_no, block in enumerate(blocks, start=1):
        if len(block) < 2 or block[0] - validation_dates < 4:
            raise ValueError("Not enough history for disjoint test blocks")
        first_val = int(block[0]) - validation_dates
        train = data.loc[data.session_date < dates[first_val]].copy()
        validation = data.loc[(data.session_date >= dates[first_val]) &
                              (data.session_date < dates[block[0]])].copy()
        test = data.loc[data.session_date.isin(dates[block])].copy()
        if min(len(train), len(validation), len(test)) < 10:
            raise ValueError("Each fold requires at least ten feature-eligible observations per split")
        yield fold_no, train, validation, test


def evaluate_walk_forward(features: pd.DataFrame, *, folds: int = 3) -> pd.DataFrame:
    """Select by validation MAE, refit before test, and compare with persistence."""
    results = []
    for test_id, subset in features.groupby("test_id"):
        for fold, train, validation, test in expanding_window_folds(subset, folds=folds):
            validation_mae = {}
            for name, model in candidates().items():
                if model is not None:
                    model.fit(train[FEATURES], train.value)
                validation_mae[name] = metrics(
                    validation.value, predict(name, model, validation)
                )["mae"]
            selected_name = min(validation_mae, key=validation_mae.get)
            selected = candidates()[selected_name]
            if selected is not None:
                pretest = pd.concat([train, validation], ignore_index=True)
                selected.fit(pretest[FEATURES], pretest.value)
            selected_prediction = predict(selected_name, selected, test)
            baseline_prediction = predict("persistence", None, test)
            selected_error = np.abs(test.value.to_numpy() - selected_prediction)
            baseline_error = np.abs(test.value.to_numpy() - baseline_prediction)
            selected_mae = float(selected_error.mean())
            baseline_mae = float(baseline_error.mean())
            # Pooled errors can hide athletes harmed by the selected model.
            athlete_errors = pd.DataFrame({
                "athlete_id": test.athlete_id.to_numpy(),
                "selected_error": selected_error, "baseline_error": baseline_error,
            }).groupby("athlete_id")[["selected_error", "baseline_error"]].mean()
            results.append({
                "test_id": test_id, "unit": test.unit.iloc[0], "fold": fold,
                "train_end": str(train.session_date.max().date()),
                "validation_start": str(validation.session_date.min().date()),
                "validation_end": str(validation.session_date.max().date()),
                "test_start": str(test.session_date.min().date()),
                "test_end": str(test.session_date.max().date()),
                "n_test": len(test), "n_athletes": test.athlete_id.nunique(),
                "selected_model": selected_name,
                "selected_mae": selected_mae, "baseline_mae": baseline_mae,
                "improvement_pct": 100 * (baseline_mae - selected_mae) / baseline_mae
                if baseline_mae > 0 else np.nan,
                "athletes_worse_pct": 100 * float(
                    (athlete_errors.selected_error > athlete_errors.baseline_error).mean()
                ),
            })
    return pd.DataFrame(results)


def run_synthetic_stress_test(*, out: Path, seeds: tuple[int, ...] = (42, 43, 44),
                              athletes: int = 48, weeks: int = 28,
                              scenarios: tuple[str, ...] = SCENARIOS,
                              folds: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat independent fictional cohorts over regimes, without test-set tuning."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide one or more distinct random seeds")
    if not scenarios or any(s not in SCENARIOS for s in scenarios):
        raise ValueError(f"Scenarios must be selected from {SCENARIOS}")
    results = []
    for scenario in scenarios:
        for seed in seeds:
            data = generate_synthetic_data(athletes=athletes, weeks=weeks,
                                           seed=seed, scenario=scenario)
            features = make_historical_features(data)
            fold_results = evaluate_walk_forward(features, folds=folds)
            fold_results.insert(0, "seed", seed)
            fold_results.insert(0, "scenario", scenario)
            results.append(fold_results)
    detail = pd.concat(results, ignore_index=True)
    summary = detail.groupby(["scenario", "test_id", "unit"], as_index=False).agg(
        evaluations=("improvement_pct", "size"),
        mean_improvement_pct=("improvement_pct", "mean"),
        min_improvement_pct=("improvement_pct", "min"),
        max_improvement_pct=("improvement_pct", "max"),
        fraction_improved=("improvement_pct", lambda x: float((x > 0).mean())),
        mean_athletes_worse_pct=("athletes_worse_pct", "mean"),
    )
    out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(out / "walk_forward_detail.csv", index=False, float_format="%.4f")
    summary.to_csv(out / "scenario_summary.csv", index=False, float_format="%.4f")
    preview = summary.round(2).to_csv(index=False)
    (out / "STRESS_TEST_REPORT.md").write_text(
        "# Synthetic forecasting stress tests\n\n"
        "**Synthetic engineering evidence only, not external validation.** "
        "Independent generator seeds are not independent real-world cohorts. "
        "Scenarios intentionally vary difficulty, not estimated frequencies of real patterns.\n\n"
        f"Seeds: {', '.join(str(s) for s in seeds)}. "
        f"Folds per seed/scenario/test: {folds}. "
        f"Fictional athletes per seed: {athletes}; weeks: {weeks}.\n\n"
        "## Scenario summary\n\n" + "~~~csv\n" + preview + "~~~\n\n"
        "Each evaluation selects an algorithm on preceding validation dates "
        "and refits on training plus validation only. The reference is predicting "
        "the preceding reading (persistence). Later folds reuse previous test "
        "dates only once those dates are in the past. Test blocks never overlap.\n\n"
        "**Interpretation:** Negative improvements mean the selected algorithm "
        "was worse than persistence. fraction_improved is a descriptive "
        "proportion of synthetic fold/seed evaluations, not a probability "
        "of success. mean_athletes_worse_pct reveals uneven individual results. "
        "Min/max across runs are not confidence intervals. No statistical "
        "significance or clinical claims are established.\n\n"
        "**Scenarios:** smooth = original low-noise generator; "
        "noisy = larger measurement errors; random_walk = independent "
        "changes without injected drops (persistence is a strong reference); "
        "irregular = missed visits from a smooth underlying process.\n\n"
        "See walk_forward_detail.csv for chronological boundaries and "
        "all per-fold negative and positive outcomes. "
        "Real-data external validation remains required.\n",
        encoding="utf-8",
    )
    return detail, summary
