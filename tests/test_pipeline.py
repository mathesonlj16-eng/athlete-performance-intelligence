import numpy as np
import pandas as pd
import pytest

from athlete_insights.anomalies import find_unusual_drops
from athlete_insights.features import FEATURES, chronological_split, make_historical_features, validate_and_sort
from athlete_insights.forecast import evaluate
from athlete_insights.synthetic import generate_synthetic_data


def example(n=14):
    dates = pd.date_range("2025-01-01", periods=n, freq="7D")
    return pd.DataFrame({
        "athlete_id": ["A"] * n,
        "session_date": dates.astype(str),
        "test_id": ["jump"] * n,
        "unit": ["cm"] * n,
        "value": np.arange(20, n+20).astype(float),
    })


def test_no_current_or_future_target_leakage():
    df = example()
    before = make_historical_features(df)
    # Editing the current or any later target cannot change features at this date.
    df.loc[8:, "value"] = 999.0
    after = make_historical_features(df)
    key = pd.Timestamp("2025-02-26")  # row 8
    first = before.loc[before.session_date == key, FEATURES]
    second = after.loc[after.session_date == key, FEATURES]
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))


def test_duplicate_rows_and_unit_mismatch_rejected():
    df = example()
    with pytest.raises(ValueError, match="Repeated"):
        validate_and_sort(pd.concat([df, df.iloc[:1]], ignore_index=True))
    df.loc[4, "unit"] = "inch"
    with pytest.raises(ValueError, match="Multiple units"):
        validate_and_sort(df)


@pytest.mark.parametrize("bad_date", [None, "", "not-a-date"])
def test_missing_or_malformed_session_dates_rejected(bad_date):
    df = example()
    df.loc[3, "session_date"] = bad_date
    with pytest.raises(ValueError, match="Invalid or missing session_date"):
        validate_and_sort(df)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1.0, 0.0])
def test_nonfinite_or_nonpositive_readings_rejected(bad_value):
    df = example()
    df.loc[3, "value"] = bad_value
    with pytest.raises(ValueError, match="finite and positive"):
        validate_and_sort(df)


def test_unsorted_input_produces_the_same_historical_features():
    df = example()
    chronological = make_historical_features(df)
    shuffled = make_historical_features(df.sample(frac=1.0, random_state=3))
    pd.testing.assert_frame_equal(chronological, shuffled)


def test_anomaly_uses_prior_sessions_only():
    df = example(13)
    df.loc[8, "value"] = 10.0
    flagged = find_unusual_drops(df)
    assert not flagged.loc[:3, "unusual_drop"].any()
    assert bool(flagged.loc[8, "unusual_drop"])
    assert flagged.loc[8, "historical_median"] > 10.0


def test_date_splits_do_not_overlap():
    df = make_historical_features(generate_synthetic_data(athletes=12, weeks=22))
    train, val, test = chronological_split(df)
    assert train.session_date.max() < val.session_date.min()
    assert val.session_date.max() < test.session_date.min()


def test_synthetic_labels_excluded_from_features():
    df = generate_synthetic_data(athletes=12, weeks=18)
    features = make_historical_features(df)
    assert "synthetic_event" not in FEATURES
    assert all(name not in FEATURES for name in ("athlete_id", "value", "session_date"))
    assert "synthetic_event" in features.columns


def test_full_research_pipeline_has_baselines():
    features = make_historical_features(generate_synthetic_data(athletes=18, weeks=22))
    summary, preds, fitted, splits = evaluate(features)
    assert len(summary) == 8  # 2 tests x 4 methods
    assert set(summary.model) == {"persistence", "recent_3_mean", "ridge", "hist_gradient_boosting"}
    assert summary.groupby("test_id").chosen_on_validation.sum().eq(1).all()
    assert set(fitted) == {"countermovement_jump", "grip_strength"}
    assert not preds.empty
    assert all(s["train_end"] < s["validation_start"] <= s["validation_end"] < s["test_start"] for s in splits.values())


def test_subjects_do_not_leak_across_splits_and_bootstrap_is_finite():
    from athlete_insights.robustness import held_out_athlete_evaluation, athlete_cluster_ci
    df = make_historical_features(generate_synthetic_data(athletes=20, weeks=24))
    held_out, detail = held_out_athlete_evaluation(df)
    assert detail["train_athletes"] + detail["validation_athletes"] + detail["test_athletes"] == 20
    assert (held_out.test_athletes == detail["test_athletes"]).all()
    summary, predictions, _, _ = evaluate(df)
    boot = athlete_cluster_ci(predictions, summary, draws=30)
    assert np.isfinite(boot[["ci_low_pct", "ci_high_pct"]]).to_numpy().all()
    assert (boot.ci_low_pct <= boot.ci_high_pct).all()
