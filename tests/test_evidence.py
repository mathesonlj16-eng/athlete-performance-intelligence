"""Evidence-suite checks: time ordering, stress controls, and reproducibility."""

import pandas as pd
import pytest

from athlete_insights.evidence import (
    SCENARIOS, evaluate_walk_forward, expanding_window_folds, run_synthetic_stress_test,
)
from athlete_insights.features import make_historical_features
from athlete_insights.synthetic import generate_synthetic_data


def test_walk_forward_test_blocks_are_disjoint_and_sequential():
    features = make_historical_features(generate_synthetic_data(athletes=14, weeks=26))
    jump = features.loc[features.test_id == "countermovement_jump"]
    folds = list(expanding_window_folds(jump, folds=3))
    assert len(folds) == 3
    test_dates = []
    for _, train, validation, test in folds:
        assert train.session_date.max() < validation.session_date.min()
        assert validation.session_date.max() < test.session_date.min()
        test_dates.extend(test.session_date.unique())
    assert len(test_dates) == len(set(test_dates))
    # Later folds may use *past* evaluation data for training, never future data.
    assert folds[0][3].session_date.max() < folds[1][3].session_date.min()
    assert folds[1][3].session_date.max() < folds[2][3].session_date.min()


def test_random_walk_is_a_genuine_no_injected_event_control():
    walk = generate_synthetic_data(athletes=12, weeks=22, seed=23, scenario="random_walk")
    assert not walk.synthetic_event.any()
    assert (walk.value > 0).all()


def test_scenario_generation_is_deterministic_and_default_is_unchanged():
    original = generate_synthetic_data(athletes=12, weeks=22, seed=7)
    explicit = generate_synthetic_data(athletes=12, weeks=22, seed=7, scenario="smooth")
    pd.testing.assert_frame_equal(original, explicit)
    for scenario in SCENARIOS:
        a = generate_synthetic_data(athletes=12, weeks=22, seed=11, scenario=scenario)
        b = generate_synthetic_data(athletes=12, weeks=22, seed=11, scenario=scenario)
        pd.testing.assert_frame_equal(a, b)


def test_walk_forward_reports_each_selected_model_vs_persistence():
    features = make_historical_features(generate_synthetic_data(athletes=14, weeks=24, seed=13))
    result = evaluate_walk_forward(features, folds=2)
    assert len(result) == 4  # 2 metrics times 2 non-overlapping test blocks
    assert (result.n_test > 0).all()
    assert (result.selected_mae >= 0).all()
    assert (result.baseline_mae > 0).all()
    assert result.athletes_worse_pct.between(0, 100).all()
    assert (result.train_end < result.validation_start).all()
    assert (result.validation_end < result.test_start).all()


def test_evidence_writes_transparent_outputs(tmp_path):
    detail, summary = run_synthetic_stress_test(
        out=tmp_path, seeds=(2,), scenarios=("random_walk",),
        athletes=12, weeks=22, folds=2,
    )
    assert len(detail) == 4
    assert len(summary) == 2
    assert (tmp_path / "walk_forward_detail.csv").is_file()
    assert (tmp_path / "scenario_summary.csv").is_file()
    report = (tmp_path / "STRESS_TEST_REPORT.md").read_text(encoding="utf-8")
    assert "not external validation" in report
    assert "random_walk" in report


@pytest.mark.parametrize("scenario", ["actual_records", "", "medical"])
def test_unknown_scenario_rejected(scenario):
    with pytest.raises(ValueError, match="Unknown synthetic scenario"):
        generate_synthetic_data(scenario=scenario)
