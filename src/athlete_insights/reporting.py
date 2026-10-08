"""Generate transparent, lightweight evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _markdown_table(df: pd.DataFrame) -> str:
    # Keep README/report readable without a markdown dependency.
    columns = list(df.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in df.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def make_charts(summary: pd.DataFrame, predictions: pd.DataFrame,
                flagged: pd.DataFrame, folder: Path) -> None:
    figure_folder = folder / "figures"
    figure_folder.mkdir(parents=True, exist_ok=True)
    for metric, rows in summary.groupby("test_id"):
        # Only the validation-selected model is refit using train + validation.
        # Compare it visually with the persistence baseline, not with train-only
        # exploratory fits that use less training data.
        selected_name = rows.loc[rows.chosen_on_validation, "model"].iloc[0]
        rows = rows.loc[rows.model.isin(["persistence", selected_name])]
        fig, ax = plt.subplots(figsize=(8, 4.6))
        rows = rows.sort_values("mae", ascending=False)
        bars = ax.barh(rows.model, rows.mae)
        for bar, selected in zip(bars, rows.chosen_on_validation):
            if selected:
                bar.set_hatch("//")
        ax.set_xlabel(f"Test-period MAE ({rows.unit.iloc[0]}) — lower is better")
        ax.set_title(f"One-step-ahead forecasting: {metric.replace('_', ' ')}")
        ax.grid(axis="x", alpha=.2)
        fig.tight_layout()
        fig.savefig(figure_folder / f"forecast_{metric}.png", dpi=150)
        plt.close(fig)

    if not flagged.empty:
        sample = flagged.loc[flagged.unusual_drop].head(1)
        if sample.empty:
            sample = flagged.head(1)
        athlete_id = sample.athlete_id.iloc[0]
        metric = sample.test_id.iloc[0]
        rows = flagged.loc[(flagged.athlete_id == athlete_id) & (flagged.test_id == metric)].sort_values("session_date")
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(rows.session_date, rows.value, marker="o", label="Observed")
        hits = rows.loc[rows.unusual_drop]
        if not hits.empty:
            ax.scatter(hits.session_date, hits.value, color="crimson", marker="x", s=100, zorder=5, label="Unusual drop")
        ax.set_title(f"Example athlete {athlete_id}: {metric.replace('_', ' ')}")
        ax.set_ylabel(rows.unit.iloc[0]); ax.set_xlabel("Session date")
        ax.legend(); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(figure_folder / "unusual_drops_example.png", dpi=150)
        plt.close(fig)


def write_report(*, out: Path, summary: pd.DataFrame, predictions: pd.DataFrame,
                 flagged: pd.DataFrame, splits: dict, demo: bool,
                 athlete_holdout: pd.DataFrame, athlete_split: dict, bootstrap: pd.DataFrame) -> None:
    with (out / "splits.json").open("w", encoding="utf-8") as fp:
        json.dump({"chronological": splits, "disjoint_athletes": athlete_split}, fp, indent=2)
    labels = "SYNTHETIC DATA — illustrative only" if demo else "USER-SUPPLIED DATA — applicability not established"
    display = summary[["test_id", "model", "chosen_on_validation", "n_test", "mae", "rmse", "mape_pct"]]
    counts = flagged.groupby("test_id")["unusual_drop"].agg(["sum", "count"]).reset_index()
    counts.columns = ["test_id", "flags", "observations"]

    selection_lines = []
    for metric, group in summary.groupby("test_id"):
        selected = group.loc[group.chosen_on_validation].iloc[0]
        persistence = group.loc[group.model == "persistence"].iloc[0]
        gain_pct = (persistence.mae - selected.mae) / persistence.mae * 100
        direction = "better" if gain_pct >= 0 else "worse"
        selection_lines.append(
            f"- **{metric}**: validation-selected **{selected.model}**; test MAE "
            f"**{selected.mae:.3f} {selected.unit}** versus persistence "
            f"**{persistence.mae:.3f} {persistence.unit}** "
            f"({abs(gain_pct):.1f}% {direction} on the untouched test dates)."
        )
    anomaly_eval = ""
    if demo and "synthetic_event" in flagged.columns:
        tp = int((flagged.unusual_drop & (flagged.synthetic_event == 1)).sum())
        fp = int((flagged.unusual_drop & (flagged.synthetic_event == 0)).sum())
        fn = int(((~flagged.unusual_drop) & (flagged.synthetic_event == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        anomaly_eval = (
            "\n## Detecting injected synthetic drops\n\n"
            f"Injected-drop detection: precision **{precision:.1%}**, recall **{recall:.1%}** "
            f"(TP={tp}, FP={fp}, FN={fn}). This measures only the artificial events "
            "in the generator, not medically relevant events or real-world accuracy.\n"
        )
    report = f"""# Athlete Performance Intelligence — evaluation report

**Dataset:** {labels}. This is an experimental research prototype, **not an injury-risk model or coaching recommendation engine**.

## Objective

Forecast the **next observed test result** for an athlete using their past test sessions, and flag large drops relative to prior sessions. Separate models are evaluated for each test type to avoid mixing measurements with different units.

## Test-set results

{_markdown_table(display)}

A lower MAE/RMSE/MAPE is better. The validation set selects the preferred method; the final test period is held out from selection. The **primary fair comparison** is the validation-selected method (refit on train + validation) against the last-value persistence baseline. Unselected model rows use train-only fits and are exploratory, not a level-training-data test-set ranking. Charts therefore show only the selected method versus persistence (the same method may be both).

## Validation-selected model versus persistence

{chr(10).join(selection_lines)}

## Uncertainty: resampling athletes, not sessions

{_markdown_table(bootstrap)}

The intervals show 95% **athlete-cluster bootstrap intervals** for improvement in test-period MAE over persistence. Intervals reflect only sampling variability in this synthetic cohort, not uncertainty about deployment to actual athletes. Confidence intervals crossing zero do not establish an improvement.

## Generalization to athletes excluded from model training

{_markdown_table(athlete_holdout)}

This is an **independent athlete-held-out experiment**, not a chronological forecast. Athlete IDs are disjoint across training, validation and test, with validation used for model selection. The test athletes still provide their own prior readings as input, but no test-athlete readings are used to fit the models. This probes a different failure mode than the out-of-time test above.

## Drop detection

{_markdown_table(counts)}

Uses each athlete's previous up-to-five results only. A flag requires at least four historical sessions, a drop of at least 6% from the historical median, and a robust z score at most -2.5. It is **not** evidence of injury, fatigue, or a medical condition.
{anomaly_eval}
## Evaluation controls and limitations

- Splits use chronological session dates with no overlap: train, validation, then test. Exact dates and sample counts appear in `splits.json`.
- Features are shifted and use only values observed before the current session. The synthetic-event label is never a prediction feature.
- The test simulates **one-step-ahead** monitoring: an actual new session becomes history before the *next* session. This is **not** a full-season forecast or a test on unseen athletes.
- Only the validation-selected model is refit on train + validation. All methods are reported on test dates, but the unselected fitted models retain their train-only fit; compare the selected model chiefly against the persistence baseline.
- Synthetic data results do **not** establish predictive accuracy on actual athletes. The generator was designed for educational validation of the pipeline, not biomechanical realism.
- No causal claims, injury diagnostics, or real client data are included. Real-data use requires consent/permissions, pseudonymization, and metric protocol checks.

## Next research milestone

Use appropriately authorized, de-identified longitudinal athlete test data; freeze the preprocessing and evaluation protocol; quantify error across sports, age bands, and individual test types; check performance against simple baselines; report uncertainty and failure cases before any practical deployment.

## Figures

- `figures/forecast_countermovement_jump.png` and `figures/forecast_grip_strength.png`: comparative test errors
- `figures/unusual_drops_example.png`: example one-athlete historical pattern
"""
    (out / "EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
