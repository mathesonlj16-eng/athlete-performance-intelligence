"""Command line entry point: reproducible demo or authorized de-identified input."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .anomalies import find_unusual_drops
from .features import make_historical_features, validate_and_sort
from .forecast import evaluate
from .reporting import make_charts, write_report
from .robustness import athlete_cluster_ci, held_out_athlete_evaluation
from .synthetic import generate_synthetic_data


def run_pipeline(data: pd.DataFrame, out: Path, *, demo: bool = False) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    clean = validate_and_sort(data)
    flagged = find_unusual_drops(clean)
    features = make_historical_features(clean)
    if features.empty:
        raise ValueError("No athletes have four or more chronological test sessions")
    summary, predictions, _, splits = evaluate(features)
    subject_summary, subject_split = held_out_athlete_evaluation(features)
    intervals = athlete_cluster_ci(predictions, summary)
    summary.to_csv(out / "forecast_comparison.csv", index=False)
    subject_summary.to_csv(out / "athlete_held_out_comparison.csv", index=False)
    intervals.to_csv(out / "athlete_bootstrap_intervals.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    flagged.to_csv(out / "unusual_drop_flags.csv", index=False)
    make_charts(summary, predictions, flagged, out)
    write_report(out=out, summary=summary, predictions=predictions, flagged=flagged,
                 splits=splits, demo=demo, athlete_holdout=subject_summary,
                 athlete_split=subject_split, bootstrap=intervals)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(prog="athlete-insights",
                                     description="Synthetic-data ML forecasting and unusual-drop research")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a fully reproducible synthetic-data demonstration")
    demo.add_argument("--athletes", type=int, default=120)
    demo.add_argument("--weeks", type=int, default=32)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--out", type=Path, default=Path("outputs/demo"))
    real = sub.add_parser("analyze", help="run on an authorized, de-identified long-format CSV")
    real.add_argument("--csv", type=Path, required=True)
    real.add_argument("--out", type=Path, default=Path("outputs/real"))
    args = parser.parse_args()
    if args.command == "demo":
        data = generate_synthetic_data(athletes=args.athletes, weeks=args.weeks, seed=args.seed)
        summary = run_pipeline(data, args.out, demo=True)
        data.to_csv(args.out / "synthetic_sessions.csv", index=False)
    else:
        data = pd.read_csv(args.csv)
        summary = run_pipeline(data, args.out, demo=False)
    print("\nRun complete. Forecast test results:\n")
    print(summary[["test_id", "model", "chosen_on_validation", "mae", "mape_pct"]].to_string(index=False))
    print(f"\nOutputs: {args.out.resolve() / 'EVALUATION_REPORT.md'}")


if __name__ == "__main__":
    main()
