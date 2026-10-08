"""Reproducible sensitivity experiments on explicitly synthetic measurements.

Designed to uncover failure modes, NOT estimate real-world predictive accuracy.
No model or perturbation setting is selected using test results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .features import make_historical_features
from .forecast import evaluate
from .synthetic import generate_synthetic_data

SCENARIOS = ("original", "measurement_noise", "missing_sessions", "late_level_shift")
DEFAULT_SEEDS = (11, 17, 23)


def apply_scenario(data: pd.DataFrame, scenario: str, *, seed: int) -> pd.DataFrame:
    """Perturb a fictional data set without using model predictions or test scores."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    out = data.copy(deep=True)
    rng = np.random.default_rng(seed + 10000)
    if scenario == "measurement_noise":
        # More measurement noise: 4% standard deviation, independent per row.
        out["value"] = (out.value.to_numpy() * np.clip(
            1 + rng.normal(0, 0.04, len(out)), 0.5, None
        )).round(3)
    elif scenario == "missing_sessions":
        # Missing-at-random visits: exclude 25% of observations. This changes
        # both history availability and the distribution of gaps between tests.
        keep = rng.random(len(out)) >= 0.25
        out = out.loc[keep].copy()
    elif scenario == "late_level_shift":
        # Simulated unannounced change in 30% of athlete trajectories,
        # beginning after roughly 82% of the originally scheduled weeks.
        # This is intentionally difficult for a model fitted on earlier data.
        athlete_ids = np.sort(out.athlete_id.unique())
        affected = set(rng.choice(athlete_ids, size=max(1, int(len(athlete_ids) * 0.3)), replace=False))
        dates = pd.to_datetime(out.session_date)
        week = ((dates - dates.min()).dt.days // 7).to_numpy()
        weeks = int(week.max()) + 1
        mask = (week >= int(0.82 * weeks)) & out.athlete_id.isin(affected).to_numpy()
        out.loc[mask, "value"] = (out.loc[mask, "value"] * 0.92).round(3)
    return out.reset_index(drop=True)


def run_stress_experiments(*, athletes: int = 48, weeks: int = 26,
                           seeds: tuple[int, ...] = DEFAULT_SEEDS) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one unique seed is required")
    details = []
    for seed in seeds:
        source = generate_synthetic_data(athletes=athletes, weeks=weeks, seed=seed)
        for scenario in SCENARIOS:
            modified = apply_scenario(source, scenario, seed=seed)
            features = make_historical_features(modified)
            summary, _, _, _ = evaluate(features)
            for metric, group in summary.groupby("test_id", sort=True):
                chosen = group.loc[group.chosen_on_validation].iloc[0]
                baseline = group.loc[group.model == "persistence"].iloc[0]
                details.append({
                    "scenario": scenario, "seed": seed, "metric": metric,
                    "n_observations": int((modified.test_id == metric).sum()),
                    "n_test": int(chosen.n_test),
                    "chosen_model": str(chosen.model),
                    "chosen_mae": float(chosen.mae),
                    "persistence_mae": float(baseline.mae),
                    "improvement_pct": round(
                        100 * (float(baseline.mae) - float(chosen.mae)) / float(baseline.mae), 2
                    ),
                })
    detailed = pd.DataFrame(details)
    # Show failures, variation, and seeds instead of cherry-picking the best run.
    grouped = (detailed.groupby(["scenario", "metric"], sort=False)
               .agg(runs=("seed", "size"), mean_improvement_pct=("improvement_pct", "mean"),
                    worst_improvement_pct=("improvement_pct", "min"),
                    best_improvement_pct=("improvement_pct", "max"),
                    runs_better_than_baseline=("improvement_pct", lambda a: int((a > 0).sum())))
               .reset_index())
    for col in ("mean_improvement_pct", "worst_improvement_pct", "best_improvement_pct"):
        grouped[col] = grouped[col].round(2)
    return detailed, grouped


def make_stress_charts(summary: pd.DataFrame, folder: Path) -> None:
    """Render both improvements and regressions; dashed line denotes parity."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folder.mkdir(parents=True, exist_ok=True)
    for metric, frame in summary.groupby("metric", sort=True):
        frame = frame.set_index("scenario").reindex(SCENARIOS).reset_index()
        y = np.arange(len(frame))
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        low = frame.mean_improvement_pct - frame.worst_improvement_pct
        high = frame.best_improvement_pct - frame.mean_improvement_pct
        ax.errorbar(frame.mean_improvement_pct, y, xerr=[low, high], fmt="o", capsize=5)
        ax.axvline(0, linestyle="--", linewidth=1, color="grey")
        ax.set_yticks(y, [s.replace("_", " ").title() for s in frame.scenario])
        ax.invert_yaxis()
        ax.set_xlabel("Mean % reduction in test MAE vs. last-value baseline (higher is better)")
        ax.set_title(f"Synthetic stress tests: {metric.replace('_', ' ').title()}")
        ax.grid(axis="x", alpha=0.15)
        for i, row in frame.iterrows():
            ax.text(row.best_improvement_pct + 0.3, i,
                    f"{int(row.runs_better_than_baseline)}/{int(row.runs)} improved",
                    va="center", fontsize=8)
        ax.set_xlim(float(frame.worst_improvement_pct.min()) - 5,
                    float(frame.best_improvement_pct.max()) + 9)
        fig.tight_layout()
        fig.savefig(folder / f"stress_{metric}.png", dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic-only stress tests for forecasting")
    parser.add_argument("--athletes", type=int, default=48)
    parser.add_argument("--weeks", type=int, default=26)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--out", type=Path, default=Path("outputs/stress"))
    args = parser.parse_args()
    detail, summary = run_stress_experiments(athletes=args.athletes, weeks=args.weeks,
                                            seeds=tuple(args.seeds))
    args.out.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out / "stress_runs.csv", index=False)
    summary.to_csv(args.out / "stress_summary.csv", index=False)
    make_stress_charts(summary, args.out / "figures")
    print(summary.to_string(index=False))
    print(f"\nReproduce: python -m athlete_insights.stress --out {args.out}")


if __name__ == "__main__":
    main()