# Synthetic stress-test findings

**Status: exploratory engineering validation, not evidence of performance on real athletes.**
This report retains unsuccessful experiments rather than showing only the original favorable demonstration.

## Protocol

- Seeds: **42, 43, 44** (three independent *synthetic generator* seeds, not independent real cohorts).
- Each seed/scenario: **48 fictional athletes × 28 potential weekly sessions × 2 test types**.
- Four specified scenarios: **smooth**, **noisy**, **irregular**, **random_walk**.
- **Three non-overlapping chronological test blocks** per metric, with expanding training data. Models are selected by validation MAE using only earlier dates and refit on all pre-test data.
- Primary comparison: validation-selected algorithm versus **persistence** (the previous test measurement). Both use the same test observations.
- Across scenarios, this yields **72 scenario × seed × fold × metric evaluations**. Nine per scenario/metric; evaluations from the same seed are correlated.
- No changes to selected algorithms' hyperparameters in the stress experiment.

Reproduce from the project root:

~~~bash
python -m pip install -e ".[dev]"
athlete-insights evidence --out outputs/evidence --seeds 42 43 44 --athletes 48 --weeks 28 --folds 3
~~~

This generates the full fold-level audit file at outputs/evidence/walk_forward_detail.csv along with a scenario_summary.csv and STRESS_TEST_REPORT.md. A rounded aggregate snapshot from the exploratory local run is in [stress_summary.csv](stress_summary.csv).

## Actual findings from the local synthetic run

| Scenario | Test | Mean MAE reduction vs persistence | Positive fold/seed results |
| --- | --- | ---: | ---: |
| Smooth | Jump | **-0.29%** | 2 of 9 |
| Smooth | Grip | **-3.40%** | 1 of 9 |
| Noisy | Jump | **+10.87%** | 9 of 9 |
| Noisy | Grip | **+7.68%** | 9 of 9 |
| Irregular | Jump | **+0.06%** | 4 of 9 |
| Irregular | Grip | **-1.81%** | 3 of 9 |
| Random walk | Jump | **0.00%** | 0 of 9 |
| Random walk | Grip | **0.00%** | 0 of 9 |

Negative means worse prediction error than the simple previous-value baseline. Zero means no improvement; notably, the model-selection procedure commonly favors persistence in the random-walk control. Across all 72 evaluations, **18 had negative measured improvements**, and several selected persistence (0% change). Avoid describing the means as statistically significant.

The *single-seed 120-athlete, 32-week* example in [EVALUATION_REPORT.md](EVALUATION_REPORT.md) reports a different experiment and more favorable numbers. It cannot be combined with the table above as though it were a replication. Changing sample size, simulated timeline and synthetic cohort changes the outcome.

## What this changes about the claim

The original demo is a successful **software-methodology demonstration** (historical features, forecasting, benchmarks, and holdout discipline). It does **not** establish that the learned models reliably outperform persistence across different conditions, even within the synthetic generator. Under a smaller smooth-data design, the overall performance advantage disappears. Higher simulated measurement noise favors learning models in this experiment, but it does not prove an advantage under realistic instrument noise.

The detail output includes **athletes_worse_pct** (percentage of test athletes whose MAE was worse than persistence), because an improvement in pooled error can conceal individual failures.

## Major remaining limitations

- The data-generating process and scenario controls are artificial and known to the researcher; these tests do **not** replace independently collected data.
- Dates in adjacent folds and results from the same synthetic cohort are correlated. Nine fold/seed evaluations should not be treated as nine independent samples for a p-value or confidence interval.
- Results depend on sample sizes, feature availability, generator assumptions, and current model settings. No claim of general robustness is warranted.
- Each forecast uses previously observed test-period sessions when predicting later sessions. This is **rolling one-step-ahead monitoring**, not a many-week-ahead forecast.
- No age/sport/site/cohort external validity test or real-world instrument calibration has been performed.
- Anomaly alerts still lack realistic injury/fatigue ground truth and must not be interpreted medically.

**Next real research step:** identify an openly licensed or expressly authorized longitudinal dataset with repeated comparable measurements, freeze the evaluation protocol *before viewing outcomes*, and run the same baselines, chronological splits, subgroup error analysis and uncertainty analysis. Only then discuss external predictive utility.
