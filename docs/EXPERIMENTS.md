# Experimental protocol and limitations

**Status:** synthetic-data methods exercise, not an externally validated predictive model.

## Research questions

1. Can a validation-selected forecasting algorithm improve next-observation accuracy over simply predicting the previous result?
2. Does a reported gain survive different synthetic seeds, extra measurement noise, missing visits, and sudden athlete-level changes?
3. Does an independently fit model work on athletes who were entirely excluded from training?

## Why compare against a trivial baseline?

Athlete measurements are autocorrelated. Repeating the most recent measurement is a strong, understandable forecast. A machine-learning method must outperform this simple rule consistently and on data it did not train on to demonstrate incremental value.

## Main fixed-seed demonstration

- 120 fictional athletes, 32 weekly sessions, two measurement types.
- Chronological train / validation / test windows (see [`../demo_results/splits.json`](../demo_results/splits.json)).
- Model choice made on validation MAE, never on test MAE.
- Last-observation persistence baseline versus rolling three-session average, Ridge and histogram gradient boosting.
- Fixed seed 42 gave 6.65% lower test MAE for countermovement jump and 10.78% lower test MAE for grip strength versus persistence. These figures reflect **one synthetic configuration**, and should not be presented without the stress experiments below.
- Disjoint-athlete split and athlete-cluster bootstrap are separately reported in the fixed-seed [`evaluation report`](../demo_results/EVALUATION_REPORT.md).

## Sensitivity experiments — all prespecified scenarios and seeds

Run `python -m athlete_insights.stress --out outputs/stress` using three independent seeds **11, 17, and 23**. Each run uses 48 fictional athletes, 26 weeks, and both metrics. The forecasting algorithms and hyperparameters are held fixed across scenarios; the selected algorithm is still determined only by validation data in each run.

| Synthetic scenario | Jump: mean MAE improvement | Grip: mean MAE improvement | Run count improved (jump / grip) |
|---|---:|---:|---:|
| Original synthetic generator | **-5.01%** | **-8.59%** | 0/3, 0/3 |
| Added independent 4% measurement noise | **+13.14%** | **+12.68%** | 3/3, 3/3 |
| Remove 25% of visits at random | **+0.96%** | **-3.34%** | 1/3, 1/3 |
| 8% persistent late drop in 30% of athletes | **-8.62%** | **-12.97%** | 0/3, 0/3 |

Positive improvement means *lower* forecasting error than the persistence baseline; negative means **worse**. Mean values are arithmetic averages across the three selected seeds, *not* statistical estimates of performance on real athlete populations. Full per-seed results and the worst/best values are in [`stress_runs.csv`](../demo_results/stress/stress_runs.csv) and [`stress_summary.csv`](../demo_results/stress/stress_summary.csv).

### Interpretation (including failures)

- The attractive single-seed result **did not reliably survive** different seeds in the unmodified generator. It must not be described as a general finding.
- Smoothing and historical averaging can help when artificial measurement noise is high, but under some conditions the simple persistence rule does better.
- Abrupt simulated shifts can make historical ML predictions lag behind the latest observation; this is an important failure mode.
- Only three seeds were evaluated. No hyperparameters were optimized against this stress suite, and the outcomes are conditional on the chosen synthetic designs.
- These tests do not establish whether any model predicts true athlete changes, fatigue, injury, or human performance in the field.

## Reproducibility and code integrity

```bash
python -m pip install -e '.[dev]'
pytest -q
athlete-insights demo --out outputs/demo
python -m athlete_insights.stress --out outputs/stress
```

`demo_results/` contains small, fully synthetic example artifacts for reviewing without installation. Running the two commands regenerates more detailed evidence (including CSV predictions and figures) outside the tracked output directory. We intentionally do **not** configure a GitHub Actions matrix for every push; verification is lightweight and can be run locally to conserve CI minutes.

## External data: not yet validated

An appropriate next step is to identify a genuinely longitudinal, legally usable dataset with stable athlete IDs, measurement units and repeated testing dates. Some openly accessible sports datasets contain just pre/post measurements or detailed recordings from one test day; they **cannot** be silently converted into 12+ weekly athlete observations. A future real-data experiment must document the source, license, unit conversion, missingness, and a new evaluation protocol before reporting results.

The application is not a diagnostic or coaching recommendation tool. No TMX customer information or real athlete records are included.