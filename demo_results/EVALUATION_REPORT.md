# Athlete Performance Intelligence — Synthetic Demonstration

**Dataset:** 100% synthetic; 120 fictional athletes observed across 32 weekly sessions. These findings establish the reproducibility of the experiment only, not real-world prediction accuracy or injury detection.

## Forecast error on chronological holdout

| Test | Selected model | Selected MAE | Previous-value baseline MAE | Improvement |
| --- | --- | ---: | ---: | ---: |
| Countermovement jump | Histogram gradient boosting | 0.7466 cm | 0.7998 cm | 6.65% |
| Grip strength | Histogram gradient boosting | 0.8670 kg | 0.9717 kg | 10.78% |

Model selection used validation dates only. Earlier training observations were used to learn forecasting patterns; predictions on later test dates were evaluated separately. The full per-model results are in `forecast_comparison.csv`.

## Uncertainty and generalization

Athlete-cluster bootstrap 95% intervals for the model's reduction in MAE relative to the previous-value baseline:

- Countermovement jump: **0.35%–12.19%**
- Grip strength: **6.13%–14.80%**

On a separate 24-athlete held-out test set, validation-selected Ridge regression improved MAE over persistence by **5.68%** for jump height and **5.84%** for grip strength. That test holds athlete identities out of fitting, but still uses each test athlete's own preceding observations for predictions. It is a different test from chronological holdout.

## Unusual-drop flags

Historical-median/MAD detection flagged **59 jump** and **48 grip** readings in 7,680 synthetic observations. Against the deliberately injected artificial drop events, precision was **75.7%** and recall **97.6%** (81 true positives, 26 false positives, two false negatives). These numbers do **not** validate injury or fatigue prediction.

## Method and limitations

- All forecast features are constructed only from preceding observations; synthetic event labels are excluded.
- Selected forecasting algorithms are chosen on validation data, not final test dates.
- The generator intentionally simulates smooth, partly predictable patterns, so apparent gains do not establish generalization to actual athlete populations.
- Complete predictions, generated datasets, flags and illustrative plots can be reproduced by running `athlete-insights demo --out outputs/demo`.
- Authorized real data and independent validation are required before practical adoption. This prototype makes no clinical claims.
