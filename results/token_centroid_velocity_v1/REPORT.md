# Per-token offset-preserving centroid spline steering

Each prefix token has its own A-fitted PCA64 and speed-centroid spline. B independently selects 8/12/16 knots per token. Edits add the source-to-target curve difference, preserving the original residual. No C/D examples fit or select curves.

The primary protected variant projects the selected curve off the same position-readout directions as the linear method. It preserves the residual relative to this projected curve. The raw curve is a fixed secondary comparison. Controls match per-token edit norms and position protection; neither downstream performance nor D selects a variant.

## D-set results

| Δspeed (m/s) | Method | Velocity gain % [95% CI] | Context drift (px) | Relative edit norm |
|---:|---|---:|---:|---:|
| -0.50 | raw | 12.52 [9.04, 16.34] | 1.569 | 0.04584 |
| -0.50 | raw_random | 1.36 [0.11, 2.69] | 0.255 | 0.04584 |
| -0.50 | protected | 12.57 [9.02, 16.30] | 1.473 | 0.04581 |
| -0.50 | protected_random | 1.33 [0.09, 2.66] | 0.255 | 0.04581 |
| -0.50 | linear_k001 | 3.46 [3.34, 3.59] | 0.035 | 0.00277 |
| -0.50 | linear_k032 | 18.11 [17.49, 18.74] | 0.371 | 0.00726 |
| -0.25 | raw | 13.70 [9.27, 18.12] | 0.988 | 0.02667 |
| -0.25 | raw_random | 0.86 [-0.74, 2.54] | 0.148 | 0.02667 |
| -0.25 | protected | 13.76 [9.41, 18.16] | 0.922 | 0.02665 |
| -0.25 | protected_random | 0.82 [-0.75, 2.49] | 0.148 | 0.02665 |
| -0.25 | linear_k001 | 3.46 [3.34, 3.58] | 0.017 | 0.00138 |
| -0.25 | linear_k032 | 18.05 [17.44, 18.68] | 0.185 | 0.00363 |
| +0.25 | raw | 5.64 [1.44, 9.81] | 1.034 | 0.02592 |
| +0.25 | raw_random | -0.77 [-2.47, 0.93] | 0.148 | 0.02592 |
| +0.25 | protected | 5.66 [1.55, 9.73] | 0.963 | 0.02590 |
| +0.25 | protected_random | -0.79 [-2.49, 0.92] | 0.148 | 0.02590 |
| +0.25 | linear_k001 | 3.45 [3.33, 3.58] | 0.017 | 0.00138 |
| +0.25 | linear_k032 | 17.96 [17.34, 18.57] | 0.184 | 0.00363 |
| +0.50 | raw | 4.19 [0.46, 7.59] | 1.779 | 0.04269 |
| +0.50 | raw_random | -0.60 [-1.88, 0.66] | 0.253 | 0.04269 |
| +0.50 | protected | 4.20 [0.61, 7.49] | 1.652 | 0.04266 |
| +0.50 | protected_random | -0.63 [-1.91, 0.64] | 0.252 | 0.04266 |
| +0.50 | linear_k001 | 3.45 [3.33, 3.57] | 0.034 | 0.00277 |
| +0.50 | linear_k032 | 17.91 [17.30, 18.52] | 0.367 | 0.00726 |

## Paired primary comparisons

| Δspeed (m/s) | Protected spline minus reference | Gain difference (percentage points) [95% CI] |
|---:|---|---:|
| -0.50 | protected_random | 11.24 [7.61, 14.96] |
| -0.50 | linear_k001 | 9.11 [5.54, 12.87] |
| -0.50 | linear_k032 | -5.53 [-9.07, -1.82] |
| -0.25 | protected_random | 12.94 [8.29, 17.71] |
| -0.25 | linear_k001 | 10.30 [5.95, 14.69] |
| -0.25 | linear_k032 | -4.29 [-8.81, 0.13] |
| +0.25 | protected_random | 6.45 [1.88, 10.93] |
| +0.25 | linear_k001 | 2.20 [-1.87, 6.22] |
| +0.25 | linear_k032 | -12.30 [-16.31, -8.32] |
| +0.50 | protected_random | 4.83 [0.85, 8.34] |
| +0.50 | linear_k001 | 0.75 [-2.85, 4.01] |
| +0.50 | linear_k032 | -13.72 [-17.39, -10.34] |

## Interpretation limits

This matches token specificity, clips, targets, frozen predictor/readouts, and (for the primary variant) construction-position constraints. It does not isolate function class: one speed-only centroid curve differs from iterated direction-aware velocity probes, and edit norms are not matched between methods. The one-probe linear reference is also reported. Source speeds are oracle labels.

Residual preservation does not guarantee preservation of every scene property or membership in the natural activation distribution. Position protection applies to construction readouts; independent context drift and perpendicular response are recorded. The frozen position readout was trained on unedited predictor outputs; teacher-token transfer remains weak. No edited videos are generated.

D was previously inspected: this is exploratory. Intervals use 1,000 paired clip bootstraps conditional on frozen fits; they omit training uncertainty and multiple-comparison correction. Random controls use one seeded direction per token.

The protocol, per-token selection records, model/source hashes, predictions, B/D metrics and paired comparisons are saved alongside this report.
