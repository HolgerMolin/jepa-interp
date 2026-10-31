# Iterated centroid-spline residualization

The fixed validation rule stopped after **4 rounds** (validation_patience); the lowest B residual MSE selected prefix **1**. For +0.50 m/s, the one-round difference edit gives **5.39%** of the physical velocity response, and the final attempted prefix gives **5.35%**, versus **17.91%** for the frozen iterated linear method.

All spline conditions here use source-to-target differences added to the untouched representation. There is no initial replacement by the source-speed centroid. This isolates iteration from the replacement effect observed in the previous experiment.

After one subtraction, fresh linear speed-probe D R² falls to **-0.0125**, while the nonlinear RBF probe retains **0.9082**. After 4 rounds, RBF R² remains **0.9050**. Thus the residual transformation leaves substantial nonlinearly decodable speed information that these additional centroid fits do not remove.

## Fixed sequence construction

Each round fits 64-dimensional PCA to A residuals, computes speed-bin centroids, and interpolates them with the original natural cubic helper. B chooses among 8/12/16 anchors by reconstruction MSE in that round’s PCA coordinates. The fitted contribution is mapped back to the original 1,024-dimensional pooled feature space, then centered by its A average. Subtracting it preserves the A mean but is not an orthogonal projection. The next PCA is fitted anew to the resulting residuals.

The maximum is 16 rounds. Stop after three rounds without reducing full-feature B residual MSE by at least 0.1% of the original B MSE relative to the last meaningful best. The selected prefix minimizes B residual MSE among all attempted prefixes, including round zero. Predictor evaluation includes powers of two up to the attempted count, plus the final attempted and selected positive prefixes. Attempted prefixes that fail validation are explicitly exploratory diagnostics.

No predictor outcome selects a round count. Residual-probe scores are diagnostic and do not select the spline sequence. A/B fit and selection are isolated from C/D. D has been inspected historically, so this remains exploratory.

| Round | Anchors | Full-feature B residual MSE | Reduction from preceding round | Meaningful improvement |
|---:|---:|---:|---:|---|
| 0 | — | 0.0778268 | — | — |
| 1 | 8 | 0.0640183 | +0.0138085 | True |
| 2 | 8 | 0.0640434 | -0.0000251 | False |
| 3 | 8 | 0.0640846 | -0.0000412 | False |
| 4 | 8 | 0.0641030 | -0.0000184 | False |

## Steering

The edit is `H′ = H + Σ_j [f_j(s + Δs) − f_j(s)]`, applied identically to each prefix token. Each fitted curve is expressed in original activation coordinates. Zero requested change is an exact identity. The one-round edit is numerically verified against the previous original centroid curve’s difference. Every condition uses the same remaining encoder, pretrained predictor, C-trained position readouts, and 184 speed-restricted D clips as the preceding experiments.

The control rotates the cumulative difference with one fixed seeded orthogonal matrix in the full 1,024D channel space. It preserves the edit norm and the uniform token mapping. There is no additional position-null projection. The velocity metric is the free-intercept time slope of decoded position change projected onto the requested change. Intervals are 1,000 paired clip bootstraps, conditional on fitted models and without correction for the sweep.

Representative +0.50 m/s condition:

| Curves | Velocity gain [95% CI] | Paired gain above one round [95% CI] | Paired gain above random [95% CI] | Context drift (px) | Relative edit norm | Pooled speed gain |
|---:|---|---|---|---:|---:|---:|
| 1 | 0.054 [0.044, 0.064] | 0.000 [0.000, 0.000] | 0.071 [0.060, 0.081] | 0.298 | 0.01120 | 0.934 |
| 2 | 0.054 [0.043, 0.064] | -0.000 [-0.002, 0.001] | 0.070 [0.060, 0.081] | 0.329 | 0.01180 | 0.938 |
| 4 | 0.054 [0.043, 0.064] | -0.000 [-0.003, 0.002] | 0.070 [0.059, 0.081] | 0.339 | 0.01199 | 0.939 |

| Δspeed | One-round gain | Final attempted gain | Paired final-minus-one gain [95% CI] | Iterated linear gain |
|---:|---:|---:|---|---:|
| -0.50 | 0.0663 | 0.0696 | 0.003 [-0.001, 0.007] | 0.1811 |
| -0.25 | 0.0656 | 0.0688 | 0.003 [0.000, 0.006] | 0.1805 |
| +0.25 | 0.0588 | 0.0597 | 0.001 [-0.002, 0.003] | 0.1796 |
| +0.50 | 0.0539 | 0.0535 | -0.000 [-0.003, 0.002] | 0.1791 |

## Residual speed information

Fresh linear and RBF-kernel ridge probes read the pooled residuals. A fits them, B chooses ridge alpha and RBF width, and all settings are saved before D scoring. RBF widths are fixed multiples of the median A squared distance. A-centered kernels and targets include an unpenalized intercept. The audit uses all 256 D clips, whereas interventions retain the established 184-clip speed-restricted subset.

**Known speed is used to construct each residual**, including on D. This is supervised residualization, not a label-free deployable erasure method. Decodability measures what these probe families can recover after that transformation; it is not a proof that all information has or has not been causally removed.

| Removed rounds | Linear D R² | Linear D MAE (m/s) | RBF D R² | RBF D MAE (m/s) |
|---:|---:|---:|---:|---:|
| 0 | 0.9853 | 0.1050 | 0.9860 | 0.1034 |
| 1 | -0.0125 | 0.9534 | 0.9082 | 0.2571 |
| 2 | -0.0007 | 0.9529 | 0.9060 | 0.2624 |
| 4 | 0.0077 | 0.9499 | 0.9050 | 0.2646 |

## Interpretation limits

Later curves need not correspond to independent copies of speed. They can recover variation previously outside PCA, correct earlier approximation errors, or fit noise. Validation stopping and the residual audit help distinguish useful progress from a training-only reduction, but cannot establish a nonlinear nullspace.

This experiment still operates on pooled features and a uniform token lift. It cannot directly coordinate token-specific changes discarded by pooling. Positive predictor effects must be interpreted alongside position drift and perpendicular responses. The frozen position readout remains the measurement bottleneck; teacher-token transfer was weak in the original experiment.

## Checks and artifacts

Tests cover telescoping residual subtraction, preservation of the A mean, zero/reversed differences, validation-prefix selection, and centered kernel prediction against an independent implementation. The run verifies first-round equality with the original centroid curve and zero-edit predictor equivalence.

- [Steering sweep](steering.png), [validation and residual audit](residuals.png).
- `protocol.json`, `selection.json`, `models/`, `fit_checks.json`, and `evaluation_selection.json` freeze construction choices.
- `effects.json` and `conditions/` retain B/D effects and per-video outputs.
- `audit_selection.json`, `residual_audit.json`, and `audit_predictions.npz` retain fresh-probe diagnostics.
- `code/` and `code_sha256.json` preserve implementation sources.
