# Original centroid-spline recipe: prefix-token predictor test

This run restores the original empirical-centroid interpolation and PCA-coordinate replacement, using the notebook helpers unchanged. The necessary adaptations are prefix-only features and a uniform token lift of the pooled edit. It does not use the previous conditional regression spline.

On the pooled path diagnostic, spline target MAE is **0.1072 m/s** versus **0.1663 m/s** for endpoint chords. For a +0.50 m/s change through the predictor, the source-to-target centroid response is **5.00%** of the expected physical velocity response (95% CI 4.02–5.93%). The previously completed iterated linear edit gives **17.91%** from the original representation.

Replacing PCA coordinates at the **original speed alone** moves the independent context-position readout by **2.262 px**, and future last-horizon position by **7.830 px**. It also shifts the decoded velocity along the original direction by approximately **-0.184 m/s**. This non-identity source replacement is why both reference points are reported below. For positive requested changes, the total response from the untouched representation is actually negative; the positive source-to-target effect does not undo the slowdown introduced by replacement.

## Recipe and token mapping

1. Mean-pool the 1,024 layer-19 tokens from frames 0–7 only. Future frames never enter encoder attention.
2. Fit 64-dimensional PCA on A. For each of 8/12/16 uniformly spaced speed anchors, average A coordinates in nearest-anchor bins.
3. Interpolate the centroids exactly with a natural cubic spline using `utils.fit_centroid_spline`. Select anchor count by B reconstruction MSE at known speed.
4. Replace the source pooled PCA coordinates with the target curve point, retaining the pooled orthogonal complement using the original `utils.replace_pca_coordinates` operation.
5. Add that pooled change identically to every token. This exactly implements the pooled edit, preserves each token’s deviation from the mean, and minimizes total squared token edit norm among lifts with the same mean change.
6. Continue the frozen encoder and predictor; apply unchanged C-trained position readouts.

Selected **8 anchors**, B reconstruction MSE 0.983408; the 64 PCs explain 97.36% of A pooled variance. Bin counts: [60, 108, 108, 108, 108, 108, 108, 60].

Unlike the prior conditional experiment, there is no ridge shrinkage of the spline, no direction/position regression, no extra position-null projection, and no replacement of the curve rule by a residual-difference edit. The curve is parameterized by speed only, as in the old speed experiment. Direction and position preservation are evaluated rather than assumed.

## Source-speed replacement is a required control

Let the original pooled PCA coordinates be z. The actual target edit is `(f(s + Δs) − z) @ PCA_components`, broadcast to all tokens. At Δs = 0, `(f(s) − z)` is generally nonzero. Subtracting source-replacement outcomes measures movement along the centroid curve after that replacement. It does not equal running a residual-difference edit through the nonlinear predictor from the untouched representation.

The untouched zero-edit pass reproduces the previous baseline. Random controls apply one fixed seeded orthogonal rotation in the same 64D PCA subspace to both source and target edits. This matches individual edit norms and source-to-target difference norms while retaining the identical uniform token-lift structure. Controls are new for this mapping and differ from the earlier per-token position-preserving random directions.

## Downstream motion

All rows use the same 184 D clips and requested velocity changes along each clip’s original direction. Gain is the time slope of decoded position change with a free intercept, projected onto the requested velocity change. One is the expected physical response; zero is no aligned velocity-like response. Confidence intervals are 1,000 paired clip bootstraps conditional on the fitted models.

| Δspeed | Centroid vs original gain [95% CI] | Centroid vs source replacement gain [95% CI] | Rotated-control source-to-target gain | Iterated linear gain |
|---:|---|---|---:|---:|
| -0.50 | 0.429 [0.393, 0.465] | 0.062 [0.046, 0.077] | 0.0010 | 0.1811 |
| -0.25 | 0.795 [0.724, 0.863] | 0.061 [0.047, 0.074] | 0.0033 | 0.1805 |
| +0.25 | -0.680 [-0.750, -0.603] | 0.055 [0.043, 0.066] | 0.0067 | 0.1796 |
| +0.50 | -0.317 [-0.354, -0.278] | 0.050 [0.040, 0.059] | 0.0088 | 0.1791 |

| Δspeed | Context drift vs original (px) | Context drift vs source replacement (px) | Full relative edit norm | Source-target relative edit norm | Pooled speed-change gain |
|---:|---:|---:|---:|---:|---:|
| -0.50 | 2.367 | 0.335 | 0.05009 | 0.01420 | 0.967 |
| -0.25 | 2.319 | 0.176 | 0.04808 | 0.00684 | 0.981 |
| +0.25 | 2.203 | 0.165 | 0.04747 | 0.00610 | 0.965 |
| +0.50 | 2.155 | 0.295 | 0.04817 | 0.01120 | 0.934 |

## Independent pooled probe check

A new scalar speed ridge is fitted only on C, with alpha selected by five-fold CV within C. It is used because the old evaluator was trained on full-clip pooled features. It does not select the spline or tune downstream effects. The path diagnostic restores the old speed targets 0.25, 2.125, and 4 m/s, all three unordered pairs, and 21 waypoints. Spline and chord share endpoints. Endpoint chords are evaluated only for this reproduction of the old pooled test; no tangent baseline is added.

On unedited prefix-pooled D features, the independent speed readout has MAE **0.1362 m/s**, R² **0.9763**. The path result above averages interior-waypoint errors across the three paths.

## Limits of the conclusion

This is a faithful refit of the original pooled recipe on prefix-only features with one explicit minimal token mapping. It does not establish how other token mappings would behave. Preserving token deviations from the mean does not guarantee preserving the model’s position or direction representation. A good pooled speed score can coexist with poor direction-preserving downstream motion, and source replacement may substantially change the scene representation.

The linear reference preserves construction position probes; this faithful centroid replacement does not impose that additional constraint. Accordingly, compare downstream response together with drift and edit magnitude. D was historically inspected, so this remains exploratory. The physical interpretation still depends on the frozen position decoder, whose independent teacher-token transfer was weak in the first experiment. No new clips or edited videos are generated.

## Validation and artifacts

Tests verify exact agreement with the old replacement helper, empirical centroids and interpolation, preservation of token deviations, minimum-norm lifting, and norm-matched rotated controls. The run checks the achieved PCA coordinates and untouched predictor baseline.

- [Downstream comparison](comparison.png), [pooled path check](pooled_path_check.png).
- `protocol.json`, `selection.json`, `model.npz`, and `evaluation_selection.json` freeze the recipe and evaluated cases.
- `pooled_path_diagnostic.json` and `pooled_speed_readout.npz` preserve the independent pooled check.
- `effects.json` and `conditions/*.npz` retain both reference comparisons and every per-video output.
- `code/` and `code_sha256.json` preserve the implementation.
