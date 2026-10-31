# Conditional spline velocity steering

For +0.50 m/s, spline steering produces **3.84%** of the physically expected velocity response (95% clip-bootstrap CI -2.28–9.55%), versus **17.91%** for the frozen iterated linear method. The spline-matched random control gives **-1.64%**.

At this requested change, independent context-position drift is **2.156 px** for the spline and **0.367 px** for linear steering. Their mean edit/raw-activation norm ratios are **0.0601** and **0.0073**, respectively.

## Fixed experiment

The task is to change speed while keeping direction and observation-boundary position fixed. This reuses the prefix-only layer-19 activations (frames 0–7), remaining encoder blocks, pretrained predictor, and frozen C-trained position readouts from the velocity experiments. No future frame enters the encoder input. The same speed-restricted B/D examples receive ±0.25 and ±0.50 m/s changes. The baseline is the previously completed iterated linear method using all B-eligible probes (saturated cap 32); its saved per-example outcomes are reused and case identities are checked.

## Curve construction and selection

Each of the 1,024 tokens and 1,024 channels has its own regression coefficients. The shared input basis is a natural cubic speed spline multiplied by periodic sine/cosine direction features and quadratic boundary-position features (1, x, y, x², xy, y²). Coordinates use known speed/direction and measured frame-7 position. This is a new token-level conditional spline regression; the old pooled centroid spline weights are not reused. There is no output PCA truncation.

A fits all parameters and input scaling. B chooses among 8/12/16 speed anchors, one/two direction harmonics, and the fixed ridge grid, using protected-activation reconstruction MSE. The existing position-probe row spaces are projected out of the fitted outputs. An exact output Gram matrix permits selection over all token/channel outputs without approximating them.

Selected: **8 speed anchors, direction order 1, 144 input features, ridge alpha 1000**. B reconstruction MSE is 2.611676, versus 2.894750 for the A-mean representation (9.78% reduction). This reconstruction score includes position/direction structure and is not a velocity-specific accuracy score.

The intervention is `H′ = H + f(s + Δs, θ, p7) − f(s, θ, p7)`. It retains the original example residual and gives exactly zero edit when Δs = 0. Direction and boundary position are held fixed in both curve evaluations; target speeds remain within the fitted domain. The edit is projected again to remove float32 position-constraint residuals. There is no fitted gain multiplier or downstream choice of edit strength.

Random controls use the same seeded position-null directions as the preceding experiment, with each token’s norm matched to its spline edit. All choices are saved before downstream D evaluation.

## Downstream comparison

Velocity slope gain is the time slope of decoded position change, with a free intercept, projected onto the requested velocity change. One means the physical response is fully recovered; zero means no aligned velocity response. It does not measure the fraction of causal information captured. Intervals use 1,000 paired clip-bootstrap resamples, conditional on the fitted models.

| Requested Δspeed | Spline gain [95% CI] | Linear gain [95% CI] | Paired spline − linear gain [95% CI] | Spline random gain |
|---:|---|---|---|---:|
| -0.50 | 0.183 [0.132, 0.233] | 0.181 [0.175, 0.187] | 0.002 [-0.050, 0.054] | 0.0269 |
| -0.25 | 0.168 [0.105, 0.226] | 0.181 [0.174, 0.187] | -0.013 [-0.075, 0.045] | 0.0224 |
| +0.25 | 0.088 [0.024, 0.148] | 0.180 [0.173, 0.186] | -0.091 [-0.154, -0.033] | -0.0096 |
| +0.50 | 0.038 [-0.023, 0.096] | 0.179 [0.173, 0.185] | -0.141 [-0.201, -0.084] | -0.0164 |

| Δspeed | Method | Context drift (px) | Future-position intercept norm (px) | Perpendicular slope (px/s) | Relative edit norm |
|---:|---|---:|---:|---:|---:|
| -0.50 | spline | 2.185 | 3.547 | 3.902 | 0.06654 |
| -0.50 | linear | 0.371 | 0.721 | 0.407 | 0.00726 |
| -0.25 | spline | 1.328 | 1.845 | 1.908 | 0.03692 |
| -0.25 | linear | 0.185 | 0.360 | 0.204 | 0.00363 |
| +0.25 | spline | 1.308 | 1.936 | 1.744 | 0.03506 |
| +0.25 | linear | 0.184 | 0.359 | 0.205 | 0.00363 |
| +0.50 | spline | 2.156 | 3.412 | 3.168 | 0.06014 |
| +0.50 | linear | 0.367 | 0.716 | 0.411 | 0.00726 |

## Interpretation and limits

At −0.50 m/s the spline response is 18.33% versus 18.11% for linear steering. At +0.50 m/s the paired spline-minus-linear gain is -14.068 [-20.128, -8.410] percentage points. The +0.50 m/s spline edit has 8.29 times the linear edit norm and 5.88 times the context-position drift. These comparisons assess this particular conditional fit, not every possible spline construction.

Assess the aligned velocity response together with position drift, perpendicular response, and edit size. A higher response with substantially larger or less selective edits is a tradeoff. This comparison evaluates the two complete steering methods; it does not separately identify the contribution of curvature.

The spline is a regularized conditional mean fitted to sparse coverage of direction, speed, and position. Conditioning and preserving the construction position probes do not guarantee a natural counterfactual or preservation of all position information. Holding boundary position fixed while changing speed also changes the implied past trajectory. Reconstruction-based selection may favor nuisance structure over the most causally useful velocity variation.

D has already been inspected in prior experiments; this is an exploratory follow-up. The position evaluator is calibrated on unedited predictor outputs, and the independent teacher-token readout transferred poorly in the original experiment. Effects therefore concern the frozen encoder–predictor–readout pipeline. The experiment does not render an edited video. Confidence intervals omit fitting uncertainty and are not corrected for the sweep. One seeded random direction is used per token.

## Verification and artifacts

Unit tests verify exact agreement with independent scikit-learn ridge fits, Gram-based validation MSE, periodicity, no extrapolation, zero/reversed edits, and projection commuting with the fitted map. The complete run independently verifies selected-model reconstruction against direct predictions, position constraints, and zero-edit agreement with the previous predictor baseline.

- Direct B MSE: 2.6116763784; Gram B MSE: 2.6116763784.
- Float64 coefficient position-constraint residual: 8.154e-16.
- [Comparison plot](comparison.png), [future displacement](future_displacement.png).
- `protocol.json`, `selection.json`, `model.npz`, `fit_checks.json`, `evaluation_selection.json` preserve construction decisions.
- `effects.json` reports B/D metrics for every method and requested change; `conditions/` stores new per-video outputs.
- Full token coefficients: `/scratch/users/hmolin/jepa-interp/spline_velocity_v1/coefficients.npy`; checksum in `fit_checks.json`.
- `code/` and `code_sha256.json` preserve source files.
