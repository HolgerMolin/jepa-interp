# Iterated token velocity interventions

For the +0.50 m/s intervention, using all validation-eligible probes changes the velocity-like response from **3.45%** of the physical expectation with one probe to **17.91%**. Erasing all discovered velocity directions gives **26.85 px** farthest-horizon position error, versus **7.59 px** for rank-matched random erasure. However, full erasure also shifts the independent context-position readout by **14.85 px**, so that damage cannot be attributed selectively to velocity. The joint-token audit still recovers substantial velocity information after erasure.

This follow-up fits separate sequences of velocity probes at each of 1,024 prefix-only layer-19 tokens. It tests coordinated steering and erasure through the frozen remaining encoder, pretrained predictor, and unchanged position evaluators from the first experiment.

## Frozen protocol

- Data, prefix, splits, checkpoint, and C-trained position evaluators are unchanged from the first experiment.
- Each token’s existing position-probe row space is protected. All velocity readouts are constrained to its orthogonal complement.
- Fit each velocity probe on A; choose ridge alpha on B using mean standardized component MAE. Remove its effective row space and refit.
- A probe qualifies for steering only if BOTH vx and vy have B R² ≥ 0.5 and both readout directions are independent.
- Stop separately for each token after three consecutive probes with BOTH component R² ≤ 0.05 and BOTH MAE ≥ 90% of the training-mean baseline on B; also stop at numerical rank exhaustion or 256 removed velocity dimensions. Fit a final residual probe at the cap.
- Probe-count sweeps specify a maximum; each token uses only its available eligible probes. Identical saturated caps are not rerun.
- Random steering controls preserve position rows and match per-token edit norms. Random erasure controls preserve position rows and match per-token removed rank.
- Downstream evaluation uses the same speed-restricted B/D clips and changes of ±0.25 and ±0.50 m/s along the original velocity direction.
- D was inspected in earlier experiments. This is a fixed exploratory follow-up; no cap is selected using downstream D performance.

The solver uses float64 equality-constrained ridge updates, periodically rebuilds the inverse, and verifies results against independent direct dual solves. Tests compare the full fit–project–refit sequence with explicit scikit-learn fits, verify C/D isolation, verify orthogonality and erasure, and compare the simultaneous edit with a direct stacked pseudoinverse.

## Different tokens require different probe counts

Eligible-probe count quantiles (min, 10%, 25%, median, 75%, 90%, max): **[1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 23.0]**.

Terminal removed-rank quantiles in the same order: **[30.0, 44.0, 54.0, 72.0, 102.0, 146.0, 256.0]**.

Stopping reasons: `{"dimension_cap": 3, "near_baseline": 1021}`. A cap stop does not establish exhaustion.

| Maximum eligible probes | Actual probes: min / median / max | Mean edited dimensions per token |
|---:|---|---:|
| 1 | 1 / 1 / 1 | 2.00 |
| 2 | 1 / 2 / 2 | 3.98 |
| 4 | 1 / 3 / 4 | 6.24 |
| 8 | 1 / 3 / 8 | 7.67 |
| 16 | 1 / 3 / 16 | 8.31 |
| 32 | 1 / 3 / 23 | 8.35 |

## Coordinated steering

The edit shifts every included effective velocity readout by the same Δv while preserving the protected position rows. Because the sequential readout row spaces are orthogonal, summing their minimum-norm edits equals the stacked constrained solution.

The primary measure below is velocity slope gain: a free-intercept regression of decoded position change against future time, projected onto the requested velocity change. A gain of 1 is physically calibrated; 0 is no velocity-like response. Intervals are 1,000 paired clip-bootstrap resamples, conditional on the fitted models, without correction for the full sweep.

Representative +0.50 m/s condition (all four changes and both B/D are in `effects.json`):

| Maximum probes | Velocity slope gain [95% CI] | Paired gain above random [95% CI] | Final projected shift (px; expected 5) | Context position drift (px) | Relative edit norm |
|---:|---|---|---:|---:|---:|
| 1 | 0.035 [0.033, 0.036] | 0.034 [0.033, 0.036] | 0.491 | 0.034 | 0.0028 |
| 2 | 0.070 [0.067, 0.074] | 0.070 [0.066, 0.074] | 0.881 | 0.109 | 0.0044 |
| 4 | 0.124 [0.119, 0.129] | 0.123 [0.118, 0.129] | 1.275 | 0.236 | 0.0060 |
| 8 | 0.163 [0.157, 0.169] | 0.162 [0.155, 0.168] | 1.500 | 0.323 | 0.0068 |
| 16 | 0.178 [0.172, 0.184] | 0.177 [0.170, 0.183] | 1.550 | 0.362 | 0.0072 |
| 32 | 0.179 [0.173, 0.185] | 0.178 [0.171, 0.184] | 1.556 | 0.367 | 0.0073 |

The matched one-probe condition here uses the same protected-position fit as the multi-probe conditions. The first experiment used a joint four-output fit, so its 3.3% slope gain is historical context rather than an exactly matched baseline.

## Erasure

Erase the A-centered coordinates in each eligible velocity subspace, and separately all discovered velocity directions including probes below the steering threshold. Positive paired error difference means targeted erasure harms prediction more than random erasure.

| Removed subspace | Farthest-horizon error (px) | Random erasure error (px) | Paired extra error [95% CI] | Context position drift (px) |
|---|---:|---:|---|---:|
| Up to 1 eligible probes | 7.374 | 7.261 | 0.114 [-0.115, 0.335] | 0.148 |
| Up to 2 eligible probes | 7.564 | 7.278 | 0.286 [-0.097, 0.634] | 0.444 |
| Up to 4 eligible probes | 8.112 | 7.285 | 0.828 [0.317, 1.303] | 0.925 |
| Up to 8 eligible probes | 8.541 | 7.298 | 1.243 [0.624, 1.790] | 1.243 |
| Up to 16 eligible probes | 8.638 | 7.307 | 1.331 [0.700, 1.901] | 1.373 |
| Up to 32 eligible probes | 8.649 | 7.307 | 1.343 [0.710, 1.912] | 1.385 |
| All discovered | 26.851 | 7.593 | 19.258 [17.640, 21.052] | 14.855 |

## Joint-token residual audit

A fresh exact dual ridge reads all 1,024×1,024 token features jointly. It fits A, selects alpha on B, and scores D. This checks information recoverable by combining tokens even when their individual residual probes are weak.

| Representation | D R² vx | D R² vy | D MAE vx (m/s) | D MAE vy (m/s) |
|---|---:|---:|---:|---:|
| original | 0.9324 | 0.9336 | 0.3408 | 0.3282 |
| velocity erased | 0.8913 | 0.8641 | 0.4368 | 0.4763 |
| random erased | 0.9323 | 0.9334 | 0.3406 | 0.3283 |
| position and velocity erased | 0.8722 | 0.8308 | 0.4753 | 0.5331 |

The position-and-velocity-erased representation is an audit diagnostic only; actual downstream erasure preserves the position directions. Residual joint decodability prevents interpreting per-token stopping as complete removal of velocity information.

## Interpretation and limitations

This experiment measures interventions in the frozen encoder–predictor–readout pipeline. It does not generate a changed video. Stronger velocity-like responses than the matched one-probe baseline support redundant information as one explanation of the initial weak edit. A weak response, or erasure without selective harm, would not establish that velocity is noncausal.

Position preservation is guaranteed only for the construction readouts. Independent context-position drift and future-position offsets must be considered alongside slope gain. Larger subspace edits can create unnatural representations or disrupt unrelated properties.

The frozen position evaluator is calibrated on unedited predictor outputs. The independently trained real-future-token position evaluator transferred poorly in the first experiment and remains an unreliable physical cross-check. Bootstrap intervals omit uncertainty in training the probes/evaluators. One seeded random basis is used per token; the controls do not average over multiple random bases.

## Artifacts

- [Steering sweep](steering_sweep.png), [erasure sweep](erasure_sweep.png), [token probe maps](token_probe_maps.png), [joint audit](joint_audit.png).
- `protocol.json` and `evaluation_selection.json` freeze the methods and evaluated cases.
- `sequence_summary.json`, `token_counts.npz`, and `probe_history.npz` describe each sequence.
- `effects.json` and `conditions/*.npz` contain all effect metrics and per-video predictions.
- `joint_audit.json`, `joint_audit_selection.json`, and `joint_audit_predictions.npz` contain the joint residual audit.
- Large sequence weights and bases remain in the scratch directory recorded in `protocol.json`.
- `code/` and `code_sha256.json` preserve this implementation.
