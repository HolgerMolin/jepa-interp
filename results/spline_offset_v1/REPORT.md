# Offset-preserving pooled PCA spline steering

The offset-preserving edit increases independent-readout target error for all three variables. It uses smaller edits and is exactly the identity when the target equals the known source label. These are pooled readout results, not predictor-motion results.

## Matched comparison

| Method | Direction (°) | Speed (m/s) | Acceleration (m/s²) |
|---|---:|---:|---:|
| Spline: replace PCA coordinates | 2.636 | 0.1783 | 0.3843 |
| Spline: preserve the offset | 5.107 | 0.1907 | 0.4247 |
| Linear: 16 probes | 3.733 | 0.0782 | 0.2244 |

Lower MAE is better. The linear result is the same fixed 16-probe setting shown previously, not selected on these results.

## Paired change in MAE

Positive values mean the offset-preserving method has higher error. Each of 5,000 bootstrap resamples resamples clips while retaining all targets for a clip; intervals condition on the frozen fits.

| Variable | Offset minus replacement | 95% paired bootstrap interval |
|---|---:|---:|
| Direction | +2.47064 | [+2.12209, +2.83047] |
| Speed | +0.01234 | [+0.00765, +0.01710] |
| Acceleration | +0.04037 | [+0.02351, +0.05803] |

All three intervals lie above zero. These are exploratory D-set follow-ups, not fresh confirmatory evaluation.

## What was held fixed

- Original block-19 full-clip pooled features, A-fitted 64D PCA, and A-fitted/B-selected centroid splines.
- Original C-trained evaluator weights and intercepts; no retraining, recalibration, or selection on D.
- Original fixed target values, D clips (292 direction; 256 speed; 256 acceleration), and equal weighting of targets and clips.
- Periodic direction spline; natural cubic scalar splines. All raw clips remain untouched.

With PCA components stored as rows C, coordinates z = (h − μ) Cᵀ, and spline f:

- Replacement: h′ = h + [f(target) − z] C.
- Offset preservation: h′ = h + [f(target) − f(source)] C.

The new method uses the known source label from clip metadata. This is an additional input relative to replacement; source labels are not inferred from a probe. No new clips or encoder/predictor runs were needed. Computation ran on oat-02 within allocation 1887909, using cached features and CPU operations.

## Why the readout error changes

Let r = z − f(source). The offset-preserving activation equals the replacement activation plus r C. With a fixed affine evaluator, its prediction therefore equals the old prediction plus r C Wᵀ. The full-space calculation agrees with this identity numerically.

Replacement removes the clip-specific residual contribution to the evaluator; preserving that contribution increases target error here. This explains why a more conservative edit can score worse on the absolute-target readout metric. It does not establish that the residual is purely nuisance, that replacement gives more natural activations, or that replacement is more causally effective through the predictor.

## Edit size and identity

| Variable | Replacement relative norm | Offset-preserving relative norm |
|---|---:|---:|
| Direction | 1.139 | 0.769 |
| Speed | 1.181 | 0.697 |
| Acceleration | 1.171 | 0.697 |

Norms are averaged across fixed targets and clips, relative to each source activation’s distance from the PCA training mean. These are not norm-matched interventions.

Requesting the source label gives exactly zero change under offset preservation. Replacement instead moves the point onto the source centroid, with mean full-space norms 8.116, 8.613, and 7.393 for direction, speed, and acceleration.

## Secondary interpolation check

The existing appendix compares spline versus straight-chord paths. Repeating those paths with the same preserved source residual for both paths gives:

| Variable | Offset-preserving spline interior MAE | Offset-preserving chord interior MAE |
|---|---:|---:|
| Direction | 5.1429 | 14.4894 |
| Speed | 0.1076 | 0.1460 |
| Acceleration | 0.3003 | 0.3524 |

All original endpoint pairs and 21 waypoints are retained. Endpoints remain matched. The chord is not the multi-probe linear baseline. Spline interpolation still beats chords, while both have higher error than their replacement versions.

## Verification and artifacts

All saved replacement predictions and fixed-target metrics were reproduced; path-prediction discrepancies were below 1.1e−13. Identity, preserved PCA offset, unchanged orthogonal complement, direction periodicity, and direct-versus-projected readout checks passed. No near-zero direction readout vectors occurred. See `checks.json`.

The protocol was saved before evaluation. `scores.json` contains aggregate/per-target scores and paired intervals; `evaluation_*.npz` stores IDs, predictions, errors, norms, and paths. `provenance.json` hashes all inputs and the code. Historical outputs in `results/spline/` and `results/steering/` were not modified. An incomplete initial run with a scalar-array shape error is archived separately in `results/spline_offset_v1_incomplete_scalar_shape/`; it is not used for any result.

Reproduce on a compute node with the project interpreter: `python evaluate_pooled_spline_offset.py --output results/spline_offset_new_run`. The destination must not exist.

## Presentation decision

The presentation uses only the offset-preserving spline method, alongside the fixed linear baseline. The replacement experiment and paired comparisons remain here as a historical record. All pooled spline values shown in the deck are 5.107° / 0.1907 m/s / 0.4247 m/s². The predictor result remains 5.39% because that experiment already used offset preservation. The methods diagram shows only the centroid-difference edit. No historical measurements have been deleted or relabeled.

## Normalized readout metric in the presentation

The deck reports readout error reduction: `100 × (1 − mean edited target MAE / mean unedited target MAE)`. This uses a shared unedited baseline for each comparison and a ratio of means. It is not R² or predictor response gain. Zero means no aggregate improvement; 100% means every readout reaches its requested target; negative values are possible. Raw MAEs remain above for auditability.

| Method | Direction | Speed | Acceleration |
|---|---:|---:|---:|
| Spline | 94.3% | 87.8% | 89.6% |
| Linear (16 probes) | 95.9% | 95.0% | 94.5% |

The appendix normalizes by no-edit errors for its own interior waypoint targets. Definitions and exact numbers are stored in `readout_error_reduction.json`. No model fitting or intervention was rerun for this metric change.
