# Velocity edits through the frozen V-JEPA predictor

This experiment edits prefix-only layer-19 tokens, continues encoder blocks 20–24 and final LayerNorm, runs the pretrained frozen predictor, and measures future disk positions with a frozen C-trained readout.

The primary readout is shared across all four prediction horizons. It was trained on unedited predictor outputs. A separate readout trained on real future encoder tokens is a cross-check; its poor transfer to predictor outputs limits claims about natural counterfactual representations.

## Data and validation

- All 1,536 constant-velocity speed-dataset clips; original A=768, B=256, C=256, D=256 partitions.
- Only frames 0–7 enter the context encoder. Targets are tubelets (8,9), (10,11), (12,13), (14,15).
- Replacing all future pixels with random pixels leaves prefix activations and predicted tokens exactly unchanged.
- Bright-disk centroids provide image-coordinate labels. The MP4s decode to orange despite the supplied blue-disk description.
- The A-calibrated world-to-image mapping is approximately (32*x+128, -32*y+128); mean metadata-to-centroid discrepancy is 0.40 pixels.
- Position readouts fit C only; five-fold cross-validation groups all horizons of a video. No evaluator is trained on edits. A fits intervention probes; B selects their alpha and reliable tokens.
- Baseline gate fixed before scoring: B mean-across-horizon and farthest-horizon errors must both beat holding the last position.
- The existing D split has been used in earlier project experiments; this is a fixed follow-up, not a newly untouched benchmark.

## Unedited prediction baseline

| Readout / baseline | +0.0625 s | +0.1458 s | +0.2292 s | +0.3125 s |
|---|---:|---:|---:|---:|
| Shared readout of predictor outputs | 6.56 | 6.74 | 7.20 | 8.17 |
| Shared readout of real future tokens | 4.25 | 4.72 | 4.99 | 5.37 |
| Real-token readout transferred to predictions | 36.02 | 40.33 | 45.31 | 49.65 |
| Hold last observed position | 4.24 | 9.91 | 15.59 | 21.26 |
| Extrapolate measured prefix centroids | 0.52 | 0.60 | 0.67 | 0.78 |

Numbers are mean Euclidean position errors in pixels on D. The predictor baseline passes the predefined B gate. Simple measured-trajectory extrapolation is much more accurate; beating the static baseline does not establish superior physics prediction.

## Intervention

All 1024 prefix tokens passed B mean velocity-component R² ≥ 0.5 and mean position-component R² ≥ 0.8. Each token has a four-output ridge probe for velocity (vx,vy) and the last observed image position (px,py). Targets are standardized using A only.

For each selected token, the minimum-norm edit satisfies W_velocity Δh = Δv and W_position Δh = 0. Velocity shifts are ±0.25 and ±0.50 m/s along the original direction. The raw probe constraint residual is 1.10e-13. This algebraic constraint does not guarantee position preservation downstream.

The fixed evaluation subset contains 184 D clips with speeds in [0.75,3.5] m/s, keeping edited speeds inside the training range. Random controls preserve the same token position probes and match each token’s edit norm, with a fixed random direction per token. Zero edits reproduce the cached baseline.

## Measured effects on D

A physically calibrated velocity edit would produce Δp(t) = (t−t₀) Δv. Velocity slope gain is the slope of the measured position change, projected onto the requested image velocity change, divided by that requested change. A gain of 1 matches the physical effect; 0 means no slope change. A free intercept separates a position offset from a velocity-like slope change.

| Δspeed (m/s) | Velocity slope gain [95% CI] | Random slope gain [95% CI] | Paired gain difference [95% CI] | Final projected shift / expected (px) |
|---|---|---|---|---|
| -0.50 | 0.033 [0.032, 0.035] | 0.002 [0.001, 0.003] | 0.031 [0.029, 0.033] | 0.338 / 5.000 |
| -0.25 | 0.033 [0.032, 0.035] | 0.002 [0.001, 0.003] | 0.031 [0.029, 0.033] | 0.169 / 2.500 |
| +0.25 | 0.033 [0.032, 0.035] | 0.002 [0.001, 0.003] | 0.031 [0.029, 0.033] | 0.168 / 2.500 |
| +0.50 | 0.033 [0.032, 0.035] | 0.002 [0.001, 0.003] | 0.031 [0.029, 0.033] | 0.336 / 5.000 |

Intervals use 1,000 paired clip-bootstrap resamples and condition on the fitted models. They do not include training uncertainty or correction across the four intervention sizes.

| Δspeed (m/s) | Position-offset intercept norm (px) | Independently decoded last-context-tubelet shift (px) | Edit norm / raw activation norm |
|---|---:|---:|---:|
| -0.50 | 0.226 | 0.068 | 0.0029 |
| -0.25 | 0.113 | 0.034 | 0.0014 |
| +0.25 | 0.113 | 0.034 | 0.0014 |
| +0.50 | 0.225 | 0.069 | 0.0029 |

## Interpretation

The velocity edits produce a positive velocity-like slope effect relative to matched random controls at all four edit sizes. The table reports the effect size rather than treating a changed construction probe as success.

The causal conclusion is restricted to this frozen encoder–predictor–position-readout pipeline. The predictor and encoder were not trained or optimized to make the edits succeed. The position decoder is calibrated on natural predictor outputs, so off-distribution decoder responses remain possible. The clean-encoder position decoder does not transfer reliably to predictor outputs and cannot validate the counterfactual positions independently. Position drift after the remaining encoder blocks is measured above; preserving a layer-19 position probe alone does not exclude that confound.

Only a four-dimensional probe row space per token was edited. Distributed or redundant velocity information may remain outside it. A weak or null effect would therefore not prove that velocity is noncausal. The test varies speed along the original direction; direction rotation and acceleration edits have not been run.

## Artifacts

- [Baseline error curves](baseline.png) and [baseline trajectories](baseline_trajectories.png).
- [Velocity effect versus horizon](velocity_effect.png) and [edited trajectories](edited_trajectories.png).
- `baseline.json`, `edit_probe_summary.json`, `intervention_results.json`, and `paired_control_comparison.json` contain detailed metrics.
- `intervention_predictions.npz` contains every evaluated video, condition, position prediction, and edit norm.
- `protocol.json` records extraction; `baseline_protocol.json` refines the assay to a shared decoder before baseline fitting; `intervention_protocol.json` and `intervention_selection.json` freeze the interventions.
- `code/` and `code_sha256.json` preserve the final experiment implementation.
