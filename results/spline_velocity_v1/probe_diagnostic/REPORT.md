# Velocity-probe diagnostic of spline edits

This diagnostic reads the completed spline edits with the existing, frozen, B-eligible iterated velocity probes. It fits no new model and does not change spline settings or predictor outcomes. The primary quantity is `probe(H + edit) − probe(H)`, projected onto the requested velocity change and divided by its magnitude. A gain of 1 means the probe reports the requested change; 0 means no aligned change; negative means the opposite direction.

There are **4274 eligible two-output probes over 1,024 tokens**, with 184 D clips. Each token gets equal weight, with its probes averaged first. First and subsequent probes are reported separately. Intervals bootstrap clips only; token/probe observations are not treated as independent samples.

| Requested Δspeed | First-probe gain [95% CI] | Subsequent-probe gain [95% CI] | All-probe gain [95% CI] | Correct-sign fraction | Within 25% vector-error fraction |
|---:|---|---|---|---:|---:|
| -0.50 | 0.336 [0.292, 0.375] | 0.104 [0.054, 0.151] | 0.173 [0.126, 0.217] | 0.680 | 0.035 |
| -0.25 | 0.334 [0.285, 0.379] | 0.083 [0.029, 0.132] | 0.157 [0.105, 0.204] | 0.652 | 0.036 |
| +0.25 | 0.318 [0.271, 0.365] | 0.048 [0.001, 0.090] | 0.128 [0.081, 0.171] | 0.627 | 0.024 |
| +0.50 | 0.290 [0.249, 0.331] | 0.020 [-0.019, 0.056] | 0.100 [0.061, 0.135] | 0.616 | 0.015 |

## Consistency and off-target changes

| Δspeed | Mean relative vector error | Mean relative perpendicular change | Mean within-token probe-gain SD |
|---:|---:|---:|---:|
| -0.50 | 0.894 | 0.208 | 0.146 |
| -0.25 | 0.923 | 0.228 | 0.157 |
| +0.25 | 0.946 | 0.226 | 0.157 |
| +0.50 | 0.953 | 0.207 | 0.147 |

Relative vector error is `||decoded Δv − requested Δv|| / ||requested Δv||`; perpendicular change is normalized by the same requested magnitude. Within-token SD is computed across eligible probes, then averaged over tokens with at least two probes and over clips. It measures disagreement of redundant readouts, not uncertainty in the mean. The per-order plot changes token membership as order increases; it is descriptive rather than a matched-token depth comparison.

## Checks and interpretation

The saved effective probe coefficients already include the sequential null-space projections, so they can be applied directly to full layer-19 tokens after subtracting the saved training mean. Predictions include the saved physical-unit intercept; it cancels in before/after differences. The reconstructed edit uses the frozen spline coefficients, identical position projection, and float32 token addition.

- Linear all-eligible operator response error versus the exact requested change: 3.775e-15.
- Direct before/after prediction versus coefficient-times-actual-edit error: 2.609e-14.
- Reconstructed spline edit-norm maximum relative difference versus the completed experiment: 1.124e-07.

The linear method satisfies these readouts by construction; its gain of 1 is a numerical check, not independent evidence of causality. Spline results here diagnose agreement with those same velocity representations. Failure to shift them does not exclude other velocity representations, and agreement would not guarantee natural edits. D has already been inspected; no settings are selected from this diagnostic.

- [Probe-response plot](probe_response.png).
- `responses.npz`: baseline predictions and every eligible probe’s per-clip velocity change.
- `summary.json`: B/D aggregates, per-order results, unedited R² diagnostics, and numerical checks.
- `protocol.json`: frozen inputs, selected probes, hashes, and aggregation rules.
