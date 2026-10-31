# V-JEPA physics experiments

Start with the **[presentation](presentation/vjepa_physics_draft.pdf)** for the
summary or **[executed notebook](experiments.ipynb)** for the walkthrough.
Both include saved outputs.

## Navigate

The notebook follows the presentation:

| Section | Topic | Saved results |
| --- | --- | --- |
| 2 | Layer-wise probing | [Plots](results/layerwise_probes.pdf) |
| 3 | Iterative nullspace probing | [Results](results/nullspace/) |
| 4 | Multi-probe steering | [Results](results/steering/) |
| 5 | Offset-preserving centroid splines | [Report](results/spline_offset_v1/REPORT.md) |
| 6 | Steering through the predictor | [Linear](results/iterated_velocity_v1/REPORT.md) · [Spline](results/token_centroid_velocity_v1/REPORT.md) |

Additional experiments and artifacts are in [results/](results/), including
[iterated splines](results/iterated_centroid_velocity_v1/REPORT.md) and
[covariance steering](results/covariance_velocity_v1/REPORT.md).

## Code and data

- [utils.py](utils.py): notebook helpers.
- [requirements.txt](requirements.txt): dependencies.
- [misc/](misc/): standalone scripts and historical code.
- [Supplied repository](vjepa-physics-takehome-4E00/): original task and dataset metadata.

To recompute the saved predictor analysis, run notebook Section 1 (both setup
cells), then Section 6; no videos, feature caches, or GPU are needed.
Full fitting and inference require external videos, caches, and model weights;
adjust the notebook's `SCRATCH` and `CHECKPOINT` paths. Those large assets are
not included in this repository.
