# Historical implementation

The historical extraction, probing, and audit scripts, detailed plan, and dependency
lock are retained for reference.
They are not active entry points: many paths assume their original location under
`<project>/scripts`, so do not run them from here. Use `../../experiments.ipynb` instead.

Original reports, splits, fitted probes, and audit findings stay in `../../results/`.
Original feature shards stay on scratch; the active notebook uses consolidated
`/scratch/users/hmolin/jepa-interp/pooled_features/{task}.npz` files with identical values.

The notebook cleanup removed four obsolete scripts: `verify_refactor.py` and
`verify_gpu_refactor.py` were one-off checks tied to an old allocation;
`scripts/plot_detail.py` duplicated the notebook's layer-wise plots; and
`scripts/summarize_results.py` generated a report claiming that nullspace and
steering experiments had not yet run. Their saved reports and verification results
remain in `results/`. Active experiment runners, tests, presentation builders,
and per-run source snapshots are retained.
