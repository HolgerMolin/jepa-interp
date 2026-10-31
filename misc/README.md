# Experiment scripts

The standalone experiment implementations, tests, Bash runners, and Slurm job
script live here. The main walkthrough is [experiments.ipynb](../experiments.ipynb).
The older, reference-only implementation is in [archive/](archive/README.md).

Run commands from the project root:

```bash
python -m misc.evaluate_pooled_spline_offset --help
python -m unittest discover -s misc -t . -p 'test_*.py'
bash misc/run_iterated_velocity.sh
sbatch misc/run_token_probes.sbatch
```

Python imports use `misc.<module>`. Data, result directories, scratch caches,
model defaults, and numerical routines keep their existing locations and behavior.
`utils.py` remains at the project root as the notebook's active shared helper.
Presentation builders and per-run source snapshots remain with their artifacts.

Source-hash checks remain enabled. Moving imports changes source hashes, so
existing frozen fitting protocols can still reject changed code; use a fresh
output directory for new fits rather than editing historical provenance.
