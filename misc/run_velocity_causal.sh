#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-velocity-causal-matplotlib
PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
OUTPUT=/home/users/hmolin/jepa-interp/results/velocity_causal_v1
CACHE=/scratch/users/hmolin/jepa-interp/velocity_causal_v1
"$PYTHON" -m unittest -v misc.test_velocity_causal
"$PYTHON" -u -m misc.velocity_causal extract --output "$OUTPUT" --cache "$CACHE"
"$PYTHON" -u -m misc.velocity_causal baseline --output "$OUTPUT" --cache "$CACHE"
