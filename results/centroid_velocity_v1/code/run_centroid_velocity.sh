#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-centroid-velocity-matplotlib
CENTROID_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$CENTROID_PYTHON" -m unittest -v test_centroid_velocity
"$CENTROID_PYTHON" -u centroid_velocity.py fit
"$CENTROID_PYTHON" -u centroid_velocity.py evaluate
"$CENTROID_PYTHON" -u report_centroid_velocity.py
