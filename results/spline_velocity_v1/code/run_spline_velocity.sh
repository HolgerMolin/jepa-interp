#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-spline-velocity-matplotlib
SPLINE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$SPLINE_PYTHON" -m unittest -v test_spline_velocity
"$SPLINE_PYTHON" -u spline_velocity.py fit
"$SPLINE_PYTHON" -u spline_velocity.py evaluate
"$SPLINE_PYTHON" -u report_spline_velocity.py
