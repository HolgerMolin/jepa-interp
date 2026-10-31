#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-spline-velocity-matplotlib
SPLINE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$SPLINE_PYTHON" -m unittest -v misc.test_spline_velocity
"$SPLINE_PYTHON" -u -m misc.spline_velocity fit
"$SPLINE_PYTHON" -u -m misc.spline_velocity evaluate
"$SPLINE_PYTHON" -u -m misc.report_spline_velocity
