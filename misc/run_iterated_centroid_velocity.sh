#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-iterated-centroid-matplotlib
RESIDUAL_SPLINE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$RESIDUAL_SPLINE_PYTHON" -m unittest -v misc.test_iterated_centroid_velocity
"$RESIDUAL_SPLINE_PYTHON" -u -m misc.iterated_centroid_velocity fit
"$RESIDUAL_SPLINE_PYTHON" -u -m misc.iterated_centroid_velocity evaluate
"$RESIDUAL_SPLINE_PYTHON" -u -m misc.iterated_centroid_velocity audit
"$RESIDUAL_SPLINE_PYTHON" -u -m misc.report_iterated_centroid_velocity
