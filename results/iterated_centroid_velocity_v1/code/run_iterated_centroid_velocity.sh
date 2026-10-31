#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-iterated-centroid-matplotlib
RESIDUAL_SPLINE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$RESIDUAL_SPLINE_PYTHON" -m unittest -v test_iterated_centroid_velocity
"$RESIDUAL_SPLINE_PYTHON" -u iterated_centroid_velocity.py fit
"$RESIDUAL_SPLINE_PYTHON" -u iterated_centroid_velocity.py evaluate
"$RESIDUAL_SPLINE_PYTHON" -u iterated_centroid_velocity.py audit
"$RESIDUAL_SPLINE_PYTHON" -u report_iterated_centroid_velocity.py
