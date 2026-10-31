#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-token-centroid-matplotlib
TOKEN_SPLINE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$TOKEN_SPLINE_PYTHON" -m unittest -v misc.test_token_centroid_velocity
"$TOKEN_SPLINE_PYTHON" -u -m misc.token_centroid_velocity fit "$@"
"$TOKEN_SPLINE_PYTHON" -u -m misc.token_centroid_velocity evaluate "$@"
