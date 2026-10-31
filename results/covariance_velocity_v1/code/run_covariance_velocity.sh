#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-covariance-matplotlib
COVARIANCE_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$COVARIANCE_PYTHON" -m unittest -v test_covariance_velocity
"$COVARIANCE_PYTHON" -u covariance_velocity.py fit
"$COVARIANCE_PYTHON" -u covariance_velocity.py evaluate
"$COVARIANCE_PYTHON" -u audit_covariance_velocity.py
"$COVARIANCE_PYTHON" -u covariance_norm_control.py
"$COVARIANCE_PYTHON" -u report_spline_residual_diagnostics.py
