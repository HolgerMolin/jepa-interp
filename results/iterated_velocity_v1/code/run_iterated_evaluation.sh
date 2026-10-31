#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-iterated-velocity-matplotlib
ITERATED_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$ITERATED_PYTHON" -m unittest -v test_iterated_velocity
"$ITERATED_PYTHON" -u evaluate_iterated_velocity.py assemble
"$ITERATED_PYTHON" -u evaluate_iterated_velocity.py evaluate
"$ITERATED_PYTHON" -u evaluate_iterated_velocity.py audit
"$ITERATED_PYTHON" -u report_iterated_velocity.py
