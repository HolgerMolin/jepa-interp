#!/usr/bin/env bash
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-iterated-velocity-matplotlib
ITERATED_PYTHON=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$ITERATED_PYTHON" -m unittest -v misc.test_iterated_velocity
"$ITERATED_PYTHON" -u -m misc.evaluate_iterated_velocity assemble
"$ITERATED_PYTHON" -u -m misc.evaluate_iterated_velocity evaluate
"$ITERATED_PYTHON" -u -m misc.evaluate_iterated_velocity audit
"$ITERATED_PYTHON" -u -m misc.report_iterated_velocity
