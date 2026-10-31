#!/usr/bin/env bash
# Use an existing allocation; request one GPU and 32 GiB of host memory.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec srun --jobid="${1:-1903060}" --overlap --exact \
  --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --gres=gpu:1 \
  /usr/bin/env OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  MPLCONFIGDIR=/tmp/jepa-full-token-matplotlib \
  /scratch/users/hmolin/jepa-interp/.venv/bin/python -u -m misc.probe_full_tokens
