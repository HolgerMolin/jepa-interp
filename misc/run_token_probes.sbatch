#!/bin/bash
#SBATCH --job-name=jepa-token-probes
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/users/hmolin/jepa-interp/results/token-probes-%j.log
#SBATCH --chdir=/home/users/hmolin/jepa-interp
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/jepa-token-probes-matplotlib
/scratch/users/hmolin/jepa-interp/.venv/bin/python -u -m misc.probe_each_token "$@"
