#!/bin/bash
#SBATCH --job-name=jepa-audit
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/users/hmolin/jepa-interp/results/diagnostic-%j.log
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export HF_HOME=/scratch/users/hmolin/jepa-interp/hf-cache
/scratch/users/hmolin/jepa-interp/.venv/bin/python -u scripts/diagnose_reproduction.py
