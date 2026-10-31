#!/bin/bash
#SBATCH --job-name=jepa-part1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/users/hmolin/jepa-interp/results/slurm-%j.log
set -euo pipefail
cd /home/users/hmolin/jepa-interp
export HF_HOME=/scratch/users/hmolin/jepa-interp/hf-cache
export MPLCONFIGDIR=/scratch/users/hmolin/jepa-interp/mpl-cache
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1
PY=/scratch/users/hmolin/jepa-interp/.venv/bin/python
"$PY" scripts/audit_data.py
"$PY" scripts/extract_features.py
"$PY" scripts/probe_layers.py
"$PY" scripts/plot_detail.py
"$PY" scripts/summarize_results.py
