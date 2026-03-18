#!/bin/bash
#SBATCH --job-name=exp
#SBATCH -p ialab
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH -w antuco
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=natan.brugueras@cenia.cl

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs outputs

# Optional: adapt these module names to your cluster.
# module purge
# module load rocm/6.4
# module load python/3.11

# Activate your prepared environment.
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate antuco_torch310
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

python extract_scenarios.py \
  --limit 1000 \
  --output outputs/scenarios_manual.csv \
  --output-format csv
