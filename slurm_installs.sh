#!/bin/bash
#SBATCH --job-name=installs
#SBATCH -p ialab
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH -w antuco
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
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

# Recommended caches on local/scratch storage.
export PYTHONNOUSERSITE=1


python -m pip install --upgrade pip

python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/rocm6.4

python -m pip install -r requirements_antuco_rocm_llm.txt

