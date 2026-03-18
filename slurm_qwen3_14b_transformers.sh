#!/bin/bash
#SBATCH --job-name=simple_inf
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

# Recommended caches on local/scratch storage.
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

# Optional (for gated/private repos): export HUGGINGFACE_HUB_TOKEN=...

# ROCm allocator tuning helps reduce fragmentation for large models.
export PYTORCH_HIP_ALLOC_CONF="garbage_collection_threshold:0.8,max_split_size_mb:512"
# Number of dataset prompts to process (override at submit time):
# sbatch --export=ALL,N_PROMPTS=50 slurm_qwen3_14b_transformers.sbatch
N_PROMPTS="${N_PROMPTS:-20}"
START_INDEX="${START_INDEX:-0}"
OUTPUT_JSONL="${OUTPUT_JSONL:-outputs/safetyconflicts_qwen3_14b_${SLURM_JOB_ID}.jsonl}"

python run_qwen3_transformers.py \
  --model-id "Qwen/Qwen3-14B" \
  --dtype bfloat16 \
  --device-map balanced \
  --max-memory-per-gpu 52GiB \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --dataset-id "hadikhalaf/safetyconflicts" \
  --dataset-split "train" \
  --prompt-column "prompt" \
  --num-prompts "${N_PROMPTS}" \
  --start-index "${START_INDEX}" \
  --output-jsonl "${OUTPUT_JSONL}"
