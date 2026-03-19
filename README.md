# spar_drafts

Small research pipeline for extracting safety/ethics conflict scenarios, perturbing them, and measuring how a local open-source model changes its responses.

## What this repo does

The workflow is:

1. Extract candidate scenarios from the Hugging Face dataset `jifanz/stress_testing_model_spec`.
2. Label them as safety or ethics conflicts using either:
   - a rule-based classifier, or
   - a DeepSeek judge that returns richer conflict metadata.
3. Generate minimal prompt perturbations with DeepSeek.
4. Run a local Qwen model on the original and perturbed prompts.
5. Ask DeepSeek whether the two model outputs are meaningfully different.

The taxonomy used for labeling lives in `fine_bases_json.json`.

## Main files

- `extract_scenarios.py`: rule-based extraction from the dataset server.
- `extract_scenarios_llm.py`: LLM-as-a-judge extraction with structured labels.
- `perturb_scenarios_llm.py`: creates minimal prompt perturbations.
- `run_qwen3_transformers.py`: runs Qwen locally with Transformers on ROCm/CUDA.
- `judge_perturbed_qwen_transformers.py`: compares original vs perturbed outputs using DeepSeek as a semantic judge.
- `requirements_antuco_rocm_llm.txt`: environment dependencies for the ROCm/HPC setup.
- `slurm_*.sh`: example Slurm jobs for installation and execution.

## Typical pipeline

### 1. Rule-based extraction

```bash
python extract_scenarios.py \
  --limit 1000 \
  --output outputs/scenarios_manual.csv \
  --output-format csv
```

### 2. LLM-based extraction

Requires `DEEPSEEK_API_KEY`.

```bash
python extract_scenarios_llm.py \
  --limit 1000 \
  --output outputs/safety_ethics_scenarios_llm.csv \
  --output-format csv
```

### 3. Prompt perturbation

```bash
python perturb_scenarios_llm.py \
  --input outputs/safety_ethics_scenarios_llm.csv \
  --output outputs/scenarios_perturbed_llm.csv
```

### 4. Local model evaluation

Runs Qwen on both prompt versions, then uses DeepSeek to judge whether the responses changed semantically.

```bash
python judge_perturbed_qwen_transformers.py \
  --input outputs/scenarios_perturbed_llm.csv \
  --output outputs/qwen_perturbation_judged.csv
```

## Requirements

- Python 3.11 recommended.
- `DEEPSEEK_API_KEY` for the DeepSeek-based steps.
- ROCm/CUDA-capable PyTorch for local model inference.
- Enough GPU memory for the selected Qwen model.

For the cluster setup used here, see `requirements_antuco_rocm_llm.txt` and the Slurm scripts.

## Outputs

The `outputs/` directory contains example artifacts from previous runs:

- extracted scenarios
- LLM-labeled scenarios
- perturbed scenarios
- downstream evaluation files

## Notes

- This repo is script-oriented and meant for experimentation rather than packaging.
- Some scripts can run without GPUs, but the Qwen inference stage requires a visible ROCm/CUDA device.
- Network access is required for the dataset server and DeepSeek API calls.
