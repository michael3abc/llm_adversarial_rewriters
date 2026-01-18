#!/usr/bin/env bash
set -euo pipefail

TRIAL_DIR="results/evaluate_rewrite"
CONFIG_SRC="src/algo.yaml"
CONFIG_SNAPSHOT="${TRIAL_DIR}/algo.yaml"
DATASET_PATH="data/orig_prompt/adl_final_25w_part1_with_cost.jsonl"
ALGORITHM_NAME="evaluate_rewrite"

mkdir -p "${TRIAL_DIR}"
cp "${CONFIG_SRC}" "${CONFIG_SNAPSHOT}"

python run_inference.py \
  --dataset "${DATASET_PATH}" \
  --algorithm "${ALGORITHM_NAME}"

python run_eval.py \
  --dataset "${DATASET_PATH}" \
  --algorithm "${ALGORITHM_NAME}"

python src/utils/reporter.py \
  --trial-dir "${TRIAL_DIR}" \
  --config "${CONFIG_SNAPSHOT}"
