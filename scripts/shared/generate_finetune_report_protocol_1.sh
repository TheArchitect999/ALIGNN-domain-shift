#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

SET_ROOT="results/derived_evidence/protocol_1"
SUMMARY_DIR="${SET_ROOT}/Summaries/finetune"
LEARNING_DIR="${SET_ROOT}/Learning Curves"
TRAINING_DIR="${SET_ROOT}/Training Curves/finetune"
PARITY_DIR="${SET_ROOT}/Parity Plots"

python scripts/shared/summarize_finetune_reports.py \
  --repo-root . \
  --results-root results/protocol_1 \
  --zero-shot-root results/zero_shot \
  --run-subdir finetune_last2 \
  --families oxide nitride \
  --Ns 10 50 100 200 500 1000 \
  --seeds 0 1 2 3 4 \
  --summary-dir "${SUMMARY_DIR}" \
  --plot-dir "${LEARNING_DIR}" \
  --title-label "canonical-protocol fine-tuning learning curve" \
  --plot-name-template "{Family} Learning Curve - protocol_1" \
  --plot-title-template "{Family} Learning Curve - protocol_1"

python scripts/shared/plot_finetune_training_curves.py \
  --repo-root . \
  --results-root results/protocol_1 \
  --run-subdir finetune_last2 \
  --families oxide nitride \
  --Ns 10 50 100 200 500 1000 \
  --seeds 0 1 2 3 4 \
  --out-dir "${TRAINING_DIR}" \
  --title-label "finetune canonical-protocol" \
  --protocol-note "pretrained ALIGNN with canonical-protocols (epochs=50, batch_size=16, learning_rate=0.0001)"

python scripts/shared/generate_finetune_parity_plots.py \
  --repo-root . \
  --results-root results/protocol_1 \
  --run-subdir finetune_last2 \
  --report-dir "${SET_ROOT}" \
  --out-dir "${PARITY_DIR}" \
  --set-number 1
