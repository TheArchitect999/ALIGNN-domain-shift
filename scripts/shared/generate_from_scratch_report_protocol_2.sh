#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

SET_ROOT="results/derived_evidence/protocol_2"
SUMMARY_DIR="${SET_ROOT}/Summaries/From Scratch"
PLOT_DIR="${SET_ROOT}/Comparison Plots"
TRAINING_DIR="${SET_ROOT}/Training Curves/From Scratch"

python scripts/shared/summarize_from_scratch_reports.py \
  --repo-root . \
  --results-root results/protocol_2 \
  --finetune-results-root results/protocol_2 \
  --zero-shot-root results/zero_shot \
  --run-subdir train_alignn_from_scratch \
  --finetune-run-subdir finetune_last2 \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --summary-dir "${SUMMARY_DIR}" \
  --plot-dir "${PLOT_DIR}" \
  --title-label "from_scratch ALIGNN-Recommended From-Scratch Comparison" \
  --plot-name-template "{Family} Comparison Plot - protocol_2" \
  --plot-title-template "{Family} Comparison Plot - protocol_2"

python scripts/shared/plot_from_scratch_training_curves.py \
  --repo-root . \
  --results-root results/protocol_2 \
  --run-subdir train_alignn_from_scratch \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --out-dir "${TRAINING_DIR}" \
  --title-label "protocol_2 From-Scratch" \
  --protocol-note "randomly initialized ALIGNN trained from scratch; epochs=300, batch_size=64, learning_rate=0.001"
