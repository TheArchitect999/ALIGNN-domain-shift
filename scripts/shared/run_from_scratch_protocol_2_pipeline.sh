#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

RESULTS_ROOT="results/protocol_2"
RUN_SUBDIR="train_alignn_from_scratch"
CONFIG_DIR="configs/protocol_2/from_scratch"
REPORT_ROOT="results/derived_evidence/protocol_2"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/From Scratch"
PLOT_DIR="${REPORT_ROOT}/Comparison Plots"

python scripts/shared/preflight_from_scratch_protocol_2.py
python scripts/shared/run_from_scratch_suite.py \
  --repo-root . \
  --results-root "$RESULTS_ROOT" \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --device cuda \
  --epochs 300 \
  --batch-size 64 \
  --lr 0.001 \
  --run-subdir "$RUN_SUBDIR" \
  --config-dir "$CONFIG_DIR" \
  --report-dir "$SUMMARY_DIR"
python scripts/shared/summarize_from_scratch_reports.py \
  --repo-root . \
  --results-root "$RESULTS_ROOT" \
  --finetune-results-root "$RESULTS_ROOT" \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --run-subdir "$RUN_SUBDIR" \
  --finetune-run-subdir finetune_last2 \
  --summary-dir "$SUMMARY_DIR" \
  --plot-dir "$PLOT_DIR" \
  --title-label "protocol_2 From-Scratch Comparison" \
  --plot-name-template "{Family} Comparison Plot - protocol_2" \
  --plot-title-template "{Family} Comparison Plot - protocol_2"
bash scripts/shared/check_from_scratch_protocol_2_status.sh . "$RUN_SUBDIR" "$SUMMARY_DIR" "$RESULTS_ROOT" "$PLOT_DIR"
