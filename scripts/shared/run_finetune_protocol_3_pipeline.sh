#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PUSH_AFTER_RUN="${PUSH_AFTER_RUN:-1}"
PUSH_FINAL_REPORTS="${PUSH_FINAL_REPORTS:-1}"

RUN_SUBDIR="finetune_last2_epochs100_bs32_lr5e5"
CONFIG_DIR="configs/protocol_3/finetune"
REPORT_ROOT="results/derived_evidence/protocol_3"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/finetune"
LEARNING_DIR="${REPORT_ROOT}/Learning Curves"
TRAINING_DIR="${REPORT_ROOT}/Training Curves/finetune"
PARITY_DIR="${REPORT_ROOT}/Parity Plots"

cd "$REPO_ROOT"

python scripts/shared/preflight_finetune_protocol_3.py

suite_args=(
  --repo-root .
  --device cuda
)

if [[ "$PUSH_AFTER_RUN" == "1" ]]; then
  suite_args+=(--git-push-after-run --git-remote "$GIT_REMOTE" --git-branch "$GIT_BRANCH")
fi

python scripts/shared/run_finetune_protocol_3_suite.py "${suite_args[@]}"

python scripts/shared/summarize_finetune_reports.py \
  --repo-root . \
  --results-root results/protocol_3 \
  --zero-shot-root results/zero_shot \
  --run-subdir "$RUN_SUBDIR" \
  --families oxide nitride \
  --Ns 10 50 100 200 500 1000 \
  --seeds 0 1 2 3 4 \
  --summary-dir "$SUMMARY_DIR" \
  --plot-dir "$LEARNING_DIR" \
  --title-label "partial fine-tuning learning curve (100 epochs, batch 32, lr 5e-5)" \
  --plot-name-template "{Family} Learning Curve - protocol_3" \
  --plot-title-template "{Family} Learning Curve - protocol_3"

python scripts/shared/plot_finetune_training_curves.py \
  --repo-root . \
  --results-root results/protocol_3 \
  --run-subdir "$RUN_SUBDIR" \
  --families oxide nitride \
  --Ns 10 50 100 200 500 1000 \
  --seeds 0 1 2 3 4 \
  --out-dir "${TRAINING_DIR}" \
  --title-label "finetune E100-B32-LR5e-5" \
  --protocol-note "pretrained ALIGNN with explicit partial fine-tuning; model.eval() then fc.train() + gcn_layers[3].train(); all other layers frozen"

python scripts/shared/generate_finetune_parity_plots.py \
  --repo-root . \
  --results-root results/protocol_3 \
  --run-subdir "$RUN_SUBDIR" \
  --report-dir "$REPORT_ROOT" \
  --out-dir "$PARITY_DIR" \
  --set-number 3

bash scripts/shared/check_finetune_protocol_3_status.sh . "$RUN_SUBDIR" "$REPORT_ROOT"

if [[ "$PUSH_FINAL_REPORTS" == "1" ]]; then
  git add -- "$REPORT_ROOT" "$CONFIG_DIR"
  if [[ -n "$(git status --porcelain -- "$REPORT_ROOT" "$CONFIG_DIR")" ]]; then
    git commit -m "colab: finalize finetune last2 epochs100 bs32 lr5e5 reports"
    git push "$GIT_REMOTE" "HEAD:${GIT_BRANCH}"
  fi
fi
