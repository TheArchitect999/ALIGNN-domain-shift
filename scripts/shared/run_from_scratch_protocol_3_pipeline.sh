#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PUSH_AFTER_RUN="${PUSH_AFTER_RUN:-1}"
PUSH_FINAL_REPORTS="${PUSH_FINAL_REPORTS:-1}"

RUN_SUBDIR="train_alignn_from_scratch_epochs100_bs32_lr5e5"
CONFIG_DIR="configs/protocol_3/from_scratch"
REPORT_ROOT="results/derived_evidence/protocol_3"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/From Scratch"
PLOT_DIR="${REPORT_ROOT}/Comparison Plots"
TRAINING_DIR="${REPORT_ROOT}/Training Curves/From Scratch"
FINETUNE_RUN_SUBDIR="finetune_last2_epochs100_bs32_lr5e5"

cd "$REPO_ROOT"

python scripts/shared/preflight_from_scratch_protocol_3.py

suite_args=(
  --repo-root .
  --device cuda
)

if [[ "$PUSH_AFTER_RUN" == "1" ]]; then
  suite_args+=(--git-push-after-run --git-remote "$GIT_REMOTE" --git-branch "$GIT_BRANCH")
fi

python scripts/shared/run_from_scratch_protocol_3_suite.py "${suite_args[@]}"

python scripts/shared/summarize_from_scratch_reports.py \
  --repo-root . \
  --results-root results/protocol_3 \
  --finetune-results-root results/protocol_3 \
  --zero-shot-root results/zero_shot \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --run-subdir "$RUN_SUBDIR" \
  --finetune-run-subdir "$FINETUNE_RUN_SUBDIR" \
  --summary-dir "$SUMMARY_DIR" \
  --plot-dir "$PLOT_DIR" \
  --title-label "from_scratch From-Scratch Comparison (100 epochs, batch 32, lr 5e-5)" \
  --plot-name-template "{Family} Comparison Plot - protocol_3" \
  --plot-title-template "{Family} Comparison Plot - protocol_3"

python scripts/shared/plot_from_scratch_training_curves.py \
  --repo-root . \
  --results-root results/protocol_3 \
  --run-subdir "$RUN_SUBDIR" \
  --families oxide nitride \
  --Ns 50 500 \
  --seeds 0 1 2 3 4 \
  --out-dir "$TRAINING_DIR" \
  --title-label "protocol_3 From-Scratch" \
  --protocol-note "randomly initialized ALIGNN trained from scratch; epochs=100, batch_size=32, learning_rate=0.00005"

bash scripts/shared/check_from_scratch_protocol_3_status.sh . "$RUN_SUBDIR" "$REPORT_ROOT"

if [[ "$PUSH_FINAL_REPORTS" == "1" ]]; then
  git add -- "$REPORT_ROOT" "$CONFIG_DIR"
  if [[ -n "$(git status --porcelain -- "$REPORT_ROOT" "$CONFIG_DIR")" ]]; then
    git commit -m "colab: finalize from_scratch from_scratch epochs100 bs32 lr5e5 reports"
    git push "$GIT_REMOTE" "HEAD:${GIT_BRANCH}"
  fi
fi
