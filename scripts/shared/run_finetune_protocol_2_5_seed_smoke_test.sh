#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
TAG="${2:-finetune_smoke}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
GIT_BRANCH="${GIT_BRANCH:-main}"
PUSH_AFTER_RUN="${PUSH_AFTER_RUN:-0}"
RUN_SUBDIR="${RUN_SUBDIR:-finetune_last2_smoke}"
CONFIG_DIR="${CONFIG_DIR:-configs/protocol_2/finetune}"
SUMMARY_DIR="${SUMMARY_DIR:-results/summaries/protocol_2/finetune Smoke}"
TRAINING_CURVE_DIR="${TRAINING_CURVE_DIR:-results/derived_evidence/protocol_2/Training Curves/finetune Smoke}"

cd "$REPO_ROOT"

python scripts/shared/preflight_finetune_protocol_2_5_seed.py

suite_args=(
  --repo-root .
  --experiment-tag "$TAG"
  --run-subdir "$RUN_SUBDIR"
  --config-dir "$CONFIG_DIR"
  --report-dir "$SUMMARY_DIR"
  --families oxide
  --Ns 50
  --seeds 0
  --device cuda
  --epochs 2
  --batch-size 64
  --min-batch-size 8
  --lr 0.001
)

if [[ "$PUSH_AFTER_RUN" == "1" ]]; then
  suite_args+=(--git-push-after-run --git-remote "$GIT_REMOTE" --git-branch "$GIT_BRANCH")
fi

python scripts/shared/run_finetune_protocol_2_5_seed_suite.py "${suite_args[@]}"
python scripts/shared/summarize_finetune_protocol_2_5_seed.py \
  --repo-root . \
  --experiment-tag "$TAG" \
  --run-subdir "$RUN_SUBDIR" \
  --families oxide \
  --Ns 50 \
  --seeds 0 \
  --out-dir "$SUMMARY_DIR"
python scripts/shared/plot_finetune_training_curves_protocol_2_5_seed.py \
  --repo-root . \
  --experiment-tag "$TAG" \
  --run-subdir "$RUN_SUBDIR" \
  --families oxide \
  --Ns 50 \
  --seeds 0 \
  --out-dir "$TRAINING_CURVE_DIR"

echo "Tagged finetune Colab smoke run finished."
