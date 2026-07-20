#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
TAG="${2:-finetune_alignn_hyperparameters_new_script}"
cd "$REPO_ROOT"

python scripts/shared/preflight_finetune_candidate_hyperparameters.py
python scripts/shared/run_finetune_candidate_hyperparameters_suite.py \
  --repo-root . \
  --experiment-tag "$TAG" \
  --device cuda \
  --epochs 300 \
  --batch-size 64 \
  --lr 0.001
python scripts/shared/summarize_finetune_candidate_hyperparameters.py \
  --repo-root . \
  --experiment-tag "$TAG"
python scripts/shared/plot_finetune_training_curves_candidate_hyperparameters.py \
  --repo-root . \
  --experiment-tag "$TAG"
scripts/shared/check_finetune_candidate_hyperparameters_status.sh . "$TAG"
