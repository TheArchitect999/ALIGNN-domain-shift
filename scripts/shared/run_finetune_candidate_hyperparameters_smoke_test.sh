#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
TAG="${2:-finetune_alignn_hyperparameters_new_script}"
cd "$REPO_ROOT"

python scripts/shared/preflight_finetune_candidate_hyperparameters.py

python scripts/shared/run_finetune_candidate_hyperparameters_suite.py \
  --repo-root . \
  --experiment-tag "$TAG" \
  --families oxide \
  --Ns 10 1000 \
  --seeds 0 \
  --device cuda \
  --epochs 300 \
  --batch-size 64 \
  --lr 0.001

echo "Smoke run complete for tag=${TAG}."
