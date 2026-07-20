#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

python scripts/shared/preflight_from_scratch_protocol_2.py
python scripts/shared/run_from_scratch_suite.py \
  --repo-root . \
  --results-root results/protocol_2 \
  --families oxide nitride \
  --Ns 50 \
  --seeds 0 \
  --device cuda \
  --epochs 5 \
  --batch-size 64 \
  --lr 0.001 \
  --run-subdir train_alignn_from_scratch_smoke \
  --config-dir configs/protocol_2/from_scratch \
  --report-dir "results/summaries/protocol_2/From Scratch Smoke"

echo "protocol_2 from-scratch smoke runs finished."
