#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
TAG="${2:-finetune_alignn_hyperparameters_new_script}"
RUN_SUBDIR="finetune_last2_${TAG}"
REPORT_DIR="results/derived_evidence/${TAG}"
cd "$REPO_ROOT"

NS=(10 50 100 200 500 1000)
SEEDS=(0 1 2)
FAMILIES=(oxide nitride)

echo "[1/4] Checking per-run tagged finetune summaries..."
missing=0
for family in "${FAMILIES[@]}"; do
  for n in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      path="results/protocol_1/${family}/N${n}_seed${seed}/${RUN_SUBDIR}/summary.json"
      if [[ -f "$path" ]]; then
        echo "OK  $path"
      else
        echo "MISS $path"
        missing=1
      fi
    done
  done
done

echo
echo "[2/4] Checking aggregate tagged finetune artifacts..."
aggregate_paths=(
  "${REPORT_DIR}/finetune_runs.csv"
  "${REPORT_DIR}/finetune_summary_by_N.csv"
  "${REPORT_DIR}/finetune_summary_wide.csv"
  "${REPORT_DIR}/finetune_summary_table.tex"
  "${REPORT_DIR}/oxide_learning_curve.png"
  "${REPORT_DIR}/oxide_learning_curve.pdf"
  "${REPORT_DIR}/nitride_learning_curve.png"
  "${REPORT_DIR}/nitride_learning_curve.pdf"
  "${REPORT_DIR}/run_suite_summary.json"
  "${REPORT_DIR}/finetune_summary_manifest.json"
)
for path in "${aggregate_paths[@]}"; do
  if [[ -f "$path" ]]; then
    echo "OK  $path"
  else
    echo "MISS $path"
    missing=1
  fi
done

echo
echo "[3/4] Validating aggregate CSV contents..."
if ! REPORT_DIR="$REPORT_DIR" python - <<'PY'
import csv
import os
import sys

families = ["oxide", "nitride"]
ns = [10, 50, 100, 200, 500, 1000]
expected_runs = 3
report_dir = os.environ["REPORT_DIR"]

bad = False
summary_rows = {}
with open(f"{report_dir}/finetune_summary_by_N.csv", newline="") as f:
    for row in csv.DictReader(f):
        summary_rows[(row["family"], int(row["N"]))] = row

for family in families:
    for n in ns:
        row = summary_rows.get((family, n))
        if row is None or int(row["runs"]) != expected_runs:
            bad = True

with open(f"{report_dir}/finetune_runs.csv", newline="") as f:
    run_rows = list(csv.DictReader(f))

if len(run_rows) != len(families) * len(ns) * expected_runs:
    bad = True

if bad:
    sys.exit(1)
PY
then
  missing=1
fi

echo
echo "[4/4] Tagged finetune status..."
if [[ "$missing" -ne 0 ]]; then
  echo "Tagged finetune status: INCOMPLETE"
  exit 1
fi

echo "Tagged finetune status: COMPLETE"
