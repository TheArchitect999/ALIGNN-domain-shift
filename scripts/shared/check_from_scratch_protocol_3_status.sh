#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
RUN_SUBDIR="${2:-train_alignn_from_scratch_epochs100_bs32_lr5e5}"
REPORT_ROOT="${3:-results/derived_evidence/protocol_3}"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/From Scratch"
TRAINING_DIR="${REPORT_ROOT}/Training Curves/From Scratch"

cd "$REPO_ROOT"

NS=(50 500)
SEEDS=(0 1 2 3 4)
FAMILIES=(oxide nitride)

echo "[1/3] Checking per-run from-scratch summaries..."
missing=0
for family in "${FAMILIES[@]}"; do
  for n in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      path="results/protocol_3/${family}/N${n}_seed${seed}/${RUN_SUBDIR}/summary.json"
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
echo "[2/3] Checking aggregate artifacts..."
aggregate_paths=(
  "${SUMMARY_DIR}/from_scratch_runs.csv"
  "${SUMMARY_DIR}/from_scratch_summary.csv"
  "${SUMMARY_DIR}/from_scratch_manifest.json"
  "${SUMMARY_DIR}/run_suite_summary.json"
  "${TRAINING_DIR}/training_curve_manifest.csv"
  "${TRAINING_DIR}/training_curve_manifest.json"
  "${TRAINING_DIR}/readme.md"
  "${TRAINING_DIR}/oxide_training_curve_grid.png"
  "${TRAINING_DIR}/oxide_training_curve_grid.pdf"
  "${TRAINING_DIR}/nitride_training_curve_grid.png"
  "${TRAINING_DIR}/nitride_training_curve_grid.pdf"
)
while IFS= read -r path; do
  [[ -n "$path" ]] && aggregate_paths+=("$path")
done < <(SUMMARY_DIR="$SUMMARY_DIR" python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["SUMMARY_DIR"]) / "from_scratch_manifest.json"
data = json.loads(manifest.read_text(encoding="utf-8"))
for family in ("oxide", "nitride"):
    plot = data.get("plots", {}).get(family, {})
    for key in ("png", "pdf"):
        value = plot.get(key)
        if value:
            print(value)
PY
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
echo "[3/3] Validating run-row count..."
if ! SUMMARY_DIR="$SUMMARY_DIR" TRAINING_DIR="$TRAINING_DIR" python - <<'PY'
import csv
import os
import sys
from pathlib import Path

summary_dir = Path(os.environ["SUMMARY_DIR"])
training_dir = Path(os.environ["TRAINING_DIR"])

with open(summary_dir / "from_scratch_runs.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
if len(rows) != 20:
    print(f"Expected 20 run rows, found {len(rows)}")
    sys.exit(1)

with open(training_dir / "training_curve_manifest.csv", newline="", encoding="utf-8") as handle:
    training_rows = list(csv.DictReader(handle))
if len(training_rows) != 20:
    print(f"Expected 20 training-curve rows, found {len(training_rows)}")
    sys.exit(1)

print("Row-count check passed (20 runs).")
PY
then
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "from_scratch from-scratch status: INCOMPLETE"
  exit 1
fi

echo "from_scratch from-scratch status: COMPLETE"
