#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
RESULTS_ROOT="${2:-results/protocol_1}"
REPORT_ROOT="${3:-results/derived_evidence/protocol_1}"
RUN_SUBDIR="${4:-train_alignn_from_scratch}"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/From Scratch"
PLOT_DIR="${REPORT_ROOT}/Comparison Plots"
TRAINING_DIR="${REPORT_ROOT}/Training Curves/From Scratch"
cd "$REPO_ROOT"

NS=(50 500)
FAMILIES=(oxide nitride)
SEEDS=(0 1 2 3 4)
EXPECTED_RUNS=20

echo "[1/3] Checking imported from-scratch summaries..."
missing=0
for family in "${FAMILIES[@]}"; do
  for n in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      path="${RESULTS_ROOT}/${family}/N${n}_seed${seed}/${RUN_SUBDIR}/summary.json"
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
echo "[2/3] Checking imported aggregate artifacts..."
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
echo "[3/3] Validating imported run-row count..."
if ! SUMMARY_DIR="$SUMMARY_DIR" TRAINING_DIR="$TRAINING_DIR" EXPECTED_RUNS="$EXPECTED_RUNS" python - <<'PY'
import csv
import os
import sys
from pathlib import Path

summary_dir = Path(os.environ["SUMMARY_DIR"])
training_dir = Path(os.environ["TRAINING_DIR"])

with open(summary_dir / "from_scratch_runs.csv", newline="") as f:
    rows = list(csv.DictReader(f))
expected = int(os.environ["EXPECTED_RUNS"])
if len(rows) != expected:
    print(f"Expected {expected} run rows, found {len(rows)}")
    sys.exit(1)

with open(training_dir / "training_curve_manifest.csv", newline="", encoding="utf-8") as handle:
    training_rows = list(csv.DictReader(handle))
if len(training_rows) != expected:
    print(f"Expected {expected} training-curve rows, found {len(training_rows)}")
    sys.exit(1)

print(f"Row-count check passed ({expected} runs).")
PY
then
  missing=1
fi

if [[ "$missing" -ne 0 ]]; then
  echo "Imported from_scratch from-scratch status: INCOMPLETE"
  exit 1
fi

echo "Imported from_scratch from-scratch status: COMPLETE"
