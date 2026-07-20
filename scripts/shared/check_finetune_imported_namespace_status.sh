#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
RESULTS_ROOT="${2:-results/protocol_1}"
REPORT_ROOT="${3:-results/derived_evidence/protocol_1}"
RUN_SUBDIR="${4:-finetune_last2}"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/finetune"
LEARNING_DIR="${REPORT_ROOT}/Learning Curves"
TRAINING_CURVE_DIR="${REPORT_ROOT}/Training Curves/finetune"
PARITY_DIR="${REPORT_ROOT}/Parity Plots"
cd "$REPO_ROOT"

NS=(10 50 100 200 500 1000)
SEEDS=(0 1 2 3 4)
FAMILIES=(oxide nitride)
EXPECTED_RUNS=60

echo "[1/4] Checking imported finetune summaries..."
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
echo "[2/4] Checking aggregate imported finetune artifacts..."
aggregate_paths=(
  "${SUMMARY_DIR}/finetune_runs.csv"
  "${SUMMARY_DIR}/finetune_summary_by_N.csv"
  "${SUMMARY_DIR}/finetune_summary_wide.csv"
  "${SUMMARY_DIR}/finetune_summary_table.tex"
  "${SUMMARY_DIR}/progress_manifest.json"
  "${SUMMARY_DIR}/finetune_summary_manifest.json"
  "${TRAINING_CURVE_DIR}/training_curve_manifest.csv"
  "${TRAINING_CURVE_DIR}/training_curve_manifest.json"
  "${TRAINING_CURVE_DIR}/readme.md"
  "${TRAINING_CURVE_DIR}/oxide_training_curve_grid.png"
  "${TRAINING_CURVE_DIR}/oxide_training_curve_grid.pdf"
  "${TRAINING_CURVE_DIR}/nitride_training_curve_grid.png"
  "${TRAINING_CURVE_DIR}/nitride_training_curve_grid.pdf"
  "${PARITY_DIR}/parity_plot_manifest.csv"
  "${PARITY_DIR}/parity_plot_manifest.json"
  "${PARITY_DIR}/readme.md"
)
while IFS= read -r path; do
  [[ -n "$path" ]] && aggregate_paths+=("$path")
done < <(SUMMARY_DIR="$SUMMARY_DIR" python - <<'PY'
import json
import os
from pathlib import Path

manifest = Path(os.environ["SUMMARY_DIR"]) / "finetune_summary_manifest.json"
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
echo "[3/4] Validating imported run-row count..."
if ! SUMMARY_DIR="$SUMMARY_DIR" TRAINING_CURVE_DIR="$TRAINING_CURVE_DIR" PARITY_DIR="$PARITY_DIR" EXPECTED_RUNS="$EXPECTED_RUNS" python - <<'PY'
import csv
import os
import sys
from pathlib import Path

summary_dir = Path(os.environ["SUMMARY_DIR"])
training_curve_dir = Path(os.environ["TRAINING_CURVE_DIR"])
parity_dir = Path(os.environ["PARITY_DIR"])

path = summary_dir / "finetune_runs.csv"
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
expected = int(os.environ["EXPECTED_RUNS"])
if len(rows) != expected:
    print(f"Expected {expected} run rows, found {len(rows)}")
    sys.exit(1)

with open(training_curve_dir / "training_curve_manifest.csv", newline="", encoding="utf-8") as handle:
    training_rows = list(csv.DictReader(handle))
if len(training_rows) != expected:
    print(f"Expected {expected} training-curve rows, found {len(training_rows)}")
    sys.exit(1)

with open(parity_dir / "parity_plot_manifest.csv", newline="", encoding="utf-8") as handle:
    parity_rows = list(csv.DictReader(handle))
if len(parity_rows) != 12:
    print(f"Expected 12 parity-plot rows, found {len(parity_rows)}")
    sys.exit(1)

print(f"Row-count check passed ({expected} runs).")
PY
then
  missing=1
fi

echo
echo "[4/4] Imported finetune status..."
if [[ "$missing" -ne 0 ]]; then
  echo "Imported finetune status: INCOMPLETE"
  exit 1
fi

echo "Imported finetune status: COMPLETE"
