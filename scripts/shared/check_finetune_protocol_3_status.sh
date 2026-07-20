#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
RUN_SUBDIR="${2:-finetune_last2_epochs100_bs32_lr5e5}"
REPORT_ROOT="${3:-results/derived_evidence/protocol_3}"
SUMMARY_DIR="${REPORT_ROOT}/Summaries/finetune"
LEARNING_DIR="${REPORT_ROOT}/Learning Curves"
TRAINING_CURVE_DIR="${REPORT_ROOT}/Training Curves/finetune"
PARITY_DIR="${REPORT_ROOT}/Parity Plots"

cd "$REPO_ROOT"

NS=(10 50 100 200 500 1000)
SEEDS=(0 1 2 3 4)
FAMILIES=(oxide nitride)

echo "[1/4] Checking per-run fine-tune summaries..."
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
echo "[2/4] Checking aggregate fine-tune artifacts..."
aggregate_paths=(
  "${SUMMARY_DIR}/finetune_runs.csv"
  "${SUMMARY_DIR}/finetune_summary_by_N.csv"
  "${SUMMARY_DIR}/finetune_summary_wide.csv"
  "${SUMMARY_DIR}/finetune_summary_table.tex"
  "${SUMMARY_DIR}/run_suite_summary.json"
  "${SUMMARY_DIR}/finetune_summary_manifest.json"
  "${SUMMARY_DIR}/progress_manifest.json"
  "${TRAINING_CURVE_DIR}/training_curve_manifest.csv"
  "${TRAINING_CURVE_DIR}/training_curve_manifest.json"
  "${PARITY_DIR}/parity_plot_manifest.csv"
  "${PARITY_DIR}/parity_plot_manifest.json"
  "${PARITY_DIR}/readme.md"
  "${TRAINING_CURVE_DIR}/readme.md"
  "${TRAINING_CURVE_DIR}/oxide_training_curve_grid.png"
  "${TRAINING_CURVE_DIR}/oxide_training_curve_grid.pdf"
  "${TRAINING_CURVE_DIR}/nitride_training_curve_grid.png"
  "${TRAINING_CURVE_DIR}/nitride_training_curve_grid.pdf"
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
echo "[3/4] Validating aggregate CSV contents..."
if ! SUMMARY_DIR="$SUMMARY_DIR" TRAINING_CURVE_DIR="$TRAINING_CURVE_DIR" PARITY_DIR="$PARITY_DIR" python - <<'PY'
import csv
import os
import sys
from pathlib import Path

report_dir = Path(os.environ["SUMMARY_DIR"])
training_curve_dir = Path(os.environ["TRAINING_CURVE_DIR"])
parity_dir = Path(os.environ["PARITY_DIR"])
families = ["oxide", "nitride"]
ns = [10, 50, 100, 200, 500, 1000]
expected_runs = 5
bad = False

summary_rows = {}
with open(report_dir / "finetune_summary_by_N.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        summary_rows[(row["family"], int(row["N"]))] = row

for family in families:
    print(f"[{family}]")
    for n in ns:
        row = summary_rows.get((family, n))
        if row is None:
            print(f"  N={n}: MISSING aggregate row")
            bad = True
            continue
        runs = int(row["runs"])
        mean_mae = float(row["mean_test_mae_eV_per_atom"])
        std_mae = float(row["std_test_mae_eV_per_atom"]) if row["std_test_mae_eV_per_atom"] else float("nan")
        status = "OK"
        if runs != expected_runs:
            status = f"BAD runs={runs}"
            bad = True
        print(f"  N={n}: runs={runs}, mean_test_mae={mean_mae:.6f}, std={std_mae:.6f} [{status}]")

with open(report_dir / "finetune_runs.csv", newline="", encoding="utf-8") as handle:
    run_rows = list(csv.DictReader(handle))
expected_total = len(families) * len(ns) * expected_runs
if len(run_rows) != expected_total:
    print(f"Run-count mismatch: expected {expected_total}, found {len(run_rows)}")
    bad = True

with open(training_curve_dir / "training_curve_manifest.csv", newline="", encoding="utf-8") as handle:
    training_rows = list(csv.DictReader(handle))
if len(training_rows) != expected_total:
    print(f"Training-curve manifest mismatch: expected {expected_total}, found {len(training_rows)}")
    bad = True

with open(parity_dir / "parity_plot_manifest.csv", newline="", encoding="utf-8") as handle:
    parity_rows = list(csv.DictReader(handle))
if len(parity_rows) != 12:
    print(f"Parity-plot manifest mismatch: expected 12, found {len(parity_rows)}")
    bad = True

if bad:
    sys.exit(1)
PY
then
  missing=1
fi

echo
echo "[4/4] Fine-tune status..."
if [[ "$missing" -ne 0 ]]; then
  echo "Fine-tune status: INCOMPLETE"
  exit 1
fi

echo "Fine-tune status: COMPLETE"
