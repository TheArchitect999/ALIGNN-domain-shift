#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

NS=(10 50 100 200 500 1000)
SEEDS=(0 1 2)
FAMILIES=(oxide nitride)

echo "[1/4] Checking per-run finetune summaries..."
missing=0
for family in "${FAMILIES[@]}"; do
  for n in "${NS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      path="results/protocol_1/${family}/N${n}_seed${seed}/finetune_last2/summary.json"
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
echo "[2/4] Checking aggregate finetune artifacts..."
aggregate_paths=(
  "results/derived_evidence/finetune/finetune_runs.csv"
  "results/derived_evidence/finetune/finetune_summary_by_N.csv"
  "results/derived_evidence/finetune/finetune_summary_wide.csv"
  "results/derived_evidence/finetune/finetune_summary_table.tex"
  "results/derived_evidence/finetune/oxide_learning_curve.png"
  "results/derived_evidence/finetune/oxide_learning_curve.pdf"
  "results/derived_evidence/finetune/nitride_learning_curve.png"
  "results/derived_evidence/finetune/nitride_learning_curve.pdf"
  "results/derived_evidence/finetune/run_suite_summary.json"
  "results/derived_evidence/finetune/finetune_summary_manifest.json"
  "results/derived_evidence/finetune_report.tex"
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
if ! python - <<'PY'
import csv
import json
import math
import os
import sys

families = ["oxide", "nitride"]
ns = [10, 50, 100, 200, 500, 1000]
expected_runs = 3
bad = False

summary_rows = {}
with open("results/derived_evidence/finetune/finetune_summary_by_N.csv", newline="") as f:
    for row in csv.DictReader(f):
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
        std_mae = float(row["std_test_mae_eV_per_atom"])
        zero = float(row["zero_shot_mae_eV_per_atom"])
        gain = float(row["transfer_gain_vs_zero_shot"])
        status = "OK"
        if runs != expected_runs:
            status = f"BAD runs={runs}"
            bad = True
        print(
            f"  N={n}: runs={runs}, mean_test_mae={mean_mae:.6f}, "
            f"std={std_mae:.6f}, zero_shot={zero:.6f}, gain={gain:.6f} [{status}]"
        )

with open("results/derived_evidence/finetune/finetune_runs.csv", newline="") as f:
    run_rows = list(csv.DictReader(f))

if len(run_rows) != len(families) * len(ns) * expected_runs:
    print(
        f"Run-count mismatch: expected {len(families) * len(ns) * expected_runs}, "
        f"found {len(run_rows)}"
    )
    bad = True

if bad:
    sys.exit(1)
PY
then
  missing=1
fi

echo
echo "[4/4] finetune status..."
if [[ "$missing" -ne 0 ]]; then
  echo "finetune status: INCOMPLETE"
  exit 1
fi

echo "finetune status: COMPLETE"
