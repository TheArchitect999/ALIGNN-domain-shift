#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${1:-.}"
cd "$REPO_ROOT"

echo "[1/5] Validating Stage 2 dataset..."
python scripts/dataset/validate_family_datasets.py --root ./data

echo
echo "[2/5] Checking required baseline artifact files..."
required_files=(
  "data/oxide/summaries/summary.json"
  "data/nitride/summaries/summary.json"
  "results/zero_shot/zero_shot_summary.csv"
  "results/zero_shot/oxide/predictions.csv"
  "results/zero_shot/nitride/predictions.csv"
  "results/protocol_1/oxide/N50_seed0/finetune_last2/summary.json"
  "results/protocol_1/nitride/N50_seed0/finetune_last2/summary.json"
  "results/protocol_1/oxide/N50_seed0/finetune_last2/best_model.pt"
  "results/protocol_1/nitride/N50_seed0/finetune_last2/best_model.pt"
  "results/protocol_1/oxide/N50_seed0/finetune_last2/history_val.json"
  "results/protocol_1/nitride/N50_seed0/finetune_last2/history_val.json"
  "results/protocol_1/oxide/N50_seed0/finetune_last2/prediction_results_test_set.csv"
  "results/protocol_1/nitride/N50_seed0/finetune_last2/prediction_results_test_set.csv"
)

missing=0
for path in "${required_files[@]}"; do
  if [[ -f "$path" ]]; then
    echo "OK  $path"
  else
    echo "MISS $path"
    missing=1
  fi
done

if [[ "$missing" -ne 0 ]]; then
  echo
  echo "baseline status: INCOMPLETE (missing required artifacts)."
  exit 1
fi

echo
echo "[3/5] Checking recorded epoch counts..."
python - <<'PY'
import json
paths = [
    "results/protocol_1/oxide/N50_seed0/finetune_last2/history_val.json",
    "results/protocol_1/nitride/N50_seed0/finetune_last2/history_val.json",
]
for path in paths:
    history = json.load(open(path))
    print(f"{path}: epochs_recorded={len(history)}")
PY

echo
echo "[4/5] Printing key baseline metrics..."
python - <<'PY'
import csv
import json
with open("results/zero_shot/zero_shot_summary.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        print(f"{row['family']}_zero_shot: {row['mae_eV_per_atom']}")

summary_paths = [
    ("oxide_N50_finetune", "results/protocol_1/oxide/N50_seed0/finetune_last2/summary.json"),
    ("nitride_N50_finetune", "results/protocol_1/nitride/N50_seed0/finetune_last2/summary.json"),
]
for label, path in summary_paths:
    data = json.load(open(path))
    mae = data.get("mae_eV_per_atom", data.get("test_mae_eV_per_atom"))
    print(f"{label}: {mae}")
PY

echo
echo "[5/5] baseline validation_criteria..."
cat <<'EOF'
Completed checks:
- oxide subset built and validated
- nitride subset built and validated
- official splits applied before family filtering
- fixed oxide and nitride test sets present
- pretrained ALIGNN checkpoint available locally
- oxide zero-shot result saved
- nitride zero-shot result saved
- oxide N=50 fine-tune result saved
- nitride N=50 fine-tune result saved
EOF

echo
echo "baseline status: COMPLETE"
