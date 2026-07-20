#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-domain_shift_data}"

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[INFO] Environment '$ENV_NAME' already exists. Reusing it."
else
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
conda install -y -c conda-forge \
  numpy=1.26.4 pandas=2.2.3 scipy=1.13.1 scikit-learn=1.5.2 \
  matplotlib=3.9.2 tqdm=4.67.1 jupyterlab=4.3.3 ipykernel=6.29.5 \
  pyyaml=6.0.2 pymatgen

python -m pip install --upgrade pip
python -m pip install "jarvis-tools==2026.1.10"

python - <<'PY'
import json, sys
mods = ["numpy","pandas","scipy","sklearn","matplotlib","pymatgen","jarvis"]
report = {}
for m in mods:
    try:
        mod = __import__(m)
        report[m] = getattr(mod, "__version__", "ok")
    except Exception as e:
        report[m] = f"ERROR: {e}"
try:
    from jarvis.db.figshare import get_db_info
    db = get_db_info()
    report["jarvis_db_keys_present"] = {k: (k in db) for k in ["dft_3d","dft_3d_2021"]}
except Exception as e:
    report["jarvis_db_keys_present"] = f"ERROR: {e}"
print(json.dumps(report, indent=2))
PY

echo "[OK] Stage-2 data environment '$ENV_NAME' is ready."
