#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-domain_shift_train}"
MODE="${2:-cpu}"  # cpu or gpu

source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
else
  conda create -y -n "$ENV_NAME" python=3.10
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip

if [[ "$MODE" == "gpu" ]]; then
  python -m pip install \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cu121
else
  python -m pip install \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    --index-url https://download.pytorch.org/whl/cpu
fi

python -m pip install \
  numpy==1.26.4 \
  alignn==2025.4.1 \
  jarvis-tools==2026.3.10 \
  pandas==2.3.3 \
  requests==2.33.1 \
  urllib3==2.6.3 \
  scikit-learn==1.7.2 \
  matplotlib==3.10.6

if [[ "$MODE" == "gpu" ]]; then
  # alignn pulls CPU-only dgl<=1.1.1 from PyPI; override it with the CUDA wheel.
  python -m pip install --no-deps --force-reinstall \
    https://data.dgl.ai/wheels/cu121/dgl-1.1.3%2Bcu121-cp310-cp310-manylinux1_x86_64.whl
else
  python -m pip install dgl==1.1.1
fi

python - <<'PY'
import dgl
import torch
import alignn
import pandas
import jarvis
from alignn.pretrained import get_figshare_model

print(
    {
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "dgl": dgl.__version__,
        "alignn": getattr(alignn, "__version__", "unknown"),
        "pandas": pandas.__version__,
    }
)
result = get_figshare_model("configs/pretrained")
model = result[0] if isinstance(result, tuple) else result
print("pretrained_ok", model.__class__.__name__)
PY
