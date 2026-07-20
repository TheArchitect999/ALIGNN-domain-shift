from __future__ import annotations

import json
import platform

import torch


def main() -> int:
    cuda_available = torch.cuda.is_available()
    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
    }
    if cuda_available:
        payload["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    print(json.dumps(payload, indent=2))
    if not cuda_available:
        raise SystemExit("torch.cuda.is_available() returned False; aborting tagged GPU sweep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
