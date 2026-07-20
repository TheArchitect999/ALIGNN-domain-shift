from __future__ import annotations

import importlib
import json
import os
import platform
import sys
from pathlib import Path

INSTALL_HINT = (
    "Bootstrap the Colab environment with "
    "`environment/setup_colab.sh`."
)
SUPPORTED_PYTHON = {(3, 10), (3, 11)}


def _optional_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return False, "missing"
    return True, getattr(module, "__version__", "unknown")


def _dgl_cuda_check(cuda_available: bool) -> tuple[bool, str]:
    try:
        import dgl  # pylint: disable=import-error
    except Exception:
        return False, "missing"
    if not cuda_available:
        return True, "skipped_no_cuda"
    try:
        graph = dgl.graph(([0], [0]))
        graph = graph.to("cuda")
        return True, str(graph.device)
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def _checkpoint_check(checkpoint_path: Path, config_path: Path) -> tuple[bool, str]:
    if not checkpoint_path.exists():
        return False, f"missing_checkpoint:{checkpoint_path}"
    if not config_path.exists():
        return False, f"missing_config:{config_path}"
    try:
        import torch  # pylint: disable=import-error
        from alignn.models.alignn import ALIGNN, ALIGNNConfig  # pylint: disable=import-error
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)

    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        model = ALIGNN(ALIGNNConfig(**raw["model"]))
        state = torch.load(checkpoint_path, map_location="cpu")
        model_state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(model_state)
        return True, model.__class__.__name__
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def main() -> int:
    repo_root = Path(".").resolve()
    checkpoint_path = repo_root / "configs/pretrained" / "checkpoint_300.pt"
    config_path = repo_root / "configs/pretrained" / "config.json"

    python_ok = sys.version_info[:2] in SUPPORTED_PYTHON
    torch_ok, torch_ver = _optional_version("torch")
    dgl_ok, dgl_ver = _optional_version("dgl")
    alignn_ok, alignn_ver = _optional_version("alignn")
    jarvis_ok, jarvis_ver = _optional_version("jarvis")
    pandas_ok, pandas_ver = _optional_version("pandas")
    matplotlib_ok, matplotlib_ver = _optional_version("matplotlib")
    payload = {
        "python": platform.python_version(),
        "python_supported": python_ok,
        "torch_installed": torch_ok,
        "torch": torch_ver,
        "dgl_installed": dgl_ok,
        "dgl": dgl_ver,
        "alignn_installed": alignn_ok,
        "alignn": alignn_ver,
        "jarvis_installed": jarvis_ok,
        "jarvis": jarvis_ver,
        "pandas_installed": pandas_ok,
        "pandas": pandas_ver,
        "matplotlib_installed": matplotlib_ok,
        "matplotlib": matplotlib_ver,
        "github_token_present": bool(os.environ.get("GITHUB_TOKEN")),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
    }

    if torch_ok:
        import torch  # pylint: disable=import-error

        payload["cuda_available"] = torch.cuda.is_available()
        payload["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            payload["cuda_device_name_0"] = torch.cuda.get_device_name(0)
    else:
        payload["cuda_available"] = False
        payload["cuda_device_count"] = 0

    dgl_cuda_ok, dgl_cuda_detail = _dgl_cuda_check(payload["cuda_available"])
    checkpoint_ok, checkpoint_detail = _checkpoint_check(checkpoint_path, config_path)

    payload["dgl_cuda_enabled"] = dgl_cuda_ok
    payload["dgl_cuda_check"] = dgl_cuda_detail
    payload["checkpoint_loadable"] = checkpoint_ok
    payload["checkpoint_check"] = checkpoint_detail

    print(json.dumps(payload, indent=2))

    if not python_ok:
        raise SystemExit(
            "Unsupported Python version for this Colab workflow. "
            "Expected Python 3.10 or 3.11."
        )

    missing_packages = []
    if not torch_ok:
        missing_packages.append("torch")
    if not dgl_ok:
        missing_packages.append("dgl")
    if not alignn_ok:
        missing_packages.append("alignn")
    if not jarvis_ok:
        missing_packages.append("jarvis")
    if not pandas_ok:
        missing_packages.append("pandas")
    if not matplotlib_ok:
        missing_packages.append("matplotlib")

    if missing_packages:
        missing_text = ", ".join(missing_packages)
        raise SystemExit(f"Missing Colab fine-tune dependencies: {missing_text}. {INSTALL_HINT}")
    if not payload["github_token_present"]:
        raise SystemExit("GITHUB_TOKEN is required for the Colab GitHub persistence workflow.")
    if not payload["cuda_available"]:
        raise SystemExit("CUDA unavailable. Configure a Colab GPU runtime before running this sweep.")
    if not dgl_cuda_ok:
        raise SystemExit(
            "DGL does not have CUDA enabled in this environment. "
            "Install the cu121 wheel before running this sweep."
        )
    if not checkpoint_ok:
        raise SystemExit(
            "The local pretrained checkpoint/config could not be loaded. "
            "Bootstrap the sparse checkout again before running this sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
