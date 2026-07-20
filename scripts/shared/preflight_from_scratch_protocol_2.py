from __future__ import annotations

import importlib
import json
import platform

INSTALL_HINT = (
    "Install the pinned environment from `environment/requirements-frozen.txt` "
    "or bootstrap it with `environment/setup_training.sh`."
)


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


def main() -> int:
    torch_ok, torch_ver = _optional_version("torch")
    alignn_ok, alignn_ver = _optional_version("alignn")
    jarvis_ok, jarvis_ver = _optional_version("jarvis")
    pandas_ok, pandas_ver = _optional_version("pandas")
    matplotlib_ok, matplotlib_ver = _optional_version("matplotlib")
    dgl_ok, dgl_ver = _optional_version("dgl")

    payload = {
        "python": platform.python_version(),
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
    payload["dgl_cuda_enabled"] = dgl_cuda_ok
    payload["dgl_cuda_check"] = dgl_cuda_detail

    print(json.dumps(payload, indent=2))

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
        raise SystemExit(f"Missing from_scratch dependencies: {missing_text}. {INSTALL_HINT}")

    if not payload["cuda_available"]:
        raise SystemExit("CUDA unavailable. Configure GPU environment before running from_scratch from-scratch suite.")
    if not dgl_cuda_ok:
        raise SystemExit(
            "DGL does not have CUDA enabled in this environment. "
            "Install a CUDA-enabled DGL wheel before running from_scratch from-scratch suite."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
