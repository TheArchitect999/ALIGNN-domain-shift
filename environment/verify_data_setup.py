#!/usr/bin/env python3
from __future__ import annotations
import importlib, json

mods = ["numpy","pandas","scipy","sklearn","matplotlib","pymatgen","jarvis"]
report = {}
for m in mods:
    try:
        mod = importlib.import_module(m)
        report[m] = {"ok": True, "version": getattr(mod, "__version__", "unknown")}
    except Exception as e:
        report[m] = {"ok": False, "error": str(e)}
try:
    from jarvis.db.figshare import get_db_info
    db = get_db_info()
    report["jarvis_db_keys_present"] = {k: (k in db) for k in ["dft_3d","dft_3d_2021"]}
except Exception as e:
    report["jarvis_db_keys_present"] = {"ok": False, "error": str(e)}
print(json.dumps(report, indent=2))
