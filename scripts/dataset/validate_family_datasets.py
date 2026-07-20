#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set


def load_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def jids(rows: List[Dict[str, str]]) -> Set[str]:
    return {row["jid"] for row in rows}


def must(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_family(root: Path, family_name: str) -> Dict[str, int]:
    family_dir = root / family_name
    manifests = family_dir / "manifests"
    summary_path = family_dir / "summaries" / "summary.json"

    all_rows = load_rows(manifests / "all.csv")
    train_rows = load_rows(manifests / "train.csv")
    val_rows = load_rows(manifests / "val.csv")
    test_rows = load_rows(manifests / "test.csv")
    pool_rows = load_rows(manifests / "pool.csv")

    all_j = jids(all_rows)
    train_j = jids(train_rows)
    val_j = jids(val_rows)
    test_j = jids(test_rows)
    pool_j = jids(pool_rows)

    must(train_j.isdisjoint(val_j), f"{family_name}: train and val overlap")
    must(train_j.isdisjoint(test_j), f"{family_name}: train and test overlap")
    must(val_j.isdisjoint(test_j), f"{family_name}: val and test overlap")
    must(pool_j == train_j | val_j, f"{family_name}: pool must equal train U val")
    must(all_j == train_j | val_j | test_j, f"{family_name}: all must equal train U val U test")

    for row in all_rows:
        has_o = str(row["has_O"]).lower() == "true"
        has_n = str(row["has_N"]).lower() == "true"
        is_nitride = str(row["is_nitride"]).lower() == "true"
        if family_name == "oxide":
            must(has_o, f"oxide family contains non-O material: {row['jid']}")
        if family_name == "nitride":
            must(is_nitride, f"nitride family contains bad member: {row['jid']}")
            must(has_n and not has_o, f"nitride family contains oxygen: {row['jid']}")

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        counts = summary.get("counts", {})
        must(int(counts.get("all", -1)) == len(all_rows), f"{family_name}: summary all count mismatch")
        must(int(counts.get("train", -1)) == len(train_rows), f"{family_name}: summary train count mismatch")
        must(int(counts.get("val", -1)) == len(val_rows), f"{family_name}: summary val count mismatch")
        must(int(counts.get("test", -1)) == len(test_rows), f"{family_name}: summary test count mismatch")
        must(int(counts.get("pool", -1)) == len(pool_rows), f"{family_name}: summary pool count mismatch")

    return {
        "all": len(all_rows),
        "train": len(train_rows),
        "val": len(val_rows),
        "test": len(test_rows),
        "pool": len(pool_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stage-2 oxide/nitride outputs.")
    parser.add_argument("--root", type=Path, required=True, help="Stage-2 output root directory.")
    args = parser.parse_args()

    oxide_counts = validate_family(args.root, "oxide")
    nitride_counts = validate_family(args.root, "nitride")
    print(json.dumps({"status": "ok", "oxide": oxide_counts, "nitride": nitride_counts}, indent=2))


if __name__ == "__main__":
    main()
