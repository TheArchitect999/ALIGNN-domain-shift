
import argparse
import csv
import io
import json
import os
import urllib.request
import zipfile
from collections import defaultdict
from typing import Iterable, Tuple

def normalize_split_name(name: str) -> str:
    n = str(name).strip().lower()
    mapping = {
        "train": "train",
        "trn": "train",
        "training": "train",
        "val": "val",
        "valid": "val",
        "validation": "val",
        "dev": "val",
        "test": "test",
        "tst": "test",
        "testing": "test",
    }
    if n not in mapping:
        raise ValueError(f"Unknown split name: {name}")
    return mapping[n]

def _extract_single_id(item):
    if item is None:
        return None
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("jid", "id", "jid_id", "file_id", "name"):
            if key in item and item[key] is not None:
                return str(item[key])
        return None
    if isinstance(item, (list, tuple)):
        # common leaderboard pattern: [jid, target] or [jid, ...]
        if not item:
            return None
        first = item[0]
        if isinstance(first, (str, int, float)):
            return str(first)
        if isinstance(first, dict):
            return _extract_single_id(first)
        return None
    if isinstance(item, (int, float)):
        return str(item)
    return None

def extract_ids(payload) -> Iterable[str]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        # common leaderboard formats:
        # 1) {jid: target, ...}
        # 2) {"ids": [...]} / {"train": ...} etc.
        # Prefer plain jid->value maps when keys look like JIDs.
        keys = list(payload.keys())
        if keys and any(str(k).startswith(("JVASP-", "jid")) for k in keys):
            return [str(k) for k in keys]
        out = []
        for key in ("ids", "jids", "data", "entries", "samples"):
            if key in payload:
                out.extend(extract_ids(payload[key]))
        return out
    if isinstance(payload, list):
        out = []
        for item in payload:
            jid = _extract_single_id(item)
            if jid is None:
                raise ValueError(f"Cannot extract id from list item: {item}")
            out.append(jid)
        return out
    raise ValueError(f"Unsupported split payload type: {type(payload)}")

def load_benchmark_json_from_zip(url: str):
    with urllib.request.urlopen(url) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob), "r") as zf:
        members = zf.namelist()
        json_members = [m for m in members if m.endswith(".json")]
        if not json_members:
            raise RuntimeError("No .json file found inside benchmark zip.")
        json_members = sorted(json_members, key=lambda x: (len(x), x))
        with zf.open(json_members[0]) as fh:
            return json.load(fh), json_members[0]

def main():
    parser = argparse.ArgumentParser(
        description="Create jid,split manifest from a JARVIS-Leaderboard benchmark json.zip"
    )
    parser.add_argument("--url", required=True, help="Raw benchmark json.zip URL")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--conflict-policy",
        choices=["exclude", "first"],
        default="exclude",
        help="How to handle IDs that appear in multiple splits. Default: exclude",
    )
    args = parser.parse_args()

    data, inner_name = load_benchmark_json_from_zip(args.url)
    if not isinstance(data, dict):
        raise RuntimeError("Top-level benchmark JSON is not a dict.")

    split_to_ids = {"train": [], "val": [], "test": []}
    raw_counts = {"train": 0, "val": 0, "test": 0}

    for raw_split in ("train", "val", "validation", "test"):
        if raw_split not in data:
            continue
        split = normalize_split_name(raw_split)
        ids = list(extract_ids(data[raw_split]))
        split_to_ids[split].extend(ids)
        raw_counts[split] += len(ids)

    id_to_splits = defaultdict(set)
    first_seen = {}
    for split in ("train", "val", "test"):
        for jid in split_to_ids[split]:
            id_to_splits[jid].add(split)
            if jid not in first_seen:
                first_seen[jid] = split

    conflicts = {jid: sorted(list(splits)) for jid, splits in id_to_splits.items() if len(splits) > 1}

    rows = {}
    if args.conflict_policy == "exclude":
        for jid, splits in id_to_splits.items():
            if len(splits) == 1:
                rows[jid] = next(iter(splits))
    else:
        rows = dict(first_seen)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["jid", "split"])
        for jid in sorted(rows):
            writer.writerow([jid, rows[jid]])

    conflicts_path = os.path.splitext(os.path.abspath(args.out))[0] + "_conflicts.csv"
    with open(conflicts_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["jid", "splits"])
        for jid in sorted(conflicts):
            writer.writerow([jid, "|".join(conflicts[jid])])

    filtered_counts = {"train": 0, "val": 0, "test": 0}
    for split in rows.values():
        filtered_counts[split] += 1

    print(json.dumps(
        {
            "inner_json": inner_name,
            "raw_counts": raw_counts,
            "n_unique_ids_before_conflict_filter": len(id_to_splits),
            "n_conflicting_ids": len(conflicts),
            "conflict_policy": args.conflict_policy,
            "filtered_counts": filtered_counts,
            "n_rows_written": len(rows),
            "out": os.path.abspath(args.out),
            "conflicts_csv": conflicts_path,
        },
        indent=2,
    ))

if __name__ == "__main__":
    main()
