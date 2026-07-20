from __future__ import annotations

import csv
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "te": "test",
    "holdout": "test",
}

SPLIT_CANDIDATE_KEYS = [
    "split",
    "dataset_split",
    "split_label",
    "subset",
    "partition",
    "set",
    "train_val_test",
    "tvt",
]

TARGET_KEY_ALIASES = [
    "formation_energy_peratom",
    "formation_energy",
    "e_form",
    "eform",
    "form_energy",
]


def canonicalize_split(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        key = value.strip().lower()
        return SPLIT_ALIASES.get(key)
    return None


def sanitize_id(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def normalize_target(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"", "na", "nan", "none", "null"}:
            return None
        try:
            value = float(stripped)
        except ValueError:
            return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def build_hill_formula(elements: Sequence[str]) -> str:
    counts = Counter(elements)
    if not counts:
        return ""
    parts: List[Tuple[str, int]] = []
    if "C" in counts:
        parts.append(("C", counts.pop("C")))
    if "H" in counts:
        parts.append(("H", counts.pop("H")))
    for el in sorted(counts):
        parts.append((el, counts[el]))
    out = []
    for el, n in parts:
        out.append(el if n == 1 else f"{el}{n}")
    return "".join(out)


def extract_elements_from_record(record: Mapping[str, Any]) -> List[str]:
    atoms = record.get("atoms")
    if isinstance(atoms, Mapping):
        elements = atoms.get("elements")
        if isinstance(elements, Sequence) and not isinstance(elements, (str, bytes)):
            return [str(x) for x in elements]
    elements = record.get("elements")
    if isinstance(elements, Sequence) and not isinstance(elements, (str, bytes)):
        return [str(x) for x in elements]
    raise KeyError("Could not find element list in record['atoms']['elements'] or record['elements']")


def classify_family(elements: Sequence[str]) -> Dict[str, bool]:
    s = set(elements)
    has_o = "O" in s
    has_n = "N" in s
    return {
        "has_O": has_o,
        "has_N": has_n,
        "is_oxide": has_o,
        "is_nitride": has_n and not has_o,
        "is_oxynitride": has_o and has_n,
    }


def get_target_value(record: Mapping[str, Any], target_key: str, target_aliases: Sequence[str]) -> Tuple[Optional[float], Optional[str]]:
    keys = [target_key] + [k for k in target_aliases if k != target_key]
    for key in keys:
        if key in record:
            value = normalize_target(record.get(key))
            return value, key
    return None, None


def jid_from_record(record: Mapping[str, Any]) -> str:
    return str(record.get("jid") or record.get("id") or record.get("material_id") or "").strip()


def _non_null_top_level_count(record: Mapping[str, Any]) -> int:
    count = 0
    for value in record.values():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        count += 1
    return count


def _record_quality(record: Mapping[str, Any], target_key: str, target_aliases: Sequence[str]) -> Tuple[int, int, int, int]:
    target_val, _ = get_target_value(record, target_key=target_key, target_aliases=target_aliases)
    try:
        elements = extract_elements_from_record(record)
        has_elements = 1 if elements else 0
    except Exception:
        has_elements = 0
    has_atoms_mapping = 1 if isinstance(record.get("atoms"), Mapping) else 0
    return (
        1 if target_val is not None else 0,
        has_elements,
        has_atoms_mapping,
        _non_null_top_level_count(record),
    )


def dedupe_records_by_jid(
    records: Sequence[Mapping[str, Any]],
    target_key: str,
    target_aliases: Sequence[str],
    max_examples: int = 20,
) -> Tuple[List[Mapping[str, Any]], Dict[str, Any]]:
    """Collapse duplicate JIDs deterministically.

    JARVIS dft_3d_2021 can contain a small number of repeated JIDs. For Stage 2 we need a
    single record per JID, so we keep the highest-quality record per JID using a deterministic
    score: has target, has elements, has atoms mapping, then number of non-null top-level keys.
    Ties keep the first-seen record.
    """
    chosen: Dict[str, Tuple[int, Tuple[int, int, int, int], Mapping[str, Any]]] = {}
    order: List[str] = []
    duplicate_counts: Counter[str] = Counter()
    duplicate_examples: Dict[str, Any] = {}

    for idx, rec in enumerate(records):
        jid = jid_from_record(rec)
        if not jid:
            continue
        score = _record_quality(rec, target_key=target_key, target_aliases=target_aliases)
        if jid not in chosen:
            chosen[jid] = (idx, score, rec)
            order.append(jid)
            continue

        duplicate_counts[jid] += 1
        prev_idx, prev_score, prev_rec = chosen[jid]
        if score > prev_score:
            chosen[jid] = (idx, score, rec)
            kept_index = idx
            kept_score = score
        else:
            kept_index = prev_idx
            kept_score = prev_score

        if len(duplicate_examples) < max_examples:
            duplicate_examples.setdefault(
                jid,
                {
                    "seen_indices": [prev_idx],
                    "scores": [list(prev_score)],
                },
            )
            duplicate_examples[jid]["seen_indices"].append(idx)
            duplicate_examples[jid]["scores"].append(list(score))
            duplicate_examples[jid]["kept_index"] = kept_index
            duplicate_examples[jid]["kept_score"] = list(kept_score)

    unique_records = [chosen[jid][2] for jid in order]
    summary = {
        "n_input_records": len(records),
        "n_unique_jids": len(unique_records),
        "n_duplicate_jids": len(duplicate_counts),
        "duplicate_jid_total_extra_rows": int(sum(duplicate_counts.values())),
        "duplicate_examples": duplicate_examples,
    }
    return unique_records, summary


def infer_split_map_from_records(records: Sequence[Mapping[str, Any]]) -> Tuple[Optional[Dict[str, str]], Optional[str]]:
    for candidate in SPLIT_CANDIDATE_KEYS:
        mapping: Dict[str, str] = {}
        ok = True
        for rec in records:
            jid = str(rec.get("jid") or rec.get("id") or rec.get("material_id") or "")
            if not jid:
                ok = False
                break
            split = canonicalize_split(rec.get(candidate))
            if split is None:
                ok = False
                break
            mapping[jid] = split
        if ok and mapping:
            return mapping, candidate
    # boolean-style keys
    for keys in [
        ("is_train", "is_val", "is_test"),
        ("train", "val", "test"),
    ]:
        mapping = {}
        ok = True
        for rec in records:
            jid = str(rec.get("jid") or rec.get("id") or rec.get("material_id") or "")
            if not jid:
                ok = False
                break
            flags = []
            for split_name, k in zip(("train", "val", "test"), keys):
                if k in rec and bool(rec.get(k)):
                    flags.append(split_name)
            if len(flags) != 1:
                ok = False
                break
            mapping[jid] = flags[0]
        if ok and mapping:
            return mapping, ",".join(keys)
    return None, None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_split_map(path: Path) -> Dict[str, str]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        obj = load_json(path)
        return _normalize_split_json(obj)
    if suffix in {".csv", ".tsv"}:
        delim = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delim)
            if reader.fieldnames is None:
                raise ValueError(f"No header row found in split file: {path}")
            field_map = {name.lower(): name for name in reader.fieldnames}
            jid_col = None
            for candidate in ("jid", "id", "material_id"):
                if candidate in field_map:
                    jid_col = field_map[candidate]
                    break
            split_col = None
            for candidate in ("split", "dataset_split", "subset", "partition", "set"):
                if candidate in field_map:
                    split_col = field_map[candidate]
                    break
            if jid_col is None or split_col is None:
                raise ValueError(
                    f"Split CSV must contain jid/id/material_id and split columns. Found: {reader.fieldnames}"
                )
            out: Dict[str, str] = {}
            for row in reader:
                jid = str(row[jid_col]).strip()
                split = canonicalize_split(row[split_col])
                if not jid or split is None:
                    raise ValueError(f"Bad row in split file {path}: {row}")
                out[jid] = split
            return out
    raise ValueError(f"Unsupported split manifest format: {path}")


def _normalize_split_json(obj: Any) -> Dict[str, str]:
    if isinstance(obj, Mapping):
        keys_lower = {str(k).lower() for k in obj.keys()}
        if keys_lower & {"train", "val", "valid", "validation", "test"}:
            out: Dict[str, str] = {}
            for raw_key, values in obj.items():
                split = canonicalize_split(raw_key)
                if split is None:
                    continue
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    raise ValueError("Split JSON lists must be arrays of JIDs")
                for jid in values:
                    out[str(jid)] = split
            return out
        out = {}
        for jid, split_val in obj.items():
            split = canonicalize_split(split_val)
            if split is None:
                raise ValueError(f"Bad split value for {jid}: {split_val}")
            out[str(jid)] = split
        return out
    if isinstance(obj, Sequence):
        out = {}
        for item in obj:
            if not isinstance(item, Mapping):
                raise ValueError("Split JSON list entries must be objects with jid and split")
            jid = str(item.get("jid") or item.get("id") or item.get("material_id") or "").strip()
            split = canonicalize_split(item.get("split") or item.get("dataset_split") or item.get("subset"))
            if not jid or split is None:
                raise ValueError(f"Bad entry in split JSON: {item}")
            out[jid] = split
        return out
    raise ValueError("Unsupported JSON split manifest format")


def generate_split_map(jids: Sequence[str], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, str]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")
    ordered = sorted(set(jids))
    rng = random.Random(seed)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_train = min(n_train, n)
    n_val = min(n_val, max(0, n - n_train))
    out: Dict[str, str] = {}
    for idx, jid in enumerate(ordered):
        if idx < n_train:
            out[jid] = "train"
        elif idx < n_train + n_val:
            out[jid] = "val"
        else:
            out[jid] = "test"
    return out


@dataclass
class Stage2BuildResult:
    global_rows: List[Dict[str, Any]]
    oxide_rows: List[Dict[str, Any]]
    nitride_rows: List[Dict[str, Any]]
    split_source: str
    split_map: Dict[str, str]
    target_key_used: str
    schema_report: Dict[str, Any]



def schema_report(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    key_counts: Counter[str] = Counter()
    candidate_split_key_counts: Dict[str, int] = {}
    jid_counts: Counter[str] = Counter()
    for rec in records:
        for k in rec.keys():
            key_counts[str(k)] += 1
        jid = jid_from_record(rec)
        if jid:
            jid_counts[jid] += 1
    for candidate in SPLIT_CANDIDATE_KEYS:
        candidate_split_key_counts[candidate] = sum(1 for rec in records if candidate in rec)
    bool_candidates = {
        key: sum(1 for rec in records if key in rec)
        for key in ("is_train", "is_val", "is_test", "train", "val", "test")
    }
    duplicate_examples = {jid: count for jid, count in list(sorted((j, c) for j, c in jid_counts.items() if c > 1))[:20]}
    return {
        "n_records": len(records),
        "n_unique_jids_detected": len(jid_counts),
        "n_duplicate_jids_detected": len([1 for _jid, c in jid_counts.items() if c > 1]),
        "duplicate_jid_examples": duplicate_examples,
        "top_level_keys_sorted": sorted(key_counts.keys()),
        "candidate_split_key_counts": candidate_split_key_counts,
        "candidate_boolean_split_key_counts": bool_candidates,
    }



def build_stage2_rows(
    records: Sequence[Mapping[str, Any]],
    target_key: str,
    split_map: Mapping[str, str],
    target_aliases: Sequence[str] = TARGET_KEY_ALIASES,
    keep_oxynitrides_in_oxide: bool = True,
) -> Stage2BuildResult:
    global_rows: List[Dict[str, Any]] = []
    oxide_rows: List[Dict[str, Any]] = []
    nitride_rows: List[Dict[str, Any]] = []
    target_key_used_any: Optional[str] = None

    seen_jids: set[str] = set()
    for rec in records:
        jid = str(rec.get("jid") or rec.get("id") or rec.get("material_id") or "").strip()
        if not jid:
            continue
        if jid in seen_jids:
            raise ValueError(f"Duplicate jid encountered: {jid}")
        seen_jids.add(jid)

        elements = extract_elements_from_record(rec)
        fam = classify_family(elements)
        target_val, actual_target_key = get_target_value(rec, target_key=target_key, target_aliases=target_aliases)
        split = split_map.get(jid)

        row = {
            "jid": jid,
            "split": split,
            "target": target_val,
            "target_key_used": actual_target_key,
            "filename": f"POSCAR-{sanitize_id(jid)}.vasp",
            "formula": str(rec.get("formula") or build_hill_formula(elements)),
            "n_atoms": len(elements),
            "elements": ";".join(elements),
            "unique_elements": ";".join(sorted(set(elements))),
            "has_O": fam["has_O"],
            "has_N": fam["has_N"],
            "is_oxide": fam["is_oxide"],
            "is_nitride": fam["is_nitride"],
            "is_oxynitride": fam["is_oxynitride"],
        }
        global_rows.append(row)
        if target_val is None or split is None:
            continue
        if target_key_used_any is None and actual_target_key is not None:
            target_key_used_any = actual_target_key
        if fam["is_oxide"] and (keep_oxynitrides_in_oxide or not fam["is_oxynitride"]):
            oxide_rows.append(dict(row))
        if fam["is_nitride"]:
            nitride_rows.append(dict(row))

    if target_key_used_any is None:
        target_key_used_any = target_key

    return Stage2BuildResult(
        global_rows=global_rows,
        oxide_rows=oxide_rows,
        nitride_rows=nitride_rows,
        split_source="",
        split_map=dict(split_map),
        target_key_used=target_key_used_any,
        schema_report=schema_report(records),
    )



def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)



def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)



def group_by_split(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for row in rows:
        split = canonicalize_split(row.get("split"))
        if split in out:
            out[split].append(dict(row))
    return out



def summary_for_family(rows: Sequence[Mapping[str, Any]], family_name: str, split_source: str, dataset_key: str, target_key: str) -> Dict[str, Any]:
    grouped = group_by_split(rows)
    pool = grouped["train"] + grouped["val"]
    n_oxynitrides = sum(bool(row.get("is_oxynitride")) for row in rows)
    return {
        "family": family_name,
        "dataset_key": dataset_key,
        "target_key": target_key,
        "split_source": split_source,
        "counts": {
            "all": len(rows),
            "train": len(grouped["train"]),
            "val": len(grouped["val"]),
            "test": len(grouped["test"]),
            "pool": len(pool),
        },
        "oxynitride_count_in_family": n_oxynitrides,
    }



def split_manifest_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped = group_by_split(rows)
    grouped["pool"] = grouped["train"] + grouped["val"]
    grouped["all"] = [dict(r) for r in rows]
    return grouped



def write_structure_files(rows: Sequence[Mapping[str, Any]], records_by_jid: Mapping[str, Mapping[str, Any]], structures_dir: Path) -> None:
    try:
        from jarvis.core.atoms import Atoms
    except ImportError as exc:
        raise ImportError(
            "jarvis-tools is required to write POSCAR files. Install jarvis-tools or run with --no-write-structures."
        ) from exc
    structures_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        jid = str(row["jid"])
        filename = str(row["filename"])
        rec = records_by_jid[jid]
        atoms = Atoms.from_dict(rec["atoms"])
        atoms.write_poscar(str(structures_dir / filename))



def make_id_prop_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([row["filename"], row["target"]])



def materialize_alignn_root(
    manifest_rows: Sequence[Mapping[str, Any]],
    structures_dir: Path,
    outdir: Path,
    link_mode: str = "auto",
) -> None:
    import shutil
    import os

    outdir.mkdir(parents=True, exist_ok=True)
    materialized_rows = list(manifest_rows)
    for row in materialized_rows:
        src = structures_dir / str(row["filename"])
        dst = outdir / str(row["filename"])
        if not src.is_file():
            raise FileNotFoundError(f"Missing structure file: {src}")
        if dst.exists():
            dst.unlink()
        chosen = link_mode
        if chosen == "auto":
            chosen = "symlink" if os.name != "nt" else "copy"
        if chosen == "symlink":
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)
        elif chosen == "copy":
            shutil.copy2(src, dst)
        else:
            raise ValueError(f"Unknown link_mode: {link_mode}")
    make_id_prop_csv(materialized_rows, outdir / "id_prop.csv")
    write_csv(materialized_rows, outdir / "manifest.csv")



def records_by_jid(records: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for rec in records:
        jid = jid_from_record(rec)
        if jid and jid not in out:
            out[jid] = rec
    return out
