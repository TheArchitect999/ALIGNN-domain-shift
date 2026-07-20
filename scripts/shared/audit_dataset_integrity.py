#!/usr/bin/env python3
"""Produce the Section 2.3B dataset-integrity evidence package.

This script is intentionally read-only with respect to datasets and results.  It
writes only inside the approved Section 2.3B evidence directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = Path(".")
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
EXPECTED_BRANCH = "main"

EVIDENCE_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3B_dataset_integrity"
)
EVIDENCE = ROOT / EVIDENCE_REL
A_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3A_checkpoint_provenance"
)
A_DIR = ROOT / A_REL

PRODUCER_REL = Path("scripts/shared/audit_dataset_integrity.py")
VALIDATOR_REL = Path("scripts/shared/validate_dataset_integrity.py")
SECTION_B_SCRIPT_RELS = {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}

SECTION_A_MANIFEST_SHA = (
    "f815f80cbe705bebcb19d37b484031f74a071859686dda2001c9a56f888769be"
)

EXPECTED_INPUT_SHA = {
    "data/manifests/dft_3d_formation_energy_peratom_splits.csv": "82e31f229775a25fb96794a8daa8b1005b264a68ed7a55e3d4be90d0a7badc79",
    "data/manifests/dft_3d_formation_energy_peratom_splits_conflicts.csv": "151b56a1c93712772dc861d0f68c5773c4a78f7054a48343c23323181e42a776",
    "data/diagnostics/global_record_catalog.csv": "c38df88550a1214d8cf7622c994df655a8f5db023cefccc7e6859671d58759df",
    "data/diagnostics/global_split_manifest.json": "bbbacf67fafdcb948c326663ca7927131af1d521374dc7d6eb88ce09ec5edbe2",
    "data/diagnostics/schema_report.json": "e08dccc9e2daa95e5aae8e8530f8ce585c47d8ddb32ce81effc21f97779aad1a",
}

GLOBAL_INPUTS = [
    "data/manifests/dft_3d_formation_energy_peratom_splits.csv",
    "data/manifests/dft_3d_formation_energy_peratom_splits_conflicts.csv",
    "data/diagnostics/global_record_catalog.csv",
    "data/diagnostics/global_split_manifest.json",
    "data/diagnostics/schema_report.json",
]
FAMILY_INPUTS = [
    f"data/{family}/manifests/{name}.csv"
    for family in ("oxide", "nitride")
    for name in ("all", "train", "val", "test", "pool")
] + [
    f"data/{family}/summaries/summary.json"
    for family in ("oxide", "nitride")
]
PROCEDURAL_INPUTS = [
    "scripts/dataset/family_dataset_lib.py",
    "scripts/dataset/build_family_datasets.py",
    "scripts/dataset/validate_family_datasets.py",
    "scripts/dataset/make_split_manifest_from_benchmark.py",
    "scripts/dataset/materialize_alignn_root.py",
    "scripts/shared/prepare_baseline_finetune_dataset.py",
]
OFFICIAL_IDS_REL = (
    A_REL / "source_artifacts/official_ids_train_val_test.json"
).as_posix()
AUTHORIZED_FIXED_INPUTS = GLOBAL_INPUTS + FAMILY_INPUTS + PROCEDURAL_INPUTS + [
    OFFICIAL_IDS_REL
]

FAMILIES = ("oxide", "nitride")
N_VALUES = (10, 50, 100, 200, 500, 1000)
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_FAMILY_COUNTS = {
    "oxide": {"all": 14991, "train": 11960, "val": 1547, "test": 1484, "pool": 13507},
    "nitride": {"all": 2288, "train": 1837, "val": 209, "test": 242, "pool": 2046},
}
EXPECTED_DUPLICATES = {
    "JVASP-100669": 2,
    "JVASP-113961": 2,
    "JVASP-116461": 3,
    "JVASP-96735": 6,
    "JVASP-97311": 3,
}
EXPECTED_UNSPLIT = {"JVASP-113961", "JVASP-116461", "JVASP-1375"}
JID_RE = re.compile(r"JVASP-[0-9]+")

ROOT_RE = re.compile(
    r"^results/protocol_(?P<set>[12])/(?P<family>oxide|nitride)/"
    r"N(?P<N>10|50|100|200|500|1000)_seed(?P<seed>[0-4])/dataset_root/"
    r"(?P<name>split_manifest\.json|id_prop\.csv)$"
)
RUN_RE = re.compile(
    r"^results/protocol_(?P<set>[123])/(?P<family>oxide|nitride)/"
    r"N(?P<N>10|50|100|200|500|1000)_seed(?P<seed>[0-4])/"
    r"(?P<method>finetune_last2|train_alignn_from_scratch|"
    r"finetune_last2_epochs100_bs32_lr5e5|"
    r"train_alignn_from_scratch_epochs100_bs32_lr5e5)/ids_train_val_test\.json$"
)
LEGACY_RE = re.compile(
    r"^results/protocol_1/(?P<family>oxide|nitride)/"
    r"N(?P<N>10|50|100|200|500|1000)_seed(?P<seed>[0-2])/"
    r"finetune_last2/ids_train_val_test\.json$"
)

FORBIDDEN_FRAGMENTS = (
    "archived_submission_materials",
    "finetune_last2_reproduction_rerun",
    "reproduction_rerun",
    "__pycache__",
)
FORBIDDEN_COMPONENTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "cache",
    "temp",
    "tmp",
}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def reject_path(relative: str | Path) -> None:
    value = Path(relative).as_posix()
    require(not value.startswith("/"), f"Absolute path rejected: {value}")
    require(".." not in Path(value).parts, f"Traversal rejected: {value}")
    require(
        value != "domain_shift-alignn-domain-shift"
        and not value.startswith("domain_shift-alignn-domain-shift/"),
        f"Nested checkout rejected: {value}",
    )
    require(
        not any(fragment in value for fragment in FORBIDDEN_FRAGMENTS),
        f"Forbidden path rejected: {value}",
    )
    require(
        not any(part.lower() in FORBIDDEN_COMPONENTS for part in Path(value).parts),
        f"Temporary/cache path rejected: {value}",
    )
    require(
        not value.endswith((".pyc", ".pyo", ".tmp", ".temp", "~")),
        f"Temporary/compiled path rejected: {value}",
    )


def authorize_scientific_input(relative: str | Path) -> None:
    """Reject every dataset/result input outside the frozen Section 2.3B allowlist."""
    value = Path(relative).as_posix()
    reject_path(value)
    run_match = RUN_RE.fullmatch(value)
    config_authorized = value.endswith("/config.json") and RUN_RE.fullmatch(
        value[: -len("config.json")] + "ids_train_val_test.json"
    )
    family_structure = re.fullmatch(
        r"data/(oxide|nitride)/structures/POSCAR-JVASP-[0-9]+\.vasp",
        value,
    )
    dataset_root_structure = re.fullmatch(
        r"results/protocol_[12]/(oxide|nitride)/"
        r"N(10|50|100|200|500|1000)_seed[0-4]/dataset_root/"
        r"POSCAR-JVASP-[0-9]+\.vasp",
        value,
    )
    require(
        value in AUTHORIZED_FIXED_INPUTS
        or ROOT_RE.fullmatch(value)
        or run_match
        or config_authorized
        or LEGACY_RE.fullmatch(value)
        or family_structure
        or dataset_root_structure,
        f"Unapproved dataset/result input rejected: {value}",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_json(name: str, value: Any) -> None:
    (EVIDENCE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(name: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with (EVIDENCE / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_status() -> list[dict[str, str]]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    rows: list[dict[str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        text = item.decode("utf-8", errors="surrogateescape")
        rows.append({"status": text[:2], "path": text[3:]})
    return rows


def status_category(path: str) -> str:
    if "finetune_last2_reproduction_rerun/" in path:
        return "rerun_staging_files"
    if path.startswith("configs/protocol_1/finetune_reproduction_rerun/"):
        return "rerun_configurations"
    if path.startswith("results/summaries/protocol_1/finetune/reproduction_rerun/"):
        return "rerun_summary_controls"
    if path in {
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
    }:
        return "governance_session"
    if path == "domain_shift-alignn-domain-shift" or path.startswith("domain_shift-alignn-domain-shift/"):
        return "nested_checkout"
    if path.startswith(A_REL.as_posix() + "/") or path in {
        "scripts/shared/audit_checkpoint_provenance.py",
        "scripts/shared/validate_checkpoint_provenance.py",
    }:
        return "section2_3A_outputs"
    if path.startswith(EVIDENCE_REL.as_posix() + "/") or path in SECTION_B_SCRIPT_RELS:
        return "section2_3B_outputs"
    return "ambiguous"


def tracked_record(relative: str, role: str, authority: str) -> dict[str, Any]:
    authorize_scientific_input(relative)
    path = ROOT / relative
    require(path.is_file(), f"Required input missing: {relative}")
    tracked = bool(git("ls-files", "--", relative).strip())
    blob = git("rev-parse", f"HEAD:{relative}").strip() if tracked else None
    stat = path.stat()
    row_count: int | None = None
    schema: Any = None
    if path.suffix == ".csv":
        fields, rows = read_csv(path)
        row_count = len(rows)
        schema = fields
    elif path.suffix == ".json":
        payload = read_json(path)
        if isinstance(payload, dict):
            schema = sorted(payload)
            if relative == OFFICIAL_IDS_REL:
                row_count = sum(len(payload[key]) for key in payload)
            elif relative.endswith("global_split_manifest.json"):
                row_count = len(payload)
            else:
                row_count = 1
        elif isinstance(payload, list):
            row_count = len(payload)
            schema = "list"
    else:
        row_count = len(path.read_text(errors="replace").splitlines())
        schema = "source_lines"
    return {
        "repository_relative_path": relative,
        "absolute_path": str(path.resolve()),
        "git_status": "tracked" if tracked else "untracked_authorized_section2_3A",
        "git_blob": blob,
        "size_bytes": stat.st_size,
        "modification_time": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "sha256": sha256(path),
        "row_count": row_count,
        "schema": json.dumps(schema, sort_keys=True),
        "intended_role": role,
        "authority_level": authority,
    }


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def expected_val_count(n_value: int) -> int:
    n_val = max(5, int(round(0.1 * n_value)))
    if n_val >= n_value:
        n_val = max(1, n_value // 5)
    return n_val


def split_stats(payload: dict[str, list[str]]) -> dict[str, Any]:
    keys = ("id_train", "id_val", "id_test")
    lists = {key: payload[key] for key in keys}
    sets = {key: set(lists[key]) for key in keys}
    return {
        "n_train": len(lists["id_train"]),
        "n_val": len(lists["id_val"]),
        "n_test": len(lists["id_test"]),
        "unique_train": len(sets["id_train"]),
        "unique_val": len(sets["id_val"]),
        "unique_test": len(sets["id_test"]),
        "train_val_overlap": len(sets["id_train"] & sets["id_val"]),
        "train_test_overlap": len(sets["id_train"] & sets["id_test"]),
        "val_test_overlap": len(sets["id_val"] & sets["id_test"]),
        "malformed_jid_count": sum(
            JID_RE.fullmatch(jid) is None for key in keys for jid in lists[key]
        ),
    }


def main() -> None:
    require(ROOT.resolve() == EXPECTED_ROOT, f"Unexpected root: {ROOT.resolve()}")
    require(git("rev-parse", "HEAD").strip() == EXPECTED_HEAD, "Unexpected HEAD")
    require(git("branch", "--show-current").strip() == EXPECTED_BRANCH, "Unexpected branch")
    section08 = ROOT / "results/derived_evidence/final_paper_factory/archived_submission_materials"
    require(not section08.exists(), "Section 08 exists; refusing to inspect")
    require(
        not git("ls-files", "--", "results/derived_evidence/final_paper_factory/archived_submission_materials").strip(),
        "Section 08 is tracked",
    )

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    status = parse_status()
    baseline = [row for row in status if status_category(row["path"]) != "section2_3B_outputs"]
    categories = Counter(status_category(row["path"]) for row in baseline)
    expected_categories = Counter(
        {
            "rerun_staging_files": 504,
            "rerun_configurations": 36,
            "rerun_summary_controls": 2,
            "governance_session": 3,
            "section2_3A_outputs": 32,
            "nested_checkout": 1,
        }
    )
    require(len(baseline) == 578, f"Unexpected baseline count: {len(baseline)}")
    require(categories == expected_categories, f"Unexpected baseline categories: {categories}")
    require(all(row["status"] == "??" for row in baseline), "Tracked/staged baseline change found")

    protected = []
    for relative in (
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
        (A_REL / "section2_3A_evidence.sha256").as_posix(),
        (A_REL / "section2_3A_validation.json").as_posix(),
    ):
        path = ROOT / relative
        protected.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    preflight = {
        "captured_at": now_utc(),
        "repository_root": str(ROOT),
        "branch": EXPECTED_BRANCH,
        "head": EXPECTED_HEAD,
        "last_three_commits": git("log", "-3", "--format=%H %s").splitlines(),
        "section08_absent": True,
        "section08_tracked": False,
        "nested_checkout_excluded": True,
        "baseline_status_count": len(baseline),
        "baseline_tracked_path_count": sum(row["status"] != "??" for row in baseline),
        "baseline_worktree_modified_count": sum(
            row["status"] != "??" and row["status"][1] not in {" ", "?"}
            for row in baseline
        ),
        "baseline_staged_count": sum(
            row["status"] != "??" and row["status"][0] not in {" ", "?"}
            for row in baseline
        ),
        "baseline_tracked_or_staged_count": sum(row["status"] != "??" for row in baseline),
        "baseline_untracked_count": sum(row["status"] == "??" for row in baseline),
        "baseline_status_categories": dict(sorted(categories.items())),
        "baseline_status_entries": baseline,
        "protected_files_before": protected,
    }
    write_json("repository_preflight.json", preflight)

    # Authoritative input inventory.
    roles = {
        **{relative: ("Global dataset/split authority", "dataset authority") for relative in GLOBAL_INPUTS},
        **{relative: ("Fixed family manifest or summary", "dataset authority") for relative in FAMILY_INPUTS},
        **{relative: ("Dataset-construction or validation procedure", "procedural authority") for relative in PROCEDURAL_INPUTS},
        OFFICIAL_IDS_REL: ("Historical split-ID multiplicity cross-check", "official provenance cross-check only"),
    }
    inventory = [tracked_record(relative, *roles[relative]) for relative in AUTHORIZED_FIXED_INPUTS]
    for relative, expected_sha in EXPECTED_INPUT_SHA.items():
        actual = next(row["sha256"] for row in inventory if row["repository_relative_path"] == relative)
        require(actual == expected_sha, f"Authoritative input SHA mismatch: {relative}")
    write_csv(
        "authoritative_dataset_input_inventory.csv",
        inventory,
        [
            "repository_relative_path", "absolute_path", "git_status", "git_blob",
            "size_bytes", "modification_time", "sha256", "row_count", "schema",
            "intended_role", "authority_level",
        ],
    )

    # Freeze the validation-allocation rule from its tracked implementation;
    # expected_val_count below is a direct standard-library recomputation.
    allocation_source = (ROOT / "scripts/shared/prepare_baseline_finetune_dataset.py").read_text()
    for required_line in (
        "n_val = max(5, int(round(0.1 * args.N)))",
        "if n_val >= args.N:",
        "n_val = max(1, args.N // 5)",
    ):
        require(required_line in allocation_source, f"Validation-size implementation changed: {required_line}")

    # Global catalog and split-map audit.
    catalog_fields, catalog = read_csv(ROOT / GLOBAL_INPUTS[2])
    split_fields, split_rows = read_csv(ROOT / GLOBAL_INPUTS[0])
    conflict_fields, conflict_rows = read_csv(ROOT / GLOBAL_INPUTS[1])
    split_json = read_json(ROOT / GLOBAL_INPUTS[3])
    schema_report = read_json(ROOT / GLOBAL_INPUTS[4])
    official_ids = read_json(ROOT / OFFICIAL_IDS_REL)
    require(catalog_fields == [
        "jid", "split", "target", "target_key_used", "filename", "formula", "n_atoms",
        "elements", "unique_elements", "has_O", "has_N", "is_oxide", "is_nitride", "is_oxynitride",
    ], "Unexpected global catalog schema")
    require(split_fields == ["jid", "split"], "Unexpected split CSV schema")
    require(conflict_fields == ["jid", "splits"], "Unexpected conflict CSV schema")

    catalog_ids = [row["jid"] for row in catalog]
    split_ids = [row["jid"] for row in split_rows]
    split_map = {row["jid"]: row["split"] for row in split_rows}
    catalog_by_jid = {row["jid"]: row for row in catalog}
    catalog_issues: Counter[str] = Counter()
    for row in catalog:
        jid = row["jid"]
        elements = row["elements"].split(";") if row["elements"] else []
        unique_elements = sorted(set(elements))
        has_o, has_n = "O" in unique_elements, "N" in unique_elements
        if JID_RE.fullmatch(jid) is None: catalog_issues["malformed_jid"] += 1
        try:
            target = float(row["target"])
            if not math.isfinite(target): catalog_issues["nonfinite_target"] += 1
        except ValueError:
            catalog_issues["nonfinite_target"] += 1
        if row["target_key_used"] != "formation_energy_peratom": catalog_issues["target_key"] += 1
        if row["filename"] != f"POSCAR-{jid}.vasp": catalog_issues["filename"] += 1
        if int(row["n_atoms"]) != len(elements): catalog_issues["n_atoms"] += 1
        if row["unique_elements"] != ";".join(unique_elements): catalog_issues["unique_elements"] += 1
        expected_flags = (has_o, has_n, has_o, has_n and not has_o, has_o and has_n)
        actual_flags = tuple(bool_value(row[key]) for key in ("has_O", "has_N", "is_oxide", "is_nitride", "is_oxynitride"))
        if actual_flags != expected_flags: catalog_issues["family_flags"] += 1
        if row["split"] != split_map.get(jid, ""): catalog_issues["split_mapping"] += 1
    require(len(catalog) == 55712 and len(set(catalog_ids)) == 55712, "Global catalog count/uniqueness mismatch")
    require(not catalog_issues, f"Global catalog issues: {catalog_issues}")
    require(len(split_rows) == len(set(split_ids)) == 55709, "Split-map count/uniqueness mismatch")
    require(list(split_json.items()) == [(row["jid"], row["split"]) for row in split_rows], "CSV/JSON split maps differ")
    split_counts = Counter(row["split"] for row in split_rows)
    require(split_counts == Counter({"train": 44567, "val": 5572, "test": 5570}), f"Split counts differ: {split_counts}")
    split_sets = {name: {row["jid"] for row in split_rows if row["split"] == name} for name in ("train", "val", "test")}
    require(not (split_sets["train"] & split_sets["val"]), "Global train/val overlap")
    require(not (split_sets["train"] & split_sets["test"]), "Global train/test overlap")
    require(not (split_sets["val"] & split_sets["test"]), "Global val/test overlap")
    unsplit = [row for row in catalog if not row["split"]]
    require({row["jid"] for row in unsplit} == EXPECTED_UNSPLIT, "Unexpected unsplit records")
    require(set(catalog_ids) - set(split_ids) == EXPECTED_UNSPLIT, "Catalog/split difference mismatch")
    require(
        {(row["jid"], row["splits"]) for row in conflict_rows}
        == {("JVASP-113961", "test|train"), ("JVASP-116461", "test|train")},
        "Conflict manifest mismatch",
    )

    official_counter = Counter(
        jid for key in ("id_train", "id_val", "id_test") for jid in official_ids[key]
    )
    require(set(official_ids) == {"id_train", "id_val", "id_test"}, "Official ID source schema differs")
    official_split_counts = {key: len(official_ids[key]) for key in ("id_train", "id_val", "id_test")}
    require(
        official_split_counts == {"id_train": 44578, "id_val": 5572, "id_test": 5572},
        f"Official ID split lengths differ: {official_split_counts}",
    )
    require(sum(official_split_counts.values()) == 55722, "Official flattened ID count differs")
    require(len(official_counter) == 55711, "Official unique-ID count differs")
    official_duplicate_by_split = {
        jid: {
            key: Counter(official_ids[key])[jid]
            for key in ("id_train", "id_val", "id_test")
        }
        for jid in EXPECTED_DUPLICATES
    }
    require(
        official_duplicate_by_split
        == {
            "JVASP-100669": {"id_train": 2, "id_val": 0, "id_test": 0},
            "JVASP-113961": {"id_train": 1, "id_val": 0, "id_test": 1},
            "JVASP-116461": {"id_train": 2, "id_val": 0, "id_test": 1},
            "JVASP-96735": {"id_train": 6, "id_val": 0, "id_test": 0},
            "JVASP-97311": {"id_train": 3, "id_val": 0, "id_test": 0},
        },
        f"Official per-split duplicate distribution differs: {official_duplicate_by_split}",
    )
    require(
        set(catalog_ids) - set(official_counter) == {"JVASP-1375"}
        and not (set(official_counter) - set(catalog_ids)),
        "Official IDs/catalog reconciliation differs",
    )
    duplicate_counts = {jid: count for jid, count in sorted(official_counter.items()) if count > 1}
    require(duplicate_counts == EXPECTED_DUPLICATES, f"Duplicate multiplicities differ: {duplicate_counts}")
    duplicate_meta = schema_report["duplicate_resolution"]
    require(duplicate_meta["n_input_records"] == 55723, "Original record count mismatch")
    require(duplicate_meta["n_unique_jids"] == 55712, "Deduplicated count mismatch")
    require(duplicate_meta["n_duplicate_jids"] == 5, "Duplicate-JID count mismatch")
    require(duplicate_meta["duplicate_jid_total_extra_rows"] == 11, "Excess duplicate rows mismatch")
    duplicate_rows = []
    for jid, multiplicity in EXPECTED_DUPLICATES.items():
        meta = duplicate_meta["duplicate_examples"][jid]
        require(len(meta["seen_indices"]) == multiplicity, f"Seen-index count differs: {jid}")
        require(meta["kept_index"] == meta["seen_indices"][0], f"Equal-score tie did not keep first record: {jid}")
        require(len({tuple(item) for item in meta["scores"]}) == 1, f"Expected equal duplicate quality scores: {jid}")
        require(catalog_ids.count(jid) == 1, f"Duplicate not present exactly once downstream: {jid}")
        duplicate_rows.append(
            {
                "jid": jid,
                "official_occurrence_count": multiplicity,
                "excess_duplicate_rows": multiplicity - 1,
                "kept_index": meta["kept_index"],
                "seen_indices": json.dumps(meta["seen_indices"]),
                "score_tie": len({tuple(item) for item in meta["scores"]}) == 1,
                "present_once_in_deduplicated_catalog": catalog_ids.count(jid) == 1,
                "resolution": "deterministic_first_seen_among_equal_quality_scores",
            }
        )
    require(sum(row["excess_duplicate_rows"] for row in duplicate_rows) == 11, "Duplicate arithmetic mismatch")
    write_csv(
        "duplicate_resolution_audit.csv",
        duplicate_rows,
        ["jid", "official_occurrence_count", "excess_duplicate_rows", "kept_index", "seen_indices", "score_tie", "present_once_in_deduplicated_catalog", "resolution"],
    )
    write_json(
        "duplicate_resolution_audit.json",
        {
            "original_record_count": 55723,
            "unique_jid_count_after_deduplication": 55712,
            "duplicate_jid_count": 5,
            "excess_duplicate_rows_removed": 11,
            "duplicate_multiplicities": EXPECTED_DUPLICATES,
            "official_id_source_split_lengths": official_split_counts,
            "official_id_source_flattened_count": 55722,
            "official_id_source_unique_jid_count": 55711,
            "official_duplicate_multiplicities_by_split": official_duplicate_by_split,
            "catalog_only_jid_absent_from_official_split_source": "JVASP-1375",
            "record_count_reconciliation": "55,722 frozen official split-ID entries plus catalog-only JVASP-1375 equals 55,723 original structural records; removing 11 excess duplicate rows yields 55,712 unique catalog JIDs.",
            "schema_report_top_level_scope": "The top-level n_records=55,712 and n_duplicate_jids_detected=0 describe the post-deduplication catalog; duplicate_resolution records the 55,723-row source state, five duplicate JIDs, and 11 excess rows.",
            "raw_structural_payload_locally_available": False,
            "limitation": "The original 55,723-record structural payload is not tracked locally; exact raw record contents and kept indices cannot be replayed from that payload. ID multiplicities, deterministic code, tracked resolution metadata, arithmetic, and the deduplicated catalog are mutually consistent.",
        },
    )

    catalog_check_counts = {
        name: catalog_issues.get(name, 0)
        for name in (
            "malformed_jid", "nonfinite_target", "target_key", "filename",
            "n_atoms", "unique_elements", "family_flags", "split_mapping",
        )
    }
    global_integrity = {
        "catalog_row_count": len(catalog),
        "catalog_unique_jid_count": len(set(catalog_ids)),
        "catalog_issue_counts": catalog_check_counts,
        "assigned_split_count": len(split_rows),
        "split_counts": dict(split_counts),
        "split_issue_counts": {
            "malformed_jid": sum(JID_RE.fullmatch(jid) is None for jid in split_ids),
            "duplicate_rows": len(split_ids) - len(set(split_ids)),
            "missing_or_invalid_split_label": sum(
                row["split"] not in {"train", "val", "test"} for row in split_rows
            ),
            "catalog_jids_missing_from_split_or_explicit_exception": len(
                set(catalog_ids) - set(split_ids) - EXPECTED_UNSPLIT
            ),
            "split_jids_missing_from_catalog": len(set(split_ids) - set(catalog_ids)),
        },
        "split_pairwise_overlaps": {"train_val": 0, "train_test": 0, "val_test": 0},
        "split_csv_json_exact_order_and_mapping_equal": True,
        "unassigned_count": len(unsplit),
        "unassigned_jids": sorted(EXPECTED_UNSPLIT),
        "status": "GLOBAL_DATASET_INTEGRITY_PASSED",
    }
    write_json("global_dataset_integrity.json", global_integrity)
    write_json("global_split_integrity.json", global_integrity)

    # Family-manifest audit.
    family_rows_by_name: dict[str, dict[str, list[dict[str, str]]]] = {}
    family_sets: dict[str, dict[str, set[str]]] = {}
    family_inventory: list[dict[str, Any]] = []
    structure_coverage: dict[str, Any] = {}
    oxynitrides: list[dict[str, Any]] = []
    for family in FAMILIES:
        family_rows_by_name[family] = {}
        family_sets[family] = {}
        for name in ("all", "train", "val", "test", "pool"):
            relative = f"data/{family}/manifests/{name}.csv"
            fields, rows = read_csv(ROOT / relative)
            ids = [row["jid"] for row in rows]
            family_rows_by_name[family][name] = rows
            family_sets[family][name] = set(ids)
            split_values = Counter(row["split"] for row in rows)
            family_inventory.append(
                {
                    "family": family,
                    "manifest": name,
                    "repository_relative_path": relative,
                    "sha256": sha256(ROOT / relative),
                    "row_count": len(rows),
                    "unique_jid_count": len(set(ids)),
                    "duplicate_jid_rows": len(ids) - len(set(ids)),
                    "split_counts": json.dumps(dict(split_values), sort_keys=True),
                    "schema": json.dumps(fields),
                    "status": "passed",
                }
            )
            require(len(rows) == EXPECTED_FAMILY_COUNTS[family][name], f"Family count mismatch: {family}/{name}")
            require(len(ids) == len(set(ids)), f"Family duplicates: {family}/{name}")
        sets = family_sets[family]
        require(not sets["train"] & sets["val"], f"Family train/val overlap: {family}")
        require(not sets["train"] & sets["test"], f"Family train/test overlap: {family}")
        require(not sets["val"] & sets["test"], f"Family val/test overlap: {family}")
        require(sets["pool"] == sets["train"] | sets["val"], f"Family pool union mismatch: {family}")
        require(sets["all"] == sets["train"] | sets["val"] | sets["test"], f"Family all union mismatch: {family}")
        require(
            family_rows_by_name[family]["pool"]
            == family_rows_by_name[family]["train"] + family_rows_by_name[family]["val"],
            f"Family pool order/content mismatch: {family}",
        )
        for split_name in ("train", "val", "test"):
            expected_rows = [row for row in family_rows_by_name[family]["all"] if row["split"] == split_name]
            require(expected_rows == family_rows_by_name[family][split_name], f"Family split filter mismatch: {family}/{split_name}")
        expected_global_rows = [
            row for row in catalog
            if row["split"] and (
                bool_value(row["is_oxide"]) if family == "oxide" else bool_value(row["is_nitride"])
            )
        ]
        require(expected_global_rows == family_rows_by_name[family]["all"], f"Global family filter mismatch: {family}")
        summary = read_json(ROOT / f"data/{family}/summaries/summary.json")
        require(summary["counts"] == EXPECTED_FAMILY_COUNTS[family], f"Family summary mismatch: {family}")
        summary_relative = f"data/{family}/summaries/summary.json"
        family_inventory.append(
            {
                "family": family,
                "manifest": "summary",
                "repository_relative_path": summary_relative,
                "sha256": sha256(ROOT / summary_relative),
                "row_count": "",
                "unique_jid_count": "",
                "duplicate_jid_rows": "",
                "split_counts": json.dumps(summary["counts"], sort_keys=True),
                "schema": json.dumps(sorted(summary)),
                "status": "passed",
            }
        )
        for row in family_rows_by_name[family]["all"]:
            has_o, has_n = bool_value(row["has_O"]), bool_value(row["has_N"])
            require(has_o if family == "oxide" else has_n and not has_o, f"Bad family member: {family}/{row['jid']}")
            require(row["split"] == split_map[row["jid"]], f"Family/global split mismatch: {family}/{row['jid']}")
            require(math.isfinite(float(row["target"])), f"Nonfinite family target: {row['jid']}")
            require(row["filename"] == f"POSCAR-{row['jid']}.vasp", f"Family filename mismatch: {row['jid']}")
            if family == "oxide" and bool_value(row["is_oxynitride"]):
                oxynitrides.append(
                    {key: row[key] for key in ("jid", "split", "formula", "target", "filename", "elements")}
                )
        structure_dir = ROOT / f"data/{family}/structures"
        expected_names = {row["filename"] for row in family_rows_by_name[family]["all"]}
        actual_names = {entry.name for entry in structure_dir.iterdir()}
        missing_names = sorted(expected_names - actual_names)
        extra_names = sorted(actual_names - expected_names)
        symlinks = sorted(entry.name for entry in structure_dir.iterdir() if entry.is_symlink())
        nonfiles = sorted(entry.name for entry in structure_dir.iterdir() if not entry.is_file())
        require(not missing_names and not extra_names and not symlinks and not nonfiles, f"Structure coverage mismatch: {family}")
        structure_coverage[family] = {
            "expected_manifest_filenames": len(expected_names),
            "directory_entries": len(actual_names),
            "missing_count": 0,
            "extra_count": 0,
            "symlink_count": 0,
            "nonfile_count": 0,
            "status": "passed",
        }
    write_csv(
        "family_manifest_inventory.csv", family_inventory,
        ["family", "manifest", "repository_relative_path", "sha256", "row_count", "unique_jid_count", "duplicate_jid_rows", "split_counts", "schema", "status"],
    )
    family_cross = sorted(family_sets["oxide"]["all"] & family_sets["nitride"]["all"])
    require(not family_cross, "Oxide/nitride family cross-membership detected")
    write_csv("family_cross_membership.csv", [{"jid": jid} for jid in family_cross], ["jid"])
    write_json("manifest_structure_coverage.json", {"families": structure_coverage, "status": "passed"})

    oxynitrides.sort(key=lambda row: row["jid"])
    ox_counts = Counter(row["split"] for row in oxynitrides)
    require(len(oxynitrides) == 499, "Oxynitride count mismatch")
    require(ox_counts == Counter({"train": 400, "val": 42, "test": 57}), f"Oxynitride split mismatch: {ox_counts}")
    write_csv("oxynitride_inventory.csv", oxynitrides, ["jid", "split", "formula", "target", "filename", "elements"])
    pure_counts = {
        name: EXPECTED_FAMILY_COUNTS["oxide"][name] - (499 if name == "all" else 442 if name == "pool" else ox_counts[name])
        for name in ("all", "train", "val", "test", "pool")
    }
    require(pure_counts == {"all": 14492, "train": 11560, "val": 1505, "test": 1427, "pool": 13065}, "Pure-oxide counts differ")
    ox_summary = {
        "approved_definition": {
            "oxide": "contains O, including O+N records",
            "nitride": "contains N and does not contain O",
            "oxynitride": "contains both O and N; retained only in oxide arm",
        },
        "oxynitride_count": len(oxynitrides),
        "oxynitride_split_counts": dict(ox_counts),
        "pure_oxide_counts": pure_counts,
        "nitride_oxynitride_count": 0,
        "sensitivity_analysis_performed": False,
    }
    write_json("oxynitride_definition_summary.json", ox_summary)
    write_json(
        "family_definition_audit.json",
        {
            "family_counts": EXPECTED_FAMILY_COUNTS,
            "family_cross_membership_count": 0,
            "oxynitride_summary": ox_summary,
            "global_family_predicate_counts_before_split_exclusion": {
                "oxide": sum(bool_value(row["is_oxide"]) for row in catalog),
                "nitride": sum(bool_value(row["is_nitride"]) for row in catalog),
                "neither": sum(not bool_value(row["is_oxide"]) and not bool_value(row["is_nitride"]) for row in catalog),
            },
            "status": "FAMILY_DEFINITIONS_AND_MANIFESTS_PASSED",
        },
    )
    family_by_jid = {
        family: {row["jid"]: row for row in family_rows_by_name[family]["all"]}
        for family in FAMILIES
    }

    # Discover exact tracked canonical dataset-root and run files.
    tracked_paths = set(git("ls-files").splitlines())
    require(
        "Ties keep the first-seen record."
        in (ROOT / "scripts/dataset/family_dataset_lib.py").read_text(),
        "Deterministic duplicate tie rule changed",
    )
    protocol_3_dataset_root_candidates = sorted(
        relative
        for relative in tracked_paths
        if relative.startswith("results/protocol_3/")
        and "/dataset_root/" in relative
        and relative.endswith(("/split_manifest.json", "/id_prop.csv"))
    )
    require(not protocol_3_dataset_root_candidates, "protocol_3 unexpectedly contains separate canonical dataset roots")
    root_matches: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for relative in tracked_paths:
        match = ROOT_RE.fullmatch(relative)
        if not match:
            continue
        reject_path(relative)
        key = (int(match["set"]), match["family"], int(match["N"]), int(match["seed"]))
        root_matches.setdefault(key, {})[match["name"]] = relative
    require(len(root_matches) == 120, f"Canonical dataset-root count differs: {len(root_matches)}")
    require(all(set(files) == {"split_manifest.json", "id_prop.csv"} for files in root_matches.values()), "Dataset-root file pair incomplete")

    root_payloads: dict[tuple[int, str, int, int], dict[str, list[str]]] = {}
    root_audit_rows: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    structure_reference_total = 0
    for key in sorted(root_matches):
        set_number, family, n_value, seed = key
        files = root_matches[key]
        authorize_scientific_input(files["split_manifest.json"])
        authorize_scientific_input(files["id_prop.csv"])
        manifest_path = ROOT / files["split_manifest.json"]
        id_prop_path = ROOT / files["id_prop.csv"]
        payload_raw = read_json(manifest_path)
        required_keys = {"family", "N", "seed", "n_train", "n_val", "n_test", "train_jids", "val_jids", "test_jids"}
        local_violations: list[str] = []
        if set(payload_raw) != required_keys: local_violations.append("manifest_schema")
        payload = {"id_train": payload_raw["train_jids"], "id_val": payload_raw["val_jids"], "id_test": payload_raw["test_jids"]}
        root_payloads[key] = payload
        stats = split_stats(payload)
        expected_val = expected_val_count(n_value)
        expected_train = n_value - expected_val
        expected_test = EXPECTED_FAMILY_COUNTS[family]["test"]
        if (payload_raw["family"], int(payload_raw["N"]), int(payload_raw["seed"])) != (family, n_value, seed): local_violations.append("path_metadata")
        if (payload_raw["n_train"], payload_raw["n_val"], payload_raw["n_test"]) != (expected_train, expected_val, expected_test): local_violations.append("declared_counts")
        if (stats["n_train"], stats["n_val"], stats["n_test"]) != (expected_train, expected_val, expected_test): local_violations.append("actual_counts")
        if stats["unique_train"] != stats["n_train"] or stats["unique_val"] != stats["n_val"] or stats["unique_test"] != stats["n_test"]: local_violations.append("duplicates")
        if stats["train_val_overlap"] or stats["train_test_overlap"] or stats["val_test_overlap"]: local_violations.append("split_overlap")
        if stats["malformed_jid_count"]: local_violations.append("malformed_jid")
        if not (set(payload["id_train"]) | set(payload["id_val"])) <= family_sets[family]["pool"]: local_violations.append("outside_family_pool")
        if (set(payload["id_train"]) | set(payload["id_val"])) & family_sets[family]["test"]: local_violations.append("fixed_test_leakage")
        fixed_test_order = [row["jid"] for row in family_rows_by_name[family]["test"]]
        if payload["id_test"] != fixed_test_order: local_violations.append("fixed_test_order_or_membership")
        with id_prop_path.open(newline="") as handle:
            id_prop_rows = list(csv.reader(handle))
        if any(len(row) != 2 for row in id_prop_rows): local_violations.append("id_prop_schema")
        expected_order = payload["id_train"] + payload["id_val"] + payload["id_test"]
        actual_order = []
        for row in id_prop_rows:
            match = re.fullmatch(r"POSCAR-(JVASP-[0-9]+)\.vasp", row[0]) if len(row) == 2 else None
            if match is None:
                local_violations.append("id_prop_filename")
                continue
            actual_order.append(match.group(1))
            expected_structure = manifest_path.parent / row[0]
            expected_structure_relative = expected_structure.relative_to(ROOT).as_posix()
            authorize_scientific_input(expected_structure_relative)
            if expected_structure_relative not in tracked_paths or not expected_structure.is_file():
                local_violations.append("dataset_root_structure_missing")
            structure_reference_total += 1
            family_row = family_by_jid[family].get(match.group(1))
            if family_row is None or not math.isclose(float(row[1]), float(family_row["target"]), rel_tol=0, abs_tol=1e-12):
                local_violations.append("id_prop_target")
        if actual_order != expected_order: local_violations.append("id_prop_order")
        local_violations = sorted(set(local_violations))
        for issue in local_violations:
            violations.append({"scope": "dataset_root", "path": files["split_manifest.json"], "check": issue, "details": "failed"})
        root_audit_rows.append(
            {
                "set": set_number, "family": family, "N": n_value, "seed": seed,
                "split_manifest_path": files["split_manifest.json"],
                "split_manifest_tracked_status": "tracked",
                "split_manifest_size_bytes": manifest_path.stat().st_size,
                "split_manifest_sha256": sha256(manifest_path),
                "id_prop_path": files["id_prop.csv"],
                "id_prop_tracked_status": "tracked",
                "id_prop_size_bytes": id_prop_path.stat().st_size,
                "id_prop_sha256": sha256(id_prop_path),
                **stats,
                "declared_n_train": payload_raw["n_train"], "declared_n_val": payload_raw["n_val"], "declared_n_test": payload_raw["n_test"],
                "id_prop_row_count": len(id_prop_rows),
                "id_prop_exact_order_match": actual_order == expected_order,
                "fixed_test_exact_order_match": payload["id_test"] == fixed_test_order,
                "family_pool_membership_pass": (set(payload["id_train"]) | set(payload["id_val"])) <= family_sets[family]["pool"],
                "structure_references_pass": "dataset_root_structure_missing" not in local_violations,
                "status": "passed" if not local_violations else "failed",
                "violations": ";".join(local_violations),
            }
        )
    require(structure_reference_total == 140760, f"Unexpected id_prop structure-reference total: {structure_reference_total}")
    write_csv(
        "canonical_dataset_root_audit.csv", root_audit_rows,
        list(root_audit_rows[0]),
    )

    # Canonical run ID files.
    run_matches: list[tuple[str, re.Match[str]]] = []
    for relative in tracked_paths:
        match = RUN_RE.fullmatch(relative)
        if match:
            reject_path(relative)
            run_matches.append((relative, match))
    require(len(run_matches) == 240, f"Canonical run count differs: {len(run_matches)}")
    run_payloads: dict[tuple[int, str, int, int, str], dict[str, list[str]]] = {}
    run_rows: list[dict[str, Any]] = []
    run_ids_union: set[str] = set()
    for relative, match in sorted(run_matches):
        authorize_scientific_input(relative)
        set_number, family, n_value, seed = int(match["set"]), match["family"], int(match["N"]), int(match["seed"])
        method_dir = match["method"]
        is_scratch = method_dir.startswith("train_alignn_from_scratch")
        expected_method = (
            method_dir in {"finetune_last2", "train_alignn_from_scratch"} if set_number in (1, 2)
            else method_dir in {"finetune_last2_epochs100_bs32_lr5e5", "train_alignn_from_scratch_epochs100_bs32_lr5e5"}
        )
        local_violations: list[str] = []
        if not expected_method: local_violations.append("method_directory")
        if is_scratch and n_value not in (50, 500): local_violations.append("scratch_N")
        payload = read_json(ROOT / relative)
        if set(payload) != {"id_train", "id_val", "id_test"}: local_violations.append("schema")
        stats = split_stats(payload)
        expected_val = expected_val_count(n_value)
        if (stats["n_train"], stats["n_val"], stats["n_test"]) != (n_value - expected_val, expected_val, EXPECTED_FAMILY_COUNTS[family]["test"]): local_violations.append("counts")
        if stats["unique_train"] != stats["n_train"] or stats["unique_val"] != stats["n_val"] or stats["unique_test"] != stats["n_test"]: local_violations.append("duplicates")
        if stats["train_val_overlap"] or stats["train_test_overlap"] or stats["val_test_overlap"]: local_violations.append("split_overlap")
        if stats["malformed_jid_count"]: local_violations.append("malformed_jid")
        if not (set(payload["id_train"]) | set(payload["id_val"])) <= family_sets[family]["pool"]: local_violations.append("outside_family_pool")
        if (set(payload["id_train"]) | set(payload["id_val"])) & family_sets[family]["test"]: local_violations.append("fixed_test_leakage")
        fixed_test_order = [row["jid"] for row in family_rows_by_name[family]["test"]]
        if payload["id_test"] != fixed_test_order: local_violations.append("fixed_test_order_or_membership")
        root_key = (set_number if set_number in (1, 2) else 2, family, n_value, seed)
        if payload != root_payloads[root_key]: local_violations.append("dataset_root_mismatch")
        config_path = (ROOT / relative).parent / "config.json"
        config_relative = config_path.relative_to(ROOT).as_posix()
        authorize_scientific_input(config_relative)
        config: dict[str, Any] = {}
        if not config_path.is_file() or config_path.relative_to(ROOT).as_posix() not in tracked_paths:
            local_violations.append("config_missing")
        else:
            config = read_json(config_path)
            if (config.get("n_train"), config.get("n_val"), config.get("n_test")) != (stats["n_train"], stats["n_val"], stats["n_test"]):
                local_violations.append("config_count_mismatch")
        local_violations = sorted(set(local_violations))
        for issue in local_violations:
            violations.append({"scope": "canonical_run", "path": relative, "check": issue, "details": "failed"})
        normalized_method = "from_scratch" if is_scratch else "finetune"
        run_payloads[(set_number, family, n_value, seed, normalized_method)] = payload
        run_ids_union.update(payload["id_train"] + payload["id_val"] + payload["id_test"])
        run_rows.append(
            {
                "set": set_number, "family": family, "method": normalized_method, "method_directory": method_dir,
                "N": n_value, "seed": seed, "repository_relative_path": relative,
                "tracked_status": "tracked", "sha256": sha256(ROOT / relative),
                "size_bytes": (ROOT / relative).stat().st_size,
                "config_path": config_relative,
                "config_tracked_status": "tracked" if config_relative in tracked_paths else "missing",
                "config_size_bytes": config_path.stat().st_size if config_path.is_file() else "",
                "config_sha256": sha256(config_path) if config_path.is_file() else "",
                "config_fields_used": "n_train;n_val;n_test",
                "config_output_dir_metadata_status": (
                    "stale_reproduction_rerun_reference_not_followed"
                    if "reproduction_rerun" in str(config.get("output_dir", ""))
                    else "historical_output_dir_not_followed"
                ),
                **stats,
                "dataset_root_exact_match": payload == root_payloads[root_key],
                "fixed_test_exact_order_match": payload["id_test"] == fixed_test_order,
                "family_pool_membership_pass": (set(payload["id_train"]) | set(payload["id_val"])) <= family_sets[family]["pool"],
                "config_counts_match": "config_count_mismatch" not in local_violations and "config_missing" not in local_violations,
                "status": "passed" if not local_violations else "failed",
                "violations": ";".join(local_violations),
            }
        )
    require(
        sum(
            row["config_output_dir_metadata_status"]
            == "stale_reproduction_rerun_reference_not_followed"
            for row in run_rows
        )
        == 36,
        "Unexpected count of stale canonical config output_dir recovery references",
    )
    write_csv("canonical_run_split_audit.csv", run_rows, list(run_rows[0]))

    # Matched sampling and nesting.
    comparisons: dict[str, Any] = {}
    comparison_violations: list[dict[str, Any]] = []
    def compare_group(name: str, pairs: Iterable[tuple[dict[str, list[str]], dict[str, list[str]], str]]) -> None:
        total = 0
        membership_mismatches = []
        ordering_mismatches = []
        for left, right, label in pairs:
            total += 1
            if any(set(left[key]) != set(right[key]) for key in ("id_train", "id_val", "id_test")):
                membership_mismatches.append(label)
            elif left != right:
                ordering_mismatches.append(label)
        comparisons[name] = {
            "comparison_count": total,
            "membership_mismatch_count": len(membership_mismatches),
            "membership_mismatches": membership_mismatches,
            "ordering_mismatch_count": len(ordering_mismatches),
            "ordering_mismatches": ordering_mismatches,
            "total_mismatch_count": len(membership_mismatches) + len(ordering_mismatches),
        }
        comparison_violations.extend(
            {"scope": name, "path": label, "check": "split_membership_mismatch", "details": "failed"}
            for label in membership_mismatches
        )
        comparison_violations.extend(
            {"scope": name, "path": label, "check": "split_ordering_mismatch", "details": "failed"}
            for label in ordering_mismatches
        )

    compare_group(
        "protocol_1_protocol_2_finetune",
        ((run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in N_VALUES for s in SEEDS),
    )
    compare_group(
        "protocol_1_promoted_seed012_vs_protocol_2_finetune",
        ((run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in N_VALUES for s in (0, 1, 2)),
    )
    compare_group(
        "protocol_1_seed34_vs_protocol_2_finetune",
        ((run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in N_VALUES for s in (3, 4)),
    )
    compare_group(
        "protocol_2_protocol_3_finetune",
        ((run_payloads[(2, f, n, s, "finetune")], run_payloads[(3, f, n, s, "finetune")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in N_VALUES for s in SEEDS),
    )
    compare_group(
        "protocol_1_protocol_2_from_scratch",
        ((run_payloads[(1, f, n, s, "from_scratch")], run_payloads[(2, f, n, s, "from_scratch")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in (50, 500) for s in SEEDS),
    )
    compare_group(
        "protocol_2_protocol_3_from_scratch",
        ((run_payloads[(2, f, n, s, "from_scratch")], run_payloads[(3, f, n, s, "from_scratch")], f"{f}/N{n}/seed{s}") for f in FAMILIES for n in (50, 500) for s in SEEDS),
    )
    compare_group(
        "finetune_from_scratch_within_set",
        ((run_payloads[(set_no, f, n, s, "finetune")], run_payloads[(set_no, f, n, s, "from_scratch")], f"set{set_no}/{f}/N{n}/seed{s}") for set_no in (1, 2, 3) for f in FAMILIES for n in (50, 500) for s in SEEDS),
    )
    root_byte_mismatches = []
    for family in FAMILIES:
        for n_value in N_VALUES:
            for seed in SEEDS:
                for name in ("split_manifest.json", "id_prop.csv"):
                    left = ROOT / root_matches[(1, family, n_value, seed)][name]
                    right = ROOT / root_matches[(2, family, n_value, seed)][name]
                    if left.read_bytes() != right.read_bytes():
                        root_byte_mismatches.append(f"{family}/N{n_value}/seed{seed}/{name}")
    comparisons["protocol_1_protocol_2_dataset_root_files"] = {"comparison_count": 120, "mismatch_count": len(root_byte_mismatches), "mismatches": root_byte_mismatches}
    comparison_violations.extend(
        {
            "scope": "protocol_1_protocol_2_dataset_root_files",
            "path": label,
            "check": "byte_mismatch",
            "details": "failed",
        }
        for label in root_byte_mismatches
    )
    nested_total, nested_violations, prefix_nested_count = 0, [], 0
    for set_no in (1, 2, 3):
        for family in FAMILIES:
            for seed in SEEDS:
                for small_n, large_n in zip(N_VALUES[:-1], N_VALUES[1:]):
                    small = run_payloads[(set_no, family, small_n, seed, "finetune")]
                    large = run_payloads[(set_no, family, large_n, seed, "finetune")]
                    small_order = small["id_train"] + small["id_val"]
                    large_order = large["id_train"] + large["id_val"]
                    nested_total += 1
                    if not set(small_order) <= set(large_order):
                        nested_violations.append(f"set{set_no}/{family}/seed{seed}/N{small_n}->N{large_n}")
                    if large_order[:len(small_order)] == small_order:
                        prefix_nested_count += 1
    comparisons["nested_sampling"] = {
        "comparison_count": nested_total,
        "set_inclusion_violation_count": len(nested_violations),
        "violations": nested_violations,
        "prefix_nested_count": prefix_nested_count,
        "prefix_nesting_required": False,
        "note": "Nestedness is set-based. Each N is repartitioned into run train/validation lists, so concatenated order is not required to be prefix-nested.",
    }
    comparison_violations.extend({"scope": "nested_sampling", "path": label, "check": "set_inclusion", "details": "failed"} for label in nested_violations)
    violations.extend(comparison_violations)
    comparisons["status"] = "passed" if not comparison_violations and not root_byte_mismatches else "failed"
    write_json("matched_sampling_consistency.json", comparisons)

    # Quarantined legacy ID inventories: basic integrity only.
    legacy_matches: list[tuple[str, re.Match[str]]] = []
    for relative in tracked_paths:
        match = LEGACY_RE.fullmatch(relative)
        if match:
            reject_path(relative)
            legacy_matches.append((relative, match))
    require(len(legacy_matches) == 36, f"Legacy split count differs: {len(legacy_matches)}")
    legacy_rows: list[dict[str, Any]] = []
    for relative, match in sorted(legacy_matches):
        authorize_scientific_input(relative)
        payload = read_json(ROOT / relative)
        schema_ok = set(payload) == {"id_train", "id_val", "id_test"}
        stats = split_stats(payload) if schema_ok else {}
        no_duplicates = schema_ok and all(
            stats[f"unique_{name}"] == stats[f"n_{name}"]
            for name in ("train", "val", "test")
        )
        basic_pass = schema_ok and no_duplicates and not any(stats[key] for key in ("train_val_overlap", "train_test_overlap", "val_test_overlap", "malformed_jid_count"))
        legacy_rows.append(
            {
                "family": match["family"], "N": int(match["N"]), "seed": int(match["seed"]),
                "repository_relative_path": relative, "sha256": sha256(ROOT / relative),
                "classification": "legacy_before_correction_quarantined",
                "schema_pass": schema_ok, "basic_split_integrity_pass": basic_pass,
                "within_split_uniqueness_pass": no_duplicates,
                "paper_evidence_eligible": False,
                "n_train": stats.get("n_train"), "n_val": stats.get("n_val"), "n_test": stats.get("n_test"),
            }
        )
    write_csv("legacy_before_correction_split_inventory.csv", legacy_rows, list(legacy_rows[0]))

    # Resolve the exact three excluded records against every canonical run.
    split_exception_rows = []
    conflicts = {row["jid"]: row["splits"] for row in conflict_rows}
    for jid in sorted(EXPECTED_UNSPLIT):
        row = catalog_by_jid[jid]
        split_exception_rows.append(
            {
                "jid": jid,
                "formula": row["formula"],
                "family": "oxide" if bool_value(row["is_oxide"]) else "nitride" if bool_value(row["is_nitride"]) else "neither",
                "catalog_split": row["split"],
                "source_exception": f"split_conflict:{conflicts[jid]}" if jid in conflicts else "absent_from_authoritative_split_source",
                "present_in_any_family_manifest": jid in family_sets["oxide"]["all"] or jid in family_sets["nitride"]["all"],
                "present_in_any_canonical_run": jid in run_ids_union,
                "resolution": "explicitly_excluded_unassigned",
                "status": "passed",
            }
        )
    require(not any(row["present_in_any_family_manifest"] or row["present_in_any_canonical_run"] for row in split_exception_rows), "Unsplit record contaminates downstream data")
    write_csv(
        "split_exception_inventory.csv", split_exception_rows,
        ["jid", "formula", "family", "catalog_split", "source_exception", "present_in_any_family_manifest", "present_in_any_canonical_run", "resolution", "status"],
    )

    write_csv("run_split_violation_inventory.csv", violations, ["scope", "path", "check", "details"])
    require(not violations, f"Canonical integrity violations detected: {violations[:10]}")
    require(all(row["status"] == "passed" for row in root_audit_rows), "Dataset-root audit failure")
    require(all(row["status"] == "passed" for row in run_rows), "Canonical run audit failure")

    decision = f"""# Section 2.3B Dataset-Integrity Decision

Status: **SECTION23B_DATASET_INTEGRITY_VALIDATED**

- The 55,723 source entries are accounted for as 55,712 unique JIDs after
  deterministic removal of 11 excess duplicate rows across five JIDs.
- The downstream split map contains 55,709 unique JIDs. Two authoritative
  train/test conflicts (`JVASP-113961`, `JVASP-116461`) and one JID absent from
  the split source (`JVASP-1375`) remain explicitly unassigned and are absent
  from all family manifests and canonical run allocations.
- The oxide arm contains O-bearing materials and retains 499 oxynitrides. The
  nitride arm contains N-bearing materials without O and therefore contains no
  oxynitrides. This is an intentional asymmetric study definition.
- All 120 canonical dataset roots and all 240 canonical run split files pass
  uniqueness, disjointness, pool, fixed-test, ordering, cross-set, cross-method,
  and set-based nesting checks.
- The 36 before-correction split files remain quarantined and are not current
  paper evidence.

No dataset or result was modified. Pure-oxide performance and embedding
sensitivity were not calculated in this phase.

## Limitation

The original 55,723-record structural payload is not tracked locally. Exact raw
duplicate-record contents and kept indices cannot be replayed from that payload.
The frozen official ID multiplicities, deterministic resolution code, tracked
resolution metadata, arithmetic, and deduplicated catalog are mutually
consistent.
"""
    (EVIDENCE / "dataset_integrity_decision.md").write_text(decision)

    print(
        json.dumps(
            {
                "global_catalog_unique_jids": len(set(catalog_ids)),
                "assigned_split_jids": len(split_rows),
                "unassigned_jids": sorted(EXPECTED_UNSPLIT),
                "oxide_count": len(family_sets["oxide"]["all"]),
                "nitride_count": len(family_sets["nitride"]["all"]),
                "oxynitride_count": len(oxynitrides),
                "canonical_dataset_roots": len(root_audit_rows),
                "canonical_runs": len(run_rows),
                "canonical_violations": len(violations),
                "legacy_splits": len(legacy_rows),
                "status": "PRODUCER_AUDIT_PASSED",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
