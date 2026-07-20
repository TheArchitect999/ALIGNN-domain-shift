#!/usr/bin/env python3
"""Independently validate the Section 2.3B dataset-integrity package.

The validator reads only tracked canonical repository inputs plus the validated
Section 2.3A official-ID artifact.  It never traverses the excluded nested
checkout and refuses to proceed if Section 08 exists or is tracked.  Its only
writes are the four final Section 2.3B handoff artifacts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


csv.field_size_limit(min(sys.maxsize, 100_000_000))


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROOT = Path(".")
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
EXPECTED_BRANCH = "main"
VERDICT = "SECTION23B_DATASET_INTEGRITY_VALIDATED"
SECTION_A_MANIFEST_SHA = (
    "f815f80cbe705bebcb19d37b484031f74a071859686dda2001c9a56f888769be"
)

EVIDENCE_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3B_dataset_integrity"
)
EVIDENCE = ROOT / EVIDENCE_REL
A_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3A_checkpoint_provenance"
)
PRODUCER_REL = Path("scripts/shared/audit_dataset_integrity.py")
VALIDATOR_REL = Path("scripts/shared/validate_dataset_integrity.py")

FINAL_NAMES = {
    "generated_output_manifest.json",
    "section2_3B_validation.json",
    "section2_3B_report.md",
    "section2_3B_evidence.sha256",
}
PRODUCER_OUTPUTS = {
    "authoritative_dataset_input_inventory.csv",
    "canonical_dataset_root_audit.csv",
    "canonical_run_split_audit.csv",
    "dataset_integrity_decision.md",
    "duplicate_resolution_audit.csv",
    "duplicate_resolution_audit.json",
    "family_cross_membership.csv",
    "family_definition_audit.json",
    "family_manifest_inventory.csv",
    "global_dataset_integrity.json",
    "global_split_integrity.json",
    "legacy_before_correction_split_inventory.csv",
    "manifest_structure_coverage.json",
    "matched_sampling_consistency.json",
    "oxynitride_definition_summary.json",
    "oxynitride_inventory.csv",
    "repository_preflight.json",
    "run_split_violation_inventory.csv",
    "split_exception_inventory.csv",
}

GLOBAL_PATHS = {
    "split_csv": "data/manifests/dft_3d_formation_energy_peratom_splits.csv",
    "conflicts": "data/manifests/dft_3d_formation_energy_peratom_splits_conflicts.csv",
    "catalog": "data/diagnostics/global_record_catalog.csv",
    "split_json": "data/diagnostics/global_split_manifest.json",
    "schema": "data/diagnostics/schema_report.json",
}
FAMILIES = ("oxide", "nitride")
SPLITS = ("train", "val", "test")
MANIFEST_NAMES = ("all", "train", "val", "test", "pool")
N_VALUES = (10, 50, 100, 200, 500, 1000)
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_COUNTS = {
    "oxide": {
        "all": 14991,
        "train": 11960,
        "val": 1547,
        "test": 1484,
        "pool": 13507,
    },
    "nitride": {
        "all": 2288,
        "train": 1837,
        "val": 209,
        "test": 242,
        "pool": 2046,
    },
}
EXPECTED_DUPLICATES = {
    "JVASP-100669": 2,
    "JVASP-113961": 2,
    "JVASP-116461": 3,
    "JVASP-96735": 6,
    "JVASP-97311": 3,
}
EXPECTED_UNASSIGNED = {"JVASP-113961", "JVASP-116461", "JVASP-1375"}
EXPECTED_GLOBAL_SHA = {
    GLOBAL_PATHS["split_csv"]: "82e31f229775a25fb96794a8daa8b1005b264a68ed7a55e3d4be90d0a7badc79",
    GLOBAL_PATHS["conflicts"]: "151b56a1c93712772dc861d0f68c5773c4a78f7054a48343c23323181e42a776",
    GLOBAL_PATHS["catalog"]: "c38df88550a1214d8cf7622c994df655a8f5db023cefccc7e6859671d58759df",
    GLOBAL_PATHS["split_json"]: "bbbacf67fafdcb948c326663ca7927131af1d521374dc7d6eb88ce09ec5edbe2",
    GLOBAL_PATHS["schema"]: "e08dccc9e2daa95e5aae8e8530f8ce585c47d8ddb32ce81effc21f97779aad1a",
}
OFFICIAL_IDS_REL = (
    A_REL / "source_artifacts/official_ids_train_val_test.json"
).as_posix()
PROCEDURAL_INPUTS = {
    "scripts/dataset/family_dataset_lib.py",
    "scripts/dataset/build_family_datasets.py",
    "scripts/dataset/validate_family_datasets.py",
    "scripts/dataset/make_split_manifest_from_benchmark.py",
    "scripts/dataset/materialize_alignn_root.py",
    "scripts/shared/prepare_baseline_finetune_dataset.py",
}
EXPECTED_INVENTORY_PATHS = set(GLOBAL_PATHS.values()) | {
    f"data/{family}/manifests/{name}.csv"
    for family in FAMILIES
    for name in MANIFEST_NAMES
} | {
    f"data/{family}/summaries/summary.json"
    for family in FAMILIES
} | PROCEDURAL_INPUTS | {OFFICIAL_IDS_REL}
NONCANONICAL_BACKUP_RELS = (
    "results/protocol_1/oxide/N50_seed0/dataset_root/"
    "id_prop.with_header_backup.csv",
    "results/protocol_2/oxide/N50_seed0/dataset_root/"
    "id_prop.with_header_backup.csv",
)

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
    ".cache", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "cache", "temp", "tmp",
}

FIXED_SCIENTIFIC_INPUTS = set(GLOBAL_PATHS.values()) | {OFFICIAL_IDS_REL} | {
    f"data/{family}/manifests/{name}.csv"
    for family in FAMILIES
    for name in MANIFEST_NAMES
} | {
    f"data/{family}/summaries/summary.json"
    for family in FAMILIES
} | PROCEDURAL_INPUTS


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace")


def tracked_reference_paths(pattern: str) -> list[str]:
    """Return tracked text files outside legacy raw roots that cite a pattern."""
    result = subprocess.run(
        [
            "git", "grep", "-I", "-l", "-E", pattern, "--", ".",
            ":(exclude)results/protocol_1/**",
            ":(exclude)results/derived_evidence/final_paper_factory/archived_submission_materials/**",
            ":(exclude)domain_shift-alignn-domain-shift/**",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode in {0, 1}, "Tracked legacy-reference scan failed")
    return result.stdout.decode("utf-8", errors="replace").splitlines()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_path(relative: str | Path) -> None:
    value = Path(relative).as_posix()
    parts = Path(value).parts
    require(not Path(value).is_absolute(), f"Absolute input rejected: {value}")
    require(".." not in parts, f"Traversal input rejected: {value}")
    require(
        value != "domain_shift-alignn-domain-shift"
        and not value.startswith("domain_shift-alignn-domain-shift/"),
        f"Nested checkout input rejected: {value}",
    )
    require(
        not any(fragment in value for fragment in FORBIDDEN_FRAGMENTS),
        f"Forbidden input rejected: {value}",
    )
    require(
        not any(part.lower() in FORBIDDEN_COMPONENTS for part in parts),
        f"Temporary/cache input rejected: {value}",
    )
    require(
        not value.endswith((".pyc", ".pyo", ".tmp", ".temp", "~")),
        f"Temporary input rejected: {value}",
    )


def authorize_scientific_input(relative: str | Path) -> None:
    """Enforce the exact dataset/result-input allowlist used by the validator."""
    value = Path(relative).as_posix()
    reject_path(value)
    config_authorized = value.endswith("/config.json") and RUN_RE.fullmatch(
        value[: -len("config.json")] + "ids_train_val_test.json"
    )
    require(
        value in FIXED_SCIENTIFIC_INPUTS
        or ROOT_RE.fullmatch(value)
        or RUN_RE.fullmatch(value)
        or LEGACY_RE.fullmatch(value)
        or config_authorized,
        f"Unapproved dataset/result input rejected: {value}",
    )


def read_json(relative: str | Path) -> Any:
    reject_path(relative)
    return json.loads((ROOT / relative).read_text())


def read_csv(relative: str | Path) -> tuple[list[str], list[dict[str, str]]]:
    reject_path(relative)
    with (ROOT / relative).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def read_evidence_csv(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (EVIDENCE / name).open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def bool_value(value: str) -> bool:
    return value.strip().lower() == "true"


def expected_val_count(n_value: int) -> int:
    n_val = max(5, int(round(0.1 * n_value)))
    return max(1, n_value // 5) if n_val >= n_value else n_val


def normalize_root_split(payload: dict[str, Any]) -> dict[str, list[str]]:
    require(
        set(payload)
        == {
            "family",
            "N",
            "seed",
            "n_train",
            "n_val",
            "n_test",
            "train_jids",
            "val_jids",
            "test_jids",
        },
        "Unexpected dataset-root split schema",
    )
    return {
        "id_train": payload["train_jids"],
        "id_val": payload["val_jids"],
        "id_test": payload["test_jids"],
    }


def split_stats(payload: dict[str, list[str]]) -> dict[str, int]:
    require(
        set(payload) == {"id_train", "id_val", "id_test"},
        "Unexpected canonical split schema",
    )
    lists = {key: payload[key] for key in ("id_train", "id_val", "id_test")}
    require(
        all(isinstance(items, list) and all(isinstance(jid, str) for jid in items)
            for items in lists.values()),
        "Canonical split contains non-string/non-list data",
    )
    sets = {key: set(items) for key, items in lists.items()}
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
            JID_RE.fullmatch(jid) is None for items in lists.values() for jid in items
        ),
    }


def parse_status() -> list[dict[str, str]]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    rows = []
    for item in raw.split(b"\0"):
        if item:
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
    if path in {"domain_shift-alignn-domain-shift", "domain_shift-alignn-domain-shift/"} or path.startswith("domain_shift-alignn-domain-shift/"):
        return "nested_checkout"
    if path.startswith(A_REL.as_posix() + "/") or path in {
        "scripts/shared/audit_checkpoint_provenance.py",
        "scripts/shared/validate_checkpoint_provenance.py",
    }:
        return "section2_3A_outputs"
    if path.startswith(EVIDENCE_REL.as_posix() + "/") or path in {
        PRODUCER_REL.as_posix(),
        VALIDATOR_REL.as_posix(),
    }:
        return "section2_3B_outputs"
    return "ambiguous"


def verify_checksum_manifest(relative: str, mode: str) -> dict[str, Any]:
    reject_path(relative)
    path = ROOT / relative
    require(path.is_file(), f"Missing checksum manifest: {relative}")
    checked = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum line in {relative}")
        expected, listed = match.groups()
        reject_path(listed)
        if mode == "root":
            candidate = ROOT / listed
        else:
            candidate = path.parent / listed
        require(candidate.is_file(), f"Checksum target missing: {listed}")
        require(sha256(candidate) == expected, f"Checksum mismatch: {listed}")
        checked += 1
    require(checked > 0, f"Empty checksum manifest: {relative}")
    return {"path": relative, "files_verified": checked, "status": "passed"}


def assert_header(name: str, expected: list[str]) -> list[dict[str, str]]:
    fields, rows = read_evidence_csv(name)
    require(fields == expected, f"Producer CSV schema mismatch: {name}: {fields}")
    return rows


def json_keys(name: str, expected: set[str]) -> dict[str, Any]:
    payload = json.loads((EVIDENCE / name).read_text())
    require(isinstance(payload, dict), f"Producer JSON is not an object: {name}")
    require(set(payload) == expected, f"Producer JSON schema mismatch: {name}")
    return payload


def compare_group(
    pairs: Iterable[tuple[dict[str, list[str]], dict[str, list[str]]]]
) -> dict[str, int]:
    total = 0
    membership_mismatches = 0
    ordering_mismatches = 0
    for left, right in pairs:
        total += 1
        membership_mismatch = any(
            set(left[key]) != set(right[key])
            for key in ("id_train", "id_val", "id_test")
        )
        membership_mismatches += membership_mismatch
        ordering_mismatches += not membership_mismatch and left != right
    return {
        "comparison_count": total,
        "membership_mismatch_count": membership_mismatches,
        "ordering_mismatch_count": ordering_mismatches,
        "total_mismatch_count": membership_mismatches + ordering_mismatches,
    }


def write_json(name: str, payload: Any) -> None:
    (EVIDENCE / name).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_report(metrics: dict[str, Any], checksum_counts: dict[str, int]) -> str:
    created = "\n".join(f"  - `{path}`" for path in metrics["created_paths"])
    hashes = "\n".join(
        f"  - `{path}`: `{digest}`" for path, digest in metrics["authoritative_hashes"].items()
    )
    categories = ", ".join(
        f"{name}={count}" for name, count in sorted(metrics["final_status_categories"].items())
    )
    return f"""# Section 2.3B Independent Validation Report

Verdict: **{VERDICT}**

The validator independently reloaded the frozen authorities and recomputed
all critical sets, intersections, counts, ordering checks, and hashes. Producer
totals were checked only after that independent recomputation.

## Safety and preceding gates

- Repository: `{EXPECTED_ROOT}`; branch `{EXPECTED_BRANCH}`; HEAD `{EXPECTED_HEAD}`.
- Section 08 is absent and untracked. Its contents were never inspected or used.
- The nested checkout, rerun staging/recovery controls, caches, temporary files,
  and unrelated user changes were excluded and untouched.
- Promotion, protocol_1 regeneration, and Section 2.3A checksum gates passed:
  {checksum_counts['promotion']}/19, {checksum_counts['regeneration']}/22, and
  {checksum_counts['section2_3A']}/31 entries.
- Section 2.3A verdict is exactly `SECTION23A_VALIDATED_KNOWN_MEMBERSHIP`; its
  checksum-manifest SHA-256 is exactly `{SECTION_A_MANIFEST_SHA}`.
- The supplied schema-report digest has only 63 hexadecimal characters. The
  recomputed tracked SHA-256 is
  `e08dccc9e2daa95e5aae8e8530f8ce585c47d8ddb32ce81effc21f97779aad1a`;
  the supplied value omitted its final `a`.

Authoritative global hashes:
{hashes}

## Duplicate and global-split reconciliation

- Frozen official IDs contain 44,578 train, 5,572 validation, and 5,572 test
  entries: 55,722 entries and 55,711 unique JIDs.
- Duplicate multiplicities are exactly `JVASP-100669`×2,
  `JVASP-113961`×2, `JVASP-116461`×3, `JVASP-96735`×6, and
  `JVASP-97311`×3: five duplicate JIDs and 11 excess rows.
- Per-split duplication is exact: 100669 train×2; 113961 train×1/test×1;
  116461 train×2/test×1; 96735 train×6; 97311 train×3.
- Arithmetic: 55,722 − 11 = 55,711 official unique JIDs; adding catalog-only
  `JVASP-1375` gives 55,723 source records and 55,712 deduplicated catalog
  JIDs; excluding the two train/test conflicts gives 55,709 assigned JIDs.
- The schema report's top-level `n_records=55,712` and
  `n_duplicate_jids_detected=0` describe the post-deduplication catalog. Its
  nested duplicate-resolution block records the original 55,723/5/11 state.
- Global train/validation/test counts are {metrics['global_train']:,} /
  {metrics['global_val']:,} / {metrics['global_test']:,}; all three are unique
  and pairwise disjoint.
- `JVASP-113961` (nitride, train/test conflict), `JVASP-116461` (oxide,
  train/test conflict), and `JVASP-1375` (nitride, absent from the official
  split source) remain blank/unassigned and occur in no family or canonical run.

## Family definitions and counts

- Oxide means contains O and retains O+N records: all/train/validation/test/pool
  = 14,991 / 11,960 / 1,547 / 1,484 / 13,507.
- Nitride means contains N without O: all/train/validation/test/pool
  = 2,288 / 1,837 / 209 / 242 / 2,046.
- Oxide/nitride cross-membership is zero. Every family split is unique,
  pairwise disjoint, equal to its required union/filter, and tied to the global split.
- Oxynitrides total 499 and remain only in oxide: train/validation/test
  = 400 / 42 / 57. Nitride oxynitrides = 0.
- Derived pure-oxide all/train/validation/test/pool counts are
  14,492 / 11,560 / 1,505 / 1,427 / 13,065. No sensitivity metric was run.

## Canonical allocation results

- Dataset roots: {metrics['dataset_roots_passed']}/{metrics['dataset_roots']} passed.
- Canonical run split files: {metrics['canonical_runs_passed']}/{metrics['canonical_runs']} passed.
- Fixed-test drift = 0; train/validation leakage = 0; canonical violations = 0.
- Cross-set membership mismatches = 0; ordering mismatches = 0;
  cross-method mismatches = 0; protocol_1 seed-0–2 promotion mismatches = 0;
  seed-3/4 mismatches = 0.
- Set-based N nesting: {metrics['nesting_comparisons']} comparisons and
  {metrics['nesting_violations']} violations.
- All 240 configs are tracked and hash-frozen. Only `n_train`, `n_val`, and
  `n_test` were used. Every `output_dir` is historical non-authoritative
  metadata and was never resolved or followed; 36 contain `_reproduction_rerun`.
- The canonical namespace contains exactly 240 run-ID files, 240 root metadata
  files, no protocol_3 dataset roots, and only the two known header-backup copies.
  Those two copies were excluded name-only, without reading or hashing.
- All {metrics['legacy_files']} before-correction split files remain
  `legacy_before_correction_quarantined`; a scoped tracked-reference scan found
  {metrics['legacy_dependency_reference_count']} downstream exact legacy split-file references.
  Legitimate `results/protocol_1/{{oxide,nitride}}/zero_shot` baselines were not reclassified.

## Outputs and repository preservation

Created (all untracked, and only these Section 2.3B paths):
{created}

Modified tracked files: none. Staged files: none.

Final Git status contains {metrics['final_status_count']} untracked paths and no
tracked or staged change. Classification: {categories}. The exact preserved
578-path baseline equals the preflight freeze; all five protected governance/
prior-evidence hashes are unchanged. `input_manifest.md`, `source_policy.md`,
and `run_session.json` were not modified.

The package contains 23 Section 2.3B evidence outputs. The checksum covers all
22 non-checksum outputs plus both scripts: 24 verified entries. The generated
manifest excludes its own hash, and the checksum excludes itself, because
including either cyclic hash is mathematically impossible.

No Git network operation, add, stage, commit, or push occurred. No dataset,
result, manuscript, prior evidence, governance file, or recovery control was
modified.

## Limitation and handoff

The original 55,723-record structural payload is not tracked locally, so raw
duplicate structure bodies cannot be replayed. Historical kept indices are
frozen metadata rather than raw-payload rederivations. This does not block the
fully reproduced downstream catalog, family, root, and run integrity checks.

The exact next subphase is **Section 2.3C — Oxynitride sensitivity and zero-shot
bootstrap uncertainty**. Section 2.3C was not started.
"""


def main() -> None:
    # Safety gates and immutable prior evidence.
    require(ROOT.resolve() == EXPECTED_ROOT, f"Unexpected repository root: {ROOT}")
    require(git("rev-parse", "HEAD").strip() == EXPECTED_HEAD, "Unexpected HEAD")
    require(git("branch", "--show-current").strip() == EXPECTED_BRANCH, "Unexpected branch")
    section08 = ROOT / "results/derived_evidence/final_paper_factory/archived_submission_materials"
    require(not section08.exists(), "Section 08 exists; refusing to inspect")
    require(
        not git("ls-files", "--", "results/derived_evidence/final_paper_factory/archived_submission_materials").strip(),
        "Section 08 has tracked content",
    )
    require(EVIDENCE.is_dir(), "Section 2.3B evidence directory is missing")
    require((ROOT / PRODUCER_REL).is_file(), "Section 2.3B producer is missing")
    require((ROOT / VALIDATOR_REL).is_file(), "Section 2.3B validator is missing")

    status = parse_status()
    baseline = [row for row in status if status_category(row["path"]) != "section2_3B_outputs"]
    baseline_categories = Counter(status_category(row["path"]) for row in baseline)
    require(not baseline_categories.get("ambiguous", 0), "Ambiguous working-tree input detected")
    require(all(row["status"] == "??" for row in baseline), "Tracked/staged change detected")
    require(
        baseline_categories
        == Counter(
            {
                "rerun_staging_files": 504,
                "rerun_configurations": 36,
                "rerun_summary_controls": 2,
                "governance_session": 3,
                "section2_3A_outputs": 32,
                "nested_checkout": 1,
            }
        ),
        f"Unexpected protected baseline: {baseline_categories}",
    )

    prior_checksums = {
        "promotion": verify_checksum_manifest(
            "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256",
            "root",
        ),
        "regeneration": verify_checksum_manifest(
            "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256",
            "local",
        ),
        "section2_3A": verify_checksum_manifest(
            (A_REL / "section2_3A_evidence.sha256").as_posix(),
            "root",
        ),
    }
    require(
        {name: result["files_verified"] for name, result in prior_checksums.items()}
        == {"promotion": 19, "regeneration": 22, "section2_3A": 31},
        "Preceding checksum-manifest entry counts differ",
    )
    section_a_manifest = ROOT / A_REL / "section2_3A_evidence.sha256"
    require(
        sha256(section_a_manifest) == SECTION_A_MANIFEST_SHA,
        "Section 2.3A checksum-manifest SHA-256 differs",
    )
    section_a_validation = read_json(A_REL / "section2_3A_validation.json")
    require(
        section_a_validation.get("verdict")
        == "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP",
        "Exact Section 2.3A verdict differs",
    )
    require(
        "protocol_1_REGENERATION_VALIDATED"
        in (ROOT / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md").read_text(),
        "protocol_1 regeneration marker missing",
    )

    # Ensure the producer package is complete before validating its contents.
    current_pre_final = {
        path.name for path in EVIDENCE.iterdir() if path.is_file() and path.name not in FINAL_NAMES
    }
    require(current_pre_final == PRODUCER_OUTPUTS, f"Producer output set differs: {current_pre_final ^ PRODUCER_OUTPUTS}")

    tracked_paths = set(git("ls-files").splitlines())
    # These paths are intentionally checked only as names in the Git index.
    # They are not opened, statted, or hashed and never enter an input inventory.
    require(
        set(NONCANONICAL_BACKUP_RELS) <= tracked_paths,
        "Expected noncanonical backup path is not tracked",
    )
    for relative, expected_hash in EXPECTED_GLOBAL_SHA.items():
        authorize_scientific_input(relative)
        require(relative in tracked_paths, f"Global authority is untracked: {relative}")
        require(sha256(ROOT / relative) == expected_hash, f"Global authority hash mismatch: {relative}")
    allocation_source_rel = "scripts/shared/prepare_baseline_finetune_dataset.py"
    authorize_scientific_input(allocation_source_rel)
    allocation_source = (ROOT / allocation_source_rel).read_text()
    for required_line in (
        "n_val = max(5, int(round(0.1 * args.N)))",
        "if n_val >= args.N:",
        "n_val = max(1, args.N // 5)",
    ):
        require(required_line in allocation_source, f"Validation allocation implementation changed: {required_line}")

    # Global catalog, split, duplicate, conflict, and unassigned recomputation.
    catalog_fields, catalog = read_csv(GLOBAL_PATHS["catalog"])
    split_fields, split_rows = read_csv(GLOBAL_PATHS["split_csv"])
    conflict_fields, conflict_rows = read_csv(GLOBAL_PATHS["conflicts"])
    split_json = read_json(GLOBAL_PATHS["split_json"])
    schema_report = read_json(GLOBAL_PATHS["schema"])
    authorize_scientific_input(OFFICIAL_IDS_REL)
    official_ids = read_json(OFFICIAL_IDS_REL)
    require(
        catalog_fields
        == [
            "jid", "split", "target", "target_key_used", "filename", "formula",
            "n_atoms", "elements", "unique_elements", "has_O", "has_N",
            "is_oxide", "is_nitride", "is_oxynitride",
        ],
        "Global catalog schema mismatch",
    )
    require(split_fields == ["jid", "split"], "Global split CSV schema mismatch")
    require(conflict_fields == ["jid", "splits"], "Conflict CSV schema mismatch")
    catalog_ids = [row["jid"] for row in catalog]
    split_ids = [row["jid"] for row in split_rows]
    split_map = {row["jid"]: row["split"] for row in split_rows}
    catalog_by_jid = {row["jid"]: row for row in catalog}
    require(len(catalog_ids) == len(set(catalog_ids)) == 55712, "Global catalog count/uniqueness failure")
    require(len(split_ids) == len(set(split_ids)) == 55709, "Global split count/uniqueness failure")
    require(list(split_json.items()) == [(row["jid"], row["split"]) for row in split_rows], "Split CSV/JSON mismatch")
    split_counts = Counter(row["split"] for row in split_rows)
    require(split_counts == Counter({"train": 44567, "val": 5572, "test": 5570}), "Global split count mismatch")
    global_sets = {name: {row["jid"] for row in split_rows if row["split"] == name} for name in SPLITS}
    require(not global_sets["train"] & global_sets["val"], "Global train/val overlap")
    require(not global_sets["train"] & global_sets["test"], "Global train/test overlap")
    require(not global_sets["val"] & global_sets["test"], "Global val/test overlap")
    require(set(catalog_ids) - set(split_ids) == EXPECTED_UNASSIGNED, "Unexpected unassigned JIDs")
    require(
        {(row["jid"], row["splits"]) for row in conflict_rows}
        == {("JVASP-113961", "test|train"), ("JVASP-116461", "test|train")},
        "Conflict set mismatch",
    )

    catalog_issue_counts: Counter[str] = Counter()
    for row in catalog:
        elements = row["elements"].split(";") if row["elements"] else []
        unique_elements = sorted(set(elements))
        has_o, has_n = "O" in unique_elements, "N" in unique_elements
        if JID_RE.fullmatch(row["jid"]) is None:
            catalog_issue_counts["malformed_jid"] += 1
        try:
            if not math.isfinite(float(row["target"])):
                catalog_issue_counts["nonfinite_target"] += 1
        except ValueError:
            catalog_issue_counts["nonfinite_target"] += 1
        if row["target_key_used"] != "formation_energy_peratom":
            catalog_issue_counts["target_key"] += 1
        if row["filename"] != f"POSCAR-{row['jid']}.vasp":
            catalog_issue_counts["filename"] += 1
        if int(row["n_atoms"]) != len(elements):
            catalog_issue_counts["atom_count"] += 1
        if row["unique_elements"] != ";".join(unique_elements):
            catalog_issue_counts["unique_elements"] += 1
        expected_flags = (has_o, has_n, has_o, has_n and not has_o, has_o and has_n)
        actual_flags = tuple(
            bool_value(row[key])
            for key in ("has_O", "has_N", "is_oxide", "is_nitride", "is_oxynitride")
        )
        if expected_flags != actual_flags:
            catalog_issue_counts["family_flags"] += 1
        if row["split"] != split_map.get(row["jid"], ""):
            catalog_issue_counts["split_mapping"] += 1
    require(not catalog_issue_counts, f"Global catalog semantic issues: {catalog_issue_counts}")

    require(set(official_ids) == {"id_train", "id_val", "id_test"}, "Official ID source schema mismatch")
    official_split_lengths = {
        key: len(official_ids[key]) for key in ("id_train", "id_val", "id_test")
    }
    require(
        official_split_lengths == {"id_train": 44578, "id_val": 5572, "id_test": 5572},
        f"Official ID split lengths mismatch: {official_split_lengths}",
    )
    official_counter = Counter(
        jid for key in ("id_train", "id_val", "id_test") for jid in official_ids[key]
    )
    require(sum(official_split_lengths.values()) == 55722, "Official flattened ID count mismatch")
    require(len(official_counter) == 55711, "Official unique-ID count mismatch")
    require(
        set(catalog_ids) - set(official_counter) == {"JVASP-1375"}
        and not (set(official_counter) - set(catalog_ids)),
        "Official ID/catalog reconciliation mismatch",
    )
    duplicate_counts = {jid: count for jid, count in sorted(official_counter.items()) if count > 1}
    require(duplicate_counts == EXPECTED_DUPLICATES, "Official duplicate multiplicities mismatch")
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
        "Official per-split duplicate distribution mismatch",
    )
    duplicate_meta = schema_report["duplicate_resolution"]
    require(schema_report["n_records"] == 55712, "Post-dedup schema-report record count mismatch")
    require(
        schema_report["n_duplicate_jids_detected"] == 0,
        "Post-dedup schema-report duplicate count mismatch",
    )
    require(duplicate_meta["n_input_records"] == 55723, "Source-record count mismatch")
    require(duplicate_meta["n_unique_jids"] == 55712, "Deduplicated count mismatch")
    require(duplicate_meta["n_duplicate_jids"] == 5, "Duplicate-JID count mismatch")
    require(duplicate_meta["duplicate_jid_total_extra_rows"] == 11, "Duplicate excess-row mismatch")
    for jid, multiplicity in EXPECTED_DUPLICATES.items():
        example = duplicate_meta["duplicate_examples"][jid]
        require(len(example["seen_indices"]) == multiplicity, f"Duplicate index count mismatch: {jid}")
        require(len(example["scores"]) == multiplicity, f"Duplicate score count mismatch: {jid}")
        require(len({tuple(score) for score in example["scores"]}) == 1, f"Duplicate score tie mismatch: {jid}")
        require(example["kept_index"] == example["seen_indices"][0], f"Duplicate first-seen tie rule mismatch: {jid}")
        require(catalog_ids.count(jid) == 1, f"Duplicate not resolved in catalog: {jid}")

    # Family definitions and manifests, independently derived from the catalog.
    family_rows: dict[str, dict[str, list[dict[str, str]]]] = {}
    family_sets: dict[str, dict[str, set[str]]] = {}
    structure_coverage: dict[str, Any] = {}
    oxynitride_rows: list[dict[str, str]] = []
    for family in FAMILIES:
        family_rows[family] = {}
        family_sets[family] = {}
        for name in MANIFEST_NAMES:
            relative = f"data/{family}/manifests/{name}.csv"
            authorize_scientific_input(relative)
            fields, rows = read_csv(relative)
            require(fields == catalog_fields, f"Family schema mismatch: {family}/{name}")
            ids = [row["jid"] for row in rows]
            require(len(rows) == EXPECTED_COUNTS[family][name], f"Family count mismatch: {family}/{name}")
            require(len(ids) == len(set(ids)), f"Family duplicate: {family}/{name}")
            family_rows[family][name] = rows
            family_sets[family][name] = set(ids)
        sets = family_sets[family]
        require(not sets["train"] & sets["val"], f"Family train/val overlap: {family}")
        require(not sets["train"] & sets["test"], f"Family train/test overlap: {family}")
        require(not sets["val"] & sets["test"], f"Family val/test overlap: {family}")
        require(sets["pool"] == sets["train"] | sets["val"], f"Pool union mismatch: {family}")
        require(sets["all"] == sets["pool"] | sets["test"], f"All union mismatch: {family}")
        require(family_rows[family]["pool"] == family_rows[family]["train"] + family_rows[family]["val"], f"Pool order mismatch: {family}")
        for split in SPLITS:
            expected = [row for row in family_rows[family]["all"] if row["split"] == split]
            require(expected == family_rows[family][split], f"Family split filter mismatch: {family}/{split}")
        expected_from_catalog = [
            row for row in catalog
            if row["split"] and (
                bool_value(row["is_oxide"])
                if family == "oxide"
                else bool_value(row["is_nitride"])
            )
        ]
        require(expected_from_catalog == family_rows[family]["all"], f"Family catalog filter mismatch: {family}")
        summary_relative = f"data/{family}/summaries/summary.json"
        authorize_scientific_input(summary_relative)
        summary = read_json(summary_relative)
        require(summary["counts"] == EXPECTED_COUNTS[family], f"Family summary mismatch: {family}")
        for row in family_rows[family]["all"]:
            has_o, has_n = bool_value(row["has_O"]), bool_value(row["has_N"])
            require(has_o if family == "oxide" else has_n and not has_o, f"Family predicate failure: {family}/{row['jid']}")
            if family == "oxide" and has_o and has_n:
                oxynitride_rows.append(
                    {key: row[key] for key in ("jid", "split", "formula", "target", "filename", "elements")}
                )
        structure_dir = ROOT / f"data/{family}/structures"
        expected_names = {row["filename"] for row in family_rows[family]["all"]}
        actual_entries = list(structure_dir.iterdir())
        actual_names = {entry.name for entry in actual_entries}
        require(expected_names == actual_names, f"Family structure coverage mismatch: {family}")
        require(all(entry.is_file() and not entry.is_symlink() for entry in actual_entries), f"Family structure type failure: {family}")
        structure_coverage[family] = {
            "expected_manifest_filenames": len(expected_names),
            "directory_entries": len(actual_names),
            "missing_count": 0,
            "extra_count": 0,
            "symlink_count": 0,
            "nonfile_count": 0,
            "status": "passed",
        }
    require(not family_sets["oxide"]["all"] & family_sets["nitride"]["all"], "Cross-family membership detected")
    oxynitride_rows.sort(key=lambda row: row["jid"])
    ox_counts = Counter(row["split"] for row in oxynitride_rows)
    require(len(oxynitride_rows) == 499, "Oxynitride total mismatch")
    require(ox_counts == Counter({"train": 400, "val": 42, "test": 57}), "Oxynitride split mismatch")
    require(not any(bool_value(row["has_O"]) for row in family_rows["nitride"]["all"]), "Oxynitride retained in nitride")

    # Discover and validate all 120 tracked canonical dataset roots.
    root_candidate_paths = {
        relative
        for relative in tracked_paths
        if re.match(r"^results/protocol_[123]/", relative)
        and "/dataset_root/" in relative
        and relative.endswith(("/split_manifest.json", "/id_prop.csv"))
    }
    root_metadata_or_copy_candidates = {
        relative
        for relative in tracked_paths
        if re.match(r"^results/protocol_[123]/", relative)
        and "/dataset_root/" in relative
        and (
            Path(relative).name.startswith("id_prop")
            or Path(relative).name.startswith("split_manifest")
        )
        and Path(relative).suffix in {".csv", ".json"}
    }
    protocol_3_dataset_root_paths = {
        relative
        for relative in tracked_paths
        if relative.startswith("results/protocol_3/")
        and "/dataset_root/" in relative
    }
    require(not protocol_3_dataset_root_paths, "protocol_3 unexpectedly has tracked dataset-root content")
    root_files: dict[tuple[int, str, int, int], dict[str, str]] = {}
    for relative in tracked_paths:
        match = ROOT_RE.fullmatch(relative)
        if match:
            reject_path(relative)
            key = (int(match["set"]), match["family"], int(match["N"]), int(match["seed"]))
            root_files.setdefault(key, {})[match["name"]] = relative
    require(len(root_files) == 120, "Canonical dataset-root count mismatch")
    require(all(set(files) == {"split_manifest.json", "id_prop.csv"} for files in root_files.values()), "Incomplete canonical dataset root")
    approved_root_paths = {relative for files in root_files.values() for relative in files.values()}
    require(
        root_candidate_paths == approved_root_paths,
        f"Unapproved canonical-like dataset-root inputs detected: {sorted(root_candidate_paths ^ approved_root_paths)}",
    )
    require(
        root_metadata_or_copy_candidates - approved_root_paths
        == set(NONCANONICAL_BACKUP_RELS),
        "Unexpected canonical-like dataset-root metadata/copy candidate",
    )
    root_payloads: dict[tuple[int, str, int, int], dict[str, list[str]]] = {}
    root_recomputed: dict[str, dict[str, Any]] = {}
    structure_reference_total = 0
    for key, files in sorted(root_files.items()):
        set_no, family, n_value, seed = key
        authorize_scientific_input(files["split_manifest.json"])
        authorize_scientific_input(files["id_prop.csv"])
        raw = read_json(files["split_manifest.json"])
        payload = normalize_root_split(raw)
        stats = split_stats(payload)
        expected_val = expected_val_count(n_value)
        expected_tuple = (n_value - expected_val, expected_val, EXPECTED_COUNTS[family]["test"])
        require((stats["n_train"], stats["n_val"], stats["n_test"]) == expected_tuple, f"Root count mismatch: {key}")
        require((raw["n_train"], raw["n_val"], raw["n_test"]) == expected_tuple, f"Root declared count mismatch: {key}")
        require((raw["family"], raw["N"], raw["seed"]) == (family, n_value, seed), f"Root metadata mismatch: {key}")
        require(stats["unique_train"] == stats["n_train"] and stats["unique_val"] == stats["n_val"] and stats["unique_test"] == stats["n_test"], f"Root duplicate: {key}")
        require(not any(stats[name] for name in ("train_val_overlap", "train_test_overlap", "val_test_overlap", "malformed_jid_count")), f"Root split violation: {key}")
        pool_ids = set(payload["id_train"]) | set(payload["id_val"])
        require(pool_ids <= family_sets[family]["pool"], f"Root outside family pool: {key}")
        require(not pool_ids & family_sets[family]["test"], f"Root fixed-test leakage: {key}")
        fixed_test = [row["jid"] for row in family_rows[family]["test"]]
        require(payload["id_test"] == fixed_test, f"Root fixed test mismatch: {key}")
        id_prop_path = ROOT / files["id_prop.csv"]
        with id_prop_path.open(newline="") as handle:
            id_prop = list(csv.reader(handle))
        require(all(len(row) == 2 for row in id_prop), f"id_prop schema mismatch: {key}")
        actual_order = []
        by_jid = {row["jid"]: row for row in family_rows[family]["all"]}
        for filename, target in id_prop:
            match = re.fullmatch(r"POSCAR-(JVASP-[0-9]+)\.vasp", filename)
            require(match is not None, f"id_prop filename mismatch: {key}/{filename}")
            jid = match.group(1)
            actual_order.append(jid)
            require(math.isclose(float(target), float(by_jid[jid]["target"]), rel_tol=0, abs_tol=1e-12), f"id_prop target mismatch: {key}/{jid}")
            structure_rel = (Path(files["id_prop.csv"]).parent / filename).as_posix()
            require(structure_rel in tracked_paths and (ROOT / structure_rel).is_file(), f"Tracked root structure missing: {key}/{filename}")
            structure_reference_total += 1
        expected_order = payload["id_train"] + payload["id_val"] + payload["id_test"]
        require(actual_order == expected_order, f"id_prop order mismatch: {key}")
        root_payloads[key] = payload
        root_recomputed[files["split_manifest.json"]] = {
            "split_manifest_sha256": sha256(ROOT / files["split_manifest.json"]),
            "id_prop_sha256": sha256(id_prop_path),
            "stats": stats,
            "id_prop_rows": len(id_prop),
        }
    require(structure_reference_total == 140760, "Root structure-reference total mismatch")

    # Discover and validate all 240 canonical run split files.
    run_candidate_paths = {
        relative
        for relative in tracked_paths
        if re.match(r"^results/protocol_[123]/", relative)
        and relative.endswith("/ids_train_val_test.json")
    }
    run_matches = []
    for relative in tracked_paths:
        match = RUN_RE.fullmatch(relative)
        if match:
            reject_path(relative)
            run_matches.append((relative, match))
    require(len(run_matches) == 240, "Canonical run count mismatch")
    approved_run_paths = {relative for relative, _ in run_matches}
    require(
        run_candidate_paths == approved_run_paths,
        f"Unapproved canonical-like run split inputs detected: {sorted(run_candidate_paths ^ approved_run_paths)}",
    )
    run_payloads: dict[tuple[int, str, int, int, str], dict[str, list[str]]] = {}
    run_recomputed: dict[str, dict[str, Any]] = {}
    all_run_ids: set[str] = set()
    stale_config_output_dir_count = 0
    for relative, match in sorted(run_matches):
        authorize_scientific_input(relative)
        set_no = int(match["set"])
        family = match["family"]
        n_value = int(match["N"])
        seed = int(match["seed"])
        method_dir = match["method"]
        is_scratch = method_dir.startswith("train_alignn_from_scratch")
        require(not is_scratch or n_value in (50, 500), f"Unexpected scratch N: {relative}")
        if set_no in (1, 2):
            require(method_dir in {"finetune_last2", "train_alignn_from_scratch"}, f"Set {set_no} method mismatch: {relative}")
        else:
            require(method_dir in {"finetune_last2_epochs100_bs32_lr5e5", "train_alignn_from_scratch_epochs100_bs32_lr5e5"}, f"protocol_3 method mismatch: {relative}")
        payload = read_json(relative)
        stats = split_stats(payload)
        expected_val = expected_val_count(n_value)
        expected_tuple = (n_value - expected_val, expected_val, EXPECTED_COUNTS[family]["test"])
        require((stats["n_train"], stats["n_val"], stats["n_test"]) == expected_tuple, f"Run count mismatch: {relative}")
        require(stats["unique_train"] == stats["n_train"] and stats["unique_val"] == stats["n_val"] and stats["unique_test"] == stats["n_test"], f"Run duplicate: {relative}")
        require(not any(stats[name] for name in ("train_val_overlap", "train_test_overlap", "val_test_overlap", "malformed_jid_count")), f"Run split violation: {relative}")
        pool_ids = set(payload["id_train"]) | set(payload["id_val"])
        require(pool_ids <= family_sets[family]["pool"], f"Run outside pool: {relative}")
        require(not pool_ids & family_sets[family]["test"], f"Run test leakage: {relative}")
        fixed_test = [row["jid"] for row in family_rows[family]["test"]]
        require(payload["id_test"] == fixed_test, f"Run fixed test mismatch: {relative}")
        root_key = (set_no if set_no in (1, 2) else 2, family, n_value, seed)
        require(payload == root_payloads[root_key], f"Run/root mismatch: {relative}")
        config_rel = (Path(relative).parent / "config.json").as_posix()
        authorize_scientific_input(config_rel)
        require(config_rel in tracked_paths and (ROOT / config_rel).is_file(), f"Run config missing: {relative}")
        config = read_json(config_rel)
        require((config.get("n_train"), config.get("n_val"), config.get("n_test")) == expected_tuple, f"Run config count mismatch: {relative}")
        method = "from_scratch" if is_scratch else "finetune"
        stale_output_dir = "reproduction_rerun" in str(config.get("output_dir", ""))
        expected_stale_output_dir = set_no == 1 and method == "finetune" and seed in (0, 1, 2)
        require(
            stale_output_dir == expected_stale_output_dir,
            f"Unexpected canonical config output_dir recovery metadata: {config_rel}",
        )
        stale_config_output_dir_count += stale_output_dir
        run_payloads[(set_no, family, n_value, seed, method)] = payload
        all_run_ids.update(payload["id_train"] + payload["id_val"] + payload["id_test"])
        run_recomputed[relative] = {
            "sha256": sha256(ROOT / relative),
            "stats": stats,
            "config_path": config_rel,
            "config_sha256": sha256(ROOT / config_rel),
            "config_size_bytes": (ROOT / config_rel).stat().st_size,
            "config_output_dir_metadata_status": (
                "stale_reproduction_rerun_reference_not_followed"
                if stale_output_dir
                else "historical_output_dir_not_followed"
            ),
        }
    require(stale_config_output_dir_count == 36, "Stale config output_dir reference count mismatch")

    # Cross-set, cross-method, byte identity, and set-based N nesting.
    comparisons: dict[str, dict[str, int]] = {}
    comparisons["protocol_1_protocol_2_finetune"] = compare_group(
        (run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")])
        for f in FAMILIES for n in N_VALUES for s in SEEDS
    )
    comparisons["protocol_2_protocol_3_finetune"] = compare_group(
        (run_payloads[(2, f, n, s, "finetune")], run_payloads[(3, f, n, s, "finetune")])
        for f in FAMILIES for n in N_VALUES for s in SEEDS
    )
    comparisons["protocol_1_protocol_2_from_scratch"] = compare_group(
        (run_payloads[(1, f, n, s, "from_scratch")], run_payloads[(2, f, n, s, "from_scratch")])
        for f in FAMILIES for n in (50, 500) for s in SEEDS
    )
    comparisons["protocol_2_protocol_3_from_scratch"] = compare_group(
        (run_payloads[(2, f, n, s, "from_scratch")], run_payloads[(3, f, n, s, "from_scratch")])
        for f in FAMILIES for n in (50, 500) for s in SEEDS
    )
    comparisons["finetune_from_scratch_within_set"] = compare_group(
        (run_payloads[(set_no, f, n, s, "finetune")], run_payloads[(set_no, f, n, s, "from_scratch")])
        for set_no in (1, 2, 3) for f in FAMILIES for n in (50, 500) for s in SEEDS
    )
    comparisons["protocol_1_promoted_seed012_vs_protocol_2_finetune"] = compare_group(
        (run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")])
        for f in FAMILIES for n in N_VALUES for s in (0, 1, 2)
    )
    comparisons["protocol_1_seed34_vs_protocol_2_finetune"] = compare_group(
        (run_payloads[(1, f, n, s, "finetune")], run_payloads[(2, f, n, s, "finetune")])
        for f in FAMILIES for n in N_VALUES for s in (3, 4)
    )
    require(
        all(result["total_mismatch_count"] == 0 for result in comparisons.values()),
        f"Matched sampling mismatch: {comparisons}",
    )
    root_byte_total = 0
    for family in FAMILIES:
        for n_value in N_VALUES:
            for seed in SEEDS:
                for filename in ("split_manifest.json", "id_prop.csv"):
                    left = ROOT / root_files[(1, family, n_value, seed)][filename]
                    right = ROOT / root_files[(2, family, n_value, seed)][filename]
                    require(left.read_bytes() == right.read_bytes(), f"protocol_1/2 root byte mismatch: {family}/{n_value}/{seed}/{filename}")
                    root_byte_total += 1
    require(root_byte_total == 120, "Root byte comparison total mismatch")
    nesting_total = 0
    nesting_violations = 0
    prefix_nested = 0
    for set_no in (1, 2, 3):
        for family in FAMILIES:
            for seed in SEEDS:
                for smaller, larger in zip(N_VALUES[:-1], N_VALUES[1:]):
                    small = run_payloads[(set_no, family, smaller, seed, "finetune")]
                    large = run_payloads[(set_no, family, larger, seed, "finetune")]
                    small_ids = small["id_train"] + small["id_val"]
                    large_ids = large["id_train"] + large["id_val"]
                    nesting_total += 1
                    nesting_violations += not set(small_ids) <= set(large_ids)
                    prefix_nested += large_ids[: len(small_ids)] == small_ids
    require(nesting_total == 150 and nesting_violations == 0, "Set-based N nesting failure")

    # Legacy files: basic integrity only and always quarantined.
    legacy_matches = []
    for relative in tracked_paths:
        match = LEGACY_RE.fullmatch(relative)
        if match:
            reject_path(relative)
            legacy_matches.append((relative, match))
    require(len(legacy_matches) == 36, "Legacy split-file count mismatch")
    legacy_recomputed: dict[str, dict[str, Any]] = {}
    for relative, _ in sorted(legacy_matches):
        authorize_scientific_input(relative)
        payload = read_json(relative)
        stats = split_stats(payload)
        require(
            all(stats[f"unique_{name}"] == stats[f"n_{name}"] for name in ("train", "val", "test")),
            f"Legacy within-split duplicate: {relative}",
        )
        require(not any(stats[name] for name in ("train_val_overlap", "train_test_overlap", "val_test_overlap", "malformed_jid_count")), f"Legacy basic integrity failure: {relative}")
        legacy_recomputed[relative] = {"sha256": sha256(ROOT / relative), "stats": stats}
    legacy_dependency_references = tracked_reference_paths(
        "(" + "|".join(re.escape(relative) for relative, _ in legacy_matches) + ")"
    )
    require(
        not legacy_dependency_references,
        f"Legacy before-correction fine-tuning path contaminates tracked downstream content: {legacy_dependency_references}",
    )

    require(not EXPECTED_UNASSIGNED & (family_sets["oxide"]["all"] | family_sets["nitride"]["all"]), "Unassigned JID in family manifest")
    require(not EXPECTED_UNASSIGNED & all_run_ids, "Unassigned JID in canonical run")

    # Producer CSV schema, row count, source hash, and critical-field checks.
    input_rows = assert_header(
        "authoritative_dataset_input_inventory.csv",
        [
            "repository_relative_path", "absolute_path", "git_status", "git_blob",
            "size_bytes", "modification_time", "sha256", "row_count", "schema",
            "intended_role", "authority_level",
        ],
    )
    require(len(input_rows) == 24, "Input-inventory row count mismatch")
    require(
        {row["repository_relative_path"] for row in input_rows}
        == EXPECTED_INVENTORY_PATHS,
        "Authoritative input-inventory path set mismatch",
    )
    for row in input_rows:
        relative = row["repository_relative_path"]
        authorize_scientific_input(relative)
        source = ROOT / relative
        require(source.is_file(), f"Inventoried source missing: {source}")
        require(row["absolute_path"] == str(source.resolve()), f"Inventoried absolute path mismatch: {source}")
        require(row["sha256"] == sha256(source), f"Inventoried source hash mismatch: {source}")
        require(int(row["size_bytes"]) == source.stat().st_size, f"Inventoried source size mismatch: {source}")
        require(
            row["modification_time"]
            == datetime.fromtimestamp(source.stat().st_mtime, timezone.utc).isoformat(),
            f"Inventoried modification time mismatch: {source}",
        )
        is_tracked = relative in tracked_paths
        require(
            row["git_status"]
            == ("tracked" if is_tracked else "untracked_authorized_section2_3A"),
            f"Inventoried Git status mismatch: {source}",
        )
        expected_blob = git("rev-parse", f"HEAD:{relative}").strip() if is_tracked else ""
        require(row["git_blob"] == expected_blob, f"Inventoried Git blob mismatch: {source}")
        require(row["intended_role"] and row["authority_level"], f"Inventoried governance fields missing: {source}")
        if source.suffix == ".csv":
            fields, source_rows = read_csv(relative)
            require(int(row["row_count"]) == len(source_rows), f"Inventoried CSV row count mismatch: {source}")
            require(json.loads(row["schema"]) == fields, f"Inventoried CSV schema mismatch: {source}")
        elif relative == OFFICIAL_IDS_REL:
            require(int(row["row_count"]) == 55722, "Official input-inventory row count mismatch")
            require(json.loads(row["schema"]) == sorted(official_ids), "Official input-inventory schema mismatch")
        elif source.suffix == ".json":
            payload = read_json(relative)
            expected_rows = len(payload) if relative == GLOBAL_PATHS["split_json"] else 1
            require(int(row["row_count"]) == expected_rows, f"Inventoried JSON row count mismatch: {source}")
            require(json.loads(row["schema"]) == sorted(payload), f"Inventoried JSON schema mismatch: {source}")
        else:
            require(int(row["row_count"]) == len(source.read_text(errors="replace").splitlines()), f"Inventoried source line count mismatch: {source}")
            require(json.loads(row["schema"]) == "source_lines", f"Inventoried source schema mismatch: {source}")
        if relative in GLOBAL_PATHS.values():
            require(row["intended_role"] == "Global dataset/split authority" and row["authority_level"] == "dataset authority", f"Inventoried global governance mismatch: {source}")
        elif relative in PROCEDURAL_INPUTS:
            require(row["intended_role"] == "Dataset-construction or validation procedure" and row["authority_level"] == "procedural authority", f"Inventoried procedural governance mismatch: {source}")
        elif relative == OFFICIAL_IDS_REL:
            require(row["intended_role"] == "Historical split-ID multiplicity cross-check" and row["authority_level"] == "official provenance cross-check only", "Official inventory governance mismatch")
        else:
            require(row["intended_role"] == "Fixed family manifest or summary" and row["authority_level"] == "dataset authority", f"Inventoried family governance mismatch: {source}")

    root_rows = assert_header(
        "canonical_dataset_root_audit.csv",
        [
            "set", "family", "N", "seed", "split_manifest_path",
            "split_manifest_tracked_status", "split_manifest_size_bytes",
            "split_manifest_sha256", "id_prop_path", "id_prop_tracked_status",
            "id_prop_size_bytes", "id_prop_sha256",
            "n_train", "n_val", "n_test", "unique_train", "unique_val",
            "unique_test", "train_val_overlap", "train_test_overlap",
            "val_test_overlap", "malformed_jid_count", "declared_n_train",
            "declared_n_val", "declared_n_test", "id_prop_row_count",
            "id_prop_exact_order_match", "fixed_test_exact_order_match",
            "family_pool_membership_pass", "structure_references_pass",
            "status", "violations",
        ],
    )
    require(len(root_rows) == 120, "Producer root-audit row count mismatch")
    for row in root_rows:
        recomputed = root_recomputed[row["split_manifest_path"]]
        require(row["split_manifest_sha256"] == recomputed["split_manifest_sha256"], "Producer root manifest hash mismatch")
        require(row["id_prop_sha256"] == recomputed["id_prop_sha256"], "Producer id_prop hash mismatch")
        require(row["split_manifest_tracked_status"] == "tracked" and row["id_prop_tracked_status"] == "tracked", "Producer root tracked status mismatch")
        require(int(row["split_manifest_size_bytes"]) == (ROOT / row["split_manifest_path"]).stat().st_size, "Producer root manifest size mismatch")
        require(int(row["id_prop_size_bytes"]) == (ROOT / row["id_prop_path"]).stat().st_size, "Producer id_prop size mismatch")
        for key, value in recomputed["stats"].items():
            require(int(row[key]) == value, f"Producer root statistic mismatch: {row['split_manifest_path']}/{key}")
        require(row["status"] == "passed" and not row["violations"], "Producer root status failure")
        require(all(bool_value(row[key]) for key in ("id_prop_exact_order_match", "fixed_test_exact_order_match", "family_pool_membership_pass", "structure_references_pass")), "Producer root boolean failure")

    run_rows = assert_header(
        "canonical_run_split_audit.csv",
        [
            "set", "family", "method", "method_directory", "N", "seed",
            "repository_relative_path", "tracked_status", "sha256", "size_bytes",
            "config_path", "config_tracked_status", "config_size_bytes",
            "config_sha256", "config_fields_used",
            "config_output_dir_metadata_status", "n_train",
            "n_val", "n_test", "unique_train", "unique_val", "unique_test",
            "train_val_overlap", "train_test_overlap", "val_test_overlap",
            "malformed_jid_count", "dataset_root_exact_match",
            "fixed_test_exact_order_match", "family_pool_membership_pass",
            "config_counts_match", "status", "violations",
        ],
    )
    require(len(run_rows) == 240, "Producer run-audit row count mismatch")
    for row in run_rows:
        recomputed = run_recomputed[row["repository_relative_path"]]
        require(row["tracked_status"] == "tracked", "Producer run tracked status mismatch")
        require(row["sha256"] == recomputed["sha256"], "Producer run hash mismatch")
        require(int(row["size_bytes"]) == (ROOT / row["repository_relative_path"]).stat().st_size, "Producer run size mismatch")
        require(row["config_path"] == recomputed["config_path"], "Producer config path mismatch")
        require(row["config_tracked_status"] == "tracked", "Producer config tracked status mismatch")
        require(int(row["config_size_bytes"]) == recomputed["config_size_bytes"], "Producer config size mismatch")
        require(row["config_sha256"] == recomputed["config_sha256"], "Producer config hash mismatch")
        require(row["config_fields_used"] == "n_train;n_val;n_test", "Producer config field-use declaration mismatch")
        require(
            row["config_output_dir_metadata_status"]
            == recomputed["config_output_dir_metadata_status"],
            "Producer config output_dir handling mismatch",
        )
        for key, value in recomputed["stats"].items():
            require(int(row[key]) == value, f"Producer run statistic mismatch: {row['repository_relative_path']}/{key}")
        require(row["status"] == "passed" and not row["violations"], "Producer run status failure")
        require(all(bool_value(row[key]) for key in ("dataset_root_exact_match", "fixed_test_exact_order_match", "family_pool_membership_pass", "config_counts_match")), "Producer run boolean failure")

    duplicate_rows = assert_header(
        "duplicate_resolution_audit.csv",
        [
            "jid", "official_occurrence_count", "excess_duplicate_rows",
            "kept_index", "seen_indices", "score_tie",
            "present_once_in_deduplicated_catalog", "resolution",
        ],
    )
    require(len(duplicate_rows) == 5, "Producer duplicate row count mismatch")
    for row in duplicate_rows:
        jid = row["jid"]
        require(int(row["official_occurrence_count"]) == EXPECTED_DUPLICATES[jid], "Producer duplicate multiplicity mismatch")
        require(int(row["excess_duplicate_rows"]) == EXPECTED_DUPLICATES[jid] - 1, "Producer duplicate excess mismatch")
        require(json.loads(row["seen_indices"]) == duplicate_meta["duplicate_examples"][jid]["seen_indices"], "Producer duplicate indices mismatch")
        require(bool_value(row["score_tie"]) and bool_value(row["present_once_in_deduplicated_catalog"]), "Producer duplicate boolean failure")

    family_inventory = assert_header(
        "family_manifest_inventory.csv",
        [
            "family", "manifest", "repository_relative_path", "sha256",
            "row_count", "unique_jid_count", "duplicate_jid_rows",
            "split_counts", "schema", "status",
        ],
    )
    require(len(family_inventory) == 12, "Producer family inventory count mismatch")
    for row in family_inventory:
        source = ROOT / row["repository_relative_path"]
        require(row["sha256"] == sha256(source), "Producer family source hash mismatch")
        require(row["status"] == "passed", "Producer family source status failure")
        if row["manifest"] != "summary":
            expected_count = EXPECTED_COUNTS[row["family"]][row["manifest"]]
            require(int(row["row_count"]) == expected_count and int(row["unique_jid_count"]) == expected_count, "Producer family source count mismatch")
            require(int(row["duplicate_jid_rows"]) == 0, "Producer family duplicate mismatch")

    cross_rows = assert_header("family_cross_membership.csv", ["jid"])
    require(not cross_rows, "Producer reports cross-family membership")
    ox_rows = assert_header("oxynitride_inventory.csv", ["jid", "split", "formula", "target", "filename", "elements"])
    require(ox_rows == oxynitride_rows, "Producer oxynitride inventory mismatch")
    violation_rows = assert_header("run_split_violation_inventory.csv", ["scope", "path", "check", "details"])
    require(not violation_rows, "Producer reports canonical violations")
    legacy_rows = assert_header(
        "legacy_before_correction_split_inventory.csv",
        [
            "family", "N", "seed", "repository_relative_path", "sha256",
            "classification", "schema_pass", "basic_split_integrity_pass",
            "within_split_uniqueness_pass", "paper_evidence_eligible",
            "n_train", "n_val", "n_test",
        ],
    )
    require(len(legacy_rows) == 36, "Producer legacy inventory count mismatch")
    for row in legacy_rows:
        recomputed = legacy_recomputed[row["repository_relative_path"]]
        require(row["sha256"] == recomputed["sha256"], "Producer legacy hash mismatch")
        require(row["classification"] == "legacy_before_correction_quarantined", "Producer legacy classification mismatch")
        require(bool_value(row["schema_pass"]) and bool_value(row["basic_split_integrity_pass"]), "Producer legacy integrity status failure")
        require(bool_value(row["within_split_uniqueness_pass"]), "Producer legacy uniqueness status failure")
        require(not bool_value(row["paper_evidence_eligible"]), "Producer marked legacy evidence eligible")

    exception_rows = assert_header(
        "split_exception_inventory.csv",
        [
            "jid", "formula", "family", "catalog_split", "source_exception",
            "present_in_any_family_manifest", "present_in_any_canonical_run",
            "resolution", "status",
        ],
    )
    require({row["jid"] for row in exception_rows} == EXPECTED_UNASSIGNED, "Producer split-exception set mismatch")
    expected_exception_details = {
        "JVASP-113961": ("Li2BN2", "nitride", "split_conflict:test|train"),
        "JVASP-116461": ("Zn3Mo3O8", "oxide", "split_conflict:test|train"),
        "JVASP-1375": ("Li3N", "nitride", "absent_from_authoritative_split_source"),
    }
    for row in exception_rows:
        require(
            (row["formula"], row["family"], row["source_exception"])
            == expected_exception_details[row["jid"]],
            f"Producer split-exception details differ: {row['jid']}",
        )
    require(all(not bool_value(row["present_in_any_family_manifest"]) and not bool_value(row["present_in_any_canonical_run"]) and row["resolution"] == "explicitly_excluded_unassigned" and row["status"] == "passed" for row in exception_rows), "Producer split-exception resolution failure")

    # Producer JSON schemas and exact critical values.
    expected_global = {
        "catalog_row_count": 55712,
        "catalog_unique_jid_count": 55712,
        "catalog_issue_counts": {
            "malformed_jid": 0,
            "nonfinite_target": 0,
            "target_key": 0,
            "filename": 0,
            "n_atoms": 0,
            "unique_elements": 0,
            "family_flags": 0,
            "split_mapping": 0,
        },
        "assigned_split_count": 55709,
        "split_counts": dict(split_counts),
        "split_issue_counts": {
            "malformed_jid": 0,
            "duplicate_rows": 0,
            "missing_or_invalid_split_label": 0,
            "catalog_jids_missing_from_split_or_explicit_exception": 0,
            "split_jids_missing_from_catalog": 0,
        },
        "split_pairwise_overlaps": {"train_val": 0, "train_test": 0, "val_test": 0},
        "split_csv_json_exact_order_and_mapping_equal": True,
        "unassigned_count": 3,
        "unassigned_jids": sorted(EXPECTED_UNASSIGNED),
        "status": "GLOBAL_DATASET_INTEGRITY_PASSED",
    }
    for name in ("global_dataset_integrity.json", "global_split_integrity.json"):
        payload = json_keys(name, set(expected_global))
        require(payload == expected_global, f"Producer global JSON mismatch: {name}")
    duplicate_json = json_keys(
        "duplicate_resolution_audit.json",
        {
            "original_record_count", "unique_jid_count_after_deduplication",
            "duplicate_jid_count", "excess_duplicate_rows_removed",
            "duplicate_multiplicities", "official_id_source_split_lengths",
            "official_id_source_flattened_count",
            "official_id_source_unique_jid_count",
            "official_duplicate_multiplicities_by_split",
            "catalog_only_jid_absent_from_official_split_source",
            "record_count_reconciliation", "schema_report_top_level_scope",
            "raw_structural_payload_locally_available", "limitation",
        },
    )
    require(
        duplicate_json["original_record_count"] == 55723
        and duplicate_json["unique_jid_count_after_deduplication"] == 55712
        and duplicate_json["duplicate_jid_count"] == 5
        and duplicate_json["excess_duplicate_rows_removed"] == 11
        and duplicate_json["duplicate_multiplicities"] == EXPECTED_DUPLICATES
        and duplicate_json["official_id_source_split_lengths"] == official_split_lengths
        and duplicate_json["official_id_source_flattened_count"] == 55722
        and duplicate_json["official_id_source_unique_jid_count"] == 55711
        and duplicate_json["official_duplicate_multiplicities_by_split"]
        == official_duplicate_by_split
        and duplicate_json["catalog_only_jid_absent_from_official_split_source"]
        == "JVASP-1375"
        and duplicate_json["raw_structural_payload_locally_available"] is False,
        "Producer duplicate JSON mismatch",
    )
    ox_summary = {
        "approved_definition": {
            "oxide": "contains O, including O+N records",
            "nitride": "contains N and does not contain O",
            "oxynitride": "contains both O and N; retained only in oxide arm",
        },
        "oxynitride_count": 499,
        "oxynitride_split_counts": dict(ox_counts),
        "pure_oxide_counts": {
            "all": 14492,
            "train": 11560,
            "val": 1505,
            "test": 1427,
            "pool": 13065,
        },
        "nitride_oxynitride_count": 0,
        "sensitivity_analysis_performed": False,
    }
    require(json_keys("oxynitride_definition_summary.json", set(ox_summary)) == ox_summary, "Producer oxynitride summary mismatch")
    family_definition = json_keys(
        "family_definition_audit.json",
        {
            "family_counts", "family_cross_membership_count",
            "oxynitride_summary", "global_family_predicate_counts_before_split_exclusion",
            "status",
        },
    )
    require(family_definition["family_counts"] == EXPECTED_COUNTS and family_definition["family_cross_membership_count"] == 0 and family_definition["oxynitride_summary"] == ox_summary and family_definition["status"] == "FAMILY_DEFINITIONS_AND_MANIFESTS_PASSED", "Producer family-definition JSON mismatch")
    require(family_definition["global_family_predicate_counts_before_split_exclusion"] == {"oxide": 14992, "nitride": 2290, "neither": 38430}, "Producer global family predicates mismatch")
    structure_json = json_keys("manifest_structure_coverage.json", {"families", "status"})
    require(structure_json == {"families": structure_coverage, "status": "passed"}, "Producer structure-coverage mismatch")
    matched = json_keys(
        "matched_sampling_consistency.json",
        {
            "protocol_1_protocol_2_finetune", "protocol_2_protocol_3_finetune",
            "protocol_1_promoted_seed012_vs_protocol_2_finetune",
            "protocol_1_seed34_vs_protocol_2_finetune",
            "protocol_1_protocol_2_from_scratch", "protocol_2_protocol_3_from_scratch",
            "finetune_from_scratch_within_set", "protocol_1_protocol_2_dataset_root_files",
            "nested_sampling", "status",
        },
    )
    for name, recomputed in comparisons.items():
        require(
            matched[name]["comparison_count"] == recomputed["comparison_count"]
            and matched[name]["membership_mismatch_count"]
            == recomputed["membership_mismatch_count"]
            and matched[name]["ordering_mismatch_count"]
            == recomputed["ordering_mismatch_count"]
            and matched[name]["total_mismatch_count"]
            == recomputed["total_mismatch_count"]
            and not matched[name]["membership_mismatches"]
            and not matched[name]["ordering_mismatches"],
            f"Producer matched-sampling mismatch: {name}",
        )
    require(matched["protocol_1_protocol_2_dataset_root_files"]["comparison_count"] == 120 and matched["protocol_1_protocol_2_dataset_root_files"]["mismatch_count"] == 0, "Producer root-byte comparison mismatch")
    require(matched["nested_sampling"]["comparison_count"] == 150 and matched["nested_sampling"]["set_inclusion_violation_count"] == 0 and matched["nested_sampling"]["prefix_nested_count"] == prefix_nested and matched["nested_sampling"]["prefix_nesting_required"] is False, "Producer nesting summary mismatch")
    require(matched["status"] == "passed", "Producer matched-sampling status failure")
    preflight = json_keys(
        "repository_preflight.json",
        {
            "captured_at", "repository_root", "branch", "head",
            "last_three_commits", "section08_absent", "section08_tracked",
            "nested_checkout_excluded", "baseline_status_count",
            "baseline_tracked_path_count", "baseline_worktree_modified_count",
            "baseline_staged_count",
            "baseline_tracked_or_staged_count", "baseline_untracked_count",
            "baseline_status_categories", "baseline_status_entries",
            "protected_files_before",
        },
    )
    require(preflight["repository_root"] == str(ROOT) and preflight["branch"] == EXPECTED_BRANCH and preflight["head"] == EXPECTED_HEAD, "Producer preflight state mismatch")
    require(preflight["section08_absent"] is True and preflight["section08_tracked"] is False and preflight["nested_checkout_excluded"] is True, "Producer preflight exclusion mismatch")
    require(
        preflight["baseline_status_count"] == 578
        and preflight["baseline_tracked_path_count"] == 0
        and preflight["baseline_worktree_modified_count"] == 0
        and preflight["baseline_staged_count"] == 0
        and preflight["baseline_tracked_or_staged_count"] == 0
        and preflight["baseline_untracked_count"] == 578,
        "Producer preflight status count mismatch",
    )
    require(preflight["baseline_status_categories"] == dict(sorted(baseline_categories.items())), "Producer preflight category mismatch")
    require(
        sorted((row["status"], row["path"]) for row in preflight["baseline_status_entries"])
        == sorted((row["status"], row["path"]) for row in baseline),
        "Protected baseline working-tree entries changed",
    )
    expected_protected_paths = {
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
        (A_REL / "section2_3A_evidence.sha256").as_posix(),
        (A_REL / "section2_3A_validation.json").as_posix(),
    }
    require(
        {record["path"] for record in preflight["protected_files_before"]}
        == expected_protected_paths,
        "Protected-file freeze set mismatch",
    )
    protected_after = []
    for record in preflight["protected_files_before"]:
        relative = record["path"]
        reject_path(relative)
        path = ROOT / relative
        require(path.is_file(), f"Protected file missing: {relative}")
        require(path.stat().st_size == record["size_bytes"], f"Protected file size changed: {relative}")
        require(sha256(path) == record["sha256"], f"Protected file hash changed: {relative}")
        protected_after.append(dict(record))
    decision_text = (EVIDENCE / "dataset_integrity_decision.md").read_text()
    require("SECTION23B_DATASET_INTEGRITY_VALIDATED" in decision_text, "Producer decision verdict missing")
    require("55,723" in decision_text and "55,712" in decision_text and "120 canonical dataset roots" in decision_text and "240 canonical run split files" in decision_text, "Producer decision critical facts missing")

    created_paths = sorted(
        [(EVIDENCE_REL / name).as_posix() for name in PRODUCER_OUTPUTS | FINAL_NAMES]
        + [PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()]
    )
    expected_final_status_entries = sorted(
        baseline
        + [{"status": "??", "path": path} for path in created_paths],
        key=lambda row: (row["path"], row["status"]),
    )
    expected_final_status_categories = dict(
        sorted((baseline_categories + Counter({"section2_3B_outputs": 25})).items())
    )
    metrics = {
        "source_records": 55723,
        "official_id_entries": 55722,
        "official_unique_jids": 55711,
        "catalog_unique": 55712,
        "duplicate_jids": 5,
        "duplicate_excess": 11,
        "assigned_jids": 55709,
        "unassigned_jids": sorted(EXPECTED_UNASSIGNED),
        "global_train": split_counts["train"],
        "global_val": split_counts["val"],
        "global_test": split_counts["test"],
        "oxide_all": len(family_sets["oxide"]["all"]),
        "nitride_all": len(family_sets["nitride"]["all"]),
        "oxynitrides": len(oxynitride_rows),
        "ox_train": ox_counts["train"],
        "ox_val": ox_counts["val"],
        "ox_test": ox_counts["test"],
        "dataset_roots": len(root_files),
        "dataset_roots_passed": len(root_files),
        "canonical_runs": len(run_matches),
        "canonical_runs_passed": len(run_matches),
        "fixed_test_drift_count": 0,
        "run_leakage_count": 0,
        "cross_set_membership_mismatch_count": 0,
        "cross_set_ordering_mismatch_count": 0,
        "cross_method_mismatch_count": 0,
        "legacy_files": len(legacy_matches),
        "legacy_dependency_reference_count": len(legacy_dependency_references),
        "nesting_comparisons": nesting_total,
        "nesting_violations": nesting_violations,
        "root_structure_references": structure_reference_total,
        "canonical_config_count": len(run_matches),
        "stale_seed012_config_output_dir_count": stale_config_output_dir_count,
        "authoritative_hashes": EXPECTED_GLOBAL_SHA,
        "created_paths": created_paths,
        "final_status_count": len(expected_final_status_entries),
        "final_status_categories": expected_final_status_categories,
    }
    checksum_counts = {key: value["files_verified"] for key, value in prior_checksums.items()}
    validation = {
        "schema_version": 1,
        "verdict": VERDICT,
        "repository": {
            "root": str(ROOT),
            "head": EXPECTED_HEAD,
            "branch": EXPECTED_BRANCH,
        },
        "safety": {
            "section08_absent": True,
            "section08_tracked": False,
            "section08_content_inspected": False,
            "nested_checkout_excluded": True,
            "forbidden_inputs_used": 0,
            "tracked_or_staged_baseline_changes": 0,
        },
        "prior_checksum_gates": prior_checksums,
        "section2_3A_exact_gate": {
            "verdict": "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP",
            "checksum_manifest_sha256": SECTION_A_MANIFEST_SHA,
            "status": "passed",
        },
        "independent_recomputation": metrics,
        "producer_package": {
            "required_output_count": len(PRODUCER_OUTPUTS),
            "all_required_outputs_present": True,
            "schemas_validated": True,
            "row_counts_validated": True,
            "embedded_source_hashes_validated": True,
            "critical_values_match": True,
        },
        "checks": {
            "global_counts_splits_duplicates_conflicts_unassigned": "passed",
            "family_counts_definitions_cross_membership_oxynitrides": "passed",
            "canonical_dataset_roots_120": "passed",
            "canonical_runs_240": "passed",
            "split_schema_counts_duplicates_disjointness_fixed_tests_pool": "passed",
            "exact_root_run_matching": "passed",
            "matched_protocol_1_protocol_2_protocol_3_and_method_consistency": "passed",
            "set_based_n_nesting_150": "passed",
            "legacy_quarantine_files_36_basic_integrity": "passed",
            "producer_output_schema_count_hash_validation": "passed",
            "no_canonical_violations": "passed",
            "no_forbidden_inputs": "passed",
            "no_unapproved_dataset_or_result_copy_ingested": "passed",
            "exact_canonical_candidate_universe": "passed",
            "legacy_dependency_contamination_scan": "passed",
            "protected_baseline_and_governance_hashes": "passed",
        },
        "excluded_noncanonical_backup_paths": [
            {
                "repository_relative_path": relative,
                "git_status": "tracked",
                "content_read": False,
                "content_hashed": False,
                "scientific_input": False,
            }
            for relative in NONCANONICAL_BACKUP_RELS
        ],
        "prompt_hash_typo": {
            "path": GLOBAL_PATHS["schema"],
            "prompt_digest_length": 63,
            "actual_sha256": EXPECTED_GLOBAL_SHA[GLOBAL_PATHS["schema"]],
            "actual_digest_length": 64,
            "correction": "append final hexadecimal character 'a'",
        },
        "limitation": (
            "The original 55,723-record structural payload is not tracked locally; "
            "raw duplicate bodies and historical kept indices cannot be replayed. "
            "Official ID multiplicities, deterministic resolution metadata, arithmetic, "
            "and the deduplicated catalog agree."
        ),
        "protected_files_after": protected_after,
        "working_tree": {
            "baseline_entry_count": len(baseline),
            "baseline_exactly_preserved": True,
            "final_status_count": len(expected_final_status_entries),
            "final_untracked_count": len(expected_final_status_entries),
            "final_tracked_modification_count": 0,
            "final_staged_count": 0,
            "final_status_categories": expected_final_status_categories,
            "final_status_entries": expected_final_status_entries,
        },
        "operation_attestations": {
            "git_network_operations": False,
            "git_add_stage_commit_push": False,
            "dataset_or_result_modification": False,
            "manuscript_or_prior_evidence_modification": False,
            "governance_file_modification": False,
        },
        "next_subphase": (
            "Section 2.3C — Oxynitride sensitivity and zero-shot bootstrap uncertainty"
        ),
    }
    write_json("section2_3B_validation.json", validation)
    (EVIDENCE / "section2_3B_report.md").write_text(build_report(metrics, checksum_counts))

    # Manifest every non-recursive evidence output and both scripts.
    input_source_sets = {
        "fixed_authoritative_inputs": sorted(EXPECTED_INVENTORY_PATHS),
        "canonical_dataset_root_metadata": sorted(approved_root_paths),
        "canonical_run_split_files": sorted(approved_run_paths),
        "canonical_run_configs": sorted(
            record["config_path"] for record in run_recomputed.values()
        ),
        "legacy_quarantine_split_files": sorted(
            relative for relative, _ in legacy_matches
        ),
        "producer_evidence_outputs": sorted(
            (EVIDENCE_REL / name).as_posix() for name in PRODUCER_OUTPUTS
        ),
        "preceding_gate_files": [
            "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256",
            "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256",
            (A_REL / "section2_3A_evidence.sha256").as_posix(),
            (A_REL / "section2_3A_validation.json").as_posix(),
            "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md",
        ],
        "protected_governance_and_prior_files": sorted(expected_protected_paths),
        "producer_script": [PRODUCER_REL.as_posix()],
        "section2_3B_scripts": [PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()],
    }
    non_file_input_sources = {
        "git_repository_state": {
            "repository_root": str(ROOT),
            "head": EXPECTED_HEAD,
            "branch": EXPECTED_BRANCH,
            "operations": [
                "git rev-parse HEAD", "git branch --show-current",
                "git log -3", "git status --porcelain=v1 -z --untracked-files=all",
                "git ls-files", "git rev-parse HEAD:<tracked-path>",
            ],
            "role": "Read-only repository identity, index, and working-tree authority",
        }
    }

    def source_sets_for(path: Path) -> list[str]:
        name = path.name
        if path in {ROOT / PRODUCER_REL, ROOT / VALIDATOR_REL}:
            return []
        if name == "repository_preflight.json":
            return [
                "protected_governance_and_prior_files",
                "non_file_input_sources.git_repository_state",
            ]
        if name == "authoritative_dataset_input_inventory.csv":
            return ["fixed_authoritative_inputs"]
        if name.startswith("global_") or name.startswith("duplicate_") or name == "split_exception_inventory.csv":
            return ["fixed_authoritative_inputs", "canonical_run_split_files"]
        if name.startswith("family_") or name.startswith("oxynitride_") or name == "manifest_structure_coverage.json":
            return ["fixed_authoritative_inputs"]
        if name == "canonical_dataset_root_audit.csv":
            return ["fixed_authoritative_inputs", "canonical_dataset_root_metadata"]
        if name == "canonical_run_split_audit.csv":
            return [
                "fixed_authoritative_inputs", "canonical_dataset_root_metadata",
                "canonical_run_split_files", "canonical_run_configs",
            ]
        if name == "legacy_before_correction_split_inventory.csv":
            return ["legacy_quarantine_split_files"]
        if name in {"matched_sampling_consistency.json", "run_split_violation_inventory.csv"}:
            return [
                "canonical_dataset_root_metadata", "canonical_run_split_files",
                "canonical_run_configs",
            ]
        if name == "dataset_integrity_decision.md":
            return [
                "fixed_authoritative_inputs", "canonical_dataset_root_metadata",
                "canonical_run_split_files", "canonical_run_configs",
                "legacy_quarantine_split_files", "producer_script",
            ]
        if name in {"section2_3B_validation.json", "section2_3B_report.md"}:
            return [
                "fixed_authoritative_inputs", "canonical_dataset_root_metadata",
                "canonical_run_split_files", "canonical_run_configs",
                "legacy_quarantine_split_files", "producer_evidence_outputs",
                "preceding_gate_files", "section2_3B_scripts",
            ]
        raise RuntimeError(f"No exact source-set mapping for generated output: {name}")

    manifest_entries = []
    manifest_candidates = sorted(
        [path for path in EVIDENCE.iterdir() if path.is_file() and path.name not in {"generated_output_manifest.json", "section2_3B_evidence.sha256"}]
        + [ROOT / PRODUCER_REL, ROOT / VALIDATOR_REL],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    for path in manifest_candidates:
        relative = path.relative_to(ROOT).as_posix()
        reject_path(relative)
        manifest_entries.append(
            {
                "repository_relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "producer": (
                    PRODUCER_REL.as_posix()
                    if path.name in PRODUCER_OUTPUTS
                    else VALIDATOR_REL.as_posix()
                    if path.name in {"section2_3B_validation.json", "section2_3B_report.md"}
                    else "source"
                ),
                "exact_input_sources": [
                    name
                    if name.startswith("non_file_input_sources.")
                    else f"input_source_sets.{name}"
                    for name in source_sets_for(path)
                ],
                "creation_timestamp": preflight["captured_at"],
                "intended_role": (
                    "Section 2.3B audit or validation evidence"
                    if path.parent == EVIDENCE
                    else "Deterministic Section 2.3B producer/validator source"
                ),
                "authority_level": (
                    "validated audit evidence"
                    if path.parent == EVIDENCE
                    else "procedural authority"
                ),
                "independent_validation_status": "passed",
            }
        )
    output_manifest = {
        "schema_version": 1,
        "generator": VALIDATOR_REL.as_posix(),
        "creation_timestamp_basis": (
            "repository_preflight.json captured_at; frozen for deterministic reruns"
        ),
        "input_source_sets": input_source_sets,
        "non_file_input_sources": non_file_input_sources,
        "derived_presence_checks": {
            "family_structure_files": (
                "Expected repository-relative POSCAR paths are derived from the exact "
                "oxide/nitride all.csv filename columns and compared with tracked HEAD "
                "directory entries; structure contents are not scientific inputs here."
            ),
            "canonical_dataset_root_structure_references": (
                "All 140,760 repository-relative POSCAR references are derived from the "
                "240 exact id_prop.csv inputs and required to exist as tracked files; "
                "their contents are not read or hashed in this split audit."
            ),
        },
        "entries": manifest_entries,
        "self_exclusions": {
            "generated_output_manifest.json": "self-referential hash excluded",
            "section2_3B_evidence.sha256": "checksum manifest excludes itself to avoid recursion",
        },
        "all_required_producer_outputs_present": True,
        "all_entries_forbidden_path_free": True,
    }
    for source_paths in input_source_sets.values():
        for relative in source_paths:
            reject_path(relative)
    write_json("generated_output_manifest.json", output_manifest)
    generated_manifest = json.loads((EVIDENCE / "generated_output_manifest.json").read_text())
    require(len(generated_manifest["entries"]) == 23, "Generated-output manifest entry count mismatch")
    required_manifest_entry_fields = {
        "repository_relative_path", "size_bytes", "sha256", "producer",
        "exact_input_sources", "creation_timestamp", "intended_role",
        "authority_level", "independent_validation_status",
    }
    for entry in generated_manifest["entries"]:
        require(set(entry) == required_manifest_entry_fields, "Generated-output manifest entry schema mismatch")
        require(entry["independent_validation_status"] == "passed", "Generated-output entry is not validated")
        for source_reference in entry["exact_input_sources"]:
            if source_reference.startswith("input_source_sets."):
                require(
                    source_reference.removeprefix("input_source_sets.") in input_source_sets,
                    f"Broken exact-input source-set reference: {source_reference}",
                )
            elif source_reference.startswith("non_file_input_sources."):
                require(
                    source_reference.removeprefix("non_file_input_sources.")
                    in non_file_input_sources,
                    f"Broken non-file source reference: {source_reference}",
                )
            else:
                raise RuntimeError(f"Unscoped exact-input source reference: {source_reference}")

    checksum_candidates = sorted(
        [path for path in EVIDENCE.iterdir() if path.is_file() and path.name != "section2_3B_evidence.sha256"]
        + [ROOT / PRODUCER_REL, ROOT / VALIDATOR_REL],
        key=lambda path: path.relative_to(ROOT).as_posix(),
    )
    checksum_lines = []
    for path in checksum_candidates:
        relative = path.relative_to(ROOT).as_posix()
        reject_path(relative)
        checksum_lines.append(f"{sha256(path)}  {relative}")
    checksum_path = EVIDENCE / "section2_3B_evidence.sha256"
    checksum_path.write_text("\n".join(checksum_lines) + "\n")

    # Final internal checksum and coverage verification.
    seen = set()
    for line in checksum_path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Generated checksum manifest is malformed")
        expected, relative = match.groups()
        reject_path(relative)
        require(relative not in seen, f"Duplicate checksum entry: {relative}")
        seen.add(relative)
        require(sha256(ROOT / relative) == expected, f"Generated checksum mismatch: {relative}")
    expected_seen = {path.relative_to(ROOT).as_posix() for path in checksum_candidates}
    require(seen == expected_seen, "Generated checksum coverage mismatch")
    require(len(seen) == 24, "Generated checksum entry count mismatch")
    require((EVIDENCE_REL / "section2_3B_evidence.sha256").as_posix() not in seen, "Checksum manifest recursively included itself")
    final_status = sorted(
        parse_status(), key=lambda row: (row["path"], row["status"])
    )
    require(final_status == expected_final_status_entries, "Final working-tree status differs from the exact preserved baseline plus Section 2.3B outputs")
    require(all(row["status"] == "??" for row in final_status), "Final tracked or staged path detected")
    final_categories = Counter(status_category(row["path"]) for row in final_status)
    require(dict(sorted(final_categories.items())) == expected_final_status_categories, "Final working-tree category mismatch")
    for record in protected_after:
        path = ROOT / record["path"]
        require(path.stat().st_size == record["size_bytes"], f"Protected file size changed after generation: {record['path']}")
        require(sha256(path) == record["sha256"], f"Protected file hash changed after generation: {record['path']}")

    print(
        json.dumps(
            {
                "verdict": VERDICT,
                "global_unique_jids": metrics["catalog_unique"],
                "canonical_dataset_roots": metrics["dataset_roots"],
                "canonical_runs": metrics["canonical_runs"],
                "canonical_violations": 0,
                "legacy_quarantine_files": metrics["legacy_files"],
                "evidence_checksum_entries": len(seen),
                "status": "VALIDATOR_PASSED",
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
