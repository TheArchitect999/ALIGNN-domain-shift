#!/usr/bin/env python3
"""Independently validate the Section 2.3C sensitivity evidence package.

The validator deliberately does not import the Section 2.3C producer.  It
re-reads the frozen predictions and embedding arrays, reconstructs membership,
replays both bootstrap protocols, and compares every generated numerical row.

Two invocation phases are supported:

* ``--pre-manifest`` validates producer outputs 1--15 and writes only
  ``section2_3C_validation.json``.
* ``--verify-only`` repeats the complete validation after the report and
  manifests exist, verifies all hashes, and writes nothing.

No source, canonical result, prior evidence package, or manuscript is ever
written.  The repository root must be supplied explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPECTED_ROOT = Path(".")
EXPECTED_BRANCH = "main"
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
VALID_VERDICT = "SECTION23C_OXYNITRIDE_BOOTSTRAP_VALIDATED"
BLOCKED_VERDICT = "SECTION23C_BLOCKED"
STABLE = "INTERPRETATION_STABLE"
MATERIAL = "INTERPRETATION_MATERIALLY_CHANGED"

EVIDENCE_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3C_oxynitride_bootstrap"
)
A_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3A_checkpoint_provenance"
)
B_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3B_dataset_integrity"
)
PRODUCER_REL = Path("scripts/shared/analyze_oxynitride_bootstrap.py")
VALIDATOR_REL = Path("scripts/shared/validate_oxynitride_bootstrap.py")

PRODUCER_OUTPUTS = {
    "repository_preflight.json",
    "authoritative_input_inventory.csv",
    "zero_shot_prediction_integrity.json",
    "zero_shot_membership.csv",
    "oxynitride_zero_shot_predictions.csv",
    "zero_shot_point_estimates.csv",
    "zero_shot_bootstrap_replicates.csv",
    "zero_shot_bootstrap_summary.csv",
    "zero_shot_bootstrap_protocol.json",
    "embedding_membership_audit.csv",
    "embedding_sensitivity_metrics.csv",
    "embedding_sensitivity_deltas.csv",
    "embedding_sensitivity_protocol.json",
    "materiality_claim_impact.md",
    "claim_evidence_map.csv",
}
FINAL_OUTPUTS = {
    "section2_3C_validation.json",
    "section2_3C_report.md",
    "generated_output_manifest.json",
    "section2_3C_evidence.sha256",
}
ALL_EVIDENCE_OUTPUTS = PRODUCER_OUTPUTS | FINAL_OUTPUTS

SECTION08_REL = Path("results/derived_evidence/final_paper_factory/archived_submission_materials")
NESTED_NAME = "domain_shift-alignn-domain-shift"

SOURCE_HASHES: dict[str, str] = {
    "results/zero_shot/oxide/predictions.csv":
        "7cf89aa9dbf0384634028f6ec002ec6dc59b112c6d53da60e63759c47112b642",
    "results/zero_shot/nitride/predictions.csv":
        "2753f6f7e2381d11ecadef9cdccbaabb75563138a4def9ac04f8f8983f2f084f",
    "results/zero_shot/zero_shot_summary.csv":
        "e08994903044a1949539703a3f1d153e3e6d367a62d506b543663b92158d9929",
    "data/oxide/manifests/test.csv":
        "1c7a3099270f991fad8a4aab25f5eec23e79f7b696d0211beaacaed2a70c2444",
    "data/nitride/manifests/test.csv":
        "ae72294c15e954a5a143ecdae9cdcaa9f074049b9118248b0e73645cbf22275c",
    (B_REL / "oxynitride_inventory.csv").as_posix():
        "e082a55b33f1f08ecabe4de95e6afbd5c8be04af38d6a7d0871fff3183688b4b",
    (B_REL / "oxynitride_definition_summary.json").as_posix():
        "6a1c10ad23aad63ada3b1a7659ab05eaa39dcb08d94b52dce40e97be625f9f19",
    "scripts/embedding_analysis/06_quantify_family_separation.py":
        "2a44a2fc608eb3311cff455e8270219667029c3651a468611b9733afb23dc93d",
    "results/reproduction/embeddings/tables/family_separation_metrics.csv":
        "e21b8a5216c9f897ba99eceddeb2c5554675a8e081034677ecf59fcaa33485fb",
    "results/embeddings/manifests/family_separation_metrics_manifest.json":
        "543d8e9e31fe99b090f460507b2f3e00cd11486fff89f47ebf0cf241398934a0",
    "results/embeddings/embeddings/test_set/structure_embeddings.npz":
        "3958735acc3b93c42edd59cdf96f9e17a25adfcffe43dec05179be58aadbf57a",
    "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv":
        "88c263772533392df61b781e91b3f024cd7e501a584504c2d80d7e00d2101169",
    "results/embeddings/embeddings/balanced_pool/structure_embeddings.npz":
        "f44d8bb4d2ab6ce78cfa17aa3e6a0ea08790de63ce265b3f2ec99d2673c8fcb2",
    "results/embeddings/embeddings/balanced_pool/structure_embedding_metadata.csv":
        "99e441a80ff80e2f8fa4abdcceb2695f147fdb143ba8164653ce94d664474633",
    "results/embeddings/subsets/fixed_test_set/metadata.csv":
        "b255906ea33f86d836ab30176d692033e4b7c7eaf5879d5e7ecbed9d9b900f26",
    "results/embeddings/subsets/balanced_pool_set/metadata.csv":
        "fadd4bdcad3771d7a0d5759a9b8b60bb2dd22b258cce04252452d1cb2b8c28da",
    "scripts/shared/evaluate_alignn_zero_shot.py":
        "e1c107d7cd670caa28ffa71ed58d770c331bf2cebb7a7cff33493d5324afaeed",
}
TERMINOLOGY_REL = (
    A_REL / "checkpoint_membership_terminology_decision.md"
).as_posix()
PRIOR_GATE_INPUTS = {
    (A_REL / "section2_3A_evidence.sha256").as_posix(),
    (B_REL / "section2_3B_evidence.sha256").as_posix(),
    (B_REL / "section2_3B_validation.json").as_posix(),
    "results/derived_evidence/protocol_1_regeneration/"
    "protocol_1_regeneration_report.md",
    "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256",
    "results/derived_evidence/protocol_1_regeneration/"
    "protocol_1_regeneration_evidence.sha256",
}

A_MANIFEST_SHA = "f815f80cbe705bebcb19d37b484031f74a071859686dda2001c9a56f888769be"
B_MANIFEST_SHA = "a02a95f9cb18f80e79240bc6eaa25c7ffea4ff9d152f333fd2a492a340c78471"
B_VALIDATION_SHA = "b07bbec3e1883003f83ce5eedac9af8ac44dacd6527ff8214cf31ebac8f50648"

SCENARIOS = (
    "inclusive_oxide_vs_nitride",
    "pure_oxide_filtered_vs_nitride",
)
EMBEDDING_SCENARIOS = (
    "inclusive_baseline_recomputed",
    "pure_oxide_filtered",
)
SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
DATASETS: dict[str, dict[str, Any]] = {
    "fixed_test_set": {
        "label": "Fixed test set",
        "folder": "test_set",
        "inclusive": (1484, 242, 57),
        "filtered": (1427, 242, 0),
        "compact": "results/embeddings/subsets/fixed_test_set/metadata.csv",
    },
    "balanced_pool_set": {
        "label": "Balanced train+val pool",
        "folder": "balanced_pool",
        "inclusive": (2046, 2046, 56),
        "filtered": (1990, 2046, 0),
        "compact": "results/embeddings/subsets/balanced_pool_set/metadata.csv",
    },
}
FAMILY_TO_LABEL = {"oxide": 0, "nitride": 1}
LABEL_TO_FAMILY = {0: "oxide", 1: "nitride"}
BOOTSTRAP_REPLICATES = 50_000
EMBEDDING_BOOTSTRAPS = 1_000
BOOTSTRAP_CHUNK = 512

PREDICTION_COLUMNS = ["jid", "filename", "target", "prediction", "abs_error"]
METRIC_ID_COLUMNS = ["dataset", "embedding_source", "metric_name", "metric_scope"]
FROZEN_METRIC_COLUMNS = [
    "dataset", "dataset_label", "embedding_source", "embedding_dim",
    "metric_name", "metric_scope", "value", "ci_low", "ci_high",
    "ci_level", "ci_method", "higher_is_better", "n_structures",
    "n_oxide", "n_nitride", "vector_space", "raw_space_primary",
    "projected_space", "preprocessing", "parameters",
]


class ValidationFailure(RuntimeError):
    """A deterministic Section 2.3C validation failure."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel_text(relative: str | Path) -> str:
    value = Path(relative).as_posix()
    require(not Path(value).is_absolute(), f"Absolute repository path rejected: {value}")
    require(".." not in Path(value).parts, f"Path traversal rejected: {value}")
    require("archived_submission_materials" not in value, f"Section 08 path rejected: {value}")
    require(
        value != NESTED_NAME and not value.startswith(NESTED_NAME + "/"),
        f"Nested-checkout path rejected: {value}",
    )
    require("finetune_last2_reproduction_rerun" not in value, f"Rerun path rejected: {value}")
    require("reproduction_rerun" not in value, f"Rerun-control path rejected: {value}")
    require("__pycache__" not in value, f"Cache path rejected: {value}")
    require(not value.endswith((".pyc", ".pyo", ".tmp", ".temp", "~")),
            f"Temporary/compiled path rejected: {value}")
    return value


def scientific_rel(relative: str | Path) -> str:
    value = rel_text(relative)
    if value.startswith("results/protocol_1/"):
        require(
            value in {
                "results/zero_shot/oxide/predictions.csv",
                "results/zero_shot/nitride/predictions.csv",
            },
            f"Unauthorized legacy result input rejected: {value}",
        )
    require(
        value in SOURCE_HASHES or value == TERMINOLOGY_REL,
        f"Scientific input is not explicitly allowlisted: {value}",
    )
    return value


def inventory_rel(relative: str | Path) -> str:
    """Authorize exactly the frozen scientific inputs and six prior gates."""
    value = rel_text(relative)
    require(
        value in SOURCE_HASHES or value == TERMINOLOGY_REL or value in PRIOR_GATE_INPUTS,
        f"Input-inventory path is not explicitly allowlisted: {value}",
    )
    if value.startswith("results/protocol_1/"):
        scientific_rel(value)
    return value


def root_path(root: Path, relative: str | Path, *, scientific: bool = False) -> Path:
    value = scientific_rel(relative) if scientific else rel_text(relative)
    candidate = root / value
    require(candidate.resolve().is_relative_to(root.resolve()), f"Escaping path rejected: {value}")
    return candidate


def git(root: Path, *args: str, allow_one: bool = False) -> str:
    process = subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    allowed = {0, 1} if allow_one else {0}
    require(process.returncode in allowed,
            f"Read-only git command failed: git {' '.join(args)}")
    return process.stdout.decode("utf-8", errors="replace")


def parse_status(root: Path) -> list[dict[str, str]]:
    process = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    rows: list[dict[str, str]] = []
    for item in process.stdout.split(b"\0"):
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
    if path in {NESTED_NAME, NESTED_NAME + "/"} or path.startswith(NESTED_NAME + "/"):
        return "nested_checkout"
    if path.startswith(A_REL.as_posix() + "/") or path in {
        "scripts/shared/audit_checkpoint_provenance.py",
        "scripts/shared/validate_checkpoint_provenance.py",
    }:
        return "section2_3A_outputs"
    if path.startswith(B_REL.as_posix() + "/") or path in {
        "scripts/shared/audit_dataset_integrity.py",
        "scripts/shared/validate_dataset_integrity.py",
    }:
        return "section2_3B_outputs"
    if path.startswith(EVIDENCE_REL.as_posix() + "/") or path in {
        PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix(),
    }:
        return "section2_3C_outputs"
    return "ambiguous"


def verify_working_tree(root: Path, phase: str) -> dict[str, Any]:
    status = parse_status(root)
    baseline = [row for row in status if status_category(row["path"]) != "section2_3C_outputs"]
    section_c = [row for row in status if status_category(row["path"]) == "section2_3C_outputs"]
    baseline_counts = Counter(status_category(row["path"]) for row in baseline)
    expected = Counter({
        "rerun_staging_files": 504,
        "rerun_configurations": 36,
        "rerun_summary_controls": 2,
        "governance_session": 3,
        "section2_3A_outputs": 32,
        "section2_3B_outputs": 25,
        "nested_checkout": 1,
    })
    require(len(baseline) == 603, f"Protected baseline count differs: {len(baseline)}")
    require(baseline_counts == expected, f"Protected baseline categories differ: {baseline_counts}")
    require(all(row["status"] == "??" for row in baseline),
            "Tracked or staged protected-baseline change detected")
    require(not baseline_counts.get("ambiguous"), "Ambiguous protected path detected")
    require(all(row["status"] == "??" for row in section_c),
            "Tracked or staged Section 2.3C path detected")
    observed = {row["path"].rstrip("/") for row in section_c}
    pre_expected = {
        (EVIDENCE_REL / name).as_posix() for name in PRODUCER_OUTPUTS
    } | {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}
    final_expected = {
        (EVIDENCE_REL / name).as_posix() for name in ALL_EVIDENCE_OUTPUTS
    } | {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}
    expected_paths = pre_expected if phase == "pre-manifest" else final_expected
    require(observed == expected_paths,
            f"Section 2.3C output set differs ({phase}): {sorted(observed ^ expected_paths)}")
    evidence = root / EVIDENCE_REL
    require(evidence.is_dir(), "Section 2.3C evidence directory is missing")
    expected_names = PRODUCER_OUTPUTS if phase == "pre-manifest" else ALL_EVIDENCE_OUTPUTS
    entries = list(evidence.iterdir())
    require(all(entry.is_file() for entry in entries),
            "Unexpected directory exists inside the Section 2.3C workspace")
    observed_names = {entry.name for entry in entries}
    require(observed_names == expected_names,
            f"On-disk Section 2.3C evidence set differs: {sorted(observed_names ^ expected_names)}")
    return {
        "all_status_count": len(status),
        "protected_baseline_count": len(baseline),
        "section2_3C_count": len(section_c),
        "baseline_categories": dict(sorted(baseline_counts.items())),
        "tracked_or_staged_count": sum(row["status"] != "??" for row in status),
        "paths": status,
    }


def verify_section08(root: Path) -> dict[str, Any]:
    section08 = root / SECTION08_REL
    require(not section08.exists(), "Section 08 exists; content was not inspected")
    tracked = git(root, "ls-files", "--", SECTION08_REL.as_posix())
    require(not tracked.strip(), "Section 08 has tracked content; content was not inspected")
    return {"absent": True, "tracked": False, "content_inspected": False}


def read_json(path: Path) -> Any:
    def reject_constant(value: str) -> Any:
        raise ValidationFailure(f"Non-finite JSON constant in {path}: {value}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, f"CSV lacks header: {path}")
        return list(reader.fieldnames), list(reader)


def write_json_once(path: Path, payload: Any) -> None:
    require(not path.exists(), f"Refusing to overwrite validation output: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    require(normalized in {"true", "false", "1", "0", "yes", "no"},
            f"Invalid boolean: {value!r}")
    return normalized in {"true", "1", "yes"}


def finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"Invalid float for {label}: {value!r}") from exc
    require(math.isfinite(result), f"Non-finite float for {label}: {value!r}")
    return result


def integer(value: Any, label: str) -> int:
    try:
        result = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationFailure(f"Invalid integer for {label}: {value!r}") from exc
    return result


def field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise ValidationFailure(f"Required field missing; expected one of {names}")


def close(actual: Any, expected: float, label: str,
          *, atol: float = 1e-12, rtol: float = 1e-10) -> None:
    value = finite_float(actual, label)
    require(math.isclose(value, expected, abs_tol=atol, rel_tol=rtol),
            f"Numeric mismatch for {label}: {value!r} != {expected!r}")


def verify_checksum_manifest(root: Path, relative: str, mode: str,
                             expected_count: int) -> dict[str, Any]:
    manifest = root_path(root, relative)
    require(manifest.is_file(), f"Missing checksum manifest: {relative}")
    checked = 0
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum line {line_number}: {relative}")
        expected_sha, listed = match.groups()
        listed = rel_text(listed)
        if listed.startswith("results/protocol_1/"):
            scientific_rel(listed)
        candidate = root / listed if mode == "root" else manifest.parent / listed
        require(candidate.resolve().is_relative_to(root.resolve()),
                f"Checksum target escapes repository: {listed}")
        require(candidate.is_file(), f"Checksum target missing: {listed}")
        require(sha256(candidate) == expected_sha, f"Checksum mismatch: {listed}")
        checked += 1
    require(checked == expected_count,
            f"Checksum count differs for {relative}: {checked} != {expected_count}")
    return {"path": relative, "verified_entries": checked, "status": "passed"}


def verify_prior_gates(root: Path) -> dict[str, Any]:
    report = root_path(
        root,
        "results/derived_evidence/protocol_1_regeneration/"
        "protocol_1_regeneration_report.md",
    )
    require("protocol_1_REGENERATION_VALIDATED" in report.read_text(encoding="utf-8"),
            "protocol_1 regeneration verdict is missing")
    results = {
        "promotion": verify_checksum_manifest(
            root,
            "results/derived_evidence/protocol_1_promotion/"
            "promotion_evidence.sha256", "root", 19,
        ),
        "regeneration": verify_checksum_manifest(
            root,
            "results/derived_evidence/protocol_1_regeneration/"
            "protocol_1_regeneration_evidence.sha256", "local", 22,
        ),
        "section2_3A": verify_checksum_manifest(
            root, (A_REL / "section2_3A_evidence.sha256").as_posix(), "root", 31,
        ),
        "section2_3B": verify_checksum_manifest(
            root, (B_REL / "section2_3B_evidence.sha256").as_posix(), "root", 24,
        ),
    }
    a_manifest = root_path(root, A_REL / "section2_3A_evidence.sha256")
    b_manifest = root_path(root, B_REL / "section2_3B_evidence.sha256")
    b_validation_path = root_path(root, B_REL / "section2_3B_validation.json")
    require(sha256(a_manifest) == A_MANIFEST_SHA, "Section 2.3A manifest digest differs")
    require(sha256(b_manifest) == B_MANIFEST_SHA, "Section 2.3B manifest digest differs")
    require(sha256(b_validation_path) == B_VALIDATION_SHA,
            "Section 2.3B validation digest differs")
    a_validation = read_json(root_path(root, A_REL / "section2_3A_validation.json"))
    b_validation = read_json(b_validation_path)
    require(a_validation.get("verdict") == "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP",
            "Section 2.3A verdict differs")
    require(b_validation.get("verdict") == "SECTION23B_DATASET_INTEGRITY_VALIDATED",
            "Section 2.3B verdict differs")
    return results


def verify_source_hashes(root: Path) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = root_path(root, relative, scientific=True)
        require(path.is_file(), f"Authoritative input missing: {relative}")
        actual = sha256(path)
        require(actual == expected, f"Authoritative input hash differs: {relative}")
        observed[relative] = actual
    terminology = root_path(root, TERMINOLOGY_REL, scientific=True)
    require(terminology.is_file(), "Terminology decision is missing")
    text = terminology.read_text(encoding="utf-8")
    require("EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED" in text,
            "Terminology decision status differs")
    require("oxide comparator" in text and "nitride target" in text,
            "Approved terminology is missing")
    observed[TERMINOLOGY_REL] = sha256(terminology)
    for relative in sorted(PRIOR_GATE_INPUTS):
        path = root_path(root, relative)
        require(path.is_file(), f"Prior-gate input missing: {relative}")
        observed[relative] = sha256(path)
    return observed


def validate_input_inventory(root: Path, source_hashes: Mapping[str, str]) -> None:
    path = root_path(root, EVIDENCE_REL / "authoritative_input_inventory.csv")
    fields, rows = read_csv_rows(path)
    required_columns = {
        "category", "repository_relative_path", "absolute_path", "git_status",
        "git_blob", "size_bytes", "modification_time", "sha256", "intended_role",
        "authority_level", "schema_or_array_shape", "before_analysis_sha256",
        "after_analysis_sha256",
    }
    require(required_columns <= set(fields),
            f"Input inventory schema lacks: {sorted(required_columns - set(fields))}")
    by_path = {row["repository_relative_path"]: row for row in rows}
    require(len(by_path) == len(rows), "Input inventory has duplicate paths")
    for relative, expected_sha in source_hashes.items():
        require(relative in by_path, f"Input inventory omits: {relative}")
        row = by_path[relative]
        path_value = inventory_rel(relative)
        source = root / path_value
        require(row["absolute_path"] == str(source.resolve()),
                f"Input inventory absolute path differs: {relative}")
        require(integer(row["size_bytes"], relative) == source.stat().st_size,
                f"Input inventory size differs: {relative}")
        for name in ("sha256", "before_analysis_sha256", "after_analysis_sha256"):
            require(row[name] == expected_sha,
                    f"Input inventory {name} differs: {relative}")
        require(row["intended_role"].strip() and row["authority_level"].strip(),
                f"Input inventory governance fields are blank: {relative}")
        require(row["schema_or_array_shape"].strip(),
                f"Input inventory schema/shape is blank: {relative}")
        tracked_probe = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", relative],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        require(tracked_probe.returncode in {0, 1},
                f"Exact tracked-status probe failed: {relative}")
        expected_status = "tracked" if tracked_probe.returncode == 0 else "untracked"
        require(row["git_status"] == expected_status,
                f"Input inventory Git status differs: {relative}")
        if expected_status == "tracked":
            expected_blob = git(root, "rev-parse", f"HEAD:{relative}").strip()
            require(row["git_blob"] == expected_blob,
                    f"Input inventory Git blob differs: {relative}")
        else:
            require(row["git_blob"].strip().lower() in {"", "not_applicable", "n/a", "none"},
                    f"Untracked input has a Git blob: {relative}")
        require(row["modification_time"].strip(),
                f"Input inventory modification time is blank: {relative}")
    require(set(by_path) == set(source_hashes),
            f"Input inventory path set differs: {sorted(set(by_path) ^ set(source_hashes))}")
    for relative in by_path:
        inventory_rel(relative)


def load_prediction_inputs(root: Path) -> dict[str, Any]:
    prediction_data: dict[str, dict[str, Any]] = {}
    for family, expected_count in (("oxide", 1484), ("nitride", 242)):
        pred_rel = f"results/zero_shot/{family}/predictions.csv"
        test_rel = f"data/{family}/manifests/test.csv"
        pred_fields, pred_rows = read_csv_rows(root_path(root, pred_rel, scientific=True))
        test_fields, test_rows = read_csv_rows(root_path(root, test_rel, scientific=True))
        require(pred_fields == PREDICTION_COLUMNS,
                f"{family} prediction schema differs: {pred_fields}")
        require(len(pred_rows) == len(test_rows) == expected_count,
                f"{family} prediction/test row count differs")
        require(len({row["jid"] for row in pred_rows}) == expected_count,
                f"{family} prediction JIDs are not unique")
        errors: list[float] = []
        for index, (prediction, manifest) in enumerate(zip(pred_rows, test_rows, strict=True)):
            require(all(prediction[name].strip() for name in PREDICTION_COLUMNS),
                    f"Missing {family} prediction value at row {index}")
            require(prediction["jid"] == manifest["jid"],
                    f"{family} prediction/test order differs at row {index}")
            require(prediction["filename"] == manifest["filename"],
                    f"{family} prediction filename differs at row {index}")
            target = finite_float(prediction["target"], f"{family} target {index}")
            manifest_target = finite_float(manifest["target"], f"{family} manifest target {index}")
            require(target == manifest_target,
                    f"{family} target differs from test manifest at row {index}")
            predicted = finite_float(prediction["prediction"], f"{family} prediction {index}")
            stored = finite_float(prediction["abs_error"], f"{family} abs_error {index}")
            recomputed = abs(target - predicted)
            require(math.isclose(stored, recomputed, abs_tol=1e-12, rel_tol=1e-12),
                    f"{family} stored absolute error differs at row {index}")
            expected_filename = f"POSCAR-{prediction['jid']}.vasp"
            require(prediction["filename"] == expected_filename,
                    f"{family} JID/filename mismatch at row {index}")
            errors.append(recomputed)
        prediction_data[family] = {
            "pred_rel": pred_rel,
            "test_rel": test_rel,
            "rows": pred_rows,
            "test_rows": test_rows,
            "errors": np.ascontiguousarray(errors, dtype=np.float64),
        }
    oxide_ids = {row["jid"] for row in prediction_data["oxide"]["rows"]}
    nitride_ids = {row["jid"] for row in prediction_data["nitride"]["rows"]}
    require(not oxide_ids & nitride_ids, "Oxide/nitride test JID overlap detected")

    _, summary_rows = read_csv_rows(root_path(root, "results/zero_shot/zero_shot_summary.csv", scientific=True))
    summary = {row["family"]: row for row in summary_rows}
    for family, expected in (("oxide", 0.03418360680813096),
                             ("nitride", 0.06954201496284854)):
        mae = float(prediction_data[family]["errors"].mean(dtype=np.float64))
        require(math.isclose(mae, expected, abs_tol=1e-12, rel_tol=0.0),
                f"{family} recomputed MAE differs from frozen expected value")
        close(summary[family]["mae_eV_per_atom"], mae, f"{family} summary MAE",
              atol=1e-12, rtol=0.0)

    _, inventory = read_csv_rows(root_path(
        root, B_REL / "oxynitride_inventory.csv", scientific=True
    ))
    require(len(inventory) == 499, "Oxynitride inventory count differs")
    require(Counter(row["split"] for row in inventory) == Counter({
        "train": 400, "val": 42, "test": 57,
    }), "Oxynitride inventory split counts differ")
    oxynitride_ids = {row["jid"] for row in inventory}
    test_oxynitride_ids = {row["jid"] for row in inventory if row["split"] == "test"}
    require(test_oxynitride_ids <= oxide_ids, "Test oxynitrides are absent from oxide predictions")
    require(not oxynitride_ids & nitride_ids, "Nitride prediction is classified as oxynitride")
    require(len(test_oxynitride_ids) == 57, "Test oxynitride count differs")
    definition = read_json(root_path(
        root, B_REL / "oxynitride_definition_summary.json", scientific=True
    ))
    require(definition.get("oxynitride_count") == 499, "Oxynitride definition total differs")
    require(definition.get("oxynitride_split_counts") == {"test": 57, "train": 400, "val": 42},
            "Oxynitride definition split counts differ")
    require(definition.get("nitride_oxynitride_count") == 0,
            "Nitride oxynitride count is nonzero")

    oxide_mask = np.array([
        row["jid"] not in test_oxynitride_ids
        for row in prediction_data["oxide"]["rows"]
    ], dtype=bool)
    require(int(oxide_mask.sum()) == 1427, "Pure-oxide prediction count differs")
    prediction_data["oxide"]["pure_mask"] = oxide_mask
    prediction_data["oxynitride_ids"] = oxynitride_ids
    prediction_data["test_oxynitride_ids"] = test_oxynitride_ids
    prediction_data["inventory"] = inventory
    return prediction_data


def expected_point_estimates(data: Mapping[str, Any]) -> dict[str, Any]:
    oxide = data["oxide"]["errors"]
    nitride = data["nitride"]["errors"]
    pure = oxide[data["oxide"]["pure_mask"]]
    removed = oxide[~data["oxide"]["pure_mask"]]
    oxide_mae = float(oxide.mean(dtype=np.float64))
    pure_mae = float(pure.mean(dtype=np.float64))
    nitride_mae = float(nitride.mean(dtype=np.float64))
    removed_mae = float(removed.mean(dtype=np.float64))
    scenarios = {
        SCENARIOS[0]: {
            "oxide_n": len(oxide), "nitride_n": len(nitride),
            "oxide_mae": oxide_mae, "nitride_mae": nitride_mae,
            "difference": nitride_mae - oxide_mae,
            "ratio": nitride_mae / oxide_mae,
        },
        SCENARIOS[1]: {
            "oxide_n": len(pure), "nitride_n": len(nitride),
            "oxide_mae": pure_mae, "nitride_mae": nitride_mae,
            "difference": nitride_mae - pure_mae,
            "ratio": nitride_mae / pure_mae,
        },
    }
    return {
        "scenarios": scenarios,
        "oxynitride_only_n": len(removed),
        "oxynitride_only_mae": removed_mae,
        "filtered_minus_inclusive_oxide_mae": pure_mae - oxide_mae,
        "relative_percent": 100.0 * (pure_mae - oxide_mae) / oxide_mae,
        "difference_delta": scenarios[SCENARIOS[1]]["difference"] - scenarios[SCENARIOS[0]]["difference"],
        "ratio_delta": scenarios[SCENARIOS[1]]["ratio"] - scenarios[SCENARIOS[0]]["ratio"],
    }


def bootstrap_family(values: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    result = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    n = len(values)
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, n, size=(stop - start, n), endpoint=False,
                               dtype=np.int64)
        result[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    return result


def recompute_zero_bootstrap(data: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    master = np.random.SeedSequence(42)
    children = master.spawn(4)
    expected_states = [
        [2684470948, 3757501821, 1691896351, 1126406280],
        [4091952314, 31242083, 366899054, 1794014678],
        [233227757, 2701265274, 3388095807, 2508111505],
        [3276785861, 872644253, 1208066006, 3985109429],
    ]
    for child, expected in zip(children, expected_states, strict=True):
        require(child.generate_state(4).tolist() == expected,
                "NumPy SeedSequence child state differs from frozen protocol")
    oxide = data["oxide"]["errors"]
    pure = oxide[data["oxide"]["pure_mask"]]
    nitride = data["nitride"]["errors"]
    family_vectors = ((oxide, nitride), (pure, nitride))
    output: dict[str, dict[str, np.ndarray]] = {}
    for scenario_index, (scenario, vectors) in enumerate(zip(SCENARIOS, family_vectors, strict=True)):
        print(f"validator zero-shot bootstrap: {scenario} (50,000 replicates)", flush=True)
        oxide_rng = np.random.default_rng(children[2 * scenario_index])
        nitride_rng = np.random.default_rng(children[2 * scenario_index + 1])
        oxide_boot = bootstrap_family(vectors[0], oxide_rng)
        nitride_boot = bootstrap_family(vectors[1], nitride_rng)
        require(np.all(np.isfinite(oxide_boot)) and np.all(oxide_boot > 0.0),
                f"Invalid oxide bootstrap denominator: {scenario}")
        difference = nitride_boot - oxide_boot
        ratio = nitride_boot / oxide_boot
        require(np.all(np.isfinite(difference)) and np.all(np.isfinite(ratio)),
                f"Non-finite bootstrap statistic: {scenario}")
        output[scenario] = {
            "oxide_mae": oxide_boot,
            "nitride_mae": nitride_boot,
            "mae_difference_nitride_minus_oxide": difference,
            "mae_ratio_nitride_over_oxide": ratio,
        }
    return output


def validate_zero_membership_outputs(root: Path, data: Mapping[str, Any]) -> None:
    membership_path = root_path(root, EVIDENCE_REL / "zero_shot_membership.csv")
    fields, rows = read_csv_rows(membership_path)
    required = {
        "family", "jid", "filename", "test_manifest_order", "is_oxynitride",
        "included_in_inclusive_comparison", "included_in_pure_oxide_sensitivity",
        "source_prediction_path", "membership_authority",
    }
    require(required <= set(fields),
            f"Zero-shot membership schema lacks: {sorted(required - set(fields))}")
    require(len(rows) == 1726, "Zero-shot membership must contain 1,726 rows")
    expected_rows: list[dict[str, Any]] = []
    ox_ids = data["test_oxynitride_ids"]
    for family in ("oxide", "nitride"):
        for index, source in enumerate(data[family]["rows"]):
            is_oxy = family == "oxide" and source["jid"] in ox_ids
            expected_rows.append({
                "family": family, "jid": source["jid"], "filename": source["filename"],
                "order": index, "is_oxy": is_oxy,
                "pure": not is_oxy,
                "source": data[family]["pred_rel"],
            })
    for index, (actual, expected) in enumerate(zip(rows, expected_rows, strict=True)):
        require(actual["family"] == expected["family"] and actual["jid"] == expected["jid"],
                f"Zero-shot membership identity/order differs at row {index}")
        require(actual["filename"] == expected["filename"],
                f"Zero-shot membership filename differs at row {index}")
        require(integer(actual["test_manifest_order"], "test_manifest_order") == expected["order"],
                f"Zero-shot membership order differs at row {index}")
        require(parse_bool(actual["is_oxynitride"]) == expected["is_oxy"],
                f"Zero-shot membership oxynitride flag differs at row {index}")
        require(parse_bool(actual["included_in_inclusive_comparison"]),
                f"Inclusive membership is false at row {index}")
        require(parse_bool(actual["included_in_pure_oxide_sensitivity"]) == expected["pure"],
                f"Pure membership differs at row {index}")
        require(actual["source_prediction_path"] == expected["source"],
                f"Prediction authority path differs at row {index}")
        require(actual["membership_authority"].strip(),
                f"Membership authority is blank at row {index}")

    oxy_path = root_path(root, EVIDENCE_REL / "oxynitride_zero_shot_predictions.csv")
    oxy_fields, oxy_rows = read_csv_rows(oxy_path)
    require({"jid", "filename", "target", "prediction"} <= set(oxy_fields),
            "Oxynitride prediction output schema differs")
    expected_oxy = [
        row for row in data["oxide"]["rows"] if row["jid"] in data["test_oxynitride_ids"]
    ]
    require(len(oxy_rows) == len(expected_oxy) == 57,
            "Oxynitride prediction output count differs")
    for index, (actual, expected) in enumerate(zip(oxy_rows, expected_oxy, strict=True)):
        require(actual["jid"] == expected["jid"] and actual["filename"] == expected["filename"],
                f"Oxynitride prediction identity differs at row {index}")
        close(actual["target"], float(expected["target"]), f"oxynitride target {index}")
        close(actual["prediction"], float(expected["prediction"]), f"oxynitride prediction {index}")
        error_value = field(
            actual, "recomputed_abs_error", "recomputed_abs_error_eV_per_atom",
            "abs_error", "absolute_error",
        )
        close(error_value, abs(float(expected["target"]) - float(expected["prediction"])),
              f"oxynitride error {index}")


def validate_point_estimates(root: Path, expected: Mapping[str, Any]) -> None:
    path = root_path(root, EVIDENCE_REL / "zero_shot_point_estimates.csv")
    _, rows = read_csv_rows(path)
    by_scenario = {row.get("scenario", ""): row for row in rows}
    require(len(rows) == 4 and len(by_scenario) == 4,
            "Point-estimate table must contain four unique scenario/diagnostic rows")
    require(set(by_scenario) == {
        *SCENARIOS, "oxynitride_only_descriptive", "sensitivity_filtered_minus_inclusive"
    }, "Point-estimate row identities differ")
    require(all(scenario in by_scenario for scenario in SCENARIOS),
            "Point-estimate table lacks primary scenarios")
    for scenario in SCENARIOS:
        row = by_scenario[scenario]
        values = expected["scenarios"][scenario]
        require(integer(field(row, "oxide_n", "n_oxide"), "oxide_n") == values["oxide_n"],
                f"Point-estimate oxide count differs: {scenario}")
        require(integer(field(row, "nitride_n", "n_nitride"), "nitride_n") == values["nitride_n"],
                f"Point-estimate nitride count differs: {scenario}")
        close(field(row, "oxide_mae_eV_per_atom", "oxide_mae"), values["oxide_mae"],
              f"point oxide MAE {scenario}")
        close(field(row, "nitride_mae_eV_per_atom", "nitride_mae"), values["nitride_mae"],
              f"point nitride MAE {scenario}")
        close(field(row, "mae_difference_nitride_minus_oxide_eV_per_atom", "mae_difference_nitride_minus_oxide", "difference"),
              values["difference"], f"point difference {scenario}")
        close(field(row, "mae_ratio_nitride_over_oxide", "ratio"), values["ratio"],
              f"point ratio {scenario}")
    serialized = json.dumps(rows, sort_keys=True)
    require("oxynitride" in serialized.lower(), "Point-estimate table omits oxynitride-only diagnostic")
    diagnostic = next((row for row in rows if "oxynitride" in row.get("scenario", "").lower()), None)
    require(diagnostic is not None, "Oxynitride-only diagnostic row is missing")
    require(integer(field(diagnostic, "oxide_n", "n_oxide"), "oxynitride-only n")
            == expected["oxynitride_only_n"], "Oxynitride-only count differs")
    close(field(diagnostic, "oxide_mae_eV_per_atom", "oxide_mae", "mae_eV_per_atom", "mae"),
          expected["oxynitride_only_mae"], "oxynitride-only MAE")
    sensitivity = by_scenario.get("sensitivity_filtered_minus_inclusive")
    require(sensitivity is not None, "Filtered-minus-inclusive sensitivity row is missing")
    sensitivity_fields = {
        "filtered_minus_inclusive_oxide_mae": (
            "filtered_minus_inclusive_oxide_mae_eV_per_atom",
            "filtered_minus_inclusive_oxide_mae", "oxide_mae_change",
        ),
        "relative_percent": (
            "percentage_change_relative_to_inclusive_oxide_mae",
            "percent_change_relative_to_inclusive_oxide_mae", "relative_percent",
        ),
        "difference_delta": (
            "filtered_minus_inclusive_difference_eV_per_atom",
            "filtered_minus_inclusive_difference", "difference_delta",
        ),
        "ratio_delta": (
            "filtered_minus_inclusive_ratio", "ratio_delta",
        ),
    }
    expected_sensitivity = {
        "filtered_minus_inclusive_oxide_mae": expected["filtered_minus_inclusive_oxide_mae"],
        "relative_percent": expected["relative_percent"],
        "difference_delta": expected["difference_delta"],
        "ratio_delta": expected["ratio_delta"],
    }
    for name, aliases in sensitivity_fields.items():
        close(field(sensitivity, *aliases), expected_sensitivity[name],
              f"zero-shot sensitivity/{name}")


def validate_bootstrap_protocol(root: Path) -> None:
    protocol = read_json(root_path(root, EVIDENCE_REL / "zero_shot_bootstrap_protocol.json"))
    encoded = json.dumps(protocol, sort_keys=True)
    replicate_count = protocol.get(
        "bootstrap_replicates",
        protocol.get("replicates_per_scenario", protocol.get("replicate_count", "")),
    )
    require(str(replicate_count) == "50000",
            "Bootstrap replicate count differs")
    require(protocol.get("master_seed", protocol.get("seed")) == 42,
            "Bootstrap master seed differs")
    require("SeedSequence" in encoded and "PCG64" in encoded,
            "Bootstrap RNG protocol is incomplete")
    require("percentile" in encoded.lower() and "linear" in encoded.lower(),
            "Bootstrap CI method differs")
    require("structure" in encoded.lower() and "independent" in encoded.lower(),
            "Bootstrap resampling-unit/independence declaration is missing")
    require(all(name in encoded for name in (
        "inclusive_oxide", "inclusive_nitride", "pure_oxide_filtered", "pure_nitride"
    )), "Bootstrap substream names differ")
    require(np.__version__ in encoded, "Bootstrap protocol omits exact NumPy version")
    for state_word in ("2684470948", "4091952314", "233227757", "3276785861"):
        require(state_word in encoded, f"Bootstrap protocol omits child state: {state_word}")


def compare_bootstrap_outputs(root: Path, bootstrap: Mapping[str, Mapping[str, np.ndarray]],
                              points: Mapping[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    path = root_path(root, EVIDENCE_REL / "zero_shot_bootstrap_replicates.csv")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames is not None, "Bootstrap replicate CSV lacks header")
        required_aliases = {
            "scenario": ("scenario",),
            "replicate": ("replicate_id", "replicate"),
            "oxide_n": ("oxide_n", "n_oxide"),
            "nitride_n": ("nitride_n", "n_nitride"),
            "oxide": ("oxide_mae_eV_per_atom", "oxide_mae"),
            "nitride": ("nitride_mae_eV_per_atom", "nitride_mae"),
            "difference": (
                "mae_difference_nitride_minus_oxide_eV_per_atom",
                "mae_difference_nitride_minus_oxide", "difference",
            ),
            "ratio": ("mae_ratio_nitride_over_oxide", "ratio"),
        }
        selected = {
            key: next((name for name in aliases if name in reader.fieldnames), None)
            for key, aliases in required_aliases.items()
        }
        require(all(selected.values()), f"Bootstrap replicate schema differs: {reader.fieldnames}")
        count = 0
        for scenario in SCENARIOS:
            arrays = bootstrap[scenario]
            for replicate in range(BOOTSTRAP_REPLICATES):
                try:
                    row = next(reader)
                except StopIteration as exc:
                    raise ValidationFailure("Bootstrap replicate CSV ended early") from exc
                require(row[selected["scenario"]] == scenario,
                        f"Bootstrap scenario order differs at row {count}")
                observed_id = integer(row[selected["replicate"]], "replicate_id")
                require(observed_id == replicate,
                        f"Bootstrap replicate ID differs at row {count}")
                point_counts = points["scenarios"][scenario]
                require(integer(row[selected["oxide_n"]], "oxide_n") == point_counts["oxide_n"]
                        and integer(row[selected["nitride_n"]], "nitride_n") == point_counts["nitride_n"],
                        f"Bootstrap sample count differs at row {count}")
                close(row[selected["oxide"]], arrays["oxide_mae"][replicate],
                      f"bootstrap oxide row {count}", atol=1e-15, rtol=1e-12)
                close(row[selected["nitride"]], arrays["nitride_mae"][replicate],
                      f"bootstrap nitride row {count}", atol=1e-15, rtol=1e-12)
                close(row[selected["difference"]],
                      arrays["mae_difference_nitride_minus_oxide"][replicate],
                      f"bootstrap difference row {count}", atol=1e-15, rtol=1e-12)
                close(row[selected["ratio"]], arrays["mae_ratio_nitride_over_oxide"][replicate],
                      f"bootstrap ratio row {count}", atol=1e-15, rtol=1e-12)
                count += 1
        require(next(reader, None) is None, "Bootstrap replicate CSV has extra rows")
        require(count == 100_000, "Bootstrap replicate row count differs")

    summary_expected: dict[str, dict[str, dict[str, float]]] = {}
    for scenario in SCENARIOS:
        summary_expected[scenario] = {}
        point = points["scenarios"][scenario]
        point_map = {
            "oxide_mae": point["oxide_mae"],
            "nitride_mae": point["nitride_mae"],
            "mae_difference_nitride_minus_oxide": point["difference"],
            "mae_ratio_nitride_over_oxide": point["ratio"],
        }
        for estimand, values in bootstrap[scenario].items():
            low, high = np.quantile(values, [0.025, 0.975], method="linear")
            summary_expected[scenario][estimand] = {
                "point_estimate": float(point_map[estimand]),
                "bootstrap_mean": float(values.mean(dtype=np.float64)),
                "bootstrap_standard_error": float(values.std(ddof=1, dtype=np.float64)),
                "ci_low": float(low), "ci_high": float(high),
            }
    _, summary_rows = read_csv_rows(root_path(
        root, EVIDENCE_REL / "zero_shot_bootstrap_summary.csv"
    ))
    require(len(summary_rows) == 8, "Bootstrap summary must contain eight rows")
    normalized: dict[tuple[str, str], Mapping[str, str]] = {}
    estimand_aliases = {
        "oxide_mae_eV_per_atom": "oxide_mae",
        "nitride_mae_eV_per_atom": "nitride_mae",
        "mae_difference_nitride_minus_oxide_eV_per_atom": "mae_difference_nitride_minus_oxide",
        "mae_ratio_nitride_over_oxide": "mae_ratio_nitride_over_oxide",
    }
    for row in summary_rows:
        raw = row["estimand"]
        estimand = estimand_aliases.get(raw, raw)
        normalized[(row["scenario"], estimand)] = row
    require(len(normalized) == 8, "Bootstrap summary has duplicate/missing identities")
    for scenario, estimands in summary_expected.items():
        for estimand, values in estimands.items():
            row = normalized.get((scenario, estimand))
            require(row is not None, f"Bootstrap summary row missing: {scenario}/{estimand}")
            for name, expected_value in values.items():
                aliases = {
                    "bootstrap_standard_error": ("bootstrap_standard_error", "bootstrap_se"),
                    "ci_low": ("ci_low", "percentile_2_5"),
                    "ci_high": ("ci_high", "percentile_97_5"),
                }.get(name, (name,))
                close(field(row, *aliases), expected_value,
                      f"bootstrap summary {scenario}/{estimand}/{name}",
                      atol=1e-15, rtol=1e-12)
            require(integer(field(row, "n_replicates", "replicate_count"), "n_replicates") == 50_000,
                    f"Bootstrap summary replicate count differs: {scenario}/{estimand}")
            close(field(row, "confidence_level", "ci_level"), 0.95,
                  f"bootstrap confidence level {scenario}/{estimand}")
            require(row["ci_method"] == "percentile" and row["quantile_method"] == "linear",
                    f"Bootstrap summary CI protocol differs: {scenario}/{estimand}")
            require(integer(row["master_seed"], "master_seed") == 42,
                    f"Bootstrap summary master seed differs: {scenario}/{estimand}")
            scenario_index = SCENARIOS.index(scenario)
            observed_oxide_spawn = re.sub(r"[\[\](), ]", "", row["oxide_spawn_key"])
            observed_nitride_spawn = re.sub(r"[\[\](), ]", "", row["nitride_spawn_key"])
            require(observed_oxide_spawn == str(2 * scenario_index),
                    f"Bootstrap oxide spawn key differs: {scenario}/{estimand}")
            require(observed_nitride_spawn == str(2 * scenario_index + 1),
                    f"Bootstrap nitride spawn key differs: {scenario}/{estimand}")
            expected_counts = points["scenarios"][scenario]
            require(integer(row["oxide_n"], "oxide_n") == expected_counts["oxide_n"]
                    and integer(row["nitride_n"], "nitride_n") == expected_counts["nitride_n"],
                    f"Bootstrap summary sample count differs: {scenario}/{estimand}")
            expected_units = "dimensionless" if estimand == "mae_ratio_nitride_over_oxide" else "eV/atom"
            require(row["units"] == expected_units,
                    f"Bootstrap summary units differ: {scenario}/{estimand}")
    return summary_expected


def validate_prediction_integrity_json(root: Path) -> None:
    payload = read_json(root_path(root, EVIDENCE_REL / "zero_shot_prediction_integrity.json"))
    encoded = json.dumps(payload, sort_keys=True).lower()
    require("1484" in encoded and "242" in encoded and "1427" in encoded and "57" in encoded,
            "Prediction-integrity evidence omits required counts")
    require("passed" in encoded, "Prediction-integrity evidence has no passed status")
    require("1e-12" in encoded or "1e-12" in repr(payload),
            "Prediction-integrity tolerance is not recorded")


def as_bool_series(series: pd.Series) -> pd.Series:
    return series.map(lambda value: parse_bool(value)).astype(bool)


def load_embedding_inputs(root: Path, data: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    oxynitride_ids = set(data["oxynitride_ids"])
    prediction_maps = {
        family: {row["jid"]: row for row in data[family]["rows"]}
        for family in ("oxide", "nitride")
    }
    # Use the same deterministic pandas float parser for both sides of the
    # secondary metadata-prediction drift audit.  The authoritative zero-shot
    # calculations above retain their independent csv-module parse.  Mixing
    # parsers here would turn last-bit parser differences into spurious exact
    # mismatch counts (while leaving the meaningful drift magnitudes unchanged).
    prediction_numeric_maps = {
        family: pd.read_csv(root_path(
            root,
            f"results/zero_shot/{family}/predictions.csv",
            scientific=True,
        )).set_index("jid")["prediction"].to_dict()
        for family in ("oxide", "nitride")
    }
    for dataset_index, (dataset, info) in enumerate(DATASETS.items()):
        folder = info["folder"]
        npz_rel = f"results/embeddings/embeddings/{folder}/structure_embeddings.npz"
        metadata_rel = f"results/embeddings/embeddings/{folder}/structure_embedding_metadata.csv"
        npz_path = root_path(root, npz_rel, scientific=True)
        metadata_path = root_path(root, metadata_rel, scientific=True)
        with np.load(npz_path, allow_pickle=False) as archive:
            require(tuple(archive.files) == SOURCES,
                    f"NPZ keys/order differ for {dataset}: {archive.files}")
            arrays = {key: archive[key] for key in archive.files}
        expected_n = sum(info["inclusive"][:2])
        for source, array in arrays.items():
            require(array.shape == (expected_n, 256),
                    f"Embedding shape differs: {dataset}/{source}/{array.shape}")
            require(array.dtype == np.float32,
                    f"Embedding dtype differs: {dataset}/{source}/{array.dtype}")
            require(np.all(np.isfinite(array)), f"Non-finite embedding: {dataset}/{source}")
        require(np.array_equal(arrays["pre_head"], arrays["last_gcn_pool"]),
                f"pre_head/last_gcn_pool arrays differ: {dataset}")
        metadata = pd.read_csv(metadata_path)
        require(len(metadata) == expected_n * 3,
                f"Embedding metadata row count differs: {dataset}")
        expected_columns = {
            "material_id", "family", "split", "target_formation_energy_peratom",
            "pretrained_prediction", "absolute_error", "embedding_source", "npz_key",
            "embedding_index", "embedding_dim", "filename", "is_oxynitride",
        }
        require(expected_columns <= set(metadata.columns),
                f"Embedding metadata schema differs: {dataset}")
        compact = pd.read_csv(root_path(root, info["compact"], scientific=True))
        require(len(compact) == expected_n, f"Compact metadata count differs: {dataset}")
        compact_ids = compact["material_id"].astype(str).tolist()
        source_frames: dict[str, pd.DataFrame] = {}
        for source in SOURCES:
            frame = metadata.loc[metadata["embedding_source"] == source].copy()
            frame["embedding_index"] = frame["embedding_index"].astype(int)
            frame = frame.sort_values("embedding_index", kind="stable").reset_index(drop=True)
            require(len(frame) == expected_n, f"Source metadata count differs: {dataset}/{source}")
            require(np.array_equal(frame["embedding_index"].to_numpy(), np.arange(expected_n)),
                    f"Embedding indices differ: {dataset}/{source}")
            require(frame["material_id"].is_unique, f"Duplicate material ID: {dataset}/{source}")
            require(frame["material_id"].astype(str).tolist() == compact_ids,
                    f"Compact/expanded metadata order differs: {dataset}/{source}")
            require(set(frame["family"]) == {"oxide", "nitride"},
                    f"Family labels differ: {dataset}/{source}")
            require((frame["npz_key"] == source).all(), f"NPZ key differs: {dataset}/{source}")
            require((frame["embedding_dim"].astype(int) == 256).all(),
                    f"Embedding dimension metadata differs: {dataset}/{source}")
            inventory_flag = frame["material_id"].astype(str).isin(oxynitride_ids)
            metadata_flag = as_bool_series(frame["is_oxynitride"])
            require((inventory_flag == metadata_flag).all(),
                    f"Oxynitride metadata differs from Section 2.3B: {dataset}/{source}")
            require(not (inventory_flag & (frame["family"] == "nitride")).any(),
                    f"Nitride marked oxynitride: {dataset}/{source}")
            counts = Counter(frame["family"])
            expected_oxide, expected_nitride, expected_oxy = info["inclusive"]
            require(counts == Counter({"oxide": expected_oxide, "nitride": expected_nitride}),
                    f"Inclusive family counts differ: {dataset}/{source}")
            require(int(inventory_flag.sum()) == expected_oxy,
                    f"Embedding oxynitride count differs: {dataset}/{source}")
            if dataset == "fixed_test_set":
                require(set(frame["split"]) == {"test"},
                        f"Fixed-test split differs: {source}")
                prediction_drifts: list[float] = []
                prediction_exact_mismatches: list[bool] = []
                for row in frame.itertuples(index=False):
                    authority = prediction_maps[str(row.family)].get(str(row.material_id))
                    require(authority is not None,
                            f"Fixed-test embedding JID absent from prediction authority: {row.material_id}")
                    require(str(row.filename) == authority["filename"],
                            f"Fixed-test embedding filename differs: {row.material_id}")
                    require(float(row.target_formation_energy_peratom) == float(authority["target"]),
                            f"Fixed-test embedding target differs: {row.material_id}")
                    metadata_prediction = float(row.pretrained_prediction)
                    canonical_prediction = float(
                        prediction_numeric_maps[str(row.family)][str(row.material_id)]
                    )
                    require(math.isclose(
                        float(row.absolute_error),
                        abs(float(row.target_formation_energy_peratom) - metadata_prediction),
                        abs_tol=1e-12, rel_tol=1e-12,
                    ), f"Embedding metadata absolute error is internally inconsistent: {row.material_id}")
                    prediction_drifts.append(abs(metadata_prediction - canonical_prediction))
                    prediction_exact_mismatches.append(metadata_prediction != canonical_prediction)
                frame["_prediction_abs_drift"] = np.asarray(prediction_drifts, dtype=np.float64)
                frame["_prediction_exact_mismatch"] = np.asarray(
                    prediction_exact_mismatches, dtype=bool
                )
                # The frozen metadata came from a separate inference pass.  Most
                # differences are float32-scale, but a small known tail is larger.
                # Freeze the exact observed audit instead of silently widening a
                # generic equality tolerance.  NPZ vectors, not these secondary
                # prediction columns, are the embedding-metric authority.
                expected_drift = {
                    "oxide": (891, 0.015854060649871826),
                    "nitride": (179, 0.027367591857910156),
                }
                for family_name, (expected_mismatches, expected_max) in expected_drift.items():
                    family_mask = frame["family"].eq(family_name).to_numpy()
                    mismatches = int(frame.loc[family_mask, "_prediction_exact_mismatch"].sum())
                    maximum = float(frame.loc[family_mask, "_prediction_abs_drift"].max())
                    require(mismatches == expected_mismatches,
                            f"Frozen metadata prediction mismatch count changed: {dataset}/{source}/{family_name}")
                    require(math.isclose(maximum, expected_max, abs_tol=1e-15, rel_tol=0.0),
                            f"Frozen metadata prediction maximum drift changed: {dataset}/{source}/{family_name}")
            else:
                split_counts = Counter(zip(frame["family"], frame["split"]))
                require(split_counts == Counter({
                    ("oxide", "train"): 1839, ("oxide", "val"): 207,
                    ("nitride", "train"): 1837, ("nitride", "val"): 209,
                }), f"Balanced-pool split counts differ: {source}")
            source_frames[source] = frame
        for source in SOURCES[1:]:
            require(source_frames[source]["material_id"].tolist()
                    == source_frames[SOURCES[0]]["material_id"].tolist(),
                    f"Source material ordering differs: {dataset}/{source}")
            require(np.array_equal(
                source_frames[source]["pretrained_prediction"].to_numpy(dtype=float),
                source_frames[SOURCES[0]]["pretrained_prediction"].to_numpy(dtype=float),
                equal_nan=True,
            ), f"Metadata prediction blocks differ: {dataset}/{source}")
        output[dataset] = {
            "dataset_index": dataset_index,
            "label": info["label"], "arrays": arrays, "frames": source_frames,
            "npz_rel": npz_rel, "metadata_rel": metadata_rel,
        }
    return output


def percentile_ci(values: Sequence[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    require(len(array) > 0, "No finite bootstrap values")
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    require(len(values) >= 2, "Too few values for family bootstrap")
    boot = [float(np.mean(values[rng.integers(0, len(values), len(values))]))
            for _ in range(EMBEDDING_BOOTSTRAPS)]
    return percentile_ci(boot)


def bootstrap_stratified_mean_ci(values: np.ndarray, y: np.ndarray,
                                 rng: np.random.Generator) -> tuple[float, float]:
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    require(len(groups) == 2 and all(len(group) >= 2 for group in groups),
            "Invalid groups for stratified mean bootstrap")
    boot: list[float] = []
    for _ in range(EMBEDDING_BOOTSTRAPS):
        sampled_values = []
        for indices in groups:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_values.append(values[sampled])
        boot.append(float(np.mean(np.concatenate(sampled_values))))
    return percentile_ci(boot)


def bootstrap_dbi_ci(x: np.ndarray, y: np.ndarray,
                     rng: np.random.Generator) -> tuple[float, float]:
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    require(len(groups) == 2 and all(len(group) >= 2 for group in groups),
            "Invalid groups for DBI bootstrap")
    boot: list[float] = []
    for _ in range(EMBEDDING_BOOTSTRAPS):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in groups
        ])
        try:
            boot.append(float(davies_bouldin_score(x[sampled], y[sampled])))
        except ValueError:
            continue
    return percentile_ci(boot)


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray,
                     rng: np.random.Generator) -> tuple[float, float]:
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    require(len(groups) == 2 and all(len(group) >= 2 for group in groups),
            "Invalid groups for AUC bootstrap")
    boot: list[float] = []
    for _ in range(EMBEDDING_BOOTSTRAPS):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True) for indices in groups
        ])
        boot.append(float(roc_auc_score(y[sampled], scores[sampled])))
    return percentile_ci(boot)


def knn_purity(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    neighbors = NearestNeighbors(n_neighbors=16, metric="euclidean")
    neighbors.fit(x)
    indices = neighbors.kneighbors(x, return_distance=False)[:, 1:]
    return np.mean(y[indices] == y[:, None], axis=1)


def logistic_auc(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[float, np.ndarray, int]:
    n_splits = min(5, min(int(np.sum(y == label)) for label in np.unique(y)))
    require(n_splits >= 2, "Too few structures for logistic CV")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan, dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=5000,
                random_state=seed + fold_index, solver="lbfgs",
            ),
        )
        model.fit(x[train_index], y[train_index])
        scores[test_index] = model.predict_proba(x[test_index])[:, 1]
    require(np.all(np.isfinite(scores)), "Non-finite logistic OOF score")
    return float(roc_auc_score(y, scores)), scores, n_splits


def metric_row(dataset: str, label: str, source: str, scenario: str,
               y: np.ndarray, metric: str, scope: str, value: float,
               ci: tuple[float, float], ci_method: str, preprocessing: str,
               higher: bool, parameters: Mapping[str, Any]) -> dict[str, Any]:
    n_oxide = int(np.sum(y == 0))
    n_nitride = int(np.sum(y == 1))
    return {
        "scenario": scenario, "dataset": dataset, "dataset_label": label,
        "embedding_source": source, "embedding_dim": 256,
        "metric_name": metric, "metric_scope": scope, "value": float(value),
        "ci_low": float(ci[0]), "ci_high": float(ci[1]), "ci_level": 0.95,
        "ci_method": ci_method, "higher_is_better": higher,
        "n_structures": int(len(y)), "n_oxide": n_oxide, "n_nitride": n_nitride,
        "vector_space": "raw_256d_embedding_vectors", "raw_space_primary": True,
        "projected_space": "none", "preprocessing": preprocessing,
        "parameters": json.dumps(dict(parameters), sort_keys=True),
        "majority_label_baseline": max(n_oxide, n_nitride) / len(y),
    }


def compute_metrics(dataset: str, label: str, source: str, scenario: str,
                    x: np.ndarray, y: np.ndarray, seed: int) -> list[dict[str, Any]]:
    require(x.dtype == np.float64, "Metric input was not promoted to float64")
    require(len(np.unique(y)) == 2, "Embedding scenario does not contain both families")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    sil = silhouette_samples(x, y, metric="euclidean")
    rows.append(metric_row(
        dataset, label, source, scenario, y, "silhouette_score", "overall_family_labels",
        float(np.mean(sil)), bootstrap_stratified_mean_ci(sil, y, rng),
        "stratified_bootstrap_over_per_structure_silhouette_values", "none", True,
        {"metric": "euclidean"},
    ))
    for label_value, family in LABEL_TO_FAMILY.items():
        values = sil[y == label_value]
        rows.append(metric_row(
            dataset, label, source, scenario, y, "silhouette_score", family,
            float(np.mean(values)), bootstrap_mean_ci(values, rng),
            "bootstrap_over_family_per_structure_silhouette_values", "none", True,
            {"metric": "euclidean"},
        ))
    dbi = float(davies_bouldin_score(x, y))
    rows.append(metric_row(
        dataset, label, source, scenario, y, "davies_bouldin_index",
        "overall_family_labels", dbi, bootstrap_dbi_ci(x, y, rng),
        "stratified_bootstrap_recomputed_index", "none", False,
        {"metric": "euclidean"},
    ))
    purity = knn_purity(x, y)
    rows.append(metric_row(
        dataset, label, source, scenario, y, "knn_family_purity",
        "overall_family_labels", float(np.mean(purity)),
        bootstrap_stratified_mean_ci(purity, y, rng),
        "stratified_bootstrap_over_per_structure_purity_values", "none", True,
        {"k_neighbors": 15, "metric": "euclidean", "self_excluded": True},
    ))
    for label_value, family in LABEL_TO_FAMILY.items():
        values = purity[y == label_value]
        rows.append(metric_row(
            dataset, label, source, scenario, y, "knn_family_purity", family,
            float(np.mean(values)), bootstrap_mean_ci(values, rng),
            "bootstrap_over_family_per_structure_purity_values", "none", True,
            {"k_neighbors": 15, "metric": "euclidean", "self_excluded": True},
        ))
    auc, scores, folds = logistic_auc(x, y, seed)
    rows.append(metric_row(
        dataset, label, source, scenario, y, "logistic_regression_family_auc",
        "overall_family_labels", auc, bootstrap_auc_ci(y, scores, rng),
        "stratified_bootstrap_over_cross_validated_out_of_fold_scores",
        "fold_local_standard_scaler_no_dimensionality_reduction", True,
        {"class_weight": "balanced", "cv_folds": folds,
         "positive_label": "nitride", "solver": "lbfgs"},
    ))
    require(len(rows) == 8, "Internal embedding metric row count differs")
    return rows


def recompute_embedding_metrics(embedding: Mapping[str, Any],
                                oxynitride_ids: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    membership: list[dict[str, Any]] = []
    for dataset, info in embedding.items():
        for source_index, source in enumerate(SOURCES):
            frame = info["frames"][source]
            array = info["arrays"][source].astype(np.float64)
            family = frame["family"].map(FAMILY_TO_LABEL).to_numpy(dtype=int)
            is_oxy = frame["material_id"].astype(str).isin(oxynitride_ids).to_numpy()
            keep_masks = {
                EMBEDDING_SCENARIOS[0]: np.ones(len(frame), dtype=bool),
                EMBEDDING_SCENARIOS[1]: ~((family == 0) & is_oxy),
            }
            nitride_ids = frame.loc[family == 1, "material_id"].astype(str).tolist()
            seed = 42 + 1000 * int(info["dataset_index"]) + source_index
            for scenario in EMBEDDING_SCENARIOS:
                keep = keep_masks[scenario]
                x = array[frame["embedding_index"].to_numpy(dtype=int)][keep]
                y = family[keep]
                kept_nitride_ids = frame.loc[keep & (family == 1), "material_id"].astype(str).tolist()
                require(kept_nitride_ids == nitride_ids,
                        f"Nitride embedding membership changed: {dataset}/{source}/{scenario}")
                expected_counts = DATASETS[dataset][
                    "inclusive" if scenario == EMBEDDING_SCENARIOS[0] else "filtered"
                ]
                counts = Counter(y)
                require((counts[0], counts[1]) == expected_counts[:2],
                        f"Filtered embedding counts differ: {dataset}/{source}/{scenario}")
                print(
                    f"validator embedding metrics: {dataset}/{source}/{scenario} "
                    f"(n={len(y)}, seed={seed})",
                    flush=True,
                )
                computed = compute_metrics(
                    dataset, str(info["label"]), source, scenario, x, y, seed
                )
                for row in computed:
                    row["source_seed"] = seed
                rows.extend(computed)
                n_oxide, n_nitride = counts[0], counts[1]
                if dataset == "fixed_test_set":
                    kept_frame = frame.loc[keep]
                    prediction_mismatch_count: int | None = int(
                        kept_frame["_prediction_exact_mismatch"].sum()
                    )
                    prediction_max_drift: float | None = float(
                        kept_frame["_prediction_abs_drift"].max()
                    )
                    prediction_status = "quantified_secondary_metadata_drift"
                else:
                    prediction_mismatch_count = None
                    prediction_max_drift = None
                    prediction_status = "not_applicable_no_canonical_prediction_comparison"
                membership.append({
                    "dataset": dataset, "embedding_source": source, "scenario": scenario,
                    "n_structures": len(y), "n_oxide": n_oxide, "n_nitride": n_nitride,
                    "n_oxynitride": int(np.sum(is_oxy[keep])),
                    "removed_oxynitrides": int(np.sum(~keep & is_oxy)),
                    "majority_label_baseline": max(n_oxide, n_nitride) / len(y),
                    "npz_path": info["npz_rel"], "metadata_path": info["metadata_rel"],
                    "membership_status": "passed",
                    "prediction_exact_mismatch_count": prediction_mismatch_count,
                    "prediction_max_absolute_drift": prediction_max_drift,
                    "prediction_comparison_status": prediction_status,
                })
    require(len(rows) == 96 and len(membership) == 12,
            "Internal embedding output counts differ")
    return rows, membership


def metric_key(row: Mapping[str, Any], include_scenario: bool = True) -> tuple[str, ...]:
    prefix = (str(row["scenario"]),) if include_scenario else ()
    return prefix + tuple(str(row[name]) for name in METRIC_ID_COLUMNS)


def validate_frozen_baseline(root: Path, recomputed: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    frozen = pd.read_csv(root_path(
        root, "results/reproduction/embeddings/tables/family_separation_metrics.csv",
        scientific=True,
    ))
    require(len(frozen) == 48 and list(frozen.columns) == FROZEN_METRIC_COLUMNS,
            "Frozen embedding metric schema/count differs")
    inclusive = {
        metric_key(row, include_scenario=False): row
        for row in recomputed if row["scenario"] == EMBEDDING_SCENARIOS[0]
    }
    require(len(inclusive) == 48, "Recomputed inclusive metric identities differ")
    maxima = {name: 0.0 for name in ("value", "ci_low", "ci_high", "ci_level")}
    for frozen_row in frozen.to_dict(orient="records"):
        key = tuple(str(frozen_row[name]) for name in METRIC_ID_COLUMNS)
        require(key in inclusive, f"Frozen metric identity missing: {key}")
        actual = inclusive[key]
        for name in maxima:
            expected = float(frozen_row[name])
            value = float(actual[name])
            maxima[name] = max(maxima[name], abs(value - expected))
            require(math.isclose(value, expected, abs_tol=1e-12, rel_tol=1e-10),
                    f"Frozen baseline reproduction failed: {key}/{name}")
        for name in (
            "dataset_label", "embedding_dim", "metric_name", "metric_scope",
            "ci_method", "n_structures", "n_oxide", "n_nitride", "vector_space",
            "projected_space", "preprocessing", "parameters",
        ):
            require(str(actual[name]) == str(frozen_row[name]),
                    f"Frozen baseline invariant differs: {key}/{name}")
        require(parse_bool(actual["higher_is_better"]) == parse_bool(frozen_row["higher_is_better"]),
                f"Frozen higher_is_better differs: {key}")
        require(parse_bool(actual["raw_space_primary"]) == parse_bool(frozen_row["raw_space_primary"]),
                f"Frozen raw_space_primary differs: {key}")
    return maxima


def validate_embedding_membership_output(root: Path,
                                         expected: Sequence[Mapping[str, Any]]) -> None:
    fields, rows = read_csv_rows(root_path(root, EVIDENCE_REL / "embedding_membership_audit.csv"))
    require(len(rows) == 12, "Embedding membership audit must contain 12 rows")
    require({
        "prediction_exact_mismatch_count", "prediction_max_absolute_drift",
        "prediction_comparison_status",
    } <= set(fields), "Embedding metadata-prediction drift audit fields are missing")
    for index, (actual, wanted) in enumerate(zip(rows, expected, strict=True)):
        for name in ("dataset", "embedding_source", "scenario", "npz_path", "metadata_path"):
            require(actual[name] == str(wanted[name]),
                    f"Embedding membership field differs at row {index}: {name}")
        for name in ("n_structures", "n_oxide", "n_nitride", "n_oxynitride", "removed_oxynitrides"):
            require(integer(actual[name], name) == int(wanted[name]),
                    f"Embedding membership count differs at row {index}: {name}")
        close(actual["majority_label_baseline"], float(wanted["majority_label_baseline"]),
              f"Embedding majority baseline row {index}")
        require(actual["membership_status"].lower() == "passed",
                f"Embedding membership status differs at row {index}")
        require(actual["prediction_comparison_status"] == wanted["prediction_comparison_status"],
                f"Embedding prediction-comparison status differs at row {index}")
        if wanted["prediction_exact_mismatch_count"] is None:
            require(actual["prediction_exact_mismatch_count"].strip() == ""
                    and actual["prediction_max_absolute_drift"].strip() == "",
                    f"Balanced-pool prediction drift should be not applicable at row {index}")
        else:
            require(integer(actual["prediction_exact_mismatch_count"], "prediction mismatch count")
                    == int(wanted["prediction_exact_mismatch_count"]),
                    f"Embedding prediction mismatch count differs at row {index}")
            close(actual["prediction_max_absolute_drift"],
                  float(wanted["prediction_max_absolute_drift"]),
                  f"Embedding prediction maximum drift row {index}",
                  atol=1e-15, rtol=0.0)


def validate_embedding_protocol(root: Path) -> None:
    protocol = read_json(root_path(root, EVIDENCE_REL / "embedding_sensitivity_protocol.json"))
    encoded = json.dumps(protocol, sort_keys=True)
    for expected in ("1000", "15", "5", "42", "raw_256d_embedding_vectors",
                     "last_alignn_pool", "StandardScaler", "lbfgs"):
        require(expected in encoded, f"Embedding protocol omits: {expected}")
    require("PCA" in encoded or "projected_space_used" in encoded,
            "Embedding protocol does not document projected-space exclusion")
    require("pre_head" in encoded and "last_gcn_pool" in encoded,
            "Embedding protocol omits source identity check")
    source_seeds = protocol.get("source_seeds")
    require(source_seeds == {
        "fixed_test_set": {"pre_head": 42, "last_alignn_pool": 43, "last_gcn_pool": 44},
        "balanced_pool_set": {
            "pre_head": 1042, "last_alignn_pool": 1043, "last_gcn_pool": 1044,
        },
    }, "Embedding protocol source-seed mapping differs")
    require("fresh" in encoded.lower() and "scenario" in encoded.lower(),
            "Embedding protocol does not record per-scenario RNG reset")
    for version in (np.__version__, pd.__version__, __import__("sklearn").__version__):
        require(version in encoded, f"Embedding protocol omits execution version: {version}")


def validate_embedding_metric_outputs(root: Path, expected: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    _, actual_rows = read_csv_rows(root_path(
        root, EVIDENCE_REL / "embedding_sensitivity_metrics.csv"
    ))
    require(len(actual_rows) == 96, "Embedding sensitivity metric count differs")
    require([metric_key(row) for row in actual_rows] == [metric_key(row) for row in expected],
            "Embedding sensitivity metric row order differs")
    by_key = {metric_key(row): row for row in actual_rows}
    require(len(by_key) == 96, "Embedding sensitivity metric identities are not unique")
    for wanted in expected:
        key = metric_key(wanted)
        actual = by_key.get(key)
        require(actual is not None, f"Embedding metric row missing: {key}")
        for name in ("value", "ci_low", "ci_high", "ci_level"):
            close(actual[name], float(wanted[name]), f"embedding metric {key}/{name}")
        for name in ("embedding_dim", "n_structures", "n_oxide", "n_nitride"):
            require(integer(actual[name], name) == int(wanted[name]),
                    f"Embedding metric integer differs: {key}/{name}")
        require(integer(actual["source_seed"], "source_seed") == int(wanted["source_seed"]),
                f"Embedding metric source seed differs: {key}")
        for name in (
            "dataset_label", "ci_method", "vector_space", "projected_space",
            "preprocessing", "parameters",
        ):
            require(str(actual[name]) == str(wanted[name]),
                    f"Embedding metric invariant differs: {key}/{name}")
        require(parse_bool(actual["higher_is_better"]) == bool(wanted["higher_is_better"]),
                f"Embedding metric direction differs: {key}")
        require(parse_bool(actual["raw_space_primary"]),
                f"Embedding metric not marked raw-space primary: {key}")
        if "majority_label_baseline" in actual:
            close(actual["majority_label_baseline"], float(wanted["majority_label_baseline"]),
                  f"embedding metric majority baseline {key}")
    expected_by_key = {metric_key(row): row for row in expected}
    deltas: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for source in SOURCES:
            identities = [
                ("silhouette_score", "overall_family_labels"),
                ("silhouette_score", "oxide"),
                ("silhouette_score", "nitride"),
                ("davies_bouldin_index", "overall_family_labels"),
                ("knn_family_purity", "overall_family_labels"),
                ("knn_family_purity", "oxide"),
                ("knn_family_purity", "nitride"),
                ("logistic_regression_family_auc", "overall_family_labels"),
            ]
            for metric, scope in identities:
                inclusive = expected_by_key[(EMBEDDING_SCENARIOS[0], dataset, source, metric, scope)]
                filtered = expected_by_key[(EMBEDDING_SCENARIOS[1], dataset, source, metric, scope)]
                delta = float(filtered["value"]) - float(inclusive["value"])
                relative = delta / abs(float(inclusive["value"]))
                deltas.append({
                    "dataset": dataset, "embedding_source": source,
                    "metric_name": metric, "metric_scope": scope,
                    "inclusive_value": float(inclusive["value"]),
                    "filtered_value": float(filtered["value"]),
                    "absolute_delta_filtered_minus_inclusive": delta,
                    "relative_delta": relative,
                    "inclusive_ci_low": float(inclusive["ci_low"]),
                    "inclusive_ci_high": float(inclusive["ci_high"]),
                    "filtered_ci_low": float(filtered["ci_low"]),
                    "filtered_ci_high": float(filtered["ci_high"]),
                    "inclusive_n_structures": int(inclusive["n_structures"]),
                    "filtered_n_structures": int(filtered["n_structures"]),
                })
    require(len(deltas) == 48, "Internal embedding delta count differs")
    _, actual_delta_rows = read_csv_rows(root_path(
        root, EVIDENCE_REL / "embedding_sensitivity_deltas.csv"
    ))
    require(len(actual_delta_rows) == 48, "Embedding sensitivity delta count differs")
    require(
        {"direction_of_improvement", "interpretation_impact"}
        <= (set(actual_delta_rows[0]) if actual_delta_rows else set()),
        "Embedding delta interpretation columns are missing",
    )
    actual_delta_map = {
        tuple(row[name] for name in METRIC_ID_COLUMNS): row for row in actual_delta_rows
    }
    require(len(actual_delta_map) == 48, "Embedding delta identities are not unique")
    require(
        [tuple(row[name] for name in METRIC_ID_COLUMNS) for row in actual_delta_rows]
        == [tuple(str(row[name]) for name in METRIC_ID_COLUMNS) for row in deltas],
        "Embedding delta row order differs",
    )
    for wanted in deltas:
        key = tuple(str(wanted[name]) for name in METRIC_ID_COLUMNS)
        row = actual_delta_map.get(key)
        require(row is not None, f"Embedding delta row missing: {key}")
        aliases = {
            "inclusive_value": ("inclusive_value",),
            "filtered_value": ("filtered_value",),
            "absolute_delta_filtered_minus_inclusive": (
                "absolute_delta_filtered_minus_inclusive", "absolute_delta", "delta"
            ),
            "relative_delta": ("relative_delta",),
            "inclusive_ci_low": ("inclusive_ci_low",),
            "inclusive_ci_high": ("inclusive_ci_high",),
            "filtered_ci_low": ("filtered_ci_low",),
            "filtered_ci_high": ("filtered_ci_high",),
        }
        for name, names in aliases.items():
            close(field(row, *names), float(wanted[name]), f"embedding delta {key}/{name}")
        for name in ("inclusive_n_structures", "filtered_n_structures"):
            require(integer(row[name], name) == int(wanted[name]),
                    f"Embedding delta sample count differs: {key}/{name}")
        require(row["direction_of_improvement"].strip()
                and row["interpretation_impact"].strip(),
                f"Embedding delta interpretation is blank: {key}")
    return deltas


def determine_materiality(zero_summary: Mapping[str, Mapping[str, Mapping[str, float]]],
                          embedding_rows: Sequence[Mapping[str, Any]],
                          points: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    inclusive = points["scenarios"][SCENARIOS[0]]
    filtered = points["scenarios"][SCENARIOS[1]]
    zero_flags = {
        "difference_sign_reversal": math.copysign(1.0, inclusive["difference"])
                                    != math.copysign(1.0, filtered["difference"]),
        "difference_zero_inclusion_changed": (
            zero_summary[SCENARIOS[0]]["mae_difference_nitride_minus_oxide"]["ci_low"] <= 0.0
            <= zero_summary[SCENARIOS[0]]["mae_difference_nitride_minus_oxide"]["ci_high"]
        ) != (
            zero_summary[SCENARIOS[1]]["mae_difference_nitride_minus_oxide"]["ci_low"] <= 0.0
            <= zero_summary[SCENARIOS[1]]["mae_difference_nitride_minus_oxide"]["ci_high"]
        ),
        "ratio_one_inclusion_changed": (
            zero_summary[SCENARIOS[0]]["mae_ratio_nitride_over_oxide"]["ci_low"] <= 1.0
            <= zero_summary[SCENARIOS[0]]["mae_ratio_nitride_over_oxide"]["ci_high"]
        ) != (
            zero_summary[SCENARIOS[1]]["mae_ratio_nitride_over_oxide"]["ci_low"] <= 1.0
            <= zero_summary[SCENARIOS[1]]["mae_ratio_nitride_over_oxide"]["ci_high"]
        ),
    }
    by_key = {metric_key(row): row for row in embedding_rows}
    embedding_flags: dict[str, Any] = {}
    changed = False
    for dataset in DATASETS:
        states: dict[str, Any] = {}
        for scenario in EMBEDDING_SCENARIOS:
            base = (scenario, dataset, "last_alignn_pool")
            auc = by_key[base + ("logistic_regression_family_auc", "overall_family_labels")]
            sil = by_key[base + ("silhouette_score", "overall_family_labels")]
            knn = by_key[base + ("knn_family_purity", "overall_family_labels")]
            majority = max(int(knn["n_oxide"]), int(knn["n_nitride"])) / int(knn["n_structures"])
            states[scenario] = {
                "auc_ci_above_half": float(auc["ci_low"]) > 0.5,
                "silhouette_positive": float(sil["value"]) > 0.0,
                "silhouette_ci_contains_zero": float(sil["ci_low"]) <= 0.0 <= float(sil["ci_high"]),
                "knn_above_majority_baseline": float(knn["value"]) > majority,
                "majority_label_baseline": majority,
            }
        comparable = (
            "auc_ci_above_half", "silhouette_positive", "silhouette_ci_contains_zero",
            "knn_above_majority_baseline",
        )
        dataset_changed = any(
            states[EMBEDDING_SCENARIOS[0]][name]
            != states[EMBEDDING_SCENARIOS[1]][name]
            for name in comparable
        )
        changed = changed or dataset_changed
        embedding_flags[dataset] = {"states": states, "threshold_state_changed": dataset_changed}
    material = any(zero_flags.values()) or changed
    return (MATERIAL if material else STABLE), {
        "zero_shot_rules": zero_flags,
        "embedding_rules": embedding_flags,
        "claim_wording_or_classification_change": material,
    }


def validate_materiality_and_claims(root: Path, interpretation: str,
                                    phase: str) -> None:
    materiality_path = root_path(root, EVIDENCE_REL / "materiality_claim_impact.md")
    materiality_text = materiality_path.read_text(encoding="utf-8")
    require(interpretation in materiality_text,
            "Materiality document does not contain independently derived outcome")
    require("nitride" in materiality_text.lower() and "oxide comparator" in materiality_text,
            "Materiality document lacks bounded terminology")
    require("metadata" in materiality_text.lower() and "drift" in materiality_text.lower(),
            "Materiality document omits quantified embedding-metadata prediction drift")
    fields, rows = read_csv_rows(root_path(root, EVIDENCE_REL / "claim_evidence_map.csv"))
    required = {
        "claim_id", "bounded_claim_text", "claim_classification", "scenario",
        "dataset", "embedding_source", "metric_or_estimand", "source_path",
        "row_selector", "units", "point_estimate", "ci_lower", "ci_upper",
        "confidence_level", "recalculation_status", "authority_level",
        "materiality_outcome", "required_caveat", "intended_future_manuscript_location",
    }
    require(required <= set(fields),
            f"Claim-evidence schema lacks: {sorted(required - set(fields))}")
    require(rows and len({row["claim_id"] for row in rows}) == len(rows),
            "Claim IDs are empty or duplicated")
    allowed_classifications = {"robust", "protocol-specific", "correlational", "unresolved"}
    allowed_locations = {"Methods", "Results", "Discussion", "Appendix"}
    prohibited = ("in-distribution", "out-of-distribution", "ood penalty",
                  "causal domain-shift effect", "intrinsic nitride difficulty",
                  "proof that chemistry alone caused")
    for row in rows:
        require(row["claim_classification"] in allowed_classifications,
                f"Invalid claim classification: {row['claim_id']}")
        require(row["intended_future_manuscript_location"] in allowed_locations,
                f"Invalid manuscript location: {row['claim_id']}")
        require(row["materiality_outcome"] == interpretation,
                f"Claim materiality differs: {row['claim_id']}")
        require(row["row_selector"].strip() and row["required_caveat"].strip(),
                f"Claim provenance/caveat is blank: {row['claim_id']}")
        require(row["bounded_claim_text"].strip() and row["recalculation_status"].strip()
                and row["authority_level"].strip() and row["metric_or_estimand"].strip(),
                f"Claim evidence/governance field is blank: {row['claim_id']}")
        source = rel_text(row["source_path"])
        allowed_claim_sources = set(SOURCE_HASHES) | {TERMINOLOGY_REL} | {
            (EVIDENCE_REL / name).as_posix() for name in PRODUCER_OUTPUTS
        }
        require(source in allowed_claim_sources,
                f"Claim source is not explicitly allowlisted: {row['claim_id']}")
        if source.startswith("results/protocol_1/"):
            scientific_rel(source)
        require((root / source).is_file(), f"Claim source is missing: {row['claim_id']}")
        claim_lower = row["bounded_claim_text"].lower()
        require(not any(term in claim_lower for term in prohibited),
                f"Claim uses prohibited terminology: {row['claim_id']}")
        for numeric_name in ("point_estimate", "ci_lower", "ci_upper", "confidence_level"):
            if row[numeric_name].strip():
                finite_float(row[numeric_name], f"{row['claim_id']}/{numeric_name}")
        if row["ci_lower"].strip() or row["ci_upper"].strip():
            require(row["ci_lower"].strip() and row["ci_upper"].strip()
                    and row["confidence_level"].strip(),
                    f"Claim CI is incomplete: {row['claim_id']}")
            lower = finite_float(row["ci_lower"], f"{row['claim_id']}/ci_lower")
            upper = finite_float(row["ci_upper"], f"{row['claim_id']}/ci_upper")
            require(lower <= upper, f"Claim CI is reversed: {row['claim_id']}")
            close(row["confidence_level"], 0.95,
                  f"claim confidence level {row['claim_id']}")
    if phase == "verify-only":
        report = root_path(root, EVIDENCE_REL / "section2_3C_report.md").read_text(encoding="utf-8")
        ids_in_report = set(re.findall(r"C23C-[A-Za-z0-9_-]+", report))
        known_ids = {row["claim_id"] for row in rows}
        require(ids_in_report, "Final report contains no claim-evidence citations")
        require(ids_in_report <= known_ids, "Report cites an unknown claim ID")
        require(interpretation in report and VALID_VERDICT in report,
                "Final report verdict/materiality differs")
        require("metadata" in report.lower() and "drift" in report.lower(),
                "Final report omits embedding-metadata prediction drift limitation")


def validate_repository_preflight(root: Path) -> None:
    preflight = read_json(root_path(root, EVIDENCE_REL / "repository_preflight.json"))
    encoded = json.dumps(preflight, sort_keys=True)
    require(str(root.resolve()) in encoded, "Repository preflight root differs")
    require(EXPECTED_HEAD in encoded and EXPECTED_BRANCH in encoded,
            "Repository preflight branch/HEAD differs")
    require("603" in encoded, "Repository preflight protected baseline count differs")
    require("section08" in encoded.lower() or "section_08" in encoded.lower(),
            "Repository preflight omits Section 08 safety gate")
    commits = preflight.get("last_three_commits")
    require(isinstance(commits, list) and len(commits) == 3,
            "Repository preflight does not record exactly three commits")
    require(EXPECTED_HEAD in json.dumps(commits),
            "Repository preflight commit list does not begin at expected HEAD")
    require("working_tree" in preflight,
            "Repository preflight omits working-tree classification")


def validate_generated_manifest(root: Path) -> None:
    path = root_path(root, EVIDENCE_REL / "generated_output_manifest.json")
    payload = read_json(path)
    entries = payload.get("entries") if isinstance(payload, dict) else None
    require(isinstance(entries, list), "Generated-output manifest lacks entries list")
    require(len(entries) == 19, "Generated-output manifest entry count differs")
    expected_paths = {
        (EVIDENCE_REL / name).as_posix()
        for name in ALL_EVIDENCE_OUTPUTS - {
            "generated_output_manifest.json", "section2_3C_evidence.sha256"
        }
    } | {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}
    observed: set[str] = set()
    for entry in entries:
        relative = field(entry, "repository_relative_path", "path")
        relative = rel_text(str(relative))
        require(relative in expected_paths, f"Unexpected generated-manifest path: {relative}")
        require(relative not in observed, f"Duplicate generated-manifest path: {relative}")
        observed.add(relative)
        target = root / relative
        require(target.is_file(), f"Generated-manifest target missing: {relative}")
        require(integer(entry["size_bytes"], "size_bytes") == target.stat().st_size,
                f"Generated-manifest size differs: {relative}")
        require(entry["sha256"] == sha256(target),
                f"Generated-manifest hash differs: {relative}")
        for name in ("producer", "exact_input_sources", "intended_role",
                     "authority_level", "independent_validation_status"):
            require(name in entry and entry[name] not in (None, "", []),
                    f"Generated-manifest governance field is blank: {relative}/{name}")
        require(str(entry["independent_validation_status"]).lower()
                in {"passed", "validated", "independently_validated"},
                f"Generated-manifest validation status is not passed: {relative}")
        timestamp = str(field(entry, "creation_timestamp", "creation_timestamp_utc"))
        require(timestamp.endswith(("Z", "+00:00")),
                f"Generated-manifest timestamp is not UTC: {relative}")
        try:
            datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValidationFailure(
                f"Generated-manifest timestamp is invalid: {relative}"
            ) from exc
    require(observed == expected_paths,
            f"Generated-output manifest path set differs: {sorted(observed ^ expected_paths)}")
    encoded = json.dumps(payload, sort_keys=True).lower()
    require("self-reference" in encoded or "self_reference" in encoded,
            "Generated manifest does not document self-reference exclusions")


def validate_final_checksum(root: Path) -> None:
    manifest = root_path(root, EVIDENCE_REL / "section2_3C_evidence.sha256")
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == 20, "Section 2.3C checksum entry count differs")
    expected_paths = {
        (EVIDENCE_REL / name).as_posix()
        for name in ALL_EVIDENCE_OUTPUTS - {"section2_3C_evidence.sha256"}
    } | {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}
    observed: set[str] = set()
    for line_number, line in enumerate(lines, 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed Section 2.3C checksum line: {line_number}")
        expected_sha, relative = match.groups()
        relative = rel_text(relative)
        require(relative in expected_paths, f"Unexpected Section 2.3C checksum path: {relative}")
        require(relative not in observed, f"Duplicate Section 2.3C checksum path: {relative}")
        observed.add(relative)
        target = root / relative
        require(target.is_file(), f"Section 2.3C checksum target missing: {relative}")
        require(sha256(target) == expected_sha, f"Section 2.3C checksum mismatch: {relative}")
    require(observed == expected_paths,
            f"Section 2.3C checksum path set differs: {sorted(observed ^ expected_paths)}")


def validate_existing_validation(root: Path, expected_payload: Mapping[str, Any]) -> None:
    actual = read_json(root_path(root, EVIDENCE_REL / "section2_3C_validation.json"))
    require(actual == expected_payload,
            "Covered Section 2.3C validation JSON differs from independent replay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True,
                        help="Exact repository root to validate")
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--pre-manifest", action="store_true",
                       help="Validate producer files 1--15 and write validation JSON once")
    phase.add_argument("--verify-only", action="store_true",
                       help="Replay all validation and hashes without writing any file")
    return parser.parse_args()


def run_validation(root: Path, phase: str) -> dict[str, Any]:
    require(root.resolve() == EXPECTED_ROOT.resolve(), f"Unexpected repository root: {root}")
    require(git(root, "rev-parse", "--show-toplevel").strip() == str(EXPECTED_ROOT),
            "Git top-level differs")
    branch = git(root, "branch", "--show-current").strip()
    head = git(root, "rev-parse", "HEAD").strip()
    require(branch == EXPECTED_BRANCH, f"Unexpected branch: {branch}")
    require(head == EXPECTED_HEAD, f"Unexpected HEAD: {head}")
    section08 = verify_section08(root)
    working_tree = verify_working_tree(root, phase)
    prior_gates = verify_prior_gates(root)
    before_hashes = verify_source_hashes(root)
    validate_repository_preflight(root)
    validate_input_inventory(root, before_hashes)

    prediction_data = load_prediction_inputs(root)
    validate_prediction_integrity_json(root)
    validate_zero_membership_outputs(root, prediction_data)
    points = expected_point_estimates(prediction_data)
    validate_point_estimates(root, points)
    validate_bootstrap_protocol(root)
    bootstrap = recompute_zero_bootstrap(prediction_data)
    zero_summary = compare_bootstrap_outputs(root, bootstrap, points)

    embedding_inputs = load_embedding_inputs(root, prediction_data)
    validate_embedding_protocol(root)
    embedding_rows, membership = recompute_embedding_metrics(
        embedding_inputs, set(prediction_data["oxynitride_ids"])
    )
    validate_embedding_membership_output(root, membership)
    baseline_maxima = validate_frozen_baseline(root, embedding_rows)
    deltas = validate_embedding_metric_outputs(root, embedding_rows)
    interpretation, materiality_rules = determine_materiality(zero_summary, embedding_rows, points)
    validate_materiality_and_claims(root, interpretation, phase)

    after_hashes = verify_source_hashes(root)
    require(before_hashes == after_hashes, "Authoritative input changed during validation")
    final_tree = verify_working_tree(root, phase)
    require(working_tree["paths"] == final_tree["paths"],
            "Working tree changed during independent validation")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "verdict": VALID_VERDICT,
        "interpretation_result": interpretation,
        "repository": {"root": str(root.resolve()), "branch": branch, "head": head},
        "safety": {
            "section08_absent": section08["absent"],
            "section08_tracked": section08["tracked"],
            "section08_content_inspected": section08["content_inspected"],
            "nested_checkout_excluded": True,
            "forbidden_inputs_used": 0,
            "canonical_inputs_modified": False,
        },
        "prior_gates": prior_gates,
        "authoritative_input_hashes": dict(sorted(before_hashes.items())),
        "zero_shot": {
            "prediction_integrity": "passed",
            "inclusive_oxide_n": 1484,
            "pure_oxide_n": 1427,
            "removed_oxynitride_n": 57,
            "nitride_n": 242,
            "nitride_membership_unchanged": True,
            "point_estimates": points,
            "bootstrap_replicates_validated": 100_000,
            "bootstrap_summary": zero_summary,
        },
        "embedding": {
            "membership_rows_validated": len(membership),
            "metric_rows_validated": len(embedding_rows),
            "delta_rows_validated": len(deltas),
            "frozen_baseline_rows_reproduced": 48,
            "frozen_baseline_max_absolute_discrepancy": baseline_maxima,
            "pre_head_last_gcn_arrays_identical": True,
            "fixed_test_secondary_metadata_prediction_drift": {
                "oxide_exact_mismatches": 891,
                "oxide_max_absolute_drift": 0.015854060649871826,
                "nitride_exact_mismatches": 179,
                "nitride_max_absolute_drift": 0.027367591857910156,
                "interpretation": (
                    "quantified provenance drift only; canonical prediction CSVs remain "
                    "zero-shot numerical authorities and NPZ vectors remain embedding authorities"
                ),
            },
        },
        "materiality_rules": materiality_rules,
        "working_tree": {
            "protected_baseline_count": final_tree["protected_baseline_count"],
            # These lifecycle counts are phase-invariant so the covered
            # pre-manifest validation JSON can be replayed byte-for-byte after
            # the report and two manifests are added.
            "pre_manifest_producer_and_script_count": 17,
            "complete_section2_3C_file_count": 21,
            "tracked_or_staged_count": final_tree["tracked_or_staged_count"],
            "baseline_categories": final_tree["baseline_categories"],
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "platform": platform.platform(),
        },
        "operation_attestations": {
            "training_or_inference": False,
            "embedding_extraction": False,
            "dependency_installation_or_download": False,
            "git_add_commit_push_or_network": False,
            "manuscript_or_canonical_result_modification": False,
        },
        "next_phase": "Section 3 — Rebuild the manuscript content",
    }
    return payload


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    phase = "verify-only" if args.verify_only else "pre-manifest"
    try:
        payload = run_validation(root, phase)
        validation_path = root_path(root, EVIDENCE_REL / "section2_3C_validation.json")
        if phase == "pre-manifest":
            write_json_once(validation_path, payload)
            print(json.dumps({
                "verdict": payload["verdict"],
                "interpretation_result": payload["interpretation_result"],
                "validation_json": validation_path.relative_to(root).as_posix(),
                "mode": phase,
            }, indent=2, allow_nan=False))
        else:
            validate_existing_validation(root, payload)
            validate_generated_manifest(root)
            validate_final_checksum(root)
            # Hash and manifest verification must be last and read-only.  Confirm
            # that neither status nor any authoritative source changed.
            verify_working_tree(root, phase)
            verify_source_hashes(root)
            print(json.dumps({
                "verdict": payload["verdict"],
                "interpretation_result": payload["interpretation_result"],
                "checksum_entries_verified": 20,
                "generated_manifest_entries_verified": 19,
                "mode": phase,
                "writes_performed": 0,
            }, indent=2, allow_nan=False))
    except (ValidationFailure, FileNotFoundError, KeyError, ValueError,
            json.JSONDecodeError, csv.Error) as exc:
        print(json.dumps({
            "verdict": BLOCKED_VERDICT,
            "mode": phase,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "writes_performed": 0,
        }, indent=2, allow_nan=False), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
