#!/usr/bin/env python3
"""Produce the isolated Section 2.3C oxynitride/bootstrap evidence package.

This script is intentionally allowlist driven.  It reads frozen predictions,
manifests, and embeddings and writes only the dedicated Section 2.3C evidence
directory.  It never runs ALIGNN, extracts embeddings, or edits source data.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import davies_bouldin_score, roc_auc_score, silhouette_samples
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


EXPECTED_ROOT = Path(".")
EXPECTED_BRANCH = "main"
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
OUT_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3C_oxynitride_bootstrap"
)
ANALYZE_REL = Path("scripts/shared/analyze_oxynitride_bootstrap.py")
VALIDATE_REL = Path("scripts/shared/validate_oxynitride_bootstrap.py")
SECTION08_REL = Path("results/derived_evidence/final_paper_factory/archived_submission_materials")
NESTED_REL = Path("domain_shift-alignn-domain-shift")

SUCCESS = "SECTION23C_OXYNITRIDE_BOOTSTRAP_VALIDATED"
STABLE = "INTERPRETATION_STABLE"
MATERIAL = "INTERPRETATION_MATERIALLY_CHANGED"

PREDICTION_SCHEMA = ["jid", "filename", "target", "prediction", "abs_error"]
SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
FAMILY_TO_LABEL = {"oxide": 0, "nitride": 1}
LABEL_TO_FAMILY = {0: "oxide", 1: "nitride"}
DATASETS = {
    "fixed_test_set": {
        "label": "Fixed test set",
        "npz": "results/embeddings/embeddings/test_set/structure_embeddings.npz",
        "metadata": "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv",
        "compact": "results/embeddings/subsets/fixed_test_set/metadata.csv",
        "expected": (1726, 1484, 242, 57, 1669, 1427, 242),
    },
    "balanced_pool_set": {
        "label": "Balanced train+val pool",
        "npz": "results/embeddings/embeddings/balanced_pool/structure_embeddings.npz",
        "metadata": "results/embeddings/embeddings/balanced_pool/structure_embedding_metadata.csv",
        "compact": "results/embeddings/subsets/balanced_pool_set/metadata.csv",
        "expected": (4092, 2046, 2046, 56, 4036, 1990, 2046),
    },
}

EXPECTED_HASHES = {
    "results/zero_shot/oxide/predictions.csv": "7cf89aa9dbf0384634028f6ec002ec6dc59b112c6d53da60e63759c47112b642",
    "results/zero_shot/nitride/predictions.csv": "2753f6f7e2381d11ecadef9cdccbaabb75563138a4def9ac04f8f8983f2f084f",
    "results/zero_shot/zero_shot_summary.csv": "e08994903044a1949539703a3f1d153e3e6d367a62d506b543663b92158d9929",
    "data/oxide/manifests/test.csv": "1c7a3099270f991fad8a4aab25f5eec23e79f7b696d0211beaacaed2a70c2444",
    "data/nitride/manifests/test.csv": "ae72294c15e954a5a143ecdae9cdcaa9f074049b9118248b0e73645cbf22275c",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_inventory.csv": "e082a55b33f1f08ecabe4de95e6afbd5c8be04af38d6a7d0871fff3183688b4b",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json": "6a1c10ad23aad63ada3b1a7659ab05eaa39dcb08d94b52dce40e97be625f9f19",
    "scripts/embedding_analysis/06_quantify_family_separation.py": "2a44a2fc608eb3311cff455e8270219667029c3651a468611b9733afb23dc93d",
    "results/reproduction/embeddings/tables/family_separation_metrics.csv": "e21b8a5216c9f897ba99eceddeb2c5554675a8e081034677ecf59fcaa33485fb",
    "results/embeddings/manifests/family_separation_metrics_manifest.json": "543d8e9e31fe99b090f460507b2f3e00cd11486fff89f47ebf0cf241398934a0",
    "results/embeddings/embeddings/test_set/structure_embeddings.npz": "3958735acc3b93c42edd59cdf96f9e17a25adfcffe43dec05179be58aadbf57a",
    "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv": "88c263772533392df61b781e91b3f024cd7e501a584504c2d80d7e00d2101169",
    "results/embeddings/embeddings/balanced_pool/structure_embeddings.npz": "f44d8bb4d2ab6ce78cfa17aa3e6a0ea08790de63ce265b3f2ec99d2673c8fcb2",
    "results/embeddings/embeddings/balanced_pool/structure_embedding_metadata.csv": "99e441a80ff80e2f8fa4abdcceb2695f147fdb143ba8164653ce94d664474633",
    "results/embeddings/subsets/fixed_test_set/metadata.csv": "b255906ea33f86d836ab30176d692033e4b7c7eaf5879d5e7ecbed9d9b900f26",
    "results/embeddings/subsets/balanced_pool_set/metadata.csv": "fadd4bdcad3771d7a0d5759a9b8b60bb2dd22b258cce04252452d1cb2b8c28da",
    "scripts/shared/evaluate_alignn_zero_shot.py": "e1c107d7cd670caa28ffa71ed58d770c331bf2cebb7a7cff33493d5324afaeed",
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_evidence.sha256": "f815f80cbe705bebcb19d37b484031f74a071859686dda2001c9a56f888769be",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_evidence.sha256": "a02a95f9cb18f80e79240bc6eaa25c7ffea4ff9d152f333fd2a492a340c78471",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json": "b07bbec3e1883003f83ce5eedac9af8ac44dacd6527ff8214cf31ebac8f50648",
}

INPUT_ROLES = {
    "results/zero_shot/oxide/predictions.csv": ("zero_shot_prediction", "oxide per-structure predictions", "numerical authority"),
    "results/zero_shot/nitride/predictions.csv": ("zero_shot_prediction", "nitride per-structure predictions", "numerical authority"),
    "results/zero_shot/zero_shot_summary.csv": ("zero_shot_summary", "frozen inclusive-MAE cross-check", "derived validation reference"),
    "data/oxide/manifests/test.csv": ("test_manifest", "fixed oxide test membership and order", "membership authority"),
    "data/nitride/manifests/test.csv": ("test_manifest", "fixed nitride test membership and order", "membership authority"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_inventory.csv": ("dataset_governance", "exact approved oxynitride JID inventory", "membership authority"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json": ("dataset_governance", "approved asymmetric family definition", "membership authority"),
    "scripts/embedding_analysis/06_quantify_family_separation.py": ("embedding_protocol", "frozen metric algorithm", "procedural authority"),
    "results/reproduction/embeddings/tables/family_separation_metrics.csv": ("embedding_metrics", "48-row baseline reproduction reference", "derived validation reference"),
    "results/embeddings/manifests/family_separation_metrics_manifest.json": ("embedding_protocol", "frozen metric parameters", "procedural authority"),
    "results/embeddings/embeddings/test_set/structure_embeddings.npz": ("embedding_array", "fixed-test raw 256D embeddings", "numerical authority"),
    "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv": ("embedding_metadata", "fixed-test embedding alignment", "membership authority"),
    "results/embeddings/embeddings/balanced_pool/structure_embeddings.npz": ("embedding_array", "balanced-pool raw 256D embeddings", "numerical authority"),
    "results/embeddings/embeddings/balanced_pool/structure_embedding_metadata.csv": ("embedding_metadata", "balanced-pool embedding alignment", "membership authority"),
    "results/embeddings/subsets/fixed_test_set/metadata.csv": ("embedding_metadata", "compact fixed-test cross-check", "secondary reference"),
    "results/embeddings/subsets/balanced_pool_set/metadata.csv": ("embedding_metadata", "compact balanced-pool cross-check", "secondary reference"),
    "scripts/shared/evaluate_alignn_zero_shot.py": ("zero_shot_protocol", "read-only prediction provenance reference; never executed", "procedural reference"),
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_membership_terminology_decision.md": ("terminology_governance", "bounded claim terminology", "governance authority"),
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_evidence.sha256": ("prior_gate", "Section 2.3A checksum gate", "governance authority"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_evidence.sha256": ("prior_gate", "Section 2.3B checksum gate", "governance authority"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json": ("prior_gate", "Section 2.3B verdict", "governance authority"),
    "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md": ("prior_gate", "Section 2.2 verdict marker", "governance authority"),
    "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256": ("prior_gate", "promotion checksum gate", "governance authority"),
    "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256": ("prior_gate", "regeneration checksum gate", "governance authority"),
}

INITIAL_OUTPUTS = [
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
]
FINAL_OUTPUTS = ["generated_output_manifest.json", "section2_3C_validation.json", "section2_3C_report.md", "section2_3C_evidence.sha256"]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    if isinstance(value, np.ndarray):
        return [native(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Refusing non-finite JSON value: {value}")
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(native(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_df(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n", float_format="%.17g")


def git(root: Path, *args: str, check: bool = True) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def verify_sha_manifest(root: Path, manifest_rel: str, cwd_rel: str = ".") -> dict[str, Any]:
    manifest = root / manifest_rel
    result = subprocess.run(
        ["shasum", "-a", "256", "-c", str(manifest if cwd_rel == "." else manifest.name)],
        cwd=root / cwd_rel,
        text=True,
        capture_output=True,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {
        "manifest": manifest_rel,
        "returncode": result.returncode,
        "entries": len(lines),
        "all_ok": result.returncode == 0 and all(line.endswith(": OK") for line in lines),
        "stderr": result.stderr.strip(),
    }


def status_entries(root: Path) -> list[dict[str, str]]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout
    records = [record for record in raw.split(b"\0") if record]
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index].decode("utf-8", "surrogateescape")
        status = record[:2]
        path = record[3:]
        entries.append({"status": status, "path": path})
        if status[0] in "RC" and index + 1 < len(records):
            index += 1
        index += 1
    return entries


def classify_path(path: str) -> str:
    if "/finetune_last2_reproduction_rerun/" in path:
        return "preserved_rerun_staging"
    if path.startswith("configs/protocol_1/finetune_reproduction_rerun/"):
        return "preserved_rerun_configuration"
    if path.startswith("results/summaries/protocol_1/finetune/reproduction_rerun/"):
        return "preserved_rerun_summary_control"
    if path in {
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
    }:
        return "governance_session"
    if path.startswith("results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/") or path in {
        "scripts/shared/audit_checkpoint_provenance.py",
        "scripts/shared/validate_checkpoint_provenance.py",
    }:
        return "section2_3A"
    if path.startswith("results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/") or path in {
        "scripts/shared/audit_dataset_integrity.py",
        "scripts/shared/validate_dataset_integrity.py",
    }:
        return "section2_3B"
    if path == "domain_shift-alignn-domain-shift/":
        return "excluded_nested_checkout"
    if path in {ANALYZE_REL.as_posix(), VALIDATE_REL.as_posix()} or path.startswith(OUT_REL.as_posix() + "/"):
        return "section2_3C"
    return "ambiguous"


def preflight(root: Path) -> dict[str, Any]:
    if root.resolve() != EXPECTED_ROOT.resolve():
        raise RuntimeError(f"Unexpected repository root: {root}")
    branch = git(root, "branch", "--show-current")
    head = git(root, "rev-parse", "HEAD")
    if branch != EXPECTED_BRANCH or head != EXPECTED_HEAD:
        raise RuntimeError(f"Unexpected branch/HEAD: {branch} {head}")
    if (root / SECTION08_REL).exists():
        raise RuntimeError("Forbidden Section 08 path exists; contents were not inspected")
    tracked08 = git(root, "ls-files", "--", SECTION08_REL.as_posix())
    if tracked08:
        raise RuntimeError("Git tracks a forbidden Section 08 path")
    entries = status_entries(root)
    for entry in entries:
        entry["classification"] = classify_path(entry["path"])
    tracked_modified = [e for e in entries if e["status"] != "??"]
    ambiguous = [e for e in entries if e["classification"] == "ambiguous"]
    if tracked_modified or ambiguous:
        raise RuntimeError(f"Unsafe working tree: tracked={tracked_modified}, ambiguous={ambiguous}")
    task_paths = {ANALYZE_REL.as_posix(), VALIDATE_REL.as_posix()}
    reconstructed = [e for e in entries if e["path"] not in task_paths and not e["path"].startswith(OUT_REL.as_posix() + "/")]
    categories: dict[str, int] = {}
    for entry in reconstructed:
        categories[entry["classification"]] = categories.get(entry["classification"], 0) + 1
    expected = {
        "preserved_rerun_staging": 504,
        "preserved_rerun_configuration": 36,
        "preserved_rerun_summary_control": 2,
        "governance_session": 3,
        "section2_3A": 32,
        "section2_3B": 25,
        "excluded_nested_checkout": 1,
    }
    if len(reconstructed) != 603 or categories != expected:
        raise RuntimeError(f"Baseline classification mismatch: total={len(reconstructed)}, categories={categories}")
    gates = {
        "promotion": verify_sha_manifest(root, "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256"),
        "regeneration": verify_sha_manifest(
            root,
            "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256",
            "results/derived_evidence/protocol_1_regeneration",
        ),
        "section2_3A": verify_sha_manifest(root, "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_evidence.sha256"),
        "section2_3B": verify_sha_manifest(root, "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_evidence.sha256"),
    }
    if not all(gate["all_ok"] for gate in gates.values()):
        raise RuntimeError(f"Prior checksum gate failed: {gates}")
    regen_report = (root / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md").read_text(encoding="utf-8")
    a_validation = json.loads((root / "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_validation.json").read_text(encoding="utf-8"))
    b_validation = json.loads((root / "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json").read_text(encoding="utf-8"))
    if "protocol_1_REGENERATION_VALIDATED" not in regen_report:
        raise RuntimeError("Missing Section 2.2 marker")
    if a_validation.get("verdict") != "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP":
        raise RuntimeError("Section 2.3A verdict mismatch")
    if b_validation.get("verdict") != "SECTION23B_DATASET_INTEGRITY_VALIDATED":
        raise RuntimeError("Section 2.3B verdict mismatch")
    return {
        "created_at_utc": utc_now(),
        "repository_root": root.as_posix(),
        "branch": branch,
        "head": head,
        "last_three_commits": git(root, "log", "-3", "--format=%H %s").splitlines(),
        "section08": {"exists": False, "tracked_paths": 0, "contents_inspected": False},
        "nested_checkout": {"path": NESTED_REL.as_posix(), "excluded": True, "contents_inspected": False},
        "observed_status_count_at_producer_start": len(entries),
        "reconstructed_pre_section2_3C_status_count": len(reconstructed),
        "tracked_modifications": len(tracked_modified),
        "staged_paths": 0,
        "ambiguous_paths": len(ambiguous),
        "baseline_category_counts": categories,
        "full_git_status": entries,
        "working_tree": {
            "observed_at_producer_start": len(entries),
            "reconstructed_pre_section2_3C_untracked": len(reconstructed),
            "tracked_modifications": len(tracked_modified),
            "staged_paths": 0,
            "ambiguous_paths": len(ambiguous),
            "category_counts": categories,
            "expected_baseline_untracked": 603,
        },
        "prior_evidence_gates": gates,
        "verdicts": {
            "section2_2": "protocol_1_REGENERATION_VALIDATED",
            "section2_3A": a_validation.get("verdict"),
            "section2_3B": b_validation.get("verdict"),
        },
        "scientific_input_policy": "explicit allowlist only",
    }


def assert_input_policy(rel: str) -> None:
    if "archived_submission_materials" in rel or rel == NESTED_REL.as_posix() or rel.startswith(NESTED_REL.as_posix() + "/"):
        raise RuntimeError(f"Forbidden input path: {rel}")
    if "finetune_last2_reproduction_rerun" in rel or "reproduction_rerun" in rel:
        raise RuntimeError(f"Recovery/staging input forbidden: {rel}")
    if rel.startswith("results/protocol_1/") and rel not in {
        "results/zero_shot/oxide/predictions.csv",
        "results/zero_shot/nitride/predictions.csv",
    }:
        raise RuntimeError(f"Unauthorized legacy input: {rel}")


def protected_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for rel in INPUT_ROLES:
        assert_input_policy(rel)
        path = root / rel
        if not path.is_file():
            raise FileNotFoundError(rel)
        hashes[rel] = sha256(path)
        expected = EXPECTED_HASHES.get(rel)
        if expected and hashes[rel] != expected:
            raise RuntimeError(f"SHA-256 mismatch for {rel}: {hashes[rel]} != {expected}")
    return hashes


def git_file_status(root: Path, rel: str) -> tuple[str, str]:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", rel],
        cwd=root,
        capture_output=True,
        check=False,
    ).returncode == 0
    if tracked:
        blob = git(root, "rev-parse", f"HEAD:{rel}", check=False)
        return "tracked", blob
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all", "--", rel, check=False)
    return ("untracked" if status.startswith("??") else "not_tracked"), ""


def schema_description(root: Path, rel: str) -> str:
    path = root / rel
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        return json.dumps({"rows": int(len(frame)), "columns": list(frame.columns)}, sort_keys=True)
    if path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            shape = {key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)} for key in payload.files}
        return json.dumps(shape, sort_keys=True)
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return json.dumps({"json_type": type(payload).__name__, "top_level_keys": sorted(payload) if isinstance(payload, dict) else []}, sort_keys=True)
    if path.suffix.lower() == ".sha256":
        return json.dumps({"checksum_entries": len([line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])})
    return json.dumps({"format": path.suffix.lower().lstrip(".") or "text"})


def input_inventory(root: Path, before: dict[str, str], after: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rel, (category, role, authority) in INPUT_ROLES.items():
        path = root / rel
        status, blob = git_file_status(root, rel)
        stat = path.stat()
        rows.append(
            {
                "category": category,
                "repository_relative_path": rel,
                "absolute_path": path.resolve().as_posix(),
                "git_status": status,
                "git_blob": blob,
                "size_bytes": stat.st_size,
                "modification_time": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
                "modification_time_utc": dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(),
                "sha256": after[rel],
                "schema_or_array_shape": schema_description(root, rel),
                "intended_role": role,
                "authority_level": authority,
                "before_analysis_sha256": before[rel],
                "after_analysis_sha256": after[rel],
                "unchanged": before[rel] == after[rel],
            }
        )
    return pd.DataFrame(rows)


def bool_series(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    if not normalized.isin(["true", "false", "1", "0"]).all():
        raise ValueError(f"Invalid Boolean values: {sorted(normalized.unique())}")
    return normalized.isin(["true", "1"]).to_numpy(dtype=bool)


def load_zero_shot(root: Path) -> dict[str, Any]:
    pred_paths = {
        "oxide": "results/zero_shot/oxide/predictions.csv",
        "nitride": "results/zero_shot/nitride/predictions.csv",
    }
    manifest_paths = {
        "oxide": "data/oxide/manifests/test.csv",
        "nitride": "data/nitride/manifests/test.csv",
    }
    expected_counts = {"oxide": 1484, "nitride": 242}
    frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, pd.DataFrame] = {}
    integrity: dict[str, Any] = {}
    for family in ("oxide", "nitride"):
        pred = pd.read_csv(root / pred_paths[family])
        manifest = pd.read_csv(root / manifest_paths[family])
        if list(pred.columns) != PREDICTION_SCHEMA:
            raise ValueError(f"{family} prediction schema mismatch: {list(pred.columns)}")
        if len(pred) != expected_counts[family] or len(manifest) != expected_counts[family]:
            raise ValueError(f"{family} count mismatch")
        if pred.isna().any().any() or pred["jid"].duplicated().any():
            raise ValueError(f"{family} predictions contain missing/duplicate values")
        numeric = pred[["target", "prediction", "abs_error"]].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{family} predictions contain non-finite values")
        expected_filenames = "POSCAR-" + pred["jid"].astype(str) + ".vasp"
        if not pred["filename"].astype(str).equals(expected_filenames):
            raise ValueError(f"{family} filename/JID mismatch")
        jid_order = pred["jid"].astype(str).tolist() == manifest["jid"].astype(str).tolist()
        filename_order = pred["filename"].astype(str).tolist() == manifest["filename"].astype(str).tolist()
        target_equal = np.array_equal(pred["target"].to_numpy(dtype=np.float64), manifest["target"].to_numpy(dtype=np.float64))
        if not jid_order or not filename_order or not target_equal:
            raise ValueError(f"{family} prediction/test-manifest alignment failed")
        recomputed = np.abs(pred["target"].to_numpy(dtype=np.float64) - pred["prediction"].to_numpy(dtype=np.float64))
        error_delta = np.abs(recomputed - pred["abs_error"].to_numpy(dtype=np.float64))
        if not np.allclose(recomputed, pred["abs_error"].to_numpy(dtype=np.float64), atol=1e-12, rtol=1e-12):
            raise ValueError(f"{family} stored abs_error mismatch")
        pred = pred.copy()
        pred["recomputed_abs_error"] = recomputed
        frames[family] = pred
        manifests[family] = manifest
        integrity[family] = {
            "prediction_path": pred_paths[family],
            "manifest_path": manifest_paths[family],
            "rows": len(pred),
            "schema_exact": True,
            "unique_jids": int(pred["jid"].nunique()),
            "finite_numeric_values": True,
            "jid_membership_and_order_exact": jid_order,
            "filename_and_order_exact": filename_order,
            "target_values_exact": target_equal,
            "max_abs_error_recomputation_difference": float(error_delta.max()),
            "mae_recomputed": float(np.mean(recomputed)),
        }
    overlap = sorted(set(frames["oxide"]["jid"]) & set(frames["nitride"]["jid"]))
    if overlap:
        raise ValueError(f"Cross-family prediction overlap: {overlap[:5]}")
    summary = pd.read_csv(root / "results/zero_shot/zero_shot_summary.csv").set_index("family")
    for family, expected in {"oxide": 0.03418360680813096, "nitride": 0.06954201496284854}.items():
        observed = integrity[family]["mae_recomputed"]
        frozen = float(summary.loc[family, "mae_eV_per_atom"])
        if abs(observed - frozen) > 1e-12 or abs(observed - expected) > 1e-12:
            raise ValueError(f"{family} zero-shot summary mismatch")
        integrity[family]["frozen_summary_mae"] = frozen
        integrity[family]["summary_absolute_difference"] = abs(observed - frozen)

    ox_inventory = pd.read_csv(root / "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_inventory.csv")
    definition = json.loads((root / "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json").read_text(encoding="utf-8"))
    if len(ox_inventory) != 499 or ox_inventory["jid"].duplicated().any():
        raise ValueError("Section 2.3B oxynitride inventory mismatch")
    split_counts = ox_inventory.groupby("split").size().to_dict()
    if split_counts != {"test": 57, "train": 400, "val": 42}:
        raise ValueError(f"Oxynitride split counts mismatch: {split_counts}")
    if definition.get("oxynitride_count") != 499 or definition.get("nitride_oxynitride_count") != 0:
        raise ValueError("Section 2.3B oxynitride definition mismatch")
    test_ox = set(ox_inventory.loc[ox_inventory["split"] == "test", "jid"].astype(str))
    oxide_mask = frames["oxide"]["jid"].astype(str).isin(test_ox).to_numpy()
    nitride_mask = frames["nitride"]["jid"].astype(str).isin(test_ox).to_numpy()
    if int(oxide_mask.sum()) != 57 or int(nitride_mask.sum()) != 0 or int((~oxide_mask).sum()) != 1427:
        raise ValueError("Prediction/oxynitride membership mismatch")
    integrity["cross_family_overlap_count"] = 0
    integrity["oxynitride_membership"] = {
        "inventory_total": len(ox_inventory),
        "split_counts": split_counts,
        "oxide_test_oxynitrides": int(oxide_mask.sum()),
        "pure_oxide_test": int((~oxide_mask).sum()),
        "nitride_removed": int(nitride_mask.sum()),
        "authority": "Section 2.3B exact JID inventory",
    }
    return {
        "frames": frames,
        "manifests": manifests,
        "integrity": integrity,
        "ox_inventory": ox_inventory,
        "test_ox_jids": test_ox,
        "oxide_mask": oxide_mask,
    }


def zero_shot_outputs(data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, Any]]:
    oxide = data["frames"]["oxide"].copy()
    nitride = data["frames"]["nitride"].copy()
    oxide_mask = data["oxide_mask"]
    membership_rows: list[dict[str, Any]] = []
    for family, frame in (("oxide", oxide), ("nitride", nitride)):
        family_mask = oxide_mask if family == "oxide" else np.zeros(len(frame), dtype=bool)
        source = f"results/zero_shot/{family}/predictions.csv"
        for index, row in frame.reset_index(drop=True).iterrows():
            membership_rows.append(
                {
                    "family": family,
                    "jid": row["jid"],
                    "filename": row["filename"],
                    "test_manifest_order": index,
                    "is_oxynitride": bool(family_mask[index]),
                    "included_in_inclusive_comparison": True,
                    "included_in_pure_oxide_sensitivity": bool(family == "nitride" or not family_mask[index]),
                    "source_prediction_path": source,
                    "membership_authority": "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_inventory.csv",
                }
            )
    membership = pd.DataFrame(membership_rows)
    removed = oxide.loc[oxide_mask].copy().reset_index(drop=True)
    removed.insert(0, "family", "oxide")
    removed.insert(1, "is_oxynitride", True)
    errors = {
        "inclusive_oxide": oxide["recomputed_abs_error"].to_numpy(dtype=np.float64),
        "inclusive_nitride": nitride["recomputed_abs_error"].to_numpy(dtype=np.float64),
        "pure_oxide_filtered": oxide.loc[~oxide_mask, "recomputed_abs_error"].to_numpy(dtype=np.float64),
        "pure_nitride": nitride["recomputed_abs_error"].to_numpy(dtype=np.float64),
        "oxynitride_only": oxide.loc[oxide_mask, "recomputed_abs_error"].to_numpy(dtype=np.float64),
    }
    inclusive_oxide = float(np.mean(errors["inclusive_oxide"]))
    pure_oxide = float(np.mean(errors["pure_oxide_filtered"]))
    nitride_mae = float(np.mean(errors["inclusive_nitride"]))
    oxynitride_mae = float(np.mean(errors["oxynitride_only"]))
    estimates = {
        "inclusive_oxide_vs_nitride": {
            "oxide_mae": inclusive_oxide,
            "nitride_mae": nitride_mae,
            "mae_difference_nitride_minus_oxide": nitride_mae - inclusive_oxide,
            "mae_ratio_nitride_over_oxide": nitride_mae / inclusive_oxide,
            "oxide_n": len(errors["inclusive_oxide"]),
            "nitride_n": len(errors["inclusive_nitride"]),
        },
        "pure_oxide_filtered_vs_nitride": {
            "oxide_mae": pure_oxide,
            "nitride_mae": nitride_mae,
            "mae_difference_nitride_minus_oxide": nitride_mae - pure_oxide,
            "mae_ratio_nitride_over_oxide": nitride_mae / pure_oxide,
            "oxide_n": len(errors["pure_oxide_filtered"]),
            "nitride_n": len(errors["pure_nitride"]),
        },
    }
    point_rows: list[dict[str, Any]] = []
    for scenario, values in estimates.items():
        point_rows.append(
            {
                "scenario": scenario,
                "oxide_n": values["oxide_n"],
                "nitride_n": values["nitride_n"],
                "oxide_mae": values["oxide_mae"],
                "nitride_mae": values["nitride_mae"],
                "mae_difference_nitride_minus_oxide": values["mae_difference_nitride_minus_oxide"],
                "mae_ratio_nitride_over_oxide": values["mae_ratio_nitride_over_oxide"],
                "oxide_mae_change_filtered_minus_inclusive": np.nan,
                "oxide_mae_percent_change": np.nan,
                "mae_difference_change_filtered_minus_inclusive": np.nan,
                "mae_ratio_change_filtered_minus_inclusive": np.nan,
                "mae_and_difference_units": "eV/atom",
                "ratio_units": "dimensionless",
                "definition": "difference=nitride-minus-oxide; ratio=nitride-over-oxide",
            }
        )
    point_rows.extend(
        [
            {
                "scenario": "oxynitride_only_descriptive",
                "oxide_n": 57,
                "nitride_n": 0,
                "oxide_mae": oxynitride_mae,
                "nitride_mae": np.nan,
                "mae_difference_nitride_minus_oxide": np.nan,
                "mae_ratio_nitride_over_oxide": np.nan,
                "oxide_mae_change_filtered_minus_inclusive": np.nan,
                "oxide_mae_percent_change": np.nan,
                "mae_difference_change_filtered_minus_inclusive": np.nan,
                "mae_ratio_change_filtered_minus_inclusive": np.nan,
                "mae_and_difference_units": "eV/atom",
                "ratio_units": "dimensionless",
                "definition": "arithmetic mean absolute error of the 57 removed oxide-arm oxynitrides",
            },
            {
                "scenario": "sensitivity_filtered_minus_inclusive",
                "oxide_n": 1427,
                "nitride_n": 242,
                "oxide_mae": np.nan,
                "nitride_mae": np.nan,
                "mae_difference_nitride_minus_oxide": np.nan,
                "mae_ratio_nitride_over_oxide": np.nan,
                "oxide_mae_change_filtered_minus_inclusive": pure_oxide - inclusive_oxide,
                "oxide_mae_percent_change": 100.0 * (pure_oxide - inclusive_oxide) / inclusive_oxide,
                "mae_difference_change_filtered_minus_inclusive": estimates["pure_oxide_filtered_vs_nitride"]["mae_difference_nitride_minus_oxide"] - estimates["inclusive_oxide_vs_nitride"]["mae_difference_nitride_minus_oxide"],
                "mae_ratio_change_filtered_minus_inclusive": estimates["pure_oxide_filtered_vs_nitride"]["mae_ratio_nitride_over_oxide"] - estimates["inclusive_oxide_vs_nitride"]["mae_ratio_nitride_over_oxide"],
                "filtered_minus_inclusive_oxide_mae_eV_per_atom": pure_oxide - inclusive_oxide,
                "percentage_change_relative_to_inclusive_oxide_mae": 100.0 * (pure_oxide - inclusive_oxide) / inclusive_oxide,
                "filtered_minus_inclusive_difference_eV_per_atom": estimates["pure_oxide_filtered_vs_nitride"]["mae_difference_nitride_minus_oxide"] - estimates["inclusive_oxide_vs_nitride"]["mae_difference_nitride_minus_oxide"],
                "filtered_minus_inclusive_ratio": estimates["pure_oxide_filtered_vs_nitride"]["mae_ratio_nitride_over_oxide"] - estimates["inclusive_oxide_vs_nitride"]["mae_ratio_nitride_over_oxide"],
                "mae_and_difference_units": "eV/atom; percent field is percent",
                "ratio_units": "dimensionless",
                "definition": "all changes are filtered minus inclusive",
            },
        ]
    )
    return membership, removed, pd.DataFrame(point_rows), errors, estimates


def bootstrap_zero_shot(errors: dict[str, np.ndarray], estimates: dict[str, Any], n_replicates: int = 50000, chunk_size: int = 512) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    master = np.random.SeedSequence(42)
    children = master.spawn(4)
    names = ["inclusive_oxide", "inclusive_nitride", "pure_oxide_filtered", "pure_nitride"]
    child_map = dict(zip(names, children))
    scenarios = [
        ("inclusive_oxide_vs_nitride", "inclusive_oxide", "inclusive_nitride"),
        ("pure_oxide_filtered_vs_nitride", "pure_oxide_filtered", "pure_nitride"),
    ]
    replicate_frames: list[pd.DataFrame] = []
    for scenario, oxide_key, nitride_key in scenarios:
        oxide = np.asarray(errors[oxide_key], dtype=np.float64)
        nitride = np.asarray(errors[nitride_key], dtype=np.float64)
        rng_o = np.random.default_rng(child_map[oxide_key])
        rng_n = np.random.default_rng(child_map[nitride_key])
        oxide_boot = np.empty(n_replicates, dtype=np.float64)
        nitride_boot = np.empty(n_replicates, dtype=np.float64)
        for start in range(0, n_replicates, chunk_size):
            stop = min(start + chunk_size, n_replicates)
            count = stop - start
            oi = rng_o.integers(0, len(oxide), size=(count, len(oxide)), dtype=np.int64)
            ni = rng_n.integers(0, len(nitride), size=(count, len(nitride)), dtype=np.int64)
            oxide_boot[start:stop] = oxide[oi].mean(axis=1, dtype=np.float64)
            nitride_boot[start:stop] = nitride[ni].mean(axis=1, dtype=np.float64)
        if not np.isfinite(oxide_boot).all() or np.any(oxide_boot == 0.0):
            raise ValueError(f"Invalid bootstrap ratio denominator in {scenario}")
        difference = nitride_boot - oxide_boot
        ratio = nitride_boot / oxide_boot
        replicate_frames.append(
            pd.DataFrame(
                {
                    "scenario": scenario,
                    "replicate": np.arange(n_replicates, dtype=np.int64),
                    "oxide_n": len(oxide),
                    "nitride_n": len(nitride),
                    "oxide_mae": oxide_boot,
                    "nitride_mae": nitride_boot,
                    "mae_difference_nitride_minus_oxide": difference,
                    "mae_ratio_nitride_over_oxide": ratio,
                }
            )
        )
    replicates = pd.concat(replicate_frames, ignore_index=True)
    summary_rows: list[dict[str, Any]] = []
    column_map = {
        "oxide_mae": "oxide_mae",
        "nitride_mae": "nitride_mae",
        "mae_difference_nitride_minus_oxide": "mae_difference_nitride_minus_oxide",
        "mae_ratio_nitride_over_oxide": "mae_ratio_nitride_over_oxide",
    }
    unit_map = {"oxide_mae": "eV/atom", "nitride_mae": "eV/atom", "mae_difference_nitride_minus_oxide": "eV/atom", "mae_ratio_nitride_over_oxide": "dimensionless"}
    scenario_keys = {
        "inclusive_oxide_vs_nitride": ("inclusive_oxide", "inclusive_nitride"),
        "pure_oxide_filtered_vs_nitride": ("pure_oxide_filtered", "pure_nitride"),
    }
    for scenario, _, _ in scenarios:
        sub = replicates[replicates["scenario"] == scenario]
        oxide_key, nitride_key = scenario_keys[scenario]
        for estimand, column in column_map.items():
            values = sub[column].to_numpy(dtype=np.float64)
            ci_low, ci_high = np.quantile(values, [0.025, 0.975], method="linear")
            summary_rows.append(
                {
                    "scenario": scenario,
                    "estimand": estimand,
                    "point_estimate": estimates[scenario][estimand],
                    "bootstrap_mean": float(np.mean(values)),
                    "bootstrap_standard_error": float(np.std(values, ddof=1)),
                    "ci_low": float(ci_low),
                    "ci_high": float(ci_high),
                    "confidence_level": 0.95,
                    "ci_method": "percentile",
                    "quantile_method": "linear",
                    "n_replicates": n_replicates,
                    "master_seed": 42,
                    "oxide_spawn_key": json.dumps(list(child_map[oxide_key].spawn_key)),
                    "nitride_spawn_key": json.dumps(list(child_map[nitride_key].spawn_key)),
                    "oxide_n": estimates[scenario]["oxide_n"],
                    "nitride_n": estimates[scenario]["nitride_n"],
                    "units": unit_map[estimand],
                }
            )
    protocol = {
        "analysis": "zero-shot chemical-family comparison uncertainty",
        "master_seed": 42,
        "seed_sequence": "numpy.random.SeedSequence",
        "seed_sequence_entropy": 42,
        "generator": "numpy.random.default_rng",
        "bit_generator": "PCG64",
        "substream_order": names,
        "substreams": {
            name: {
                "spawn_key": list(child_map[name].spawn_key),
                "generated_state_uint32": child_map[name].generate_state(4).tolist(),
            }
            for name in names
        },
        "n_replicates_per_scenario": n_replicates,
        "bootstrap_replicates": n_replicates,
        "confidence_level": 0.95,
        "interval": "percentile",
        "quantile_method": "linear",
        "resampling_unit": "structure/prediction row",
        "resampling_design": "independent within each family back to original scenario-specific sample size",
        "paired_bootstrap": False,
        "point_estimate": "full-data statistic",
        "difference_orientation": "nitride MAE minus oxide MAE",
        "ratio_orientation": "nitride MAE divided by oxide MAE",
        "numeric_dtype": "float64",
        "bootstrap_standard_error_ddof": 1,
        "chunk_size": chunk_size,
        "saved_resampled_indices": False,
        "errors_recomputed_from": "abs(target - prediction)",
        "numpy_version": np.__version__,
        "causal_claim_authorized": False,
    }
    return replicates, pd.DataFrame(summary_rows), protocol


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return math.nan, math.nan
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_bootstrap <= 0:
        return math.nan, math.nan
    boot = [float(np.mean(values[rng.integers(0, len(values), len(values))])) for _ in range(n_bootstrap)]
    return percentile_ci(boot)


def bootstrap_stratified_mean_ci(values: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if n_bootstrap <= 0:
        return math.nan, math.nan
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if any(len(indices) < 2 for indices in groups):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled_values = []
        for indices in groups:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_values.append(values[sampled])
        boot.append(float(np.mean(np.concatenate(sampled_values))))
    return percentile_ci(boot)


def bootstrap_dbi_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return math.nan, math.nan
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if len(groups) < 2 or any(len(indices) < 2 for indices in groups):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in groups])
        try:
            boot.append(float(davies_bouldin_score(x[sampled], y[sampled])))
        except ValueError:
            continue
    return percentile_ci(boot)


def bootstrap_auc_ci(y: np.ndarray, scores: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return math.nan, math.nan
    groups = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if len(groups) < 2 or any(len(indices) < 2 for indices in groups):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in groups])
        try:
            boot.append(float(roc_auc_score(y[sampled], scores[sampled])))
        except ValueError:
            continue
    return percentile_ci(boot)


def knn_purity(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if k < 1 or k >= len(y):
        raise ValueError(f"Invalid k={k} for n={len(y)}")
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(x)
    indices = neighbors.kneighbors(x, return_distance=False)[:, 1:]
    return np.mean(y[indices] == y[:, None], axis=1)


def logistic_auc(x: np.ndarray, y: np.ndarray, cv_folds: int, seed: int) -> tuple[float, np.ndarray, int]:
    min_class = min(int(np.sum(y == label)) for label in np.unique(y))
    n_splits = min(cv_folds, min_class)
    if n_splits < 2:
        return math.nan, np.full(len(y), np.nan), n_splits
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan, dtype=float)
    for fold_index, (train_index, test_index) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed + fold_index,
                solver="lbfgs",
            ),
        )
        model.fit(x[train_index], y[train_index])
        scores[test_index] = model.predict_proba(x[test_index])[:, 1]
    return float(roc_auc_score(y, scores)), scores, n_splits


def family_counts(y: np.ndarray) -> dict[str, int]:
    return {family: int(np.sum(y == label)) for label, family in LABEL_TO_FAMILY.items()}


def metric_row(
    dataset: str,
    dataset_label: str,
    source: str,
    x: np.ndarray,
    y: np.ndarray,
    scenario: str,
    source_seed: int,
    metric_name: str,
    metric_scope: str,
    value: float,
    ci_low: float,
    ci_high: float,
    ci_method: str,
    preprocessing: str,
    higher_is_better: bool,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    counts = family_counts(y)
    return {
        "dataset": dataset,
        "dataset_label": dataset_label,
        "embedding_source": source,
        "embedding_dim": int(x.shape[1]),
        "metric_name": metric_name,
        "metric_scope": metric_scope,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 0.95,
        "ci_method": ci_method,
        "higher_is_better": higher_is_better,
        "n_structures": int(len(y)),
        "n_oxide": counts["oxide"],
        "n_nitride": counts["nitride"],
        "vector_space": "raw_256d_embedding_vectors",
        "raw_space_primary": True,
        "projected_space": "none",
        "preprocessing": preprocessing,
        "parameters": json.dumps(parameters, sort_keys=True),
        "scenario": scenario,
        "source_seed": source_seed,
        "majority_label_baseline": max(counts.values()) / len(y),
    }


def compute_embedding_metrics(
    dataset: str,
    dataset_label: str,
    source: str,
    x: np.ndarray,
    y: np.ndarray,
    scenario: str,
    source_seed: int,
    n_bootstrap: int = 1000,
    k: int = 15,
    cv_folds: int = 5,
) -> list[dict[str, Any]]:
    if len(np.unique(y)) != 2 or not np.isfinite(x).all():
        raise ValueError(f"Invalid embedding data for {dataset}/{source}/{scenario}")
    rng = np.random.default_rng(source_seed)
    rows: list[dict[str, Any]] = []
    print(f"{dataset}/{source}/{scenario}: silhouette")
    sil = silhouette_samples(x, y, metric="euclidean")
    low, high = bootstrap_stratified_mean_ci(sil, y, rng, n_bootstrap)
    rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "silhouette_score", "overall_family_labels", float(np.mean(sil)), low, high, "stratified_bootstrap_over_per_structure_silhouette_values", "none", True, {"metric": "euclidean"}))
    for label, family in LABEL_TO_FAMILY.items():
        values = sil[y == label]
        low, high = bootstrap_mean_ci(values, rng, n_bootstrap)
        rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "silhouette_score", family, float(np.mean(values)), low, high, "bootstrap_over_family_per_structure_silhouette_values", "none", True, {"metric": "euclidean"}))
    print(f"{dataset}/{source}/{scenario}: Davies-Bouldin")
    dbi = float(davies_bouldin_score(x, y))
    low, high = bootstrap_dbi_ci(x, y, rng, n_bootstrap)
    rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "davies_bouldin_index", "overall_family_labels", dbi, low, high, "stratified_bootstrap_recomputed_index", "none", False, {"metric": "euclidean"}))
    print(f"{dataset}/{source}/{scenario}: kNN purity")
    purity = knn_purity(x, y, k)
    low, high = bootstrap_stratified_mean_ci(purity, y, rng, n_bootstrap)
    rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "knn_family_purity", "overall_family_labels", float(np.mean(purity)), low, high, "stratified_bootstrap_over_per_structure_purity_values", "none", True, {"k_neighbors": k, "metric": "euclidean", "self_excluded": True}))
    for label, family in LABEL_TO_FAMILY.items():
        values = purity[y == label]
        low, high = bootstrap_mean_ci(values, rng, n_bootstrap)
        rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "knn_family_purity", family, float(np.mean(values)), low, high, "bootstrap_over_family_per_structure_purity_values", "none", True, {"k_neighbors": k, "metric": "euclidean", "self_excluded": True}))
    print(f"{dataset}/{source}/{scenario}: frozen logistic probe")
    auc, scores, actual_folds = logistic_auc(x, y, cv_folds, source_seed)
    low, high = bootstrap_auc_ci(y, scores, rng, n_bootstrap)
    rows.append(metric_row(dataset, dataset_label, source, x, y, scenario, source_seed, "logistic_regression_family_auc", "overall_family_labels", auc, low, high, "stratified_bootstrap_over_cross_validated_out_of_fold_scores", "fold_local_standard_scaler_no_dimensionality_reduction", True, {"cv_folds": actual_folds, "class_weight": "balanced", "solver": "lbfgs", "positive_label": "nitride"}))
    return rows


def load_embedding_inputs(root: Path, test_ox_jids: set[str], zero_data: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    metrics_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    duplicate_arrays: dict[str, bool] = {}
    source_seeds: dict[str, dict[str, int]] = {}
    prediction_drift: dict[str, Any] = {}
    for dataset_index, (dataset, info) in enumerate(DATASETS.items()):
        npz_path = root / info["npz"]
        metadata_path = root / info["metadata"]
        compact_path = root / info["compact"]
        metadata = pd.read_csv(metadata_path)
        compact = pd.read_csv(compact_path)
        expected_total, expected_oxide, expected_nitride, expected_oxynitride, filtered_total, filtered_oxide, filtered_nitride = info["expected"]
        with np.load(npz_path) as payload:
            if tuple(payload.files) != SOURCES:
                raise ValueError(f"Unexpected NPZ keys/order for {dataset}: {payload.files}")
            raw_arrays = {key: payload[key] for key in payload.files}
            arrays = {key: payload[key].astype(np.float64) for key in payload.files}
        duplicate_arrays[dataset] = bool(np.array_equal(raw_arrays["pre_head"], raw_arrays["last_gcn_pool"]))
        if len(metadata) != expected_total * 3 or len(compact) != expected_total:
            raise ValueError(f"Embedding metadata count mismatch for {dataset}")
        first_ids: list[str] | None = None
        for source_index, source in enumerate(SOURCES):
            source_seed = 42 + 1000 * dataset_index + source_index
            source_seeds.setdefault(dataset, {})[source] = source_seed
            source_meta = metadata.loc[metadata["embedding_source"] == source].copy()
            source_meta["embedding_index"] = source_meta["embedding_index"].astype(int)
            source_meta = source_meta.sort_values("embedding_index").reset_index(drop=True)
            if len(source_meta) != expected_total or not np.array_equal(source_meta["embedding_index"].to_numpy(), np.arange(expected_total)):
                raise ValueError(f"Embedding index mismatch for {dataset}/{source}")
            x = arrays[source]
            if x.shape != (expected_total, 256) or raw_arrays[source].dtype != np.float32 or not np.isfinite(x).all():
                raise ValueError(f"Embedding array mismatch for {dataset}/{source}")
            if not (source_meta["npz_key"].astype(str) == source).all() or not (source_meta["embedding_dim"].astype(int) == 256).all():
                raise ValueError(f"Embedding metadata key/dimension mismatch for {dataset}/{source}")
            ids = source_meta["material_id"].astype(str).tolist()
            if len(set(ids)) != expected_total:
                raise ValueError(f"Duplicate embedding material IDs for {dataset}/{source}")
            if first_ids is None:
                first_ids = ids
                compact_ids = compact["material_id"].astype(str).tolist()
                if ids != compact_ids:
                    raise ValueError(f"Compact metadata order mismatch for {dataset}")
            elif ids != first_ids:
                raise ValueError(f"Embedding-source order mismatch for {dataset}/{source}")
            y = source_meta["family"].map(FAMILY_TO_LABEL).to_numpy(dtype=int)
            if np.any(pd.isna(source_meta["family"].map(FAMILY_TO_LABEL))):
                raise ValueError(f"Unknown family in {dataset}/{source}")
            flags = bool_series(source_meta["is_oxynitride"])
            membership_flags = source_meta["material_id"].astype(str).isin(test_ox_jids).to_numpy()
            if not np.array_equal(flags, membership_flags):
                raise ValueError(f"Oxynitride metadata/JID inventory mismatch for {dataset}/{source}")
            if np.any(flags & (y == 1)):
                raise ValueError(f"Nitride oxynitride flag found for {dataset}/{source}")
            counts = family_counts(y)
            if counts != {"oxide": expected_oxide, "nitride": expected_nitride} or int(flags.sum()) != expected_oxynitride:
                raise ValueError(f"Embedding family count mismatch for {dataset}/{source}")
            exact_mismatch_array: np.ndarray | None = None
            prediction_abs_drift_array: np.ndarray | None = None
            if dataset == "fixed_test_set":
                expected_ids = zero_data["frames"]["oxide"]["jid"].astype(str).tolist() + zero_data["frames"]["nitride"]["jid"].astype(str).tolist()
                if ids != expected_ids:
                    raise ValueError(f"Fixed-test embedding/prediction order mismatch for {source}")
                expected_filenames = zero_data["frames"]["oxide"]["filename"].astype(str).tolist() + zero_data["frames"]["nitride"]["filename"].astype(str).tolist()
                if source_meta["filename"].astype(str).tolist() != expected_filenames:
                    raise ValueError(f"Fixed-test embedding filename mismatch for {source}")
                expected_targets = np.concatenate([
                    zero_data["frames"]["oxide"]["target"].to_numpy(dtype=np.float64),
                    zero_data["frames"]["nitride"]["target"].to_numpy(dtype=np.float64),
                ])
                expected_predictions = np.concatenate([
                    zero_data["frames"]["oxide"]["prediction"].to_numpy(dtype=np.float64),
                    zero_data["frames"]["nitride"]["prediction"].to_numpy(dtype=np.float64),
                ])
                if not np.array_equal(source_meta["target_formation_energy_peratom"].to_numpy(dtype=np.float64), expected_targets):
                    raise ValueError(f"Fixed-test embedding target mismatch for {source}")
                metadata_predictions = source_meta["pretrained_prediction"].to_numpy(dtype=np.float64)
                prediction_abs_drift = np.abs(metadata_predictions - expected_predictions)
                exact_mismatch = metadata_predictions != expected_predictions
                beyond_audit_tolerance = ~np.isclose(metadata_predictions, expected_predictions, atol=2e-6, rtol=1e-6)
                worst_index = int(np.argmax(prediction_abs_drift))
                prediction_drift[source] = {
                    "exact_mismatch_count": int(exact_mismatch.sum()),
                    "exact_mismatch_oxide": int(np.sum(exact_mismatch & (y == 0))),
                    "exact_mismatch_nitride": int(np.sum(exact_mismatch & (y == 1))),
                    "maximum_absolute_drift_oxide": float(np.max(prediction_abs_drift[y == 0])),
                    "maximum_absolute_drift_nitride": float(np.max(prediction_abs_drift[y == 1])),
                    "beyond_atol2e6_rtol1e6_count": int(beyond_audit_tolerance.sum()),
                    "beyond_tolerance_oxide": int(np.sum(beyond_audit_tolerance & (y == 0))),
                    "beyond_tolerance_nitride": int(np.sum(beyond_audit_tolerance & (y == 1))),
                    "maximum_absolute_drift": float(prediction_abs_drift[worst_index]),
                    "worst_material_id": ids[worst_index],
                    "status": "quantified_secondary_metadata_drift",
                }
                exact_mismatch_array = exact_mismatch
                prediction_abs_drift_array = prediction_abs_drift
            filtered = ~flags
            y_filtered = y[filtered]
            if len(y_filtered) != filtered_total or family_counts(y_filtered) != {"oxide": filtered_oxide, "nitride": filtered_nitride}:
                raise ValueError(f"Filtered embedding count mismatch for {dataset}/{source}")
            scenarios = [
                ("inclusive_baseline_recomputed", x, y, 0, np.ones(len(y), dtype=bool)),
                ("pure_oxide_filtered", x[filtered], y_filtered, expected_oxynitride, filtered),
            ]
            for scenario, scenario_x, scenario_y, removed, scenario_mask in scenarios:
                scenario_counts = family_counts(scenario_y)
                if exact_mismatch_array is not None and prediction_abs_drift_array is not None:
                    prediction_exact_mismatch_count: int | str = int(exact_mismatch_array[scenario_mask].sum())
                    prediction_max_absolute_drift: float | str = float(prediction_abs_drift_array[scenario_mask].max())
                    prediction_comparison_status = "quantified_secondary_metadata_drift"
                else:
                    prediction_exact_mismatch_count = ""
                    prediction_max_absolute_drift = ""
                    prediction_comparison_status = "not_applicable_no_canonical_prediction_comparison"
                membership_rows.append(
                    {
                        "dataset": dataset,
                        "embedding_source": source,
                        "scenario": scenario,
                        "n_structures": len(scenario_y),
                        "n_oxide": scenario_counts["oxide"],
                        "n_nitride": scenario_counts["nitride"],
                        "n_oxynitride": expected_oxynitride if scenario == "inclusive_baseline_recomputed" else 0,
                        "removed_oxynitrides": removed,
                        "majority_label_baseline": max(scenario_counts.values()) / len(scenario_y),
                        "npz_path": info["npz"],
                        "metadata_path": info["metadata"],
                        "membership_status": "passed",
                        "membership_detail": "validated_exact_JID_target_filename_filter_no_replacement_no_rebalancing",
                        "prediction_exact_mismatch_count": prediction_exact_mismatch_count,
                        "prediction_max_absolute_drift": prediction_max_absolute_drift,
                        "prediction_comparison_status": prediction_comparison_status,
                    }
                )
                metrics_rows.extend(compute_embedding_metrics(dataset, info["label"], source, scenario_x, scenario_y, scenario, source_seed))
    metrics = pd.DataFrame(metrics_rows)
    if len(metrics) != 96 or len(membership_rows) != 12:
        raise ValueError(f"Embedding output count mismatch: metrics={len(metrics)}, membership={len(membership_rows)}")

    original = pd.read_csv(root / "results/reproduction/embeddings/tables/family_separation_metrics.csv")
    original_columns = list(original.columns)
    baseline = metrics.loc[metrics["scenario"] == "inclusive_baseline_recomputed", original_columns].copy()
    keys = ["dataset", "embedding_source", "metric_name", "metric_scope"]
    original = original.sort_values(keys).reset_index(drop=True)
    baseline = baseline.sort_values(keys).reset_index(drop=True)
    if len(original) != 48 or not original[keys].equals(baseline[keys]):
        raise ValueError("Frozen embedding baseline row identity mismatch")
    numeric_columns = ["embedding_dim", "value", "ci_low", "ci_high", "ci_level", "n_structures", "n_oxide", "n_nitride"]
    max_differences: dict[str, float] = {}
    for column in numeric_columns:
        left = original[column].to_numpy(dtype=np.float64)
        right = baseline[column].to_numpy(dtype=np.float64)
        max_differences[column] = float(np.max(np.abs(left - right)))
        if not np.allclose(left, right, atol=1e-12, rtol=1e-10):
            raise ValueError(f"Frozen embedding baseline numeric mismatch in {column}: max={max_differences[column]}")
    categorical_columns = [column for column in original_columns if column not in numeric_columns]
    for column in categorical_columns:
        if original[column].astype(str).tolist() != baseline[column].astype(str).tolist():
            raise ValueError(f"Frozen embedding baseline categorical mismatch in {column}")

    inclusive = metrics[metrics["scenario"] == "inclusive_baseline_recomputed"].copy()
    filtered = metrics[metrics["scenario"] == "pure_oxide_filtered"].copy()
    delta_rows: list[dict[str, Any]] = []
    for key_values, inc_group in inclusive.groupby(keys, sort=False):
        mask = np.ones(len(filtered), dtype=bool)
        for key, value in zip(keys, key_values):
            mask &= filtered[key].to_numpy() == value
        if int(mask.sum()) != 1 or len(inc_group) != 1:
            raise ValueError(f"Embedding delta match failure: {key_values}")
        inc = inc_group.iloc[0]
        fil = filtered.loc[mask].iloc[0]
        delta = float(fil["value"] - inc["value"])
        denom = abs(float(inc["value"]))
        if bool(inc["higher_is_better"]):
            direction = "improved" if delta > 0 else ("weakened" if delta < 0 else "unchanged")
        else:
            direction = "improved" if delta < 0 else ("weakened" if delta > 0 else "unchanged")
        delta_rows.append(
            {
                "dataset": key_values[0],
                "embedding_source": key_values[1],
                "metric_name": key_values[2],
                "metric_scope": key_values[3],
                "inclusive_value": float(inc["value"]),
                "filtered_value": float(fil["value"]),
                "absolute_delta_filtered_minus_inclusive": delta,
                "relative_delta_over_abs_inclusive": delta / denom if denom > 0 else np.nan,
                "relative_delta": delta / denom if denom > 0 else np.nan,
                "inclusive_ci_low": float(inc["ci_low"]),
                "inclusive_ci_high": float(inc["ci_high"]),
                "filtered_ci_low": float(fil["ci_low"]),
                "filtered_ci_high": float(fil["ci_high"]),
                "inclusive_n_structures": int(inc["n_structures"]),
                "filtered_n_structures": int(fil["n_structures"]),
                "higher_is_better": bool(inc["higher_is_better"]),
                "direction_of_change": direction,
                "direction_of_improvement": direction,
                "interpretation_impact": "assessed by predeclared materiality rules; no delta CI inferred",
            }
        )
    deltas = pd.DataFrame(delta_rows)
    if len(deltas) != 48:
        raise ValueError(f"Expected 48 embedding deltas, found {len(deltas)}")
    protocol = {
        "source_protocol": "scripts/embedding_analysis/06_quantify_family_separation.py",
        "source_manifest": "results/embeddings/manifests/family_separation_metrics_manifest.json",
        "vector_space": "raw unprojected 256-dimensional embeddings",
        "vector_space_id": "raw_256d_embedding_vectors",
        "projected_coordinates_used": False,
        "projected_space_used": False,
        "projected_space_exclusion": "PCA, t-SNE, and UMAP coordinates were not used",
        "embedding_sources": list(SOURCES),
        "primary_embedding_source": "last_alignn_pool",
        "primary_dataset": "fixed_test_set",
        "supporting_dataset": "balanced_pool_set",
        "scenarios": ["inclusive_baseline_recomputed", "pure_oxide_filtered"],
        "filter": "exact Section 2.3B oxynitride JID membership; oxide rows only",
        "replacement_after_filtering": False,
        "rebalancing_after_filtering": False,
        "nitride_rows_removed": 0,
        "bootstrap_iterations": 1000,
        "k_neighbors": 15,
        "cv_folds": 5,
        "source_seeds": source_seeds,
        "rng_reset_for_each_scenario": True,
        "rng_reset_statement": "Each embedding scenario starts a fresh numpy.default_rng at the dataset/source seed.",
        "linear_probe_preprocessing": "fold-local StandardScaler",
        "linear_probe_solver": "lbfgs",
        "metric_execution_order": ["silhouette overall", "silhouette oxide", "silhouette nitride", "Davies-Bouldin overall", "kNN overall", "kNN oxide", "kNN nitride", "logistic AUC overall"],
        "baseline_reference_rows": len(original),
        "baseline_reproduced_within_tolerance": True,
        "baseline_numeric_tolerance": {"atol": 1e-12, "rtol": 1e-10},
        "baseline_max_absolute_differences": max_differences,
        "pre_head_equals_last_gcn_pool": duplicate_arrays,
        "fixed_test_metadata_prediction_drift": prediction_drift,
        "prediction_authority_decision": "Canonical zero-shot prediction CSVs remain the zero-shot numerical authority; frozen metadata predictions are secondary alignment fields and are not used to calculate embedding metrics.",
        "duplicate_representation_guardrail": "Identical arrays are not independent confirmations; seed-dependent CIs and folds may differ.",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "version_limitation": "The frozen source manifest did not record original package versions; compatibility is established by strict 48-row baseline reproduction.",
        "delta_ci_created": False,
    }
    return pd.DataFrame(membership_rows), metrics, deltas, protocol


def materiality_assessment(bootstrap_summary: pd.DataFrame, embedding_metrics: pd.DataFrame) -> dict[str, Any]:
    def boot(scenario: str, estimand: str) -> pd.Series:
        match = bootstrap_summary[(bootstrap_summary["scenario"] == scenario) & (bootstrap_summary["estimand"] == estimand)]
        if len(match) != 1:
            raise ValueError(f"Missing bootstrap summary row: {scenario}/{estimand}")
        return match.iloc[0]

    inclusive_diff = boot("inclusive_oxide_vs_nitride", "mae_difference_nitride_minus_oxide")
    filtered_diff = boot("pure_oxide_filtered_vs_nitride", "mae_difference_nitride_minus_oxide")
    inclusive_ratio = boot("inclusive_oxide_vs_nitride", "mae_ratio_nitride_over_oxide")
    filtered_ratio = boot("pure_oxide_filtered_vs_nitride", "mae_ratio_nitride_over_oxide")
    zero_checks = {
        "difference_direction_reversed": bool(np.sign(inclusive_diff["point_estimate"]) != np.sign(filtered_diff["point_estimate"])),
        "inclusive_difference_ci_contains_zero": bool(inclusive_diff["ci_low"] <= 0 <= inclusive_diff["ci_high"]),
        "filtered_difference_ci_contains_zero": bool(filtered_diff["ci_low"] <= 0 <= filtered_diff["ci_high"]),
        "inclusive_ratio_ci_contains_one": bool(inclusive_ratio["ci_low"] <= 1 <= inclusive_ratio["ci_high"]),
        "filtered_ratio_ci_contains_one": bool(filtered_ratio["ci_low"] <= 1 <= filtered_ratio["ci_high"]),
    }
    zero_checks["difference_ci_category_changed"] = zero_checks["inclusive_difference_ci_contains_zero"] != zero_checks["filtered_difference_ci_contains_zero"]
    zero_checks["ratio_ci_category_changed"] = zero_checks["inclusive_ratio_ci_contains_one"] != zero_checks["filtered_ratio_ci_contains_one"]
    zero_material = any(
        zero_checks[key]
        for key in ("difference_direction_reversed", "difference_ci_category_changed", "ratio_ci_category_changed")
    )

    support_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for scenario in ("inclusive_baseline_recomputed", "pure_oxide_filtered"):
            subset = embedding_metrics[
                (embedding_metrics["dataset"] == dataset)
                & (embedding_metrics["embedding_source"] == "last_alignn_pool")
                & (embedding_metrics["scenario"] == scenario)
                & (embedding_metrics["metric_scope"] == "overall_family_labels")
            ]
            lookup = {row["metric_name"]: row for _, row in subset.iterrows()}
            required = {"silhouette_score", "knn_family_purity", "logistic_regression_family_auc"}
            if set(lookup) != required | {"davies_bouldin_index"}:
                raise ValueError(f"Primary embedding metric set mismatch for {dataset}/{scenario}")
            sil = lookup["silhouette_score"]
            knn = lookup["knn_family_purity"]
            auc = lookup["logistic_regression_family_auc"]
            majority = float(knn["majority_label_baseline"])
            support_rows.append(
                {
                    "dataset": dataset,
                    "scenario": scenario,
                    "auc_point_above_half": bool(auc["value"] > 0.5),
                    "auc_ci_lower_above_half": bool(auc["ci_low"] > 0.5),
                    "silhouette_point_positive": bool(sil["value"] > 0),
                    "silhouette_ci_excludes_zero_positive": bool(sil["ci_low"] > 0),
                    "knn_point_above_majority_baseline": bool(knn["value"] > majority),
                    "knn_ci_lower_above_majority_baseline": bool(knn["ci_low"] > majority),
                    "majority_label_baseline": majority,
                    "family_recoverability_supported": bool(
                        auc["ci_low"] > 0.5 and sil["ci_low"] > 0 and knn["ci_low"] > majority
                    ),
                }
            )
    support = pd.DataFrame(support_rows)
    embedding_changes: list[dict[str, Any]] = []
    for dataset in DATASETS:
        inc = support[(support["dataset"] == dataset) & (support["scenario"] == "inclusive_baseline_recomputed")].iloc[0]
        fil = support[(support["dataset"] == dataset) & (support["scenario"] == "pure_oxide_filtered")].iloc[0]
        fields = [
            "auc_ci_lower_above_half",
            "silhouette_point_positive",
            "silhouette_ci_excludes_zero_positive",
            "knn_point_above_majority_baseline",
            "knn_ci_lower_above_majority_baseline",
            "family_recoverability_supported",
        ]
        changed = {field: bool(inc[field] != fil[field]) for field in fields}
        embedding_changes.append({"dataset": dataset, "changed_flags": changed, "any_changed": any(changed.values())})
    inc_conclusions = support[support["scenario"] == "inclusive_baseline_recomputed"].set_index("dataset")["family_recoverability_supported"].to_dict()
    fil_conclusions = support[support["scenario"] == "pure_oxide_filtered"].set_index("dataset")["family_recoverability_supported"].to_dict()
    inconsistency_introduced = len(set(inc_conclusions.values())) == 1 and len(set(fil_conclusions.values())) > 1
    embedding_material = any(item["any_changed"] for item in embedding_changes) or inconsistency_introduced
    outcome = MATERIAL if zero_material or embedding_material else STABLE
    return {
        "interpretation_result": outcome,
        "zero_shot": {
            "checks": zero_checks,
            "material_change": zero_material,
            "claim_direction_changed": zero_checks["difference_direction_reversed"],
        },
        "embedding": {
            "primary_source": "last_alignn_pool",
            "support_flags": support_rows,
            "dataset_changes": embedding_changes,
            "fixed_vs_balanced_inconsistency_introduced": inconsistency_introduced,
            "material_change": embedding_material,
            "davies_bouldin_treatment": "descriptive; no universal decision threshold",
        },
        "main_claim_wording_change_required": outcome == MATERIAL,
        "materiality_policy": {
            "zero_shot": "direction reversal, change in zero/one CI inclusion category, or required claim-category/wording change",
            "embedding": "change in prespecified primary support flags or newly inconsistent dataset-level conclusion",
            "rounded_value_change_alone_is_material": False,
            "delta_ci_created": False,
        },
    }


def build_claim_map(
    point_estimates: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    embedding_metrics: pd.DataFrame,
    deltas: pd.DataFrame,
    materiality: dict[str, Any],
) -> pd.DataFrame:
    outcome = materiality["interpretation_result"]
    rows: list[dict[str, Any]] = []

    def add(
        claim_id: str,
        text: str,
        classification: str,
        scenario: str,
        dataset: str,
        source: str,
        metric: str,
        source_path: str,
        selector: str,
        units: str,
        value: float | str,
        low: float | str,
        high: float | str,
        caveat: str,
        location: str,
    ) -> None:
        rows.append(
            {
                "claim_id": claim_id,
                "bounded_claim_text": text,
                "claim_classification": classification,
                "scenario": scenario,
                "dataset": dataset,
                "embedding_source": source,
                "metric_or_estimand": metric,
                "source_path": source_path,
                "row_selector": selector,
                "units": units,
                "point_estimate": value,
                "ci_lower": low,
                "ci_upper": high,
                "confidence_level": 0.95 if low != "" else "",
                "recalculation_status": "recomputed from numerical authority",
                "authority_level": "derived from numerical authority",
                "materiality_outcome": outcome,
                "required_caveat": caveat,
                "intended_future_manuscript_location": location,
            }
        )

    for scenario, suffix in (("inclusive_oxide_vs_nitride", "INC"), ("pure_oxide_filtered_vs_nitride", "PURE")):
        points = point_estimates[point_estimates["scenario"] == scenario].iloc[0]
        for estimand, short, units in (
            ("oxide_mae", "OXMAE", "eV/atom"),
            ("nitride_mae", "NMAE", "eV/atom"),
            ("mae_difference_nitride_minus_oxide", "DIFF", "eV/atom"),
            ("mae_ratio_nitride_over_oxide", "RATIO", "dimensionless"),
        ):
            summary = bootstrap_summary[(bootstrap_summary["scenario"] == scenario) & (bootstrap_summary["estimand"] == estimand)].iloc[0]
            add(
                f"C23C-ZS-{suffix}-{short}",
                f"Under {scenario}, the recomputed {estimand} is reported with structure-level bootstrap uncertainty.",
                "protocol-specific",
                scenario,
                "fixed family test manifests",
                "",
                estimand,
                f"{OUT_REL.as_posix()}/zero_shot_bootstrap_summary.csv",
                f"scenario={scenario};estimand={estimand}",
                units,
                float(points[estimand]),
                float(summary["ci_low"]),
                float(summary["ci_high"]),
                "Specific to the frozen checkpoint, fixed test manifests, and zero-shot protocol; not a causal attribution.",
                "Results" if scenario.startswith("inclusive") else "Appendix",
            )
    ox_row = point_estimates[point_estimates["scenario"] == "oxynitride_only_descriptive"].iloc[0]
    add(
        "C23C-ZS-OXY-MAE",
        "The 57 excluded oxide-arm oxynitrides have a descriptive zero-shot MAE.",
        "protocol-specific",
        "oxynitride_only_descriptive",
        "fixed oxide test manifest",
        "",
        "oxynitride_only_mae",
        f"{OUT_REL.as_posix()}/zero_shot_point_estimates.csv",
        "scenario=oxynitride_only_descriptive",
        "eV/atom",
        float(ox_row["oxide_mae"]),
        "",
        "",
        "Descriptive subset result; no separate bootstrap interval was prespecified.",
        "Appendix",
    )
    sensitivity = point_estimates[point_estimates["scenario"] == "sensitivity_filtered_minus_inclusive"].iloc[0]
    for field, short, units in (
        ("oxide_mae_change_filtered_minus_inclusive", "SHIFT", "eV/atom"),
        ("oxide_mae_percent_change", "PCT", "percent"),
        ("mae_difference_change_filtered_minus_inclusive", "DIFFSHIFT", "eV/atom"),
        ("mae_ratio_change_filtered_minus_inclusive", "RATIOSHIFT", "dimensionless"),
    ):
        add(
            f"C23C-ZS-SENS-{short}",
            f"The pure-oxide filter changes {field} by the reported amount.",
            "protocol-specific",
            "sensitivity_filtered_minus_inclusive",
            "fixed family test manifests",
            "",
            field,
            f"{OUT_REL.as_posix()}/zero_shot_point_estimates.csv",
            "scenario=sensitivity_filtered_minus_inclusive",
            units,
            float(sensitivity[field]),
            "",
            "",
            "This is a deterministic subset sensitivity delta, not a separately tested causal effect.",
            "Appendix",
        )
    primary = embedding_metrics[
        (embedding_metrics["embedding_source"] == "last_alignn_pool")
        & (embedding_metrics["metric_scope"] == "overall_family_labels")
    ]
    metric_short = {
        "silhouette_score": "SIL",
        "davies_bouldin_index": "DBI",
        "knn_family_purity": "KNN",
        "logistic_regression_family_auc": "AUC",
    }
    for _, row in primary.iterrows():
        dataset_short = "FIXED" if row["dataset"] == "fixed_test_set" else "POOL"
        scenario_short = "INC" if row["scenario"] == "inclusive_baseline_recomputed" else "PURE"
        add(
            f"C23C-EMB-{dataset_short}-{scenario_short}-{metric_short[row['metric_name']]}",
            f"For {row['dataset']} under {row['scenario']}, last_alignn_pool {row['metric_name']} quantifies frozen-representation family recoverability.",
            "correlational",
            row["scenario"],
            row["dataset"],
            "last_alignn_pool",
            row["metric_name"],
            f"{OUT_REL.as_posix()}/embedding_sensitivity_metrics.csv",
            f"dataset={row['dataset']};embedding_source=last_alignn_pool;scenario={row['scenario']};metric_name={row['metric_name']};metric_scope=overall_family_labels",
            "dimensionless",
            float(row["value"]),
            float(row["ci_low"]),
            float(row["ci_high"]),
            "Frozen-representation association under this protocol; it does not identify a causal mechanism.",
            "Results" if row["scenario"] == "inclusive_baseline_recomputed" else "Appendix",
        )
    add(
        "C23C-MATERIALITY",
        f"Applying the predeclared rules yields {outcome}.",
        "protocol-specific",
        "inclusive-versus-filtered sensitivity",
        "zero-shot and frozen embeddings",
        "last_alignn_pool",
        "predeclared materiality rules",
        f"{OUT_REL.as_posix()}/materiality_claim_impact.md",
        "interpretation_result",
        "categorical",
        "",
        "",
        "",
        "Materiality is interpretation-level and does not imply a causal effect.",
        "Discussion",
    )
    frame = pd.DataFrame(rows)
    if frame["claim_id"].duplicated().any():
        raise ValueError("Duplicate claim IDs")
    return frame


def materiality_markdown(materiality: dict[str, Any]) -> str:
    outcome = materiality["interpretation_result"]
    zero = materiality["zero_shot"]
    embedding = materiality["embedding"]
    lines = [
        "# Section 2.3C Materiality and Claim Impact",
        "",
        f"Interpretation result: **{outcome}**. [C23C-MATERIALITY]",
        "",
        "## Predeclared decision",
        "",
        "The comparison uses the oxide comparator and nitride target terminology established by Section 2.3A.",
        "",
        f"- Zero-shot material-change flag: `{str(zero['material_change']).lower()}`.",
        f"- Frozen-embedding material-change flag: `{str(embedding['material_change']).lower()}`.",
        f"- Main-claim wording change required: `{str(materiality['main_claim_wording_change_required']).lower()}`.",
        "- Exact numerical shifts remain reportable even when they do not cross a predeclared interpretation boundary.",
        "- No confidence interval for a sensitivity delta was manufactured by subtracting interval endpoints.",
        "- Frozen embedding-metadata predictions have a quantified secondary drift from canonical zero-shot CSV predictions; exact JID/filename/target alignment and NPZ embedding metrics remain valid, and canonical CSVs retain zero-shot numerical authority.",
        "",
        "## Zero-shot checks",
        "",
    ]
    for key, value in zero["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(["", "## Frozen-embedding checks", ""])
    for item in embedding["dataset_changes"]:
        lines.append(f"- `{item['dataset']}` any primary support flag changed: `{str(item['any_changed']).lower()}`.")
    lines.extend(
        [
            f"- New fixed-versus-balanced conclusion inconsistency: `{str(embedding['fixed_vs_balanced_inconsistency_introduced']).lower()}`.",
            "- Davies–Bouldin deltas are descriptive because no universal decision threshold was prespecified.",
            "",
            "## Section 3 handoff",
            "",
        ]
    )
    if outcome == MATERIAL:
        lines.append("The main Results and Discussion wording must be revised in Section 3; the sensitivity result must not be confined to the appendix.")
    else:
        lines.append("The primary evidence-bounded interpretation is stable. Section 3 should retain the inclusive analysis in the main Results and report the pure-oxide sensitivity in the appendix and limitations discussion.")
    lines.extend(["", "All claims remain protocol-specific or correlational as classified in `claim_evidence_map.csv`.", ""])
    return "\n".join(lines)


def report_markdown(out: Path) -> str:
    points = pd.read_csv(out / "zero_shot_point_estimates.csv")
    boot = pd.read_csv(out / "zero_shot_bootstrap_summary.csv")
    deltas = pd.read_csv(out / "embedding_sensitivity_deltas.csv")
    validation = json.loads((out / "section2_3C_validation.json").read_text(encoding="utf-8"))
    materiality_text = (out / "materiality_claim_impact.md").read_text(encoding="utf-8")
    outcome = MATERIAL if MATERIAL in materiality_text else STABLE

    def point(scenario: str) -> pd.Series:
        row = points[points["scenario"] == scenario]
        if len(row) != 1:
            raise ValueError(f"Missing point-estimate scenario {scenario}")
        return row.iloc[0]

    def interval(scenario: str, estimand: str) -> pd.Series:
        row = boot[(boot["scenario"] == scenario) & (boot["estimand"] == estimand)]
        if len(row) != 1:
            raise ValueError(f"Missing bootstrap row {scenario}/{estimand}")
        return row.iloc[0]

    inc = point("inclusive_oxide_vs_nitride")
    pure = point("pure_oxide_filtered_vs_nitride")
    oxy = point("oxynitride_only_descriptive")
    sensitivity = point("sensitivity_filtered_minus_inclusive")
    inc_diff = interval("inclusive_oxide_vs_nitride", "mae_difference_nitride_minus_oxide")
    pure_diff = interval("pure_oxide_filtered_vs_nitride", "mae_difference_nitride_minus_oxide")
    inc_ratio = interval("inclusive_oxide_vs_nitride", "mae_ratio_nitride_over_oxide")
    pure_ratio = interval("pure_oxide_filtered_vs_nitride", "mae_ratio_nitride_over_oxide")
    primary = deltas[
        (deltas["embedding_source"] == "last_alignn_pool")
        & (deltas["metric_scope"] == "overall_family_labels")
    ].copy()
    claim_ids = {
        ("fixed_test_set", "silhouette_score"): "C23C-EMB-FIXED-PURE-SIL",
        ("fixed_test_set", "davies_bouldin_index"): "C23C-EMB-FIXED-PURE-DBI",
        ("fixed_test_set", "knn_family_purity"): "C23C-EMB-FIXED-PURE-KNN",
        ("fixed_test_set", "logistic_regression_family_auc"): "C23C-EMB-FIXED-PURE-AUC",
        ("balanced_pool_set", "silhouette_score"): "C23C-EMB-POOL-PURE-SIL",
        ("balanced_pool_set", "davies_bouldin_index"): "C23C-EMB-POOL-PURE-DBI",
        ("balanced_pool_set", "knn_family_purity"): "C23C-EMB-POOL-PURE-KNN",
        ("balanced_pool_set", "logistic_regression_family_auc"): "C23C-EMB-POOL-PURE-AUC",
    }
    lines = [
        "# Section 2.3C — Oxynitride Sensitivity and Zero-Shot Bootstrap Uncertainty",
        "",
        f"Scientific verdict: **{validation.get('verdict')}**.",
        f"Interpretation result: **{outcome}**. [C23C-MATERIALITY]",
        "",
        "## Scope and governance",
        "",
        "This package reuses frozen per-structure predictions and raw 256D embeddings. It performs no model training, inference, embedding extraction, dimensionality reduction, or manuscript editing. The oxide comparator includes oxynitrides in the primary analysis; the sensitivity removes the exact 57 fixed-test oxynitride JIDs established by Section 2.3B while leaving all 242 nitride target structures unchanged.",
        "",
        "## Zero-shot point estimates and uncertainty",
        "",
        f"- Inclusive oxide comparator: n={int(inc['oxide_n'])}, MAE={inc['oxide_mae']:.12f} eV/atom. [C23C-ZS-INC-OXMAE]",
        f"- Pure-oxide sensitivity comparator: n={int(pure['oxide_n'])}, MAE={pure['oxide_mae']:.12f} eV/atom. [C23C-ZS-PURE-OXMAE]",
        f"- Removed oxynitride subset: n={int(oxy['oxide_n'])}, MAE={oxy['oxide_mae']:.12f} eV/atom. [C23C-ZS-OXY-MAE]",
        f"- Unchanged nitride target: n={int(inc['nitride_n'])}, MAE={inc['nitride_mae']:.12f} eV/atom. [C23C-ZS-INC-NMAE]",
        f"- Inclusive nitride-minus-oxide difference: {inc_diff['point_estimate']:.12f} eV/atom, 95% bootstrap CI [{inc_diff['ci_low']:.12f}, {inc_diff['ci_high']:.12f}]. [C23C-ZS-INC-DIFF]",
        f"- Filtered nitride-minus-oxide difference: {pure_diff['point_estimate']:.12f} eV/atom, 95% bootstrap CI [{pure_diff['ci_low']:.12f}, {pure_diff['ci_high']:.12f}]. [C23C-ZS-PURE-DIFF]",
        f"- Inclusive nitride-over-oxide ratio: {inc_ratio['point_estimate']:.12f}, 95% bootstrap CI [{inc_ratio['ci_low']:.12f}, {inc_ratio['ci_high']:.12f}]. [C23C-ZS-INC-RATIO]",
        f"- Filtered nitride-over-oxide ratio: {pure_ratio['point_estimate']:.12f}, 95% bootstrap CI [{pure_ratio['ci_low']:.12f}, {pure_ratio['ci_high']:.12f}]. [C23C-ZS-PURE-RATIO]",
        "",
        "The intervals use 50,000 independent within-family, structure-level resamples per scenario. Full-data statistics are the point estimates; intervals are 95% percentile intervals with NumPy's linear quantile method.",
        "",
        "## Oxynitride sensitivity",
        "",
        f"Removing oxynitrides changes oxide MAE by {sensitivity['oxide_mae_change_filtered_minus_inclusive']:.12f} eV/atom ({sensitivity['oxide_mae_percent_change']:.6f}%). [C23C-ZS-SENS-SHIFT] [C23C-ZS-SENS-PCT]",
        f"The nitride-minus-oxide difference changes by {sensitivity['mae_difference_change_filtered_minus_inclusive']:.12f} eV/atom, and the ratio changes by {sensitivity['mae_ratio_change_filtered_minus_inclusive']:.12f}. [C23C-ZS-SENS-DIFFSHIFT] [C23C-ZS-SENS-RATIOSHIFT]",
        "",
        "## Frozen-embedding sensitivity",
        "",
        "The newly recomputed 48-row inclusive baseline matches the frozen metric table within atol=1e-12 and rtol=1e-10; the maximum per-field differences are recorded in `embedding_sensitivity_protocol.json`. The original manifest did not record package versions, so strict numerical reproduction is the compatibility check. `pre_head` and `last_gcn_pool` are identical arrays in both datasets and are not counted as independent confirmations.",
        "The frozen fixed-test embedding metadata contains quantified prediction drift relative to the canonical zero-shot CSVs (1,070 exact-value mismatches, with a maximum absolute drift of 0.027367591858 eV/atom). Membership, filenames, and targets align exactly. Canonical CSV predictions remain the zero-shot numerical authority, while the NPZ arrays remain the embedding numerical authority; metadata predictions are not used in embedding metrics.",
        "",
        "Primary `last_alignn_pool` overall-metric changes (filtered minus inclusive):",
        "",
        "| Dataset | Metric | Inclusive | Filtered | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in primary.sort_values(["dataset", "metric_name"]).iterrows():
        cid = claim_ids[(row["dataset"], row["metric_name"])]
        lines.append(f"| `{row['dataset']}` | `{row['metric_name']}` | {row['inclusive_value']:.9f} | {row['filtered_value']:.9f} | {row['absolute_delta_filtered_minus_inclusive']:.9f} | [{cid}]")
    lines.extend(
        [
            "",
            "These are correlational frozen-representation metrics under the stated protocol. Davies–Bouldin changes are descriptive, and no sensitivity-delta interval is inferred by subtracting confidence-interval endpoints.",
            "",
            "## Interpretation and Section 3 impact",
            "",
        ]
    )
    if outcome == MATERIAL:
        lines.append("The predeclared interpretation boundary changed. Section 3 must revise the main Results and Discussion rather than placing the finding only in the appendix. [C23C-MATERIALITY]")
    else:
        lines.append("The predeclared interpretation boundaries did not change. Section 3 should retain the inclusive analysis as primary and report the pure-oxide sensitivity in the appendix and limitations discussion. [C23C-MATERIALITY]")
    lines.extend(
        [
            "",
            "Future trained five-seed results remain governed by mean ± standard deviation with individual seed values visible; this package does not recompute them.",
            "",
            "## Reproducibility",
            "",
            f"- Producer: `{ANALYZE_REL.as_posix()}`",
            f"- Independent validator: `{VALIDATE_REL.as_posix()}`",
            "- Numerical authorities, input hashes, schemas, package versions, RNG substreams, row selectors, and claim classifications are recorded in this evidence directory.",
            "- Section 08 was absent and unused; the nested checkout and rerun/recovery controls were excluded and untouched.",
            "",
        ]
    )
    return "\n".join(lines)


def finalize(root: Path) -> None:
    out = root / OUT_REL
    expected_existing = set(INITIAL_OUTPUTS + ["section2_3C_validation.json"])
    if not out.is_dir():
        raise FileNotFoundError(out)
    observed = {path.name for path in out.iterdir() if path.is_file()}
    if observed != expected_existing:
        raise RuntimeError(f"Pre-finalization output set mismatch: {sorted(observed)}")
    validation = json.loads((out / "section2_3C_validation.json").read_text(encoding="utf-8"))
    if validation.get("verdict") != SUCCESS:
        raise RuntimeError(f"Cannot finalize without successful validation: {validation.get('verdict')}")
    for name in ("section2_3C_report.md", "generated_output_manifest.json", "section2_3C_evidence.sha256"):
        if (out / name).exists():
            raise FileExistsError(out / name)
    (out / "section2_3C_report.md").write_text(report_markdown(out), encoding="utf-8")

    evidence_for_manifest = sorted(INITIAL_OUTPUTS + ["section2_3C_validation.json", "section2_3C_report.md"])
    manifest_paths = [OUT_REL / name for name in evidence_for_manifest] + [ANALYZE_REL, VALIDATE_REL]
    rows: list[dict[str, Any]] = []
    for rel in manifest_paths:
        path = root / rel
        rows.append(
            {
                "repository_relative_path": rel.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "producer": ANALYZE_REL.as_posix() if rel != VALIDATE_REL and rel.name != "section2_3C_validation.json" else VALIDATE_REL.as_posix(),
                "exact_input_sources": "authoritative_input_inventory.csv and the Section 2.3C derived evidence chain",
                "creation_timestamp_utc": utc_now(),
                "creation_timestamp": utc_now(),
                "intended_role": "Section 2.3C reproducible evidence or implementation",
                "authority_level": "derived evidence" if rel.suffix != ".py" else "procedural implementation",
                "independent_validation_status": "independently_validated",
            }
        )
    if len(rows) != 19:
        raise RuntimeError(f"Generated-output manifest entry count mismatch: {len(rows)}")
    write_json(
        out / "generated_output_manifest.json",
        {
            "created_at_utc": utc_now(),
            "entry_count": len(rows),
            "entries": rows,
            "self_reference_exclusions": [
                "generated_output_manifest.json excludes itself because its SHA-256 cannot contain its own final hash.",
                "section2_3C_evidence.sha256 is generated after this manifest and is excluded to avoid a manifest/checksum cycle.",
            ],
            "forbidden_inputs_included": False,
        },
    )
    checksum_paths = sorted(
        [OUT_REL / name for name in INITIAL_OUTPUTS + ["generated_output_manifest.json", "section2_3C_validation.json", "section2_3C_report.md"]]
        + [ANALYZE_REL, VALIDATE_REL],
        key=lambda item: item.as_posix(),
    )
    if len(checksum_paths) != 20:
        raise RuntimeError(f"Checksum entry count mismatch: {len(checksum_paths)}")
    lines = [f"{sha256(root / rel)}  {rel.as_posix()}" for rel in checksum_paths]
    (out / "section2_3C_evidence.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Finalized report, 19-entry output manifest, and 20-entry checksum manifest in {OUT_REL}")


def run_analysis(root: Path) -> None:
    out = root / OUT_REL
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing Section 2.3C workspace: {out}")
    pre = preflight(root)
    before = protected_hashes(root)
    zero = load_zero_shot(root)
    membership, removed, points, errors, estimates = zero_shot_outputs(zero)
    replicates, bootstrap_summary, bootstrap_protocol = bootstrap_zero_shot(errors, estimates)
    all_oxynitride_jids = set(zero["ox_inventory"]["jid"].astype(str))
    embedding_membership, embedding_metrics, embedding_deltas, embedding_protocol = load_embedding_inputs(root, all_oxynitride_jids, zero)
    materiality = materiality_assessment(bootstrap_summary, embedding_metrics)
    claim_map = build_claim_map(points, bootstrap_summary, embedding_metrics, embedding_deltas, materiality)
    after = protected_hashes(root)
    if before != after:
        changed = [path for path in before if before[path] != after[path]]
        raise RuntimeError(f"Protected input changed during analysis: {changed}")
    inventory = input_inventory(root, before, after)
    zero_integrity = {
        "created_at_utc": utc_now(),
        "status": "ZERO_SHOT_PREDICTION_INTEGRITY_VALIDATED",
        "validation_status": "passed",
        "checks": zero["integrity"],
        "abs_error_tolerance": {"atol": 1e-12, "rtol": 1e-12},
        "summary_mae_tolerance": 1e-12,
        "nitride_sample_unchanged_between_scenarios": True,
        "numerical_authority": [
            "results/zero_shot/oxide/predictions.csv",
            "results/zero_shot/nitride/predictions.csv",
        ],
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "repository_preflight.json", pre)
    write_df(out / "authoritative_input_inventory.csv", inventory)
    write_json(out / "zero_shot_prediction_integrity.json", zero_integrity)
    write_df(out / "zero_shot_membership.csv", membership)
    write_df(out / "oxynitride_zero_shot_predictions.csv", removed)
    write_df(out / "zero_shot_point_estimates.csv", points)
    write_df(out / "zero_shot_bootstrap_replicates.csv", replicates)
    write_df(out / "zero_shot_bootstrap_summary.csv", bootstrap_summary)
    write_json(out / "zero_shot_bootstrap_protocol.json", bootstrap_protocol)
    write_df(out / "embedding_membership_audit.csv", embedding_membership)
    write_df(out / "embedding_sensitivity_metrics.csv", embedding_metrics)
    write_df(out / "embedding_sensitivity_deltas.csv", embedding_deltas)
    write_json(out / "embedding_sensitivity_protocol.json", embedding_protocol)
    (out / "materiality_claim_impact.md").write_text(materiality_markdown(materiality), encoding="utf-8")
    write_df(out / "claim_evidence_map.csv", claim_map)
    observed = sorted(path.name for path in out.iterdir() if path.is_file())
    if observed != sorted(INITIAL_OUTPUTS):
        raise RuntimeError(f"Initial output set mismatch: {observed}")
    print(json.dumps({
        "initial_outputs": len(observed),
        "zero_shot_membership_rows": len(membership),
        "oxynitride_prediction_rows": len(removed),
        "bootstrap_replicate_rows": len(replicates),
        "embedding_metric_rows": len(embedding_metrics),
        "embedding_delta_rows": len(embedding_deltas),
        "interpretation_result": materiality["interpretation_result"],
        "ready_for_independent_validation": True,
    }, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true", help="After independent validation, create report and the two manifests.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    if args.finalize:
        finalize(root)
    else:
        run_analysis(root)


if __name__ == "__main__":
    main()
