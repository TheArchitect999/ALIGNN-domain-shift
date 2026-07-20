#!/usr/bin/env python3
"""Build the Section 2.3D terminology and claim-reconciliation package.

This producer is deliberately allowlist driven.  It reads tracked scientific
and editorial material plus three explicitly permitted historical governance
files, and writes only outputs 1--12 in the dedicated Section 2.3D directory.
It never enters rerun staging, the nested checkout, or forbidden Section 08.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

EXPECTED_ROOT = Path(".")
EXPECTED_BRANCH = "main"
EXPECTED_HEAD = "9e2c640bec56fd67d77989cc45845d6d241bf28d"
EXPECTED_SUBJECT = "evidence: checkpoint Section 2.3C oxynitride sensitivity and bootstrap analysis"
EXPECTED_UPSTREAM = "main...origin/main [ahead 5, behind 3]"
SUCCESS = "SECTION23D_TERMINOLOGY_CLAIMS_VALIDATED"
BLOCKED = "SECTION23D_BLOCKED"

OUT_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3D_terminology_claim_reconciliation"
)
PRODUCER_REL = Path("scripts/shared/audit_terminology_claims.py")
VALIDATOR_REL = Path("scripts/shared/validate_terminology_claims.py")
SECTION08_REL = Path("results/derived_evidence/final_paper_factory/archived_submission_materials")
NESTED_REL = "domain_shift-alignn-domain-shift"

PRODUCER_OUTPUTS = (
    "audit_scope_manifest.json",
    "authoritative_input_inventory.csv",
    "claim_bearing_file_inventory.csv",
    "terminology_occurrence_inventory.csv",
    "terminology_disposition.csv",
    "numerical_claim_reconciliation.csv",
    "rewrite_action_matrix.csv",
    "future_manuscript_claim_registry.csv",
    "governance_reconciliation.md",
    "section3_rewrite_specification.md",
    "coverage_summary.json",
    "repository_preflight.json",
)
FINAL_OUTPUTS = (
    "generated_output_manifest.json",
    "section2_3D_validation.json",
    "section2_3D_report.md",
    "section2_3D_evidence.sha256",
)
ALL_OUTPUTS = PRODUCER_OUTPUTS + FINAL_OUTPUTS
ALL_NEW_RELS = {
    PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix(),
    *{(OUT_REL / name).as_posix() for name in ALL_OUTPUTS},
}

ROOTS = (
    "readme.md", "docs", "results/derived_evidence/readme.md",
    "results/derived_evidence/final_paper_factory/00_source_of_truth",
    "results/derived_evidence/final_paper_factory/01_blueprints",
    "results/derived_evidence/final_paper_factory/02_figure_memos",
    "results/derived_evidence/final_paper_factory/03_section_inputs",
    "results/derived_evidence/final_paper_factory/04_drafts",
    "results/derived_evidence/final_paper_factory/05_reviewed_drafts",
    "results/derived_evidence/final_paper_factory/06_template_ready",
    "results/derived_evidence/final_paper_factory/07_final_qc",
    "results/derived_evidence/protocol_1", "results/derived_evidence/protocol_2",
    "results/derived_evidence/protocol_3", "results/reproduction/embeddings",
    "results/zero_shot", "results/derived_evidence/provenance",
    "results/derived_evidence/protocol_1_promotion",
    "results/derived_evidence/protocol_1_regeneration",
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap",
)
PROSE_SUFFIXES = {".md", ".txt", ".tex", ".rst"}

STRUCTURED_SCAN_RULES: dict[str, tuple[str, ...] | str] = {
    "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv":
        ("interpretation_label", "ambiguity_note", "formula"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_support_map_v3.csv":
        ("claim_text", "claim", "support_note", "notes", "guardrail"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_to_number_source_map_v3.csv":
        ("claim_text", "notes", "mapping_note"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.csv":
        ("description", "intended_role", "notes", "claim"),
    "results/derived_evidence/protocol_1_regeneration/manuscript_claim_impact.csv":
        ("existing_sentence_or_identifier", "old_numerical_claim",
         "corrected_numerical_value_or_interpretation", "required_action_in_section_3"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv":
        ("bounded_claim_text", "required_caveat"),
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_validation.json": "all_string_values",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json": "all_string_values",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_validation.json": "all_string_values",
}

GOVERNANCE_HASHES = {
    "results/derived_evidence/input_manifest.md":
        "735044376ffe562f61fc04343a7cbe1c294f7cf83b8add7e6d65bdc595af6494",
    "results/derived_evidence/source_policy.md":
        "0fc51ef87db6cb8c3615cb90cc8110ee26f89491208a296373b8ad63a8a3bacb",
    "results/derived_evidence/run_session.json":
        "647ab1653e2fb8d9e7bbc8843d0f034d3614ec17ee1e7f4b2cca24f6de40cdcb",
}

GATE_MANIFESTS = (
    ("results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256", "repo", 19),
    ("results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256", "parent", 22),
    ("results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_evidence.sha256", "repo", 31),
    ("results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_evidence.sha256", "repo", 24),
    ("results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_evidence.sha256", "repo", 20),
)

AUTHORITIES: dict[str, tuple[str, str, str, str]] = {
    **{p: ("historical_governance", "hash-pinned historical governance input",
           "governance authority", "historical") for p in GOVERNANCE_HASHES},
    "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv":
        ("corrected_protocol_1", "validated five-seed protocol_1 aggregates and checkpoint vectors", "numerical authority", "current"),
    "results/derived_evidence/protocol_1_regeneration/manuscript_claim_impact.csv":
        ("corrected_protocol_1", "462-row stale-claim quarantine and rewrite map", "governance authority", "current"),
    "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md":
        ("corrected_protocol_1", "Section 2.2 validation verdict", "validation authority", "current"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv":
        ("source_of_truth_v3", "versioned paper-number registry", "numerical authority", "current"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.md":
        ("source_of_truth_v3", "human-readable number registry", "secondary reference", "current"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_support_map_v3.csv":
        ("source_of_truth_v3", "claim support map subject to Section 2.3D conflict resolutions", "secondary reference", "current_with_supersessions"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_to_number_source_map_v3.csv":
        ("source_of_truth_v3", "claim-number map subject to Section 2.3D conflict resolutions", "secondary reference", "current_with_supersessions"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.csv":
        ("source_of_truth_v3", "versioned evidence routing map", "governance authority", "current"),
    "results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.md":
        ("source_of_truth_v3", "human-readable evidence routing map", "secondary reference", "current"),
    "results/derived_evidence/final_paper_factory/03_section_inputs/combined_paper_results_III_and_IV_protocol_1_correction_packet_v1.md":
        ("corrected_handoff", "combined corrected Section 3 handoff", "future manuscript input", "current"),
    "results/derived_evidence/final_paper_factory/03_section_inputs/joint_comparison_packet_v2_protocol_1_corrected.md":
        ("corrected_handoff", "joint comparison rewrite input", "future manuscript input", "current"),
    "results/derived_evidence/final_paper_factory/03_section_inputs/oxide_results_packet_v2_protocol_1_corrected.md":
        ("corrected_handoff", "oxide rewrite input", "future manuscript input", "current"),
    "results/derived_evidence/final_paper_factory/03_section_inputs/nitride_results_packet_v2_protocol_1_corrected.md":
        ("corrected_handoff", "nitride rewrite input", "future manuscript input", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_membership_terminology_decision.md":
        ("checkpoint_provenance", "bounded checkpoint-membership terminology", "governance authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv":
        ("checkpoint_provenance", "exact fixed-test/checkpoint JID overlaps", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/dataset_integrity_decision.md":
        ("dataset_integrity", "validated dataset-definition decision", "governance authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json":
        ("dataset_integrity", "dataset-integrity validation verdict and facts", "validation authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_dataset_integrity.json":
        ("dataset_integrity", "deduplication and assignment counts", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_split_integrity.json":
        ("dataset_integrity", "global split counts and exceptions", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/family_definition_audit.json":
        ("dataset_integrity", "asymmetric family-definition counts", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json":
        ("dataset_integrity", "oxynitride definition and counts", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv":
        ("zero_shot_sensitivity", "30 bounded uncertainty and embedding claims", "governance authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_bootstrap_summary.csv":
        ("zero_shot_sensitivity", "structure-bootstrap point estimates and intervals", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_point_estimates.csv":
        ("zero_shot_sensitivity", "zero-shot sensitivity point estimates", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv":
        ("embedding_sensitivity", "recomputed frozen-embedding metrics", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_deltas.csv":
        ("embedding_sensitivity", "inclusive-versus-filtered metric deltas", "numerical authority", "current"),
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/materiality_claim_impact.md":
        ("zero_shot_sensitivity", "predeclared materiality decision", "governance authority", "current"),
}
for _manifest, _mode, _count in GATE_MANIFESTS:
    AUTHORITIES.setdefault(_manifest, ("prerequisite_gate", "checksum gate", "validation authority", "current"))

CHECKPOINT_COMMITS = (
    ("acf373d5ebbaaf5b2d8d8fbc8aa1319e89eba18c", "evidence: checkpoint Section 2.3A pretrained provenance audit"),
    ("3d723cb5e503868a737665b30dc526c365f3365b", "evidence: checkpoint Section 2.3B dataset integrity audit"),
    (EXPECTED_HEAD, EXPECTED_SUBJECT),
)

ACTIONS = (
    "KEEP_SUPPORTED", "KEEP_WITH_QUALIFICATION", "KEEP_LITERATURE_CONTEXT",
    "KEEP_HISTORICAL_RECORD", "REPLACE_TERMINOLOGY", "REPLACE_NUMBER",
    "REWRITE_INTERPRETATION", "WITHDRAW_UNSUPPORTED", "DO_NOT_REUSE",
    "NONCLAIM_METADATA",
)
CLAIM_CLASSES = (
    "robust", "protocol-specific", "correlational", "unsupported",
    "procedural", "literature-context", "historical-only", "nonclaim",
)

H = r"[ \t\u00a0\-\u2010-\u2015]+"
PATTERN_SPECS = (
    ("ood_penalty", rf"OOD{H}penalt(?:y|ies)"),
    ("out_of_distribution", rf"out{H}of{H}distribution"),
    ("in_distribution", rf"in{H}distribution"),
    ("oxide_pretrained", rf"oxide{H}pre[-\u2010-\u2015]?train(?:ed|ing)"),
    ("oxide_pretrained", rf"pre[-\u2010-\u2015]?train(?:ed|ing){H}(?:(?:primarily|exclusively){H})?(?:on|using){H}(?:the{H})?oxides?"),
    ("source_domain", rf"source{H}domain"),
    ("target_domain", rf"target{H}domain"),
    ("domain_gap", rf"domain{H}gap"),
    ("distribution_shift", rf"distribution{H}shift"),
    ("domain_shift", rf"domain{H}shift"),
    ("transfer_benefit", rf"transfer{H}benefit"),
    ("transfer_gain", rf"transfer{H}gain"),
    ("adaptation_onset", rf"adaptation{H}onset"),
    ("genuine_adaptation", rf"genuine{H}adaptation"),
    ("no_parameter_updates", rf"no{H}parameter{H}updates?"),
    ("byte_identity", rf"byte{H}identit(?:y|ical)"),
    ("zero_shot_state", rf"zero[-\u2010-\u2015]?shot{H}state"),
    ("checkpoint_pinned", rf"(?:pinned|does{H}not{H}move)"),
    ("inert", r"inert(?:ness)?"),
    ("causal_language", r"(?:explain(?:s|ed|ing|ation)?|caus(?:e|es|ed|ing|al|ally|ality)|driv(?:e|es|en|ing)|mechanis(?:m|ms|tic|tically))"),
    ("significance_language", r"signific(?:ant|antly|ance)"),
    ("robustness_language", r"robust(?:ness|ly)?"),
    ("generalization_language", r"generali[sz](?:e|es|ed|ing|ation|ability|able)"),
    ("ood", r"OOD"),
)
COMPILED_PATTERNS = tuple(
    (name, re.compile(rf"(?<![\w])(?:{pattern})(?![\w])", re.IGNORECASE))
    for name, pattern in PATTERN_SPECS
)


class AuditFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditFailure(message)


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    process = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    require(process.returncode == 0, f"git {' '.join(args)} failed: {process.stderr.decode(errors='replace').strip()}")
    return process.stdout if binary else process.stdout.decode("utf-8", errors="surrogateescape").strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csv_dump(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def text_dump(path: Path, text: str) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.write_text(text.rstrip() + "\n", encoding="utf-8", newline="\n")


def safe_rel(value: str) -> str:
    rel = Path(value).as_posix()
    require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, f"Unsafe path: {value}")
    require("archived_submission_materials" not in rel, f"Forbidden Section 08 path: {rel}")
    require(not (rel == NESTED_REL or rel.startswith(NESTED_REL + "/")), f"Nested checkout path: {rel}")
    require("finetune_last2_reproduction_rerun" not in rel, f"Rerun staging path: {rel}")
    require("finetune_reproduction_rerun" not in rel, f"Rerun config path: {rel}")
    require("Summaries/finetune/reproduction_rerun" not in rel, f"Temporary summary path: {rel}")
    return rel


def parse_status(root: Path) -> list[dict[str, str]]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=root, capture_output=True, check=True,
    ).stdout
    records = [part for part in raw.split(b"\0") if part]
    entries: list[dict[str, str]] = []
    index = 0
    while index < len(records):
        record = records[index].decode("utf-8", errors="surrogateescape")
        code, path = record[:2], record[3:]
        entries.append({"status": code, "path": path})
        if code[0] in "RC":
            index += 1
        index += 1
    return sorted(entries, key=lambda item: (item["path"], item["status"]))


def status_category(path: str) -> str:
    if path in ALL_NEW_RELS:
        return "section23d_new"
    if "/finetune_last2_reproduction_rerun/" in path:
        return "corrected_rerun_staging_artifact"
    if path.startswith("configs/protocol_1/finetune_reproduction_rerun/"):
        return "rerun_configuration_recovery_control"
    if path.startswith("results/summaries/protocol_1/finetune/reproduction_rerun/"):
        return "temporary_rerun_summary"
    if path in GOVERNANCE_HASHES:
        return "historical_governance_session_file"
    if path == NESTED_REL or path.startswith(NESTED_REL + "/"):
        return "nested_checkout_entry"
    return "ambiguous"


def verify_section08(root: Path) -> dict[str, Any]:
    candidate = root / SECTION08_REL
    exists = candidate.exists() or candidate.is_symlink()
    tracked = str(git(root, "ls-files", "--", SECTION08_REL.as_posix())).splitlines()
    status = [x for x in parse_status(root) if x["path"].startswith(SECTION08_REL.as_posix() + "/")]
    require(not exists and not tracked and not status, "Section 08 exists or has Git entries")
    return {"path": SECTION08_REL.as_posix(), "filesystem_absent": True,
            "tracked_entries": 0, "status_entries": 0, "content_inspected": False}


def verify_baseline(root: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    require(str(git(root, "rev-parse", "--show-toplevel")) == str(root), "Unexpected repository root")
    head = str(git(root, "rev-parse", "HEAD"))
    branch = str(git(root, "branch", "--show-current"))
    subject = str(git(root, "show", "-s", "--format=%s", "HEAD"))
    branch_line = str(git(root, "status", "--short", "--branch", "--untracked-files=no")).splitlines()[0]
    upstream = branch_line.removeprefix("## ")
    require((head, branch, subject, upstream) ==
            (EXPECTED_HEAD, EXPECTED_BRANCH, EXPECTED_SUBJECT, EXPECTED_UPSTREAM),
            f"Starting Git identity differs: {head}, {branch}, {subject}, {upstream}")
    entries = parse_status(root)
    baseline = [item for item in entries if status_category(item["path"]) != "section23d_new"]
    new = [item for item in entries if status_category(item["path"]) == "section23d_new"]
    require(all(item["status"] == "??" for item in baseline), "Tracked or staged baseline change detected")
    counts = Counter(status_category(item["path"]) for item in baseline)
    expected = {
        "corrected_rerun_staging_artifact": 504,
        "rerun_configuration_recovery_control": 36,
        "temporary_rerun_summary": 2,
        "historical_governance_session_file": 3,
        "nested_checkout_entry": 1,
    }
    require(len(baseline) == 546 and dict(counts) == expected,
            f"Unexpected filtered baseline: {len(baseline)} {dict(counts)}")
    require(not any(status_category(item["path"]) == "ambiguous" for item in entries),
            "Ambiguous working-tree entry detected")
    require(all(item["path"] in ALL_NEW_RELS and item["status"] == "??" for item in new),
            "Unexpected Section 2.3D status entry")
    log = str(git(root, "log", "-6", "--format=%H%x09%s")).splitlines()
    for commit, wanted_subject in CHECKPOINT_COMMITS:
        actual = str(git(root, "show", "-s", "--format=%s", commit))
        require(actual == wanted_subject, f"Checkpoint subject mismatch: {commit}")
    ancestry = [str(git(root, "merge-base", "--is-ancestor", CHECKPOINT_COMMITS[i][0],
                        CHECKPOINT_COMMITS[i + 1][0])) == "" for i in range(2)]
    require(all(ancestry), "Section 2.3 checkpoint order is invalid")
    captured = str(git(root, "show", "-s", "--format=%cI", "HEAD"))
    baseline_serialized = "\n".join(f"{x['status']} {x['path']}" for x in baseline) + "\n"
    state = {
        "head": head, "branch": branch, "head_subject": subject,
        "cached_upstream_relationship": upstream, "captured_at_frozen_head_time": captured,
        "latest_six_commits": log, "tracked_modifications": 0, "staged_files": 0,
        "filtered_original_untracked_count": len(baseline),
        "filtered_original_untracked_categories": dict(sorted(counts.items())),
        "filtered_original_status_sha256": sha256_bytes(baseline_serialized.encode()),
        "filtered_original_status_entries": baseline,
        "section23d_entries_present_at_preflight": new,
        "excluded_content_preservation_method":
            "Exact Git-status path/status equality; excluded content was not opened or hashed by this phase.",
    }
    return state, baseline


def verify_manifest(root: Path, rel: str, mode: str, expected_count: int) -> dict[str, Any]:
    manifest = root / rel
    lines = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(len(lines) == expected_count, f"Unexpected entry count in {rel}")
    base = manifest.parent if mode == "parent" else root
    checked: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum line in {rel}")
        expected_hash, named = match.groups()
        candidate = (base / named).resolve()
        require(candidate.is_relative_to(root.resolve()), f"Checksum path escapes repository: {named}")
        relative = candidate.relative_to(root.resolve()).as_posix()
        safe_rel(relative)
        require(candidate.is_file() and not candidate.is_symlink(), f"Missing checksum target: {relative}")
        require(sha256(candidate) == expected_hash, f"Checksum mismatch: {relative}")
        checked.append(relative)
    return {"manifest": rel, "base_mode": mode, "entry_count": len(checked), "all_passed": True}


def verify_gates(root: Path) -> list[dict[str, Any]]:
    results = [verify_manifest(root, *spec) for spec in GATE_MANIFESTS]
    report = (root / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md").read_text(encoding="utf-8")
    require("protocol_1_REGENERATION_VALIDATED" in report, "protocol_1 regeneration verdict missing")
    verdict_paths = {
        "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP": "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_validation.json",
        "SECTION23B_DATASET_INTEGRITY_VALIDATED": "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json",
        "SECTION23C_OXYNITRIDE_BOOTSTRAP_VALIDATED": "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_validation.json",
    }
    for verdict, rel in verdict_paths.items():
        payload = json.loads((root / rel).read_text(encoding="utf-8"))
        require(payload.get("verdict") == verdict, f"Prerequisite verdict mismatch: {rel}")
        results.append({"verdict_file": rel, "verdict": verdict, "passed": True})
    results.append({"verdict_file": "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md",
                    "verdict": "protocol_1_REGENERATION_VALIDATED", "passed": True})
    return results


def verify_governance(root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for rel, expected in GOVERNANCE_HASHES.items():
        value = sha256(root / rel)
        require(value == expected, f"Historical governance hash mismatch: {rel}")
        actual[rel] = value
    return actual


def governance_secret_scan(root: Path) -> dict[str, Any]:
    """Conservative local scan of the explicitly allowlisted session record."""
    rel = "results/derived_evidence/run_session.json"
    text = (root / rel).read_text(encoding="utf-8")
    patterns = {
        "private_key_block": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "github_token": r"gh[pousr]_[A-Za-z0-9]{30,}",
        "aws_access_key": r"AKIA[0-9A-Z]{16}",
        "assigned_secret": r"(?i)(?:api[_-]?key|access[_-]?token|secret)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{20,}",
    }
    matches = [name for name, pattern in patterns.items() if re.search(pattern, text)]
    require(not matches, f"Potential secret detected in run_session.json: {matches}")
    return {"path": rel, "patterns_checked": sorted(patterns), "detected_secret_count": 0}


def tracked_corpus(root: Path) -> list[str]:
    raw = git(root, "ls-files", "-z", "--", *ROOTS, binary=True)
    assert isinstance(raw, bytes)
    paths = sorted(part.decode("utf-8") for part in raw.split(b"\0") if part)
    require(len(paths) == 1553, f"Allowlisted tracked corpus changed: {len(paths)}")
    for rel in paths:
        safe_rel(rel)
        mode = str(git(root, "ls-files", "-s", "--", rel)).split()[0]
        require(mode == "100644", f"Non-regular tracked corpus entry: {rel} mode={mode}")
    return paths


def currentness(rel: str) -> str:
    low = rel.lower()
    if any(token in low for token in ("_v1.", "_v2.", "polished_v2", "before_correction", "old_", "legacy")):
        return "historical"
    if rel in STRUCTURED_SCAN_RULES or "_v3" in low or "polished_v3" in low or "polished_v4" in low or "protocol_1_corrected" in low:
        return "current_or_current_audit_target"
    if "/04_drafts/" in rel or "/05_reviewed_drafts/" in rel or "/06_template_ready/" in rel:
        return "editorial_version_requires_rebuild"
    if "/evidence/" in rel:
        return "current_evidence_record"
    return "mixed_or_historical_audit_corpus"


def file_classification(rel: str) -> str:
    suffix = Path(rel).suffix.lower()
    low = rel.lower()
    if suffix not in PROSE_SUFFIXES | {".csv", ".json"}:
        return "Non-claim metadata"
    if "literature" in low or "bibliograph" in low or "references" in low:
        return "Literature context"
    if rel in AUTHORITIES:
        level = AUTHORITIES[rel][2]
        if level == "numerical authority":
            return "Current numerical authority"
        return "Current procedural/governance authority"
    if any(token in low for token in ("polished_v3", "polished_v4", "template_ready_v3")):
        return "Editorial audit target"
    if "protocol_1_corrected" in low or "correction_packet" in low or "_v3" in low:
        return "Future manuscript input"
    if currentness(rel) == "historical":
        return "Historical-only"
    if "/evidence/" in rel or "/provenance/" in rel:
        return "Evidence record"
    return "Editorial audit target"


def claim_status(rel: str) -> tuple[str, str, str]:
    suffix = Path(rel).suffix.lower()
    if suffix in PROSE_SUFFIXES:
        return ("broad_prose", "included", "Tracked user-facing prose under an explicit allowlisted root")
    if rel in STRUCTURED_SCAN_RULES:
        return ("structured_claim_fields", "included", "Explicit claim/evidence CSV or validation JSON; selected text fields only")
    if suffix in {".csv", ".json"}:
        return ("structured_inventory_only", "not_scanned", "Structured numeric or metadata file; inventoried but not prose-scanned")
    return ("nonprose_inventory_only", "not_scanned", "Binary, figure, checksum, or implementation artifact; explicitly outside prose scan")


def corpus_inventory(root: Path, paths: list[str]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for rel in paths:
        path = root / rel
        digest = sha256(path)
        hashes[rel] = digest
        tier, bearing, rationale = claim_status(rel)
        rows.append({
            "repository_relative_path": rel, "sha256": digest,
            "corpus_tier": tier, "file_classification": file_classification(rel),
            "version_currentness": currentness(rel), "claim_bearing_status": bearing,
            "audit_inclusion_rationale": rationale,
        })
    return rows, hashes


def authority_inventory(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tracked_set = set(str(git(root, "ls-files", "--", *AUTHORITIES.keys())).splitlines())
    for rel in sorted(AUTHORITIES):
        safe_rel(rel)
        path = root / rel
        require(path.is_file() and not path.is_symlink(), f"Authority missing or symlinked: {rel}")
        category, role, level, state = AUTHORITIES[rel]
        tracked = "tracked" if rel in tracked_set else "untracked"
        require(tracked == ("untracked" if rel in GOVERNANCE_HASHES else "tracked"),
                f"Unexpected Git status for authority: {rel}")
        modified = dt.datetime.fromtimestamp(path.stat().st_mtime, tz=dt.timezone.utc).isoformat()
        rows.append({
            "category": category, "repository_relative_path": rel,
            "git_status": tracked, "size_bytes": path.stat().st_size,
            "modification_time_utc": modified, "sha256": sha256(path),
            "intended_role": role, "authority_level": level,
            "current_historical_status": state,
        })
    return rows


def accepted_spans(text: str) -> list[tuple[int, int, str, str]]:
    candidates: list[tuple[int, int, int, str, str]] = []
    for priority, (normalized, pattern) in enumerate(COMPILED_PATTERNS):
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), priority, normalized, match.group(0)))
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[2], x[3]))
    accepted: list[tuple[int, int, str, str]] = []
    occupied_until = -1
    for start, end, _priority, normalized, matched in candidates:
        if start < occupied_until:
            continue
        accepted.append((start, end, normalized, matched))
        occupied_until = end
    return accepted


def bounded(text: str, start: int, end: int, width: int = 180) -> str:
    left = max(0, start - width)
    right = min(len(text), end + width)
    return re.sub(r"\s+", " ", text[left:right]).strip()


def context_class(rel: str, text: str, start: int, classification: str) -> str:
    low = text.lower()
    before = text[:start]
    if before.count("`") % 2 == 1 or re.search(r"(?:^|\s)(?:[\w.-]+/){2,}", text):
        return "metadata"
    if classification == "Literature context" or any(x in low for x in ("et al.", "cited literature", "related work", "prior work")):
        return "literature"
    if classification == "Historical-only" or any(x in low for x in ("historical quotation", "obsolete wording", "stale claim")):
        return "history"
    if classification in {"Evidence record", "Current procedural/governance authority"} or any(
            x in low for x in ("do not", "must not", "does not establish", "withdraw", "prohibited", "unsafe", "not causal", "not a mechanism")):
        return "evidence_record"
    return "project_claim"


def scan_text_unit(rel: str, line_number: int, source_label: str, text: str,
                   excerpt_bytes: bytes, classification: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start, end, normalized, matched in accepted_spans(text):
        excerpt_hash = sha256_bytes(excerpt_bytes)
        seed = f"{rel}\0{line_number}\0{start}\0{end}\0{normalized}\0{excerpt_hash}\0{source_label}"
        rows.append({
            "occurrence_id": "T23D-" + sha256_bytes(seed.encode())[:24],
            "path": rel, "line_number": line_number,
            "column_start": start + 1, "column_end": end,
            "matched_term": matched, "normalized_term": normalized,
            "bounded_context": (f"[{source_label}] " if source_label else "") + bounded(text, start, end),
            "excerpt_sha256": excerpt_hash, "file_classification": classification,
            "context_classification": context_class(rel, text, start, classification),
        })
    return rows


def walk_json_strings(value: Any, pointer: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for key in sorted(value):
            escaped = str(key).replace("~", "~0").replace("/", "~1")
            yield from walk_json_strings(value[key], pointer + "/" + escaped)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from walk_json_strings(item, pointer + f"/{index}")
    elif isinstance(value, str):
        yield pointer or "/", value


def scan_corpus(root: Path, paths: list[str]) -> list[dict[str, Any]]:
    occurrences: list[dict[str, Any]] = []
    for rel in paths:
        suffix = Path(rel).suffix.lower()
        classification = file_classification(rel)
        if suffix in PROSE_SUFFIXES:
            raw = (root / rel).read_bytes()
            text = raw.decode("utf-8")
            for line_number, line in enumerate(text.splitlines(), start=1):
                occurrences.extend(scan_text_unit(rel, line_number, "", line,
                                                   line.encode("utf-8"), classification))
        elif rel in STRUCTURED_SCAN_RULES and suffix == ".csv":
            with (root / rel).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                wanted = set(STRUCTURED_SCAN_RULES[rel])
                for row_number, row in enumerate(reader, start=1):
                    for column in sorted(set(row) & wanted):
                        value = row.get(column) or ""
                        occurrences.extend(scan_text_unit(
                            rel, row_number + 1, f"csv:{column}", value,
                            value.encode("utf-8"), classification))
        elif rel in STRUCTURED_SCAN_RULES and suffix == ".json":
            payload = json.loads((root / rel).read_text(encoding="utf-8"))
            for virtual_line, (pointer, value) in enumerate(walk_json_strings(payload), start=1):
                occurrences.extend(scan_text_unit(
                    rel, virtual_line, f"json-pointer:{pointer}", value,
                    value.encode("utf-8"), classification))
    occurrences.sort(key=lambda row: (row["path"], int(row["line_number"]),
                                       int(row["column_start"]), row["normalized_term"],
                                       row["occurrence_id"]))
    require(len({row["occurrence_id"] for row in occurrences}) == len(occurrences),
            "Duplicate occurrence IDs")
    return occurrences


def disposition(row: dict[str, Any]) -> dict[str, Any]:
    term = row["normalized_term"]
    context = row["context_classification"]
    text = row["bounded_context"].lower()
    guardrail = any(x in text for x in ("do not", "must not", "does not establish", "withdraw",
                                        "prohibited", "unsafe", "not causal", "not a mechanism",
                                        "not uniformly inert", "no blanket"))
    action = "KEEP_WITH_QUALIFICATION"
    claim_class = "protocol-specific"
    rationale = "Retain only with the bounded protocol and evidence caveats in the Section 3 contract."
    replacement = ""
    evidence = "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_membership_terminology_decision.md"
    selector = "approved_wording"
    caveat = "Do not infer statistical distribution status or causality from family labels alone."
    location = "Discussion"
    if context == "metadata":
        action, claim_class = "NONCLAIM_METADATA", "nonclaim"
        rationale, evidence, selector, caveat, location = (
            "The match is a path, identifier, or code-like token rather than a scientific claim.", "", "", "", "Not applicable")
    elif context == "literature":
        action, claim_class = "KEEP_LITERATURE_CONTEXT", "literature-context"
        rationale = "Permitted only as accurate, cited literature terminology; it is not project evidence."
        evidence, selector, caveat, location = "", "", "Attribute the terminology explicitly to the cited source.", "Related Work"
    elif context == "history":
        claim_class = "historical-only"
        if term in {"ood", "ood_penalty", "in_distribution", "out_of_distribution", "oxide_pretrained",
                    "causal_language", "inert", "adaptation_onset", "genuine_adaptation"}:
            action = "DO_NOT_REUSE"
            rationale = "Historical wording is preserved for audit only and is forbidden in the rebuilt manuscript."
        else:
            action = "KEEP_HISTORICAL_RECORD"
            rationale = "Historical occurrence may remain in the audit trail but is not an authority."
        caveat, location = "Historical-only; never cite as current evidence.", "Historical record only"
    elif context == "evidence_record" and guardrail:
        action, claim_class = "KEEP_SUPPORTED", "procedural"
        rationale = "This occurrence records a current prohibition, limitation, or corrected guardrail."
        caveat, location = "Retain the negation or prohibition; do not quote the prohibited phrase without it.", "Methods/Limitations"
    elif term in {"ood", "ood_penalty", "in_distribution", "out_of_distribution"}:
        action, claim_class = "WITHDRAW_UNSUPPORTED", "unsupported"
        replacement = "oxide comparator / nitride target / chemical-family performance gap"
        rationale = "Exact JID non-overlap does not establish statistical ID/OOD or source/target-domain membership."
        caveat, location = "State 0/1484 and 0/242 exact overlaps without distribution-status inference.", "Title/Abstract/Results/Discussion"
    elif term in {"source_domain", "target_domain", "domain_gap"}:
        action, claim_class = "REPLACE_TERMINOLOGY", "protocol-specific"
        replacement = "oxide comparator / nitride target / chemical-family performance gap"
        rationale = "Use the verified family-comparison language instead of an unverified domain-membership label."
        caveat, location = "Do not infer statistical distribution status from family labels.", "Title/Abstract/Results/Discussion"
    elif term == "oxide_pretrained":
        action, claim_class = "WITHDRAW_UNSUPPORTED", "unsupported"
        replacement = "JARVIS-DFT pretrained ALIGNN checkpoint"
        rationale = "The official checkpoint is not established as oxide-only pretraining."
        caveat, location = "Do not characterize checkpoint training chemistry beyond verified membership facts.", "Methods"
    elif term == "causal_language":
        action, claim_class = "REWRITE_INTERPRETATION", "correlational"
        replacement = "is associated with / is consistent with / supports a representation-level interpretation"
        evidence = "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv"
        selector = "claim_classification=correlational"
        rationale = "Frozen-embedding separation is correlational and protocol-specific, not a causal mechanism."
        caveat, location = "No causal attribution or independent-confirmation claim.", "Results/Discussion"
    elif term in {"inert", "adaptation_onset", "genuine_adaptation", "checkpoint_pinned",
                  "zero_shot_state", "no_parameter_updates", "byte_identity"}:
        evidence = "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv"
        selector = "family=nitride;N=200"
        caveat = "best_epoch=1 selects the end-of-epoch-1 checkpoint; it does not prove no update or byte identity."
        location = "Results/Discussion"
        if guardrail or "49,1,1,1,1" in text:
            action, claim_class = "KEEP_SUPPORTED", "procedural"
            rationale = "Corrected wording rejects the obsolete blanket checkpoint-timing interpretation."
        else:
            action, claim_class = "WITHDRAW_UNSUPPORTED", "unsupported"
            replacement = "mixed checkpoint timing at nitride N=200 (49,1,1,1,1)"
            rationale = "The corrected five-seed vector invalidates inert-through-N=200 and onset-at-N=500 claims."
    elif term in {"domain_shift", "distribution_shift"}:
        action = "KEEP_WITH_QUALIFICATION" if "consistent" in text or context == "evidence_record" else "REWRITE_INTERPRETATION"
        claim_class = "correlational"
        replacement = "domain-shift-consistent evidence / chemical-family shift"
        rationale = "Family labels, performance gaps, and embedding separation do not independently prove domain shift."
        caveat, location = "Frame as study motivation or qualified evidence, not a proven distribution diagnosis.", "Title/Abstract/Discussion"
    elif term in {"transfer_benefit", "transfer_gain"}:
        action, claim_class = "KEEP_WITH_QUALIFICATION", "protocol-specific"
        evidence = "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv"
        selector = "number_id=CN_TRANSFER_BENEFIT_*"
        rationale = "Transfer-benefit and zero-shot-gain estimands must remain distinct."
        caveat = "Transfer benefit is scratch mean MAE minus fine-tuned mean MAE and exists only at N=50/500."
        location = "Results"
    elif term == "significance_language":
        action, claim_class = "REWRITE_INTERPRETATION", "protocol-specific"
        replacement = "the difference interval lies above zero / the ratio interval lies above one"
        evidence = "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_bootstrap_summary.csv"
        selector = "scenario=inclusive_oxide_vs_nitride"
        rationale = "The recorded bootstrap protocol authorizes an interval relationship, not generic significance wording."
        caveat, location = "State bootstrap level, resampling unit, and orientation.", "Results"
    elif term in {"robustness_language", "generalization_language"}:
        action, claim_class = "KEEP_WITH_QUALIFICATION", "protocol-specific"
        replacement = "robust under the tested protocol / observed in the evaluated families and conditions"
        rationale = "Robustness and generalization must be bounded to tested protocols, families, seeds, and sample sizes."
        caveat, location = "protocol_2/3 are robustness evidence only; do not claim universal generalization.", "Results/Discussion"
    return {
        "occurrence_id": row["occurrence_id"], "action": action,
        "replacement_wording": replacement, "claim_classification": claim_class,
        "rationale": rationale, "authoritative_evidence_path": evidence,
        "selector": selector, "required_caveat": caveat,
        "intended_future_manuscript_location": location,
    }


def read_csv(root: Path, rel: str) -> list[dict[str, str]]:
    with (root / rel).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def numerical_reconciliation(root: Path) -> list[dict[str, Any]]:
    impact_rel = "results/derived_evidence/protocol_1_regeneration/manuscript_claim_impact.csv"
    impact = read_csv(root, impact_rel)
    require(len(impact) == 462, f"Expected 462 protocol_1 impact rows, got {len(impact)}")
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(impact, start=1):
        require(source.get("old_protocol_1_claim_status") == "STALE_PENDING_SECTION3_REWRITE",
                f"Impact row {index} is not quarantined")
        source_hash = sha256_bytes(json.dumps(source, sort_keys=True, ensure_ascii=False,
                                              separators=(",", ":")).encode())
        action = "WITHDRAW_UNSUPPORTED" if source.get("claim_status") == "unsupported" else (
            "REPLACE_NUMBER" if source.get("old_numerical_claim", "").strip() else "REWRITE_INTERPRETATION")
        rows.append({
            "claim_id": f"S1-IMPACT-{index:04d}", "source_impact_row": index,
            "source_impact_row_sha256": source_hash,
            "current_path": source.get("file", ""), "current_location": source.get("section_or_line", ""),
            "old_claim_or_value": source.get("old_numerical_claim") or source.get("existing_sentence_or_identifier", ""),
            "authority_source": impact_rel, "row_selector": f"data_row={index}",
            "corrected_value": source.get("corrected_numerical_value_or_interpretation", ""),
            "units": "mixed; use selected v3 authority", "orientation": "defined by corrected interpretation",
            "rounding_policy": "full precision in registry; protocol_1 display generally five decimals",
            "action": action, "future_manuscript_location": "Section 3 rewrite target",
        })
    conflicts = (
        ("AUTH-CONFLICT-CLM02", "claim_support_map_v3.csv:CLM_02", "nitride zero-shot 0.06955",
         "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_point_estimates.csv",
         "scenario=inclusive_oxide_vs_nitride", "0.0695420149628485 (display 0.06954)", "eV/atom", "REPLACE_NUMBER"),
        ("AUTH-CONFLICT-CLM04", "claim_to_number_source_map_v3.csv:CLM_04", "nitride inert through N=200; adaptation begins N=500",
         "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv",
         "family=nitride;N=200", "best epochs [49,1,1,1,1]; blanket claim withdrawn", "epoch", "WITHDRAW_UNSUPPORTED"),
        ("AUTH-CONFLICT-CLM08", "claim_support_map_v3.csv:CLM_08", "last_alignn_pool is the strongest view",
         "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv",
         "dataset=fixed_test_set;embedding_source=last_alignn_pool;scenario=inclusive_baseline_recomputed",
         "metric-specific primary view; Davies-Bouldin does not support blanket strongest wording", "dimensionless", "REWRITE_INTERPRETATION"),
        ("AUTH-CONFLICT-CLM09", "claim_to_number_source_map_v3.csv:CLM_09", "distance-error rho/q/hard-easy statistics from drifted metadata predictions",
         "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv",
         "claim_id=C23C-*", "not reauthorized; recompute with canonical CSV errors before reuse", "not authorized", "DO_NOT_REUSE"),
        ("AUTH-CONFLICT-CLM11", "claim_support_map_v3.csv:CLM_11", "pre_head and last_gcn_pool are numerically identical",
         "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_report.md",
         "embedding duplicate guardrail", "raw arrays identical; stochastic derived metrics may differ", "dimensionless", "REWRITE_INTERPRETATION"),
    )
    for cid, location, old, authority, selector, corrected, units, action in conflicts:
        rows.append({
            "claim_id": cid, "source_impact_row": "", "source_impact_row_sha256": "",
            "current_path": "results/derived_evidence/final_paper_factory/00_source_of_truth/", "current_location": location,
            "old_claim_or_value": old, "authority_source": authority, "row_selector": selector,
            "corrected_value": corrected, "units": units,
            "orientation": "as stated; difference=nitride-minus-oxide; ratio=nitride-over-oxide",
            "rounding_policy": "retain full precision; five-decimal paper display where applicable",
            "action": action, "future_manuscript_location": "Section 3 source-of-truth reconciliation",
        })
    return rows


def future_registry(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    canonical_rel = "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv"
    for source in read_csv(root, canonical_rel):
        number_id = source["number_id"]
        upper = number_id.upper()
        if "EA_KNN5" in upper or any(token in upper for token in ("SPEARMAN", "HARD_EASY")):
            continue
        classification = "correlational" if "EA_" in upper else "protocol-specific"
        caveat = source.get("ambiguity_note") or "Bounded to the frozen dataset, checkpoint, and recorded protocol."
        if "EA_" in upper:
            caveat += " Correlational only; no causal mechanism or independent-confirmation claim."
        rows.append({
            "stable_claim_id": number_id,
            "bounded_future_claim": source.get("interpretation_label") or f"Report {source.get('metric_name')} for the selected condition.",
            "classification": classification, "numerical_authority": canonical_rel,
            "selector": f"number_id={number_id}", "value_or_qualitative_result": source.get("value", ""),
            "units": source.get("unit") or "dimensionless",
            "orientation": "difference=nitride-minus-oxide; ratio=nitride-over-oxide; otherwise not applicable",
            "rounding_policy": source.get("display_precision") or "retain full precision; manuscript display per metric convention",
            "uncertainty_convention": "sample SD ddof=1 for five-seed aggregates; otherwise as selected authority states",
            "required_caveat": caveat,
            "manuscript_location": "Main text" if source.get("paper_visibility") == "main_text" else "Appendix",
        })
    c_rel = "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv"
    for source in read_csv(root, c_rel):
        value = source.get("point_estimate") or source.get("materiality_outcome") or "qualitative result"
        ci = ""
        if source.get("ci_lower") and source.get("ci_upper"):
            ci = f"; 95% CI [{source['ci_lower']}, {source['ci_upper']}]"
        metric = source.get("metric_or_estimand", "")
        orientation = "nitride-minus-oxide" if "difference" in metric else (
            "nitride-over-oxide" if "ratio" in metric else "not applicable")
        rows.append({
            "stable_claim_id": source["claim_id"], "bounded_future_claim": source["bounded_claim_text"],
            "classification": source["claim_classification"], "numerical_authority": source["source_path"],
            "selector": source["row_selector"], "value_or_qualitative_result": value + ci,
            "units": source.get("units") or "categorical", "orientation": orientation,
            "rounding_policy": "retain full precision in registry; round transparently in prose",
            "uncertainty_convention": "structure-level nonparametric percentile bootstrap under Section 2.3C protocol" if source.get("confidence_level") else "no interval prespecified",
            "required_caveat": source["required_caveat"],
            "manuscript_location": source["intended_future_manuscript_location"],
        })
    manual = (
        ("PROV-CHECKPOINT-OXIDE-OVERLAP", "Oxide fixed-test exact overlap with unique official checkpoint-training JIDs is 0/1484.",
         "protocol-specific", "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv", "family=oxide", "0/1484", "count", "Exact JID non-overlap does not establish statistical ID/OOD status.", "Methods/Results"),
        ("PROV-CHECKPOINT-NITRIDE-OVERLAP", "Nitride fixed-test exact overlap with unique official checkpoint-training JIDs is 0/242.",
         "protocol-specific", "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv", "family=nitride", "0/242", "count", "Exact JID non-overlap does not establish statistical ID/OOD status.", "Methods/Results"),
        ("DATA-SOURCE-DEDUP", "The source has 55,723 entries and 55,712 unique catalog JIDs after removing 11 excess rows across five duplicated JIDs.",
         "procedural", "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_dataset_integrity.json", "/", "55723;55712;11;5", "count", "The complete original structural payload is not locally tracked for replay.", "Methods"),
        ("DATA-GLOBAL-SPLIT", "The governed downstream split assigns 55,709 unique JIDs with pairwise-disjoint train, validation, and test sets.",
         "procedural", "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_split_integrity.json", "/", "55709", "count", "Two train/test conflicts and one absent split-source JID remain unassigned.", "Methods"),
        ("DATA-FAMILY-DEFINITION", "Oxide is O-bearing including O+N; nitride is N-bearing without O, an intentional asymmetric definition retaining 499 oxynitrides.",
         "procedural", "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json", "/", "499 oxynitrides", "count", "Do not call the inclusive oxide arm pure oxide.", "Methods/Limitations"),
        ("DATA-CANONICAL-AUDIT", "All 120 canonical dataset roots and 240 canonical run split files passed the validated integrity audit.",
         "procedural", "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json", "/verdict", "120/120;240/240", "count", "Bounded to the committed canonical roots and run splits.", "Methods/Appendix"),
        ("protocol_1-NITRIDE-N200-EPOCHS", "Nitride N=200 selected best epochs [49,1,1,1,1] across the five seeds.",
         "protocol-specific", "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv", "family=nitride;N=200", "[49,1,1,1,1]", "epoch", "Epoch 1 is an end-of-epoch checkpoint and does not prove no update or byte identity.", "Results"),
        ("protocol_1-ALL-FT-WORSE-ZS", "Fine-tuning is worse than zero-shot in all 12 canonical protocol_1 family-by-N conditions.",
         "robust", "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv", "all_rows", "12/12 negative zero_shot_minus_finetune_mean", "condition", "Five-seed arithmetic means; sample SD uses ddof=1.", "Results"),
        ("EMBED-DISTANCE-ERROR-QUARANTINE", "Distance-error correlation statistics from drifted embedding metadata are not authorized for reuse pending canonical-error recomputation.",
         "procedural", "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_report.md", "prediction_drift_guardrail", "DO_NOT_REUSE", "categorical", "NPZ embeddings remain embedding authority; canonical CSV predictions remain error authority.", "Methods/Limitations"),
    )
    for cid, claim, classification, authority, selector, value, units, caveat, location in manual:
        rows.append({
            "stable_claim_id": cid, "bounded_future_claim": claim, "classification": classification,
            "numerical_authority": authority, "selector": selector,
            "value_or_qualitative_result": value, "units": units,
            "orientation": "difference=nitride-minus-oxide; ratio=nitride-over-oxide; otherwise not applicable",
            "rounding_policy": "retain exact counts/full precision; five decimals for protocol_1 MAE display",
            "uncertainty_convention": "as selected authority states; seed SD and structure bootstrap are distinct",
            "required_caveat": caveat, "manuscript_location": location,
        })
    rows.sort(key=lambda row: row["stable_claim_id"])
    require(len({r["stable_claim_id"] for r in rows}) == len(rows), "Duplicate future claim IDs")
    return rows


def rewrite_matrix(occurrences: list[dict[str, Any]], dispositions: list[dict[str, Any]],
                   numerical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    disp = {row["occurrence_id"]: row for row in dispositions}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for occurrence in occurrences:
        d = disp[occurrence["occurrence_id"]]
        if d["action"] in {"KEEP_SUPPORTED", "KEEP_LITERATURE_CONTEXT", "KEEP_HISTORICAL_RECORD", "NONCLAIM_METADATA"}:
            continue
        key = (occurrence["path"], f"line {occurrence['line_number']}")
        priority = "critical" if d["action"] in {"WITHDRAW_UNSUPPORTED", "DO_NOT_REUSE", "REPLACE_NUMBER"} else "high"
        rows[key] = {
            "file": key[0], "section_or_line_anchor": key[1],
            "currentness": currentness(key[0]), "priority": priority,
            "required_action": d["action"], "dependency": d["required_caveat"],
            "replacement_source": d["authoritative_evidence_path"],
            "section3_disposition": d["replacement_wording"] or d["rationale"],
        }
    for claim in numerical[:462]:
        key = (claim["current_path"], claim["current_location"] or f"impact row {claim['source_impact_row']}")
        rows[key] = {
            "file": key[0], "section_or_line_anchor": key[1], "currentness": currentness(key[0]),
            "priority": "critical", "required_action": claim["action"],
            "dependency": "Validated five-seed protocol_1 v3 authority",
            "replacement_source": claim["authority_source"],
            "section3_disposition": claim["corrected_value"],
        }
    return [rows[key] for key in sorted(rows)]


def aggregate_digest(mapping: dict[str, str]) -> str:
    return sha256_bytes(b"".join(
        rel.encode() + b"\0" + digest.encode() + b"\n" for rel, digest in sorted(mapping.items())))


def governance_markdown() -> str:
    return """# Section 2.3D governance reconciliation

## Decision

The original `input_manifest.md`, `source_policy.md`, and `run_session.json` are hash-pinned historical artifacts. They were verified before and after this phase and were not modified.

- `input_manifest.md` requires a versioned successor because its protocol_1 hashes and readiness state predate the corrected rerun.
- `source_policy.md` requires a versioned successor because its protocol_1 quarantine and manuscript-blocking gate are now satisfied.
- `run_session.json` records the 36-run Colab session, is referenced by permanent evidence, and contains no detected secret. The subsequent controlled checkpoint must decide explicitly whether it is permanent provenance or a recovery record.
- The Section 08 prohibition remains permanent.
- Raw results continue to override prose for numerical claims.
- protocol_1 v3 is the canonical primary numerical package.
- protocol_2 and 3 remain robustness and protocol-sensitivity evidence only.

No successor governance file was promoted in Section 2.3D. The next controlled checkpoint must version successors rather than silently edit these hash-pinned originals.
"""


def section3_spec() -> str:
    return """# Section 3 rewrite specification

## Binding authority order

Use raw or independently recomputed results first, validated Sections 2.2/2.3 second, v3 source-of-truth and corrected handoff packets third, and existing manuscripts only as audit targets. Never recover a desired sentence from an older draft.

## Terminology contract

- Replace project-level ID/OOD, OOD-penalty, source-domain, and target-domain conclusions with *oxide comparator*, *nitride target*, *chemical-family performance gap*, or *chemical-family MAE difference/ratio*.
- Exact checkpoint-training overlap is 0/1484 oxide and 0/242 nitride JIDs; this does not establish statistical distribution status.
- Use *domain-shift-consistent evidence* or *chemical-family shift* only with an explicit qualification. Family labels, performance gaps, and embedding separation do not prove domain shift.
- Describe the checkpoint as a JARVIS-DFT pretrained ALIGNN checkpoint, not as pretrained on oxides.

## protocol_1 contract

- Replace every one of the 462 quarantined entries using the reconciliation table; no old number is reusable.
- Report arithmetic five-seed means and sample SD (`ddof=1`), retaining visible seed-level points in figures.
- Fine-tuning is worse than zero-shot in all 12 canonical conditions.
- Nitride N=200 best epochs are `[49,1,1,1,1]`; withdraw inert-through-N=200 and adaptation-at-N=500 wording.
- `best_epoch=1` is the end-of-epoch-1 checkpoint, not proof of byte identity or absence of updates.
- Scratch and transfer-benefit claims are limited to N=50 and N=500. Transfer benefit is scratch mean MAE minus fine-tuned mean MAE.
- protocol_2/3 provide robustness evidence only.

## Dataset contract

State 55,723 source entries, 55,712 unique catalog JIDs, 11 excess rows across five duplicate JIDs, and 55,709 assigned JIDs. Two train/test conflicts and one split-source-absent JID remain unassigned. Oxide is O-bearing including O+N (499 oxynitrides); nitride is N-bearing without O. This asymmetry is intentional. All 120 roots and 240 run splits passed. Do not claim local replay of the complete original structural payload.

## Zero-shot and sensitivity contract

Difference is nitride minus oxide; ratio is nitride divided by oxide. Intervals are structure-level nonparametric bootstrap percentile intervals under the recorded protocol. State interval relationships rather than generic statistical significance. Do not construct sensitivity intervals by subtracting endpoints. Inclusive analysis is primary; pure-oxide filtering belongs in appendix and limitations. `INTERPRETATION_STABLE` does not mean zero change or removal of every definition limitation.

## Embedding contract

Embedding evidence is correlational and protocol-specific. Use *associated with*, *consistent with*, or *supports a representation-level interpretation*. Raw `pre_head` and `last_gcn_pool` arrays are duplicates, not independent confirmations; stochastic derived metrics can differ. Canonical prediction CSVs remain error authority. Distance-error rho/q/hard-easy claims from drifted metadata are `DO_NOT_REUSE` until recomputed against canonical errors.

## Assembly gate

Every numerical manuscript sentence must map to one registry row with an exact path, selector, units, orientation, rounding rule, uncertainty convention, caveat, and intended location. Apply the rewrite matrix before copying any text into the document template.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    root = args.repo.resolve()
    require(root == EXPECTED_ROOT, f"Expected {EXPECTED_ROOT}, got {root}")
    out = root / OUT_REL
    require(not out.exists(), f"Section 2.3D output directory already exists: {out}")

    preflight, baseline = verify_baseline(root)
    preflight["section08"] = verify_section08(root)
    preflight["prerequisite_checks"] = verify_gates(root)
    preflight["historical_governance_hashes_before"] = verify_governance(root)
    preflight["run_session_secret_scan"] = governance_secret_scan(root)
    preflight["remote_operations_performed"] = False
    preflight["excluded_paths_opened_or_hashed"] = False

    paths = tracked_corpus(root)
    corpus_rows, before_hashes = corpus_inventory(root, paths)
    authority_rows = authority_inventory(root)
    occurrences = scan_corpus(root, paths)
    dispositions = [disposition(row) for row in occurrences]
    require(len(dispositions) == len(occurrences) and
            {r["occurrence_id"] for r in dispositions} == {r["occurrence_id"] for r in occurrences},
            "Occurrence/disposition coverage is not one-to-one")
    require(all(row["action"] in ACTIONS and row["claim_classification"] in CLAIM_CLASSES
                for row in dispositions), "Unclassified disposition")
    numerical = numerical_reconciliation(root)
    registry = future_registry(root)
    rewrites = rewrite_matrix(occurrences, dispositions, numerical)

    after_hashes = {rel: sha256(root / rel) for rel in paths}
    require(before_hashes == after_hashes, "Selected tracked corpus changed during audit")
    governance_after = verify_governance(root)
    require(preflight["historical_governance_hashes_before"] == governance_after,
            "Historical governance changed during audit")
    preflight["historical_governance_hashes_after_producer"] = governance_after
    preflight["selected_tracked_input_count"] = len(paths)
    preflight["selected_tracked_input_aggregate_sha256_before"] = aggregate_digest(before_hashes)
    preflight["selected_tracked_input_aggregate_sha256_after"] = aggregate_digest(after_hashes)
    preflight["original_546_status_preserved_during_producer"] = (
        [x for x in parse_status(root) if status_category(x["path"]) != "section23d_new"] == baseline)
    require(preflight["original_546_status_preserved_during_producer"], "Original status baseline changed")

    prose_count = sum(Path(p).suffix.lower() in PROSE_SUFFIXES for p in paths)
    structured_count = sum(Path(p).suffix.lower() in {".csv", ".json"} for p in paths)
    scope = {
        "schema_version": "1.0", "section": "2.3D",
        "expected_head": EXPECTED_HEAD, "explicit_tracked_roots": list(ROOTS),
        "forbidden_path_pattern": "**/archived_submission_materials/**",
        "protected_exclusions": [NESTED_REL + "/", "**/finetune_last2_reproduction_rerun/**",
            "configs/protocol_1/finetune_reproduction_rerun/",
            "results/summaries/protocol_1/finetune/reproduction_rerun/"],
        "tracked_inventory_count": len(paths), "broad_prose_extensions": sorted(PROSE_SUFFIXES),
        "broad_prose_file_count": prose_count, "structured_file_count": structured_count,
        "structured_scan_rules": {k: v if isinstance(v, str) else list(v)
                                  for k, v in sorted(STRUCTURED_SCAN_RULES.items())},
        "terminology_regex_version": "section23d-unicode-longest-first-v1",
        "terminology_patterns": [{"normalized_term": n, "pattern": p} for n, p in PATTERN_SPECS],
        "matching_policy": "case-insensitive, Unicode-horizontal-separator aware, longest-first, non-overlapping, line bounded",
        "action_taxonomy": list(ACTIONS), "claim_classification_taxonomy": list(CLAIM_CLASSES),
        "selector_grammar": {"csv": "semicolon-separated field=value predicates; data_row=N is one-based",
                             "json": "RFC 6901 JSON Pointer", "special": ["all_rows", "approved_wording", "prediction_drift_guardrail"]},
        "authority_precedence": ["raw or independently recomputed numerical evidence",
            "validated Section 2.2 and 2.3 packages", "v3 source-of-truth and corrected handoff packets",
            "results/derived_evidence/manuscripts as audit targets", "historical files as non-authoritative prose"],
        "producer_outputs": list(PRODUCER_OUTPUTS), "validator_final_outputs": list(FINAL_OUTPUTS),
        "generated_manifest_policy": "16 entries: 14 evidence files excluding itself/checksum plus two scripts",
        "checksum_policy": "17 entries: all 15 other evidence files plus two scripts; excludes itself",
        "determinism_note": "Rows are sorted and content is reproducible for this frozen workspace snapshot; input mtimes are observational.",
    }
    coverage = {
        "tracked_allowlisted_files": len(paths), "broad_prose_files": prose_count,
        "structured_inventory_files": structured_count,
        "structured_claim_files_scanned": len(STRUCTURED_SCAN_RULES),
        "prose_files_with_occurrences": len({r["path"] for r in occurrences if Path(r["path"]).suffix.lower() in PROSE_SUFFIXES}),
        "terminology_occurrences": len(occurrences),
        "terminology_occurrences_by_term": dict(sorted(Counter(r["normalized_term"] for r in occurrences).items())),
        "terminology_context_counts": dict(sorted(Counter(r["context_classification"] for r in occurrences).items())),
        "disposition_counts": dict(sorted(Counter(r["action"] for r in dispositions).items())),
        "claim_classification_counts": dict(sorted(Counter(r["claim_classification"] for r in dispositions).items())),
        "protocol_1_impact_rows_expected": 462, "protocol_1_impact_rows_covered": 462,
        "stale_protocol_1_rows_authorized_for_reuse": 0,
        "authority_conflicts_resolved": 5, "numerical_reconciliation_rows": len(numerical),
        "future_manuscript_registry_rows": len(registry), "rewrite_action_rows": len(rewrites),
        "producer_validator_agreement": "pending independent validator",
    }

    out.mkdir(parents=True, exist_ok=False)
    json_dump(out / "audit_scope_manifest.json", scope)
    csv_dump(out / "authoritative_input_inventory.csv",
             ["category", "repository_relative_path", "git_status", "size_bytes",
              "modification_time_utc", "sha256", "intended_role", "authority_level",
              "current_historical_status"], authority_rows)
    csv_dump(out / "claim_bearing_file_inventory.csv",
             ["repository_relative_path", "sha256", "corpus_tier", "file_classification",
              "version_currentness", "claim_bearing_status", "audit_inclusion_rationale"], corpus_rows)
    csv_dump(out / "terminology_occurrence_inventory.csv",
             ["occurrence_id", "path", "line_number", "column_start", "column_end",
              "matched_term", "normalized_term", "bounded_context", "excerpt_sha256",
              "file_classification", "context_classification"], occurrences)
    csv_dump(out / "terminology_disposition.csv",
             ["occurrence_id", "action", "replacement_wording", "claim_classification",
              "rationale", "authoritative_evidence_path", "selector", "required_caveat",
              "intended_future_manuscript_location"], dispositions)
    csv_dump(out / "numerical_claim_reconciliation.csv",
             ["claim_id", "source_impact_row", "source_impact_row_sha256", "current_path",
              "current_location", "old_claim_or_value", "authority_source", "row_selector",
              "corrected_value", "units", "orientation", "rounding_policy", "action",
              "future_manuscript_location"], numerical)
    csv_dump(out / "rewrite_action_matrix.csv",
             ["file", "section_or_line_anchor", "currentness", "priority", "required_action",
              "dependency", "replacement_source", "section3_disposition"], rewrites)
    csv_dump(out / "future_manuscript_claim_registry.csv",
             ["stable_claim_id", "bounded_future_claim", "classification", "numerical_authority",
              "selector", "value_or_qualitative_result", "units", "orientation", "rounding_policy",
              "uncertainty_convention", "required_caveat", "manuscript_location"], registry)
    text_dump(out / "governance_reconciliation.md", governance_markdown())
    text_dump(out / "section3_rewrite_specification.md", section3_spec())
    json_dump(out / "coverage_summary.json", coverage)
    json_dump(out / "repository_preflight.json", preflight)
    print(json.dumps({"producer_outputs": len(PRODUCER_OUTPUTS),
                      "tracked_corpus": len(paths), "occurrences": len(occurrences),
                      "protocol_1_impact_rows": 462, "ready_for_independent_validation": True}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except AuditFailure as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        raise SystemExit(1)
