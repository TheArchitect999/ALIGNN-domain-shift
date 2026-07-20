#!/usr/bin/env python3
"""Independently validate and finalize Section 2.3D evidence.

The validator does not import the producer.  ``--finalize`` independently
rebuilds the corpus and terminology scan, validates outputs 1--12, and writes
outputs 13--16 exactly once.  ``--verify-only`` repeats all checks, verifies
the manifest/checksum exact sets, and performs no writes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

EXPECTED_ROOT = Path(".")
EXPECTED_HEAD = "9e2c640bec56fd67d77989cc45845d6d241bf28d"
EXPECTED_BRANCH = "main"
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

PRODUCER_OUTPUTS = {
    "audit_scope_manifest.json", "authoritative_input_inventory.csv",
    "claim_bearing_file_inventory.csv", "terminology_occurrence_inventory.csv",
    "terminology_disposition.csv", "numerical_claim_reconciliation.csv",
    "rewrite_action_matrix.csv", "future_manuscript_claim_registry.csv",
    "governance_reconciliation.md", "section3_rewrite_specification.md",
    "coverage_summary.json", "repository_preflight.json",
}
FINAL_OUTPUTS = {
    "generated_output_manifest.json", "section2_3D_validation.json",
    "section2_3D_report.md", "section2_3D_evidence.sha256",
}
ALL_OUTPUTS = PRODUCER_OUTPUTS | FINAL_OUTPUTS
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
    "results/derived_evidence/input_manifest.md": "735044376ffe562f61fc04343a7cbe1c294f7cf83b8add7e6d65bdc595af6494",
    "results/derived_evidence/source_policy.md": "0fc51ef87db6cb8c3615cb90cc8110ee26f89491208a296373b8ad63a8a3bacb",
    "results/derived_evidence/run_session.json": "647ab1653e2fb8d9e7bbc8843d0f034d3614ec17ee1e7f4b2cca24f6de40cdcb",
}
GATE_MANIFESTS = (
    ("results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256", "repo", 19),
    ("results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256", "parent", 22),
    ("results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_evidence.sha256", "repo", 31),
    ("results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_evidence.sha256", "repo", 24),
    ("results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_evidence.sha256", "repo", 20),
)

AUTHORITY_PATHS = {
    *GOVERNANCE_HASHES,
    *(spec[0] for spec in GATE_MANIFESTS),
    "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv",
    "results/derived_evidence/protocol_1_regeneration/manuscript_claim_impact.csv",
    "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.md",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_support_map_v3.csv",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/claim_to_number_source_map_v3.csv",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.csv",
    "results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.md",
    "results/derived_evidence/final_paper_factory/03_section_inputs/combined_paper_results_III_and_IV_protocol_1_correction_packet_v1.md",
    "results/derived_evidence/final_paper_factory/03_section_inputs/joint_comparison_packet_v2_protocol_1_corrected.md",
    "results/derived_evidence/final_paper_factory/03_section_inputs/oxide_results_packet_v2_protocol_1_corrected.md",
    "results/derived_evidence/final_paper_factory/03_section_inputs/nitride_results_packet_v2_protocol_1_corrected.md",
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_membership_terminology_decision.md",
    "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/dataset_integrity_decision.md",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_dataset_integrity.json",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/global_split_integrity.json",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/family_definition_audit.json",
    "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_bootstrap_summary.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_point_estimates.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_deltas.csv",
    "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/materiality_claim_impact.md",
}
NUMERICAL_AUTHORITIES = {
    path for path in AUTHORITY_PATHS if path.endswith(("aggregate_recomputation.csv", "canonical_numbers_v3.csv",
        "checkpoint_test_overlap.csv", "global_dataset_integrity.json", "global_split_integrity.json",
        "family_definition_audit.json", "oxynitride_definition_summary.json", "zero_shot_bootstrap_summary.csv",
        "zero_shot_point_estimates.csv", "embedding_sensitivity_metrics.csv", "embedding_sensitivity_deltas.csv"))
}

ACTIONS = {
    "KEEP_SUPPORTED", "KEEP_WITH_QUALIFICATION", "KEEP_LITERATURE_CONTEXT",
    "KEEP_HISTORICAL_RECORD", "REPLACE_TERMINOLOGY", "REPLACE_NUMBER",
    "REWRITE_INTERPRETATION", "WITHDRAW_UNSUPPORTED", "DO_NOT_REUSE",
    "NONCLAIM_METADATA",
}
CLAIM_CLASSES = {
    "robust", "protocol-specific", "correlational", "unsupported",
    "procedural", "literature-context", "historical-only", "nonclaim",
}

H = r"[ \t\u00a0\-\u2010-\u2015]+"
PATTERN_SPECS = (
    ("ood_penalty", rf"OOD{H}penalt(?:y|ies)"),
    ("out_of_distribution", rf"out{H}of{H}distribution"),
    ("in_distribution", rf"in{H}distribution"),
    ("oxide_pretrained", rf"oxide{H}pre[-\u2010-\u2015]?train(?:ed|ing)"),
    ("oxide_pretrained", rf"pre[-\u2010-\u2015]?train(?:ed|ing){H}(?:(?:primarily|exclusively){H})?(?:on|using){H}(?:the{H})?oxides?"),
    ("source_domain", rf"source{H}domain"), ("target_domain", rf"target{H}domain"),
    ("domain_gap", rf"domain{H}gap"), ("distribution_shift", rf"distribution{H}shift"),
    ("domain_shift", rf"domain{H}shift"), ("transfer_benefit", rf"transfer{H}benefit"),
    ("transfer_gain", rf"transfer{H}gain"), ("adaptation_onset", rf"adaptation{H}onset"),
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
COMPILED_PATTERNS = tuple((name, re.compile(rf"(?<![\w])(?:{pattern})(?![\w])", re.I))
                          for name, pattern in PATTERN_SPECS)


class ValidationFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def git(root: Path, *args: str, binary: bool = False) -> str | bytes:
    proc = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    require(proc.returncode == 0, f"git {' '.join(args)} failed")
    return proc.stdout if binary else proc.stdout.decode("utf-8", errors="surrogateescape").strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_rel(value: str) -> str:
    rel = Path(value).as_posix()
    require(not Path(rel).is_absolute() and ".." not in Path(rel).parts, f"Unsafe path: {rel}")
    require("archived_submission_materials" not in rel, f"Forbidden Section 08 path: {rel}")
    require(not (rel == NESTED_REL or rel.startswith(NESTED_REL + "/")), f"Nested path: {rel}")
    require("finetune_last2_reproduction_rerun" not in rel, f"Staging path: {rel}")
    require("finetune_reproduction_rerun" not in rel, f"Config recovery path: {rel}")
    require("Summaries/finetune/reproduction_rerun" not in rel, f"Temporary summary path: {rel}")
    return rel


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json_once(path: Path, payload: Any) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False,
                               allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def write_text_once(path: Path, value: str) -> None:
    require(not path.exists(), f"Refusing to overwrite: {path}")
    path.write_text(value.rstrip() + "\n", encoding="utf-8", newline="\n")


def parse_status(root: Path) -> list[dict[str, str]]:
    raw = subprocess.run(["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
                         cwd=root, capture_output=True, check=True).stdout
    parts = [x for x in raw.split(b"\0") if x]
    rows: list[dict[str, str]] = []
    i = 0
    while i < len(parts):
        rec = parts[i].decode("utf-8", errors="surrogateescape")
        rows.append({"status": rec[:2], "path": rec[3:]})
        if rec[0] in "RC":
            i += 1
        i += 1
    return sorted(rows, key=lambda x: (x["path"], x["status"]))


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


def verify_repo(root: Path, phase: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    require(str(git(root, "rev-parse", "HEAD")) == EXPECTED_HEAD, "HEAD mismatch")
    require(str(git(root, "branch", "--show-current")) == EXPECTED_BRANCH, "Branch mismatch")
    require(str(git(root, "show", "-s", "--format=%s", "HEAD")) == EXPECTED_SUBJECT, "Subject mismatch")
    line = str(git(root, "status", "--short", "--branch", "--untracked-files=no")).splitlines()[0]
    require(line.removeprefix("## ") == EXPECTED_UPSTREAM, "Cached upstream relationship mismatch")
    require(subprocess.run(["git", "diff", "--quiet"], cwd=root).returncode == 0, "Tracked worktree change")
    require(subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode == 0, "Staged change")
    require(not (root / SECTION08_REL).exists() and not (root / SECTION08_REL).is_symlink(), "Section 08 exists")
    require(not str(git(root, "ls-files", "--", SECTION08_REL.as_posix())), "Section 08 tracked")
    status = parse_status(root)
    baseline = [x for x in status if status_category(x["path"]) != "section23d_new"]
    new = [x for x in status if status_category(x["path"]) == "section23d_new"]
    expected_counts = {"corrected_rerun_staging_artifact": 504,
        "rerun_configuration_recovery_control": 36, "temporary_rerun_summary": 2,
        "historical_governance_session_file": 3, "nested_checkout_entry": 1}
    require(len(baseline) == 546 and dict(Counter(status_category(x["path"]) for x in baseline)) == expected_counts,
            "Original 546-entry baseline changed")
    require(all(x["status"] == "??" for x in baseline + new), "Non-untracked status entry present")
    require(not any(status_category(x["path"]) == "ambiguous" for x in status), "Ambiguous status entry")
    expected_new_names = {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()} | {
        (OUT_REL / name).as_posix() for name in (PRODUCER_OUTPUTS if phase == "finalize" else ALL_OUTPUTS)}
    require({x["path"] for x in new} == expected_new_names,
            f"Unexpected Section 2.3D path set in {phase}: {len(new)}")
    expected_new_count = 14 if phase == "finalize" else 18
    require(len(new) == expected_new_count, f"Expected {expected_new_count} new paths, got {len(new)}")
    preflight = json.loads((root / OUT_REL / "repository_preflight.json").read_text(encoding="utf-8"))
    require(preflight["filtered_original_status_entries"] == baseline,
            "Original status list differs from producer preflight")
    return baseline, new


def verify_checksum_manifest(root: Path, rel: str, mode: str, expected_count: int) -> None:
    manifest = root / rel
    lines = [x for x in manifest.read_text(encoding="utf-8").splitlines() if x.strip()]
    require(len(lines) == expected_count, f"Checksum count mismatch: {rel}")
    base = manifest.parent if mode == "parent" else root
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum: {rel}")
        wanted, named = match.groups()
        path = (base / named).resolve()
        require(path.is_relative_to(root.resolve()), f"Checksum escape: {named}")
        safe_rel(path.relative_to(root.resolve()).as_posix())
        require(path.is_file() and not path.is_symlink() and sha256(path) == wanted,
                f"Checksum failure: {named}")


def verify_prerequisites(root: Path) -> None:
    for spec in GATE_MANIFESTS:
        verify_checksum_manifest(root, *spec)
    report = (root / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md").read_text(encoding="utf-8")
    require("protocol_1_REGENERATION_VALIDATED" in report, "Regeneration verdict missing")
    verdicts = {
        "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/section2_3A_validation.json": "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP",
        "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json": "SECTION23B_DATASET_INTEGRITY_VALIDATED",
        "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/section2_3C_validation.json": "SECTION23C_OXYNITRIDE_BOOTSTRAP_VALIDATED",
    }
    for rel, verdict in verdicts.items():
        require(json.loads((root / rel).read_text(encoding="utf-8")).get("verdict") == verdict,
                f"Verdict mismatch: {rel}")
    for rel, wanted in GOVERNANCE_HASHES.items():
        require(sha256(root / rel) == wanted, f"Governance hash mismatch: {rel}")


def tracked_corpus(root: Path) -> list[str]:
    raw = git(root, "ls-files", "-z", "--", *ROOTS, binary=True)
    assert isinstance(raw, bytes)
    paths = sorted(x.decode("utf-8") for x in raw.split(b"\0") if x)
    require(len(paths) == 1553, f"Corpus size mismatch: {len(paths)}")
    for rel in paths:
        safe_rel(rel)
        mode = str(git(root, "ls-files", "-s", "--", rel)).split()[0]
        require(mode == "100644", f"Nonregular corpus path: {rel}")
    return paths


def currentness(rel: str) -> str:
    low = rel.lower()
    if any(x in low for x in ("_v1.", "_v2.", "polished_v2", "before_correction", "old_", "legacy")):
        return "historical"
    if rel in STRUCTURED_SCAN_RULES or "_v3" in low or "polished_v3" in low or "polished_v4" in low or "protocol_1_corrected" in low:
        return "current_or_current_audit_target"
    if "/04_drafts/" in rel or "/05_reviewed_drafts/" in rel or "/06_template_ready/" in rel:
        return "editorial_version_requires_rebuild"
    if "/evidence/" in rel:
        return "current_evidence_record"
    return "mixed_or_historical_audit_corpus"


def file_classification(rel: str) -> str:
    suffix, low = Path(rel).suffix.lower(), rel.lower()
    if suffix not in PROSE_SUFFIXES | {".csv", ".json"}:
        return "Non-claim metadata"
    if "literature" in low or "bibliograph" in low or "references" in low:
        return "Literature context"
    if rel in AUTHORITY_PATHS:
        return "Current numerical authority" if rel in NUMERICAL_AUTHORITIES else "Current procedural/governance authority"
    if any(x in low for x in ("polished_v3", "polished_v4", "template_ready_v3")):
        return "Editorial audit target"
    if "protocol_1_corrected" in low or "correction_packet" in low or "_v3" in low:
        return "Future manuscript input"
    if currentness(rel) == "historical":
        return "Historical-only"
    if "/evidence/" in rel or "/provenance/" in rel:
        return "Evidence record"
    return "Editorial audit target"


def context_class(rel: str, text: str, start: int, classification: str) -> str:
    low, before = text.lower(), text[:start]
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


def accepted_spans(text: str) -> list[tuple[int, int, str, str]]:
    candidates: list[tuple[int, int, int, str, str]] = []
    for priority, (term, pattern) in enumerate(COMPILED_PATTERNS):
        for match in pattern.finditer(text):
            candidates.append((match.start(), match.end(), priority, term, match.group(0)))
    candidates.sort(key=lambda x: (x[0], -(x[1] - x[0]), x[2], x[3]))
    accepted: list[tuple[int, int, str, str]] = []
    occupied = -1
    for start, end, _priority, term, matched in candidates:
        if start < occupied:
            continue
        accepted.append((start, end, term, matched))
        occupied = end
    return accepted


def bounded(text: str, start: int, end: int, width: int = 180) -> str:
    return re.sub(r"\s+", " ", text[max(0, start-width):min(len(text), end+width)]).strip()


def scan_unit(rel: str, line: int, label: str, text: str, excerpt: bytes,
              classification: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    excerpt_hash = sha256_bytes(excerpt)
    for start, end, term, matched in accepted_spans(text):
        seed = f"{rel}\0{line}\0{start}\0{end}\0{term}\0{excerpt_hash}\0{label}"
        rows.append({
            "occurrence_id": "T23D-" + sha256_bytes(seed.encode())[:24], "path": rel,
            "line_number": str(line), "column_start": str(start + 1), "column_end": str(end),
            "matched_term": matched, "normalized_term": term,
            "bounded_context": (f"[{label}] " if label else "") + bounded(text, start, end),
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


def independent_scan(root: Path, paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for rel in paths:
        suffix, classification = Path(rel).suffix.lower(), file_classification(rel)
        if suffix in PROSE_SUFFIXES:
            text = (root / rel).read_bytes().decode("utf-8")
            for number, line in enumerate(text.splitlines(), 1):
                rows.extend(scan_unit(rel, number, "", line, line.encode(), classification))
        elif rel in STRUCTURED_SCAN_RULES and suffix == ".csv":
            with (root / rel).open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                wanted = set(STRUCTURED_SCAN_RULES[rel])
                for number, source in enumerate(reader, 1):
                    for column in sorted(set(source) & wanted):
                        value = source.get(column) or ""
                        rows.extend(scan_unit(rel, number + 1, f"csv:{column}", value,
                                              value.encode(), classification))
        elif rel in STRUCTURED_SCAN_RULES and suffix == ".json":
            payload = json.loads((root / rel).read_text(encoding="utf-8"))
            for number, (pointer, value) in enumerate(walk_json_strings(payload), 1):
                rows.extend(scan_unit(rel, number, f"json-pointer:{pointer}", value,
                                      value.encode(), classification))
    rows.sort(key=lambda row: (row["path"], int(row["line_number"]), int(row["column_start"]),
                               row["normalized_term"], row["occurrence_id"]))
    require(len(rows) == len({r["occurrence_id"] for r in rows}), "Duplicate independent occurrence ID")
    return rows


def verify_corpus_and_occurrences(root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    paths = tracked_corpus(root)
    inventory = read_csv(root / OUT_REL / "claim_bearing_file_inventory.csv")
    require([r["repository_relative_path"] for r in inventory] == paths, "Corpus inventory path/order mismatch")
    for row in inventory:
        rel = row["repository_relative_path"]
        require(row["sha256"] == sha256(root / rel), f"Corpus hash mismatch: {rel}")
    expected = independent_scan(root, paths)
    actual = read_csv(root / OUT_REL / "terminology_occurrence_inventory.csv")
    require(actual == expected, f"Producer/validator terminology scan mismatch: {len(actual)} vs {len(expected)}")
    return inventory, actual


def verify_dispositions(root: Path, occurrences: list[dict[str, str]]) -> list[dict[str, str]]:
    rows = read_csv(root / OUT_REL / "terminology_disposition.csv")
    require(len(rows) == len(occurrences), "Disposition count mismatch")
    require(len({r["occurrence_id"] for r in rows}) == len(rows), "Duplicate disposition")
    require({r["occurrence_id"] for r in rows} == {r["occurrence_id"] for r in occurrences},
            "Occurrence/disposition join mismatch")
    for row in rows:
        require(row["action"] in ACTIONS and row["claim_classification"] in CLAIM_CLASSES,
                f"Invalid taxonomy: {row['occurrence_id']}")
        if row["action"] in {"REPLACE_TERMINOLOGY", "REWRITE_INTERPRETATION", "WITHDRAW_UNSUPPORTED"}:
            require(row["replacement_wording"] and row["rationale"] and row["required_caveat"],
                    f"Incomplete action: {row['occurrence_id']}")
        if row["claim_classification"] == "unsupported":
            require(row["action"] in {"WITHDRAW_UNSUPPORTED", "DO_NOT_REUSE"},
                    f"Unsupported claim not withdrawn/quarantined: {row['occurrence_id']}")
    by_id = {r["occurrence_id"]: r for r in occurrences}
    for row in rows:
        occ = by_id[row["occurrence_id"]]
        if occ["context_classification"] == "project_claim" and occ["normalized_term"] in {
                "ood", "ood_penalty", "in_distribution", "out_of_distribution"}:
            require(row["action"] == "WITHDRAW_UNSUPPORTED", "Project ID/OOD conclusion authorized")
        if occ["context_classification"] == "project_claim" and occ["normalized_term"] == "causal_language":
            require(row["action"] in {"REWRITE_INTERPRETATION", "WITHDRAW_UNSUPPORTED"},
                    "Causal project claim retained")
    return rows


def row_hash(row: dict[str, str]) -> str:
    return sha256_bytes(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode())


def verify_numerical(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / OUT_REL / "numerical_claim_reconciliation.csv")
    impact = read_csv(root / "results/derived_evidence/protocol_1_regeneration/manuscript_claim_impact.csv")
    require(len(impact) == 462, "Impact authority no longer has 462 rows")
    protocol_1 = [r for r in rows if r["claim_id"].startswith("S1-IMPACT-")]
    require(len(protocol_1) == 462, "Not all 462 protocol_1 impact rows reconciled")
    require({int(r["source_impact_row"]) for r in protocol_1} == set(range(1, 463)), "protocol_1 row coverage mismatch")
    for row in protocol_1:
        index = int(row["source_impact_row"])
        source = impact[index - 1]
        require(source["old_protocol_1_claim_status"] == "STALE_PENDING_SECTION3_REWRITE", "protocol_1 stale gate missing")
        require(row["source_impact_row_sha256"] == row_hash(source), "Impact row hash mismatch")
        require(row["corrected_value"] == source["corrected_numerical_value_or_interpretation"], "Impact correction mismatch")
        require(row["action"] in {"REPLACE_NUMBER", "REWRITE_INTERPRETATION", "WITHDRAW_UNSUPPORTED", "DO_NOT_REUSE"},
                "Stale protocol_1 value authorized")
    conflict_actions = {r["claim_id"]: r["action"] for r in rows if r["claim_id"].startswith("AUTH-CONFLICT-")}
    require(conflict_actions == {"AUTH-CONFLICT-CLM02": "REPLACE_NUMBER",
        "AUTH-CONFLICT-CLM04": "WITHDRAW_UNSUPPORTED", "AUTH-CONFLICT-CLM08": "REWRITE_INTERPRETATION",
        "AUTH-CONFLICT-CLM09": "DO_NOT_REUSE", "AUTH-CONFLICT-CLM11": "REWRITE_INTERPRETATION"},
        "Authority conflict resolutions missing")
    aggregate = read_csv(root / "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv")
    require(len(aggregate) == 12 and all(float(r["zero_shot_minus_finetune_mean_eV_per_atom"]) < 0 for r in aggregate),
            "All-12 fine-tuning/zero-shot invariant failed")
    n200 = [r for r in aggregate if r["family"] == "nitride" and r["N"] == "200"]
    require(len(n200) == 1 and n200[0]["raw_seed_best_epochs"].replace(" ", "") == "[49,1,1,1,1]",
            "Nitride N=200 checkpoint vector mismatch")
    overlap = read_csv(root / "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv")
    require({r["family"]: (r["fixed_test_count"], r["overlap_count"]) for r in overlap} ==
            {"oxide": ("1484", "0"), "nitride": ("242", "0")}, "Checkpoint overlap mismatch")
    b = json.loads((root / "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/section2_3B_validation.json").read_text())
    ir = b["independent_recomputation"]
    require((ir["source_records"], ir["catalog_unique"], ir["duplicate_excess"], ir["duplicate_jids"],
             ir["assigned_jids"], ir["oxynitrides"], ir["dataset_roots_passed"], ir["canonical_runs_passed"]) ==
            (55723, 55712, 11, 5, 55709, 499, 120, 240), "Dataset fact reconciliation mismatch")
    return rows


def match_csv_selector(path: Path, selector: str) -> int:
    predicates: dict[str, str] = {}
    for part in selector.split(";"):
        require("=" in part, f"Invalid CSV selector: {selector}")
        key, value = part.split("=", 1)
        predicates[key] = value
    rows = read_csv(path)
    return sum(all(row.get(key) == value for key, value in predicates.items()) for row in rows)


def resolve_json_pointer(payload: Any, pointer: str) -> Any:
    if pointer in {"", "/"}:
        return payload
    require(pointer.startswith("/"), f"Invalid JSON pointer: {pointer}")
    current = payload
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def verify_registry(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / OUT_REL / "future_manuscript_claim_registry.csv")
    require(len(rows) == len({r["stable_claim_id"] for r in rows}), "Duplicate registry claim ID")
    require(sum(r["stable_claim_id"].startswith("C23C-") for r in rows) == 30, "C23C registry coverage mismatch")
    require(not any("EA_KNN5" in r["stable_claim_id"] or "SPEARMAN" in r["stable_claim_id"] or
                    "HARD_EASY" in r["stable_claim_id"] for r in rows), "Quarantined distance-error claim authorized")
    special = {"all_rows", "interpretation_result", "prediction_drift_guardrail", "embedding duplicate guardrail"}
    for row in rows:
        for key in ("bounded_future_claim", "classification", "numerical_authority", "selector",
                    "value_or_qualitative_result", "units", "orientation", "rounding_policy",
                    "uncertainty_convention", "required_caveat", "manuscript_location"):
            require(row[key].strip(), f"Blank registry field {key}: {row['stable_claim_id']}")
        rel = safe_rel(row["numerical_authority"])
        path = root / rel
        require(path.is_file() and not path.is_symlink(), f"Registry authority missing: {rel}")
        selector = row["selector"]
        if path.suffix == ".csv":
            if selector == "all_rows":
                require(len(read_csv(path)) == 12, "all_rows selector does not target protocol_1 aggregate")
            elif "*" in selector:
                require(row["classification"] == "protocol-specific", "Wildcard selector on non-protocol claim")
            else:
                require(match_csv_selector(path, selector) == 1,
                        f"Registry selector not unique: {row['stable_claim_id']} {selector}")
        elif path.suffix == ".json":
            resolve_json_pointer(json.loads(path.read_text(encoding="utf-8")), selector)
        else:
            require(selector in special, f"Unauthorized prose selector: {selector}")
    canonical = {r["number_id"]: r for r in read_csv(root / "results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv")}
    for row in rows:
        if row["stable_claim_id"].startswith("CN_"):
            require(row["stable_claim_id"] in canonical and
                    row["value_or_qualitative_result"] == canonical[row["stable_claim_id"]]["value"],
                    f"Canonical registry value mismatch: {row['stable_claim_id']}")
    c_map = {r["claim_id"]: r for r in read_csv(root / "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/claim_evidence_map.csv")}
    require(set(c_map) <= {r["stable_claim_id"] for r in rows}, "C23C claim map not fully imported")
    return rows


def verify_authority_inventory(root: Path) -> list[dict[str, str]]:
    rows = read_csv(root / OUT_REL / "authoritative_input_inventory.csv")
    require({r["repository_relative_path"] for r in rows} == AUTHORITY_PATHS,
            "Authority inventory allowlist mismatch")
    for row in rows:
        rel = safe_rel(row["repository_relative_path"])
        require(row["sha256"] == sha256(root / rel), f"Authority hash mismatch: {rel}")
        require(row["authority_level"] and row["intended_role"], f"Authority metadata blank: {rel}")
    return rows


def verify_governance_docs(root: Path) -> None:
    governance = (root / OUT_REL / "governance_reconciliation.md").read_text(encoding="utf-8")
    for phrase in ("hash-pinned historical artifacts", "versioned successor", "Section 08 prohibition remains permanent",
                   "Raw results continue to override prose", "protocol_1 v3 is the canonical primary numerical package",
                   "protocol_2 and 3 remain robustness", "No successor governance file was promoted"):
        require(phrase in governance, f"Governance reconciliation omission: {phrase}")
    spec = (root / OUT_REL / "section3_rewrite_specification.md").read_text(encoding="utf-8")
    for phrase in ("462 quarantined", "[49,1,1,1,1]", "scratch mean MAE minus fine-tuned mean MAE",
                   "55,723 source entries", "499 oxynitrides", "nitride minus oxide",
                   "correlational and protocol-specific", "DO_NOT_REUSE"):
        require(phrase in spec, f"Section 3 specification omission: {phrase}")


def semantic_checks(root: Path, phase: str) -> dict[str, Any]:
    verify_prerequisites(root)
    baseline, new = verify_repo(root, phase)
    authority = verify_authority_inventory(root)
    inventory, occurrences = verify_corpus_and_occurrences(root)
    dispositions = verify_dispositions(root, occurrences)
    numerical = verify_numerical(root)
    registry = verify_registry(root)
    rewrites = read_csv(root / OUT_REL / "rewrite_action_matrix.csv")
    require(rewrites and all(r["required_action"] in ACTIONS for r in rewrites), "Rewrite matrix invalid")
    verify_governance_docs(root)
    coverage = json.loads((root / OUT_REL / "coverage_summary.json").read_text(encoding="utf-8"))
    require(coverage["tracked_allowlisted_files"] == len(inventory) == 1553, "Coverage corpus mismatch")
    require(coverage["terminology_occurrences"] == len(occurrences) == len(dispositions), "Coverage occurrence mismatch")
    require(coverage["protocol_1_impact_rows_covered"] == 462 and
            coverage["stale_protocol_1_rows_authorized_for_reuse"] == 0, "Coverage stale claim mismatch")
    scope = json.loads((root / OUT_REL / "audit_scope_manifest.json").read_text(encoding="utf-8"))
    require(scope["tracked_inventory_count"] == 1553 and
            scope["structured_scan_rules"] == {k: v if isinstance(v, str) else list(v)
                                               for k, v in sorted(STRUCTURED_SCAN_RULES.items())},
            "Scope manifest differs from independent allowlist")
    return {
        "baseline": baseline, "new": new, "authority_count": len(authority),
        "corpus_count": len(inventory), "occurrence_count": len(occurrences),
        "disposition_counts": dict(sorted(Counter(r["action"] for r in dispositions).items())),
        "numerical_reconciliation_count": len(numerical), "protocol_1_impact_count": 462,
        "registry_count": len(registry), "rewrite_count": len(rewrites),
    }


def report_text(checks: dict[str, Any]) -> str:
    counts = checks["disposition_counts"]
    lines = [
        "# Section 2.3D terminology and claim reconciliation report", "",
        f"**Verdict: {SUCCESS}**", "", "## Scope and validation", "",
        f"- Starting HEAD: `{EXPECTED_HEAD}` on `main`.",
        "- All five prerequisite checksum packages and four prerequisite verdicts passed.",
        f"- Explicit tracked corpus: {checks['corpus_count']} files; independently rescanned terminology occurrences: {checks['occurrence_count']}.",
        f"- Numerical reconciliation rows: {checks['numerical_reconciliation_count']}; all 462 stale protocol_1 impact rows are covered and none is authorized for reuse.",
        f"- Future-manuscript registry rows: {checks['registry_count']}; rewrite actions: {checks['rewrite_count']}.",
        "- Producer and independent validator agreed on the corpus, hashes, occurrence spans, selectors, and protected working-tree baseline.",
        "", "## Dispositions", "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(counts.items()))
    lines += [
        "", "## Binding decisions", "",
        "- Project ID/OOD and OOD-penalty conclusions are withdrawn; use bounded chemical-family terminology.",
        "- The checkpoint is a JARVIS-DFT pretrained ALIGNN checkpoint, not an oxide-only checkpoint.",
        "- Nitride N=200 has best epochs `[49,1,1,1,1]`; inert-through-N=200 and adaptation-at-N=500 claims are withdrawn.",
        "- protocol_1 v3 is canonical; protocol_2/3 remain robustness evidence.",
        "- Dataset counts and the intentional inclusive-oxide/nitride asymmetry match Section 2.3B.",
        "- Zero-shot difference and ratio use nitride-minus-oxide and nitride-over-oxide; uncertainty follows Section 2.3C.",
        "- Inclusive oxide is primary and pure-oxide filtering is appendix/limitations sensitivity evidence with `INTERPRETATION_STABLE`.",
        "- Embedding claims are correlational. Duplicate raw embedding views are not independent confirmations, and drifted-metadata distance-error statistics are `DO_NOT_REUSE` pending canonical-error recomputation.",
        "", "## Governance and repository safety", "",
        "The three hash-pinned governance/session files remained unchanged. Versioned successors are required in the controlled checkpoint; none was promoted here. Section 08 remained absent and forbidden. The nested checkout and all staging/recovery paths were excluded. No tracked file was modified, nothing was staged, and no commit or remote operation occurred.",
        "", "## Handoff", "",
        "The exact next phase is **Controlled Section 2.3D evidence and governance checkpoint**. Do not begin Section 3 until that checkpoint is reviewed.",
    ]
    return "\n".join(lines) + "\n"


def finalize(root: Path, checks: dict[str, Any]) -> None:
    out = root / OUT_REL
    validation = {
        "schema_version": 1, "verdict": SUCCESS, "section": "2.3D",
        "repository": {"head": EXPECTED_HEAD, "branch": EXPECTED_BRANCH,
                       "cached_upstream_relationship": EXPECTED_UPSTREAM},
        "checks": {
            "prerequisite_checksums_and_verdicts": "passed",
            "section08_absent_and_unread": "passed", "tracked_and_staged_clean": "passed",
            "original_546_status_entries_preserved": "passed",
            "explicit_1553_file_corpus_rebuilt": "passed",
            "producer_validator_occurrence_agreement": "passed",
            "one_to_one_occurrence_disposition_coverage": "passed",
            "all_462_protocol_1_rows_quarantined_and_reconciled": "passed",
            "authority_conflicts_resolved": "passed", "future_claim_selectors_resolved": "passed",
            "unsupported_id_ood_and_causal_claims_not_authorized": "passed",
            "dataset_zero_shot_sensitivity_embedding_policies": "passed",
            "historical_governance_hashes_unchanged": "passed",
        },
        "counts": {k: v for k, v in checks.items() if k not in {"baseline", "new"}},
        "package_expectations": {"evidence_files": 16, "scripts": 2,
                                 "generated_manifest_entries": 16, "checksum_entries": 17,
                                 "final_untracked_entries": 564},
        "operation_attestations": {"existing_file_modified": False, "git_stage_commit_or_push": False,
                                   "remote_operation": False, "forbidden_or_excluded_input_used": False,
                                   "manuscript_rewrite_started": False},
        "next_phase": "Controlled Section 2.3D evidence and governance checkpoint",
    }
    write_json_once(out / "section2_3D_validation.json", validation)
    write_text_once(out / "section2_3D_report.md", report_text(checks))

    manifest_targets = sorted(
        [OUT_REL / name for name in (ALL_OUTPUTS - {"generated_output_manifest.json", "section2_3D_evidence.sha256"})]
        + [PRODUCER_REL, VALIDATOR_REL], key=lambda p: p.as_posix())
    require(len(manifest_targets) == 16, "Generated manifest target arithmetic failed")
    manifest = {
        "schema_version": 1, "section": "2.3D", "entry_count": 16,
        "self_exclusions": [
            (OUT_REL / "generated_output_manifest.json").as_posix(),
            (OUT_REL / "section2_3D_evidence.sha256").as_posix(),
        ],
        "entries": [{"repository_relative_path": path.as_posix(),
                     "size_bytes": (root / path).stat().st_size,
                     "sha256": sha256(root / path)} for path in manifest_targets],
    }
    write_json_once(out / "generated_output_manifest.json", manifest)
    checksum_targets = sorted(
        [OUT_REL / name for name in (ALL_OUTPUTS - {"section2_3D_evidence.sha256"})]
        + [PRODUCER_REL, VALIDATOR_REL], key=lambda p: p.as_posix())
    require(len(checksum_targets) == 17, "Checksum target arithmetic failed")
    write_text_once(out / "section2_3D_evidence.sha256", "\n".join(
        f"{sha256(root / path)}  {path.as_posix()}" for path in checksum_targets))


def verify_final_package(root: Path) -> None:
    out = root / OUT_REL
    require({p.name for p in out.iterdir()} == ALL_OUTPUTS, "Final evidence directory set mismatch")
    manifest = json.loads((out / "generated_output_manifest.json").read_text(encoding="utf-8"))
    expected_manifest = sorted(
        [(OUT_REL / name).as_posix() for name in ALL_OUTPUTS - {"generated_output_manifest.json", "section2_3D_evidence.sha256"}]
        + [PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()])
    require(manifest.get("entry_count") == 16 and
            [x["repository_relative_path"] for x in manifest.get("entries", [])] == expected_manifest,
            "Generated manifest exact set mismatch")
    for entry in manifest["entries"]:
        path = root / safe_rel(entry["repository_relative_path"])
        require(path.stat().st_size == entry["size_bytes"] and sha256(path) == entry["sha256"],
                f"Generated manifest hash mismatch: {path}")
    lines = [x for x in (out / "section2_3D_evidence.sha256").read_text(encoding="utf-8").splitlines() if x]
    expected_checksum = sorted(
        [(OUT_REL / name).as_posix() for name in ALL_OUTPUTS - {"section2_3D_evidence.sha256"}]
        + [PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()])
    require(len(lines) == 17, "Final checksum entry count mismatch")
    parsed: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, "Malformed final checksum line")
        wanted, rel = match.groups()
        require(sha256(root / safe_rel(rel)) == wanted, f"Final checksum mismatch: {rel}")
        parsed.append(rel)
    require(parsed == expected_checksum, "Final checksum exact set/order mismatch")
    validation = json.loads((out / "section2_3D_validation.json").read_text(encoding="utf-8"))
    require(validation.get("verdict") == SUCCESS, "Final validation verdict mismatch")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--finalize", action="store_true")
    mode.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.repo.resolve()
    require(root == EXPECTED_ROOT, f"Expected {EXPECTED_ROOT}, got {root}")
    phase = "verify-only" if args.verify_only else "finalize"
    checks = semantic_checks(root, phase)
    if args.finalize:
        finalize(root, checks)
        # The just-created final package is checked without rewriting anything.
        verify_repo(root, "verify-only")
        verify_final_package(root)
    else:
        verify_final_package(root)
    print(json.dumps({"verdict": SUCCESS, "mode": phase,
                      "corpus_files": checks["corpus_count"],
                      "terminology_occurrences": checks["occurrence_count"],
                      "protocol_1_rows_covered": checks["protocol_1_impact_count"],
                      "final_package_files": 18}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValidationFailure, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"{BLOCKED}: {exc}", file=sys.stderr)
        raise SystemExit(1)
