#!/usr/bin/env python3
"""Produce the Section 2.3A checkpoint-provenance evidence package."""

from __future__ import annotations

import binascii
import csv
import hashlib
import json
import re
import struct
import subprocess
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3A_checkpoint_provenance"
)
EVIDENCE = ROOT / EVIDENCE_REL
SOURCE = EVIDENCE / "source_artifacts"
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
EXPECTED_CHECKPOINT_SHA = (
    "bce5cdafa06dc26ad8ddb3ceeb2bef7593c218dd66825e7cb5381c156317458f"
)
EXPECTED_CONFIG_SHA = (
    "abfb9b6922e90157210e7583ccdd41eea9204df08794489654ea1f4f67bd2589"
)
EXPECTED_VERSION = "9835fe0d4b313e2522034ff39f0ebdbfecde99a2"
FIGSHARE_ARTICLE_ID = 17005681
FIGSHARE_FILE_ID = 31458679
FIGSHARE_FILE_SIZE = 47_501_164
FIGSHARE_FILE_MD5 = "3de8105dfd5bfb8f9b70b00158a00b35"
ZIP_TAIL_BASE = 47_400_000

CHECKPOINT = ROOT / "models/pretrained/checkpoint_300.pt"
CONFIG = ROOT / "configs/pretrained/config.json"
LOCAL_README = ROOT / "configs/pretrained/README.md"
OXIDE_TEST = ROOT / "data/oxide/manifests/test.csv"
NITRIDE_TEST = ROOT / "data/nitride/manifests/test.csv"

SCRIPT_RELS = {
    "scripts/shared/audit_checkpoint_provenance.py",
    "scripts/shared/validate_checkpoint_provenance.py",
}
FORBIDDEN_PARTS = (
    "archived_submission_materials",
    "finetune_last2_reproduction_rerun",
    "reproduction_rerun",
    "__pycache__",
)
FORBIDDEN_TEMP_COMPONENTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "cache",
    "temp",
    "tmp",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crc32(path: Path) -> str:
    value = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value = zlib.crc32(chunk, value)
    return f"{value & 0xFFFFFFFF:08x}"


def git(*args: str, check: bool = True) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8", errors="replace")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def assert_safe_relative(relative: str) -> None:
    normalized = relative.replace("\\", "/")
    parts = Path(normalized).parts
    if normalized == "domain_shift-alignn-domain-shift" or normalized.startswith(
        "domain_shift-alignn-domain-shift/"
    ):
        raise RuntimeError(f"Nested checkout path rejected: {relative}")
    if any(part in normalized for part in FORBIDDEN_PARTS):
        raise RuntimeError(f"Forbidden path rejected: {relative}")
    if (
        any(part.lower() in FORBIDDEN_TEMP_COMPONENTS for part in parts)
        or normalized.endswith((".pyc", ".pyo", ".tmp", ".temp", "~"))
    ):
        raise RuntimeError(f"Temporary/cache path rejected: {relative}")


def tracked_and_blob(relative: str) -> tuple[bool, str | None]:
    assert_safe_relative(relative)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    blob = git("rev-parse", f"HEAD:{relative}").strip() if tracked else None
    return tracked, blob


def file_record(path: Path, role: str) -> dict[str, Any]:
    relative = path.relative_to(ROOT).as_posix()
    assert_safe_relative(relative)
    tracked, blob = tracked_and_blob(relative)
    stat = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "git_blob": blob,
        "git_status": "tracked" if tracked else "untracked",
        "intended_role": role,
        "modification_time": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "repository_relative_path": relative,
        "sha256": sha256(path),
        "size_bytes": stat.st_size,
    }


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
        return "rerun_staging_results"
    if path.startswith("configs/protocol_1/finetune_reproduction_rerun/"):
        return "rerun_configs"
    if path.startswith(
        "results/summaries/protocol_1/finetune/reproduction_rerun/"
    ):
        return "rerun_summary_controls"
    if path in {
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
    }:
        return "prior_governance_or_session"
    if path.startswith("domain_shift-alignn-domain-shift/"):
        return "nested_checkout"
    if path.startswith(EVIDENCE_REL.as_posix() + "/") or path in SCRIPT_RELS:
        return "section2_3A_outputs"
    return "ambiguous"


def tracked_match_category(path: str) -> str:
    """Classify a tracked search hit without treating it as provenance by name alone."""
    if path.startswith("configs/pretrained/"):
        return "local_checkpoint_bundle"
    if path == "alignn_references/alignn-main/alignn/pretrained.py":
        return "exact_model_registry_pointer"
    if path.startswith("Results_") and path.endswith("ids_train_val_test.json"):
        return "downstream_experiment_split_not_checkpoint_membership"
    if path.startswith("data/"):
        return "downstream_project_dataset_or_manifest"
    if path.startswith("alignn_references/"):
        return "generic_or_vendored_alignn_reference"
    if path.startswith("configs/") or path.startswith("scripts/"):
        return "project_configuration_or_code"
    return "other_tracked_reference"


def tracked_text_search(terms: list[str]) -> list[str]:
    """Search tracked, provenance-relevant text while excluding forbidden/manuscript areas."""
    allowed_roots = (
        "readme.md",
        "REPORT_PLAN.txt",
        "Project_Task",
        "artifacts",
        "docs",
        "env",
        "manifests",
        "requirements",
        "configs/pretrained/README.md",
        "configs/pretrained/config.json",
        "alignn_references/alignn-main/alignn",
        "data",
        "configs",
        "scripts",
    )
    excluded_globs = (
        ":(exclude,glob)**/*.pt",
        ":(exclude,glob)**/*.csv",
        ":(exclude,glob)**/*.zip",
        ":(exclude,glob)**/*.pkl",
        ":(exclude,glob)alignn_references/**/*.json",
    )
    command = ["git", "grep", "-l", "-I", "-F"]
    for term in terms:
        command.extend(["-e", term])
    command.extend(["--", *allowed_roots, *excluded_globs])
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"Tracked provenance search failed for {terms!r}: "
            + result.stderr.decode(errors="replace")
        )
    matched: set[str] = set()
    for path in result.stdout.decode(errors="replace").splitlines():
        assert_safe_relative(path)
        matched.add(path)
    return sorted(matched)


def source_artifact_record(
    path: Path,
    *,
    source_url: str,
    retrieval_description: str,
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "repository_relative_path": path.relative_to(ROOT).as_posix(),
        "retrieval_timestamp": datetime.fromtimestamp(
            stat.st_mtime, timezone.utc
        ).isoformat(),
        "retrieval_description": retrieval_description,
        "sha256": sha256(path),
        "size_bytes": stat.st_size,
        "source_url": source_url,
    }


def parse_central_directory(tail: bytes) -> dict[str, Any]:
    eocd_at = tail.rfind(b"PK\x05\x06")
    if eocd_at < 0:
        raise RuntimeError("Official archive EOCD not present in frozen range")
    eocd = struct.unpack_from("<4s4H2LH", tail, eocd_at)
    entry_count = eocd[4]
    directory_size = eocd[5]
    directory_offset = eocd[6]
    cursor = directory_offset - ZIP_TAIL_BASE
    files: list[dict[str, Any]] = []
    for _ in range(entry_count):
        fields = struct.unpack_from("<4s6H3L5H2L", tail, cursor)
        if fields[0] != b"PK\x01\x02":
            raise RuntimeError("Invalid official ZIP central-directory record")
        name_len, extra_len, comment_len = fields[10], fields[11], fields[12]
        name = tail[cursor + 46 : cursor + 46 + name_len].decode()
        files.append(
            {
                "compressed_size": fields[8],
                "compression_method": fields[4],
                "crc32": f"{fields[7]:08x}",
                "local_header_offset": fields[16],
                "name": name,
                "uncompressed_size": fields[9],
            }
        )
        cursor += 46 + name_len + extra_len + comment_len
    return {
        "archive_size_bytes": FIGSHARE_FILE_SIZE,
        "central_directory_offset": directory_offset,
        "central_directory_size": directory_size,
        "entry_count": entry_count,
        "files": files,
        "range_start": ZIP_TAIL_BASE,
    }


def extract_local_zip_entry(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    payload = path.read_bytes()
    fields = struct.unpack_from("<4s5H3L2H", payload, 0)
    if fields[0] != b"PK\x03\x04":
        raise RuntimeError(f"Invalid local ZIP entry range: {path}")
    compression = fields[3]
    expected_crc = fields[6]
    compressed_size = fields[7]
    expected_size = fields[8]
    name_len, extra_len = fields[9], fields[10]
    name = payload[30 : 30 + name_len].decode()
    start = 30 + name_len + extra_len
    compressed = payload[start : start + compressed_size]
    if compression == 8:
        extracted = zlib.decompress(compressed, -15)
    elif compression == 0:
        extracted = compressed
    else:
        raise RuntimeError(f"Unsupported ZIP compression method: {compression}")
    actual_crc = binascii.crc32(extracted) & 0xFFFFFFFF
    if actual_crc != expected_crc or len(extracted) != expected_size:
        raise RuntimeError(f"Official ZIP entry checksum/size failure: {name}")
    return name, extracted, {
        "compressed_size": compressed_size,
        "compression_method": compression,
        "crc32": f"{actual_crc:08x}",
        "name": name,
        "range_file_sha256": sha256(path),
        "uncompressed_sha256": hashlib.sha256(extracted).hexdigest(),
        "uncompressed_size": len(extracted),
    }


def load_test_manifest(path: Path, family: str) -> tuple[dict[str, Any], set[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [row["jid"].strip() for row in rows]
    splits = Counter(row["split"].strip() for row in rows)
    missing = sum(not value for value in ids)
    unique = set(ids)
    record = file_record(path, f"Fixed {family} downstream test identifiers")
    record.update(
        {
            "duplicate_jid_rows": len(ids) - len(unique),
            "family": family,
            "missing_jids": missing,
            "row_count": len(rows),
            "split_values": dict(splits),
            "unique_jid_count": len(unique),
        }
    )
    return record, unique


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    SOURCE.mkdir(parents=True, exist_ok=True)

    head = git("rev-parse", "HEAD").strip()
    branch = git("branch", "--show-current").strip()
    if head != EXPECTED_HEAD or branch != "main":
        raise RuntimeError(f"Unexpected Git state: {branch=} {head=}")
    section08 = ROOT / "results/derived_evidence/final_paper_factory/archived_submission_materials"
    if section08.exists():
        raise RuntimeError("Section 08 exists; refusing to inspect it")
    if git("ls-files", "--", section08.relative_to(ROOT).as_posix()).strip():
        raise RuntimeError("Section 08 has tracked paths")

    required_sources = {
        "figshare_v1": SOURCE / "figshare_article_17005681_v1_metadata.json",
        "figshare_v9": SOURCE / "figshare_article_17005681_v9_metadata.json",
        "github_commit": SOURCE / "github_commit_9835fe0_metadata.json",
        "official_registry": SOURCE / "alignn_pretrained_main.py",
        "zip_tail": SOURCE / "figshare_file_31458679_zip_tail_47400000_47501163.bin",
        "config_range": SOURCE / "figshare_file_31458679_config_entry_44785352_44786114.bin",
        "ids_range": SOURCE / "figshare_file_31458679_ids_entry_44799446_44985113.bin",
    }
    missing = [str(path) for path in required_sources.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen official source artifacts: {missing}")

    status = parse_status()
    classified = Counter(status_category(row["path"]) for row in status)
    baseline = [row for row in status if status_category(row["path"]) != "section2_3A_outputs"]
    baseline_categories = Counter(status_category(row["path"]) for row in baseline)
    if baseline_categories.get("ambiguous", 0):
        raise RuntimeError("Ambiguous pre-existing working-tree paths detected")
    if any(row["status"] != "??" for row in baseline):
        raise RuntimeError("Tracked modifications detected in the baseline working tree")

    governance = []
    for relative in (
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
    ):
        path = ROOT / relative
        governance.append(
            {
                "path": relative,
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "modification_time": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )

    preflight = {
        "baseline_status_categories": dict(sorted(baseline_categories.items())),
        "baseline_status_count": len(baseline),
        "baseline_status_entries": baseline,
        "baseline_tracked_status_count": sum(
            row["status"] != "??" for row in baseline
        ),
        "baseline_untracked_status_count": sum(
            row["status"] == "??" for row in baseline
        ),
        "branch": branch,
        "captured_at": utc_now(),
        "current_status_categories": dict(sorted(classified.items())),
        "current_status_count": len(status),
        "expected_head": EXPECTED_HEAD,
        "governance_files_before": governance,
        "head": head,
        "last_three_commits": [
            {"hash": line.split(" ", 1)[0], "subject": line.split(" ", 1)[1]}
            for line in git("log", "-3", "--format=%H %s").splitlines()
        ],
        "nested_checkout_excluded": True,
        "repository_root": str(ROOT),
        "section08_absent": True,
        "section08_tracked": False,
        "tracked_baseline_change_count": 0,
    }
    write_json(EVIDENCE / "repository_preflight.json", preflight)

    local_records = [
        file_record(CHECKPOINT, "Frozen pretrained ALIGNN weights"),
        file_record(CONFIG, "Official-bundle-equivalent model configuration"),
        file_record(LOCAL_README, "Local checkpoint bundle handoff note"),
    ]
    for record, path in zip(local_records, (CHECKPOINT, CONFIG, LOCAL_README)):
        record["crc32"] = crc32(path)
        record["introduced_by_commit"] = git(
            "log", "--diff-filter=A", "-1", "--format=%H", "--", record["repository_relative_path"]
        ).strip()
    write_csv(
        EVIDENCE / "checkpoint_file_inventory.csv",
        local_records,
        [
            "repository_relative_path",
            "absolute_path",
            "git_status",
            "git_blob",
            "introduced_by_commit",
            "size_bytes",
            "modification_time",
            "sha256",
            "crc32",
            "intended_role",
        ],
    )

    config = json.loads(CONFIG.read_text())
    if sha256(CHECKPOINT) != EXPECTED_CHECKPOINT_SHA:
        raise RuntimeError("Local checkpoint SHA-256 mismatch")
    if sha256(CONFIG) != EXPECTED_CONFIG_SHA:
        raise RuntimeError("Local config SHA-256 mismatch")
    if (
        config.get("version") != EXPECTED_VERSION
        or config.get("dataset") != "dft_3d"
        or config.get("target") != "formation_energy_peratom"
    ):
        raise RuntimeError("Local config expected fields mismatch")

    figshare_v1 = json.loads(required_sources["figshare_v1"].read_text())
    figshare_v9 = json.loads(required_sources["figshare_v9"].read_text())
    github_commit = json.loads(required_sources["github_commit"].read_text())
    official_file = next(
        item for item in figshare_v1["files"] if item["id"] == FIGSHARE_FILE_ID
    )
    if (
        official_file["name"] != "configs/pretrained.zip"
        or official_file["size"] != FIGSHARE_FILE_SIZE
        or official_file["computed_md5"] != FIGSHARE_FILE_MD5
    ):
        raise RuntimeError("Official Figshare file metadata mismatch")
    registry = required_sources["official_registry"].read_text()
    if (
        '"configs/pretrained"' not in registry
        or "figshare.com/ndownloader/files/31458679" not in registry
    ):
        raise RuntimeError("Official ALIGNN registry mapping not found")
    if github_commit.get("sha") != EXPECTED_VERSION:
        raise RuntimeError("Config version is not the expected official GitHub commit")

    central = parse_central_directory(required_sources["zip_tail"].read_bytes())
    central_by_name = {item["name"]: item for item in central["files"]}
    checkpoint_member = central_by_name[
        "models/pretrained/checkpoint_300.pt"
    ]
    config_member = central_by_name["configs/pretrained/config.json"]
    ids_member = central_by_name[
        "configs/pretrained/ids_train_val_test.json"
    ]
    if (
        checkpoint_member["uncompressed_size"] != CHECKPOINT.stat().st_size
        or checkpoint_member["crc32"] != crc32(CHECKPOINT)
    ):
        raise RuntimeError("Local checkpoint does not match official archive member metadata")

    config_name, official_config_bytes, config_range = extract_local_zip_entry(
        required_sources["config_range"]
    )
    ids_name, official_ids_bytes, ids_range = extract_local_zip_entry(
        required_sources["ids_range"]
    )
    if config_name != config_member["name"] or ids_name != ids_member["name"]:
        raise RuntimeError("Official entry-range names do not match central directory")
    if official_config_bytes != CONFIG.read_bytes():
        raise RuntimeError("Local config bytes differ from official archive config")

    official_config_path = SOURCE / "official_config.json"
    official_ids_path = SOURCE / "official_ids_train_val_test.json"
    official_config_path.write_bytes(official_config_bytes)
    official_ids_path.write_bytes(official_ids_bytes)

    ids_payload = json.loads(official_ids_bytes)
    expected_keys = {"id_train", "id_val", "id_test"}
    if set(ids_payload) != expected_keys:
        raise RuntimeError("Unexpected official ID-list schema")
    counters = {key: Counter(values) for key, values in ids_payload.items()}
    id_sets = {key: set(values) for key, values in ids_payload.items()}
    malformed = {
        key: sorted(value for value in values if not re.fullmatch(r"JVASP-[0-9]+", value))
        for key, values in ids_payload.items()
    }
    if any(malformed.values()):
        raise RuntimeError(f"Malformed official checkpoint identifiers: {malformed}")

    cross_memberships: dict[str, list[str]] = defaultdict(list)
    for split_key, values in id_sets.items():
        for value in values:
            cross_memberships[value].append(split_key.removeprefix("id_"))
    csv_rows: list[dict[str, Any]] = []
    for split_key in ("id_train", "id_val", "id_test"):
        split = split_key.removeprefix("id_")
        seen: Counter[str] = Counter()
        for position, original in enumerate(ids_payload[split_key], start=1):
            seen[original] += 1
            csv_rows.append(
                {
                    "checkpoint_split": split,
                    "cross_split_memberships": ";".join(sorted(cross_memberships[original])),
                    "duplicate_within_split": counters[split_key][original] > 1,
                    "jid": original.strip(),
                    "normalization_status": "unchanged_valid_jid",
                    "occurrence_index_for_jid_in_split": seen[original],
                    "original_identifier": original,
                    "row_position_in_split": position,
                    "split_occurrence_count": counters[split_key][original],
                    "source_identifier": f"figshare_file_{FIGSHARE_FILE_ID}",
                }
            )
    write_csv(
        EVIDENCE / "checkpoint_training_ids.csv",
        csv_rows,
        [
            "jid",
            "original_identifier",
            "checkpoint_split",
            "row_position_in_split",
            "occurrence_index_for_jid_in_split",
            "split_occurrence_count",
            "duplicate_within_split",
            "cross_split_memberships",
            "normalization_status",
            "source_identifier",
        ],
    )

    pairwise = {
        "train_test": sorted(id_sets["id_train"] & id_sets["id_test"]),
        "train_val": sorted(id_sets["id_train"] & id_sets["id_val"]),
        "val_test": sorted(id_sets["id_val"] & id_sets["id_test"]),
    }
    split_summary = {
        key.removeprefix("id_"): {
            "duplicate_identifiers": {
                jid: count for jid, count in sorted(counters[key].items()) if count > 1
            },
            "malformed_identifier_count": len(malformed[key]),
            "raw_count": len(ids_payload[key]),
            "unique_count": len(id_sets[key]),
        }
        for key in ("id_train", "id_val", "id_test")
    }
    training_summary = {
        "archive_entry": ids_name,
        "archive_entry_crc32": ids_range["crc32"],
        "archive_entry_sha256": ids_range["uncompressed_sha256"],
        "archive_entry_size_bytes": ids_range["uncompressed_size"],
        "pairwise_split_intersections": pairwise,
        "source_artifact": official_ids_path.relative_to(ROOT).as_posix(),
        "source_identifier": f"Figshare file {FIGSHARE_FILE_ID}",
        "splits": split_summary,
        "status": "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED",
    }
    write_json(EVIDENCE / "checkpoint_training_ids_summary.json", training_summary)

    oxide_record, oxide_ids = load_test_manifest(OXIDE_TEST, "oxide")
    nitride_record, nitride_ids = load_test_manifest(NITRIDE_TEST, "nitride")
    if (
        oxide_record["unique_jid_count"] != 1484
        or nitride_record["unique_jid_count"] != 242
        or oxide_record["duplicate_jid_rows"]
        or nitride_record["duplicate_jid_rows"]
        or oxide_record["split_values"] != {"test": 1484}
        or nitride_record["split_values"] != {"test": 242}
    ):
        raise RuntimeError("Fixed test manifest validation failed")
    fixed_records = [oxide_record, nitride_record]
    write_csv(
        EVIDENCE / "fixed_test_inventory.csv",
        fixed_records,
        [
            "family",
            "repository_relative_path",
            "absolute_path",
            "git_status",
            "git_blob",
            "size_bytes",
            "modification_time",
            "sha256",
            "row_count",
            "unique_jid_count",
            "duplicate_jid_rows",
            "missing_jids",
            "split_values",
            "intended_role",
        ],
    )

    training_unique = id_sets["id_train"]
    overlaps = {
        "oxide": sorted(training_unique & oxide_ids),
        "nitride": sorted(training_unique & nitride_ids),
    }
    test_records = {record["family"]: record for record in fixed_records}
    overlap_rows = []
    for family in ("oxide", "nitride"):
        count = len(overlaps[family])
        total = test_records[family]["unique_jid_count"]
        overlap_rows.append(
            {
                "checkpoint_training_entry_count": len(ids_payload["id_train"]),
                "checkpoint_training_unique_jid_count": len(training_unique),
                "family": family,
                "fixed_test_count": total,
                "fixed_test_manifest_sha256": test_records[family]["sha256"],
                "overlap_count": count,
                "overlap_percent_of_fixed_test": 100.0 * count / total,
                "overlap_status": "exact_authoritative_jid_intersection",
                "training_id_source_sha256": ids_range["uncompressed_sha256"],
            }
        )
        write_csv(
            EVIDENCE / f"checkpoint_test_overlap_{family}_jids.csv",
            [{"jid": jid} for jid in overlaps[family]],
            ["jid"],
        )
    write_csv(
        EVIDENCE / "checkpoint_test_overlap.csv",
        overlap_rows,
        [
            "family",
            "fixed_test_count",
            "checkpoint_training_entry_count",
            "checkpoint_training_unique_jid_count",
            "overlap_count",
            "overlap_percent_of_fixed_test",
            "overlap_status",
            "training_id_source_sha256",
            "fixed_test_manifest_sha256",
        ],
    )
    write_json(
        EVIDENCE / "checkpoint_test_overlap.json",
        {
            "definition": "Exact JID intersection with unique official id_train entries",
            "families": overlap_rows,
            "overlap_jids": overlaps,
            "status": "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED",
        },
    )

    membership = {
        "checkpoint_training_entry_count": len(ids_payload["id_train"]),
        "checkpoint_training_unique_jid_count": len(training_unique),
        "exact_training_ids_found": True,
        "official_archive_file_id": FIGSHARE_FILE_ID,
        "official_archive_member": ids_name,
        "oxide_fixed_test_overlap_count": len(overlaps["oxide"]),
        "nitride_fixed_test_overlap_count": len(overlaps["nitride"]),
        "status": "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED",
    }
    write_json(EVIDENCE / "checkpoint_training_membership_status.json", membership)

    central_summary = {
        **central,
        "archive_file_id": FIGSHARE_FILE_ID,
        "archive_md5_from_official_metadata": FIGSHARE_FILE_MD5,
        "archive_md5_locally_recomputed": False,
        "archive_url": official_file["download_url"],
        "config_range": config_range,
        "ids_range": ids_range,
        "local_checkpoint_crc32": crc32(CHECKPOINT),
        "local_checkpoint_sha256": sha256(CHECKPOINT),
    }
    write_json(SOURCE / "official_archive_central_directory.json", central_summary)

    registry_url = "https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py"
    registry_raw_url = (
        "https://raw.githubusercontent.com/usnistgov/alignn/main/alignn/pretrained.py"
    )
    source_inventory = {
        "inventory_created_at": utc_now(),
        "sources": [
            {
                "artifact_id": FIGSHARE_ARTICLE_ID,
                "authority": "primary official distribution metadata",
                "doi": figshare_v1["doi"],
                "evidence_strength": "authoritative exact archive identity and official MD5",
                "model_or_archive_identifier": f"Figshare article {FIGSHARE_ARTICLE_ID} v1; file {FIGSHARE_FILE_ID}",
                "provides_checksum": True,
                "provides_exact_training_ids": True,
                "provides_original_config": True,
                "publisher_or_owner": "Figshare record published by Kamal Choudhary / NIST ALIGNN project",
                "relevant_documented_facts": [
                    "Names configs/pretrained.zip",
                    f"Assigns immutable file ID {FIGSHARE_FILE_ID}",
                    f"Records archive size {FIGSHARE_FILE_SIZE} bytes and MD5 {FIGSHARE_FILE_MD5}",
                    "The archive central directory records config.json and ids_train_val_test.json beside checkpoint_300.pt",
                ],
                "retrieval_timestamp": datetime.fromtimestamp(
                    required_sources["figshare_v1"].stat().st_mtime, timezone.utc
                ).isoformat(),
                "source_type": "official Figshare article metadata API response",
                "title": figshare_v1["title"],
                "uncertainty": "The full archive MD5 is official metadata and was not locally recomputed because downloading a replacement model was prohibited.",
                "url": figshare_v1["url_public_html"],
                "version": figshare_v1["version"],
            },
            {
                "artifact_id": FIGSHARE_ARTICLE_ID,
                "authority": "primary official current distribution metadata",
                "doi": figshare_v9["doi"],
                "evidence_strength": "authoritative confirmation that original file ID remains in the current record",
                "model_or_archive_identifier": f"Figshare article {FIGSHARE_ARTICLE_ID} v9; retained file {FIGSHARE_FILE_ID}",
                "provides_checksum": True,
                "provides_exact_training_ids": True,
                "provides_original_config": True,
                "publisher_or_owner": "Figshare record published by Kamal Choudhary / NIST ALIGNN project",
                "relevant_documented_facts": [
                    "Current article version retains the original formation-energy model archive",
                    f"Retains immutable file ID {FIGSHARE_FILE_ID}",
                ],
                "retrieval_timestamp": datetime.fromtimestamp(
                    required_sources["figshare_v9"].stat().st_mtime, timezone.utc
                ).isoformat(),
                "source_type": "official Figshare article metadata API response",
                "title": figshare_v9["title"],
                "uncertainty": "Later article versions add other files; provenance uses the immutable original file ID rather than assuming every version is byte-identical.",
                "url": figshare_v9["url_public_html"],
                "version": figshare_v9["version"],
            },
            {
                "authority": "primary official model registry",
                "evidence_strength": "authoritative exact model-name to Figshare-file mapping",
                "model_or_archive_identifier": "configs/pretrained -> Figshare file 31458679",
                "provides_checksum": False,
                "provides_exact_training_ids": False,
                "provides_original_config": False,
                "publisher_or_owner": "NIST usnistgov/alignn repository",
                "relevant_documented_facts": [
                    "Maps the exact local bundle name to Figshare file 31458679"
                ],
                "retrieval_timestamp": datetime.fromtimestamp(
                    required_sources["official_registry"].stat().st_mtime, timezone.utc
                ).isoformat(),
                "source_type": "official GitHub source file",
                "title": "ALIGNN official pretrained-model registry",
                "uncertainty": "The live main-branch file is mutable; its retrieved bytes are frozen locally and are corroborated by the tracked vendored registry.",
                "url": registry_url,
            },
            {
                "authority": "primary official source-control record",
                "evidence_strength": "authoritative identity for the config version field",
                "model_or_archive_identifier": EXPECTED_VERSION,
                "provides_checksum": False,
                "provides_exact_training_ids": False,
                "provides_original_config": False,
                "publisher_or_owner": "NIST usnistgov/alignn repository",
                "relevant_documented_facts": [
                    "The exact local config version resolves to an official ALIGNN commit",
                    f"Commit message: {github_commit['commit']['message']}",
                ],
                "retrieval_timestamp": datetime.fromtimestamp(
                    required_sources["github_commit"].stat().st_mtime, timezone.utc
                ).isoformat(),
                "source_type": "official GitHub commit API response",
                "title": f"usnistgov/alignn commit {EXPECTED_VERSION}",
                "uncertainty": "Commit identity does not alone prove checkpoint training membership; membership is supplied by the official model archive.",
                "url": github_commit["html_url"],
            },
        ],
        "frozen_source_artifacts": [
            source_artifact_record(
                required_sources["figshare_v1"],
                source_url="https://api.figshare.com/v2/articles/17005681/versions/1",
                retrieval_description="Unmodified official version-1 metadata response",
            ),
            source_artifact_record(
                required_sources["figshare_v9"],
                source_url="https://api.figshare.com/v2/articles/17005681/versions/9",
                retrieval_description="Unmodified official version-9 metadata response",
            ),
            source_artifact_record(
                required_sources["github_commit"],
                source_url=f"https://api.github.com/repos/usnistgov/alignn/commits/{EXPECTED_VERSION}",
                retrieval_description="Unmodified official GitHub commit API response",
            ),
            source_artifact_record(
                required_sources["official_registry"],
                source_url=registry_raw_url,
                retrieval_description="Unmodified official main-branch pretrained registry bytes",
            ),
            source_artifact_record(
                required_sources["zip_tail"],
                source_url=official_file["download_url"] + "#bytes=47400000-47501163",
                retrieval_description="Unmodified HTTP byte range containing ZIP central directory and EOCD",
            ),
            source_artifact_record(
                required_sources["config_range"],
                source_url=official_file["download_url"] + "#bytes=44785352-44786114",
                retrieval_description="Unmodified HTTP byte range containing config.json local ZIP entry",
            ),
            source_artifact_record(
                required_sources["ids_range"],
                source_url=official_file["download_url"] + "#bytes=44799446-44985113",
                retrieval_description="Unmodified HTTP byte range containing ids_train_val_test.json local ZIP entry",
            ),
            source_artifact_record(
                official_config_path,
                source_url=official_file["download_url"] + "#member=configs/pretrained/config.json",
                retrieval_description="Deterministically extracted exact bytes from frozen official config range",
            ),
            source_artifact_record(
                official_ids_path,
                source_url=official_file["download_url"] + "#member=configs/pretrained/ids_train_val_test.json",
                retrieval_description="Deterministically extracted exact bytes from frozen official ID-list range",
            ),
        ],
    }
    write_json(EVIDENCE / "official_source_inventory.json", source_inventory)

    search_definitions = [
        (
            "checkpoint_identity_and_checksum",
            ["checkpoint_300.pt", "configs/pretrained", EXPECTED_CHECKPOINT_SHA],
            "Only bundle/registry references are checkpoint-specific; checksum text was assessed separately from name-only matches.",
        ),
        (
            "config_version_and_source_identifiers",
            [EXPECTED_VERSION, "31458679", "figshare.com/ndownloader/files/31458679"],
            "The tracked registry pointer is accepted; generic version/config references are contextual only.",
        ),
        (
            "dataset_labels",
            ["dft_3d", "dft_3d_2021"],
            "Dataset-name matches are not accepted as exact historical checkpoint membership.",
        ),
        (
            "split_identifier_terms",
            ["id_train_val_test.json", "ids_train_val_test.json", "train_ids", "val_ids", "test_ids"],
            "Downstream experiment splits and generic ALIGNN manifests are explicitly rejected as substitutes for checkpoint membership.",
        ),
    ]
    search_rows: list[dict[str, Any]] = []
    for label, terms, reason in search_definitions:
        paths = tracked_text_search(terms)
        categorized = [
            {"category": tracked_match_category(path), "path": path} for path in paths
        ]
        search_rows.append(
            {
                "authoritative": any(
                    item["category"] == "exact_model_registry_pointer"
                    for item in categorized
                ),
                "checkpoint_specific": any(
                    item["category"] in {
                        "local_checkpoint_bundle", "exact_model_registry_pointer"
                    }
                    for item in categorized
                ),
                "location": "tracked repository provenance-relevant roots (manuscripts and forbidden paths excluded)",
                "matched_path_count": len(paths),
                "matched_paths": categorized,
                "reason": reason,
                "result": label,
                "search_terms": terms,
                "search_type": "tracked_repository_literal_text_search",
            }
        )

    id_list_paths = []
    for path in git("ls-files", "--", "results/protocol_1", "results/protocol_1", "results/protocol_2", "results/protocol_3", "alignn_references").splitlines():
        assert_safe_relative(path)
        lowered = Path(path).name.lower()
        if lowered in {"id_train_val_test.json", "ids_train_val_test.json"}:
            id_list_paths.append(path)
    search_rows.append(
        {
            "authoritative": False,
            "checkpoint_specific": False,
            "location": "tracked filename inventory in downstream results and vendored references",
            "matched_path_count": len(id_list_paths),
            "matched_paths": [
                {"category": tracked_match_category(path), "path": path}
                for path in sorted(id_list_paths)
            ],
            "reason": "Every tracked split-list candidate was classified; downstream experiment allocations cannot establish checkpoint pretraining membership.",
            "result": "split_manifest_filename_inventory",
            "search_terms": ["id_train_val_test.json", "ids_train_val_test.json"],
            "search_type": "tracked_repository_filename_search",
        }
    )
    search_rows.extend(
        [
            {
                "authoritative": True,
                "checkpoint_specific": True,
                "location": figshare_v1["url_public_html"],
                "matched_path_count": None,
                "matched_paths": [],
                "reason": "Official version-1 article identifies the exact named archive, immutable file ID, size, and MD5.",
                "result": "accepted_official_archive_identity",
                "search_terms": ["configs/pretrained.zip", str(FIGSHARE_FILE_ID)],
                "search_type": "official_external_metadata",
            },
            {
                "authoritative": True,
                "checkpoint_specific": True,
                "location": official_file["download_url"],
                "matched_path_count": None,
                "matched_paths": [],
                "reason": "Frozen byte ranges show checkpoint, byte-identical config, and exact split IDs in the same official archive.",
                "result": "accepted_exact_membership",
                "search_terms": ["checkpoint_300.pt", "config.json", "ids_train_val_test.json"],
                "search_type": "official_external_archive_range_metadata",
            },
            {
                "authoritative": True,
                "checkpoint_specific": True,
                "location": github_commit["html_url"],
                "matched_path_count": None,
                "matched_paths": [],
                "reason": "Official repository resolves the local config version to an exact ALIGNN commit; this is version provenance, not membership evidence.",
                "result": "accepted_version_identity",
                "search_terms": [EXPECTED_VERSION],
                "search_type": "official_external_source_control",
            },
        ]
    )
    write_json(EVIDENCE / "provenance_search_log.json", {"searches": search_rows})
    search_md = ["# Section 2.3A Provenance Search Log", ""]
    for row in search_rows:
        search_md.extend(
            [
                f"## {row['result']}",
                "",
                f"- Location: `{row['location']}`",
                f"- Search type: {row['search_type']}",
                f"- Terms: `{json.dumps(row['search_terms'])}`",
                f"- Authoritative: {row['authoritative']}",
                f"- Checkpoint-specific: {row['checkpoint_specific']}",
                f"- Matched tracked paths: {row['matched_path_count']}",
                f"- Reason: {row['reason']}",
                "",
            ]
        )
    (EVIDENCE / "provenance_search_log.md").write_text("\n".join(search_md))

    config_fields = {
        key: config.get(key)
        for key in (
            "version",
            "dataset",
            "target",
            "random_seed",
            "n_train",
            "n_val",
            "n_test",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "epochs",
            "batch_size",
            "weight_decay",
            "learning_rate",
            "warmup_steps",
            "criterion",
            "optimizer",
            "scheduler",
            "atom_features",
            "neighbor_strategy",
            "cutoff",
            "max_neighbors",
        )
    }
    config_fields["model"] = config["model"]
    provenance = {
        "archive_binding": {
            "checkpoint_crc32_matches_official_member": True,
            "checkpoint_size_matches_official_member": True,
            "config_bytes_match_official_member": True,
            "official_archive_file_id": FIGSHARE_FILE_ID,
            "official_archive_member_count": central["entry_count"],
            "official_archive_md5": FIGSHARE_FILE_MD5,
            "official_archive_md5_recomputed_locally": False,
        },
        "checkpoint": local_records[0],
        "config": local_records[1],
        "config_fields": config_fields,
        "documented_training_dataset": {
            "exact_snapshot_label_in_archive_metadata": None,
            "local_config_label": "dft_3d",
            "official_distribution_description": "ALIGNN models on JARVIS-DFT dataset",
            "official_model_registry_context": "JARVIS-DFT formation-energy-per-atom pretrained model",
        },
        "exact_membership": membership,
        "official_identifier": {
            "article_doi": figshare_v1["doi"],
            "article_id": FIGSHARE_ARTICLE_ID,
            "archive_file_id": FIGSHARE_FILE_ID,
            "archive_name": official_file["name"],
            "download_url": official_file["download_url"],
        },
        "schema_version": 1,
    }
    write_json(EVIDENCE / "checkpoint_provenance.json", provenance)

    provenance_md = f"""# Section 2.3A Checkpoint Provenance

## Verdict basis

The local checkpoint SHA-256 is `{sha256(CHECKPOINT)}`. Its size
({CHECKPOINT.stat().st_size:,} bytes) and CRC32 (`{crc32(CHECKPOINT)}`) match the
`checkpoint_300.pt` member in official Figshare file `{FIGSHARE_FILE_ID}`. The
local `config.json` is byte-identical to the official archive member. That same
archive contains `ids_train_val_test.json`, so its membership list is bound to
the checkpoint at archive level.

## Official source

- Model: `configs/pretrained`
- Figshare article: `{figshare_v1['title']}`
- DOI: `{figshare_v1['doi']}`
- File ID: `{FIGSHARE_FILE_ID}`
- Archive: `{official_file['name']}`
- Official archive MD5: `{FIGSHARE_FILE_MD5}` (recorded by Figshare; the full
  archive was deliberately not downloaded and this MD5 was not recomputed)
- Config version: `{config['version']}`, verified as an official
  `usnistgov/alignn` commit

## Dataset statement

The local config identifies `dft_3d`; the official distribution describes the
bundle as trained on JARVIS-DFT. The Figshare metadata does not provide a more
specific snapshot label, so `dft_3d`, `dft_3d_2021`, and any current JARVIS
snapshot must not be silently conflated.

## Exact split membership

- Training entries: {len(ids_payload['id_train']):,}
- Unique training JIDs: {len(training_unique):,}
- Validation entries/unique: {len(ids_payload['id_val']):,}/{len(id_sets['id_val']):,}
- Test entries/unique: {len(ids_payload['id_test']):,}/{len(id_sets['id_test']):,}
- Duplicate training identifiers: {json.dumps(split_summary['train']['duplicate_identifiers'], sort_keys=True)}
- Training/test split intersection in the official metadata: {pairwise['train_test']}

These internal duplicate/cross-split facts are reported rather than repaired.
Downstream overlap uses the unique official training-ID set.

## Fixed-test overlap

- Oxide: {len(overlaps['oxide'])} of {len(oxide_ids)} JIDs ({100*len(overlaps['oxide'])/len(oxide_ids):.6f}%)
- Nitride: {len(overlaps['nitride'])} of {len(nitride_ids)} JIDs ({100*len(overlaps['nitride'])/len(nitride_ids):.6f}%)

This establishes exact JID non-membership in the recorded checkpoint training
list. It does not by itself establish a broad distributional or OOD label.
"""
    (EVIDENCE / "checkpoint_provenance.md").write_text(provenance_md)

    terminology = f"""# Checkpoint-Membership Terminology Decision

Status: **EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED**

The official archive provides an exact training-ID list tied to the same
checkpoint/config bundle. Exact intersection shows:

- oxide fixed test: **{len(overlaps['oxide'])}/{len(oxide_ids)}** JIDs in the unique official training list;
- nitride fixed test: **{len(overlaps['nitride'])}/{len(nitride_ids)}** JIDs in the unique official training list.

Allowed statements are limited to these measured JID-membership facts. Zero JID
overlap does not prove that either chemical family is statistically
in-distribution or out-of-distribution.

Do not use `in-distribution`, `out-of-distribution`, `OOD`, or `OOD penalty` as
checkpoint-membership conclusions. Prefer `oxide comparator`, `nitride target`,
`chemical-family performance gap`, and `domain-shift-consistent evidence`.
"""
    (EVIDENCE / "checkpoint_membership_terminology_decision.md").write_text(terminology)

    print(
        json.dumps(
            {
                "checkpoint_sha256": sha256(CHECKPOINT),
                "config_version": config["version"],
                "membership_status": membership["status"],
                "nitride_overlap": len(overlaps["nitride"]),
                "oxide_overlap": len(overlaps["oxide"]),
                "training_entries": len(ids_payload["id_train"]),
                "training_unique_jids": len(training_unique),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
