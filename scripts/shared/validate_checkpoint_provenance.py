#!/usr/bin/env python3
"""Independently validate the Section 2.3A checkpoint-provenance package.

This validator is deliberately self-contained.  It recomputes identities,
archive relationships, membership statistics, fixed-test inventories, and
overlaps from the authorized inputs rather than accepting producer totals.
"""

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
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_REL = Path(
    "results/derived_evidence/provenance_dataset_closure/"
    "2_3A_checkpoint_provenance"
)
EVIDENCE = ROOT / EVIDENCE_REL
SOURCE = EVIDENCE / "source_artifacts"

EXPECTED_ROOT = Path(".")
EXPECTED_BRANCH = "main"
EXPECTED_HEAD = "577fcb8ecb3ad7d9e90a46e211627ef5f30993b3"
EXPECTED_CHECKPOINT_SHA = (
    "bce5cdafa06dc26ad8ddb3ceeb2bef7593c218dd66825e7cb5381c156317458f"
)
EXPECTED_CONFIG_SHA = (
    "abfb9b6922e90157210e7583ccdd41eea9204df08794489654ea1f4f67bd2589"
)
EXPECTED_README_SHA = (
    "89d77400accfd9a619e8dacc6fb965e8e56569a515bab4c5bdd6d3012f10d1ca"
)
EXPECTED_VERSION = "9835fe0d4b313e2522034ff39f0ebdbfecde99a2"
EXPECTED_CHECKPOINT_SIZE = 48_614_953
EXPECTED_CHECKPOINT_CRC = "a7b440ce"
EXPECTED_CONFIG_SIZE = 1_611
EXPECTED_CONFIG_CRC = "bb0c6caa"
EXPECTED_README_SIZE = 782

FIGSHARE_ARTICLE_ID = 17_005_681
FIGSHARE_FILE_ID = 31_458_679
FIGSHARE_FILE_NAME = "configs/pretrained.zip"
FIGSHARE_FILE_SIZE = 47_501_164
FIGSHARE_FILE_MD5 = "3de8105dfd5bfb8f9b70b00158a00b35"
ZIP_TAIL_BASE = 47_400_000
EXPECTED_IDS_SHA = (
    "4572a078db04801698c3d1f432a95a2f5e02270bffda6ffca3f5a3b40a5584ae"
)

CHECKPOINT_REL = Path("models/pretrained/checkpoint_300.pt")
CONFIG_REL = Path("configs/pretrained/config.json")
README_REL = Path("configs/pretrained/README.md")
OXIDE_TEST_REL = Path("data/oxide/manifests/test.csv")
NITRIDE_TEST_REL = Path("data/nitride/manifests/test.csv")

PRODUCER_REL = Path("scripts/shared/audit_checkpoint_provenance.py")
VALIDATOR_REL = Path("scripts/shared/validate_checkpoint_provenance.py")
VALIDATION_NAME = "section2_3A_validation.json"
REPORT_NAME = "section2_3A_report.md"
OUTPUT_MANIFEST_NAME = "generated_output_manifest.json"
CHECKSUM_NAME = "section2_3A_evidence.sha256"

FORBIDDEN_FRAGMENTS = (
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


def run_git(*args: str) -> str:
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


def reject_path(relative: str | Path, *, allow_preflight_exclusion: bool = False) -> None:
    value = Path(relative).as_posix()
    if value.startswith("./"):
        value = value[2:]
    require(not value.startswith("/"), f"Absolute path rejected: {value}")
    require(".." not in Path(value).parts, f"Traversal path rejected: {value}")
    require(
        value != "domain_shift-alignn-domain-shift"
        and not value.startswith("domain_shift-alignn-domain-shift/"),
        f"Nested checkout path rejected: {value}",
    )
    if allow_preflight_exclusion:
        return
    require(
        not any(fragment in value for fragment in FORBIDDEN_FRAGMENTS),
        f"Forbidden input path rejected: {value}",
    )
    require(
        not any(part.lower() in FORBIDDEN_TEMP_COMPONENTS for part in Path(value).parts),
        f"Temporary/cache path rejected: {value}",
    )
    require(
        not value.endswith((".pyc", ".pyo", ".tmp", ".temp", "~")),
        f"Temporary/compiled path rejected: {value}",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_git_status() -> list[dict[str, str]]:
    raw = subprocess.run(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    rows: list[dict[str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        text = item.decode("utf-8", errors="surrogateescape")
        rows.append({"status": text[:2], "path": text[3:]})
    return rows


def is_section_output(path: str) -> bool:
    return path.startswith(EVIDENCE_REL.as_posix() + "/") or path in {
        PRODUCER_REL.as_posix(),
        VALIDATOR_REL.as_posix(),
    }


def verify_sha_manifest(manifest: Path, base: Path) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"Malformed checksum line {line_number}: {manifest}")
        expected, relative = match.groups()
        reject_path(relative)
        path = base / relative
        require(path.is_file(), f"Checksum target missing: {path}")
        actual = sha256(path)
        require(actual == expected, f"Checksum mismatch: {path}")
        checked.append({"path": str(path.relative_to(ROOT)), "sha256": actual})
    require(bool(checked), f"Empty checksum manifest: {manifest}")
    return {"manifest": str(manifest.relative_to(ROOT)), "verified_file_count": len(checked)}


def parse_central_directory(tail: bytes) -> dict[str, Any]:
    eocd_at = tail.rfind(b"PK\x05\x06")
    require(eocd_at >= 0, "Official archive EOCD missing from frozen tail")
    signature, disk, cd_disk, disk_entries, total_entries, cd_size, cd_offset, comment_len = (
        struct.unpack_from("<4s4H2LH", tail, eocd_at)
    )
    require(signature == b"PK\x05\x06", "Invalid EOCD signature")
    require(disk == 0 and cd_disk == 0, "Multi-disk ZIP is unsupported")
    require(disk_entries == total_entries, "Inconsistent ZIP entry totals")
    require(eocd_at + 22 + comment_len <= len(tail), "Truncated EOCD comment")
    cursor = cd_offset - ZIP_TAIL_BASE
    require(cursor >= 0, "Central directory lies outside frozen range")
    entries: list[dict[str, Any]] = []
    for _ in range(total_entries):
        fields = struct.unpack_from("<4s6H3L5H2L", tail, cursor)
        require(fields[0] == b"PK\x01\x02", "Invalid central-directory entry")
        name_len, extra_len, entry_comment_len = fields[10:13]
        end = cursor + 46 + name_len + extra_len + entry_comment_len
        require(end <= len(tail), "Truncated central-directory entry")
        name = tail[cursor + 46 : cursor + 46 + name_len].decode("utf-8")
        entries.append(
            {
                "name": name,
                "compression_method": fields[4],
                "crc32": f"{fields[7]:08x}",
                "compressed_size": fields[8],
                "uncompressed_size": fields[9],
                "local_header_offset": fields[16],
            }
        )
        cursor = end
    require(cursor == cd_offset - ZIP_TAIL_BASE + cd_size, "ZIP directory size mismatch")
    return {
        "entry_count": total_entries,
        "central_directory_offset": cd_offset,
        "central_directory_size": cd_size,
        "files": entries,
    }


def extract_range_entry(path: Path) -> tuple[str, bytes, dict[str, Any]]:
    payload = path.read_bytes()
    require(len(payload) >= 30, f"Truncated ZIP local entry: {path}")
    fields = struct.unpack_from("<4s5H3L2H", payload, 0)
    require(fields[0] == b"PK\x03\x04", f"Invalid ZIP local-entry signature: {path}")
    compression = fields[3]
    expected_crc = fields[6]
    compressed_size = fields[7]
    uncompressed_size = fields[8]
    name_len, extra_len = fields[9], fields[10]
    data_start = 30 + name_len + extra_len
    data_end = data_start + compressed_size
    require(data_end <= len(payload), f"Truncated compressed ZIP payload: {path}")
    name = payload[30 : 30 + name_len].decode("utf-8")
    compressed = payload[data_start:data_end]
    if compression == 8:
        extracted = zlib.decompress(compressed, -15)
    elif compression == 0:
        extracted = compressed
    else:
        raise RuntimeError(f"Unsupported compression method {compression}: {path}")
    actual_crc = binascii.crc32(extracted) & 0xFFFFFFFF
    require(actual_crc == expected_crc, f"ZIP entry CRC mismatch: {name}")
    require(len(extracted) == uncompressed_size, f"ZIP entry size mismatch: {name}")
    return name, extracted, {
        "compression_method": compression,
        "compressed_size": compressed_size,
        "uncompressed_size": uncompressed_size,
        "crc32": f"{actual_crc:08x}",
        "uncompressed_sha256": hashlib.sha256(extracted).hexdigest(),
        "range_sha256": sha256(path),
    }


def validate_local_bundle() -> dict[str, Any]:
    expected = {
        CHECKPOINT_REL: (EXPECTED_CHECKPOINT_SHA, EXPECTED_CHECKPOINT_SIZE, EXPECTED_CHECKPOINT_CRC),
        CONFIG_REL: (EXPECTED_CONFIG_SHA, EXPECTED_CONFIG_SIZE, EXPECTED_CONFIG_CRC),
        README_REL: (EXPECTED_README_SHA, EXPECTED_README_SIZE, None),
    }
    rows: dict[str, Any] = {}
    inventory = {row["repository_relative_path"]: row for row in read_csv(EVIDENCE / "checkpoint_file_inventory.csv")}
    require(set(inventory) == {path.as_posix() for path in expected}, "Checkpoint inventory paths differ")
    for relative, (expected_sha, expected_size, expected_crc) in expected.items():
        reject_path(relative)
        path = ROOT / relative
        require(path.is_file(), f"Local checkpoint-bundle file missing: {relative}")
        actual_sha = sha256(path)
        actual_size = path.stat().st_size
        actual_crc = crc32(path)
        require(actual_sha == expected_sha, f"SHA-256 mismatch: {relative}")
        require(actual_size == expected_size, f"Size mismatch: {relative}")
        if expected_crc is not None:
            require(actual_crc == expected_crc, f"CRC32 mismatch: {relative}")
        tracked = bool(run_git("ls-files", "--", relative.as_posix()).strip())
        require(tracked, f"Bundle path is not tracked: {relative}")
        blob = run_git("rev-parse", f"HEAD:{relative.as_posix()}").strip()
        introduced = run_git(
            "log", "--diff-filter=A", "-1", "--format=%H", "--", relative.as_posix()
        ).strip()
        producer = inventory[relative.as_posix()]
        require(producer["sha256"] == actual_sha, f"Producer SHA mismatch: {relative}")
        require(int(producer["size_bytes"]) == actual_size, f"Producer size mismatch: {relative}")
        require(producer["crc32"] == actual_crc, f"Producer CRC mismatch: {relative}")
        require(producer["git_blob"] == blob, f"Producer Git blob mismatch: {relative}")
        require(producer["introduced_by_commit"] == introduced, f"Producer add-commit mismatch: {relative}")
        rows[relative.as_posix()] = {
            "sha256": actual_sha,
            "size_bytes": actual_size,
            "crc32": actual_crc,
            "git_blob": blob,
            "introduced_by_commit": introduced,
        }

    config = read_json(ROOT / CONFIG_REL)
    expected_top = {
        "version": EXPECTED_VERSION,
        "dataset": "dft_3d",
        "target": "formation_energy_peratom",
        "random_seed": 123,
        "epochs": 300,
        "n_train": None,
        "n_val": None,
        "n_test": None,
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "atom_features": "cgcnn",
        "neighbor_strategy": "k-nearest",
        "cutoff": 8.0,
        "max_neighbors": 12,
        "optimizer": "adamw",
        "scheduler": "onecycle",
        "criterion": "mse",
        "batch_size": 64,
        "learning_rate": 0.001,
    }
    for key, expected_value in expected_top.items():
        require(config.get(key) == expected_value, f"Config field mismatch: {key}")
    expected_model = {
        "name": "alignn",
        "alignn_layers": 4,
        "gcn_layers": 4,
        "embedding_features": 64,
        "hidden_features": 256,
        "output_features": 1,
    }
    for key, expected_value in expected_model.items():
        require(config.get("model", {}).get(key) == expected_value, f"Model config mismatch: {key}")
    provenance = read_json(EVIDENCE / "checkpoint_provenance.json")
    producer_fields = provenance["config_fields"]
    for key, value in expected_top.items():
        require(producer_fields.get(key) == value, f"Producer config fact mismatch: {key}")
    for key, value in expected_model.items():
        require(producer_fields["model"].get(key) == value, f"Producer model fact mismatch: {key}")
    rows["parsed_config"] = {**expected_top, "model": expected_model}
    return rows


def validate_official_archive() -> tuple[dict[str, Any], dict[str, list[str]]]:
    required = {
        "v1": SOURCE / "figshare_article_17005681_v1_metadata.json",
        "v9": SOURCE / "figshare_article_17005681_v9_metadata.json",
        "github": SOURCE / "github_commit_9835fe0_metadata.json",
        "registry": SOURCE / "alignn_pretrained_main.py",
        "tail": SOURCE / "figshare_file_31458679_zip_tail_47400000_47501163.bin",
        "config_range": SOURCE / "figshare_file_31458679_config_entry_44785352_44786114.bin",
        "ids_range": SOURCE / "figshare_file_31458679_ids_entry_44799446_44985113.bin",
        "official_config": SOURCE / "official_config.json",
        "official_ids": SOURCE / "official_ids_train_val_test.json",
    }
    for name, path in required.items():
        require(path.is_file(), f"Frozen official source missing ({name}): {path}")

    v1 = read_json(required["v1"])
    v9 = read_json(required["v9"])
    github = read_json(required["github"])
    require(v1.get("id") == FIGSHARE_ARTICLE_ID, "Figshare v1 article ID mismatch")
    require(v1.get("version") == 1, "Figshare version-1 metadata mismatch")
    require(v9.get("id") == FIGSHARE_ARTICLE_ID, "Figshare v9 article ID mismatch")
    require(v9.get("version") == 9, "Figshare current-version metadata mismatch")
    require(v1.get("doi") == "10.6084/m9.figshare.17005681.v1", "Figshare v1 DOI mismatch")
    require(v1.get("title") == "ALIGNN models on JARVIS-DFT dataset", "Figshare title mismatch")
    official = next((item for item in v1.get("files", []) if item.get("id") == FIGSHARE_FILE_ID), None)
    require(official is not None, "Official model archive file ID missing")
    require(official.get("name") == FIGSHARE_FILE_NAME, "Official archive name mismatch")
    require(official.get("size") == FIGSHARE_FILE_SIZE, "Official archive size mismatch")
    require(official.get("computed_md5") == FIGSHARE_FILE_MD5, "Official archive MD5 mismatch")
    current = next((item for item in v9.get("files", []) if item.get("id") == FIGSHARE_FILE_ID), None)
    require(current is not None, "Current Figshare record no longer contains official file ID")
    require(github.get("sha") == EXPECTED_VERSION, "Official GitHub commit identity mismatch")
    registry = required["registry"].read_text()
    require('"configs/pretrained"' in registry, "Official model registry name missing")
    require("figshare.com/ndownloader/files/31458679" in registry, "Official model registry URL missing")

    central = parse_central_directory(required["tail"].read_bytes())
    require(central["entry_count"] == 19, "Unexpected official archive member count")
    by_name = {entry["name"]: entry for entry in central["files"]}
    checkpoint_name = "models/pretrained/checkpoint_300.pt"
    config_name = "configs/pretrained/config.json"
    ids_name = "configs/pretrained/ids_train_val_test.json"
    require({checkpoint_name, config_name, ids_name} <= set(by_name), "Required official ZIP member missing")
    checkpoint_member = by_name[checkpoint_name]
    require(checkpoint_member["uncompressed_size"] == (ROOT / CHECKPOINT_REL).stat().st_size, "Official/local checkpoint size differs")
    require(checkpoint_member["crc32"] == crc32(ROOT / CHECKPOINT_REL), "Official/local checkpoint CRC differs")

    extracted_config_name, extracted_config, config_details = extract_range_entry(required["config_range"])
    extracted_ids_name, extracted_ids, ids_details = extract_range_entry(required["ids_range"])
    require(extracted_config_name == config_name, "Frozen config range member name mismatch")
    require(extracted_ids_name == ids_name, "Frozen ID range member name mismatch")
    require(config_details["crc32"] == by_name[config_name]["crc32"], "Config range/central CRC mismatch")
    require(ids_details["crc32"] == by_name[ids_name]["crc32"], "ID range/central CRC mismatch")
    require(config_details["uncompressed_size"] == by_name[config_name]["uncompressed_size"], "Config range/central size mismatch")
    require(ids_details["uncompressed_size"] == by_name[ids_name]["uncompressed_size"], "ID range/central size mismatch")
    require(extracted_config == (ROOT / CONFIG_REL).read_bytes(), "Official config is not byte-identical locally")
    require(extracted_config == required["official_config"].read_bytes(), "Extracted official config artifact differs")
    require(extracted_ids == required["official_ids"].read_bytes(), "Extracted official ID artifact differs")
    require(ids_details["uncompressed_sha256"] == EXPECTED_IDS_SHA, "Official ID-list SHA mismatch")

    ids_payload = json.loads(extracted_ids)
    require(set(ids_payload) == {"id_train", "id_val", "id_test"}, "Official split schema mismatch")
    for key, values in ids_payload.items():
        require(isinstance(values, list), f"Official split is not a list: {key}")
        require(all(isinstance(value, str) for value in values), f"Non-string official ID: {key}")

    central_record = read_json(SOURCE / "official_archive_central_directory.json")
    require(central_record["entry_count"] == central["entry_count"], "Producer central entry count mismatch")
    producer_by_name = {entry["name"]: entry for entry in central_record["files"]}
    for member in (checkpoint_name, config_name, ids_name):
        require(producer_by_name[member] == by_name[member], f"Producer central record mismatch: {member}")

    inventory = read_json(EVIDENCE / "official_source_inventory.json")
    sources = inventory.get("sources", [])
    require(len(sources) >= 4, "Official source inventory is incomplete")
    required_source_fields = {
        "authority",
        "retrieval_timestamp",
        "source_type",
        "publisher_or_owner",
        "model_or_archive_identifier",
        "relevant_documented_facts",
        "provides_checksum",
        "provides_original_config",
        "provides_exact_training_ids",
        "evidence_strength",
        "uncertainty",
        "url",
    }
    for index, source in enumerate(sources, start=1):
        require(required_source_fields <= set(source), f"Official source {index} lacks required metadata fields")
    require(
        any(str(FIGSHARE_FILE_ID) in str(item.get("model_or_archive_identifier")) for item in sources),
        "Official file ID absent from source inventory",
    )
    require(
        any(item.get("model_or_archive_identifier") == EXPECTED_VERSION for item in sources),
        "Config commit absent from source inventory",
    )
    frozen_artifacts = inventory.get("frozen_source_artifacts", [])
    require(len(frozen_artifacts) == 9, "Frozen official-source artifact inventory is incomplete")
    for index, artifact in enumerate(frozen_artifacts, start=1):
        required_artifact_fields = {
            "repository_relative_path",
            "retrieval_description",
            "retrieval_timestamp",
            "sha256",
            "size_bytes",
            "source_url",
        }
        require(required_artifact_fields <= set(artifact), f"Frozen source artifact {index} lacks required fields")
        relative = artifact["repository_relative_path"]
        reject_path(relative)
        path = ROOT / relative
        require(path.is_file(), f"Frozen source artifact missing: {relative}")
        require(path.stat().st_size == artifact["size_bytes"], f"Frozen source size mismatch: {relative}")
        require(sha256(path) == artifact["sha256"], f"Frozen source SHA mismatch: {relative}")

    archive_facts = {
        "article_id": FIGSHARE_ARTICLE_ID,
        "file_id": FIGSHARE_FILE_ID,
        "file_name": FIGSHARE_FILE_NAME,
        "archive_size_bytes": FIGSHARE_FILE_SIZE,
        "official_archive_md5": FIGSHARE_FILE_MD5,
        "full_archive_md5_locally_recomputed": False,
        "member_count": central["entry_count"],
        "checkpoint_member_crc32": checkpoint_member["crc32"],
        "config_byte_identical": True,
        "id_list_sha256": ids_details["uncompressed_sha256"],
        "config_version_commit": github["sha"],
    }
    provenance = read_json(EVIDENCE / "checkpoint_provenance.json")
    binding = provenance.get("archive_binding", {})
    require(binding.get("checkpoint_crc32_matches_official_member") is True, "Producer checkpoint CRC binding flag differs")
    require(binding.get("checkpoint_size_matches_official_member") is True, "Producer checkpoint size binding flag differs")
    require(binding.get("config_bytes_match_official_member") is True, "Producer config binding flag differs")
    require(binding.get("official_archive_file_id") == FIGSHARE_FILE_ID, "Producer archive file ID differs")
    require(binding.get("official_archive_md5") == FIGSHARE_FILE_MD5, "Producer archive MD5 differs")
    require(binding.get("official_archive_md5_recomputed_locally") is False, "Producer overstates local MD5 verification")
    identifier = provenance.get("official_identifier", {})
    require(identifier.get("article_id") == FIGSHARE_ARTICLE_ID, "Producer article ID differs")
    require(identifier.get("archive_file_id") == FIGSHARE_FILE_ID, "Producer official identifier differs")
    require(identifier.get("archive_name") == FIGSHARE_FILE_NAME, "Producer archive name differs")
    dataset = provenance.get("documented_training_dataset", {})
    require(dataset.get("local_config_label") == "dft_3d", "Producer local dataset label differs")
    require(dataset.get("exact_snapshot_label_in_archive_metadata") is None, "Producer overstates snapshot identity")
    provenance_md = re.sub(r"\s+", " ", (EVIDENCE / "checkpoint_provenance.md").read_text())
    for fact in (
        EXPECTED_CHECKPOINT_SHA,
        str(FIGSHARE_FILE_ID),
        EXPECTED_VERSION,
        "Training entries: 44,578",
        "Unique training JIDs: 44,569",
        "Oxide: 0 of 1484 JIDs",
        "Nitride: 0 of 242 JIDs",
        "does not by itself establish a broad distributional or OOD label",
    ):
        require(fact in provenance_md, f"Checkpoint provenance prose lacks validated fact: {fact}")
    return archive_facts, ids_payload


def validate_membership(ids_payload: dict[str, list[str]]) -> dict[str, Any]:
    keys = ("id_train", "id_val", "id_test")
    counters = {key: Counter(ids_payload[key]) for key in keys}
    sets = {key: set(ids_payload[key]) for key in keys}
    malformed = {
        key: sorted(value for value in ids_payload[key] if re.fullmatch(r"JVASP-[0-9]+", value) is None)
        for key in keys
    }
    require(not any(malformed.values()), f"Malformed checkpoint JIDs: {malformed}")
    expected_counts = {
        "id_train": (44_578, 44_569),
        "id_val": (5_572, 5_572),
        "id_test": (5_572, 5_572),
    }
    for key, (raw_count, unique_count) in expected_counts.items():
        require(len(ids_payload[key]) == raw_count, f"Unexpected raw ID count: {key}")
        require(len(sets[key]) == unique_count, f"Unexpected unique ID count: {key}")
    train_duplicates = {jid: count for jid, count in counters["id_train"].items() if count > 1}
    require(
        train_duplicates == {
            "JVASP-100669": 2,
            "JVASP-116461": 2,
            "JVASP-96735": 6,
            "JVASP-97311": 3,
        },
        "Unexpected official training-list duplicates",
    )
    pairwise = {
        "train_test": sorted(sets["id_train"] & sets["id_test"]),
        "train_val": sorted(sets["id_train"] & sets["id_val"]),
        "val_test": sorted(sets["id_val"] & sets["id_test"]),
    }
    require(pairwise["train_test"] == ["JVASP-113961", "JVASP-116461"], "Unexpected train/test intersection")
    require(pairwise["train_val"] == [], "Unexpected train/validation intersection")
    require(pairwise["val_test"] == [], "Unexpected validation/test intersection")

    summary = read_json(EVIDENCE / "checkpoint_training_ids_summary.json")
    require(summary["status"] == "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED", "Producer membership summary status mismatch")
    require(summary["archive_entry_sha256"] == EXPECTED_IDS_SHA, "Producer membership source SHA mismatch")
    require(summary["pairwise_split_intersections"] == pairwise, "Producer pairwise intersections mismatch")
    for key in keys:
        split = key.removeprefix("id_")
        expected_summary = {
            "duplicate_identifiers": {
                jid: count for jid, count in sorted(counters[key].items()) if count > 1
            },
            "malformed_identifier_count": 0,
            "raw_count": len(ids_payload[key]),
            "unique_count": len(sets[key]),
        }
        require(summary["splits"][split] == expected_summary, f"Producer split summary mismatch: {split}")

    csv_rows = read_csv(EVIDENCE / "checkpoint_training_ids.csv")
    require(len(csv_rows) == sum(len(ids_payload[key]) for key in keys), "Training-ID CSV row count mismatch")
    memberships: dict[str, list[str]] = defaultdict(list)
    for key in keys:
        for jid in sets[key]:
            memberships[jid].append(key.removeprefix("id_"))
    cursor = 0
    for key in keys:
        split = key.removeprefix("id_")
        seen: Counter[str] = Counter()
        for position, original in enumerate(ids_payload[key], start=1):
            row = csv_rows[cursor]
            cursor += 1
            seen[original] += 1
            expected_row = {
                "jid": original,
                "original_identifier": original,
                "checkpoint_split": split,
                "row_position_in_split": str(position),
                "occurrence_index_for_jid_in_split": str(seen[original]),
                "split_occurrence_count": str(counters[key][original]),
                "duplicate_within_split": str(counters[key][original] > 1),
                "cross_split_memberships": ";".join(sorted(memberships[original])),
                "normalization_status": "unchanged_valid_jid",
                "source_identifier": f"figshare_file_{FIGSHARE_FILE_ID}",
            }
            require(row == expected_row, f"Training-ID CSV differs at row {cursor}")

    membership = read_json(EVIDENCE / "checkpoint_training_membership_status.json")
    require(membership["status"] == "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED", "Membership status mismatch")
    require(membership["exact_training_ids_found"] is True, "Exact membership flag is false")
    require(membership["checkpoint_training_entry_count"] == 44_578, "Membership raw count mismatch")
    require(membership["checkpoint_training_unique_jid_count"] == 44_569, "Membership unique count mismatch")
    require(membership["official_archive_file_id"] == FIGSHARE_FILE_ID, "Membership archive ID mismatch")
    return {
        "raw_counts": {key.removeprefix("id_"): len(ids_payload[key]) for key in keys},
        "unique_counts": {key.removeprefix("id_"): len(sets[key]) for key in keys},
        "training_duplicate_identifiers": dict(sorted(train_duplicates.items())),
        "pairwise_split_intersections": pairwise,
        "training_ids": sets["id_train"],
    }


def load_fixed_test(relative: Path, family: str, expected_count: int, expected_sha: str) -> dict[str, Any]:
    reject_path(relative)
    path = ROOT / relative
    require(path.is_file(), f"Fixed test manifest missing: {relative}")
    actual_sha = sha256(path)
    require(actual_sha == expected_sha, f"Fixed test SHA mismatch: {family}")
    rows = read_csv(path)
    require(rows and "jid" in rows[0] and "split" in rows[0], f"Malformed fixed test CSV: {family}")
    ids = [row["jid"].strip() for row in rows]
    require(all(re.fullmatch(r"JVASP-[0-9]+", jid) for jid in ids), f"Malformed fixed test JID: {family}")
    unique = set(ids)
    splits = Counter(row["split"].strip() for row in rows)
    require(len(rows) == expected_count, f"Fixed test row count mismatch: {family}")
    require(len(unique) == expected_count, f"Fixed test unique count mismatch: {family}")
    require(len(ids) - len(unique) == 0, f"Fixed test duplicates found: {family}")
    require(splits == {"test": expected_count}, f"Fixed test split inconsistency: {family}")
    return {
        "family": family,
        "path": relative.as_posix(),
        "sha256": actual_sha,
        "row_count": len(rows),
        "unique_jid_count": len(unique),
        "duplicate_jid_rows": len(ids) - len(unique),
        "split_values": dict(splits),
        "ids": unique,
    }


def validate_fixed_tests_and_overlap(training_ids: set[str]) -> dict[str, Any]:
    fixed = {
        "oxide": load_fixed_test(
            OXIDE_TEST_REL,
            "oxide",
            1_484,
            "1c7a3099270f991fad8a4aab25f5eec23e79f7b696d0211beaacaed2a70c2444",
        ),
        "nitride": load_fixed_test(
            NITRIDE_TEST_REL,
            "nitride",
            242,
            "ae72294c15e954a5a143ecdae9cdcaa9f074049b9118248b0e73645cbf22275c",
        ),
    }
    inventory_rows = {row["family"]: row for row in read_csv(EVIDENCE / "fixed_test_inventory.csv")}
    require(set(inventory_rows) == {"oxide", "nitride"}, "Producer fixed-test inventory families differ")
    overlaps: dict[str, list[str]] = {}
    for family, record in fixed.items():
        producer = inventory_rows[family]
        require(producer["repository_relative_path"] == record["path"], f"Fixed inventory path mismatch: {family}")
        require(producer["sha256"] == record["sha256"], f"Fixed inventory SHA mismatch: {family}")
        require(int(producer["row_count"]) == record["row_count"], f"Fixed inventory row mismatch: {family}")
        require(int(producer["unique_jid_count"]) == record["unique_jid_count"], f"Fixed inventory unique mismatch: {family}")
        require(int(producer["duplicate_jid_rows"]) == 0, f"Fixed inventory duplicate mismatch: {family}")
        overlaps[family] = sorted(training_ids & record["ids"])

    overlap_json = read_json(EVIDENCE / "checkpoint_test_overlap.json")
    require(overlap_json["status"] == "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED", "Overlap JSON status mismatch")
    require(overlap_json["overlap_jids"] == overlaps, "Producer overlap JID sets differ")
    overlap_csv = {row["family"]: row for row in read_csv(EVIDENCE / "checkpoint_test_overlap.csv")}
    require(set(overlap_csv) == {"oxide", "nitride"}, "Overlap CSV families differ")
    family_json = {row["family"]: row for row in overlap_json["families"]}
    for family, values in overlaps.items():
        count = len(values)
        total = fixed[family]["unique_jid_count"]
        expected_percent = 100.0 * count / total
        require(count == 0, f"Unexpected checkpoint/test overlap for {family}: {count}")
        row = overlap_csv[family]
        require(int(row["fixed_test_count"]) == total, f"Overlap fixed count mismatch: {family}")
        require(int(row["checkpoint_training_entry_count"]) == 44_578, f"Overlap raw train count mismatch: {family}")
        require(int(row["checkpoint_training_unique_jid_count"]) == 44_569, f"Overlap unique train count mismatch: {family}")
        require(int(row["overlap_count"]) == count, f"Overlap count mismatch: {family}")
        require(float(row["overlap_percent_of_fixed_test"]) == expected_percent, f"Overlap percent mismatch: {family}")
        require(row["training_id_source_sha256"] == EXPECTED_IDS_SHA, f"Overlap ID source SHA mismatch: {family}")
        require(row["fixed_test_manifest_sha256"] == fixed[family]["sha256"], f"Overlap test SHA mismatch: {family}")
        require(row["overlap_status"] == "exact_authoritative_jid_intersection", f"Overlap status mismatch: {family}")
        require(family_json[family]["overlap_count"] == count, f"Overlap JSON count mismatch: {family}")
        jid_rows = read_csv(EVIDENCE / f"checkpoint_test_overlap_{family}_jids.csv")
        require([item["jid"] for item in jid_rows] == values, f"Overlap JID CSV mismatch: {family}")

    membership = read_json(EVIDENCE / "checkpoint_training_membership_status.json")
    require(membership["oxide_fixed_test_overlap_count"] == 0, "Membership oxide overlap mismatch")
    require(membership["nitride_fixed_test_overlap_count"] == 0, "Membership nitride overlap mismatch")
    return {
        family: {
            key: value
            for key, value in record.items()
            if key != "ids"
        }
        | {"overlap_count": len(overlaps[family]), "overlap_jids": overlaps[family]}
        for family, record in fixed.items()
    }


def validate_terminology() -> dict[str, Any]:
    path = EVIDENCE / "checkpoint_membership_terminology_decision.md"
    text = path.read_text()
    normalized_text = re.sub(r"\s+", " ", text)
    required_phrases = (
        "EXACT_CHECKPOINT_TRAINING_MEMBERSHIP_VERIFIED",
        "oxide fixed test: **0/1484**",
        "nitride fixed test: **0/242**",
        "Zero JID overlap does not prove",
        "Do not use `in-distribution`, `out-of-distribution`, `OOD`, or `OOD penalty`",
        "oxide comparator",
        "nitride target",
        "chemical-family performance gap",
        "domain-shift-consistent evidence",
    )
    for phrase in required_phrases:
        require(phrase in normalized_text, f"Terminology decision lacks required phrase: {phrase}")
    return {
        "broad_distribution_labels_not_authorized": True,
        "allowed_precise_statement": "zero exact JID intersection with unique official id_train entries",
        "prescribed_terms": [
            "oxide comparator",
            "nitride target",
            "chemical-family performance gap",
            "domain-shift-consistent evidence",
        ],
    }


def validate_search_inventory() -> dict[str, Any]:
    payload = read_json(EVIDENCE / "provenance_search_log.json")
    searches = payload.get("searches", [])
    require(len(searches) >= 5, "Provenance search inventory is incomplete")
    required = {
        "location",
        "search_type",
        "search_terms",
        "result",
        "matched_path_count",
        "matched_paths",
        "authoritative",
        "checkpoint_specific",
        "reason",
    }
    tracked_paths = set(run_git("ls-files").splitlines())
    for index, row in enumerate(searches, start=1):
        require(required <= set(row), f"Provenance search row {index} lacks required fields")
        if row["matched_path_count"] is None:
            require(row["matched_paths"] == [], f"External search row unexpectedly lists tracked paths: row {index}")
        else:
            require(row["matched_path_count"] == len(row["matched_paths"]), f"Search match count differs: row {index}")
        for match in row["matched_paths"]:
            relative = match["path"]
            reject_path(relative)
            require(relative in tracked_paths, f"Search result is not a tracked path: {relative}")
    require(any(row["result"] == "accepted_exact_membership" for row in searches), "Search log lacks exact-membership source")
    search_md = (EVIDENCE / "provenance_search_log.md").read_text()
    for row in searches:
        require(str(row["result"]) in search_md, f"Human-readable search log omits result: {row['result']}")
    return {"search_count": len(searches), "accepted_exact_membership_source": True}


def validate_preflight_and_governance() -> dict[str, Any]:
    require(ROOT.resolve() == EXPECTED_ROOT, f"Unexpected repository root: {ROOT.resolve()}")
    head = run_git("rev-parse", "HEAD").strip()
    branch = run_git("branch", "--show-current").strip()
    require(head == EXPECTED_HEAD, f"Unexpected HEAD: {head}")
    require(branch == EXPECTED_BRANCH, f"Unexpected branch: {branch}")
    section08 = ROOT / "results/derived_evidence/final_paper_factory/archived_submission_materials"
    require(not section08.exists(), "Section 08 exists; validator stopped without inspecting it")
    require(
        not run_git("ls-files", "--", "results/derived_evidence/final_paper_factory/archived_submission_materials").strip(),
        "Git tracks a Section 08 path",
    )

    preflight = read_json(EVIDENCE / "repository_preflight.json")
    require(preflight["repository_root"] == str(EXPECTED_ROOT), "Producer root mismatch")
    require(preflight["head"] == head and preflight["expected_head"] == EXPECTED_HEAD, "Producer HEAD mismatch")
    require(preflight["branch"] == branch, "Producer branch mismatch")
    require(preflight["section08_absent"] is True and preflight["section08_tracked"] is False, "Producer Section 08 gate mismatch")
    require(preflight["nested_checkout_excluded"] is True, "Producer did not record nested-checkout exclusion")

    status = parse_git_status()
    require(all(row["status"] == "??" for row in status), "Tracked working-tree modifications detected")
    current_baseline = [row for row in status if not is_section_output(row["path"])]
    producer_baseline = preflight["baseline_status_entries"]
    require(current_baseline == producer_baseline, "Pre-existing working-tree baseline changed during Section 2.3A")
    require(preflight.get("baseline_tracked_status_count") == 0, "Producer tracked-status count is not zero")
    require(
        preflight.get("baseline_untracked_status_count") == len(producer_baseline),
        "Producer untracked-status count differs from baseline",
    )
    require(preflight["baseline_status_count"] == 546, "Unexpected pre-existing status count")

    governance_current: dict[str, Any] = {}
    before = {item["path"]: item for item in preflight["governance_files_before"]}
    expected_governance = {
        "results/derived_evidence/input_manifest.md",
        "results/derived_evidence/source_policy.md",
        "results/derived_evidence/run_session.json",
    }
    require(set(before) == expected_governance, "Governance baseline paths differ")
    for relative in sorted(expected_governance):
        reject_path(relative)
        path = ROOT / relative
        require(path.is_file(), f"Governance file missing: {relative}")
        current = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        require(current["sha256"] == before[relative]["sha256"], f"Governance file changed: {relative}")
        require(current["size_bytes"] == before[relative]["size_bytes"], f"Governance file size changed: {relative}")
        governance_current[relative] = current

    promotion = ROOT / "results/derived_evidence/protocol_1_promotion/promotion_evidence.sha256"
    regeneration = ROOT / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_evidence.sha256"
    report = ROOT / "results/derived_evidence/protocol_1_regeneration/protocol_1_regeneration_report.md"
    require("protocol_1_REGENERATION_VALIDATED" in report.read_text(), "Section 2.2 validation marker missing")
    prior_manifests = [
        verify_sha_manifest(promotion, ROOT),
        verify_sha_manifest(regeneration, regeneration.parent),
    ]
    return {
        "repository_root": str(ROOT),
        "branch": branch,
        "head": head,
        "last_three_commits": run_git("log", "-3", "--format=%H %s").splitlines(),
        "section08_absent": True,
        "section08_untracked": True,
        "nested_checkout_excluded": True,
        "tracked_status_count": 0,
        "baseline_status_count": len(current_baseline),
        "current_status_count_before_validator_outputs": len(status),
        "governance_files_unchanged": governance_current,
        "section2_2_marker_verified": True,
        "prior_checksum_manifests": prior_manifests,
    }


def purpose_for(path: Path) -> tuple[str, str, str, bool]:
    relative = path.relative_to(ROOT).as_posix()
    if relative == PRODUCER_REL.as_posix():
        return ("Section 2.3A deterministic evidence producer", "procedural authority", relative, True)
    if relative == VALIDATOR_REL.as_posix():
        return ("Section 2.3A independent validator", "procedural authority", relative, True)
    name = path.name
    if "source_artifacts/" in relative:
        if name == "official_archive_central_directory.json":
            return (
                "Deterministically parsed official ZIP central-directory evidence",
                "validated derived evidence",
                PRODUCER_REL.as_posix(),
                True,
            )
        if name in {"official_config.json", "official_ids_train_val_test.json"}:
            return (
                "Exact bytes extracted from a checksum-validated official ZIP member range",
                "provenance authority from official external artifact",
                PRODUCER_REL.as_posix(),
                True,
            )
        return (
            "Frozen unmodified official provenance source artifact",
            "provenance authority from official external source",
            "read-only official source retrieval plus producer freeze",
            True,
        )
    roles = {
        "repository_preflight.json": "Repository and safety-gate baseline",
        "checkpoint_file_inventory.csv": "Local checkpoint-bundle identity inventory",
        "checkpoint_provenance.json": "Machine-readable checkpoint provenance",
        "checkpoint_provenance.md": "Human-readable checkpoint provenance",
        "official_source_inventory.json": "Official-source inventory",
        "provenance_search_log.json": "Machine-readable provenance search log",
        "provenance_search_log.md": "Human-readable provenance search log",
        "fixed_test_inventory.csv": "Fixed-test identity inventory",
        "checkpoint_training_membership_status.json": "Checkpoint membership verdict",
        "checkpoint_training_ids.csv": "Exact official checkpoint split IDs",
        "checkpoint_training_ids_summary.json": "Official split-ID audit summary",
        "checkpoint_test_overlap.csv": "Machine-readable fixed-test overlap summary",
        "checkpoint_test_overlap.json": "Structured fixed-test overlap evidence",
        "checkpoint_test_overlap_oxide_jids.csv": "Exact oxide overlap JID list",
        "checkpoint_test_overlap_nitride_jids.csv": "Exact nitride overlap JID list",
        "checkpoint_membership_terminology_decision.md": "Conservative terminology decision",
        VALIDATION_NAME: "Independent Section 2.3A validation record",
        REPORT_NAME: "Section 2.3A validation handoff report",
    }
    producer = VALIDATOR_REL.as_posix() if name in {VALIDATION_NAME, REPORT_NAME} else PRODUCER_REL.as_posix()
    authority = "validated derived evidence" if name not in {"repository_preflight.json"} else "procedural evidence"
    return (roles.get(name, "Section 2.3A evidence artifact"), authority, producer, True)


def inputs_for(path: Path) -> list[str]:
    relative = path.relative_to(ROOT).as_posix()
    name = path.name
    local_bundle = [CHECKPOINT_REL.as_posix(), CONFIG_REL.as_posix(), README_REL.as_posix()]
    official_core = [
        (SOURCE / "figshare_article_17005681_v1_metadata.json").relative_to(ROOT).as_posix(),
        (SOURCE / "figshare_article_17005681_v9_metadata.json").relative_to(ROOT).as_posix(),
        (SOURCE / "alignn_pretrained_main.py").relative_to(ROOT).as_posix(),
        (SOURCE / "github_commit_9835fe0_metadata.json").relative_to(ROOT).as_posix(),
        (SOURCE / "figshare_file_31458679_zip_tail_47400000_47501163.bin").relative_to(ROOT).as_posix(),
        (SOURCE / "figshare_file_31458679_config_entry_44785352_44786114.bin").relative_to(ROOT).as_posix(),
        (SOURCE / "figshare_file_31458679_ids_entry_44799446_44985113.bin").relative_to(ROOT).as_posix(),
    ]
    fixed_tests = [OXIDE_TEST_REL.as_posix(), NITRIDE_TEST_REL.as_posix()]
    if relative in {PRODUCER_REL.as_posix(), VALIDATOR_REL.as_posix()}:
        return ["approved Section 2.3A specification", "authorized repository state"]
    if "source_artifacts/" in relative:
        inventory_path = EVIDENCE / "official_source_inventory.json"
        if inventory_path.is_file() and name != "official_source_inventory.json":
            inventory = read_json(inventory_path)
            match = next(
                (
                    item
                    for item in inventory.get("frozen_source_artifacts", [])
                    if item["repository_relative_path"] == relative
                ),
                None,
            )
            if match is not None:
                return [match["source_url"]]
        if name == "official_archive_central_directory.json":
            return [official_core[4], "Figshare file 31458679 official metadata"]
        if name == "official_config.json":
            return [official_core[5]]
        if name == "official_ids_train_val_test.json":
            return [official_core[6]]
        return ["official provenance retrieval recorded in official_source_inventory.json"]
    if name == "repository_preflight.json":
        return [
            "local Git HEAD/branch/status",
            "results/derived_evidence/input_manifest.md",
            "results/derived_evidence/source_policy.md",
            "results/derived_evidence/run_session.json",
        ]
    if name == "checkpoint_file_inventory.csv":
        return local_bundle + ["local Git object metadata"]
    if name in {"checkpoint_provenance.json", "checkpoint_provenance.md"}:
        return local_bundle + official_core
    if name == "official_source_inventory.json":
        return official_core
    if name in {"provenance_search_log.json", "provenance_search_log.md"}:
        return ["tracked repository provenance search"] + official_core
    if name == "fixed_test_inventory.csv":
        return fixed_tests
    if name in {
        "checkpoint_training_membership_status.json",
        "checkpoint_training_ids.csv",
        "checkpoint_training_ids_summary.json",
    }:
        return [
            (SOURCE / "official_ids_train_val_test.json").relative_to(ROOT).as_posix(),
            official_core[4],
            official_core[6],
        ]
    if name.startswith("checkpoint_test_overlap"):
        return [
            (SOURCE / "official_ids_train_val_test.json").relative_to(ROOT).as_posix(),
            *fixed_tests,
        ]
    if name == "checkpoint_membership_terminology_decision.md":
        return [(EVIDENCE / "checkpoint_test_overlap.json").relative_to(ROOT).as_posix()]
    if name in {VALIDATION_NAME, REPORT_NAME}:
        return [
            "all authorized Section 2.3A evidence artifacts",
            "both Section 2.2 checksum manifests",
            PRODUCER_REL.as_posix(),
            VALIDATOR_REL.as_posix(),
        ]
    return ["authorized Section 2.3A inputs"]


def build_generated_output_manifest(created_at: str) -> dict[str, Any]:
    evidence_names = (
        "repository_preflight.json",
        "checkpoint_file_inventory.csv",
        "checkpoint_provenance.json",
        "checkpoint_provenance.md",
        "official_source_inventory.json",
        "provenance_search_log.json",
        "provenance_search_log.md",
        "fixed_test_inventory.csv",
        "checkpoint_training_membership_status.json",
        "checkpoint_training_ids.csv",
        "checkpoint_training_ids_summary.json",
        "checkpoint_test_overlap.csv",
        "checkpoint_test_overlap.json",
        "checkpoint_test_overlap_oxide_jids.csv",
        "checkpoint_test_overlap_nitride_jids.csv",
        "checkpoint_membership_terminology_decision.md",
        VALIDATION_NAME,
        REPORT_NAME,
    )
    source_names = (
        "alignn_pretrained_main.py",
        "figshare_article_17005681_v1_metadata.json",
        "figshare_article_17005681_v9_metadata.json",
        "figshare_file_31458679_config_entry_44785352_44786114.bin",
        "figshare_file_31458679_ids_entry_44799446_44985113.bin",
        "figshare_file_31458679_zip_tail_47400000_47501163.bin",
        "github_commit_9835fe0_metadata.json",
        "official_archive_central_directory.json",
        "official_config.json",
        "official_ids_train_val_test.json",
    )
    evidence_files = [EVIDENCE / name for name in evidence_names]
    evidence_files += [SOURCE / name for name in source_names]
    for path in evidence_files:
        require(path.is_file(), f"Required generated/frozen evidence output missing: {path}")
    files = evidence_files + [ROOT / PRODUCER_REL, ROOT / VALIDATOR_REL]
    records: list[dict[str, Any]] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        reject_path(relative)
        role, authority, producer, validated = purpose_for(path)
        records.append(
            {
                "repository_relative_path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
                "producer": producer,
                "input_sources": inputs_for(path),
                "creation_timestamp": datetime.fromtimestamp(
                    path.stat().st_mtime, timezone.utc
                ).isoformat(),
                "intended_role": role,
                "authority_level": authority,
                "independently_validated": validated,
            }
        )
    return {
        "schema_version": 1,
        "created_at": created_at,
        "producer": VALIDATOR_REL.as_posix(),
        "entries": records,
        "entry_count": len(records),
        "self_reference_policy": {
            "generated_output_manifest": (
                "Excluded from its own entry table because a file cannot contain its final file-level SHA-256; "
                "its final SHA-256 is recorded in section2_3A_evidence.sha256."
            ),
            "section2_3A_evidence.sha256": (
                "Excluded from the entry table and from itself to avoid checksum self-reference."
            ),
        },
    }


def validate_output_manifest(payload: dict[str, Any]) -> None:
    require(payload["entry_count"] == len(payload["entries"]), "Generated-output manifest count mismatch")
    seen: set[str] = set()
    for entry in payload["entries"]:
        required_fields = {
            "repository_relative_path",
            "size_bytes",
            "sha256",
            "producer",
            "input_sources",
            "creation_timestamp",
            "intended_role",
            "authority_level",
            "independently_validated",
        }
        require(required_fields <= set(entry), "Generated-output manifest record is incomplete")
        relative = entry["repository_relative_path"]
        reject_path(relative)
        reject_path(entry["producer"])
        for source in entry["input_sources"]:
            require(
                not any(fragment in source for fragment in FORBIDDEN_FRAGMENTS)
                and "domain_shift-alignn-domain-shift/" not in source,
                f"Forbidden input reference in generated-output manifest: {relative}",
            )
        require(relative not in seen, f"Duplicate generated-output path: {relative}")
        seen.add(relative)
        path = ROOT / relative
        require(path.is_file(), f"Generated-output target missing: {relative}")
        require(path.stat().st_size == entry["size_bytes"], f"Generated-output size mismatch: {relative}")
        require(sha256(path) == entry["sha256"], f"Generated-output SHA mismatch: {relative}")
        require(entry["independently_validated"] is True, f"Output not marked independently validated: {relative}")


def write_evidence_checksum() -> dict[str, Any]:
    checksum_path = EVIDENCE / CHECKSUM_NAME
    manifest = read_json(EVIDENCE / OUTPUT_MANIFEST_NAME)
    files = [ROOT / entry["repository_relative_path"] for entry in manifest["entries"]]
    files.append(EVIDENCE / OUTPUT_MANIFEST_NAME)
    # The explicit list comes from the already path-guarded generated-output
    # manifest, avoiding traversal into any unapproved directory.
    files = sorted(files)
    lines: list[str] = []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        reject_path(relative)
        lines.append(f"{sha256(path)}  {relative}")
    checksum_path.write_text("\n".join(lines) + "\n")
    result = verify_sha_manifest(checksum_path, ROOT)
    result["self_excluded"] = True
    return result


def main() -> None:
    require(EVIDENCE.is_dir(), f"Section 2.3A evidence workspace missing: {EVIDENCE}")
    validated_at = utc_now()
    checks: list[dict[str, Any]] = []

    preflight = validate_preflight_and_governance()
    checks.append({"check": "safety_preflight_governance_and_section2_2", "status": "passed"})
    bundle = validate_local_bundle()
    checks.append({"check": "local_checkpoint_bundle", "status": "passed"})
    archive, ids_payload = validate_official_archive()
    checks.append({"check": "official_archive_binding", "status": "passed"})
    search = validate_search_inventory()
    checks.append({"check": "provenance_search_inventory", "status": "passed"})
    membership = validate_membership(ids_payload)
    checks.append({"check": "exact_checkpoint_membership", "status": "passed"})
    fixed_overlap = validate_fixed_tests_and_overlap(membership["training_ids"])
    checks.append({"check": "fixed_tests_and_exact_overlap", "status": "passed"})
    terminology = validate_terminology()
    checks.append({"check": "conservative_terminology", "status": "passed"})

    # Sets are only an in-memory recomputation aid and must not enter JSON.
    membership_json = {key: value for key, value in membership.items() if key != "training_ids"}
    validation = {
        "schema_version": 1,
        "validated_at": validated_at,
        "verdict": "SECTION23A_VALIDATED_KNOWN_MEMBERSHIP",
        "passed": True,
        "failed_check_count": 0,
        "checks": checks,
        "repository": preflight,
        "local_checkpoint_bundle": bundle,
        "official_archive": archive,
        "provenance_search": search,
        "checkpoint_membership": membership_json,
        "fixed_tests_and_overlap": fixed_overlap,
        "terminology_decision": terminology,
        "limitations": [
            "The official full-archive MD5 was frozen from Figshare metadata but was not locally recomputed because the full 47.5 MB archive was deliberately not downloaded.",
            "The authoritative split file contains two training/test cross-memberships; this fact is preserved, and downstream overlap uses unique id_train entries only.",
            "The official metadata names JARVIS-DFT/dft_3d but does not establish a more specific historical dataset snapshot label.",
        ],
        "forbidden_inputs_used": False,
        "manuscript_or_canonical_result_modified": False,
        "git_network_or_mutating_operation_performed": False,
    }
    write_json(EVIDENCE / VALIDATION_NAME, validation)

    report = f"""# Section 2.3A Validation Report

Verdict: **SECTION23A_VALIDATED_KNOWN_MEMBERSHIP**

## Safety and repository state

- Repository: `{preflight['repository_root']}`
- Branch: `{preflight['branch']}`
- HEAD: `{preflight['head']}`
- Section 08: absent and untracked; never inspected or used
- Nested checkout: excluded and untouched
- Pre-existing rerun/recovery controls: status-preserved and never hashed as scientific inputs
- Section 2.2 marker and both evidence checksum manifests: independently verified
- Tracked manuscript/canonical-result modifications: none

## Checkpoint provenance

- Local checkpoint SHA-256: `{EXPECTED_CHECKPOINT_SHA}`
- Local checkpoint size/CRC32: `{EXPECTED_CHECKPOINT_SIZE:,}` bytes / `{EXPECTED_CHECKPOINT_CRC}`
- Config SHA-256: `{EXPECTED_CONFIG_SHA}`
- Config version: `{EXPECTED_VERSION}`
- Local dataset / target: `dft_3d` / `formation_energy_peratom`
- Official distribution: Figshare article `{FIGSHARE_ARTICLE_ID}`, file `{FIGSHARE_FILE_ID}` (`{FIGSHARE_FILE_NAME}`)
- Official distribution statement: ALIGNN models on the JARVIS-DFT dataset
- Archive binding: checkpoint member size/CRC32 match; local config is byte-identical to the official archive member

The official archive's full MD5 (`{FIGSHARE_FILE_MD5}`) is reported by Figshare.
It was not locally recomputed because the full archive was deliberately not downloaded.

## Exact membership and overlap

- Official training entries / unique JIDs: **44,578 / 44,569**
- Official validation entries / unique JIDs: **5,572 / 5,572**
- Official test entries / unique JIDs: **5,572 / 5,572**
- Training-list duplicate identifiers: `{json.dumps(membership_json['training_duplicate_identifiers'], sort_keys=True)}`
- Official train/test intersection: `{membership_json['pairwise_split_intersections']['train_test']}`
- Oxide fixed test overlap: **0/1,484 (0.0%)**
- Nitride fixed test overlap: **0/242 (0.0%)**

The overlap result is the independently recomputed exact JID intersection with
unique official `id_train` entries. It does not authorize broad
in-distribution/out-of-distribution conclusions.

## Terminology decision

Use `oxide comparator`, `nitride target`, `chemical-family performance gap`,
and `domain-shift-consistent evidence`. Do not use `in-distribution`,
`out-of-distribution`, `OOD`, or `OOD penalty` as checkpoint-membership
conclusions.

## Preservation

`input_manifest.md`, `source_policy.md`, and `run_session.json` retain their
preflight SHA-256 identities. No Git network operation, stage, commit, or push
was performed by this validator.

The exact next subphase is **Section 2.3B — Dataset integrity and split audit**.
This report does not begin that subphase.
"""
    (EVIDENCE / REPORT_NAME).write_text(report)

    output_manifest = build_generated_output_manifest(validated_at)
    write_json(EVIDENCE / OUTPUT_MANIFEST_NAME, output_manifest)
    validate_output_manifest(read_json(EVIDENCE / OUTPUT_MANIFEST_NAME))
    checksum_result = write_evidence_checksum()
    require(checksum_result["verified_file_count"] > 0, "Final evidence checksum is empty")

    print(
        json.dumps(
            {
                "verdict": validation["verdict"],
                "validated_checks": len(checks),
                "generated_output_manifest_entries": output_manifest["entry_count"],
                "evidence_checksum_entries": checksum_result["verified_file_count"],
                "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA,
                "training_entries": 44_578,
                "training_unique_jids": 44_569,
                "oxide_overlap": 0,
                "nitride_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
