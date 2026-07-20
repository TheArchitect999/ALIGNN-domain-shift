#!/usr/bin/env python3
"""Validate the released supplementary package from a clean checkout.

The check is read-only and uses only public files. It verifies document
structure, local links, the 240-run table, figure inventory, and recorded hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown_targets(markdown: str) -> list[str]:
    return [match.group(1).strip() for match in re.finditer(r"!?\[[^\]]*\]\(([^)]+)\)", markdown)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    root = args.repo_root.expanduser().resolve()
    supplement = root / "paper/supplementary"
    markdown_path = supplement / "supplementary_materials.md"
    pdf_path = supplement / "supplementary_materials.pdf"
    run_table = supplement / "data/all_240_training_runs.csv"
    figure_manifest = supplement / "data/figure_manifest.csv"
    failures: list[str] = []

    required = (markdown_path, pdf_path, run_table, figure_manifest)
    for path in required:
        if not path.is_file():
            failures.append(f"missing required artifact: {path.relative_to(root)}")

    markdown = markdown_path.read_text(encoding="utf-8") if markdown_path.is_file() else ""
    expected_title = "# Supplementary Materials for"
    if not markdown.startswith(expected_title):
        failures.append("supplement title is missing or noncanonical")

    sections = [int(value) for value in re.findall(r"(?m)^## S(\d+)\.", markdown)]
    if sections != list(range(1, 14)):
        failures.append(f"expected ordered sections S1-S13; found {sections}")

    forbidden = {
        "named venue": "A" + "CM",
        "named region": "Ch" + "ina",
        "course identifier": "RES" + "201",
        "legacy protocol label": "hyperparameter" + " set",
        "private workstation": "/" + "Users" + "/",
    }
    lowered = markdown.lower()
    for label, token in forbidden.items():
        if token.lower() in lowered:
            failures.append(f"{label} remains in supplementary markdown")

    local_links = 0
    for target in markdown_targets(markdown):
        target = target.split("#", 1)[0].strip().strip("<>")
        if not target or re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I):
            continue
        local_links += 1
        resolved = (markdown_path.parent / target).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            failures.append(f"local link escapes repository: {target}")
            continue
        if not resolved.exists():
            failures.append(f"broken local link: {target}")

    run_rows: list[dict[str, str]] = []
    if run_table.is_file():
        with run_table.open(encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
        if len(run_rows) != 240:
            failures.append(f"expected 240 released runs; found {len(run_rows)}")
        expected_counts = {
            "set_id": {"1": 80, "2": 80, "3": 80},
            "experiment_type": {"fine_tune": 180, "from_scratch": 60},
            "family": {"oxide": 120, "nitride": 120},
            "validation_status": {"validated": 240},
        }
        for field, expected in expected_counts.items():
            observed = dict(Counter(row.get(field, "") for row in run_rows))
            if observed != expected:
                failures.append(f"unexpected {field} counts: {observed}")

    figure_rows: list[dict[str, str]] = []
    if figure_manifest.is_file():
        with figure_manifest.open(encoding="utf-8", newline="") as handle:
            figure_rows = list(csv.DictReader(handle))
        if len(figure_rows) != 24:
            failures.append(f"expected 24 run-grid manifest rows; found {len(figure_rows)}")
        for row in figure_rows:
            output = root / row.get("output_path", "")
            if not output.is_file():
                failures.append(f"missing manifested figure: {row.get('output_path', '')}")
                continue
            if sha256(output) != row.get("output_sha256"):
                failures.append(f"figure hash mismatch: {row.get('output_path', '')}")
            sources = row.get("source_paths", "").split(";")
            hashes = row.get("source_sha256s", "").split(";")
            if len(sources) != len(hashes):
                failures.append(f"source/hash count mismatch: {row.get('trace_id', '')}")
                continue
            for source, expected_hash in zip(sources, hashes):
                source_path = root / source
                if not source_path.is_file():
                    failures.append(f"missing figure source: {source}")
                elif sha256(source_path) != expected_hash:
                    failures.append(f"figure source hash mismatch: {source}")

    figure_files = sorted((supplement / "figures").glob("*.png"))
    if len(figure_files) != 28:
        failures.append(f"expected 28 supplementary PNG figures; found {len(figure_files)}")
    if pdf_path.is_file():
        if pdf_path.stat().st_size < 1_000_000:
            failures.append("supplementary PDF is unexpectedly small")
        if pdf_path.read_bytes()[:5] != b"%PDF-":
            failures.append("supplementary PDF signature is invalid")

    payload = {
        "status": "SUPPLEMENTARY_PACKAGE_VALIDATED" if not failures else "SUPPLEMENTARY_PACKAGE_FAILED",
        "sections": len(sections),
        "run_rows": len(run_rows),
        "figure_manifest_rows": len(figure_rows),
        "figure_files": len(figure_files),
        "local_links": local_links,
        "failures": failures,
    }
    print(json.dumps(payload, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
