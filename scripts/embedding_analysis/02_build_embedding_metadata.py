#!/usr/bin/env python
"""Build canonical embedding metadata tables and analysis subset manifests.

This script does not extract embeddings and does not modify earlier extraction
outputs. It reads the fixed test-set embedding metadata plus frozen manifests,
then writes additive analysis tables under results/embeddings.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_DIR = "results/embeddings"
DEFAULT_EMBEDDING_METADATA = (
    "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv"
)
DEFAULT_EMBEDDING_MANIFEST = (
    "results/embeddings/manifests/test_set_embedding_manifest.json"
)
DEFAULT_SEED = 42
FAMILIES = ("oxide", "nitride")
POOL_FAMILIES = ("oxide", "nitride")
SPLIT_ORDER = ("train", "val", "test")


EMBEDDING_MASTER_FIELDS = [
    "structure_id",
    "material_id",
    "family",
    "split",
    "formula",
    "target_formation_energy_peratom",
    "pretrained_prediction",
    "absolute_error",
    "embedding_npz_path",
    "embedding_source",
    "embedding_npz_key",
    "embedding_row_index",
    "embedding_dim",
    "layer_name",
    "module_path",
    "filename",
    "structure_path",
    "source_manifest",
    "n_atoms",
    "is_oxide",
    "is_nitride",
    "is_oxynitride",
]

STRUCTURE_MASTER_FIELDS = [
    "structure_id",
    "material_id",
    "family",
    "split",
    "formula",
    "target_formation_energy_peratom",
    "pretrained_prediction",
    "absolute_error",
    "embedding_available",
    "embedding_npz_path",
    "embedding_row_index",
    "embedding_sources",
    "embedding_source_count",
    "filename",
    "structure_path",
    "source_manifest",
    "n_atoms",
    "is_oxide",
    "is_nitride",
    "is_oxynitride",
]

SUBSET_METADATA_FIELDS = [
    "subset_name",
    "subset_order",
    "structure_id",
    "material_id",
    "family",
    "split",
    "formula",
    "target_formation_energy_peratom",
    "pretrained_prediction",
    "absolute_error",
    "embedding_available",
    "embedding_npz_path",
    "embedding_row_index",
    "embedding_sources",
    "embedding_source_count",
    "filename",
    "structure_path",
    "source_manifest",
    "source_row_index",
    "n_atoms",
    "is_oxide",
    "is_nitride",
    "is_oxynitride",
    "selection_role",
    "sampling_seed",
    "sampled",
    "sample_pool_size",
    "sample_size",
    "sample_draw_rank",
]

ID_FIELDS = [
    "subset_name",
    "subset_order",
    "material_id",
    "family",
    "split",
    "selection_role",
    "source_manifest",
    "source_row_index",
    "sample_draw_rank",
]

COUNT_FIELDS = ["subset_name", "family", "split", "count"]


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def normalize_path_text(value: str, root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    return relpath(path, root)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DOMAIN_SHIFT embedding-analysis metadata and subset manifests."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-metadata", default=DEFAULT_EMBEDDING_METADATA)
    parser.add_argument("--embedding-manifest", default=DEFAULT_EMBEDDING_MANIFEST)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing metadata/subset outputs. Existing embeddings are never overwritten.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned counts and outputs without writing files.",
    )
    return parser.parse_args()


def output_paths(root: Path, output_root: Path) -> dict[str, Path]:
    report_dir = root / "results" / "derived_evidence" / "embedding_analysis"
    return {
        "master_embedding_metadata_csv": output_root
        / "metadata"
        / "fixed_test_set_embedding_master.csv",
        "master_structure_metadata_csv": output_root
        / "metadata"
        / "fixed_test_set_structure_master.csv",
        "subset_counts_csv": output_root / "manifests" / "subset_counts.csv",
        "subset_manifest_json": output_root / "manifests" / "subset_manifest.json",
        "embedding_metadata_manifest_json": output_root
        / "manifests"
        / "embedding_metadata_manifest.json",
        "report_subset_design_md": report_dir / "subset_design.md",
        "report_subset_counts_csv": report_dir / "subset_counts.csv",
    }


def refuse_existing(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing embedding-analysis metadata outputs. "
            "Use --allow-overwrite to regenerate them:\n"
            f"{joined}"
        )


def load_embedding_npz_path(embedding_manifest: Path, root: Path) -> str:
    payload = json.loads(embedding_manifest.read_text(encoding="utf-8"))
    paths = payload.get("paths", {})
    npz_path = paths.get("npz")
    if not npz_path:
        raise KeyError(f"Manifest does not contain paths.npz: {embedding_manifest}")
    return normalize_path_text(str(npz_path), root)


def build_fixed_test_masters(
    embedding_metadata_csv: Path,
    embedding_npz_path: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_rows = read_csv(embedding_metadata_csv)
    if not source_rows:
        raise ValueError(f"No rows found in {embedding_metadata_csv}")

    long_rows: list[dict[str, Any]] = []
    grouped: OrderedDict[str, dict[str, Any]] = OrderedDict()

    for row in source_rows:
        material_id = row["material_id"]
        embedding_source = row["embedding_source"]
        embedding_index = row["embedding_index"]
        embedding_dim = row["embedding_dim"]
        long_row = {
            "structure_id": material_id,
            "material_id": material_id,
            "family": row["family"],
            "split": row["split"],
            "formula": row.get("formula", ""),
            "target_formation_energy_peratom": row.get(
                "target_formation_energy_peratom", ""
            ),
            "pretrained_prediction": row.get("pretrained_prediction", ""),
            "absolute_error": row.get("absolute_error", ""),
            "embedding_npz_path": embedding_npz_path,
            "embedding_source": embedding_source,
            "embedding_npz_key": row.get("npz_key", embedding_source),
            "embedding_row_index": embedding_index,
            "embedding_dim": embedding_dim,
            "layer_name": row.get("layer_name", ""),
            "module_path": row.get("module_path", ""),
            "filename": row.get("filename", ""),
            "structure_path": row.get("structure_path", ""),
            "source_manifest": row.get("source_manifest", ""),
            "n_atoms": row.get("n_atoms", ""),
            "is_oxide": row.get("is_oxide", ""),
            "is_nitride": row.get("is_nitride", ""),
            "is_oxynitride": row.get("is_oxynitride", ""),
        }
        long_rows.append(long_row)

        if material_id not in grouped:
            grouped[material_id] = {
                "structure_id": material_id,
                "material_id": material_id,
                "family": row["family"],
                "split": row["split"],
                "formula": row.get("formula", ""),
                "target_formation_energy_peratom": row.get(
                    "target_formation_energy_peratom", ""
                ),
                "pretrained_prediction": row.get("pretrained_prediction", ""),
                "absolute_error": row.get("absolute_error", ""),
                "embedding_available": "True",
                "embedding_npz_path": embedding_npz_path,
                "embedding_row_index": embedding_index,
                "embedding_sources": set(),
                "embedding_source_count": 0,
                "filename": row.get("filename", ""),
                "structure_path": row.get("structure_path", ""),
                "source_manifest": row.get("source_manifest", ""),
                "n_atoms": row.get("n_atoms", ""),
                "is_oxide": row.get("is_oxide", ""),
                "is_nitride": row.get("is_nitride", ""),
                "is_oxynitride": row.get("is_oxynitride", ""),
            }
        elif str(grouped[material_id]["embedding_row_index"]) != str(embedding_index):
            raise ValueError(
                f"Inconsistent embedding row index for {material_id}: "
                f"{grouped[material_id]['embedding_row_index']} vs {embedding_index}"
            )
        grouped[material_id]["embedding_sources"].add(embedding_source)

    structure_rows: list[dict[str, Any]] = []
    for row in grouped.values():
        sources = sorted(row["embedding_sources"])
        row["embedding_sources"] = ";".join(sources)
        row["embedding_source_count"] = len(sources)
        structure_rows.append(row)

    return long_rows, structure_rows


def manifest_row_to_structure_row(
    raw: dict[str, str],
    family: str,
    source_manifest: str,
    source_row_index: int | str,
) -> dict[str, Any]:
    material_id = raw["jid"]
    filename = raw.get("filename", "")
    return {
        "structure_id": material_id,
        "material_id": material_id,
        "family": family,
        "split": raw.get("split", ""),
        "formula": raw.get("formula", ""),
        "target_formation_energy_peratom": raw.get("target", ""),
        "pretrained_prediction": "",
        "absolute_error": "",
        "embedding_available": "False",
        "embedding_npz_path": "",
        "embedding_row_index": "",
        "embedding_sources": "",
        "embedding_source_count": 0,
        "filename": filename,
        "structure_path": f"data/{family}/structures/{filename}",
        "source_manifest": source_manifest,
        "source_row_index": source_row_index,
        "n_atoms": raw.get("n_atoms", ""),
        "is_oxide": raw.get("is_oxide", ""),
        "is_nitride": raw.get("is_nitride", ""),
        "is_oxynitride": raw.get("is_oxynitride", ""),
    }


def load_manifest_rows(root: Path, family: str, subset: str) -> list[dict[str, Any]]:
    source_manifest = f"data/{family}/manifests/{subset}.csv"
    path = root / source_manifest
    rows = []
    for index, raw in enumerate(read_csv(path)):
        rows.append(manifest_row_to_structure_row(raw, family, source_manifest, index))
    return rows


def subset_row(
    base: dict[str, Any],
    subset_name: str,
    subset_order: int,
    selection_role: str,
    sampling_seed: int | str = "",
    sampled: str = "",
    sample_pool_size: int | str = "",
    sample_size: int | str = "",
    sample_draw_rank: int | str = "",
) -> dict[str, Any]:
    row = {field: base.get(field, "") for field in STRUCTURE_MASTER_FIELDS}
    row.update(
        {
            "subset_name": subset_name,
            "subset_order": subset_order,
            "source_row_index": base.get("source_row_index", ""),
            "selection_role": selection_role,
            "sampling_seed": sampling_seed,
            "sampled": sampled,
            "sample_pool_size": sample_pool_size,
            "sample_size": sample_size,
            "sample_draw_rank": sample_draw_rank,
        }
    )
    return row


def build_subsets(
    root: Path,
    fixed_test_rows: list[dict[str, Any]],
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    subsets: dict[str, list[dict[str, Any]]] = {}

    subsets["fixed_test_set"] = [
        subset_row(row, "fixed_test_set", index, "all_fixed_test")
        for index, row in enumerate(fixed_test_rows)
    ]

    oxide_pool = load_manifest_rows(root, "oxide", "pool")
    nitride_pool = load_manifest_rows(root, "nitride", "pool")
    sample_size = len(nitride_pool)
    if len(oxide_pool) < sample_size:
        raise ValueError("Oxide pool is smaller than nitride pool; cannot balance.")

    sampled_indices = random.Random(seed).sample(range(len(oxide_pool)), sample_size)
    draw_rank_by_index = {row_index: rank for rank, row_index in enumerate(sampled_indices)}
    selected_oxide = [oxide_pool[index] for index in sorted(sampled_indices)]
    balanced_rows: list[dict[str, Any]] = []
    for row in selected_oxide:
        source_index = int(row["source_row_index"])
        balanced_rows.append(
            subset_row(
                row,
                "balanced_pool_set",
                len(balanced_rows),
                "oxide_random_sample_matched_to_nitride_pool",
                sampling_seed=seed,
                sampled="True",
                sample_pool_size=len(oxide_pool),
                sample_size=sample_size,
                sample_draw_rank=draw_rank_by_index[source_index],
            )
        )
    for row in nitride_pool:
        balanced_rows.append(
            subset_row(
                row,
                "balanced_pool_set",
                len(balanced_rows),
                "all_nitride_pool",
                sampling_seed=seed,
                sampled="False",
                sample_pool_size=len(nitride_pool),
                sample_size=sample_size,
            )
        )
    subsets["balanced_pool_set"] = balanced_rows

    subsets["oxide_reference_pool"] = [
        subset_row(row, "oxide_reference_pool", index, "all_oxide_pool")
        for index, row in enumerate(oxide_pool)
    ]

    return subsets


def count_subset(rows: list[dict[str, Any]], subset_name: str) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    families = [family for family in ("nitride", "oxide") if any(row["family"] == family for row in rows)]
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        for split in SPLIT_ORDER:
            count = sum(1 for row in family_rows if row["split"] == split)
            if count:
                counts.append(
                    {
                        "subset_name": subset_name,
                        "family": family,
                        "split": split,
                        "count": count,
                    }
                )
    for family in families:
        family_rows = [row for row in rows if row["family"] == family]
        counts.append(
            {
                "subset_name": subset_name,
                "family": family,
                "split": "all",
                "count": len(family_rows),
            }
        )
    counts.append(
        {
            "subset_name": subset_name,
            "family": "all",
            "split": "all",
            "count": len(rows),
        }
    )
    return counts


def count_all_subsets(subsets: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    counts: list[dict[str, Any]] = []
    for subset_name in ("fixed_test_set", "balanced_pool_set", "oxide_reference_pool"):
        counts.extend(count_subset(subsets[subset_name], subset_name))
    return counts


def id_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{field: row.get(field, "") for field in ID_FIELDS} for row in rows]


def counts_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def write_subset_files(
    output_root: Path,
    root: Path,
    subsets: dict[str, list[dict[str, Any]]],
    seed: int,
) -> list[dict[str, Any]]:
    subset_entries: list[dict[str, Any]] = []
    for subset_name, rows in subsets.items():
        subset_dir = output_root / "subsets" / subset_name
        metadata_csv = subset_dir / "metadata.csv"
        ids_csv = subset_dir / "ids.csv"
        ids_json = subset_dir / "ids.json"
        manifest_json = subset_dir / "manifest.json"

        ids = id_records(rows)
        write_csv(metadata_csv, rows, SUBSET_METADATA_FIELDS)
        write_csv(ids_csv, ids, ID_FIELDS)
        write_json(ids_json, {"id_records": ids})
        write_json(
            manifest_json,
            {
                "created_by": "scripts/embedding_analysis/02_build_embedding_metadata.py",
                "subset_name": subset_name,
                "row_count": len(rows),
                "random_seed": seed if subset_name == "balanced_pool_set" else None,
                "counts_by_family": counts_by(rows, "family"),
                "counts_by_split": counts_by(rows, "split"),
                "metadata_csv": str(metadata_csv.resolve()),
                "ids_csv": str(ids_csv.resolve()),
                "ids_json": str(ids_json.resolve()),
            },
        )
        subset_entries.append(
            {
                "subset_name": subset_name,
                "row_count": len(rows),
                "metadata_csv": str(metadata_csv.resolve()),
                "ids_csv": str(ids_csv.resolve()),
                "ids_json": str(ids_json.resolve()),
                "manifest_json": str(manifest_json.resolve()),
            }
        )
    return subset_entries


def write_subset_design(path: Path, counts: list[dict[str, Any]]) -> None:
    lines = [
        "# embedding_analysis Embedding Subset Design",
        "",
        "This file defines the canonical metadata and row subsets to use before plotting or dimensionality reduction.",
        "",
        "## Master Test Metadata",
        "",
        "- Long embedding table: `results/embeddings/metadata/fixed_test_set_embedding_master.csv`",
        "- Structure-level table: `results/embeddings/metadata/fixed_test_set_structure_master.csv`",
        "- The long table has one row per structure per embedding source.",
        "- The structure-level table has one row per fixed-test structure and records the shared NPZ row index plus available embedding sources.",
        "",
        "## Named Subsets",
        "",
        "- `fixed_test_set`: all oxide test structures plus all nitride test structures. Embeddings and prediction/error metadata are available.",
        "- `balanced_pool_set`: all nitride train+val pool structures plus a random oxide train+val pool sample of equal size using seed `42`. This is the balanced visualization and separation-analysis subset.",
        "- `oxide_reference_pool`: all oxide train+val pool structures. This is the oxide distance-to-manifold reference subset.",
        "",
        "Pool subsets are metadata/subset manifests only at this stage. Their embeddings and pretrained predictions are intentionally blank until a later extraction step is run for pool data.",
        "",
        "## Sampling Rule",
        "",
        "The oxide side of `balanced_pool_set` is sampled with `random.Random(42).sample(...)` over the canonical oxide pool manifest row indices. The selected oxide rows are saved in canonical manifest order, and each row keeps its random draw rank in `sample_draw_rank`.",
        "",
        "## Counts",
        "",
        "| Subset | Family | Split | Count |",
        "|---|---|---|---|",
    ]
    for row in counts:
        lines.append(
            f"| `{row['subset_name']}` | `{row['family']}` | `{row['split']}` | {row['count']} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "| Subset | IDs | Metadata | Manifest |",
            "|---|---|---|---|",
        ]
    )
    for subset_name in ("fixed_test_set", "balanced_pool_set", "oxide_reference_pool"):
        lines.append(
            "| "
            f"`{subset_name}` | "
            f"`results/embeddings/subsets/{subset_name}/ids.csv` | "
            f"`results/embeddings/subsets/{subset_name}/metadata.csv` | "
            f"`results/embeddings/subsets/{subset_name}/manifest.json` |"
        )
    lines.extend(["", "Exact IDs are saved in each subset's `ids.csv` and `ids.json`.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def dry_run_report(subsets: dict[str, list[dict[str, Any]]], counts: list[dict[str, Any]]) -> None:
    print("Dry-run only. Planned subset counts:")
    for row in counts:
        print(f"  {row['subset_name']}: {row['family']} / {row['split']} = {row['count']}")
    for subset_name, rows in subsets.items():
        print(f"{subset_name}: {len(rows)} IDs")


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_root = (root / args.output_dir).resolve()
    embedding_metadata_csv = (root / args.embedding_metadata).resolve()
    embedding_manifest_json = (root / args.embedding_manifest).resolve()

    if not embedding_metadata_csv.exists():
        raise FileNotFoundError(f"Missing embedding metadata: {embedding_metadata_csv}")
    if not embedding_manifest_json.exists():
        raise FileNotFoundError(f"Missing embedding manifest: {embedding_manifest_json}")

    paths = output_paths(root, output_root)
    embedding_npz_path = load_embedding_npz_path(embedding_manifest_json, root)
    embedding_master, structure_master = build_fixed_test_masters(
        embedding_metadata_csv, embedding_npz_path
    )
    subsets = build_subsets(root, structure_master, args.seed)
    counts = count_all_subsets(subsets)

    if args.dry_run:
        dry_run_report(subsets, counts)
        return

    subset_output_paths = []
    for subset_name in subsets:
        subset_dir = output_root / "subsets" / subset_name
        subset_output_paths.extend(
            [
                subset_dir / "metadata.csv",
                subset_dir / "ids.csv",
                subset_dir / "ids.json",
                subset_dir / "manifest.json",
            ]
        )
    refuse_existing(
        list(paths.values()) + subset_output_paths,
        allow_overwrite=args.allow_overwrite,
    )

    write_csv(
        paths["master_embedding_metadata_csv"],
        embedding_master,
        EMBEDDING_MASTER_FIELDS,
    )
    write_csv(
        paths["master_structure_metadata_csv"],
        structure_master,
        STRUCTURE_MASTER_FIELDS,
    )
    write_csv(paths["subset_counts_csv"], counts, COUNT_FIELDS)
    write_csv(paths["report_subset_counts_csv"], counts, COUNT_FIELDS)
    subset_entries = write_subset_files(output_root, root, subsets, args.seed)

    write_json(
        paths["subset_manifest_json"],
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/02_build_embedding_metadata.py",
            "random_seed": args.seed,
            "sampling": {
                "balanced_pool_set": {
                    "nitride_pool_rule": "include all rows from data/nitride/manifests/pool.csv",
                    "oxide_pool_rule": "sample len(nitride_pool) row indices from data/oxide/manifests/pool.csv with random.Random(seed).sample",
                    "output_order": "selected oxide rows in canonical manifest order, then all nitride pool rows in canonical manifest order",
                }
            },
            "counts": counts,
            "subsets": subset_entries,
        },
    )
    write_json(
        paths["embedding_metadata_manifest_json"],
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/02_build_embedding_metadata.py",
            "command": " ".join(sys.argv),
            "random_seed": args.seed,
            "fixed_test_structure_count": len(structure_master),
            "fixed_test_embedding_rows": len(embedding_master),
            "embedding_sources": sorted(
                {
                    row["embedding_source"]
                    for row in embedding_master
                    if row.get("embedding_source")
                }
            ),
            "inputs": {
                "embedding_metadata": relpath(embedding_metadata_csv, root),
                "embedding_manifest": relpath(embedding_manifest_json, root),
                "fixed_test_manifests": [
                    "data/oxide/manifests/test.csv",
                    "data/nitride/manifests/test.csv",
                ],
                "pool_manifests": [
                    "data/oxide/manifests/pool.csv",
                    "data/nitride/manifests/pool.csv",
                ],
            },
            "outputs": {
                key.removesuffix("_json").removesuffix("_csv").removesuffix("_md"): relpath(
                    path, root
                )
                for key, path in paths.items()
            },
        },
    )
    write_subset_design(paths["report_subset_design_md"], counts)

    print("Embedding metadata and subset manifests complete.")
    print(f"Fixed test structures: {len(structure_master)}")
    print(f"Fixed test embedding rows: {len(embedding_master)}")
    for subset_name, rows in subsets.items():
        print(f"{subset_name}: {len(rows)} rows")
    print(f"Global subset manifest: {relpath(paths['subset_manifest_json'], root)}")


if __name__ == "__main__":
    main()
