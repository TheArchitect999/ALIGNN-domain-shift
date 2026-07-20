#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from family_dataset_lib import (
    TARGET_KEY_ALIASES,
    build_stage2_rows,
    canonicalize_split,
    dedupe_records_by_jid,
    generate_split_map,
    infer_split_map_from_records,
    load_json,
    load_split_map,
    materialize_alignn_root,
    records_by_jid,
    split_manifest_rows,
    summary_for_family,
    write_csv,
    write_json,
    write_structure_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build protocol-compliant oxide and nitride subset manifests from the "
            "JARVIS dft_3d_2021 dataset, preserving the original train/val/test split "
            "before family filtering."
        )
    )
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument(
        "--dataset-key",
        default="dft_3d_2021",
        help="jarvis.db.figshare dataset key to load via jarvis-tools (default: dft_3d_2021).",
    )
    src.add_argument(
        "--dataset-json",
        type=Path,
        help="Path to a pre-downloaded JARVIS JSON file, e.g. jdft_3d-8-18-2021.json.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("./cache/jarvis"),
        help="Directory used by jarvis-tools to cache downloaded zip files.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("./data_stage2"),
        help="Output directory for manifests, summaries, and optional POSCAR files.",
    )
    parser.add_argument(
        "--target-key",
        default="formation_energy_peratom",
        help="Target key to extract from each JARVIS record (default: formation_energy_peratom).",
    )
    parser.add_argument(
        "--target-aliases",
        nargs="*",
        default=list(TARGET_KEY_ALIASES),
        help="Fallback target-key aliases to try if --target-key is absent.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        help=(
            "Optional CSV/TSV/JSON file mapping jid -> split. Use this when the dataset does "
            "not expose the official split labels directly."
        ),
    )
    parser.add_argument(
        "--strict-official-splits",
        action="store_true",
        default=True,
        help=(
            "Fail if an official split mapping cannot be found or supplied. This is the default "
            "and is the correct mode for the registered protocol."
        ),
    )
    parser.add_argument(
        "--allow-generated-split",
        action="store_true",
        help=(
            "Allow a deterministic fallback split if official JARVIS split IDs cannot be found. "
            "Use only as a last resort and disclose it in the report."
        ),
    )
    parser.add_argument(
        "--generated-split-seed",
        type=int,
        default=123,
        help="Seed for the deterministic fallback split, only used with --allow-generated-split.",
    )
    parser.add_argument(
        "--keep-oxynitrides-in-oxide",
        action="store_true",
        default=True,
        help=(
            "Keep O-containing N compounds in the oxide arm. This matches the final task wording: "
            "oxides are structures containing O; nitrides are structures containing N and no O."
        ),
    )
    parser.add_argument(
        "--no-write-structures",
        action="store_true",
        help="Only write manifests and summaries; do not materialize POSCAR structure files.",
    )
    parser.add_argument(
        "--materialize-pool-and-test",
        action="store_true",
        help=(
            "Also create ALIGNN-ready root directories for each family's pool and test manifests. "
            "Useful for later zero-shot and fine-tuning stages."
        ),
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "symlink", "copy"],
        default="auto",
        help="How to populate ALIGNN-ready directories when --materialize-pool-and-test is used.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Write only schema/split diagnostics, then stop before exporting family data.",
    )
    return parser.parse_args()



def load_records(args: argparse.Namespace) -> List[Mapping[str, Any]]:
    if args.dataset_json is not None:
        obj = load_json(args.dataset_json)
        if not isinstance(obj, list):
            raise ValueError(f"Expected a JSON array of records in {args.dataset_json}")
        return obj
    try:
        from jarvis.db.figshare import data as jdata
    except ImportError as exc:
        raise ImportError(
            "jarvis-tools is required to load --dataset-key. Either install jarvis-tools or use --dataset-json."
        ) from exc
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    return jdata(args.dataset_key, store_dir=str(args.cache_dir))



def determine_split_map(args: argparse.Namespace, records: Sequence[Mapping[str, Any]]) -> tuple[Dict[str, str], str]:
    if args.splits_file is not None:
        split_map = load_split_map(args.splits_file)
        return split_map, f"provided:{args.splits_file}"

    inferred, source_key = infer_split_map_from_records(records)
    if inferred is not None and source_key is not None:
        return inferred, f"inferred_from_record_key:{source_key}"

    if args.allow_generated_split:
        jids = [str(r.get("jid") or r.get("id") or r.get("material_id") or "").strip() for r in records]
        jids = [jid for jid in jids if jid]
        split_map = generate_split_map(jids, seed=args.generated_split_seed)
        return split_map, f"generated_seed:{args.generated_split_seed}"

    raise RuntimeError(
        "Could not locate an official split mapping in the dataset records and no --splits-file was supplied. "
        "Run once with --inspect-only to see candidate keys, or provide a split manifest."
    )



def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    raw_records = load_records(args)
    records, duplicate_report = dedupe_records_by_jid(
        raw_records,
        target_key=args.target_key,
        target_aliases=args.target_aliases,
    )
    split_map, split_source = determine_split_map(args, records)

    build_result = build_stage2_rows(
        records=records,
        target_key=args.target_key,
        target_aliases=args.target_aliases,
        split_map=split_map,
        keep_oxynitrides_in_oxide=args.keep_oxynitrides_in_oxide,
    )

    build_result.split_source = split_source
    build_result.schema_report["duplicate_resolution"] = duplicate_report
    build_result.schema_report["n_records_after_dedup"] = len(records)

    diagnostics_dir = args.outdir / "diagnostics"
    write_json(build_result.schema_report, diagnostics_dir / "schema_report.json")
    write_json({jid: split for jid, split in sorted(split_map.items())}, diagnostics_dir / "global_split_manifest.json")
    write_csv(build_result.global_rows, diagnostics_dir / "global_record_catalog.csv")

    if args.inspect_only:
        print(json.dumps({
            "message": "inspect-only complete",
            "n_raw_records": len(raw_records),
            "n_records_after_dedup": len(records),
            "split_source": split_source,
            "schema_report": build_result.schema_report,
        }, indent=2))
        return

    record_lookup = records_by_jid(records)

    family_rows_map = {
        "oxide": build_result.oxide_rows,
        "nitride": build_result.nitride_rows,
    }

    for family_name, rows in family_rows_map.items():
        family_dir = args.outdir / family_name
        manifests_dir = family_dir / "manifests"
        structures_dir = family_dir / "structures"
        summaries_dir = family_dir / "summaries"
        family_dir.mkdir(parents=True, exist_ok=True)

        if not args.no_write_structures:
            write_structure_files(rows, record_lookup, structures_dir)

        family_manifests = split_manifest_rows(rows)
        for split_name, split_rows in family_manifests.items():
            write_csv(split_rows, manifests_dir / f"{split_name}.csv")

        summary = summary_for_family(
            rows=rows,
            family_name=family_name,
            split_source=split_source,
            dataset_key=(args.dataset_key if args.dataset_json is None else str(args.dataset_json)),
            target_key=build_result.target_key_used,
        )
        write_json(summary, summaries_dir / "summary.json")

        if args.materialize_pool_and_test and not args.no_write_structures:
            alignn_roots_dir = family_dir / "alignn_ready"
            for split_name in ("pool", "test"):
                materialize_alignn_root(
                    manifest_rows=family_manifests[split_name],
                    structures_dir=structures_dir,
                    outdir=alignn_roots_dir / split_name,
                    link_mode=args.link_mode,
                )

    print(json.dumps(
        {
            "status": "ok",
            "split_source": split_source,
            "target_key_used": build_result.target_key_used,
            "oxide_counts": summary_for_family(
                build_result.oxide_rows,
                family_name="oxide",
                split_source=split_source,
                dataset_key=(args.dataset_key if args.dataset_json is None else str(args.dataset_json)),
                target_key=build_result.target_key_used,
            )["counts"],
            "nitride_counts": summary_for_family(
                build_result.nitride_rows,
                family_name="nitride",
                split_source=split_source,
                dataset_key=(args.dataset_key if args.dataset_json is None else str(args.dataset_json)),
                target_key=build_result.target_key_used,
            )["counts"],
            "outdir": str(args.outdir.resolve()),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
