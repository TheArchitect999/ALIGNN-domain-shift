#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from family_dataset_lib import materialize_alignn_root


def load_rows(path: Path):
    import csv

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Turn a stage-2 manifest into an ALIGNN-ready root_dir with id_prop.csv and structure links/copies."
    )
    parser.add_argument("--manifest", type=Path, required=True, help="Path to a stage-2 manifest CSV.")
    parser.add_argument("--structures-dir", type=Path, required=True, help="Directory containing canonical POSCAR files.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output root_dir for ALIGNN.")
    parser.add_argument("--link-mode", choices=["auto", "symlink", "copy"], default="auto")
    args = parser.parse_args()

    rows = load_rows(args.manifest)
    materialize_alignn_root(rows, args.structures_dir, args.outdir, link_mode=args.link_mode)
    print(f"[OK] materialized {len(rows)} rows to {args.outdir}")


if __name__ == "__main__":
    main()
