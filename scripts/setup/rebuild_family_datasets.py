#!/usr/bin/env python3
"""Rebuild and validate the oxide/nitride family datasets from JARVIS-DFT.

The committed official split manifest is mandatory by default.  This wrapper
does not silently generate a replacement split, because doing so would change
the fixed-test evidence reported by the study.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLITS = ROOT / "data/manifests/dft_3d_formation_energy_peratom_splits.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset-key", default="dft_3d_2021")
    source.add_argument("--dataset-json", type=Path)
    parser.add_argument("--splits-file", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/jarvis")
    parser.add_argument("--outdir", type=Path, default=ROOT / "data")
    parser.add_argument("--link-mode", choices=("auto", "symlink", "copy"), default="auto")
    parser.add_argument("--no-write-structures", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    builder = ROOT / "scripts/dataset/build_family_datasets.py"
    validator = ROOT / "scripts/dataset/validate_family_datasets.py"
    if not args.splits_file.is_file():
        raise SystemExit(f"Official split manifest not found: {args.splits_file}")

    command = [
        sys.executable, str(builder),
        "--cache-dir", str(args.cache_dir),
        "--outdir", str(args.outdir),
        "--splits-file", str(args.splits_file),
        "--materialize-pool-and-test",
        "--link-mode", args.link_mode,
    ]
    if args.dataset_json:
        command.extend(("--dataset-json", str(args.dataset_json)))
    else:
        command.extend(("--dataset-key", args.dataset_key))
    if args.no_write_structures:
        command.append("--no-write-structures")
    run(command)

    if not args.skip_validation:
        run([sys.executable, str(validator), "--root", str(args.outdir)])


if __name__ == "__main__":
    main()

