#!/usr/bin/env python3
"""Recompute the public zero-shot family bootstrap from released predictions."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--oxide", type=Path, default=Path("results/zero_shot/oxide/predictions.csv"))
    parser.add_argument("--nitride", type=Path, default=Path("results/zero_shot/nitride/predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/reproduction/zero_shot_bootstrap.csv"))
    parser.add_argument("--replicates", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def errors(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8", newline="") as handle:
        values = np.asarray([float(row["abs_error"]) for row in csv.DictReader(handle)])
    if values.size == 0 or not np.isfinite(values).all():
        raise SystemExit(f"Invalid or empty absolute-error column: {path}")
    return values


def bootstrap_means(
    values: np.ndarray,
    rng: np.random.Generator,
    replicates: int,
    batch_size: int,
) -> np.ndarray:
    output = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        output[start:stop] = values[indices].mean(axis=1)
    return output


def summary(name: str, point: float, samples: np.ndarray) -> dict[str, object]:
    low, high = np.quantile(samples, [0.025, 0.975], method="linear")
    return {
        "estimand": name,
        "point_estimate": point,
        "bootstrap_mean": float(samples.mean()),
        "bootstrap_standard_error": float(samples.std(ddof=1)),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence_level": 0.95,
        "ci_method": "percentile",
    }


def main() -> None:
    args = parse_args()
    if args.replicates <= 0 or args.batch_size <= 0:
        raise SystemExit("--replicates and --batch-size must be positive")
    root = args.repo_root.resolve()
    oxide = errors(resolve(root, args.oxide))
    nitride = errors(resolve(root, args.nitride))

    oxide_seed, nitride_seed = np.random.SeedSequence(args.seed).spawn(2)
    oxide_samples = bootstrap_means(
        oxide, np.random.default_rng(oxide_seed), args.replicates, args.batch_size
    )
    nitride_samples = bootstrap_means(
        nitride, np.random.default_rng(nitride_seed), args.replicates, args.batch_size
    )
    rows = [
        summary("oxide_mae", float(oxide.mean()), oxide_samples),
        summary("nitride_mae", float(nitride.mean()), nitride_samples),
        summary(
            "nitride_to_oxide_mae_ratio",
            float(nitride.mean() / oxide.mean()),
            nitride_samples / oxide_samples,
        ),
    ]
    for row in rows:
        row.update(
            {
                "n_replicates": args.replicates,
                "master_seed": args.seed,
                "oxide_n": len(oxide),
                "nitride_n": len(nitride),
            }
        )

    output = resolve(root, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} bootstrap summaries to {output}")
    for row in rows:
        print(
            f"  {row['estimand']}: {row['point_estimate']:.6f} "
            f"[{row['ci_low']:.6f}, {row['ci_high']:.6f}]"
        )


if __name__ == "__main__":
    main()
