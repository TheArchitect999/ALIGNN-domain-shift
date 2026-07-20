#!/usr/bin/env python
"""Direct UMAP analysis for extracted ALIGNN structure embeddings."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_mpl_cache = Path("results/embeddings/cache/matplotlib").resolve()
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

_numba_cache = Path("results/embeddings/cache/numba").resolve()
_numba_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("NUMBA_CACHE_DIR", str(_numba_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_DIR = "results/embeddings/coordinates/umap"
DEFAULT_FIGURE_DIR = "results/reproduction/embeddings/figures/umap"
DEFAULT_REPORT_DIR = "results/reproduction/embeddings"
DEFAULT_SEED = 42
EMBEDDING_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
N_NEIGHBORS_VALUES = (30, 15, 50)
MAIN_N_NEIGHBORS = 30
DEFAULT_MIN_DIST = 0.1
DATASETS = {
    "fixed_test_set": {
        "folder": "test_set",
        "label": "Fixed test set",
    },
    "balanced_pool_set": {
        "folder": "balanced_pool",
        "label": "Balanced train+val pool",
    },
}
FAMILY_COLORS = {
    "oxide": "#D55E00",
    "nitride": "#0072B2",
}
GROUP_COLORS = {
    "easy_bottom_20pct": "#009E73",
    "hard_top_20pct": "#CC79A7",
}


@dataclass
class EmbeddingDataset:
    subset_name: str
    label: str
    npz_path: Path
    metadata_path: Path
    arrays: dict[str, np.ndarray]
    metadata: pd.DataFrame


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run direct standardized UMAP on extracted structure embeddings."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--embedding-root",
        default="results/embeddings/embeddings",
        help="Root containing test_set and balanced_pool embeddings.",
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        nargs="+",
        default=list(N_NEIGHBORS_VALUES),
        help="UMAP n_neighbors values to run. Include 30 for the main run.",
    )
    parser.add_argument("--min-dist", type=float, default=DEFAULT_MIN_DIST)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing UMAP coordinate, manifest, and figure outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without fitting UMAP.",
    )
    return parser.parse_args()


def import_umap() -> Any:
    try:
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: umap-learn is required for direct UMAP analysis. "
            "Install it in the active environment, then rerun this script."
        ) from exc
    return umap


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def planned_output_paths(
    output_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    n_neighbors_values: list[int],
) -> list[Path]:
    paths: list[Path] = [
        output_dir / "umap_coordinates_all_sources.csv",
        output_dir / "umap_parameter_manifest.csv",
        output_dir / "umap_manifest.json",
        output_dir / "hard_easy_nitride_test_ids.csv",
        report_dir / "umap_notes.md",
    ]
    for source in EMBEDDING_SOURCES:
        paths.append(output_dir / f"{source}_umap_coordinates.csv")
        for n_neighbors in n_neighbors_values:
            label = f"n{n_neighbors}"
            for run_name in (
                "fixed_test_set",
                "balanced_pool_set",
                "balanced_pool_transform_fixed_test_nitride",
            ):
                paths.append(output_dir / f"{source}_{run_name}_{label}_coordinates.csv")
            for stem in (
                "fixed_test_family",
                "balanced_pool_family",
                "nitride_test_abs_error",
                "hard_easy_nitrides_on_balanced_pool",
            ):
                paths.append(figure_dir / f"{source}_{stem}_{label}.png")
                paths.append(figure_dir / f"{source}_{stem}_{label}.pdf")
    return paths


def refuse_existing(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing UMAP outputs. Use --allow-overwrite "
            f"to regenerate them:\n{joined}"
        )


def load_dataset(root: Path, embedding_root: Path, subset_name: str, info: dict[str, Any]) -> EmbeddingDataset:
    folder = embedding_root / info["folder"]
    npz_path = folder / "structure_embeddings.npz"
    metadata_path = folder / "structure_embedding_metadata.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing embedding matrix file: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata file: {metadata_path}")
    with np.load(npz_path) as payload:
        arrays = {key: payload[key].astype(np.float64) for key in payload.files}
    metadata = pd.read_csv(metadata_path)
    metadata["subset_name"] = subset_name
    metadata["embedding_npz_path"] = relpath(npz_path, root)
    return EmbeddingDataset(
        subset_name=subset_name,
        label=info["label"],
        npz_path=npz_path,
        metadata_path=metadata_path,
        arrays=arrays,
        metadata=metadata,
    )


def source_metadata(dataset: EmbeddingDataset, source: str) -> pd.DataFrame:
    if source not in dataset.arrays:
        raise KeyError(f"{source} is missing from {dataset.npz_path}")
    df = dataset.metadata[dataset.metadata["embedding_source"] == source].copy()
    if df.empty:
        raise ValueError(f"No metadata rows for source {source} in {dataset.metadata_path}")
    df["embedding_index"] = df["embedding_index"].astype(int)
    df = df.sort_values("embedding_index").reset_index(drop=True)
    expected = np.arange(len(df))
    observed = df["embedding_index"].to_numpy()
    if not np.array_equal(observed, expected):
        raise ValueError(
            f"Embedding indices for {dataset.subset_name}/{source} are not contiguous."
        )
    if dataset.arrays[source].shape[0] != len(df):
        raise ValueError(
            f"Array/metadata row mismatch for {dataset.subset_name}/{source}: "
            f"{dataset.arrays[source].shape[0]} vs {len(df)}"
        )
    return df


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )


def make_umap_reducer(umap_module: Any, n_neighbors: int, min_dist: float, seed: int) -> Any:
    return umap_module.UMAP(
        n_components=2,
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )


def fit_direct_umap(
    matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    umap_module: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.shape[0] <= n_neighbors:
        raise ValueError(
            f"UMAP n_neighbors {n_neighbors} requires more than {n_neighbors} rows, got {matrix.shape[0]}."
        )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    reducer = make_umap_reducer(umap_module, n_neighbors, min_dist, seed)
    coords = reducer.fit_transform(scaled)
    diagnostics = {
        "standardized_before_umap": True,
        "pre_reduction": "none",
        "standardization_scope": "run_input",
        "umap_transform_mode": "fit_transform",
    }
    return np.asarray(coords, dtype=float), diagnostics


def fit_balanced_transform_fixed_nitrides(
    balanced_matrix: np.ndarray,
    fixed_nitride_matrix: np.ndarray,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    umap_module: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if balanced_matrix.shape[0] <= n_neighbors:
        raise ValueError(
            f"UMAP n_neighbors {n_neighbors} requires more than {n_neighbors} balanced rows, "
            f"got {balanced_matrix.shape[0]}."
        )
    scaler = StandardScaler()
    balanced_scaled = scaler.fit_transform(balanced_matrix)
    fixed_nitride_scaled = scaler.transform(fixed_nitride_matrix)
    reducer = make_umap_reducer(umap_module, n_neighbors, min_dist, seed)
    balanced_coords = reducer.fit_transform(balanced_scaled)
    fixed_nitride_coords = reducer.transform(fixed_nitride_scaled)
    coords = np.vstack([balanced_coords, fixed_nitride_coords])
    diagnostics = {
        "standardized_before_umap": True,
        "pre_reduction": "none",
        "standardization_scope": "balanced_pool_set",
        "umap_transform_mode": "fit_balanced_transform_fixed_test_nitride",
    }
    return np.asarray(coords, dtype=float), diagnostics


def metadata_rows(
    df: pd.DataFrame,
    coords: np.ndarray,
    source: str,
    run_name: str,
    n_neighbors: int,
    min_dist: float,
    seed: int,
    diagnostics: dict[str, Any],
    row_offset: int = 0,
    error_group_by_id: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    error_group_by_id = error_group_by_id or {}
    for idx, row in df.reset_index(drop=True).iterrows():
        material_id = row["material_id"]
        rows.append(
            {
                "embedding_source": source,
                "run_name": run_name,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "random_state": seed,
                "standardized_before_umap": True,
                "pre_reduction": "none",
                "metric": "euclidean",
                "standardization_scope": diagnostics["standardization_scope"],
                "umap_transform_mode": diagnostics["umap_transform_mode"],
                "subset_name": row.get("subset_name", ""),
                "material_id": material_id,
                "family": row["family"],
                "split": row["split"],
                "formula": row.get("formula", ""),
                "target_formation_energy_peratom": row.get(
                    "target_formation_energy_peratom", ""
                ),
                "pretrained_prediction": row.get("pretrained_prediction", ""),
                "absolute_error": row.get("absolute_error", ""),
                "error_group": error_group_by_id.get(material_id, ""),
                "umap1": float(coords[idx + row_offset, 0]),
                "umap2": float(coords[idx + row_offset, 1]),
                "embedding_npz_path": row["embedding_npz_path"],
                "embedding_npz_key": row.get("npz_key", source),
                "embedding_row_index": int(row["embedding_index"]),
                "embedding_dim": int(row["embedding_dim"]),
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
        )
    return rows


def hard_easy_groups(fixed_df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, dict[str, str], int]:
    nitride = fixed_df[fixed_df["family"] == "nitride"].copy()
    nitride["absolute_error_float"] = numeric_series(nitride, "absolute_error")
    nitride = nitride.dropna(subset=["absolute_error_float"])
    nitride = nitride.sort_values(
        ["absolute_error_float", "material_id"], ascending=[True, True]
    ).reset_index(drop=True)
    n_each = int(math.ceil(0.20 * len(nitride)))
    easy = nitride.head(n_each).copy()
    hard = nitride.tail(n_each).copy()
    easy["error_group"] = "easy_bottom_20pct"
    hard["error_group"] = "hard_top_20pct"
    groups = pd.concat([easy, hard], ignore_index=True)
    groups["embedding_source"] = source
    group_by_id = dict(zip(groups["material_id"], groups["error_group"]))
    return groups, group_by_id, n_each


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_family(
    df: pd.DataFrame,
    title: str,
    figure_dir: Path,
    stem: str,
    alpha: float,
    size: float,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    for family in ("oxide", "nitride"):
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=size,
            color=FAMILY_COLORS[family],
            alpha=alpha,
            linewidths=0,
            label=f"{family} (n={len(sub)})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.4)
    return save_figure(fig, figure_dir, stem)


def plot_nitride_error(df: pd.DataFrame, title: str, figure_dir: Path, stem: str) -> list[Path]:
    nitride = df[df["family"] == "nitride"].copy()
    nitride["absolute_error_float"] = numeric_series(nitride, "absolute_error")
    nitride = nitride.dropna(subset=["absolute_error_float"])
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    scatter = ax.scatter(
        nitride["umap1"],
        nitride["umap2"],
        c=nitride["absolute_error_float"],
        cmap="magma_r",
        s=34,
        alpha=0.88,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute zero-shot error (eV/atom)")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    return save_figure(fig, figure_dir, stem)


def plot_overlay(df: pd.DataFrame, title: str, figure_dir: Path, stem: str) -> list[Path]:
    balanced = df[df["subset_name"] == "balanced_pool_set"]
    groups = df[df["error_group"].isin(["easy_bottom_20pct", "hard_top_20pct"])]
    fig, ax = plt.subplots(figsize=(5.9, 4.55))
    for family in ("oxide", "nitride"):
        sub = balanced[balanced["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=12,
            color=FAMILY_COLORS[family],
            alpha=0.16,
            linewidths=0,
            label=f"balanced {family}",
        )
    for group, label in (
        ("easy_bottom_20pct", "easy nitride test, bottom 20%"),
        ("hard_top_20pct", "hard nitride test, top 20%"),
    ):
        sub = groups[groups["error_group"] == group]
        ax.scatter(
            sub["umap1"],
            sub["umap2"],
            s=48,
            color=GROUP_COLORS[group],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.92,
            label=f"{label} (n={len(sub)})",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.05)
    return save_figure(fig, figure_dir, stem)


def write_notes(
    path: Path,
    parameter_rows: list[dict[str, Any]],
    figure_paths: list[str],
    hard_easy_count: int,
) -> None:
    main_rows = [
        row
        for row in parameter_rows
        if int(row["n_neighbors"]) == MAIN_N_NEIGHBORS
        and row["run_name"] in {"fixed_test_set", "balanced_pool_set"}
    ]
    lines = [
        "# UMAP Notes",
        "",
        "This is the direct UMAP analysis of standardized ALIGNN structure embeddings.",
        "",
        "## Method",
        "",
        "- Embedding dimensions are standardized with `StandardScaler` before every UMAP fit.",
        "- The standardized embeddings are fed directly into UMAP.",
        "- No PCA pre-reduction or other dimensionality reduction is used as preprocessing.",
        "- Main run parameters: `n_components=2`, `n_neighbors=30`, `min_dist=0.1`, `metric='euclidean'`, `random_state=42`.",
        "- Sensitivity appendix runs use `n_neighbors=15` and `n_neighbors=50` with the same direct-standardized input policy.",
        "",
        "## Interpretation",
        "",
        "UMAP is used here as a descriptive nonlinear embedding view and not as proof of a causal mechanism.",
        "",
        "## Overlay Fit",
        "",
        "For the hard/easy nitride overlay, UMAP is fit on the standardized `balanced_pool_set` manifold. Fixed-test nitride embeddings are standardized with the balanced-pool scaler and transformed onto that fitted UMAP view.",
        "",
        "## Hard/Easy Nitride Definition",
        "",
        f"- Hard nitrides are the top 20% of fixed-test nitrides by absolute zero-shot error, using `ceil(0.20 * n) = {hard_easy_count}` records.",
        f"- Easy nitrides are the bottom 20%, also `{hard_easy_count}` records.",
        "",
        "## Main Run Diagnostics",
        "",
        "| Embedding source | Run | n_neighbors | min_dist | Rows | Transform mode |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in main_rows:
        lines.append(
            f"| `{row['embedding_source']}` | `{row['run_name']}` | {row['n_neighbors']} | "
            f"{float(row['min_dist']):.3f} | {row['n_rows']} | `{row['umap_transform_mode']}` |"
        )
    lines.extend(["", "## Figures", ""])
    for fig_path in sorted(figure_paths):
        lines.append(f"- `{fig_path}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    output_dir = (root / args.output_dir).resolve()
    figure_dir = (root / args.figure_dir).resolve()
    report_dir = (root / args.report_dir).resolve()
    embedding_root = (root / args.embedding_root).resolve()
    n_neighbors_values = [int(value) for value in args.n_neighbors]

    refuse_existing(
        planned_output_paths(output_dir, figure_dir, report_dir, n_neighbors_values),
        allow_overwrite=args.allow_overwrite,
    )

    datasets = {
        name: load_dataset(root, embedding_root, name, info)
        for name, info in DATASETS.items()
    }
    for source in EMBEDDING_SOURCES:
        for dataset in datasets.values():
            source_metadata(dataset, source)

    if args.dry_run:
        available = importlib.util.find_spec("umap") is not None
        print("Dry-run only. UMAP inputs:")
        for name, dataset in datasets.items():
            print(f"{name}: {dataset.npz_path}")
            for source in EMBEDDING_SOURCES:
                print(f"  {source}: {dataset.arrays[source].shape}")
        print(f"n_neighbors values: {n_neighbors_values}")
        print(f"min_dist: {args.min_dist}")
        print(f"UMAP dependency available: {available}")
        print(f"Coordinates output: {relpath(output_dir, root)}")
        print(f"Figures output: {relpath(figure_dir, root)}")
        return

    umap_module = import_umap()
    setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    hard_easy_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    hard_easy_count = 0

    for source in EMBEDDING_SOURCES:
        source_rows: list[dict[str, Any]] = []
        fixed_meta = source_metadata(datasets["fixed_test_set"], source)
        balanced_meta = source_metadata(datasets["balanced_pool_set"], source)
        groups, error_group_by_id, hard_easy_count = hard_easy_groups(fixed_meta, source)
        hard_easy_rows.extend(groups.to_dict(orient="records"))

        for n_neighbors in n_neighbors_values:
            print(f"{source}: running direct UMAP n_neighbors={n_neighbors} fixed_test_set")
            fixed_coords, fixed_diag = fit_direct_umap(
                datasets["fixed_test_set"].arrays[source],
                n_neighbors,
                args.min_dist,
                args.seed,
                umap_module,
            )
            fixed_rows = metadata_rows(
                fixed_meta,
                fixed_coords,
                source,
                "fixed_test_set",
                n_neighbors,
                args.min_dist,
                args.seed,
                fixed_diag,
                error_group_by_id=error_group_by_id,
            )
            fixed_df = pd.DataFrame(fixed_rows)
            source_rows.extend(fixed_rows)
            parameter_rows.append(
                {
                    "embedding_source": source,
                    "run_name": "fixed_test_set",
                    "n_neighbors": n_neighbors,
                    "min_dist": args.min_dist,
                    "n_rows": len(fixed_rows),
                    "n_dimensions": int(datasets["fixed_test_set"].arrays[source].shape[1]),
                    "standardized_before_umap": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "metric": "euclidean",
                    "random_state": args.seed,
                    **fixed_diag,
                }
            )

            print(f"{source}: running direct UMAP n_neighbors={n_neighbors} balanced_pool_set")
            balanced_coords, balanced_diag = fit_direct_umap(
                datasets["balanced_pool_set"].arrays[source],
                n_neighbors,
                args.min_dist,
                args.seed,
                umap_module,
            )
            balanced_rows = metadata_rows(
                balanced_meta,
                balanced_coords,
                source,
                "balanced_pool_set",
                n_neighbors,
                args.min_dist,
                args.seed,
                balanced_diag,
            )
            balanced_df = pd.DataFrame(balanced_rows)
            source_rows.extend(balanced_rows)
            parameter_rows.append(
                {
                    "embedding_source": source,
                    "run_name": "balanced_pool_set",
                    "n_neighbors": n_neighbors,
                    "min_dist": args.min_dist,
                    "n_rows": len(balanced_rows),
                    "n_dimensions": int(datasets["balanced_pool_set"].arrays[source].shape[1]),
                    "standardized_before_umap": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "metric": "euclidean",
                    "random_state": args.seed,
                    **balanced_diag,
                }
            )

            print(
                f"{source}: running UMAP balanced manifold transform n_neighbors={n_neighbors}"
            )
            nitride_mask = fixed_meta["family"] == "nitride"
            fixed_nitride_meta = fixed_meta[nitride_mask].copy().reset_index(drop=True)
            fixed_nitride_array = datasets["fixed_test_set"].arrays[source][nitride_mask.to_numpy()]
            overlay_coords, overlay_diag = fit_balanced_transform_fixed_nitrides(
                datasets["balanced_pool_set"].arrays[source],
                fixed_nitride_array,
                n_neighbors,
                args.min_dist,
                args.seed,
                umap_module,
            )
            overlay_balanced_rows = metadata_rows(
                balanced_meta,
                overlay_coords,
                source,
                "balanced_pool_transform_fixed_test_nitride",
                n_neighbors,
                args.min_dist,
                args.seed,
                overlay_diag,
            )
            overlay_nitride_rows = metadata_rows(
                fixed_nitride_meta,
                overlay_coords,
                source,
                "balanced_pool_transform_fixed_test_nitride",
                n_neighbors,
                args.min_dist,
                args.seed,
                overlay_diag,
                row_offset=len(balanced_meta),
                error_group_by_id=error_group_by_id,
            )
            overlay_rows = overlay_balanced_rows + overlay_nitride_rows
            overlay_df = pd.DataFrame(overlay_rows)
            source_rows.extend(overlay_rows)
            parameter_rows.append(
                {
                    "embedding_source": source,
                    "run_name": "balanced_pool_transform_fixed_test_nitride",
                    "n_neighbors": n_neighbors,
                    "min_dist": args.min_dist,
                    "n_rows": len(overlay_rows),
                    "n_dimensions": int(datasets["balanced_pool_set"].arrays[source].shape[1]),
                    "standardized_before_umap": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "metric": "euclidean",
                    "random_state": args.seed,
                    **overlay_diag,
                }
            )

            label = f"n{n_neighbors}"
            write_csv(
                output_dir / f"{source}_fixed_test_set_{label}_coordinates.csv",
                fixed_rows,
                list(fixed_rows[0].keys()),
            )
            write_csv(
                output_dir / f"{source}_balanced_pool_set_{label}_coordinates.csv",
                balanced_rows,
                list(balanced_rows[0].keys()),
            )
            write_csv(
                output_dir
                / f"{source}_balanced_pool_transform_fixed_test_nitride_{label}_coordinates.csv",
                overlay_rows,
                list(overlay_rows[0].keys()),
            )

            figure_paths.extend(
                relpath(path, root)
                for path in plot_family(
                    fixed_df,
                    f"{source}: fixed test set, UMAP n={n_neighbors}",
                    figure_dir,
                    f"{source}_fixed_test_family_{label}",
                    alpha=0.66,
                    size=18,
                )
            )
            figure_paths.extend(
                relpath(path, root)
                for path in plot_family(
                    balanced_df,
                    f"{source}: balanced pool set, UMAP n={n_neighbors}",
                    figure_dir,
                    f"{source}_balanced_pool_family_{label}",
                    alpha=0.42,
                    size=12,
                )
            )
            figure_paths.extend(
                relpath(path, root)
                for path in plot_nitride_error(
                    fixed_df,
                    f"{source}: fixed-test nitrides by error, UMAP n={n_neighbors}",
                    figure_dir,
                    f"{source}_nitride_test_abs_error_{label}",
                )
            )
            figure_paths.extend(
                relpath(path, root)
                for path in plot_overlay(
                    overlay_df,
                    f"{source}: hard/easy nitrides on balanced manifold, UMAP n={n_neighbors}",
                    figure_dir,
                    f"{source}_hard_easy_nitrides_on_balanced_pool_{label}",
                )
            )
            print(
                f"{source} n={n_neighbors}: fixed rows={len(fixed_rows)}, "
                f"balanced rows={len(balanced_rows)}, overlay rows={len(overlay_rows)}"
            )

        all_rows.extend(source_rows)
        write_csv(
            output_dir / f"{source}_umap_coordinates.csv",
            source_rows,
            list(source_rows[0].keys()),
        )

    write_csv(
        output_dir / "umap_coordinates_all_sources.csv",
        all_rows,
        list(all_rows[0].keys()),
    )
    write_csv(
        output_dir / "umap_parameter_manifest.csv",
        parameter_rows,
        list(parameter_rows[0].keys()),
    )
    hard_easy_fields = [
        "embedding_source",
        "material_id",
        "family",
        "split",
        "formula",
        "absolute_error",
        "absolute_error_float",
        "error_group",
        "embedding_index",
    ]
    write_csv(
        output_dir / "hard_easy_nitride_test_ids.csv",
        hard_easy_rows,
        hard_easy_fields,
    )
    write_json(
        output_dir / "umap_manifest.json",
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/05_plot_umap.py",
            "command": " ".join(sys.argv),
            "method": {
                "standardization": "StandardScaler fit per UMAP fit",
                "pre_reduction": "none",
                "metric": "euclidean",
                "min_dist": args.min_dist,
                "n_components": 2,
                "random_state": args.seed,
                "n_neighbors_values": n_neighbors_values,
                "main_n_neighbors": MAIN_N_NEIGHBORS,
                "sensitivity_n_neighbors": [
                    value for value in n_neighbors_values if value != MAIN_N_NEIGHBORS
                ],
                "overlay_policy": (
                    "fit UMAP on balanced_pool_set and transform fixed-test nitrides "
                    "after applying the balanced-pool StandardScaler"
                ),
            },
            "run_contexts": [
                "fixed_test_set",
                "balanced_pool_set",
                "balanced_pool_transform_fixed_test_nitride",
            ],
            "embedding_sources": list(EMBEDDING_SOURCES),
            "parameter_manifest": relpath(output_dir / "umap_parameter_manifest.csv", root),
            "coordinates": relpath(output_dir / "umap_coordinates_all_sources.csv", root),
            "figures": figure_paths,
        },
    )
    write_notes(report_dir / "umap_notes.md", parameter_rows, figure_paths, hard_easy_count)

    print(f"Coordinates: {relpath(output_dir, root)}")
    print(f"Figures: {relpath(figure_dir, root)}")
    print(f"Notes: {relpath(report_dir / 'umap_notes.md', root)}")


if __name__ == "__main__":
    main()
