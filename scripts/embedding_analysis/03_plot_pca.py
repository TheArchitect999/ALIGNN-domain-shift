#!/usr/bin/env python
"""Standalone PCA analysis for extracted ALIGNN structure embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_DIR = "results/embeddings/coordinates/pca"
DEFAULT_FIGURE_DIR = "results/reproduction/embeddings/figures/pca"
DEFAULT_TABLE_DIR = "results/reproduction/embeddings/tables"
DEFAULT_REPORT_DIR = "results/reproduction/embeddings"
DEFAULT_SEED = 42
EMBEDDING_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
DATASETS = {
    "fixed_test_set": {
        "folder": "test_set",
        "label": "Fixed test set",
        "fit": False,
    },
    "balanced_pool_set": {
        "folder": "balanced_pool",
        "label": "Balanced train+val pool",
        "fit": True,
    },
    "oxide_reference_pool": {
        "folder": "oxide_reference_pool",
        "label": "Oxide reference train+val pool",
        "fit": False,
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
        description="Run standalone standardized PCA on extracted structure embeddings."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--embedding-root",
        default="results/embeddings/embeddings",
        help="Root containing test_set, balanced_pool, and oxide_reference_pool embeddings.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing PCA coordinate, table, and figure outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without fitting PCA.",
    )
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
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


def existing_outputs(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def planned_output_paths(
    root: Path,
    output_dir: Path,
    figure_dir: Path,
    table_dir: Path,
    report_dir: Path,
) -> list[Path]:
    paths: list[Path] = [
        output_dir / "pca_coordinates_all_sources.csv",
        output_dir / "pca_explained_variance.csv",
        output_dir / "pca_manifest.json",
        output_dir / "hard_easy_nitride_test_ids.csv",
        table_dir / "pca_explained_variance.csv",
        report_dir / "pca_notes.md",
    ]
    for source in EMBEDDING_SOURCES:
        paths.append(output_dir / f"{source}_pca_coordinates.csv")
        for stem in (
            "fixed_test_family",
            "balanced_pool_family",
            "nitride_test_abs_error",
            "hard_easy_nitrides_on_balanced_pool",
        ):
            paths.append(figure_dir / f"{source}_{stem}.png")
            paths.append(figure_dir / f"{source}_{stem}.pdf")
    return paths


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


def pct(value: float) -> float:
    return float(value) * 100.0


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def coordinate_rows(
    dataset: EmbeddingDataset,
    source: str,
    coords: np.ndarray,
    variance_ratio: np.ndarray,
    pca_fit_subset: str,
) -> list[dict[str, Any]]:
    df = source_metadata(dataset, source)
    rows: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append(
            {
                "embedding_source": source,
                "subset_name": dataset.subset_name,
                "pca_fit_subset": pca_fit_subset,
                "material_id": row["material_id"],
                "family": row["family"],
                "split": row["split"],
                "formula": row.get("formula", ""),
                "target_formation_energy_peratom": row.get(
                    "target_formation_energy_peratom", ""
                ),
                "pretrained_prediction": row.get("pretrained_prediction", ""),
                "absolute_error": row.get("absolute_error", ""),
                "pca1": float(coords[idx, 0]),
                "pca2": float(coords[idx, 1]),
                "pc1_explained_variance_pct": pct(variance_ratio[0]),
                "pc2_explained_variance_pct": pct(variance_ratio[1]),
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
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
        }
    )


def axis_labels(variance_ratio: np.ndarray) -> tuple[str, str]:
    return (
        f"PC1 ({pct(variance_ratio[0]):.1f}% variance)",
        f"PC2 ({pct(variance_ratio[1]):.1f}% variance)",
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path.as_posix())
    plt.close(fig)
    return paths


def plot_family(
    df: pd.DataFrame,
    variance_ratio: np.ndarray,
    title: str,
    figure_dir: Path,
    stem: str,
    alpha: float,
    size: float,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    for family in ("oxide", "nitride"):
        sub = df[df["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["pca1"],
            sub["pca2"],
            s=size,
            color=FAMILY_COLORS[family],
            alpha=alpha,
            linewidths=0,
            label=f"{family} (n={len(sub)})",
        )
    x_label, y_label = axis_labels(variance_ratio)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.4)
    return save_figure(fig, figure_dir, stem)


def plot_nitride_error(
    df: pd.DataFrame,
    variance_ratio: np.ndarray,
    title: str,
    figure_dir: Path,
    stem: str,
) -> list[str]:
    nitride = df[df["family"] == "nitride"].copy()
    nitride["absolute_error_float"] = numeric_series(nitride, "absolute_error")
    nitride = nitride.dropna(subset=["absolute_error_float"])
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    scatter = ax.scatter(
        nitride["pca1"],
        nitride["pca2"],
        c=nitride["absolute_error_float"],
        cmap="magma_r",
        s=34,
        alpha=0.88,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute zero-shot error (eV/atom)")
    x_label, y_label = axis_labels(variance_ratio)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    return save_figure(fig, figure_dir, stem)


def hard_easy_ids(fixed_df: pd.DataFrame, source: str) -> tuple[pd.DataFrame, int]:
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
    return groups, n_each


def plot_hard_easy_overlay(
    balanced_df: pd.DataFrame,
    groups: pd.DataFrame,
    variance_ratio: np.ndarray,
    title: str,
    figure_dir: Path,
    stem: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(5.9, 4.55))
    for family in ("oxide", "nitride"):
        sub = balanced_df[balanced_df["family"] == family]
        if sub.empty:
            continue
        ax.scatter(
            sub["pca1"],
            sub["pca2"],
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
            sub["pca1"],
            sub["pca2"],
            s=48,
            color=GROUP_COLORS[group],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.92,
            label=f"{label} (n={len(sub)})",
        )
    x_label, y_label = axis_labels(variance_ratio)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.05)
    return save_figure(fig, figure_dir, stem)


def write_notes(
    path: Path,
    variance_rows: list[dict[str, Any]],
    figure_paths: list[str],
    hard_easy_count: int,
) -> None:
    lines = [
        "# PCA Notes",
        "",
        "PCA is used here as a standalone linear baseline visualization and variance summary.",
        "",
        "## Method",
        "",
        "- Each embedding source is standardized dimension-wise with `StandardScaler` before PCA.",
        "- The scaler and PCA basis are fit on `balanced_pool_set` for each embedding source.",
        "- `fixed_test_set` and `oxide_reference_pool` are projected into the same fitted coordinate system.",
        "- PCA uses two components and a fixed seed of `42`; the full SVD solver is deterministic.",
        "- Axis labels in all figures report the explained variance percentages from the fitted balanced-pool PCA basis.",
        "",
        "## Hard/Easy Nitride Definition",
        "",
        f"- Hard nitrides are the top 20% of fixed-test nitrides by absolute zero-shot error, using `ceil(0.20 * n) = {hard_easy_count}` records.",
        f"- Easy nitrides are the bottom 20% by the same rule, also `{hard_easy_count}` records.",
        "",
        "## Explained Variance",
        "",
        "| Embedding source | PC1 % | PC2 % | Cumulative 2D % |",
        "|---|---:|---:|---:|",
    ]
    for row in variance_rows:
        lines.append(
            f"| `{row['embedding_source']}` | {row['pc1_explained_variance_pct']:.3f} | "
            f"{row['pc2_explained_variance_pct']:.3f} | "
            f"{row['cumulative_2d_explained_variance_pct']:.3f} |"
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
    table_dir = (root / args.table_dir).resolve()
    report_dir = (root / args.report_dir).resolve()
    embedding_root = (root / args.embedding_root).resolve()

    outputs = planned_output_paths(root, output_dir, figure_dir, table_dir, report_dir)
    if not args.allow_overwrite:
        existing = existing_outputs(outputs)
        if existing:
            joined = "\n".join(str(path) for path in existing)
            raise FileExistsError(
                "Refusing to overwrite existing PCA outputs. Use --allow-overwrite "
                f"to regenerate them:\n{joined}"
            )

    datasets = {
        name: load_dataset(root, embedding_root, name, info)
        for name, info in DATASETS.items()
    }
    for source in EMBEDDING_SOURCES:
        for dataset in datasets.values():
            source_metadata(dataset, source)

    if args.dry_run:
        print("Dry-run only. PCA inputs:")
        for name, dataset in datasets.items():
            print(f"{name}: {dataset.npz_path}")
            for source in EMBEDDING_SOURCES:
                print(f"  {source}: {dataset.arrays[source].shape}")
        print(f"Coordinates output: {relpath(output_dir, root)}")
        print(f"Figures output: {relpath(figure_dir, root)}")
        return

    setup_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    all_coordinate_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    hard_easy_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []
    hard_easy_count = 0

    for source in EMBEDDING_SOURCES:
        fit_dataset = datasets["balanced_pool_set"]
        fit_matrix = fit_dataset.arrays[source]
        scaler = StandardScaler()
        scaled_fit = scaler.fit_transform(fit_matrix)
        pca = PCA(n_components=2, svd_solver="full", random_state=args.seed)
        pca.fit(scaled_fit)
        variance_ratio = pca.explained_variance_ratio_

        source_rows: list[dict[str, Any]] = []
        coord_frames: dict[str, pd.DataFrame] = {}
        for subset_name, dataset in datasets.items():
            coords = pca.transform(scaler.transform(dataset.arrays[source]))
            rows = coordinate_rows(
                dataset=dataset,
                source=source,
                coords=coords,
                variance_ratio=variance_ratio,
                pca_fit_subset="balanced_pool_set",
            )
            source_rows.extend(rows)
            coord_frames[subset_name] = pd.DataFrame(rows)

        all_coordinate_rows.extend(source_rows)
        write_csv(
            output_dir / f"{source}_pca_coordinates.csv",
            source_rows,
            list(source_rows[0].keys()),
        )

        variance_row = {
            "embedding_source": source,
            "pca_fit_subset": "balanced_pool_set",
            "standardized_before_pca": True,
            "n_fit_structures": int(fit_matrix.shape[0]),
            "n_embedding_dimensions": int(fit_matrix.shape[1]),
            "pc1_explained_variance_ratio": float(variance_ratio[0]),
            "pc2_explained_variance_ratio": float(variance_ratio[1]),
            "pc1_explained_variance_pct": pct(variance_ratio[0]),
            "pc2_explained_variance_pct": pct(variance_ratio[1]),
            "cumulative_2d_explained_variance_pct": pct(float(np.sum(variance_ratio))),
            "pc1_explained_variance": float(pca.explained_variance_[0]),
            "pc2_explained_variance": float(pca.explained_variance_[1]),
            "pc1_singular_value": float(pca.singular_values_[0]),
            "pc2_singular_value": float(pca.singular_values_[1]),
        }
        variance_rows.append(variance_row)

        figure_paths.extend(
            relpath(Path(path), root)
            for path in plot_family(
                coord_frames["fixed_test_set"],
                variance_ratio,
                f"{source}: fixed test set",
                figure_dir,
                f"{source}_fixed_test_family",
                alpha=0.66,
                size=18,
            )
        )
        figure_paths.extend(
            relpath(Path(path), root)
            for path in plot_family(
                coord_frames["balanced_pool_set"],
                variance_ratio,
                f"{source}: balanced pool set",
                figure_dir,
                f"{source}_balanced_pool_family",
                alpha=0.42,
                size=12,
            )
        )
        figure_paths.extend(
            relpath(Path(path), root)
            for path in plot_nitride_error(
                coord_frames["fixed_test_set"],
                variance_ratio,
                f"{source}: fixed-test nitrides by zero-shot error",
                figure_dir,
                f"{source}_nitride_test_abs_error",
            )
        )
        groups, hard_easy_count = hard_easy_ids(coord_frames["fixed_test_set"], source)
        hard_easy_rows.extend(groups.to_dict(orient="records"))
        figure_paths.extend(
            relpath(Path(path), root)
            for path in plot_hard_easy_overlay(
                coord_frames["balanced_pool_set"],
                groups,
                variance_ratio,
                f"{source}: hard/easy fixed-test nitrides on balanced manifold",
                figure_dir,
                f"{source}_hard_easy_nitrides_on_balanced_pool",
            )
        )

        print(
            f"{source}: PC1={pct(variance_ratio[0]):.3f}%, "
            f"PC2={pct(variance_ratio[1]):.3f}%, "
            f"coordinates={len(source_rows)}"
        )

    write_csv(
        output_dir / "pca_coordinates_all_sources.csv",
        all_coordinate_rows,
        list(all_coordinate_rows[0].keys()),
    )
    write_csv(
        output_dir / "pca_explained_variance.csv",
        variance_rows,
        list(variance_rows[0].keys()),
    )
    write_csv(
        table_dir / "pca_explained_variance.csv",
        variance_rows,
        list(variance_rows[0].keys()),
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
        "pca1",
        "pca2",
        "embedding_row_index",
    ]
    write_csv(
        output_dir / "hard_easy_nitride_test_ids.csv",
        hard_easy_rows,
        hard_easy_fields,
    )
    manifest = {
        "created_at_unix": time.time(),
        "created_by": "scripts/embedding_analysis/03_plot_pca.py",
        "command": " ".join(sys.argv),
        "random_seed": args.seed,
        "method": {
            "standardization": "StandardScaler fit on balanced_pool_set per embedding source",
            "pca_fit_subset": "balanced_pool_set",
            "n_components": 2,
            "svd_solver": "full",
            "role": "standalone linear baseline visualization and variance summary",
        },
        "embedding_sources": list(EMBEDDING_SOURCES),
        "datasets": {
            name: {
                "npz": relpath(dataset.npz_path, root),
                "metadata": relpath(dataset.metadata_path, root),
                "n_structures": int(dataset.arrays["pre_head"].shape[0]),
            }
            for name, dataset in datasets.items()
        },
        "outputs": {
            "coordinates_dir": relpath(output_dir, root),
            "figure_dir": relpath(figure_dir, root),
            "explained_variance_csv": relpath(
                table_dir / "pca_explained_variance.csv", root
            ),
            "notes": relpath(report_dir / "pca_notes.md", root),
            "figures": figure_paths,
        },
    }
    write_json(output_dir / "pca_manifest.json", manifest)
    write_notes(report_dir / "pca_notes.md", variance_rows, figure_paths, hard_easy_count)

    print(f"Coordinates: {relpath(output_dir, root)}")
    print(f"Figures: {relpath(figure_dir, root)}")
    print(f"Explained variance table: {relpath(table_dir / 'pca_explained_variance.csv', root)}")
    print(f"Notes: {relpath(report_dir / 'pca_notes.md', root)}")


if __name__ == "__main__":
    main()
