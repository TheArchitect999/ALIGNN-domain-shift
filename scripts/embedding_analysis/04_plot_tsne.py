#!/usr/bin/env python
"""Direct t-SNE analysis for extracted ALIGNN structure embeddings."""

from __future__ import annotations

import argparse
import csv
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_DIR = "results/embeddings/coordinates/tsne"
DEFAULT_FIGURE_DIR = "results/reproduction/embeddings/figures/tsne"
DEFAULT_REPORT_DIR = "results/reproduction/embeddings"
DEFAULT_SEED = 42
EMBEDDING_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
PERPLEXITIES = (30, 15, 50)
MAIN_PERPLEXITY = 30
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
        description="Run direct standardized t-SNE on extracted structure embeddings."
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
        "--perplexities",
        type=int,
        nargs="+",
        default=list(PERPLEXITIES),
        help="Perplexities to run. Include 30 for the main run.",
    )
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing t-SNE coordinate, manifest, and figure outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned outputs without fitting t-SNE.",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def planned_output_paths(output_dir: Path, figure_dir: Path, report_dir: Path, perplexities: list[int]) -> list[Path]:
    paths: list[Path] = [
        output_dir / "tsne_coordinates_all_sources.csv",
        output_dir / "tsne_parameter_manifest.csv",
        output_dir / "tsne_manifest.json",
        output_dir / "hard_easy_nitride_test_ids.csv",
        report_dir / "tsne_notes.md",
    ]
    for source in EMBEDDING_SOURCES:
        paths.append(output_dir / f"{source}_tsne_coordinates.csv")
        for perplexity in perplexities:
            label = f"p{perplexity}"
            for run_name in (
                "fixed_test_set",
                "balanced_pool_set",
                "balanced_pool_plus_fixed_test_nitride",
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
            "Refusing to overwrite existing t-SNE outputs. Use --allow-overwrite "
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


def fit_direct_tsne(matrix: np.ndarray, perplexity: int, seed: int, max_iter: int) -> tuple[np.ndarray, dict[str, Any]]:
    if matrix.shape[0] <= perplexity:
        raise ValueError(
            f"t-SNE perplexity {perplexity} requires more than {perplexity} rows, got {matrix.shape[0]}."
        )
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    tsne = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        metric="euclidean",
        method="barnes_hut",
        angle=0.5,
        max_iter=max_iter,
    )
    coords = tsne.fit_transform(scaled)
    diagnostics = {
        "n_iter": int(getattr(tsne, "n_iter_", -1)),
        "kl_divergence": float(getattr(tsne, "kl_divergence_", np.nan)),
        "standardized_before_tsne": True,
        "pre_reduction": "none",
    }
    return coords, diagnostics


def metadata_rows(
    df: pd.DataFrame,
    coords: np.ndarray,
    source: str,
    run_name: str,
    perplexity: int,
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
                "perplexity": perplexity,
                "random_state": seed,
                "standardized_before_tsne": True,
                "pre_reduction": "none",
                "tsne_init": "pca",
                "metric": "euclidean",
                "learning_rate": "auto",
                "kl_divergence": diagnostics["kl_divergence"],
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
                "tsne1": float(coords[idx + row_offset, 0]),
                "tsne2": float(coords[idx + row_offset, 1]),
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
            sub["tsne1"],
            sub["tsne2"],
            s=size,
            color=FAMILY_COLORS[family],
            alpha=alpha,
            linewidths=0,
            label=f"{family} (n={len(sub)})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.4)
    return save_figure(fig, figure_dir, stem)


def plot_nitride_error(df: pd.DataFrame, title: str, figure_dir: Path, stem: str) -> list[Path]:
    nitride = df[df["family"] == "nitride"].copy()
    nitride["absolute_error_float"] = numeric_series(nitride, "absolute_error")
    nitride = nitride.dropna(subset=["absolute_error_float"])
    fig, ax = plt.subplots(figsize=(5.7, 4.4))
    scatter = ax.scatter(
        nitride["tsne1"],
        nitride["tsne2"],
        c=nitride["absolute_error_float"],
        cmap="magma_r",
        s=34,
        alpha=0.88,
        linewidths=0,
    )
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Absolute zero-shot error (eV/atom)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
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
            sub["tsne1"],
            sub["tsne2"],
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
            sub["tsne1"],
            sub["tsne2"],
            s=48,
            color=GROUP_COLORS[group],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.92,
            label=f"{label} (n={len(sub)})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.legend(frameon=False, loc="best", markerscale=1.05)
    return save_figure(fig, figure_dir, stem)


def write_notes(path: Path, parameter_rows: list[dict[str, Any]], figure_paths: list[str], hard_easy_count: int) -> None:
    main_rows = [
        row
        for row in parameter_rows
        if int(row["perplexity"]) == MAIN_PERPLEXITY
        and row["run_name"] in {"fixed_test_set", "balanced_pool_set"}
    ]
    lines = [
        "# t-SNE Notes",
        "",
        "This is the direct t-SNE analysis of standardized ALIGNN structure embeddings.",
        "",
        "## Method",
        "",
        "- Embedding dimensions are standardized with `StandardScaler` before every t-SNE fit.",
        "- The standardized embeddings are fed directly into t-SNE.",
        "- No PCA pre-reduction or other dimensionality reduction is used as preprocessing.",
        "- `init='pca'` is used only for t-SNE initialization, as allowed by the analysis spec.",
        "- Main run parameters: `n_components=2`, `perplexity=30`, `learning_rate='auto'`, `metric='euclidean'`, `random_state=42`.",
        "- Sensitivity appendix runs use perplexity `15` and `50` with the same direct-standardized input policy.",
        "",
        "## Interpretation",
        "",
        "t-SNE mainly preserves local neighborhood relationships. Distances between separated clusters, global geometry, and apparent gaps between far-apart groups should not be interpreted too strongly.",
        "",
        "## Overlay Fit",
        "",
        "t-SNE has no stable out-of-sample transform in this workflow. The hard/easy nitride overlay is therefore fit directly on `balanced_pool_set` plus fixed-test nitride structures for each embedding source and perplexity.",
        "",
        "## Hard/Easy Nitride Definition",
        "",
        f"- Hard nitrides are the top 20% of fixed-test nitrides by absolute zero-shot error, using `ceil(0.20 * n) = {hard_easy_count}` records.",
        f"- Easy nitrides are the bottom 20%, also `{hard_easy_count}` records.",
        "",
        "## Main Run Diagnostics",
        "",
        "| Embedding source | Run | Perplexity | Rows | KL divergence |",
        "|---|---|---:|---:|---:|",
    ]
    for row in main_rows:
        lines.append(
            f"| `{row['embedding_source']}` | `{row['run_name']}` | {row['perplexity']} | "
            f"{row['n_rows']} | {float(row['kl_divergence']):.5f} |"
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
    perplexities = [int(p) for p in args.perplexities]

    refuse_existing(
        planned_output_paths(output_dir, figure_dir, report_dir, perplexities),
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
        print("Dry-run only. t-SNE inputs:")
        for name, dataset in datasets.items():
            print(f"{name}: {dataset.npz_path}")
            for source in EMBEDDING_SOURCES:
                print(f"  {source}: {dataset.arrays[source].shape}")
        print(f"Perplexities: {perplexities}")
        print(f"Coordinates output: {relpath(output_dir, root)}")
        print(f"Figures output: {relpath(figure_dir, root)}")
        return

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

        for perplexity in perplexities:
            print(f"{source}: running direct t-SNE perplexity={perplexity} fixed_test_set")
            fixed_coords, fixed_diag = fit_direct_tsne(
                datasets["fixed_test_set"].arrays[source],
                perplexity,
                args.seed,
                args.max_iter,
            )
            fixed_rows = metadata_rows(
                fixed_meta,
                fixed_coords,
                source,
                "fixed_test_set",
                perplexity,
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
                    "perplexity": perplexity,
                    "n_rows": len(fixed_rows),
                    "n_dimensions": int(datasets["fixed_test_set"].arrays[source].shape[1]),
                    "standardized_before_tsne": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "learning_rate": "auto",
                    "init": "pca",
                    "metric": "euclidean",
                    "random_state": args.seed,
                    "max_iter": args.max_iter,
                    **fixed_diag,
                }
            )

            print(f"{source}: running direct t-SNE perplexity={perplexity} balanced_pool_set")
            balanced_coords, balanced_diag = fit_direct_tsne(
                datasets["balanced_pool_set"].arrays[source],
                perplexity,
                args.seed,
                args.max_iter,
            )
            balanced_rows = metadata_rows(
                balanced_meta,
                balanced_coords,
                source,
                "balanced_pool_set",
                perplexity,
                args.seed,
                balanced_diag,
            )
            balanced_df = pd.DataFrame(balanced_rows)
            source_rows.extend(balanced_rows)
            parameter_rows.append(
                {
                    "embedding_source": source,
                    "run_name": "balanced_pool_set",
                    "perplexity": perplexity,
                    "n_rows": len(balanced_rows),
                    "n_dimensions": int(datasets["balanced_pool_set"].arrays[source].shape[1]),
                    "standardized_before_tsne": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "learning_rate": "auto",
                    "init": "pca",
                    "metric": "euclidean",
                    "random_state": args.seed,
                    "max_iter": args.max_iter,
                    **balanced_diag,
                }
            )

            print(f"{source}: running direct t-SNE perplexity={perplexity} balanced overlay")
            nitride_mask = fixed_meta["family"] == "nitride"
            fixed_nitride_meta = fixed_meta[nitride_mask].copy().reset_index(drop=True)
            fixed_nitride_array = datasets["fixed_test_set"].arrays[source][nitride_mask.to_numpy()]
            overlay_matrix = np.vstack(
                [datasets["balanced_pool_set"].arrays[source], fixed_nitride_array]
            )
            overlay_coords, overlay_diag = fit_direct_tsne(
                overlay_matrix,
                perplexity,
                args.seed,
                args.max_iter,
            )
            overlay_balanced_rows = metadata_rows(
                balanced_meta,
                overlay_coords,
                source,
                "balanced_pool_plus_fixed_test_nitride",
                perplexity,
                args.seed,
                overlay_diag,
            )
            overlay_nitride_rows = metadata_rows(
                fixed_nitride_meta,
                overlay_coords,
                source,
                "balanced_pool_plus_fixed_test_nitride",
                perplexity,
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
                    "run_name": "balanced_pool_plus_fixed_test_nitride",
                    "perplexity": perplexity,
                    "n_rows": len(overlay_rows),
                    "n_dimensions": int(overlay_matrix.shape[1]),
                    "standardized_before_tsne": True,
                    "pre_reduction": "none",
                    "n_components": 2,
                    "learning_rate": "auto",
                    "init": "pca",
                    "metric": "euclidean",
                    "random_state": args.seed,
                    "max_iter": args.max_iter,
                    **overlay_diag,
                }
            )

            label = f"p{perplexity}"
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
                / f"{source}_balanced_pool_plus_fixed_test_nitride_{label}_coordinates.csv",
                overlay_rows,
                list(overlay_rows[0].keys()),
            )

            figure_paths.extend(
                relpath(path, root)
                for path in plot_family(
                    fixed_df,
                    f"{source}: fixed test set, t-SNE p={perplexity}",
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
                    f"{source}: balanced pool set, t-SNE p={perplexity}",
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
                    f"{source}: fixed-test nitrides by error, t-SNE p={perplexity}",
                    figure_dir,
                    f"{source}_nitride_test_abs_error_{label}",
                )
            )
            figure_paths.extend(
                relpath(path, root)
                for path in plot_overlay(
                    overlay_df,
                    f"{source}: hard/easy nitrides on balanced manifold, t-SNE p={perplexity}",
                    figure_dir,
                    f"{source}_hard_easy_nitrides_on_balanced_pool_{label}",
                )
            )
            print(
                f"{source} p={perplexity}: "
                f"fixed KL={fixed_diag['kl_divergence']:.5f}, "
                f"balanced KL={balanced_diag['kl_divergence']:.5f}, "
                f"overlay KL={overlay_diag['kl_divergence']:.5f}"
            )

        all_rows.extend(source_rows)
        write_csv(
            output_dir / f"{source}_tsne_coordinates.csv",
            source_rows,
            list(source_rows[0].keys()),
        )

    write_csv(
        output_dir / "tsne_coordinates_all_sources.csv",
        all_rows,
        list(all_rows[0].keys()),
    )
    write_csv(
        output_dir / "tsne_parameter_manifest.csv",
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
        output_dir / "tsne_manifest.json",
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/04_plot_tsne.py",
            "command": " ".join(sys.argv),
            "method": {
                "standardization": "StandardScaler fit per t-SNE run",
                "pre_reduction": "none",
                "init": "pca initialization only",
                "metric": "euclidean",
                "learning_rate": "auto",
                "n_components": 2,
                "random_state": args.seed,
                "perplexities": perplexities,
                "main_perplexity": MAIN_PERPLEXITY,
                "sensitivity_perplexities": [
                    p for p in perplexities if p != MAIN_PERPLEXITY
                ],
            },
            "run_contexts": [
                "fixed_test_set",
                "balanced_pool_set",
                "balanced_pool_plus_fixed_test_nitride",
            ],
            "embedding_sources": list(EMBEDDING_SOURCES),
            "parameter_manifest": relpath(output_dir / "tsne_parameter_manifest.csv", root),
            "coordinates": relpath(output_dir / "tsne_coordinates_all_sources.csv", root),
            "figures": figure_paths,
        },
    )
    write_notes(report_dir / "tsne_notes.md", parameter_rows, figure_paths, hard_easy_count)

    print(f"Coordinates: {relpath(output_dir, root)}")
    print(f"Figures: {relpath(figure_dir, root)}")
    print(f"Notes: {relpath(report_dir / 'tsne_notes.md', root)}")


if __name__ == "__main__":
    main()
