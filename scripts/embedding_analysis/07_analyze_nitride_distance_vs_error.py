#!/usr/bin/env python
"""Test nitride error against distance from the oxide reference manifold."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


DEFAULT_EMBEDDING_ROOT = "results/embeddings/embeddings"
DEFAULT_STATS_CSV = "results/reproduction/embeddings/tables/nitride_distance_error_stats.csv"
DEFAULT_DISTANCE_CSV = "results/reproduction/embeddings/tables/nitride_distance_error_by_structure.csv"
DEFAULT_FIGURE_DIR = "results/reproduction/embeddings/figures/distance_vs_error"
DEFAULT_SUMMARY_MD = "results/reproduction/embeddings/domain_shift_hypothesis_test.md"
DEFAULT_MANIFEST = "results/embeddings/manifests/nitride_distance_error_manifest.json"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAPS = 5000
DEFAULT_PERMUTATIONS = 10000
DEFAULT_K_OXIDE_NEIGHBORS = 5
EMBEDDING_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
DISTANCE_METRIC_LABELS = {
    "oxide_centroid_distance": "Oxide centroid distance",
    "oxide_knn5_mean_distance": "Mean 5NN oxide distance",
    "oxide_mahalanobis_lw_distance": "Mahalanobis distance",
}
PRIMARY_DISTANCE_METRICS = ("oxide_centroid_distance", "oxide_knn5_mean_distance")
GROUP_COLORS = {
    "easy_bottom_20pct": "#009E73",
    "hard_top_20pct": "#CC79A7",
}


@dataclass
class EmbeddingDataset:
    subset_name: str
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
        description="Analyze whether poorly predicted nitrides are farther from oxide reference embeddings."
    )
    parser.add_argument("--embedding-root", default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--stats-csv", default=DEFAULT_STATS_CSV)
    parser.add_argument("--distance-csv", default=DEFAULT_DISTANCE_CSV)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--permutation-iterations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--k-oxide-neighbors", type=int, default=DEFAULT_K_OXIDE_NEIGHBORS)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing hypothesis-test tables, figures, and manifest.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned tests without computing outputs.",
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


def planned_figure_paths(figure_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for source in EMBEDDING_SOURCES:
        for metric in DISTANCE_METRIC_LABELS:
            for stem in (
                f"{source}_{metric}_vs_abs_error",
                f"{source}_{metric}_hard_easy_boxplot",
            ):
                paths.append(figure_dir / f"{stem}.png")
                paths.append(figure_dir / f"{stem}.pdf")
    return paths


def refuse_existing(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing distance-vs-error outputs. "
            f"Use --allow-overwrite to regenerate them:\n{joined}"
        )


def load_dataset(root: Path, embedding_root: Path, subset_name: str) -> EmbeddingDataset:
    folder = embedding_root / subset_name
    npz_path = folder / "structure_embeddings.npz"
    metadata_path = folder / "structure_embedding_metadata.csv"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing embedding matrix file: {npz_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing embedding metadata file: {metadata_path}")
    with np.load(npz_path) as payload:
        arrays = {key: payload[key].astype(np.float64) for key in payload.files}
    metadata = pd.read_csv(metadata_path)
    metadata["embedding_npz_path"] = relpath(npz_path, root)
    return EmbeddingDataset(
        subset_name=subset_name,
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


def nitride_test_view(test_set: EmbeddingDataset, source: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    metadata = source_metadata(test_set, source)
    mask = metadata["family"].eq("nitride").to_numpy()
    nitride_meta = metadata[mask].copy().reset_index(drop=True)
    nitride_vectors = test_set.arrays[source][mask]
    nitride_meta["absolute_error_float"] = pd.to_numeric(
        nitride_meta["absolute_error"], errors="coerce"
    )
    valid = nitride_meta["absolute_error_float"].notna().to_numpy()
    nitride_meta = nitride_meta[valid].copy().reset_index(drop=True)
    nitride_vectors = nitride_vectors[valid]
    errors = nitride_meta["absolute_error_float"].to_numpy(dtype=float)
    if len(nitride_meta) == 0:
        raise ValueError(f"No fixed-test nitrides with absolute_error for {source}.")
    return nitride_meta, nitride_vectors, errors


def oxide_reference_view(oxide_reference: EmbeddingDataset, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    metadata = source_metadata(oxide_reference, source)
    oxide_meta = metadata[metadata["family"].eq("oxide")].copy().reset_index(drop=True)
    oxide_vectors = oxide_reference.arrays[source][oxide_meta["embedding_index"].to_numpy()]
    if len(oxide_meta) == 0:
        raise ValueError(f"No oxide reference records for {source}.")
    return oxide_meta, oxide_vectors


def hard_easy_labels(errors: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.lexsort((np.arange(len(errors)), errors))
    n_each = int(math.ceil(0.20 * len(errors)))
    labels = np.full(len(errors), "middle_60pct", dtype=object)
    labels[order[:n_each]] = "easy_bottom_20pct"
    labels[order[-n_each:]] = "hard_top_20pct"
    return labels, n_each


def compute_mahalanobis_if_stable(
    oxide_vectors: np.ndarray,
    nitride_vectors: np.ndarray,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    n_reference, dim = oxide_vectors.shape
    diagnostics: dict[str, Any] = {
        "mahalanobis_status": "not_attempted",
        "mahalanobis_condition_number": math.nan,
        "mahalanobis_shrinkage": math.nan,
    }
    if n_reference < 5 * dim:
        diagnostics["mahalanobis_status"] = "skipped_reference_count_lt_5x_dim"
        return None, diagnostics
    try:
        estimator = LedoitWolf().fit(oxide_vectors)
        covariance = estimator.covariance_
        condition_number = float(np.linalg.cond(covariance))
        diagnostics["mahalanobis_condition_number"] = condition_number
        diagnostics["mahalanobis_shrinkage"] = float(estimator.shrinkage_)
        if not np.isfinite(condition_number) or condition_number > 1.0e12:
            diagnostics["mahalanobis_status"] = "skipped_unstable_condition_number"
            return None, diagnostics
        diff = nitride_vectors - estimator.location_
        squared = np.einsum("ij,jk,ik->i", diff, estimator.precision_, diff)
        distances = np.sqrt(np.maximum(squared, 0.0))
        if not np.all(np.isfinite(distances)):
            diagnostics["mahalanobis_status"] = "skipped_nonfinite_distances"
            return None, diagnostics
        diagnostics["mahalanobis_status"] = "computed_ledoit_wolf_stable"
        return distances, diagnostics
    except Exception as exc:  # pragma: no cover - defensive diagnostic path
        diagnostics["mahalanobis_status"] = f"skipped_exception_{type(exc).__name__}"
        return None, diagnostics


def compute_distances(
    oxide_vectors: np.ndarray,
    nitride_vectors: np.ndarray,
    k_neighbors: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    centroid = oxide_vectors.mean(axis=0)
    distances = {
        "oxide_centroid_distance": np.linalg.norm(nitride_vectors - centroid, axis=1),
    }
    if k_neighbors >= len(oxide_vectors):
        raise ValueError(
            f"k_neighbors must be less than oxide reference count: {k_neighbors} >= {len(oxide_vectors)}"
        )
    neighbors = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
    neighbors.fit(oxide_vectors)
    knn_distances, _ = neighbors.kneighbors(nitride_vectors, return_distance=True)
    distances[f"oxide_knn{k_neighbors}_mean_distance"] = knn_distances.mean(axis=1)
    mahalanobis, diagnostics = compute_mahalanobis_if_stable(oxide_vectors, nitride_vectors)
    if mahalanobis is not None:
        distances["oxide_mahalanobis_lw_distance"] = mahalanobis
    return distances, diagnostics


def corr_value(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def centered_for_corr(values: np.ndarray, method: str) -> np.ndarray:
    if method == "spearman":
        values = rankdata(values, method="average")
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    return centered


def fast_corr_from_centered(x_centered: np.ndarray, y_centered: np.ndarray) -> float:
    denom = math.sqrt(float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denom == 0:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denom)


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    if iterations <= 0:
        return math.nan, math.nan
    values: list[float] = []
    n = len(x)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        value = corr_value(x[idx], y[idx], method)
        if np.isfinite(value):
            values.append(value)
    return percentile_ci(values)


def permutation_corr_p_value(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
    iterations: int,
    alternative: str = "greater",
) -> float:
    observed = corr_value(x, y, method)
    x_centered = centered_for_corr(x, method)
    y_centered = centered_for_corr(y, method)
    count = 0
    for _ in range(iterations):
        permuted = rng.permutation(y_centered)
        value = fast_corr_from_centered(x_centered, permuted)
        if alternative == "greater":
            count += int(value >= observed)
        elif alternative == "two-sided":
            count += int(abs(value) >= abs(observed))
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    return float((count + 1) / (iterations + 1))


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    low, high = np.percentile(arr, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_group_diff_ci(
    easy_values: np.ndarray,
    hard_values: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
    iterations: int,
) -> tuple[float, float]:
    if iterations <= 0:
        return math.nan, math.nan
    values: list[float] = []
    for _ in range(iterations):
        easy_sample = rng.choice(easy_values, size=len(easy_values), replace=True)
        hard_sample = rng.choice(hard_values, size=len(hard_values), replace=True)
        if statistic == "mean":
            values.append(float(np.mean(hard_sample) - np.mean(easy_sample)))
        elif statistic == "median":
            values.append(float(np.median(hard_sample) - np.median(easy_sample)))
        else:
            raise ValueError(f"Unknown group statistic: {statistic}")
    return percentile_ci(values)


def permutation_group_diff_p_value(
    easy_values: np.ndarray,
    hard_values: np.ndarray,
    observed: float,
    statistic: str,
    rng: np.random.Generator,
    iterations: int,
    alternative: str = "greater",
) -> float:
    combined = np.concatenate([easy_values, hard_values])
    n_easy = len(easy_values)
    count = 0
    for _ in range(iterations):
        shuffled = rng.permutation(combined)
        easy_perm = shuffled[:n_easy]
        hard_perm = shuffled[n_easy:]
        if statistic == "mean":
            value = float(np.mean(hard_perm) - np.mean(easy_perm))
        elif statistic == "median":
            value = float(np.median(hard_perm) - np.median(easy_perm))
        else:
            raise ValueError(f"Unknown group statistic: {statistic}")
        if alternative == "greater":
            count += int(value >= observed)
        elif alternative == "two-sided":
            count += int(abs(value) >= abs(observed))
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
    return float((count + 1) / (iterations + 1))


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return adjusted.tolist()
    finite_indices = np.flatnonzero(finite)
    finite_p = p[finite]
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    m = len(ranked)
    adjusted_ranked = np.empty(m, dtype=float)
    running = 1.0
    for i in range(m - 1, -1, -1):
        running = min(running, ranked[i] * m / (i + 1))
        adjusted_ranked[i] = running
    adjusted_values = np.empty(m, dtype=float)
    adjusted_values[order] = np.minimum(adjusted_ranked, 1.0)
    adjusted[finite_indices] = adjusted_values
    return adjusted.tolist()


def add_corr_rows(
    rows: list[dict[str, Any]],
    source: str,
    distance_metric: str,
    distances: np.ndarray,
    errors: np.ndarray,
    n_oxide_reference: int,
    mahalanobis_diagnostics: dict[str, Any],
    rng: np.random.Generator,
    bootstrap_iterations: int,
    permutation_iterations: int,
    k_neighbors: int,
    hard_count: int,
    easy_count: int,
) -> None:
    for method, label in (("spearman", "spearman_correlation"), ("pearson", "pearson_correlation")):
        value = corr_value(distances, errors, method)
        ci_low, ci_high = bootstrap_corr_ci(
            distances, errors, method, rng, bootstrap_iterations
        )
        p_value = permutation_corr_p_value(
            distances, errors, method, rng, permutation_iterations, alternative="greater"
        )
        rows.append(
            base_stat_row(
                source,
                distance_metric,
                label,
                "nitride_abs_error_vs_distance",
                value,
                ci_low,
                ci_high,
                p_value,
                "permutation_pairing_errors_to_distances",
                "greater",
                n_oxide_reference,
                len(errors),
                k_neighbors,
                hard_count,
                easy_count,
                mahalanobis_diagnostics,
            )
        )


def add_group_rows(
    rows: list[dict[str, Any]],
    source: str,
    distance_metric: str,
    distances: np.ndarray,
    groups: np.ndarray,
    n_oxide_reference: int,
    mahalanobis_diagnostics: dict[str, Any],
    rng: np.random.Generator,
    bootstrap_iterations: int,
    permutation_iterations: int,
    k_neighbors: int,
    hard_count: int,
    easy_count: int,
) -> None:
    easy_values = distances[groups == "easy_bottom_20pct"]
    hard_values = distances[groups == "hard_top_20pct"]
    for statistic, row_name in (
        ("mean", "hard_minus_easy_mean_distance"),
        ("median", "hard_minus_easy_median_distance"),
    ):
        if statistic == "mean":
            value = float(np.mean(hard_values) - np.mean(easy_values))
            hard_summary = float(np.mean(hard_values))
            easy_summary = float(np.mean(easy_values))
        else:
            value = float(np.median(hard_values) - np.median(easy_values))
            hard_summary = float(np.median(hard_values))
            easy_summary = float(np.median(easy_values))
        ci_low, ci_high = bootstrap_group_diff_ci(
            easy_values, hard_values, statistic, rng, bootstrap_iterations
        )
        p_value = permutation_group_diff_p_value(
            easy_values,
            hard_values,
            value,
            statistic,
            rng,
            permutation_iterations,
            alternative="greater",
        )
        row = base_stat_row(
            source,
            distance_metric,
            row_name,
            "hard_top_20pct_vs_easy_bottom_20pct",
            value,
            ci_low,
            ci_high,
            p_value,
            "permutation_group_labels_within_hard_easy_union",
            "greater",
            n_oxide_reference,
            len(distances),
            k_neighbors,
            hard_count,
            easy_count,
            mahalanobis_diagnostics,
        )
        row["hard_group_distance_summary"] = hard_summary
        row["easy_group_distance_summary"] = easy_summary
        rows.append(row)


def base_stat_row(
    source: str,
    distance_metric: str,
    statistic: str,
    comparison: str,
    value: float,
    ci_low: float,
    ci_high: float,
    p_value: float,
    p_value_method: str,
    alternative: str,
    n_oxide_reference: int,
    n_nitrides: int,
    k_neighbors: int,
    hard_count: int,
    easy_count: int,
    mahalanobis_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "embedding_source": source,
        "distance_metric": distance_metric,
        "distance_label": DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric),
        "statistic": statistic,
        "comparison": comparison,
        "value": value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_level": 0.95,
        "p_value": p_value,
        "p_value_bh_fdr_within_statistic": math.nan,
        "p_value_method": p_value_method,
        "test_alternative": alternative,
        "n_nitrides": n_nitrides,
        "n_oxide_reference": n_oxide_reference,
        "hard_nitrides_count": hard_count,
        "easy_nitrides_count": easy_count,
        "k_oxide_neighbors": k_neighbors if "knn" in distance_metric else "",
        "analysis_space": "raw_256d_embedding_vectors",
        "error_metric": "absolute_zero_shot_error",
        "hard_group_distance_summary": "",
        "easy_group_distance_summary": "",
        "mahalanobis_status": mahalanobis_diagnostics.get("mahalanobis_status", ""),
        "mahalanobis_condition_number": mahalanobis_diagnostics.get(
            "mahalanobis_condition_number", math.nan
        ),
        "mahalanobis_shrinkage": mahalanobis_diagnostics.get(
            "mahalanobis_shrinkage", math.nan
        ),
    }


def add_fdr_adjustment(rows: list[dict[str, Any]]) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    for statistic in sorted(df["statistic"].unique()):
        indices = df.index[df["statistic"] == statistic].tolist()
        adjusted = benjamini_hochberg(df.loc[indices, "p_value"].astype(float).tolist())
        for idx, value in zip(indices, adjusted):
            rows[idx]["p_value_bh_fdr_within_statistic"] = value


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


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[Path]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for ext in ("png", "pdf"):
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_scatter(
    df: pd.DataFrame,
    source: str,
    distance_metric: str,
    figure_dir: Path,
) -> list[Path]:
    x = df[distance_metric].to_numpy(dtype=float)
    y = df["absolute_error_float"].to_numpy(dtype=float)
    groups = df["error_group"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.7, 4.35))
    middle = groups == "middle_60pct"
    ax.scatter(
        x[middle],
        y[middle],
        s=26,
        color="#6A6A6A",
        alpha=0.48,
        linewidths=0,
        label=f"middle 60% (n={int(np.sum(middle))})",
    )
    for group, label in (
        ("easy_bottom_20pct", "easy bottom 20%"),
        ("hard_top_20pct", "hard top 20%"),
    ):
        mask = groups == group
        ax.scatter(
            x[mask],
            y[mask],
            s=42,
            color=GROUP_COLORS[group],
            edgecolors="black",
            linewidths=0.35,
            alpha=0.88,
            label=f"{label} (n={int(np.sum(mask))})",
        )
    if len(x) >= 2 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
        ax.plot(x_line, slope * x_line + intercept, color="black", linewidth=1.0, alpha=0.7)
    spearman = float(spearmanr(x, y).statistic)
    pearson = float(pearsonr(x, y).statistic)
    ax.text(
        0.03,
        0.97,
        f"Spearman rho = {spearman:.3f}\nPearson r = {pearson:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86},
    )
    ax.set_xlabel(DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric))
    ax.set_ylabel("Nitride absolute zero-shot error (eV/atom)")
    ax.set_title(f"{source}: nitride error vs oxide distance")
    ax.legend(frameon=False, loc="best")
    return save_figure(fig, figure_dir, f"{source}_{distance_metric}_vs_abs_error")


def plot_boxplot(
    df: pd.DataFrame,
    source: str,
    distance_metric: str,
    figure_dir: Path,
    seed: int,
) -> list[Path]:
    easy = df[df["error_group"] == "easy_bottom_20pct"][distance_metric].to_numpy(dtype=float)
    hard = df[df["error_group"] == "hard_top_20pct"][distance_metric].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(4.9, 4.25))
    bp = ax.boxplot(
        [easy, hard],
        tick_labels=["Easy\nbottom 20%", "Hard\ntop 20%"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"linewidth": 0.9},
        whiskerprops={"linewidth": 0.9},
        capprops={"linewidth": 0.9},
    )
    for patch, color in zip(
        bp["boxes"],
        [GROUP_COLORS["easy_bottom_20pct"], GROUP_COLORS["hard_top_20pct"]],
    ):
        patch.set_facecolor(color)
        patch.set_alpha(0.58)
    for position, values, color in (
        (1, easy, GROUP_COLORS["easy_bottom_20pct"]),
        (2, hard, GROUP_COLORS["hard_top_20pct"]),
    ):
        jitter = rng.normal(0.0, 0.045, size=len(values))
        ax.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            color=color,
            edgecolors="black",
            linewidths=0.25,
            alpha=0.68,
        )
    mean_diff = float(np.mean(hard) - np.mean(easy))
    ax.text(
        0.03,
        0.97,
        f"Mean hard - easy = {mean_diff:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.86},
    )
    ax.set_ylabel(DISTANCE_METRIC_LABELS.get(distance_metric, distance_metric))
    ax.set_title(f"{source}: hard vs easy nitride distance")
    return save_figure(fig, figure_dir, f"{source}_{distance_metric}_hard_easy_boxplot")


def distance_rows_for_source(
    source: str,
    nitride_meta: pd.DataFrame,
    distances: dict[str, np.ndarray],
    error_groups: np.ndarray,
    oxide_reference_count: int,
    k_neighbors: int,
    mahalanobis_diagnostics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, row in nitride_meta.reset_index(drop=True).iterrows():
        record = {
            "embedding_source": source,
            "material_id": row["material_id"],
            "family": row["family"],
            "split": row["split"],
            "formula": row.get("formula", ""),
            "absolute_error": row.get("absolute_error", ""),
            "absolute_error_float": float(row["absolute_error_float"]),
            "error_group": error_groups[idx],
            "embedding_row_index": int(row["embedding_index"]),
            "embedding_dim": int(row["embedding_dim"]),
            "n_oxide_reference": oxide_reference_count,
            "k_oxide_neighbors": k_neighbors,
            "mahalanobis_status": mahalanobis_diagnostics.get("mahalanobis_status", ""),
            "mahalanobis_condition_number": mahalanobis_diagnostics.get(
                "mahalanobis_condition_number", math.nan
            ),
            "mahalanobis_shrinkage": mahalanobis_diagnostics.get(
                "mahalanobis_shrinkage", math.nan
            ),
        }
        for metric, values in distances.items():
            record[metric] = float(values[idx])
        rows.append(record)
    return rows


def format_effect(value: float, low: float, high: float, p_adj: float) -> str:
    if not np.isfinite(value):
        return "NA"
    return f"{value:.3f} [{low:.3f}, {high:.3f}], q={p_adj:.4f}"


def evidence_statement(stats_df: pd.DataFrame) -> str:
    required = stats_df[
        stats_df["distance_metric"].isin(PRIMARY_DISTANCE_METRICS)
        & stats_df["statistic"].isin(["spearman_correlation", "hard_minus_easy_mean_distance"])
    ].copy()
    supportive = (
        (required["value"].astype(float) > 0)
        & (required["p_value_bh_fdr_within_statistic"].astype(float) < 0.05)
    )
    if len(required) > 0 and supportive.all():
        return (
            "The raw-space results support the project domain-shift interpretation: "
            "nitrides with larger zero-shot errors are consistently farther from the oxide "
            "reference region across the required centroid and kNN distance measures."
        )
    if supportive.any():
        return (
            "The raw-space results partially support the project domain-shift interpretation: "
            "some distance/error tests are positive after FDR adjustment, but the evidence is not uniform."
        )
    return (
        "The raw-space results do not support the project domain-shift interpretation under these tests: "
        "distance from the oxide reference region is not reliably larger for poorly predicted nitrides."
    )


def write_summary(
    path: Path,
    stats_rows: list[dict[str, Any]],
    figure_paths: list[str],
    seed: int,
    bootstrap_iterations: int,
    permutation_iterations: int,
    k_neighbors: int,
) -> None:
    stats_df = pd.DataFrame(stats_rows)
    lines = [
        "# Domain-Shift Hypothesis Test",
        "",
        "Hypothesis: poorly predicted fixed-test nitrides lie farther from the oxide reference manifold in pretrained ALIGNN representation space.",
        "",
        "## Method",
        "",
        "- Distances are computed on the original extracted 256D structure embeddings.",
        "- No PCA, t-SNE, UMAP, or other projection is used for the statistical tests.",
        "- Oxide reference manifold proxy: `oxide_reference_pool` embeddings.",
        "- Test structures: fixed-test nitrides with absolute zero-shot error metadata.",
        f"- Required distances: Euclidean distance to the oxide reference centroid and mean Euclidean distance to the k={k_neighbors} nearest oxide reference embeddings.",
        "- Ledoit-Wolf Mahalanobis distance is included only where the oxide covariance estimate passes the stability gate.",
        "- Hard nitrides are the top 20% by absolute zero-shot error; easy nitrides are the bottom 20%.",
        f"- Bootstrap confidence intervals use {bootstrap_iterations} iterations with seed {seed}.",
        f"- Permutation tests use {permutation_iterations} iterations and the directional alternative that larger distances correspond to larger errors or harder nitrides.",
        "",
        "## Interpretation",
        "",
        evidence_statement(stats_df),
        "",
        "This evidence is associative. It does not prove that distance from the oxide reference region causes prediction error.",
        "",
        "## Primary Raw-Space Results",
        "",
        "| Embedding level | Distance | Spearman rho | Pearson r | Hard - easy mean distance |",
        "|---|---|---:|---:|---:|",
    ]
    for source in EMBEDDING_SOURCES:
        for metric in PRIMARY_DISTANCE_METRICS:
            sub = stats_df[
                (stats_df["embedding_source"] == source)
                & (stats_df["distance_metric"] == metric)
            ]
            cells: dict[str, str] = {}
            for statistic in (
                "spearman_correlation",
                "pearson_correlation",
                "hard_minus_easy_mean_distance",
            ):
                row = sub[sub["statistic"] == statistic]
                if row.empty:
                    cells[statistic] = "NA"
                    continue
                record = row.iloc[0]
                cells[statistic] = format_effect(
                    float(record["value"]),
                    float(record["ci_low"]),
                    float(record["ci_high"]),
                    float(record["p_value_bh_fdr_within_statistic"]),
                )
            lines.append(
                f"| `{source}` | {DISTANCE_METRIC_LABELS[metric]} | "
                f"{cells['spearman_correlation']} | {cells['pearson_correlation']} | "
                f"{cells['hard_minus_easy_mean_distance']} |"
            )
    mahalanobis = stats_df[stats_df["distance_metric"] == "oxide_mahalanobis_lw_distance"]
    if not mahalanobis.empty:
        lines.extend(
            [
                "",
                "## Supplemental Mahalanobis Check",
                "",
                "| Embedding level | Status | Covariance condition number | Shrinkage | Spearman rho |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for source in EMBEDDING_SOURCES:
            sub = mahalanobis[
                (mahalanobis["embedding_source"] == source)
                & (mahalanobis["statistic"] == "spearman_correlation")
            ]
            if sub.empty:
                continue
            record = sub.iloc[0]
            lines.append(
                f"| `{source}` | `{record['mahalanobis_status']}` | "
                f"{float(record['mahalanobis_condition_number']):.3g} | "
                f"{float(record['mahalanobis_shrinkage']):.4f} | "
                f"{format_effect(float(record['value']), float(record['ci_low']), float(record['ci_high']), float(record['p_value_bh_fdr_within_statistic']))} |"
            )
    lines.extend(["", "## Figures", ""])
    for figure_path in sorted(figure_paths):
        lines.append(f"- `{figure_path}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    embedding_root = (root / args.embedding_root).resolve()
    stats_csv = (root / args.stats_csv).resolve()
    distance_csv = (root / args.distance_csv).resolve()
    figure_dir = (root / args.figure_dir).resolve()
    summary_md = (root / args.summary_md).resolve()
    manifest_path = (root / args.manifest).resolve()

    refuse_existing(
        [
            stats_csv,
            distance_csv,
            summary_md,
            manifest_path,
            *planned_figure_paths(figure_dir),
        ],
        allow_overwrite=args.allow_overwrite,
    )

    test_set = load_dataset(root, embedding_root, "test_set")
    oxide_reference = load_dataset(root, embedding_root, "oxide_reference_pool")

    for source in EMBEDDING_SOURCES:
        nitride_test_view(test_set, source)
        oxide_reference_view(oxide_reference, source)

    if args.dry_run:
        print("Dry-run only. Distance-vs-error inputs:")
        for source in EMBEDDING_SOURCES:
            nitride_meta, nitride_vectors, errors = nitride_test_view(test_set, source)
            oxide_meta, oxide_vectors = oxide_reference_view(oxide_reference, source)
            labels, n_each = hard_easy_labels(errors)
            print(
                f"{source}: nitrides={nitride_vectors.shape}, oxide_reference={oxide_vectors.shape}, "
                f"hard={int(np.sum(labels == 'hard_top_20pct'))}, "
                f"easy={int(np.sum(labels == 'easy_bottom_20pct'))}, n_each={n_each}"
            )
        print(f"k_oxide_neighbors: {args.k_oxide_neighbors}")
        print(f"bootstrap_iterations: {args.bootstrap_iterations}")
        print(f"permutation_iterations: {args.permutation_iterations}")
        print(f"Stats CSV: {relpath(stats_csv, root)}")
        print(f"Figure dir: {relpath(figure_dir, root)}")
        return

    setup_plot_style()
    stats_rows: list[dict[str, Any]] = []
    distance_rows: list[dict[str, Any]] = []
    figure_paths: list[str] = []

    for source_idx, source in enumerate(EMBEDDING_SOURCES):
        print(f"{source}: loading nitride fixed-test and oxide reference embeddings")
        nitride_meta, nitride_vectors, errors = nitride_test_view(test_set, source)
        oxide_meta, oxide_vectors = oxide_reference_view(oxide_reference, source)
        error_groups, n_each = hard_easy_labels(errors)

        print(f"{source}: computing raw-space distances to oxide reference")
        distances, mahalanobis_diagnostics = compute_distances(
            oxide_vectors,
            nitride_vectors,
            args.k_oxide_neighbors,
        )
        source_distance_rows = distance_rows_for_source(
            source,
            nitride_meta,
            distances,
            error_groups,
            len(oxide_meta),
            args.k_oxide_neighbors,
            mahalanobis_diagnostics,
        )
        distance_rows.extend(source_distance_rows)
        plot_df = pd.DataFrame(source_distance_rows)

        for metric_idx, (metric, values) in enumerate(distances.items()):
            rng = np.random.default_rng(args.seed + source_idx * 100 + metric_idx)
            print(f"{source}/{metric}: correlations and hard/easy comparisons")
            add_corr_rows(
                stats_rows,
                source,
                metric,
                values,
                errors,
                len(oxide_meta),
                mahalanobis_diagnostics,
                rng,
                args.bootstrap_iterations,
                args.permutation_iterations,
                args.k_oxide_neighbors,
                n_each,
                n_each,
            )
            add_group_rows(
                stats_rows,
                source,
                metric,
                values,
                error_groups,
                len(oxide_meta),
                mahalanobis_diagnostics,
                rng,
                args.bootstrap_iterations,
                args.permutation_iterations,
                args.k_oxide_neighbors,
                n_each,
                n_each,
            )
            figure_paths.extend(
                relpath(path, root) for path in plot_scatter(plot_df, source, metric, figure_dir)
            )
            figure_paths.extend(
                relpath(path, root)
                for path in plot_boxplot(plot_df, source, metric, figure_dir, args.seed + source_idx)
            )

    add_fdr_adjustment(stats_rows)

    stats_fields = [
        "embedding_source",
        "distance_metric",
        "distance_label",
        "statistic",
        "comparison",
        "value",
        "ci_low",
        "ci_high",
        "ci_level",
        "p_value",
        "p_value_bh_fdr_within_statistic",
        "p_value_method",
        "test_alternative",
        "n_nitrides",
        "n_oxide_reference",
        "hard_nitrides_count",
        "easy_nitrides_count",
        "k_oxide_neighbors",
        "analysis_space",
        "error_metric",
        "hard_group_distance_summary",
        "easy_group_distance_summary",
        "mahalanobis_status",
        "mahalanobis_condition_number",
        "mahalanobis_shrinkage",
    ]
    distance_fields = [
        "embedding_source",
        "material_id",
        "family",
        "split",
        "formula",
        "absolute_error",
        "absolute_error_float",
        "error_group",
        "embedding_row_index",
        "embedding_dim",
        "n_oxide_reference",
        "k_oxide_neighbors",
        "oxide_centroid_distance",
        f"oxide_knn{args.k_oxide_neighbors}_mean_distance",
        "oxide_mahalanobis_lw_distance",
        "mahalanobis_status",
        "mahalanobis_condition_number",
        "mahalanobis_shrinkage",
    ]
    write_csv(stats_csv, stats_rows, stats_fields)
    write_csv(distance_csv, distance_rows, distance_fields)
    write_summary(
        summary_md,
        stats_rows,
        figure_paths,
        args.seed,
        args.bootstrap_iterations,
        args.permutation_iterations,
        args.k_oxide_neighbors,
    )
    write_json(
        manifest_path,
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/07_analyze_nitride_distance_vs_error.py",
            "inputs": {
                "fixed_test_set_npz": relpath(test_set.npz_path, root),
                "fixed_test_set_metadata": relpath(test_set.metadata_path, root),
                "oxide_reference_pool_npz": relpath(oxide_reference.npz_path, root),
                "oxide_reference_pool_metadata": relpath(oxide_reference.metadata_path, root),
            },
            "outputs": {
                "stats_csv": relpath(stats_csv, root),
                "distance_csv": relpath(distance_csv, root),
                "summary_md": relpath(summary_md, root),
                "figure_dir": relpath(figure_dir, root),
            },
            "embedding_sources": list(EMBEDDING_SOURCES),
            "distance_metrics": list(DISTANCE_METRIC_LABELS),
            "primary_distance_metrics": list(PRIMARY_DISTANCE_METRICS),
            "analysis_space": "raw_256d_embedding_vectors",
            "projection_used_for_tests": False,
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "permutation_iterations": args.permutation_iterations,
            "k_oxide_neighbors": args.k_oxide_neighbors,
            "hard_easy_definition": "top/bottom 20% of fixed-test nitrides by absolute zero-shot error",
        },
    )
    print(f"Stats: {relpath(stats_csv, root)}")
    print(f"Distances: {relpath(distance_csv, root)}")
    print(f"Figures: {relpath(figure_dir, root)}")
    print(f"Summary: {relpath(summary_md, root)}")
    print(f"Manifest: {relpath(manifest_path, root)}")


if __name__ == "__main__":
    main()
