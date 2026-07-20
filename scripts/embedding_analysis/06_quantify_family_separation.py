#!/usr/bin/env python
"""Quantify oxide-vs-nitride separation in raw ALIGNN embedding space."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    davies_bouldin_score,
    roc_auc_score,
    silhouette_samples,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_OUTPUT_CSV = "results/reproduction/embeddings/tables/family_separation_metrics.csv"
DEFAULT_SUMMARY_MD = "results/reproduction/embeddings/family_separation_summary.md"
DEFAULT_MANIFEST = "results/embeddings/manifests/family_separation_metrics_manifest.json"
DEFAULT_EMBEDDING_ROOT = "results/embeddings/embeddings"
DEFAULT_SEED = 42
DEFAULT_BOOTSTRAPS = 1000
DEFAULT_K_NEIGHBORS = 15
DEFAULT_CV_FOLDS = 5
EMBEDDING_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")
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
FAMILY_TO_LABEL = {"oxide": 0, "nitride": 1}
LABEL_TO_FAMILY = {0: "oxide", 1: "nitride"}


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
        description="Compute family-separation metrics on raw extracted embeddings."
    )
    parser.add_argument("--embedding-root", default=DEFAULT_EMBEDDING_ROOT)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--summary-md", default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap-iterations", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    parser.add_argument("--cv-folds", type=int, default=DEFAULT_CV_FOLDS)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing family-separation metric outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print planned comparisons without computing metrics.",
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


def refuse_existing(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing family-separation outputs. "
            f"Use --allow-overwrite to regenerate them:\n{joined}"
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
    unknown = sorted(set(df["family"]) - set(FAMILY_TO_LABEL))
    if unknown:
        raise ValueError(f"Unexpected family labels in {dataset.metadata_path}: {unknown}")
    return df


def family_labels(df: pd.DataFrame) -> np.ndarray:
    return df["family"].map(FAMILY_TO_LABEL).to_numpy(dtype=int)


def family_counts(y: np.ndarray) -> dict[str, int]:
    return {
        family: int(np.sum(y == label))
        for label, family in LABEL_TO_FAMILY.items()
    }


def percentile_ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return math.nan, math.nan
    low, high = np.percentile(arr, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_mean_ci(
    values: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2 or n_bootstrap <= 0:
        return math.nan, math.nan
    boot = [
        float(np.mean(values[rng.integers(0, len(values), len(values))]))
        for _ in range(n_bootstrap)
    ]
    return percentile_ci(boot)


def bootstrap_stratified_mean_ci(
    values: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if n_bootstrap <= 0:
        return math.nan, math.nan
    group_indices = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if any(len(indices) < 2 for indices in group_indices):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled_values = []
        for indices in group_indices:
            sampled = rng.choice(indices, size=len(indices), replace=True)
            sampled_values.append(values[sampled])
        boot.append(float(np.mean(np.concatenate(sampled_values))))
    return percentile_ci(boot)


def bootstrap_davies_bouldin_ci(
    x: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return math.nan, math.nan
    group_indices = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if len(group_indices) < 2 or any(len(indices) < 2 for indices in group_indices):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in group_indices]
        )
        try:
            boot.append(float(davies_bouldin_score(x[sampled], y[sampled])))
        except ValueError:
            continue
    return percentile_ci(boot)


def bootstrap_auc_ci(
    y: np.ndarray,
    scores: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return math.nan, math.nan
    group_indices = [np.flatnonzero(y == label) for label in sorted(np.unique(y))]
    if len(group_indices) < 2 or any(len(indices) < 2 for indices in group_indices):
        return math.nan, math.nan
    boot: list[float] = []
    for _ in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in group_indices]
        )
        try:
            boot.append(float(roc_auc_score(y[sampled], scores[sampled])))
        except ValueError:
            continue
    return percentile_ci(boot)


def knn_family_purity(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    if k < 1:
        raise ValueError(f"k-neighbors must be positive, got {k}.")
    if k >= len(y):
        raise ValueError(f"k-neighbors must be less than n_structures, got k={k}, n={len(y)}.")
    neighbors = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
    neighbors.fit(x)
    indices = neighbors.kneighbors(x, return_distance=False)[:, 1:]
    neighbor_labels = y[indices]
    return np.mean(neighbor_labels == y[:, None], axis=1)


def cross_validated_logistic_auc(
    x: np.ndarray,
    y: np.ndarray,
    cv_folds: int,
    seed: int,
) -> tuple[float, np.ndarray, int]:
    min_class = min(int(np.sum(y == label)) for label in np.unique(y))
    n_splits = min(cv_folds, min_class)
    if n_splits < 2:
        return math.nan, np.full(len(y), np.nan), n_splits
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = np.full(len(y), np.nan, dtype=float)
    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(x, y)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=5000,
                random_state=seed + fold_idx,
                solver="lbfgs",
            ),
        )
        model.fit(x[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(x[test_idx])[:, 1]
    auc = float(roc_auc_score(y, scores))
    return auc, scores, n_splits


def add_metric_row(
    rows: list[dict[str, Any]],
    dataset: EmbeddingDataset,
    source: str,
    y: np.ndarray,
    metric_name: str,
    metric_scope: str,
    value: float,
    ci_low: float,
    ci_high: float,
    ci_method: str,
    preprocessing: str,
    higher_is_better: bool,
    parameters: dict[str, Any] | None = None,
) -> None:
    counts = family_counts(y)
    rows.append(
        {
            "dataset": dataset.subset_name,
            "dataset_label": dataset.label,
            "embedding_source": source,
            "embedding_dim": int(dataset.arrays[source].shape[1]),
            "metric_name": metric_name,
            "metric_scope": metric_scope,
            "value": value,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "ci_level": 0.95,
            "ci_method": ci_method,
            "higher_is_better": higher_is_better,
            "n_structures": int(len(y)),
            "n_oxide": counts["oxide"],
            "n_nitride": counts["nitride"],
            "vector_space": "raw_256d_embedding_vectors",
            "raw_space_primary": True,
            "projected_space": "none",
            "preprocessing": preprocessing,
            "parameters": json.dumps(parameters or {}, sort_keys=True),
        }
    )


def compute_metrics_for_source(
    dataset: EmbeddingDataset,
    source: str,
    bootstrap_iterations: int,
    k_neighbors: int,
    cv_folds: int,
    seed: int,
) -> list[dict[str, Any]]:
    metadata = source_metadata(dataset, source)
    x = dataset.arrays[source]
    y = family_labels(metadata)
    if len(np.unique(y)) != 2:
        raise ValueError(f"{dataset.subset_name}/{source} must contain both oxide and nitride.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []

    print(f"{dataset.subset_name}/{source}: silhouette samples in raw space")
    sil_samples = silhouette_samples(x, y, metric="euclidean")
    low, high = bootstrap_stratified_mean_ci(sil_samples, y, rng, bootstrap_iterations)
    add_metric_row(
        rows,
        dataset,
        source,
        y,
        "silhouette_score",
        "overall_family_labels",
        float(np.mean(sil_samples)),
        low,
        high,
        "stratified_bootstrap_over_per_structure_silhouette_values",
        "none",
        True,
        {"metric": "euclidean"},
    )
    for label, family in LABEL_TO_FAMILY.items():
        values = sil_samples[y == label]
        low, high = bootstrap_mean_ci(values, rng, bootstrap_iterations)
        add_metric_row(
            rows,
            dataset,
            source,
            y,
            "silhouette_score",
            family,
            float(np.mean(values)),
            low,
            high,
            "bootstrap_over_family_per_structure_silhouette_values",
            "none",
            True,
            {"metric": "euclidean"},
        )

    print(f"{dataset.subset_name}/{source}: Davies-Bouldin index in raw space")
    try:
        dbi = float(davies_bouldin_score(x, y))
        low, high = bootstrap_davies_bouldin_ci(x, y, rng, bootstrap_iterations)
        ci_method = "stratified_bootstrap_recomputed_index"
    except ValueError:
        dbi = math.nan
        low, high = math.nan, math.nan
        ci_method = "not_meaningful_for_available_family_labels"
    add_metric_row(
        rows,
        dataset,
        source,
        y,
        "davies_bouldin_index",
        "overall_family_labels",
        dbi,
        low,
        high,
        ci_method,
        "none",
        False,
        {"metric": "euclidean"},
    )

    print(f"{dataset.subset_name}/{source}: kNN family purity in raw space")
    purity = knn_family_purity(x, y, k_neighbors)
    low, high = bootstrap_stratified_mean_ci(purity, y, rng, bootstrap_iterations)
    add_metric_row(
        rows,
        dataset,
        source,
        y,
        "knn_family_purity",
        "overall_family_labels",
        float(np.mean(purity)),
        low,
        high,
        "stratified_bootstrap_over_per_structure_purity_values",
        "none",
        True,
        {"k_neighbors": k_neighbors, "metric": "euclidean", "self_excluded": True},
    )
    for label, family in LABEL_TO_FAMILY.items():
        values = purity[y == label]
        low, high = bootstrap_mean_ci(values, rng, bootstrap_iterations)
        add_metric_row(
            rows,
            dataset,
            source,
            y,
            "knn_family_purity",
            family,
            float(np.mean(values)),
            low,
            high,
            "bootstrap_over_family_per_structure_purity_values",
            "none",
            True,
            {"k_neighbors": k_neighbors, "metric": "euclidean", "self_excluded": True},
        )

    print(f"{dataset.subset_name}/{source}: logistic-regression family AUC on frozen embeddings")
    auc, oof_scores, actual_folds = cross_validated_logistic_auc(x, y, cv_folds, seed)
    low, high = bootstrap_auc_ci(y, oof_scores, rng, bootstrap_iterations)
    add_metric_row(
        rows,
        dataset,
        source,
        y,
        "logistic_regression_family_auc",
        "overall_family_labels",
        auc,
        low,
        high,
        "stratified_bootstrap_over_cross_validated_out_of_fold_scores",
        "fold_local_standard_scaler_no_dimensionality_reduction",
        True,
        {
            "cv_folds": actual_folds,
            "class_weight": "balanced",
            "solver": "lbfgs",
            "positive_label": "nitride",
        },
    )

    return rows


def format_metric(value: float, low: float, high: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if np.isfinite(low) and np.isfinite(high):
        return f"{value:.4f} [{low:.4f}, {high:.4f}]"
    return f"{value:.4f}"


def write_summary(path: Path, rows: list[dict[str, Any]], seed: int, bootstrap_iterations: int) -> None:
    df = pd.DataFrame(rows)
    primary = df[df["metric_scope"] == "overall_family_labels"].copy()
    metric_order = [
        "silhouette_score",
        "davies_bouldin_index",
        "knn_family_purity",
        "logistic_regression_family_auc",
    ]
    lines = [
        "# Family Separation Summary",
        "",
        "These metrics quantify oxide-vs-nitride separation on the original extracted 256D structure embedding vectors.",
        "",
        "## Scope",
        "",
        "- Raw-space metrics are primary.",
        "- PCA, t-SNE, and UMAP coordinates are not used in this table.",
        "- Silhouette, Davies-Bouldin, and kNN purity use Euclidean distances in raw embedding space with no preprocessing.",
        "- Logistic regression AUC is a frozen-embedding linear probe. It uses fold-local standardization inside cross-validation for optimization/regularization, with no dimensionality reduction and no ALIGNN retraining.",
        f"- Bootstrap confidence intervals use {bootstrap_iterations} iterations with seed {seed}.",
        "",
        "## Overall Metrics",
        "",
        "| Dataset | Embedding level | Silhouette | Davies-Bouldin | kNN purity | Logistic AUC |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for source in EMBEDDING_SOURCES:
            sub = primary[
                (primary["dataset"] == dataset)
                & (primary["embedding_source"] == source)
            ]
            cells: dict[str, str] = {}
            for metric in metric_order:
                row = sub[sub["metric_name"] == metric]
                if row.empty:
                    cells[metric] = "NA"
                else:
                    record = row.iloc[0]
                    cells[metric] = format_metric(
                        float(record["value"]),
                        float(record["ci_low"]),
                        float(record["ci_high"]),
                    )
            lines.append(
                f"| `{dataset}` | `{source}` | {cells['silhouette_score']} | "
                f"{cells['davies_bouldin_index']} | {cells['knn_family_purity']} | "
                f"{cells['logistic_regression_family_auc']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Larger silhouette, kNN purity, and logistic AUC indicate stronger family separation.",
            "- Smaller Davies-Bouldin index indicates stronger family separation.",
            "- Strong raw-space separation indicates that family identity is recoverable from frozen ALIGNN embeddings; it does not by itself identify a causal mechanism for domain shift.",
            "- Projected 2D plots should be read as descriptive support, not as the primary separation evidence.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    embedding_root = (root / args.embedding_root).resolve()
    output_csv = (root / args.output_csv).resolve()
    summary_md = (root / args.summary_md).resolve()
    manifest_path = (root / args.manifest).resolve()

    refuse_existing(
        [output_csv, summary_md, manifest_path],
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
        print("Dry-run only. Raw-space family separation inputs:")
        for name, dataset in datasets.items():
            print(f"{name}: {dataset.npz_path}")
            for source in EMBEDDING_SOURCES:
                meta = source_metadata(dataset, source)
                counts = meta.groupby("family").size().to_dict()
                print(f"  {source}: {dataset.arrays[source].shape}, counts={counts}")
        print(f"k_neighbors: {args.k_neighbors}")
        print(f"cv_folds: {args.cv_folds}")
        print(f"bootstrap_iterations: {args.bootstrap_iterations}")
        print(f"Output CSV: {relpath(output_csv, root)}")
        print(f"Summary: {relpath(summary_md, root)}")
        return

    all_rows: list[dict[str, Any]] = []
    for dataset_name, dataset in datasets.items():
        for source_idx, source in enumerate(EMBEDDING_SOURCES):
            source_seed = args.seed + 1000 * list(DATASETS).index(dataset_name) + source_idx
            rows = compute_metrics_for_source(
                dataset,
                source,
                args.bootstrap_iterations,
                args.k_neighbors,
                args.cv_folds,
                source_seed,
            )
            all_rows.extend(rows)

    fieldnames = [
        "dataset",
        "dataset_label",
        "embedding_source",
        "embedding_dim",
        "metric_name",
        "metric_scope",
        "value",
        "ci_low",
        "ci_high",
        "ci_level",
        "ci_method",
        "higher_is_better",
        "n_structures",
        "n_oxide",
        "n_nitride",
        "vector_space",
        "raw_space_primary",
        "projected_space",
        "preprocessing",
        "parameters",
    ]
    write_csv(output_csv, all_rows, fieldnames)
    write_summary(summary_md, all_rows, args.seed, args.bootstrap_iterations)
    write_json(
        manifest_path,
        {
            "created_at_unix": time.time(),
            "created_by": "scripts/embedding_analysis/06_quantify_family_separation.py",
            "inputs": {
                name: {
                    "npz": relpath(dataset.npz_path, root),
                    "metadata": relpath(dataset.metadata_path, root),
                }
                for name, dataset in datasets.items()
            },
            "outputs": {
                "metrics_csv": relpath(output_csv, root),
                "summary_md": relpath(summary_md, root),
            },
            "embedding_sources": list(EMBEDDING_SOURCES),
            "metrics": [
                "silhouette_score",
                "davies_bouldin_index",
                "knn_family_purity",
                "logistic_regression_family_auc",
            ],
            "space": "raw_256d_embedding_vectors",
            "projected_space_used": False,
            "seed": args.seed,
            "bootstrap_iterations": args.bootstrap_iterations,
            "k_neighbors": args.k_neighbors,
            "cv_folds": args.cv_folds,
        },
    )
    print(f"Metrics: {relpath(output_csv, root)}")
    print(f"Summary: {relpath(summary_md, root)}")
    print(f"Manifest: {relpath(manifest_path, root)}")


if __name__ == "__main__":
    main()
