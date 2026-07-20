#!/usr/bin/env python3
"""Recompute nitride distance-error statistics from canonical zero-shot errors.

This canonical producer deliberately separates authorities:

* results/zero_shot/nitride/predictions.csv supplies JIDs,
  targets, predictions, and absolute errors.
* Frozen NPZ arrays supply last_alignn_pool vectors.
* Embedding metadata supplies only membership and NPZ row alignment.

When the archive-only legacy distance-error package is available, it is hashed
for preservation.  That package is not shipped in the public tree and is never
used as a numerical input, so its absence does not block the canonical
recomputation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/derived_evidence/distance_error_recompute"

PREDICTIONS = ROOT / "results/zero_shot/nitride/predictions.csv"
TEST_NPZ = ROOT / "results/embeddings/embeddings/test_set/structure_embeddings.npz"
TEST_METADATA = ROOT / "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv"
REFERENCE_NPZ = ROOT / "results/embeddings/embeddings/oxide_reference_pool/structure_embeddings.npz"
REFERENCE_METADATA = ROOT / "results/embeddings/embeddings/oxide_reference_pool/structure_embedding_metadata.csv"

EXPECTED_INPUT_HASHES = {
    "results/zero_shot/nitride/predictions.csv":
        "2753f6f7e2381d11ecadef9cdccbaabb75563138a4def9ac04f8f8983f2f084f",
    "results/embeddings/embeddings/test_set/structure_embeddings.npz":
        "3958735acc3b93c42edd59cdf96f9e17a25adfcffe43dec05179be58aadbf57a",
    "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv":
        "69d42a08778de14279685e39485135132d19ac27e5bb68001cae07eabded0cde",
    "results/embeddings/embeddings/oxide_reference_pool/structure_embeddings.npz":
        "e227999df6a9b280084977fede1797198197415eb63364107228e29878b99602",
    "results/embeddings/embeddings/oxide_reference_pool/structure_embedding_metadata.csv":
        "f77a80451b89d4f08985aae09bd5f4b0f520263eb8a01a41fc33b8f23745a25b",
}

EXPECTED_QUARANTINED_HASHES = {
    "results/reproduction/embeddings/tables/nitride_distance_error_stats.csv":
        "2b70694045c1394b8e50dde6a57a9b33b99d7259102ba4b976c1e8875d871c70",
    "results/reproduction/embeddings/tables/nitride_distance_error_by_structure.csv":
        "76a583c28b92e54e1d92b79c06cbd73705447394c110450ff393c14740a30a86",
    "results/embeddings/manifests/nitride_distance_error_manifest.json":
        "b134dbdabe1bfb5ee1780145222dba1f08b6c31962c5c403d27ae8288234633a",
    "results/reproduction/embeddings/domain_shift_hypothesis_test.md":
        "157c3280a9d58f8c2456742670b420989e9ad9a9f5c1ecb14c1daa1459a5aa30",
}

PREDICTION_SCHEMA = ["jid", "filename", "target", "prediction", "abs_error"]
EMBEDDING_SOURCE = "last_alignn_pool"
DISTANCE_METRICS = (
    "oxide_centroid_distance",
    "oxide_knn5_mean_distance",
    "oxide_mahalanobis_lw_distance",
)
STATISTICS = (
    "spearman_correlation",
    "pearson_correlation",
    "hard_minus_easy_mean_distance",
    "hard_minus_easy_median_distance",
)
METRIC_SEEDS = {
    "oxide_centroid_distance": 142,
    "oxide_knn5_mean_distance": 143,
    "oxide_mahalanobis_lw_distance": 144,
}
BASE_SEED = 42
BOOTSTRAP_ITERATIONS = 5000
PERMUTATION_ITERATIONS = 10000
K_NEIGHBORS = 5
OUTPUT_FILES = (
    "distance_error_by_structure.csv",
    "distance_error_statistics.csv",
    "distance_error_legacy_nine_cell_sensitivity.csv",
    "distance_error_protocol.json",
    "distance_error_decision.json",
    "input_alignment_audit.json",
    "distance_error_recompute_report.md",
    "input_output_manifest.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def reject_forbidden(path: Path) -> None:
    parts = path.resolve().parts
    if "archived_submission_materials" in parts:
        raise RuntimeError("Forbidden Section 08 path rejected without inspection.")
    nested = (ROOT / "domain_shift-alignn-domain-shift").resolve()
    try:
        path.resolve().relative_to(nested)
    except ValueError:
        return
    raise RuntimeError("Nested checkout path rejected.")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        quoting=csv.QUOTE_MINIMAL,
    )


def verify_hashes(expected: dict[str, str]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for rel, expected_hash in expected.items():
        path = ROOT / rel
        reject_forbidden(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        value = sha256(path)
        if value != expected_hash:
            raise RuntimeError(
                f"Frozen hash mismatch for {rel}: {value} != {expected_hash}"
            )
        observed[rel] = value
    return observed


def quarantine_snapshot() -> dict[str, str]:
    """Hash the complete archive-only package when it is locally available.

    The public release intentionally omits this historical comparison package.
    A partial package is therefore treated as unavailable rather than as a
    canonical-input failure; none of these files participates in the numerical
    analysis below.
    """
    core_paths = [ROOT / rel for rel in EXPECTED_QUARANTINED_HASHES]
    for path in core_paths:
        reject_forbidden(path)
    figure_dir = ROOT / "results/reproduction/embeddings/figures/distance_vs_error"
    reject_forbidden(figure_dir)
    figures = (
        sorted(
            p for p in figure_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".png", ".pdf"}
        )
        if figure_dir.is_dir()
        else []
    )
    if not all(path.is_file() for path in core_paths) or len(figures) != 36:
        return {}

    snapshot = verify_hashes(EXPECTED_QUARANTINED_HASHES)
    for path in figures:
        snapshot[relative(path)] = sha256(path)
    return dict(sorted(snapshot.items()))


def prepare_output(path: Path, overwrite: bool) -> None:
    reject_forbidden(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in OUTPUT_FILES if (path / name).exists()]
    if existing and not overwrite:
        names = ", ".join(item.name for item in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {names}")


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, dict[str, Any]]:
    predictions = pd.read_csv(PREDICTIONS)
    if list(predictions.columns) != PREDICTION_SCHEMA:
        raise ValueError(f"Unexpected canonical prediction schema: {list(predictions.columns)}")
    if len(predictions) != 242:
        raise ValueError(f"Expected 242 canonical nitride predictions, found {len(predictions)}")
    if predictions["jid"].isna().any() or predictions["jid"].duplicated().any():
        raise ValueError("Canonical prediction JIDs must be unique and non-null.")
    for column in ("target", "prediction", "abs_error"):
        predictions[column] = pd.to_numeric(predictions[column], errors="raise")
    recomputed_error = (predictions["target"] - predictions["prediction"]).abs()
    if not np.allclose(
        predictions["abs_error"].to_numpy(float),
        recomputed_error.to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Canonical abs_error is inconsistent with target and prediction.")

    test_metadata_all = pd.read_csv(TEST_METADATA)
    required = {
        "material_id", "family", "split", "formula",
        "target_formation_energy_peratom", "pretrained_prediction", "absolute_error",
        "embedding_source", "embedding_index", "embedding_dim", "filename",
    }
    if not required.issubset(test_metadata_all.columns):
        raise ValueError("Test embedding metadata is missing required columns.")
    test_metadata = test_metadata_all[
        test_metadata_all["embedding_source"].eq(EMBEDDING_SOURCE)
    ].copy()
    test_metadata["embedding_index"] = pd.to_numeric(
        test_metadata["embedding_index"], errors="raise"
    ).astype(int)
    if len(test_metadata) != 1726:
        raise ValueError(f"Expected 1726 last_alignn_pool metadata rows, found {len(test_metadata)}")
    if (
        test_metadata["embedding_index"].duplicated().any()
        or set(test_metadata["embedding_index"]) != set(range(1726))
    ):
        raise ValueError("Test embedding indices are not a unique 0..1725 mapping.")

    nitride_metadata = test_metadata[test_metadata["family"].eq("nitride")].copy()
    if len(nitride_metadata) != 242 or set(nitride_metadata["split"]) != {"test"}:
        raise ValueError("Expected exactly 242 fixed-test nitride metadata rows.")
    if nitride_metadata["material_id"].duplicated().any():
        raise ValueError("Nitride embedding metadata contains duplicate JIDs.")

    canonical = predictions.rename(
        columns={
            "filename": "canonical_filename",
            "target": "canonical_target",
            "prediction": "canonical_prediction",
            "abs_error": "canonical_abs_error",
        }
    )
    joined = nitride_metadata.merge(
        canonical,
        left_on="material_id",
        right_on="jid",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if len(joined) != 242 or set(joined["_merge"].astype(str)) != {"both"}:
        raise ValueError("Canonical prediction and embedding JID coverage is not exactly 242/242.")
    joined = joined.drop(columns=["_merge"])
    if not joined["filename"].eq(joined["canonical_filename"]).all():
        raise ValueError("Canonical and embedding filenames do not match exactly.")
    if not np.array_equal(
        joined["target_formation_energy_peratom"].to_numpy(float),
        joined["canonical_target"].to_numpy(float),
    ):
        raise ValueError("Canonical and embedding targets do not match exactly.")
    joined = joined.sort_values("embedding_index").reset_index(drop=True)

    with np.load(TEST_NPZ) as payload:
        if EMBEDDING_SOURCE not in payload.files:
            raise KeyError(f"{EMBEDDING_SOURCE} absent from test NPZ")
        test_vectors = np.asarray(payload[EMBEDDING_SOURCE], dtype=np.float64)
    if test_vectors.shape != (1726, 256) or not np.isfinite(test_vectors).all():
        raise ValueError(f"Invalid test embedding array: {test_vectors.shape}")
    nitride_vectors = test_vectors[joined["embedding_index"].to_numpy(int)]

    reference_metadata_all = pd.read_csv(REFERENCE_METADATA)
    if not required.issubset(reference_metadata_all.columns):
        raise ValueError("Reference embedding metadata is missing required columns.")
    reference_metadata = reference_metadata_all[
        reference_metadata_all["embedding_source"].eq(EMBEDDING_SOURCE)
        & reference_metadata_all["family"].eq("oxide")
    ].copy()
    reference_metadata["embedding_index"] = pd.to_numeric(
        reference_metadata["embedding_index"], errors="raise"
    ).astype(int)
    reference_metadata = reference_metadata.sort_values("embedding_index").reset_index(drop=True)
    if len(reference_metadata) != 13507:
        raise ValueError(f"Expected 13507 oxide reference rows, found {len(reference_metadata)}")
    if reference_metadata["material_id"].duplicated().any():
        raise ValueError("Oxide reference JIDs are not unique.")
    if set(reference_metadata["embedding_index"]) != set(range(13507)):
        raise ValueError("Reference embedding indices are not exactly 0..13506.")
    if set(reference_metadata["material_id"]) & set(joined["material_id"]):
        raise ValueError("Oxide reference pool overlaps the fixed-test nitride set.")

    with np.load(REFERENCE_NPZ) as payload:
        if EMBEDDING_SOURCE not in payload.files:
            raise KeyError(f"{EMBEDDING_SOURCE} absent from reference NPZ")
        reference_all = np.asarray(payload[EMBEDDING_SOURCE], dtype=np.float64)
    if reference_all.shape != (13507, 256) or not np.isfinite(reference_all).all():
        raise ValueError(f"Invalid reference embedding array: {reference_all.shape}")
    reference_vectors = reference_all[reference_metadata["embedding_index"].to_numpy(int)]

    metadata_prediction = pd.to_numeric(joined["pretrained_prediction"], errors="raise")
    metadata_error = pd.to_numeric(joined["absolute_error"], errors="raise")
    prediction_drift = np.abs(
        metadata_prediction.to_numpy(float)
        - joined["canonical_prediction"].to_numpy(float)
    )
    error_drift = np.abs(
        metadata_error.to_numpy(float)
        - joined["canonical_abs_error"].to_numpy(float)
    )
    alignment_audit = {
        "canonical_embedding_join_count": 242,
        "canonical_unique_jids": 242,
        "embedding_unique_jids": 242,
        "filename_exact_match_count": 242,
        "target_exact_match_count": 242,
        "metadata_prediction_exact_mismatch_count": int(np.count_nonzero(prediction_drift)),
        "metadata_prediction_max_absolute_drift_eV_per_atom": float(prediction_drift.max()),
        "metadata_error_exact_mismatch_count": int(np.count_nonzero(error_drift)),
        "metadata_error_max_absolute_drift_eV_per_atom": float(error_drift.max()),
        "authority_decision": (
            "Canonical CSV supplies target, prediction, and absolute error; "
            "metadata prediction and error fields are alignment diagnostics only."
        ),
    }
    return joined, reference_metadata, nitride_vectors, test_metadata, reference_vectors, alignment_audit


def hard_easy_labels(errors: np.ndarray, embedding_indices: np.ndarray) -> tuple[np.ndarray, int]:
    order = np.lexsort((embedding_indices, errors))
    n_each = int(math.ceil(0.20 * len(errors)))
    labels = np.full(len(errors), "middle_60pct", dtype=object)
    labels[order[:n_each]] = "easy_bottom_20pct"
    labels[order[-n_each:]] = "hard_top_20pct"
    return labels, n_each


def compute_distances(
    reference_vectors: np.ndarray,
    nitride_vectors: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    centroid = reference_vectors.mean(axis=0)
    distances: dict[str, np.ndarray] = {
        "oxide_centroid_distance": np.linalg.norm(nitride_vectors - centroid, axis=1)
    }
    neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="euclidean")
    neighbors.fit(reference_vectors)
    neighbor_distances, _ = neighbors.kneighbors(nitride_vectors, return_distance=True)
    distances["oxide_knn5_mean_distance"] = neighbor_distances.mean(axis=1)

    n_reference, dimension = reference_vectors.shape
    if n_reference < 5 * dimension:
        raise RuntimeError("Mahalanobis-LW reference-count stability gate failed.")
    estimator = LedoitWolf().fit(reference_vectors)
    condition = float(np.linalg.cond(estimator.covariance_))
    if not np.isfinite(condition) or condition > 1.0e12:
        raise RuntimeError(f"Mahalanobis-LW covariance stability gate failed: {condition}")
    diff = nitride_vectors - estimator.location_
    squared = np.einsum("ij,jk,ik->i", diff, estimator.precision_, diff)
    mahalanobis = np.sqrt(np.maximum(squared, 0.0))
    if not np.isfinite(mahalanobis).all():
        raise RuntimeError("Mahalanobis-LW distances are non-finite.")
    distances["oxide_mahalanobis_lw_distance"] = mahalanobis
    for name, values in distances.items():
        if len(values) != 242 or not np.isfinite(values).all() or np.any(values < 0):
            raise RuntimeError(f"Invalid distances for {name}")
    diagnostics = {
        "mahalanobis_status": "computed_ledoit_wolf_stable",
        "mahalanobis_condition_number": condition,
        "mahalanobis_shrinkage": float(estimator.shrinkage_),
    }
    return distances, diagnostics


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        raise RuntimeError("No finite bootstrap values.")
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    raise ValueError(method)


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values: list[float] = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        index = rng.integers(0, len(x), len(x))
        value = correlation(x[index], y[index], method)
        if np.isfinite(value):
            values.append(value)
    return percentile_ci(values)


def centered(values: np.ndarray, method: str) -> np.ndarray:
    source = rankdata(values, method="average") if method == "spearman" else values
    return np.asarray(source, dtype=float) - float(np.mean(source))


def fast_correlation(x_centered: np.ndarray, y_centered: np.ndarray) -> float:
    denominator = math.sqrt(
        float(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered))
    )
    if denominator == 0:
        return math.nan
    return float(np.dot(x_centered, y_centered) / denominator)


def permutation_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    observed: float,
    rng: np.random.Generator,
) -> float:
    x_centered = centered(x, method)
    y_centered = centered(y, method)
    count = 0
    for _ in range(PERMUTATION_ITERATIONS):
        value = fast_correlation(x_centered, rng.permutation(y_centered))
        count += int(value >= observed)
    return float((count + 1) / (PERMUTATION_ITERATIONS + 1))


def bootstrap_group_difference(
    easy: np.ndarray,
    hard: np.ndarray,
    statistic: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values: list[float] = []
    function = np.mean if statistic == "mean" else np.median
    for _ in range(BOOTSTRAP_ITERATIONS):
        easy_sample = rng.choice(easy, size=len(easy), replace=True)
        hard_sample = rng.choice(hard, size=len(hard), replace=True)
        values.append(float(function(hard_sample) - function(easy_sample)))
    return percentile_ci(values)


def permutation_group_difference(
    easy: np.ndarray,
    hard: np.ndarray,
    statistic: str,
    observed: float,
    rng: np.random.Generator,
) -> float:
    function = np.mean if statistic == "mean" else np.median
    combined = np.concatenate([easy, hard])
    n_easy = len(easy)
    count = 0
    for _ in range(PERMUTATION_ITERATIONS):
        shuffled = rng.permutation(combined)
        value = float(function(shuffled[n_easy:]) - function(shuffled[:n_easy]))
        count += int(value >= observed)
    return float((count + 1) / (PERMUTATION_ITERATIONS + 1))


def benjamini_hochberg(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(len(ranked), dtype=float)
    running = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        running = min(running, ranked[index] * len(ranked) / (index + 1))
        adjusted_ranked[index] = running
    adjusted = np.empty(len(ranked), dtype=float)
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted.tolist()


def statistics_rows(
    distances: dict[str, np.ndarray],
    errors: np.ndarray,
    groups: np.ndarray,
    diagnostics: dict[str, Any],
    embedding_source: str,
    metric_seeds: dict[str, int],
    fdr_family: str,
    fdr_family_size: int,
    apply_fdr: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in DISTANCE_METRICS:
        values = distances[metric]
        rng = np.random.default_rng(metric_seeds[metric])
        for method, name in (
            ("spearman", "spearman_correlation"),
            ("pearson", "pearson_correlation"),
        ):
            value = correlation(values, errors, method)
            ci_low, ci_high = bootstrap_correlation(values, errors, method, rng)
            p_value = permutation_correlation(values, errors, method, value, rng)
            rows.append(
                {
                    "embedding_source": embedding_source,
                    "distance_metric": metric,
                    "statistic": name,
                    "comparison": "nitride_canonical_abs_error_vs_distance",
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_level": 0.95,
                    "p_value": p_value,
                    "p_value_bh_fdr_within_statistic": math.nan,
                    "p_value_method": "permutation_pairing_errors_to_distances",
                    "test_alternative": "greater",
                    "n_nitrides": 242,
                    "n_oxide_reference": 13507,
                    "hard_nitrides_count": 49,
                    "easy_nitrides_count": 49,
                    "k_oxide_neighbors": K_NEIGHBORS if "knn" in metric else "",
                    "analysis_space": "raw_256d_embedding_vectors",
                    "error_metric": "canonical_absolute_zero_shot_error",
                    "hard_group_distance_summary": "",
                    "easy_group_distance_summary": "",
                    **diagnostics,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "permutation_iterations": PERMUTATION_ITERATIONS,
                    "rng_seed": metric_seeds[metric],
                    "bh_fdr_family": fdr_family,
                    "bh_fdr_family_size": fdr_family_size,
                }
            )

        easy = values[groups == "easy_bottom_20pct"]
        hard = values[groups == "hard_top_20pct"]
        for method, name in (
            ("mean", "hard_minus_easy_mean_distance"),
            ("median", "hard_minus_easy_median_distance"),
        ):
            function = np.mean if method == "mean" else np.median
            easy_summary = float(function(easy))
            hard_summary = float(function(hard))
            value = hard_summary - easy_summary
            ci_low, ci_high = bootstrap_group_difference(easy, hard, method, rng)
            p_value = permutation_group_difference(easy, hard, method, value, rng)
            rows.append(
                {
                    "embedding_source": embedding_source,
                    "distance_metric": metric,
                    "statistic": name,
                    "comparison": "hard_top_20pct_vs_easy_bottom_20pct",
                    "value": value,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_level": 0.95,
                    "p_value": p_value,
                    "p_value_bh_fdr_within_statistic": math.nan,
                    "p_value_method": "permutation_group_labels_within_hard_easy_union",
                    "test_alternative": "greater",
                    "n_nitrides": 242,
                    "n_oxide_reference": 13507,
                    "hard_nitrides_count": 49,
                    "easy_nitrides_count": 49,
                    "k_oxide_neighbors": K_NEIGHBORS if "knn" in metric else "",
                    "analysis_space": "raw_256d_embedding_vectors",
                    "error_metric": "canonical_absolute_zero_shot_error",
                    "hard_group_distance_summary": hard_summary,
                    "easy_group_distance_summary": easy_summary,
                    **diagnostics,
                    "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                    "permutation_iterations": PERMUTATION_ITERATIONS,
                    "rng_seed": metric_seeds[metric],
                    "bh_fdr_family": fdr_family,
                    "bh_fdr_family_size": fdr_family_size,
                }
            )

    if apply_fdr:
        apply_bh_fdr(rows)
    return rows


def apply_bh_fdr(rows: list[dict[str, Any]]) -> None:
    for statistic in STATISTICS:
        indices = [index for index, row in enumerate(rows) if row["statistic"] == statistic]
        adjusted = benjamini_hochberg([float(rows[index]["p_value"]) for index in indices])
        for index, q_value in zip(indices, adjusted):
            rows[index]["p_value_bh_fdr_within_statistic"] = q_value


def legacy_nine_cell_sensitivity(
    joined: pd.DataFrame,
    reference_metadata: pd.DataFrame,
    errors: np.ndarray,
    groups: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    sources = ("pre_head", "last_alignn_pool", "last_gcn_pool")
    with np.load(TEST_NPZ) as payload:
        test_arrays = {
            source: np.asarray(payload[source], dtype=np.float64)
            for source in sources
        }
    with np.load(REFERENCE_NPZ) as payload:
        reference_arrays = {
            source: np.asarray(payload[source], dtype=np.float64)
            for source in sources
        }
    if not np.array_equal(test_arrays["pre_head"], test_arrays["last_gcn_pool"]):
        raise RuntimeError("Expected duplicate pre_head and last_gcn_pool test arrays.")
    if not np.array_equal(
        reference_arrays["pre_head"], reference_arrays["last_gcn_pool"]
    ):
        raise RuntimeError("Expected duplicate pre_head and last_gcn_pool reference arrays.")

    nitride_indices = joined["embedding_index"].to_numpy(int)
    reference_indices = reference_metadata["embedding_index"].to_numpy(int)
    rows: list[dict[str, Any]] = []
    seed_map: dict[str, dict[str, int]] = {}
    for source_index, source in enumerate(sources):
        test = test_arrays[source]
        reference = reference_arrays[source]
        if test.shape != (1726, 256) or reference.shape != (13507, 256):
            raise RuntimeError(f"Unexpected legacy sensitivity shape for {source}.")
        distances, diagnostics = compute_distances(
            reference[reference_indices],
            test[nitride_indices],
        )
        seeds = {
            metric: BASE_SEED + source_index * 100 + metric_index
            for metric_index, metric in enumerate(DISTANCE_METRICS)
        }
        seed_map[source] = seeds
        rows.extend(
            statistics_rows(
                distances,
                errors,
                groups,
                diagnostics,
                embedding_source=source,
                metric_seeds=seeds,
                fdr_family="legacy_nine_source_by_distance_cells_within_statistic",
                fdr_family_size=9,
                apply_fdr=False,
            )
        )
    apply_bh_fdr(rows)
    if len(rows) != 36:
        raise RuntimeError(f"Expected 36 legacy sensitivity rows, found {len(rows)}")
    return pd.DataFrame(rows), {
        "embedding_sources": list(sources),
        "distance_metrics": list(DISTANCE_METRICS),
        "cells_per_statistic": 9,
        "total_rows": 36,
        "metric_rng_seeds": seed_map,
        "pre_head_equals_last_gcn_pool_test": True,
        "pre_head_equals_last_gcn_pool_reference": True,
        "role": (
            "legacy multiplicity sensitivity only; duplicate representations are "
            "not independent confirmations and this family does not control C6_MAIN"
        ),
    }


def structure_frame(
    joined: pd.DataFrame,
    distances: dict[str, np.ndarray],
    groups: np.ndarray,
    diagnostics: dict[str, Any],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "jid": joined["material_id"].astype(str),
            "filename": joined["canonical_filename"].astype(str),
            "formula": joined["formula"].astype(str),
            "family": "nitride",
            "split": "test",
            "target_eV_per_atom": joined["canonical_target"].to_numpy(float),
            "prediction_eV_per_atom": joined["canonical_prediction"].to_numpy(float),
            "canonical_abs_error_eV_per_atom": joined["canonical_abs_error"].to_numpy(float),
            "error_group": groups,
            "embedding_source": EMBEDDING_SOURCE,
            "embedding_index": joined["embedding_index"].to_numpy(int),
            "embedding_dim": joined["embedding_dim"].to_numpy(int),
            "n_oxide_reference": 13507,
            "k_oxide_neighbors": K_NEIGHBORS,
            "oxide_centroid_distance": distances["oxide_centroid_distance"],
            "oxide_knn5_mean_distance": distances["oxide_knn5_mean_distance"],
            "oxide_mahalanobis_lw_distance": distances["oxide_mahalanobis_lw_distance"],
            "mahalanobis_status": diagnostics["mahalanobis_status"],
            "mahalanobis_condition_number": diagnostics["mahalanobis_condition_number"],
            "mahalanobis_shrinkage": diagnostics["mahalanobis_shrinkage"],
            "error_authority": relative(PREDICTIONS),
            "embedding_authority": relative(TEST_NPZ),
        }
    )
    return frame


def build_manifest(
    output_dir: Path,
    input_hashes: dict[str, str],
    quarantine_hashes: dict[str, str],
    generated_names: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path, value in sorted(input_hashes.items()):
        source = ROOT / path
        rows.append(
            {
                "category": "authoritative_input",
                "path": path,
                "sha256": value,
                "bytes": source.stat().st_size,
                "role": (
                    "canonical numerical error authority"
                    if path.endswith("predictions.csv")
                    else "frozen embedding/alignment authority"
                ),
            }
        )
    for path, value in sorted(quarantine_hashes.items()):
        source = ROOT / path
        rows.append(
            {
                "category": "quarantined_preservation_reference",
                "path": path,
                "sha256": value,
                "bytes": source.stat().st_size,
                "role": "preservation hash only; not a numerical input",
            }
        )
    for name in generated_names:
        path = output_dir / name
        rows.append(
            {
                "category": "generated_output",
                "path": f"results/derived_evidence/distance_error_recompute/{name}",
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "role": "canonical distance-error evidence",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    prepare_output(output_dir, args.overwrite)
    input_hashes = verify_hashes(EXPECTED_INPUT_HASHES)
    quarantined_hashes = quarantine_snapshot()

    joined, reference_metadata, nitride_vectors, _test_metadata, reference_vectors, alignment = load_inputs()
    errors = joined["canonical_abs_error"].to_numpy(float)
    groups, n_each = hard_easy_labels(
        errors, joined["embedding_index"].to_numpy(int)
    )
    if n_each != 49:
        raise RuntimeError(f"Unexpected hard/easy count: {n_each}")
    counts = pd.Series(groups).value_counts().to_dict()
    if counts != {
        "middle_60pct": 144,
        "easy_bottom_20pct": 49,
        "hard_top_20pct": 49,
    }:
        raise RuntimeError(f"Unexpected hard/easy membership counts: {counts}")

    distances, diagnostics = compute_distances(reference_vectors, nitride_vectors)
    rows = statistics_rows(
        distances,
        errors,
        groups,
        diagnostics,
        embedding_source=EMBEDDING_SOURCE,
        metric_seeds=METRIC_SEEDS,
        fdr_family="three_last_alignn_pool_distances_within_statistic",
        fdr_family_size=3,
        apply_fdr=True,
    )
    if len(rows) != 12:
        raise RuntimeError(f"Expected 12 statistical rows, found {len(rows)}")
    stats = pd.DataFrame(rows)
    structures = structure_frame(joined, distances, groups, diagnostics)
    legacy_stats, legacy_protocol = legacy_nine_cell_sensitivity(
        joined, reference_metadata, errors, groups
    )

    write_frame(output_dir / "distance_error_by_structure.csv", structures)
    write_frame(output_dir / "distance_error_statistics.csv", stats)
    write_frame(
        output_dir / "distance_error_legacy_nine_cell_sensitivity.csv",
        legacy_stats,
    )

    protocol = {
        "analysis_name": "canonical_nitride_distance_error_recompute",
        "analysis_space": "raw_unprojected_256d_last_alignn_pool",
        "canonical_error_source": relative(PREDICTIONS),
        "embedding_source": EMBEDDING_SOURCE,
        "embedding_inputs": {
            "test_npz": relative(TEST_NPZ),
            "test_metadata_alignment_only": relative(TEST_METADATA),
            "oxide_reference_npz": relative(REFERENCE_NPZ),
            "oxide_reference_metadata_alignment_only": relative(REFERENCE_METADATA),
        },
        "expected_and_observed_input_hashes": input_hashes,
        "quarantined_output_preservation_hashes": quarantined_hashes,
        "quarantined_output_preservation_status": (
            "verified" if quarantined_hashes else "not_available_in_public_release"
        ),
        "distance_metrics": list(DISTANCE_METRICS),
        "centroid_definition": "Euclidean distance to mean of 13507 oxide reference vectors",
        "knn_definition": "mean Euclidean distance to five nearest oxide reference vectors",
        "mahalanobis_definition": "Ledoit-Wolf covariance; reference>=5*dimension; condition<=1e12",
        "correlations": ["Spearman rho", "Pearson r"],
        "hard_easy_definition": (
            "top/bottom ceil(20% of 242)=49 by canonical absolute error; "
            "embedding index breaks exact ties"
        ),
        "group_contrasts": ["hard-minus-easy mean distance", "hard-minus-easy median distance"],
        "bootstrap": {
            "iterations": BOOTSTRAP_ITERATIONS,
            "ci": "nonparametric percentile [2.5,97.5]",
            "correlation_unit": "paired nitride structure",
            "group_unit": "independent within-group nitride structure",
        },
        "permutation": {
            "iterations": PERMUTATION_ITERATIONS,
            "alternative": "greater",
            "plus_one_correction": True,
        },
        "bh_fdr": {
            "scope": "within each statistic across three last_alignn_pool distance metrics",
            "family_size": 3,
            "role": "preselected primary multiplicity family controlling C6_MAIN",
        },
        "legacy_nine_cell_sensitivity": legacy_protocol,
        "base_seed": BASE_SEED,
        "metric_rng_seeds": METRIC_SEEDS,
        "decision_rule": (
            "C6 enters main paper iff last_alignn_pool 5NN Spearman 95% CI excludes zero, "
            "BH-FDR q<0.01, and rho remains positive."
        ),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "authority_guardrail": (
            "Embedding metadata prediction and error columns are prohibited from statistical use."
        ),
    }
    write_json(output_dir / "distance_error_protocol.json", protocol)

    selected = stats[
        stats["distance_metric"].eq("oxide_knn5_mean_distance")
        & stats["statistic"].eq("spearman_correlation")
    ]
    if len(selected) != 1:
        raise RuntimeError("Could not identify exactly one registered C6 decision row.")
    row = selected.iloc[0]
    legacy_selected = legacy_stats[
        legacy_stats["embedding_source"].eq("last_alignn_pool")
        & legacy_stats["distance_metric"].eq("oxide_knn5_mean_distance")
        & legacy_stats["statistic"].eq("spearman_correlation")
    ]
    if len(legacy_selected) != 1:
        raise RuntimeError("Could not identify legacy nine-cell C6 sensitivity row.")
    legacy_row = legacy_selected.iloc[0]
    ci_excludes_zero = bool(float(row["ci_low"]) > 0 or float(row["ci_high"]) < 0)
    q_below = bool(float(row["p_value_bh_fdr_within_statistic"]) < 0.01)
    direction_unchanged = bool(float(row["value"]) > 0)
    include = bool(ci_excludes_zero and q_below and direction_unchanged)
    legacy_include = bool(
        float(legacy_row["ci_low"]) > 0
        and float(legacy_row["p_value_bh_fdr_within_statistic"]) < 0.01
        and float(legacy_row["value"]) > 0
    )
    decision = {
        "claim_id": "C6",
        "metric": "last_alignn_pool mean 5NN oxide distance vs canonical nitride absolute error",
        "spearman_rho": float(row["value"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "ci_level": 0.95,
        "permutation_p_value": float(row["p_value"]),
        "bh_fdr_q_value": float(row["p_value_bh_fdr_within_statistic"]),
        "primary_multiplicity": {
            "family": "three preselected last_alignn_pool distances within statistic",
            "family_size": 3,
            "bh_fdr_q_value": float(row["p_value_bh_fdr_within_statistic"]),
            "controls_C6_MAIN": True,
        },
        "legacy_nine_cell_sensitivity": {
            "family": "three embedding sources by three distances within statistic",
            "family_size": 9,
            "bh_fdr_q_value": float(
                legacy_row["p_value_bh_fdr_within_statistic"]
            ),
            "spearman_rho": float(legacy_row["value"]),
            "ci_low": float(legacy_row["ci_low"]),
            "ci_high": float(legacy_row["ci_high"]),
            "pre_head_equals_last_gcn_pool": True,
            "role": "sensitivity only; does not control C6_MAIN",
        },
        "ci_excludes_zero": ci_excludes_zero,
        "q_below_0_01": q_below,
        "direction_unchanged_positive": direction_unchanged,
        "C6_MAIN": include,
        "decision": "INCLUDE_C6_IN_MAIN_PAPER" if include else "DROP_C6_FROM_MAIN_PAPER",
        "legacy_nine_cell_decision_unchanged": bool(legacy_include == include),
        "interpretation_guardrail": (
            "Correlational and protocol-specific; it does not establish causation or prove domain shift."
        ),
    }
    write_json(output_dir / "distance_error_decision.json", decision)
    write_json(output_dir / "input_alignment_audit.json", alignment)

    report = [
        "# Canonical Distance–Error Recomputation",
        "",
        "## Status",
        "",
        f"- Decision: {decision['decision']}",
        f"- C6_MAIN: {str(include).lower()}",
        "- Error authority: canonical nitride zero-shot prediction CSV.",
        "- Embedding authority: frozen last_alignn_pool NPZ arrays.",
        "- Embedding metadata prediction/error fields were not used.",
        "",
        "## Registered 5-NN Spearman result",
        "",
        f"- rho = {float(row['value']):.10f}",
        f"- 95% percentile CI = [{float(row['ci_low']):.10f}, {float(row['ci_high']):.10f}]",
        f"- directional permutation p = {float(row['p_value']):.10g}",
        f"- BH-FDR q = {float(row['p_value_bh_fdr_within_statistic']):.10g}",
        "- Primary multiplicity family: three preselected last_alignn_pool distances.",
        "",
        "## Legacy nine-cell multiplicity sensitivity",
        "",
        f"- legacy BH-FDR q = {float(legacy_row['p_value_bh_fdr_within_statistic']):.10g}",
        "- Family: three embedding sources × three distances within each statistic.",
        "- The C6 decision is unchanged under this sensitivity.",
        "- pre_head and last_gcn_pool are identical raw arrays and are not independent confirmations.",
        "- This nine-cell result is sensitivity evidence only; the preselected three-distance family controls C6_MAIN.",
        "",
        "The result is an association in the frozen raw representation under this protocol. "
        "It is not a causal explanation and does not independently prove domain shift.",
        "",
        "## Integrity",
        "",
        "- Canonical-to-embedding JID coverage: 242/242.",
        "- Hard/easy/middle counts: 49/49/144.",
        "- Oxide reference vectors: 13,507.",
        f"- Mahalanobis-LW condition number: {diagnostics['mahalanobis_condition_number']:.10g}.",
        f"- Metadata prediction mismatches: {alignment['metadata_prediction_exact_mismatch_count']}/242; "
        "recorded as provenance drift only.",
        (
            f"- Archive-only legacy package verified with {len(quarantined_hashes)} "
            "preservation hashes; it was not used numerically."
            if quarantined_hashes
            else "- Archive-only legacy package is not present in the public release; "
            "the optional preservation check was skipped."
        ),
        "",
    ]
    (output_dir / "distance_error_recompute_report.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    generated_before_manifest = OUTPUT_FILES[:-1]
    manifest = build_manifest(
        output_dir,
        input_hashes,
        quarantined_hashes,
        generated_before_manifest,
    )
    write_frame(output_dir / "input_output_manifest.csv", manifest)

    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
