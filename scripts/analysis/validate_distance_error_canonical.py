#!/usr/bin/env python3
"""Independently validate the public canonical distance-error package."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results/derived_evidence/distance_error_recompute"
PRODUCER = ROOT / "scripts/analysis/recompute_distance_error_canonical.py"

PREDICTIONS = ROOT / "results/zero_shot/nitride/predictions.csv"
TEST_NPZ = ROOT / "results/embeddings/embeddings/test_set/structure_embeddings.npz"
TEST_METADATA = ROOT / "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv"
REFERENCE_NPZ = ROOT / "results/embeddings/embeddings/oxide_reference_pool/structure_embeddings.npz"
REFERENCE_METADATA = ROOT / "results/embeddings/embeddings/oxide_reference_pool/structure_embedding_metadata.csv"

EXPECTED_HASHES = {
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

PRODUCER_OUTPUTS = (
    "distance_error_by_structure.csv",
    "distance_error_statistics.csv",
    "distance_error_legacy_nine_cell_sensitivity.csv",
    "distance_error_protocol.json",
    "distance_error_decision.json",
    "input_alignment_audit.json",
    "distance_error_recompute_report.md",
    "input_output_manifest.csv",
)
DISTANCES = (
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
SEEDS = {
    "oxide_centroid_distance": 142,
    "oxide_knn5_mean_distance": 143,
    "oxide_mahalanobis_lw_distance": 144,
}
BOOTSTRAPS = 5000
PERMUTATIONS = 10000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reject_forbidden(path: Path) -> None:
    if "archived_submission_materials" in path.resolve().parts:
        raise RuntimeError("Forbidden Section 08 path rejected without inspection.")
    nested = (ROOT / "domain_shift-alignn-domain-shift").resolve()
    try:
        path.resolve().relative_to(nested)
    except ValueError:
        return
    raise RuntimeError("Nested checkout path rejected.")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_frozen_hashes() -> dict[str, str]:
    observed: dict[str, str] = {}
    for rel, expected in EXPECTED_HASHES.items():
        path = ROOT / rel
        reject_forbidden(path)
        require(path.is_file(), f"Missing frozen file: {rel}")
        value = sha256(path)
        require(value == expected, f"Frozen hash mismatch: {rel}")
        observed[rel] = value
    return observed


def verify_quarantined_snapshot(output: Path) -> dict[str, str]:
    protocol_path = output / "distance_error_protocol.json"
    require(protocol_path.is_file(), "Missing protocol needed for quarantine verification.")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    snapshot = protocol.get("quarantined_output_preservation_hashes", {})
    require(isinstance(snapshot, dict), "Quarantine snapshot must be a JSON object.")
    if not snapshot:
        require(
            protocol.get("quarantined_output_preservation_status")
            == "not_available_in_public_release",
            "Empty quarantine snapshot must be marked unavailable.",
        )
        return {}
    require(len(snapshot) == 40, "Expected four core plus 36 figure quarantine hashes.")
    require(
        protocol.get("quarantined_output_preservation_status") == "verified",
        "Complete quarantine snapshot must be marked verified.",
    )
    for rel, expected in snapshot.items():
        path = ROOT / rel
        reject_forbidden(path)
        require(path.is_file(), f"Missing quarantined preservation target: {rel}")
        require(sha256(path) == expected, f"Quarantined output changed: {rel}")
    return dict(sorted(snapshot.items()))


def quantile(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    require(len(array) > 0, "No finite bootstrap values.")
    low, high = np.percentile(array, [2.5, 97.5])
    return float(low), float(high)


def correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    return float(pearsonr(x, y).statistic)


def bootstrap_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    collected: list[float] = []
    for _ in range(BOOTSTRAPS):
        selection = rng.integers(0, len(x), len(x))
        value = correlation(x[selection], y[selection], method)
        if np.isfinite(value):
            collected.append(value)
    return quantile(collected)


def normalized(values: np.ndarray, method: str) -> np.ndarray:
    source = rankdata(values, method="average") if method == "spearman" else values
    source = np.asarray(source, dtype=float)
    return source - source.mean()


def dot_corr(x: np.ndarray, y: np.ndarray) -> float:
    denominator = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    return math.nan if denominator == 0 else float(np.dot(x, y) / denominator)


def permutation_corr(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    observed: float,
    rng: np.random.Generator,
) -> float:
    x0 = normalized(x, method)
    y0 = normalized(y, method)
    exceed = 0
    for _ in range(PERMUTATIONS):
        exceed += int(dot_corr(x0, rng.permutation(y0)) >= observed)
    return float((exceed + 1) / (PERMUTATIONS + 1))


def bootstrap_groups(
    easy: np.ndarray,
    hard: np.ndarray,
    kind: str,
    rng: np.random.Generator,
) -> tuple[float, float]:
    function = np.mean if kind == "mean" else np.median
    values: list[float] = []
    for _ in range(BOOTSTRAPS):
        easy_draw = rng.choice(easy, len(easy), replace=True)
        hard_draw = rng.choice(hard, len(hard), replace=True)
        values.append(float(function(hard_draw) - function(easy_draw)))
    return quantile(values)


def permutation_groups(
    easy: np.ndarray,
    hard: np.ndarray,
    kind: str,
    observed: float,
    rng: np.random.Generator,
) -> float:
    function = np.mean if kind == "mean" else np.median
    combined = np.concatenate([easy, hard])
    exceed = 0
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(combined)
        value = float(function(shuffled[len(easy):]) - function(shuffled[:len(easy)]))
        exceed += int(value >= observed)
    return float((exceed + 1) / (PERMUTATIONS + 1))


def bh(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = np.empty(len(p))
    running = 1.0
    for index in range(len(p) - 1, -1, -1):
        running = min(running, ranked[index] * len(p) / (index + 1))
        adjusted_ranked[index] = running
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(adjusted_ranked, 1.0)
    return adjusted.tolist()


def independently_load_and_compute() -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    predictions = pd.read_csv(PREDICTIONS)
    require(
        list(predictions.columns) == ["jid", "filename", "target", "prediction", "abs_error"],
        "Canonical prediction schema changed.",
    )
    require(len(predictions) == 242 and predictions["jid"].nunique() == 242,
            "Canonical nitride membership changed.")
    expected_error = (predictions["target"] - predictions["prediction"]).abs()
    require(np.allclose(predictions["abs_error"], expected_error, rtol=0, atol=1e-12),
            "Canonical absolute errors are inconsistent.")

    metadata = pd.read_csv(TEST_METADATA)
    metadata = metadata[
        metadata["embedding_source"].eq("last_alignn_pool")
        & metadata["family"].eq("nitride")
    ].copy()
    metadata["embedding_index"] = metadata["embedding_index"].astype(int)
    merged = metadata.merge(
        predictions,
        left_on="material_id",
        right_on="jid",
        validate="one_to_one",
        how="inner",
        suffixes=("_metadata", "_canonical"),
    ).sort_values("embedding_index").reset_index(drop=True)
    require(len(merged) == 242, "Canonical/embedding join is not 242/242.")
    require(merged["filename_metadata"].eq(merged["filename_canonical"]).all(),
            "Filename alignment failed.")
    require(np.array_equal(
        merged["target_formation_energy_peratom"].to_numpy(float),
        merged["target"].to_numpy(float),
    ), "Target alignment failed.")

    with np.load(TEST_NPZ) as payload:
        test = np.asarray(payload["last_alignn_pool"], dtype=np.float64)
    nitride = test[merged["embedding_index"].to_numpy(int)]

    reference_metadata = pd.read_csv(REFERENCE_METADATA)
    reference_metadata = reference_metadata[
        reference_metadata["embedding_source"].eq("last_alignn_pool")
        & reference_metadata["family"].eq("oxide")
    ].copy()
    reference_metadata["embedding_index"] = reference_metadata["embedding_index"].astype(int)
    reference_metadata = reference_metadata.sort_values("embedding_index")
    require(len(reference_metadata) == 13507, "Reference membership changed.")
    require(reference_metadata["material_id"].nunique() == 13507,
            "Reference JIDs are not unique.")
    with np.load(REFERENCE_NPZ) as payload:
        reference_all = np.asarray(payload["last_alignn_pool"], dtype=np.float64)
    reference = reference_all[reference_metadata["embedding_index"].to_numpy(int)]

    distances: dict[str, np.ndarray] = {}
    distances["oxide_centroid_distance"] = np.linalg.norm(
        nitride - reference.mean(axis=0), axis=1
    )
    model = NearestNeighbors(n_neighbors=5, metric="euclidean").fit(reference)
    distances["oxide_knn5_mean_distance"] = model.kneighbors(
        nitride, return_distance=True
    )[0].mean(axis=1)
    covariance = LedoitWolf().fit(reference)
    condition = float(np.linalg.cond(covariance.covariance_))
    require(np.isfinite(condition) and condition <= 1e12,
            "Independent Mahalanobis stability check failed.")
    delta = nitride - covariance.location_
    distances["oxide_mahalanobis_lw_distance"] = np.sqrt(
        np.maximum(np.einsum("ij,jk,ik->i", delta, covariance.precision_, delta), 0)
    )

    error = merged["abs_error"].to_numpy(float)
    order = np.lexsort((merged["embedding_index"].to_numpy(int), error))
    labels = np.full(242, "middle_60pct", dtype=object)
    labels[order[:49]] = "easy_bottom_20pct"
    labels[order[-49:]] = "hard_top_20pct"

    structure = pd.DataFrame(
        {
            "jid": merged["jid"],
            "target_eV_per_atom": merged["target"],
            "prediction_eV_per_atom": merged["prediction"],
            "canonical_abs_error_eV_per_atom": merged["abs_error"],
            "error_group": labels,
            "embedding_index": merged["embedding_index"],
            **distances,
        }
    )

    rows: list[dict[str, Any]] = []
    for metric in DISTANCES:
        values = distances[metric]
        rng = np.random.default_rng(SEEDS[metric])
        for method, statistic in (
            ("spearman", "spearman_correlation"),
            ("pearson", "pearson_correlation"),
        ):
            point = correlation(values, error, method)
            low, high = bootstrap_corr(values, error, method, rng)
            p_value = permutation_corr(values, error, method, point, rng)
            rows.append({
                "distance_metric": metric, "statistic": statistic,
                "value": point, "ci_low": low, "ci_high": high,
                "p_value": p_value,
            })
        easy = values[labels == "easy_bottom_20pct"]
        hard = values[labels == "hard_top_20pct"]
        for kind, statistic in (
            ("mean", "hard_minus_easy_mean_distance"),
            ("median", "hard_minus_easy_median_distance"),
        ):
            function = np.mean if kind == "mean" else np.median
            point = float(function(hard) - function(easy))
            low, high = bootstrap_groups(easy, hard, kind, rng)
            p_value = permutation_groups(easy, hard, kind, point, rng)
            rows.append({
                "distance_metric": metric, "statistic": statistic,
                "value": point, "ci_low": low, "ci_high": high,
                "p_value": p_value,
                "hard_group_distance_summary": float(function(hard)),
                "easy_group_distance_summary": float(function(easy)),
            })
    for statistic in STATISTICS:
        indices = [i for i, row in enumerate(rows) if row["statistic"] == statistic]
        adjusted = bh([float(rows[i]["p_value"]) for i in indices])
        for index, value in zip(indices, adjusted):
            rows[index]["p_value_bh_fdr_within_statistic"] = value

    sources = ("pre_head", "last_alignn_pool", "last_gcn_pool")
    with np.load(TEST_NPZ) as payload:
        legacy_test = {
            source: np.asarray(payload[source], dtype=np.float64)
            for source in sources
        }
    with np.load(REFERENCE_NPZ) as payload:
        legacy_reference = {
            source: np.asarray(payload[source], dtype=np.float64)
            for source in sources
        }
    require(
        np.array_equal(legacy_test["pre_head"], legacy_test["last_gcn_pool"]),
        "pre_head and last_gcn_pool test arrays are no longer duplicates.",
    )
    require(
        np.array_equal(
            legacy_reference["pre_head"], legacy_reference["last_gcn_pool"]
        ),
        "pre_head and last_gcn_pool reference arrays are no longer duplicates.",
    )
    nitride_indices = merged["embedding_index"].to_numpy(int)
    reference_indices = reference_metadata["embedding_index"].to_numpy(int)
    legacy_rows: list[dict[str, Any]] = []
    for source_index, source in enumerate(sources):
        source_test = legacy_test[source][nitride_indices]
        source_reference = legacy_reference[source][reference_indices]
        source_distances: dict[str, np.ndarray] = {
            "oxide_centroid_distance": np.linalg.norm(
                source_test - source_reference.mean(axis=0), axis=1
            )
        }
        source_neighbors = NearestNeighbors(
            n_neighbors=5, metric="euclidean"
        ).fit(source_reference)
        source_distances["oxide_knn5_mean_distance"] = source_neighbors.kneighbors(
            source_test, return_distance=True
        )[0].mean(axis=1)
        source_covariance = LedoitWolf().fit(source_reference)
        source_condition = float(np.linalg.cond(source_covariance.covariance_))
        require(
            np.isfinite(source_condition) and source_condition <= 1e12,
            f"Legacy Mahalanobis stability failed for {source}.",
        )
        source_delta = source_test - source_covariance.location_
        source_distances["oxide_mahalanobis_lw_distance"] = np.sqrt(
            np.maximum(
                np.einsum(
                    "ij,jk,ik->i",
                    source_delta,
                    source_covariance.precision_,
                    source_delta,
                ),
                0,
            )
        )
        for metric_index, metric in enumerate(DISTANCES):
            values = source_distances[metric]
            rng = np.random.default_rng(42 + source_index * 100 + metric_index)
            for method, statistic in (
                ("spearman", "spearman_correlation"),
                ("pearson", "pearson_correlation"),
            ):
                point = correlation(values, error, method)
                low, high = bootstrap_corr(values, error, method, rng)
                p_value = permutation_corr(values, error, method, point, rng)
                legacy_rows.append({
                    "embedding_source": source,
                    "distance_metric": metric,
                    "statistic": statistic,
                    "value": point,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                })
            easy = values[labels == "easy_bottom_20pct"]
            hard = values[labels == "hard_top_20pct"]
            for kind, statistic in (
                ("mean", "hard_minus_easy_mean_distance"),
                ("median", "hard_minus_easy_median_distance"),
            ):
                function = np.mean if kind == "mean" else np.median
                point = float(function(hard) - function(easy))
                low, high = bootstrap_groups(easy, hard, kind, rng)
                p_value = permutation_groups(easy, hard, kind, point, rng)
                legacy_rows.append({
                    "embedding_source": source,
                    "distance_metric": metric,
                    "statistic": statistic,
                    "value": point,
                    "ci_low": low,
                    "ci_high": high,
                    "p_value": p_value,
                    "hard_group_distance_summary": float(function(hard)),
                    "easy_group_distance_summary": float(function(easy)),
                })
    require(len(legacy_rows) == 36, "Legacy sensitivity must contain 36 rows.")
    for statistic in STATISTICS:
        indices = [
            i for i, row in enumerate(legacy_rows)
            if row["statistic"] == statistic
        ]
        require(len(indices) == 9, f"Legacy FDR family is not nine for {statistic}.")
        adjusted = bh([float(legacy_rows[i]["p_value"]) for i in indices])
        for index, value in zip(indices, adjusted):
            legacy_rows[index]["p_value_bh_fdr_within_statistic"] = value

    diagnostics = {
        "mahalanobis_condition_number": condition,
        "mahalanobis_shrinkage": float(covariance.shrinkage_),
        "hard_count": int(np.sum(labels == "hard_top_20pct")),
        "easy_count": int(np.sum(labels == "easy_bottom_20pct")),
        "middle_count": int(np.sum(labels == "middle_60pct")),
        "pre_head_equals_last_gcn_pool_test": True,
        "pre_head_equals_last_gcn_pool_reference": True,
    }
    return structure, rows, legacy_rows, diagnostics


def compare_numeric(actual: float, expected: float, label: str) -> None:
    require(
        math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12),
        f"Numerical mismatch for {label}: {actual} != {expected}",
    )


def validate_outputs(
    output: Path,
    expected_structure: pd.DataFrame,
    expected_rows: list[dict[str, Any]],
    expected_legacy_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    for name in PRODUCER_OUTPUTS:
        require((output / name).is_file(), f"Missing producer output: {name}")

    structure = pd.read_csv(output / "distance_error_by_structure.csv")
    require(len(structure) == 242 and structure["jid"].nunique() == 242,
            "Structure output must contain 242 unique JIDs.")
    require(structure["jid"].tolist() == expected_structure["jid"].tolist(),
            "Structure JID ordering differs.")
    require(structure["error_group"].tolist() == expected_structure["error_group"].tolist(),
            "Hard/easy labels differ.")
    for column in (
        "target_eV_per_atom",
        "prediction_eV_per_atom",
        "canonical_abs_error_eV_per_atom",
        *DISTANCES,
    ):
        require(
            np.allclose(
                structure[column].to_numpy(float),
                expected_structure[column].to_numpy(float),
                rtol=1e-12,
                atol=1e-12,
            ),
            f"Structure column differs: {column}",
        )
    require(structure["error_authority"].eq(
        "results/zero_shot/nitride/predictions.csv"
    ).all(), "Canonical error authority label differs.")

    stats = pd.read_csv(output / "distance_error_statistics.csv")
    require(len(stats) == 12, "Statistics output must contain 12 rows.")
    require(set(stats["embedding_source"]) == {"last_alignn_pool"},
            "Only last_alignn_pool is permitted.")
    require(set(stats["distance_metric"]) == set(DISTANCES), "Distance metric set differs.")
    require(set(stats["statistic"]) == set(STATISTICS), "Statistic set differs.")
    require(set(stats["bh_fdr_family_size"].astype(int)) == {3},
            "BH-FDR family size must be three.")
    for expected in expected_rows:
        selected = stats[
            stats["distance_metric"].eq(expected["distance_metric"])
            & stats["statistic"].eq(expected["statistic"])
        ]
        require(len(selected) == 1, f"Missing unique statistic row: {expected}")
        row = selected.iloc[0]
        for key in (
            "value", "ci_low", "ci_high", "p_value",
            "p_value_bh_fdr_within_statistic",
        ):
            compare_numeric(row[key], expected[key], f"{expected['distance_metric']}/{expected['statistic']}/{key}")
        if "hard_group_distance_summary" in expected:
            compare_numeric(
                row["hard_group_distance_summary"],
                expected["hard_group_distance_summary"],
                "hard summary",
            )
            compare_numeric(
                row["easy_group_distance_summary"],
                expected["easy_group_distance_summary"],
                "easy summary",
            )
        require(float(row["ci_low"]) <= float(row["value"]) <= float(row["ci_high"]),
                "Point estimate lies outside its bootstrap interval.")
        require(0 <= float(row["p_value"]) <= 1, "Invalid p-value.")
        require(0 <= float(row["p_value_bh_fdr_within_statistic"]) <= 1, "Invalid q-value.")

    legacy = pd.read_csv(
        output / "distance_error_legacy_nine_cell_sensitivity.csv"
    )
    require(len(legacy) == 36, "Legacy sensitivity output must contain 36 rows.")
    require(
        set(legacy["embedding_source"]) == {
            "pre_head", "last_alignn_pool", "last_gcn_pool"
        },
        "Legacy sensitivity embedding-source set differs.",
    )
    require(set(legacy["bh_fdr_family_size"].astype(int)) == {9},
            "Legacy BH-FDR family size must be nine.")
    for expected in expected_legacy_rows:
        selected_legacy = legacy[
            legacy["embedding_source"].eq(expected["embedding_source"])
            & legacy["distance_metric"].eq(expected["distance_metric"])
            & legacy["statistic"].eq(expected["statistic"])
        ]
        require(
            len(selected_legacy) == 1,
            f"Missing unique legacy sensitivity row: {expected}",
        )
        legacy_row = selected_legacy.iloc[0]
        for key in (
            "value", "ci_low", "ci_high", "p_value",
            "p_value_bh_fdr_within_statistic",
        ):
            compare_numeric(
                legacy_row[key],
                expected[key],
                f"legacy/{expected['embedding_source']}/"
                f"{expected['distance_metric']}/{expected['statistic']}/{key}",
            )
        if "hard_group_distance_summary" in expected:
            compare_numeric(
                legacy_row["hard_group_distance_summary"],
                expected["hard_group_distance_summary"],
                "legacy hard summary",
            )
            compare_numeric(
                legacy_row["easy_group_distance_summary"],
                expected["easy_group_distance_summary"],
                "legacy easy summary",
            )

    protocol = json.loads((output / "distance_error_protocol.json").read_text())
    require(protocol["canonical_error_source"] ==
            "results/zero_shot/nitride/predictions.csv",
            "Protocol canonical authority differs.")
    require(protocol["embedding_source"] == "last_alignn_pool",
            "Protocol embedding source differs.")
    require(protocol["bh_fdr"]["family_size"] == 3, "Protocol BH family differs.")
    require(
        protocol["legacy_nine_cell_sensitivity"]["cells_per_statistic"] == 9,
        "Protocol legacy BH family differs.",
    )
    require(
        protocol["legacy_nine_cell_sensitivity"][
            "pre_head_equals_last_gcn_pool_test"
        ] is True,
        "Protocol duplicate-representation guardrail differs.",
    )
    require(protocol["bootstrap"]["iterations"] == BOOTSTRAPS, "Bootstrap count differs.")
    require(protocol["permutation"]["iterations"] == PERMUTATIONS, "Permutation count differs.")

    decision = json.loads((output / "distance_error_decision.json").read_text())
    selected = stats[
        stats["distance_metric"].eq("oxide_knn5_mean_distance")
        & stats["statistic"].eq("spearman_correlation")
    ].iloc[0]
    expected_include = bool(
        (float(selected["ci_low"]) > 0 or float(selected["ci_high"]) < 0)
        and float(selected["p_value_bh_fdr_within_statistic"]) < 0.01
        and float(selected["value"]) > 0
    )
    require(bool(decision["C6_MAIN"]) == expected_include, "C6 decision rule differs.")
    require(decision["decision"] == (
        "INCLUDE_C6_IN_MAIN_PAPER" if expected_include else "DROP_C6_FROM_MAIN_PAPER"
    ), "C6 decision label differs.")
    compare_numeric(decision["spearman_rho"], selected["value"], "decision rho")
    compare_numeric(decision["ci_low"], selected["ci_low"], "decision CI low")
    compare_numeric(decision["ci_high"], selected["ci_high"], "decision CI high")
    compare_numeric(
        decision["bh_fdr_q_value"],
        selected["p_value_bh_fdr_within_statistic"],
        "decision q",
    )
    legacy_selected = legacy[
        legacy["embedding_source"].eq("last_alignn_pool")
        & legacy["distance_metric"].eq("oxide_knn5_mean_distance")
        & legacy["statistic"].eq("spearman_correlation")
    ].iloc[0]
    compare_numeric(
        decision["legacy_nine_cell_sensitivity"]["bh_fdr_q_value"],
        legacy_selected["p_value_bh_fdr_within_statistic"],
        "legacy decision q",
    )
    require(
        decision["legacy_nine_cell_sensitivity"]["family_size"] == 9,
        "Legacy decision family size differs.",
    )
    require(
        decision["legacy_nine_cell_sensitivity"]["role"]
        == "sensitivity only; does not control C6_MAIN",
        "Legacy sensitivity role differs.",
    )
    legacy_include = bool(
        float(legacy_selected["ci_low"]) > 0
        and float(legacy_selected["p_value_bh_fdr_within_statistic"]) < 0.01
        and float(legacy_selected["value"]) > 0
    )
    require(
        bool(decision["legacy_nine_cell_decision_unchanged"])
        == bool(legacy_include == expected_include),
        "Legacy sensitivity decision-consistency flag differs.",
    )

    manifest = pd.read_csv(output / "input_output_manifest.csv")
    manifest_categories = set(manifest["category"])
    require(
        manifest_categories
        in (
            {"authoritative_input", "generated_output"},
            {
                "authoritative_input",
                "quarantined_preservation_reference",
                "generated_output",
            },
        ),
        "Manifest categories differ.",
    )
    for row in manifest.itertuples(index=False):
        if row.category == "generated_output":
            candidate = output / Path(str(row.path)).name
        else:
            candidate = ROOT / str(row.path)
        require(candidate.is_file(), f"Manifest path is broken: {row.path}")
        require(sha256(candidate) == str(row.sha256), f"Manifest hash differs: {row.path}")

    return {
        "C6_MAIN": expected_include,
        "decision": decision["decision"],
        "spearman_rho": float(selected["value"]),
        "ci_low": float(selected["ci_low"]),
        "ci_high": float(selected["ci_high"]),
        "bh_fdr_q_value": float(selected["p_value_bh_fdr_within_statistic"]),
        "legacy_nine_cell_bh_fdr_q_value": float(
            legacy_selected["p_value_bh_fdr_within_statistic"]
        ),
        "legacy_nine_cell_decision_unchanged": bool(
            legacy_include == expected_include
        ),
    }


def validate_determinism(output: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with tempfile.TemporaryDirectory(prefix="domain_shift_distance_error_a_") as first_text, \
            tempfile.TemporaryDirectory(prefix="domain_shift_distance_error_b_") as second_text:
        first = Path(first_text) / "package"
        second = Path(second_text) / "package"
        for target in (first, second):
            completed = subprocess.run(
                [sys.executable, str(PRODUCER), "--output-dir", str(target)],
                cwd=ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )
            require(
                completed.returncode == 0,
                f"Determinism producer run failed: {completed.stderr}\n{completed.stdout}",
            )
        hashes: dict[str, str] = {}
        for name in PRODUCER_OUTPUTS:
            first_hash = sha256(first / name)
            second_hash = sha256(second / name)
            final_hash = sha256(output / name)
            require(first_hash == second_hash, f"Two producer runs differ: {name}")
            require(first_hash == final_hash, f"Final output differs from deterministic rerun: {name}")
            hashes[name] = first_hash
        return hashes


def write_validation(
    output: Path,
    payload: dict[str, Any],
) -> None:
    validation_path = output / "distance_error_validation.json"
    validation_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    checksum_paths = [
        PRODUCER,
        Path(__file__).resolve(),
        *(output / name for name in PRODUCER_OUTPUTS),
        validation_path,
    ]
    lines = [
        f"{sha256(path)}  {path.resolve().relative_to(ROOT.resolve()).as_posix()}"
        for path in checksum_paths
    ]
    (output / "distance_error_evidence.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    reject_forbidden(output)
    require(output.is_dir(), f"Missing output directory: {output}")
    frozen = verify_frozen_hashes()
    quarantined_before = verify_quarantined_snapshot(output)
    expected_structure, expected_rows, expected_legacy_rows, diagnostics = (
        independently_load_and_compute()
    )
    decision = validate_outputs(
        output,
        expected_structure,
        expected_rows,
        expected_legacy_rows,
    )
    deterministic_hashes = validate_determinism(output)
    frozen_after = verify_frozen_hashes()
    quarantined_after = verify_quarantined_snapshot(output)
    require(frozen == frozen_after, "A canonical frozen input changed during validation.")
    require(
        quarantined_before == quarantined_after,
        "A quarantined output changed during validation.",
    )
    payload = {
        "status": "DISTANCE_ERROR_RECOMPUTE_VALIDATED",
        "READY_FOR_validation_check": True,
        "producer": "scripts/analysis/recompute_distance_error_canonical.py",
        "validator": "scripts/analysis/validate_distance_error_canonical.py",
        "canonical_join": {
            "validated": 242,
            "expected": 242,
            "missing": 0,
            "extra": 0,
        },
        "embedding": {
            "source": "last_alignn_pool",
            "test_shape": [1726, 256],
            "nitride_count": 242,
            "oxide_reference_shape": [13507, 256],
            **diagnostics,
        },
        "statistics": {
            "rows_validated": 12,
            "bootstrap_iterations": BOOTSTRAPS,
            "permutation_iterations": PERMUTATIONS,
            "bh_fdr_family_size": 3,
            "independent_recomputation_match": True,
        },
        "legacy_nine_cell_sensitivity": {
            "rows_validated": 36,
            "cells_per_statistic": 9,
            "bh_fdr_q_value": decision[
                "legacy_nine_cell_bh_fdr_q_value"
            ],
            "decision_unchanged": decision[
                "legacy_nine_cell_decision_unchanged"
            ],
            "pre_head_equals_last_gcn_pool": True,
            "role": "sensitivity only; primary q3 controls C6_MAIN",
        },
        "decision": decision,
        "determinism": {
            "two_fresh_runs_byte_identical": True,
            "fresh_runs_match_final_package": True,
            "producer_output_hashes": deterministic_hashes,
        },
        "frozen_hashes_verified_before_and_after": frozen_after,
        "quarantined_outputs_unchanged": True if quarantined_after else None,
        "quarantined_output_hashes_verified": len(quarantined_after),
        "quarantined_preservation_status": (
            "verified" if quarantined_after else "not_available_in_public_release"
        ),
        "safety": {
            "section08_inspected": False,
            "section08_used": False,
            "nested_checkout_used": False,
            "metadata_predictions_used_numerically": False,
            "canonical_inputs_modified": False,
        },
    }
    write_validation(output, payload)
    evidence_path = output / "distance_error_evidence.sha256"
    require(evidence_path.is_file(), "Evidence checksum file was not written.")
    print(json.dumps({
        "status": payload["status"],
        "READY_FOR_validation_check": payload["READY_FOR_validation_check"],
        **decision,
        "evidence_checksum_entries": len(evidence_path.read_text().splitlines()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
