#!/usr/bin/env python
"""Optional appendix analysis of pretrained ALIGNN element embeddings.

This script is intentionally separate from the structure-level domain-shift
analysis. It reads the local pretrained checkpoint/config only, constructs an
element-level atom-embedding table for Z=1..80, and writes appendix-only plots
and tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
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
from jarvis.core.specie import Specie, atomic_numbers_to_symbols, get_node_attributes
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DEFAULT_CHECKPOINT = "models/pretrained/checkpoint_300.pt"
DEFAULT_CONFIG = "configs/pretrained/config.json"
DEFAULT_ARTIFACT_DIR = "results/embeddings/element_appendix"
DEFAULT_FIGURE_DIR = "results/reproduction/embeddings/figures/element_appendix"
DEFAULT_TABLE_DIR = "results/reproduction/embeddings/tables"
DEFAULT_REPORT = "results/reproduction/embeddings/appendix_element_embeddings.md"
DEFAULT_SEED = 42
DEFAULT_PERMUTATIONS = 5000
ELEMENT_Z_MIN = 1
ELEMENT_Z_MAX = 80
TSNE_PERPLEXITY = 15
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
PROPERTY_SPECS = {
    "electronegativity": {
        "label": "Electronegativity",
        "missing_policy": "values <= 0 treated as missing",
    },
    "atomic_radius": {
        "label": "Atomic radius",
        "missing_policy": "nonfinite values treated as missing",
    },
    "valence_electrons": {
        "label": "Valence electrons",
        "missing_policy": "nonfinite values treated as missing",
    },
}
CHEMISTRY_COLORS = {
    "alkali metal": "#E69F00",
    "alkaline earth metal": "#F0E442",
    "transition metal": "#0072B2",
    "lanthanide": "#56B4E9",
    "post-transition metal": "#999999",
    "metalloid": "#CC79A7",
    "nonmetal": "#009E73",
    "halogen": "#D55E00",
    "noble gas": "#8A8A8A",
    "other": "#333333",
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create optional appendix plots/tables for pretrained atom embeddings."
    )
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--table-dir", default=DEFAULT_TABLE_DIR)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Replace existing appendix element-embedding outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect checkpoint/config and planned outputs without writing analysis files.",
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


def planned_outputs(artifact_dir: Path, figure_dir: Path, table_dir: Path, report: Path) -> list[Path]:
    return [
        artifact_dir / "element_embedding_table.npy",
        artifact_dir / "element_embedding_table.csv",
        artifact_dir / "atom_embedding_layer0_weight.npy",
        artifact_dir / "element_embedding_coordinates.csv",
        artifact_dir / "element_embedding_appendix_manifest.json",
        table_dir / "element_embedding_property_correlations.csv",
        figure_dir / "element_embedding_pca.png",
        figure_dir / "element_embedding_pca.pdf",
        figure_dir / "element_embedding_tsne.png",
        figure_dir / "element_embedding_tsne.pdf",
        figure_dir / "element_embedding_umap.png",
        figure_dir / "element_embedding_umap.pdf",
        report,
    ]


def refuse_existing(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing appendix element-embedding outputs. "
            f"Use --allow-overwrite to regenerate them:\n{joined}"
        )


def load_checkpoint_model_state(checkpoint_path: Path) -> dict[str, Any]:
    import torch

    try:
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    if not isinstance(model_state, dict):
        raise TypeError(f"Checkpoint did not contain a model state dict: {checkpoint_path}")
    return model_state


def element_symbol(z: int) -> str:
    return atomic_numbers_to_symbols([z])[0]


def chemistry_class(symbol: str, z: int) -> str:
    if symbol in {"He", "Ne", "Ar", "Kr", "Xe", "Rn"}:
        return "noble gas"
    if symbol in {"F", "Cl", "Br", "I", "At"}:
        return "halogen"
    if symbol in {"H", "C", "N", "O", "P", "S", "Se"}:
        return "nonmetal"
    if symbol in {"B", "Si", "Ge", "As", "Sb", "Te"}:
        return "metalloid"
    if symbol in {"Li", "Na", "K", "Rb", "Cs", "Fr"}:
        return "alkali metal"
    if symbol in {"Be", "Mg", "Ca", "Sr", "Ba", "Ra"}:
        return "alkaline earth metal"
    if 57 <= z <= 71:
        return "lanthanide"
    if symbol in {"Al", "Ga", "In", "Sn", "Tl", "Pb", "Bi", "Po"}:
        return "post-transition metal"
    if (
        21 <= z <= 30
        or 39 <= z <= 48
        or 72 <= z <= 80
    ):
        return "transition metal"
    return "other"


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if np.isfinite(result) else math.nan


def element_properties(symbol: str) -> dict[str, float]:
    specie = Specie(symbol)
    electronegativity = finite_or_nan(specie.X)
    if electronegativity <= 0:
        electronegativity = math.nan
    atomic_radius = finite_or_nan(specie.atomic_rad)
    valence_parts = [
        finite_or_nan(specie.element_property(name))
        for name in ("nsvalence", "npvalence", "ndvalence", "nfvalence")
    ]
    valence_electrons = float(np.nansum(valence_parts))
    if not np.isfinite(valence_electrons):
        valence_electrons = math.nan
    return {
        "electronegativity": electronegativity,
        "atomic_radius": atomic_radius,
        "valence_electrons": valence_electrons,
    }


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def construct_element_table(
    model_state: dict[str, Any],
    config_payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    atom_features = config_payload.get("atom_features", "cgcnn")
    if atom_features != "cgcnn":
        raise ValueError(
            "This appendix constructor expects the local pretrained config to use "
            f"atom_features='cgcnn'; found {atom_features!r}."
        )

    direct_key = next(
        (key for key in ("atom_embedding.weight", "module.atom_embedding.weight") if key in model_state),
        None,
    )
    if direct_key is not None:
        table = model_state[direct_key].detach().cpu().numpy().astype(np.float64)
        if table.shape[0] <= ELEMENT_Z_MAX:
            raise ValueError(f"{direct_key} has too few rows for Z=1..80: {table.shape}")
        return (
            table[ELEMENT_Z_MIN : ELEMENT_Z_MAX + 1],
            table,
            {
                "embedding_source": direct_key,
                "construction": "direct_embedding_table_rows_1_to_80_padding_row_0_skipped",
                "raw_weight_shape": list(table.shape),
                "post_batchnorm_silu": False,
            },
        )

    linear_key = "atom_embedding.layer.0.weight"
    bias_key = "atom_embedding.layer.0.bias"
    bn_weight_key = "atom_embedding.layer.1.weight"
    bn_bias_key = "atom_embedding.layer.1.bias"
    bn_mean_key = "atom_embedding.layer.1.running_mean"
    bn_var_key = "atom_embedding.layer.1.running_var"
    required = [
        linear_key,
        bias_key,
        bn_weight_key,
        bn_bias_key,
        bn_mean_key,
        bn_var_key,
    ]
    missing = [key for key in required if key not in model_state]
    if missing:
        raise KeyError(
            "Could not find a direct atom embedding table or the expected ALIGNN "
            f"atom_embedding MLP keys. Missing: {missing}"
        )

    weight = model_state[linear_key].detach().cpu().numpy().astype(np.float64)
    bias = model_state[bias_key].detach().cpu().numpy().astype(np.float64)
    bn_weight = model_state[bn_weight_key].detach().cpu().numpy().astype(np.float64)
    bn_bias = model_state[bn_bias_key].detach().cpu().numpy().astype(np.float64)
    bn_mean = model_state[bn_mean_key].detach().cpu().numpy().astype(np.float64)
    bn_var = model_state[bn_var_key].detach().cpu().numpy().astype(np.float64)
    eps = 1e-5

    feature_rows: list[np.ndarray] = []
    for z in range(ELEMENT_Z_MIN, ELEMENT_Z_MAX + 1):
        symbol = element_symbol(z)
        feature_rows.append(np.asarray(get_node_attributes(symbol, atom_features="cgcnn"), dtype=np.float64))
    features = np.vstack(feature_rows)
    if features.shape[1] != weight.shape[1]:
        raise ValueError(
            f"CGCNN feature width {features.shape[1]} does not match {linear_key} width {weight.shape[1]}."
        )
    linear = features @ weight.T + bias
    normalized = (linear - bn_mean) / np.sqrt(bn_var + eps)
    table = silu(normalized * bn_weight + bn_bias)
    return (
        table,
        weight,
        {
            "embedding_source": "atom_embedding.layer.0.weight plus BatchNorm1d and SiLU",
            "construction": (
                "effective_element_embeddings_for_Z_1_to_80_from_cgcnn_features_"
                "through_pretrained_atom_embedding_mlp_eval_mode"
            ),
            "raw_weight_key": linear_key,
            "raw_weight_shape": list(weight.shape),
            "cgcnn_feature_shape": list(features.shape),
            "post_batchnorm_silu": True,
            "padding_row_status": (
                "no explicit padding row exists for this CGCNN-feature MLP; "
                "rows are constructed directly for Z=1..80"
            ),
        },
    )


def build_element_metadata(table: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row_index, z in enumerate(range(ELEMENT_Z_MIN, ELEMENT_Z_MAX + 1)):
        symbol = element_symbol(z)
        props = element_properties(symbol)
        rows.append(
            {
                "row_index": row_index,
                "Z": z,
                "symbol": symbol,
                "chemistry_class": chemistry_class(symbol, z),
                "electronegativity": props["electronegativity"],
                "atomic_radius": props["atomic_radius"],
                "valence_electrons": props["valence_electrons"],
                "embedding_dim": table.shape[1],
            }
        )
    return pd.DataFrame(rows)


def fit_coordinates(table: np.ndarray, seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    scaled = StandardScaler().fit_transform(table)
    pca = PCA(n_components=2, random_state=seed)
    pca_coords = pca.fit_transform(scaled)
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        metric="euclidean",
        max_iter=1000,
    )
    tsne_coords = tsne.fit_transform(scaled)
    umap_module = import_umap()
    reducer = umap_module.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="euclidean",
        random_state=seed,
        transform_seed=seed,
        n_jobs=1,
    )
    umap_coords = reducer.fit_transform(scaled)
    coords = pd.DataFrame(
        {
            "pca1": pca_coords[:, 0],
            "pca2": pca_coords[:, 1],
            "tsne1": tsne_coords[:, 0],
            "tsne2": tsne_coords[:, 1],
            "umap1": umap_coords[:, 0],
            "umap2": umap_coords[:, 1],
        }
    )
    diagnostics = {
        "standardized_before_projection": True,
        "pca_explained_variance_ratio": [float(v) for v in pca.explained_variance_ratio_],
        "tsne_perplexity": TSNE_PERPLEXITY,
        "tsne_init": "pca initialization only",
        "tsne_kl_divergence": float(getattr(tsne, "kl_divergence_", np.nan)),
        "umap_n_neighbors": UMAP_N_NEIGHBORS,
        "umap_min_dist": UMAP_MIN_DIST,
    }
    return coords, diagnostics


def import_umap() -> Any:
    try:
        import umap  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "UMAP is required for this appendix script. Install umap-learn in the active "
            f"environment. Original error: {exc}"
        ) from exc
    return umap


def setup_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.16,
            "grid.linewidth": 0.6,
        }
    )


def save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    root = repo_root()
    for ext in ("png", "pdf"):
        path = figure_dir / f"{stem}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        paths.append(relpath(path, root))
    plt.close(fig)
    return paths


def plot_embedding(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    figure_dir: Path,
    stem: str,
) -> list[str]:
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    for class_name, color in CHEMISTRY_COLORS.items():
        sub = df[df["chemistry_class"] == class_name]
        if sub.empty:
            continue
        ax.scatter(
            sub[x_col],
            sub[y_col],
            s=42,
            color=color,
            edgecolors="white",
            linewidths=0.45,
            alpha=0.9,
            label=class_name,
        )
    for _, row in df.iterrows():
        ax.text(row[x_col], row[y_col], row["symbol"], fontsize=7, ha="center", va="center")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    return save_figure(fig, figure_dir, stem)


def centered(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values - float(np.mean(values))


def corr_value(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def fast_corr_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x_centered = x - np.mean(x, axis=0, keepdims=True)
    y_centered = centered(y)
    denom = np.sqrt(np.sum(x_centered * x_centered, axis=0) * np.sum(y_centered * y_centered))
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.sum(x_centered * y_centered[:, None], axis=0) / denom
    return out


def max_abs_permutation_p_values(
    x: np.ndarray,
    y: np.ndarray,
    observed: np.ndarray,
    method: str,
    rng: np.random.Generator,
    permutations: int,
) -> np.ndarray:
    if method == "spearman":
        x_for_perm = np.apply_along_axis(rankdata, 0, x)
        y_for_perm = rankdata(y, method="average")
    else:
        x_for_perm = x
        y_for_perm = y
    counts = np.zeros(x.shape[1], dtype=int)
    observed_abs = np.abs(observed)
    for _ in range(permutations):
        permuted = rng.permutation(y_for_perm)
        perm_corr = fast_corr_matrix(x_for_perm, permuted)
        max_abs = float(np.nanmax(np.abs(perm_corr)))
        counts += max_abs >= observed_abs
    return (counts + 1) / (permutations + 1)


def property_correlations(
    table: np.ndarray,
    metadata: pd.DataFrame,
    seed: int,
    permutations: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for property_name, spec in PROPERTY_SPECS.items():
        y_all = metadata[property_name].to_numpy(dtype=float)
        mask = np.isfinite(y_all)
        x = table[mask]
        y = y_all[mask]
        for method in ("spearman", "pearson"):
            observed = np.array(
                [corr_value(x[:, dim], y, method) for dim in range(x.shape[1])],
                dtype=float,
            )
            p_adj = max_abs_permutation_p_values(
                x,
                y,
                observed,
                method,
                rng,
                permutations,
            )
            for dim, (corr, p_value) in enumerate(zip(observed, p_adj)):
                rows.append(
                    {
                        "property": property_name,
                        "property_label": spec["label"],
                        "method": method,
                        "embedding_dimension": dim,
                        "correlation": corr,
                        "abs_correlation": abs(corr),
                        "max_abs_permutation_p_value": p_value,
                        "n_elements": int(mask.sum()),
                        "multiple_comparison_scope": "all_embedding_dimensions_within_property_and_method",
                        "missing_policy": spec["missing_policy"],
                    }
                )
    return rows


def write_report(
    path: Path,
    manifest: dict[str, Any],
    corr_rows: list[dict[str, Any]],
    figure_paths: list[str],
) -> None:
    corr_df = pd.DataFrame(corr_rows)
    lines = [
        "# Appendix: Element Embedding Sanity Check",
        "",
        "This appendix-only analysis inspects the pretrained atom-embedding layer as a chemistry sanity check. It is not part of the main oxide-vs-nitride structure-level domain-shift result.",
        "",
        "## Scope",
        "",
        "- Source checkpoint: `models/pretrained/checkpoint_300.pt`.",
        "- Source config: `configs/pretrained/config.json`.",
        "- Elements included: atomic numbers `Z=1..80`.",
        "- The local ALIGNN config uses CGCNN atom features and an atom-embedding MLP rather than a literal `nn.Embedding` lookup table.",
        "- The saved element table therefore uses each element's CGCNN feature vector passed through the pretrained `atom_embedding` MLP in eval mode.",
        "- This is a chemistry sanity check only; structure-level raw-space analyses remain the primary domain-shift evidence.",
        "",
        "## Projection Outputs",
        "",
        f"- PCA explained variance: PC1 `{manifest['projection_diagnostics']['pca_explained_variance_ratio'][0] * 100:.2f}%`, PC2 `{manifest['projection_diagnostics']['pca_explained_variance_ratio'][1] * 100:.2f}%`.",
        f"- t-SNE: direct on standardized element embeddings, perplexity `{manifest['projection_diagnostics']['tsne_perplexity']}`.",
        f"- UMAP: direct on standardized element embeddings, n_neighbors `{manifest['projection_diagnostics']['umap_n_neighbors']}`, min_dist `{manifest['projection_diagnostics']['umap_min_dist']}`.",
        "",
        "## Strongest Property-Dimension Correlations",
        "",
        "Permutation p-values control the max absolute correlation across all embedding dimensions within each property and correlation method.",
        "",
        "| Property | Method | Dimension | Correlation | Max-abs permutation p | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    if not corr_df.empty:
        top = (
            corr_df.sort_values(["property", "method", "abs_correlation"], ascending=[True, True, False])
            .groupby(["property", "method"], as_index=False)
            .head(3)
        )
        for _, row in top.iterrows():
            lines.append(
                f"| {row['property_label']} | {row['method']} | {int(row['embedding_dimension'])} | "
                f"{float(row['correlation']):.3f} | {float(row['max_abs_permutation_p_value']):.4f} | "
                f"{int(row['n_elements'])} |"
            )
    lines.extend(["", "## Files", ""])
    for key, value in manifest["outputs"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Figures", ""])
    for figure_path in figure_paths:
        lines.append(f"- `{figure_path}`")
    lines.extend(
        [
            "",
            "## Interpretation Guardrail",
            "",
            "Element-level patterns can indicate whether the pretrained atom-embedding layer encodes broad chemical trends, but they should not be interpreted as evidence for the oxide-vs-nitride structure-level mechanism.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = repo_root()
    checkpoint = (root / args.checkpoint).resolve()
    config = (root / args.config).resolve()
    artifact_dir = (root / args.artifact_dir).resolve()
    figure_dir = (root / args.figure_dir).resolve()
    table_dir = (root / args.table_dir).resolve()
    report = (root / args.report).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not config.exists():
        raise FileNotFoundError(f"Missing config: {config}")

    refuse_existing(
        planned_outputs(artifact_dir, figure_dir, table_dir, report),
        allow_overwrite=args.allow_overwrite,
    )

    config_payload = json.loads(config.read_text(encoding="utf-8"))
    model_state = load_checkpoint_model_state(checkpoint)
    table, raw_weight, construction = construct_element_table(model_state, config_payload)

    if args.dry_run:
        print("Dry-run only. Optional element embedding appendix:")
        print(f"checkpoint: {relpath(checkpoint, root)}")
        print(f"config: {relpath(config, root)}")
        print(f"element table shape: {table.shape}")
        print(f"raw weight shape: {raw_weight.shape}")
        print(f"construction: {construction['construction']}")
        print(f"figures: {relpath(figure_dir, root)}")
        print(f"report: {relpath(report, root)}")
        return

    setup_plot_style()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metadata = build_element_metadata(table)
    coords, projection_diagnostics = fit_coordinates(table, args.seed)
    embedding_columns = pd.DataFrame(
        table,
        columns=[f"embedding_{dim:03d}" for dim in range(table.shape[1])],
    )
    element_df = pd.concat([metadata, coords, embedding_columns], axis=1)

    corr_rows = property_correlations(table, metadata, args.seed, args.permutations)

    np.save(artifact_dir / "element_embedding_table.npy", table)
    np.save(artifact_dir / "atom_embedding_layer0_weight.npy", raw_weight)
    element_df.to_csv(artifact_dir / "element_embedding_table.csv", index=False)
    element_df[
        [
            "Z",
            "symbol",
            "chemistry_class",
            "pca1",
            "pca2",
            "tsne1",
            "tsne2",
            "umap1",
            "umap2",
        ]
    ].to_csv(artifact_dir / "element_embedding_coordinates.csv", index=False)
    write_csv(
        table_dir / "element_embedding_property_correlations.csv",
        corr_rows,
        [
            "property",
            "property_label",
            "method",
            "embedding_dimension",
            "correlation",
            "abs_correlation",
            "max_abs_permutation_p_value",
            "n_elements",
            "multiple_comparison_scope",
            "missing_policy",
        ],
    )

    figure_paths: list[str] = []
    figure_paths.extend(
        plot_embedding(
            element_df,
            "pca1",
            "pca2",
            "Element embedding PCA",
            f"PC1 ({projection_diagnostics['pca_explained_variance_ratio'][0] * 100:.1f}%)",
            f"PC2 ({projection_diagnostics['pca_explained_variance_ratio'][1] * 100:.1f}%)",
            figure_dir,
            "element_embedding_pca",
        )
    )
    figure_paths.extend(
        plot_embedding(
            element_df,
            "tsne1",
            "tsne2",
            "Element embedding direct t-SNE",
            "t-SNE 1",
            "t-SNE 2",
            figure_dir,
            "element_embedding_tsne",
        )
    )
    figure_paths.extend(
        plot_embedding(
            element_df,
            "umap1",
            "umap2",
            "Element embedding direct UMAP",
            "UMAP 1",
            "UMAP 2",
            figure_dir,
            "element_embedding_umap",
        )
    )

    manifest = {
        "created_at_unix": time.time(),
        "created_by": "scripts/embedding_analysis/09_optional_element_embedding_appendix.py",
        "appendix_only": True,
        "not_main_domain_shift_result": True,
        "inputs": {
            "checkpoint": relpath(checkpoint, root),
            "config": relpath(config, root),
        },
        "outputs": {
            "element_embedding_table_npy": relpath(artifact_dir / "element_embedding_table.npy", root),
            "element_embedding_table_csv": relpath(artifact_dir / "element_embedding_table.csv", root),
            "raw_atom_embedding_weight_npy": relpath(artifact_dir / "atom_embedding_layer0_weight.npy", root),
            "coordinates_csv": relpath(artifact_dir / "element_embedding_coordinates.csv", root),
            "property_correlations_csv": relpath(
                table_dir / "element_embedding_property_correlations.csv",
                root,
            ),
            "figure_dir": relpath(figure_dir, root),
            "report": relpath(report, root),
        },
        "element_range": {"z_min": ELEMENT_Z_MIN, "z_max": ELEMENT_Z_MAX, "n": table.shape[0]},
        "embedding_dim": table.shape[1],
        "construction": construction,
        "projection_diagnostics": projection_diagnostics,
        "correlation_properties": list(PROPERTY_SPECS),
        "permutation_iterations": args.permutations,
        "seed": args.seed,
        "figures": figure_paths,
    }
    write_json(artifact_dir / "element_embedding_appendix_manifest.json", manifest)
    write_report(report, manifest, corr_rows, figure_paths)

    print(f"Element embeddings: {relpath(artifact_dir / 'element_embedding_table.csv', root)}")
    print(f"Property correlations: {relpath(table_dir / 'element_embedding_property_correlations.csv', root)}")
    print(f"Figures: {relpath(figure_dir, root)}")
    print(f"Report: {relpath(report, root)}")


if __name__ == "__main__":
    main()
