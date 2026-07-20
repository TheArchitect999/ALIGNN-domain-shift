#!/usr/bin/env python3
"""Generate the paper figures from governed evidence only.

The producer deliberately writes a new package under
``results/derived_evidence/figures_v1``.  It never reads an old
figure, an editorial manuscript, the excluded nested checkout, or forbidden
Section 08.  Dense plot coordinates are written beside the images so every
mark remains auditable even when it is not a printed numerical annotation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


_CACHE = Path(tempfile.gettempdir()) / "domain_shift_paper_figures_matplotlib"
_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "domain_shift_paper_figures_xdg"))
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams.update(
    {
        "svg.hashsalt": "domain_shift-paper-figures-v1",
        "svg.fonttype": "none",
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.55,
        "savefig.dpi": 360,
    }
)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


FAMILIES = ("oxide", "nitride")
NS = (10, 50, 100, 200, 500, 1000)
SEEDS = (0, 1, 2, 3, 4)
COMPARISON_NS = (50, 500)
PNG_DPI = 360
OXIDE_COLOR = "#0072B2"
NITRIDE_COLOR = "#D55E00"
FAMILY_STYLE = {
    "oxide": {"color": OXIDE_COLOR, "marker": "o", "linestyle": "-"},
    "nitride": {"color": NITRIDE_COLOR, "marker": "s", "linestyle": "--"},
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def rel(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reject_forbidden_input(root: Path, path: Path) -> None:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"Paper input must be inside the repository: {path}") from exc
    if "archived_submission_materials" in relative.parts:
        raise SystemExit(f"Forbidden Section 08 input: {relative}")
    if relative.parts and relative.parts[0] == root.name:
        raise SystemExit(f"Excluded nested-checkout input: {relative}")


def require_file(root: Path, path: Path) -> Path:
    reject_forbidden_input(root, path)
    if not path.is_file():
        raise SystemExit(f"Missing required input: {rel(root, path)}")
    return path


def require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SystemExit(f"{label} is missing columns: {', '.join(missing)}")


def assert_close(left: float, right: float, label: str, atol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol):
        raise SystemExit(f"{label} mismatch: {left!r} != {right!r}")


def json_native(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False, default=json_native) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise SystemExit(f"Refusing to write empty plot-data CSV: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: json_native(value) for key, value in row.items()} for row in rows])


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = [line.rstrip() for line in text.splitlines() if not line.lstrip().startswith("<dc:date>")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    fig.savefig(
        png,
        dpi=PNG_DPI,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Software": "ALIGNN domain-shift deterministic figure producer"},
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "ALIGNN domain-shift deterministic figure producer", "Date": None},
    )
    plt.close(fig)
    normalize_svg(svg)
    return [png, svg]


def input_paths(root: Path) -> dict[str, Path]:
    paths = {
        "oxide_summary": root / "data/oxide/summaries/summary.json",
        "nitride_summary": root / "data/nitride/summaries/summary.json",
        "family_definition": root / "results/derived_evidence/provenance_dataset_closure/2_3B_dataset_integrity/oxynitride_definition_summary.json",
        "checkpoint_overlap": root / "results/derived_evidence/provenance_dataset_closure/2_3A_checkpoint_provenance/checkpoint_test_overlap.csv",
        "zero_shot_bootstrap": root / "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/zero_shot_bootstrap_summary.csv",
        "finetune_runs": root / "results/summaries/protocol_1/finetune/finetune_runs.csv",
        "protocol_1_aggregate": root / "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv",
        "comparison_seeds": root / "results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_seed_values.csv",
        "comparison_groups": root / "results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv",
        "test_embeddings": root / "results/embeddings/embeddings/test_set/structure_embeddings.npz",
        "test_embedding_metadata": root / "results/embeddings/embeddings/test_set/structure_embedding_metadata.csv",
        "balanced_embeddings": root / "results/embeddings/embeddings/balanced_pool/structure_embeddings.npz",
        "balanced_embedding_metadata": root / "results/embeddings/embeddings/balanced_pool/structure_embedding_metadata.csv",
        "embedding_metrics": root / "results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv",
        "distance_decision": root / "results/derived_evidence/distance_error_recompute/distance_error_decision.json",
        "distance_statistics": root / "results/derived_evidence/distance_error_recompute/distance_error_statistics.csv",
        "distance_by_structure": root / "results/derived_evidence/distance_error_recompute/distance_error_by_structure.csv",
        "distance_validation": root / "results/derived_evidence/distance_error_recompute/distance_error_validation.json",
    }
    return {label: require_file(root, path) for label, path in paths.items()}


def validate_finetune_runs(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, {"family", "N", "seed", "test_mae_eV_per_atom", "best_epoch"}, "protocol_1 fine-tuning run table")
    frame = frame.copy()
    frame["family"] = frame["family"].astype(str).str.lower()
    frame["N"] = pd.to_numeric(frame["N"], errors="raise").astype(int)
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    frame["test_mae_eV_per_atom"] = pd.to_numeric(frame["test_mae_eV_per_atom"], errors="raise")
    frame["best_epoch"] = pd.to_numeric(frame["best_epoch"], errors="raise").astype(int)
    if frame.duplicated(["family", "N", "seed"]).any():
        raise SystemExit("protocol_1 fine-tuning run table contains duplicate family/N/seed rows")
    expected = {(family, n_value, seed) for family in FAMILIES for n_value in NS for seed in SEEDS}
    actual = set(zip(frame["family"], frame["N"], frame["seed"]))
    if actual != expected or len(frame) != 60:
        raise SystemExit(f"protocol_1 fine-tuning run matrix mismatch: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    if not np.isfinite(frame[["test_mae_eV_per_atom", "best_epoch"]].to_numpy(dtype=float)).all():
        raise SystemExit("protocol_1 run table contains non-finite values")
    return frame.sort_values(["family", "N", "seed"]).reset_index(drop=True)


def validate_protocol_1_aggregates(runs: pd.DataFrame, aggregate: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        aggregate,
        {
            "family",
            "N",
            "runs",
            "raw_seed_test_mae_eV_per_atom",
            "raw_seed_best_epochs",
            "mean_test_mae_eV_per_atom",
            "sample_std_test_mae_eV_per_atom_ddof1",
            "zero_shot_mae_eV_per_atom",
            "validation_status",
        },
        "protocol_1 aggregate recomputation",
    )
    aggregate = aggregate.copy()
    aggregate["family"] = aggregate["family"].astype(str).str.lower()
    aggregate["N"] = pd.to_numeric(aggregate["N"], errors="raise").astype(int)
    if len(aggregate) != 12 or aggregate.duplicated(["family", "N"]).any():
        raise SystemExit("protocol_1 aggregate recomputation must contain 12 unique family/N rows")
    index = aggregate.set_index(["family", "N"])
    for (family, n_value), group in runs.groupby(["family", "N"], sort=True):
        supplied = index.loc[(family, n_value)]
        if int(supplied["runs"]) != 5 or supplied["validation_status"] != "PASS":
            raise SystemExit(f"protocol_1 aggregate is not validated for {family}/N={n_value}")
        values = group.sort_values("seed")["test_mae_eV_per_atom"].to_numpy(dtype=float)
        epochs = group.sort_values("seed")["best_epoch"].to_numpy(dtype=int).tolist()
        assert_close(values.mean(), supplied["mean_test_mae_eV_per_atom"], f"protocol_1 mean {family}/N={n_value}")
        assert_close(values.std(ddof=1), supplied["sample_std_test_mae_eV_per_atom_ddof1"], f"protocol_1 sample SD {family}/N={n_value}")
        supplied_values = np.asarray(json.loads(supplied["raw_seed_test_mae_eV_per_atom"]), dtype=float)
        if not np.allclose(values, supplied_values, rtol=1e-10, atol=1e-12):
            raise SystemExit(f"protocol_1 seed vector mismatch for {family}/N={n_value}")
        if epochs != list(map(int, json.loads(supplied["raw_seed_best_epochs"]))):
            raise SystemExit(f"protocol_1 best-epoch vector mismatch for {family}/N={n_value}")
    return aggregate.sort_values(["family", "N"]).reset_index(drop=True)


def draw_box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, *, face: str, edge: str, fontsize: float = 8.0) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        linewidth=1.0,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.25)


def render_figure1(root: Path, out: Path, paths: dict[str, Path]) -> tuple[list[Path], list[dict[str, Any]]]:
    oxide = json.loads(paths["oxide_summary"].read_text(encoding="utf-8"))
    nitride = json.loads(paths["nitride_summary"].read_text(encoding="utf-8"))
    definition = json.loads(paths["family_definition"].read_text(encoding="utf-8"))
    overlap = pd.read_csv(paths["checkpoint_overlap"])
    if int(oxide["counts"]["test"]) != 1484 or int(nitride["counts"]["test"]) != 242:
        raise SystemExit("Family fixed-test counts do not match governed values")
    if set(overlap["family"]) != set(FAMILIES) or not (pd.to_numeric(overlap["overlap_count"]) == 0).all():
        raise SystemExit("Checkpoint fixed-test overlap gate failed")
    if definition["approved_definition"]["oxide"] != "contains O, including O+N records":
        raise SystemExit("Unexpected oxide family definition")
    rows = [
        {"fact_id": "F1-OXIDE-DEFINITION", "value": definition["approved_definition"]["oxide"], "units": "text", "source_path": rel(root, paths["family_definition"]), "selector": "approved_definition.oxide"},
        {"fact_id": "F1-NITRIDE-DEFINITION", "value": definition["approved_definition"]["nitride"], "units": "text", "source_path": rel(root, paths["family_definition"]), "selector": "approved_definition.nitride"},
        {"fact_id": "F1-OXIDE-NTEST", "value": oxide["counts"]["test"], "units": "structures", "source_path": rel(root, paths["oxide_summary"]), "selector": "counts.test"},
        {"fact_id": "F1-NITRIDE-NTEST", "value": nitride["counts"]["test"], "units": "structures", "source_path": rel(root, paths["nitride_summary"]), "selector": "counts.test"},
        {"fact_id": "F1-OXYNITRIDE-COUNT", "value": definition["oxynitride_count"], "units": "structures", "source_path": rel(root, paths["family_definition"]), "selector": "oxynitride_count"},
        {"fact_id": "F1-OXIDE-OVERLAP", "value": int(overlap.loc[overlap.family == "oxide", "overlap_count"].iloc[0]), "units": "JIDs", "source_path": rel(root, paths["checkpoint_overlap"]), "selector": "family=oxide;column=overlap_count"},
        {"fact_id": "F1-NITRIDE-OVERLAP", "value": int(overlap.loc[overlap.family == "nitride", "overlap_count"].iloc[0]), "units": "JIDs", "source_path": rel(root, paths["checkpoint_overlap"]), "selector": "family=nitride;column=overlap_count"},
    ]
    write_csv(out / "figure1_design_data.csv", rows)

    fig, ax = plt.subplots(figsize=(7.2, 3.60))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    draw_box(ax, (0.18, 0.84), 0.64, 0.115, "JARVIS-DFT pretrained ALIGNN checkpoint\nformation-energy prediction; exact test-JID overlap = 0", face="#F4F4F4", edge="#222222", fontsize=9.2)
    draw_box(ax, (0.035, 0.57), 0.40, 0.16, "Oxide comparator\nO-bearing, including O+N\nfixed test: n = 1,484", face="#E8F2F8", edge=OXIDE_COLOR, fontsize=8.6)
    draw_box(ax, (0.565, 0.57), 0.40, 0.16, "Nitride target\nN-bearing and O-free\nfixed test: n = 242", face="#FBEDE7", edge=NITRIDE_COLOR, fontsize=8.6)
    ax.annotate("", xy=(0.235, 0.735), xytext=(0.42, 0.84), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#333333"})
    ax.annotate("", xy=(0.765, 0.735), xytext=(0.58, 0.84), arrowprops={"arrowstyle": "-|>", "lw": 1.0, "color": "#333333"})
    draw_box(ax, (0.025, 0.27), 0.29, 0.18, "Zero-shot evaluation\nno parameter updates\nsame fixed test sets", face="#FFF8E5", edge="#8A6D1D", fontsize=8.0)
    draw_box(ax, (0.355, 0.27), 0.29, 0.18, "Partial fine-tuning\nlast GCN block + output head\nN = 10, 50, 100, 200, 500, 1,000", face="#F2EDF8", edge="#6F4A8E", fontsize=7.8)
    draw_box(ax, (0.685, 0.27), 0.29, 0.18, "From-scratch baseline\nrandom initialization\nN = 50 and 500", face="#EDF6EC", edge="#39733A", fontsize=8.0)
    for x in (0.17, 0.50, 0.83):
        ax.annotate("", xy=(x, 0.46), xytext=(0.50, 0.56), arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#555555", "connectionstyle": "arc3,rad=0"})
    draw_box(
        ax,
        (0.12, 0.045),
        0.76,
        0.115,
        "Audited outputs\nzero-shot uncertainty • five-seed learning curves\npretrained-versus-scratch comparison • frozen-representation geometry",
        face="#FAFAFA",
        edge="#555555",
        fontsize=7.6,
    )
    for x in (0.17, 0.50, 0.83):
        ax.annotate("", xy=(0.50, 0.16), xytext=(x, 0.27), arrowprops={"arrowstyle": "-|>", "lw": 0.75, "color": "#777777"})
    return save_figure(fig, out, "figure1_study_design"), rows


def zero_shot_rows(paths: dict[str, Path]) -> pd.DataFrame:
    frame = pd.read_csv(paths["zero_shot_bootstrap"])
    require_columns(frame, {"scenario", "estimand", "point_estimate", "ci_low", "ci_high", "n_replicates", "oxide_n", "nitride_n", "units"}, "zero-shot bootstrap summary")
    wanted = frame[(frame.scenario == "inclusive_oxide_vs_nitride") & frame.estimand.isin(["oxide_mae", "nitride_mae"])].copy()
    if len(wanted) != 2:
        raise SystemExit("Expected exactly two inclusive zero-shot MAE rows")
    wanted["family"] = wanted["estimand"].str.replace("_mae", "", regex=False)
    for column in ("point_estimate", "ci_low", "ci_high"):
        wanted[column] = pd.to_numeric(wanted[column], errors="raise")
    if not ((wanted.ci_low < wanted.point_estimate) & (wanted.point_estimate < wanted.ci_high)).all():
        raise SystemExit("Zero-shot confidence intervals do not bracket point estimates")
    return wanted.sort_values("family").reset_index(drop=True)


def render_figure2(root: Path, out: Path, paths: dict[str, Path], runs: pd.DataFrame, aggregate: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    zero = zero_shot_rows(paths)
    zero_by_family = dict(zip(zero.family, zero.point_estimate))
    aggregate_index = aggregate.set_index(["family", "N"])
    learning_rows: list[dict[str, Any]] = []
    for row in runs.itertuples(index=False):
        agg = aggregate_index.loc[(row.family, row.N)]
        learning_rows.append(
            {
                "family": row.family,
                "N": int(row.N),
                "seed": int(row.seed),
                "test_mae_eV_per_atom": float(row.test_mae_eV_per_atom),
                "five_seed_mean_eV_per_atom": float(agg.mean_test_mae_eV_per_atom),
                "sample_sd_eV_per_atom_ddof1": float(agg.sample_std_test_mae_eV_per_atom_ddof1),
                "zero_shot_mae_eV_per_atom": float(agg.zero_shot_mae_eV_per_atom),
                "source_path": rel(root, paths["finetune_runs"]),
                "source_selector": f"family={row.family};N={row.N};seed={row.seed}",
            }
        )
    zero_plot_rows = []
    for row in zero.itertuples(index=False):
        zero_plot_rows.append(
            {
                "family": row.family,
                "mae_eV_per_atom": float(row.point_estimate),
                "ci_low_eV_per_atom": float(row.ci_low),
                "ci_high_eV_per_atom": float(row.ci_high),
                "confidence_level": 0.95,
                "bootstrap_replicates": int(row.n_replicates),
                "n_structures": int(row.oxide_n if row.family == "oxide" else row.nitride_n),
                "source_path": rel(root, paths["zero_shot_bootstrap"]),
                "source_selector": f"scenario=inclusive_oxide_vs_nitride;estimand={row.estimand}",
            }
        )
    write_csv(out / "figure2_zero_shot_plot_data.csv", zero_plot_rows)
    write_csv(out / "figure2_learning_curve_plot_data.csv", learning_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), constrained_layout=True, gridspec_kw={"width_ratios": [0.82, 1.55]})
    ax = axes[0]
    display = zero.set_index("family").loc[list(FAMILIES)]
    bars = ax.bar(
        [0, 1],
        display.point_estimate,
        yerr=[display.point_estimate - display.ci_low, display.ci_high - display.point_estimate],
        capsize=4,
        color=["white", "#A8A8A8"],
        edgecolor=[OXIDE_COLOR, NITRIDE_COLOR],
        linewidth=1.4,
        hatch=["///", "xx"],
        error_kw={"ecolor": "#222222", "elinewidth": 1.0, "capthick": 1.0},
    )
    ax.set_xticks([0, 1], ["Oxide\n(n=1,484)", "Nitride\n(n=242)"])
    ax.set_ylabel("Zero-shot test MAE (eV/atom)")
    ax.set_title("(a) Zero-shot family gap")
    ax.set_ylim(0, float(display.ci_high.max()) * 1.22)
    ax.grid(axis="y")
    for bar, value in zip(bars, display.point_estimate):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{value:.4f}", ha="center", va="bottom", fontsize=8)

    ax = axes[1]
    seed_offsets = {seed: 10 ** ((seed - 2) * 0.012) for seed in SEEDS}
    for family in FAMILIES:
        style = FAMILY_STYLE[family]
        family_runs = runs[runs.family == family]
        for seed in SEEDS:
            sub = family_runs[family_runs.seed == seed].sort_values("N")
            ax.scatter(
                sub.N.to_numpy(dtype=float) * seed_offsets[seed],
                sub.test_mae_eV_per_atom,
                marker=style["marker"],
                s=18,
                facecolors="none",
                edgecolors=style["color"],
                linewidths=0.75,
                alpha=0.72,
                zorder=2,
            )
        means = family_runs.groupby("N", as_index=False).test_mae_eV_per_atom.mean().sort_values("N")
        ax.plot(means.N, means.test_mae_eV_per_atom, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.7, markersize=4.5, label=f"{family.capitalize()} five-seed mean", zorder=3)
        ax.axhline(zero_by_family[family], color=style["color"], linestyle=":" if family == "oxide" else "-.", linewidth=1.2, label=f"{family.capitalize()} zero-shot")
    ax.set_xscale("log")
    ax.set_xticks(NS, [str(value) for value in NS])
    ax.set_xlabel("Fine-tuning size N")
    ax.set_ylabel("Test MAE (eV/atom)")
    ax.set_title("(b) Protocol 1 learning curves")
    ax.legend(frameon=False, loc="upper left", ncol=1, handlelength=2.8)
    ax.grid(True, which="major")
    return save_figure(fig, out, "figure2_zero_shot_and_learning_curves"), [out / "figure2_zero_shot_plot_data.csv", out / "figure2_learning_curve_plot_data.csv"]


def load_comparison(paths: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seeds = pd.read_csv(paths["comparison_seeds"])
    groups = pd.read_csv(paths["comparison_groups"])
    require_columns(seeds, {"family", "N", "method", "seed", "test_mae_eV_per_atom"}, "comparison seed table")
    require_columns(groups, {"family", "N", "finetune_mean_test_mae_eV_per_atom", "finetune_std_test_mae_eV_per_atom", "scratch_mean_test_mae_eV_per_atom", "scratch_std_test_mae_eV_per_atom"}, "comparison group table")
    seeds["N"] = pd.to_numeric(seeds.N, errors="raise").astype(int)
    seeds["seed"] = pd.to_numeric(seeds.seed, errors="raise").astype(int)
    seeds["test_mae_eV_per_atom"] = pd.to_numeric(seeds.test_mae_eV_per_atom, errors="raise")
    expected = {(family, n_value, method, seed) for family in FAMILIES for n_value in COMPARISON_NS for method in ("finetune", "from_scratch") for seed in SEEDS}
    actual = set(zip(seeds.family, seeds.N, seeds.method, seeds.seed))
    if actual != expected or len(seeds) != 40 or seeds.duplicated(["family", "N", "method", "seed"]).any():
        raise SystemExit("comparison seed matrix is not the expected 40 rows")
    if len(groups) != 4 or groups.duplicated(["family", "N"]).any():
        raise SystemExit("comparison group table must have four unique rows")
    indexed = groups.set_index(["family", "N"])
    for (family, n_value, method), group in seeds.groupby(["family", "N", "method"], sort=True):
        prefix = "finetune" if method == "finetune" else "scratch"
        supplied = indexed.loc[(family, n_value)]
        assert_close(group.test_mae_eV_per_atom.mean(), supplied[f"{prefix}_mean_test_mae_eV_per_atom"], f"comparison mean {family}/{n_value}/{method}")
        assert_close(group.test_mae_eV_per_atom.std(ddof=1), supplied[f"{prefix}_std_test_mae_eV_per_atom"], f"comparison SD {family}/{n_value}/{method}")
    return seeds.sort_values(["family", "N", "method", "seed"]), groups.sort_values(["family", "N"])


def render_figure3(root: Path, out: Path, paths: dict[str, Path], runs: pd.DataFrame) -> tuple[list[Path], list[Path]]:
    seeds, groups = load_comparison(paths)
    comparison_rows = []
    group_index = groups.set_index(["family", "N"])
    for row in seeds.itertuples(index=False):
        prefix = "finetune" if row.method == "finetune" else "scratch"
        summary = group_index.loc[(row.family, row.N)]
        comparison_rows.append(
            {
                "family": row.family,
                "N": int(row.N),
                "method": row.method,
                "seed": int(row.seed),
                "test_mae_eV_per_atom": float(row.test_mae_eV_per_atom),
                "five_seed_mean_eV_per_atom": float(summary[f"{prefix}_mean_test_mae_eV_per_atom"]),
                "sample_sd_eV_per_atom_ddof1": float(summary[f"{prefix}_std_test_mae_eV_per_atom"]),
                "source_path": rel(root, paths["comparison_seeds"]),
                "source_selector": f"family={row.family};N={row.N};method={row.method};seed={row.seed}",
            }
        )
    epoch_rows = []
    means = runs.groupby(["family", "N"], as_index=False).best_epoch.mean().rename(columns={"best_epoch": "five_seed_mean_best_epoch"})
    epoch_frame = runs[["family", "N", "seed", "best_epoch"]].merge(means, on=["family", "N"], validate="many_to_one")
    for row in epoch_frame.itertuples(index=False):
        epoch_rows.append(
            {
                "family": row.family,
                "N": int(row.N),
                "seed": int(row.seed),
                "best_epoch": int(row.best_epoch),
                "five_seed_mean_best_epoch": float(row.five_seed_mean_best_epoch),
                "source_path": rel(root, paths["finetune_runs"]),
                "source_selector": f"family={row.family};N={row.N};seed={row.seed};column=best_epoch",
                "interpretation_guardrail": "End-of-epoch checkpoint selection; not proof of no optimizer update or byte identity with zero-shot.",
            }
        )
    write_csv(out / "figure3_comparison_plot_data.csv", comparison_rows)
    write_csv(out / "figure3_best_epoch_plot_data.csv", epoch_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), constrained_layout=True)
    ax = axes[0]
    categories = [("oxide", 50), ("oxide", 500), ("nitride", 50), ("nitride", 500)]
    method_style = {
        "finetune": {"label": "Fine-tuned", "marker": "o", "face": "white", "offset": -0.16},
        "from_scratch": {"label": "From scratch", "marker": "^", "face": "#777777", "offset": 0.16},
    }
    for cat_idx, (family, n_value) in enumerate(categories):
        for method in ("finetune", "from_scratch"):
            style = method_style[method]
            sub = seeds[(seeds.family == family) & (seeds.N == n_value) & (seeds.method == method)].sort_values("seed")
            x_center = cat_idx + style["offset"]
            jitter = np.linspace(-0.055, 0.055, len(sub))
            edge = FAMILY_STYLE[family]["color"]
            ax.scatter(np.full(len(sub), x_center) + jitter, sub.test_mae_eV_per_atom, marker=style["marker"], s=22, facecolors=style["face"], edgecolors=edge, linewidths=0.8, alpha=0.86, zorder=3)
            mean = float(sub.test_mae_eV_per_atom.mean())
            std = float(sub.test_mae_eV_per_atom.std(ddof=1))
            ax.errorbar(x_center, mean, yerr=std, fmt="D", color="#111111", markersize=3.7, capsize=3, linewidth=1.0, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(range(4), ["Oxide\nN=50", "Oxide\nN=500", "Nitride\nN=50", "Nitride\nN=500"])
    ax.set_ylabel("Test MAE (eV/atom, log scale)")
    ax.set_title("(a) Pretrained versus random initialization")
    ax.grid(axis="y", which="both")
    handles = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#333333", markerfacecolor="white", label="Fine-tuned", markersize=5),
        Line2D([0], [0], marker="^", color="none", markeredgecolor="#333333", markerfacecolor="#777777", label="From scratch", markersize=5),
        Line2D([0], [0], marker="D", color="#111111", label="Five-seed mean ± sample SD", markersize=4),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right")

    ax = axes[1]
    seed_offsets = {seed: 10 ** ((seed - 2) * 0.012) for seed in SEEDS}
    for family in FAMILIES:
        style = FAMILY_STYLE[family]
        sub_family = runs[runs.family == family]
        for seed in SEEDS:
            sub = sub_family[sub_family.seed == seed].sort_values("N")
            ax.scatter(sub.N.to_numpy(dtype=float) * seed_offsets[seed], sub.best_epoch, marker=style["marker"], s=18, facecolors="none", edgecolors=style["color"], linewidths=0.75, alpha=0.75, zorder=2)
        summary = sub_family.groupby("N", as_index=False).best_epoch.mean().sort_values("N")
        ax.plot(summary.N, summary.best_epoch, color=style["color"], marker=style["marker"], linestyle=style["linestyle"], linewidth=1.7, markersize=4.5, label=f"{family.capitalize()} mean", zorder=3)
    ax.set_xscale("log")
    ax.set_xticks(NS, [str(value) for value in NS])
    ax.set_xlabel("Fine-tuning size N")
    ax.set_ylabel("Selected checkpoint epoch")
    ax.set_ylim(0, 52)
    ax.set_title("(b) Checkpoint-selection depth")
    ax.legend(frameon=False, loc="upper left")
    return save_figure(fig, out, "figure3_pretraining_benefit_and_engagement"), [out / "figure3_comparison_plot_data.csv", out / "figure3_best_epoch_plot_data.csv"]


def embedding_view(npz_path: Path, metadata_path: Path, source: str = "last_alignn_pool") -> tuple[np.ndarray, pd.DataFrame]:
    with np.load(npz_path) as payload:
        if source not in payload.files:
            raise SystemExit(f"{source} missing from {npz_path}")
        matrix = payload[source].astype(np.float64)
    metadata = pd.read_csv(metadata_path)
    require_columns(metadata, {"material_id", "family", "embedding_source", "embedding_index"}, f"embedding metadata {metadata_path.name}")
    metadata = metadata[metadata.embedding_source == source].copy()
    metadata["embedding_index"] = pd.to_numeric(metadata.embedding_index, errors="raise").astype(int)
    metadata = metadata.sort_values("embedding_index").reset_index(drop=True)
    if len(metadata) != matrix.shape[0] or not np.array_equal(metadata.embedding_index.to_numpy(), np.arange(len(metadata))):
        raise SystemExit(f"Embedding array/metadata alignment failed for {npz_path}")
    if metadata.material_id.duplicated().any():
        raise SystemExit(f"Duplicate material IDs in {metadata_path}")
    return matrix, metadata


def render_figure4(root: Path, out: Path, paths: dict[str, Path]) -> tuple[list[Path], list[Path], dict[str, Any]]:
    balanced_matrix, balanced_meta = embedding_view(paths["balanced_embeddings"], paths["balanced_embedding_metadata"])
    test_matrix, test_meta = embedding_view(paths["test_embeddings"], paths["test_embedding_metadata"])
    if balanced_meta.family.value_counts().to_dict() != {"oxide": 2046, "nitride": 2046}:
        raise SystemExit("Balanced-pool family counts failed")
    if test_meta.family.value_counts().to_dict() != {"oxide": 1484, "nitride": 242}:
        raise SystemExit("Fixed-test family counts failed")
    scaler = StandardScaler()
    fit = scaler.fit_transform(balanced_matrix)
    pca = PCA(n_components=2, svd_solver="full", random_state=42)
    pca.fit(fit)
    coords = pca.transform(scaler.transform(test_matrix))
    pca_rows = [
        {
            "jid": row.material_id,
            "family": row.family,
            "embedding_source": "last_alignn_pool",
            "embedding_row_index": int(row.embedding_index),
            "pca1": float(coords[idx, 0]),
            "pca2": float(coords[idx, 1]),
            "pca_fit_subset": "balanced_pool",
            "preprocessing": "StandardScaler fit on balanced_pool",
            "source_npz_path": rel(root, paths["test_embeddings"]),
            "source_npz_key": "last_alignn_pool",
        }
        for idx, row in test_meta.iterrows()
    ]
    variance_rows = [
        {
            "embedding_source": "last_alignn_pool",
            "pca_fit_subset": "balanced_pool",
            "component": component,
            "explained_variance_ratio": float(pca.explained_variance_ratio_[component - 1]),
            "explained_variance_percent": float(100 * pca.explained_variance_ratio_[component - 1]),
            "source_npz_path": rel(root, paths["balanced_embeddings"]),
            "source_npz_key": "last_alignn_pool",
        }
        for component in (1, 2)
    ]
    write_csv(out / "figure4_pca_plot_data.csv", pca_rows)
    write_csv(out / "figure4_pca_explained_variance.csv", variance_rows)

    decision = json.loads(paths["distance_decision"].read_text(encoding="utf-8"))
    if decision.get("C6_MAIN") is not True or decision.get("decision") != "INCLUDE_C6_IN_MAIN_PAPER":
        raise SystemExit("The current governed distance decision does not authorize the Figure 4b scatter")
    if not (decision.get("ci_excludes_zero") and decision.get("q_below_0_01") and decision.get("direction_unchanged_positive")):
        raise SystemExit("Distance decision sub-gates are inconsistent with C6_MAIN=true")
    distance = pd.read_csv(paths["distance_by_structure"])
    stats = pd.read_csv(paths["distance_statistics"])
    require_columns(distance, {"jid", "canonical_abs_error_eV_per_atom", "error_group", "oxide_knn5_mean_distance"}, "canonical distance-by-structure table")
    require_columns(stats, {"embedding_source", "distance_metric", "statistic", "value", "ci_low", "ci_high", "p_value", "p_value_bh_fdr_within_statistic"}, "canonical distance statistics")
    if len(distance) != 242 or distance.jid.duplicated().any():
        raise SystemExit("Canonical distance scatter table must contain 242 unique nitride JIDs")
    selected = stats[(stats.embedding_source == "last_alignn_pool") & (stats.distance_metric == "oxide_knn5_mean_distance") & (stats.statistic == "spearman_correlation")]
    if len(selected) != 1:
        raise SystemExit("Canonical 5-NN Spearman selector must match exactly one row")
    stat = selected.iloc[0]
    for decision_key, stat_key in (("spearman_rho", "value"), ("ci_low", "ci_low"), ("ci_high", "ci_high"), ("permutation_p_value", "p_value"), ("bh_fdr_q_value", "p_value_bh_fdr_within_statistic")):
        assert_close(decision[decision_key], stat[stat_key], f"distance decision {decision_key}")
    numeric = distance[["canonical_abs_error_eV_per_atom", "oxide_knn5_mean_distance"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy()).all():
        raise SystemExit("Canonical distance scatter contains non-finite values")
    distance_rows = []
    for row in distance.itertuples(index=False):
        distance_rows.append(
            {
                "jid": row.jid,
                "canonical_abs_error_eV_per_atom": float(row.canonical_abs_error_eV_per_atom),
                "oxide_knn5_mean_distance": float(row.oxide_knn5_mean_distance),
                "error_group": row.error_group,
                "source_path": rel(root, paths["distance_by_structure"]),
                "source_selector": f"jid={row.jid}",
            }
        )
    stat_rows = [
        {
            "claim_id": decision["claim_id"],
            "embedding_source": "last_alignn_pool",
            "distance_metric": "oxide_knn5_mean_distance",
            "statistic": "spearman_correlation",
            "value": float(stat.value),
            "ci_low": float(stat.ci_low),
            "ci_high": float(stat.ci_high),
            "permutation_p_value": float(stat.p_value),
            "bh_fdr_q_value": float(stat.p_value_bh_fdr_within_statistic),
            "decision": decision["decision"],
            "interpretation_guardrail": decision["interpretation_guardrail"],
            "source_path": rel(root, paths["distance_statistics"]),
            "source_selector": "embedding_source=last_alignn_pool;distance_metric=oxide_knn5_mean_distance;statistic=spearman_correlation",
        }
    ]
    write_csv(out / "figure4_distance_error_plot_data.csv", distance_rows)
    write_csv(out / "figure4_distance_error_statistics.csv", stat_rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.95), constrained_layout=True)
    ax = axes[0]
    pca_frame = pd.DataFrame(pca_rows)
    for family in FAMILIES:
        sub = pca_frame[pca_frame.family == family]
        style = FAMILY_STYLE[family]
        ax.scatter(sub.pca1, sub.pca2, marker=style["marker"], s=12 if family == "oxide" else 20, facecolors="none" if family == "nitride" else style["color"], edgecolors=style["color"], linewidths=0.55, alpha=0.34 if family == "oxide" else 0.72, label=f"{family.capitalize()} (n={len(sub):,})")
    ax.set_xlabel(f"Principal component 1 ({variance_rows[0]['explained_variance_percent']:.1f}%)")
    ax.set_ylabel(f"Principal component 2 ({variance_rows[1]['explained_variance_percent']:.1f}%)")
    ax.set_title("(a) Frozen ALIGNN embeddings (descriptive PCA)")
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    distance_frame = pd.DataFrame(distance_rows)
    group_styles = {
        "easy_bottom_20pct": ("v", "#0072B2", "Easy bottom 20%"),
        "middle_60pct": ("o", "#777777", "Middle 60%"),
        "hard_top_20pct": ("^", "#D55E00", "Hard top 20%"),
    }
    for group, (marker, color, label) in group_styles.items():
        sub = distance_frame[distance_frame.error_group == group]
        ax.scatter(sub.oxide_knn5_mean_distance, sub.canonical_abs_error_eV_per_atom, marker=marker, s=22 if group != "middle_60pct" else 14, facecolors="none" if group != "middle_60pct" else color, edgecolors=color, linewidths=0.6, alpha=0.75 if group != "middle_60pct" else 0.42, label=f"{label} (n={len(sub)})")
    x = distance_frame.oxide_knn5_mean_distance.to_numpy(dtype=float)
    y = distance_frame.canonical_abs_error_eV_per_atom.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 100)
    ax.plot(xline, slope * xline + intercept, color="#111111", linewidth=1.0, linestyle="-", label="Linear visual guide")
    ax.text(0.03, 0.97, f"Spearman ρ = {float(stat.value):.3f}\n95% CI [{float(stat.ci_low):.3f}, {float(stat.ci_high):.3f}]\nBH-FDR q = {float(stat.p_value_bh_fdr_within_statistic):.6f}", transform=ax.transAxes, ha="left", va="top", fontsize=7.4, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.90})
    ax.set_xlabel("Mean distance to 5 nearest oxide references")
    ax.set_ylabel("Nitride absolute zero-shot error (eV/atom)")
    ax.set_title("(b) Canonical distance–error association")
    ax.legend(
        frameon=True,
        loc="upper right",
        fontsize=6.6,
        facecolor="white",
        edgecolor="#BBBBBB",
        framealpha=0.95,
        borderpad=0.35,
        labelspacing=0.25,
        handletextpad=0.45,
    )
    outputs = save_figure(fig, out, "figure4_representation_geometry")
    return outputs, [out / "figure4_pca_plot_data.csv", out / "figure4_pca_explained_variance.csv", out / "figure4_distance_error_plot_data.csv", out / "figure4_distance_error_statistics.csv"], decision


def captions_text() -> str:
    return """# Paper figure captions and accessibility descriptions

## Figure 1

**Caption.** Figure 1. Study design for auditing a JARVIS-DFT pretrained ALIGNN formation-energy checkpoint across an oxide comparator and nitride target. The same fixed family test sets are used for zero-shot evaluation and partial fine-tuning; matched from-scratch baselines are available only at N = 50 and N = 500. Family definitions are intentionally asymmetric: the oxide arm is O-bearing and includes O+N structures, whereas the nitride arm is N-bearing and O-free.

**Alt text.** Workflow diagram showing one JARVIS-DFT pretrained ALIGNN checkpoint evaluated on O-bearing oxide and O-free nitride families through zero-shot, partial fine-tuning at six training sizes, matched from-scratch baselines at N = 50 and 500, and frozen-embedding analysis.

**Long description.** The shared pretrained checkpoint branches to oxide and nitride fixed-test arms, whose exact checkpoint-training JID overlap is zero. Both arms feed zero-shot evaluation and partial fine-tuning of the last GCN block plus output head at N = 10, 50, 100, 200, 500, and 1,000. From-scratch comparison is restricted to N = 50 and 500. The workflow produces audited error, data-efficiency, initialization, and representation evidence.

## Figure 2

**Caption.** Figure 2. Zero-shot chemical-family performance and Protocol 1 fine-tuning curves. (a) Bars show fixed-test mean absolute error (MAE); error bars are 95% percentile intervals from 50,000 structure-level bootstrap replicates. (b) Open markers show individual fine-tuning seeds and lines show arithmetic five-seed means under Protocol 1; horizontal lines show family-specific zero-shot MAE. Every Protocol 1 mean remains above its corresponding zero-shot line, and the nitride mean reaches its maximum at N = 200.

**Alt text.** Two-panel plot: nitride zero-shot MAE is about twice oxide with nonoverlapping bootstrap intervals, while five-seed Protocol 1 curves remain above both family zero-shot references and nitride peaks at N = 200.

**Long description.** Panel a compares oxide MAE 0.0342 and nitride MAE 0.0695 eV per atom with structure-bootstrap intervals. Panel b shows all five seed results at six training sizes for each family, their mean curves, and the two zero-shot references. The oxide curve declines toward but does not reach its zero-shot level after N = 50. The nitride curve is non-monotonic and peaks at N = 200 before declining at larger N.

## Figure 3

**Caption.** Figure 3. Pretrained-initialization benefit and selected-checkpoint depth under Protocol 1. (a) Seed points and five-seed mean ± sample standard deviation compare fine-tuned and from-scratch MAE at N = 50 and N = 500 on a logarithmic scale. (b) Seed points and mean lines show the validation-selected checkpoint epoch. An epoch-1 selection denotes the end-of-epoch-1 checkpoint and does not establish byte identity with the zero-shot checkpoint or absence of parameter updates.

**Alt text.** Two-panel plot: fine-tuned models have far lower MAE than random initialization at N = 50 and 500, while oxide selects later checkpoints from N = 50 and nitride is epoch-1 through N = 100, mixed at N = 200, and later at N at least 500.

**Long description.** Panel a shows a large gap between fine-tuned and from-scratch test MAE for both families and both supported sizes. Panel b shows five selected epochs per family and size plus their means. Oxide seeds select checkpoints later than epoch 1 from N = 50 onward. Nitride seeds all select epoch 1 through N = 100, are heterogeneous at N = 200 with epochs 49, 1, 1, 1, and 1, and all select later checkpoints at N = 500 and 1,000.

## Figure 4

**Caption.** Figure 4. Frozen-representation geometry and canonical within-nitride distance–error association. (a) Descriptive standardized principal-component projection of fixed-test `last_alignn_pool` embeddings, with the basis fitted on the balanced train–validation pool. (b) Canonical nitride absolute zero-shot error versus mean Euclidean distance to the five nearest oxide-reference embeddings in the raw 256-dimensional space. The positive Spearman association is protocol-specific and correlational; it does not identify a causal mechanism.

**Alt text.** Two-panel plot: a descriptive PCA separates many oxide and nitride frozen embeddings, and canonical nitride error tends to increase with 5-nearest-oxide embedding distance, with Spearman rho about 0.346 and a positive 95% interval.

**Long description.** Panel a projects 1,484 oxide and 242 nitride fixed-test embeddings into a common PCA basis fitted without test labels on the balanced pool. Panel b plots 242 nitride structures using canonical zero-shot errors and raw-space 5-nearest-neighbor distances to the oxide reference pool. The recomputed Spearman correlation is 0.346 with 95% bootstrap interval 0.225 to 0.463 and BH-FDR q approximately 0.000150. These results are an association under the recorded representation and distance protocol, not a causal explanation.
"""


def render_package(root: Path, out: Path, paths: dict[str, Path]) -> tuple[list[Path], dict[str, Any]]:
    out.mkdir(parents=True, exist_ok=True)
    runs = validate_finetune_runs(pd.read_csv(paths["finetune_runs"]))
    aggregate = validate_protocol_1_aggregates(runs, pd.read_csv(paths["protocol_1_aggregate"]))
    artifacts: list[Path] = []
    figure1, _ = render_figure1(root, out, paths)
    artifacts.extend(figure1)
    artifacts.append(out / "figure1_design_data.csv")
    figure2, data2 = render_figure2(root, out, paths, runs, aggregate)
    artifacts.extend(figure2 + data2)
    figure3, data3 = render_figure3(root, out, paths, runs)
    artifacts.extend(figure3 + data3)
    figure4, data4, decision = render_figure4(root, out, paths)
    artifacts.extend(figure4 + data4)
    captions = out / "figure_captions_and_alt_text.md"
    captions.write_text(captions_text(), encoding="utf-8")
    artifacts.append(captions)
    return artifacts, decision


def artifact_record(root: Path, path: Path, artifact_id: str | None = None) -> dict[str, Any]:
    return {
        "artifact_id": artifact_id or path.stem,
        "path": rel(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def deterministic_replay(root: Path, final_dir: Path, paths: dict[str, Path], final_artifacts: list[Path]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="domain_shift_figures_replay_") as temp:
        replay_dir = Path(temp) / "figures_v1"
        replay_artifacts, _ = render_package(root, replay_dir, paths)
        final_by_name = {path.name: path for path in final_artifacts}
        replay_by_name = {path.name: path for path in replay_artifacts}
        if set(final_by_name) != set(replay_by_name):
            raise SystemExit("Deterministic replay artifact-name set differs")
        mismatches = []
        for name in sorted(final_by_name):
            if sha256(final_by_name[name]) != sha256(replay_by_name[name]):
                mismatches.append(name)
        if mismatches:
            raise SystemExit(f"Deterministic replay hash mismatch: {', '.join(mismatches)}")
        return {"passed": True, "compared_artifact_count": len(final_by_name), "mismatches": []}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", default="paper/figures")
    parser.add_argument("--skip-replay", action="store_true", help="Debug only; validation_check_1 requires replay.")
    args = parser.parse_args()
    root = Path(args.repo_root).expanduser().resolve() if args.repo_root else repo_root()
    forbidden = root / "results/derived_evidence/final_paper_factory/archived_submission_materials"
    if forbidden.exists():
        raise SystemExit("Forbidden Section 08 exists; refusing to generate paper figures")
    out = (root / args.output_dir).resolve()
    reject_forbidden_input(root, out)
    paths = input_paths(root)
    artifacts, decision = render_package(root, out, paths)
    replay = {"passed": False, "reason": "skipped"} if args.skip_replay else deterministic_replay(root, out, paths, artifacts)
    if not replay["passed"]:
        raise SystemExit("Deterministic replay is required")
    producer = Path(__file__).resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    input_records = [
        {
            "input_id": label,
            "path": rel(root, path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "authority": "raw_or_validated_numerical_authority",
        }
        for label, path in sorted(paths.items())
    ]
    figure_ids = {
        "figure1_study_design.png": "FIG1_STUDY_DESIGN_PNG",
        "figure1_study_design.svg": "FIG1_STUDY_DESIGN_SVG",
        "figure2_zero_shot_and_learning_curves.png": "FIG2_ZERO_SHOT_LEARNING_PNG",
        "figure2_zero_shot_and_learning_curves.svg": "FIG2_ZERO_SHOT_LEARNING_SVG",
        "figure3_pretraining_benefit_and_engagement.png": "FIG3_PRETRAINING_ENGAGEMENT_PNG",
        "figure3_pretraining_benefit_and_engagement.svg": "FIG3_PRETRAINING_ENGAGEMENT_SVG",
        "figure4_representation_geometry.png": "FIG4_REPRESENTATION_DISTANCE_PNG",
        "figure4_representation_geometry.svg": "FIG4_REPRESENTATION_DISTANCE_SVG",
    }
    manifest = {
        "schema_version": 1,
        "status": "FIGURES_V1_GENERATED_AND_DETERMINISTIC",
        "git_commit": commit,
        "producer": {"path": rel(root, producer), "sha256": sha256(producer)},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
            "scikit_learn": sklearn.__version__,
            "png_dpi": PNG_DPI,
        },
        "source_governance": {
            "forbidden_section_08_absent": True,
            "old_figures_read": False,
            "nested_checkout_used": False,
            "dense_points_traced_by_plot_data_csv": True,
        },
        "distance_error_decision": {
            "claim_id": decision["claim_id"],
            "decision": decision["decision"],
            "C6_MAIN": decision["C6_MAIN"],
            "spearman_rho": decision["spearman_rho"],
            "ci_low": decision["ci_low"],
            "ci_high": decision["ci_high"],
            "bh_fdr_q_value": decision["bh_fdr_q_value"],
            "interpretation_guardrail": decision["interpretation_guardrail"],
        },
        "deterministic_replay": replay,
        "inputs": input_records,
        "artifacts": [artifact_record(root, path, figure_ids.get(path.name)) for path in sorted(artifacts)],
        "figure_panel_sources": {
            "Figure 1": ["oxide_summary", "nitride_summary", "family_definition", "checkpoint_overlap"],
            "Figure 2a": ["zero_shot_bootstrap"],
            "Figure 2b": ["finetune_runs", "protocol_1_aggregate"],
            "Figure 3a": ["comparison_seeds", "comparison_groups"],
            "Figure 3b": ["finetune_runs", "protocol_1_aggregate"],
            "Figure 4a": ["balanced_embeddings", "balanced_embedding_metadata", "test_embeddings", "test_embedding_metadata"],
            "Figure 4b": ["distance_decision", "distance_statistics", "distance_by_structure", "distance_validation"],
        },
    }
    write_json(out / "generation_manifest.json", manifest)
    print(json.dumps({"status": manifest["status"], "artifacts": len(artifacts), "output_dir": rel(root, out), "deterministic_replay": replay["passed"], "C6_MAIN": decision["C6_MAIN"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
