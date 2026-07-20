"""Build corrected five-seed protocol_1 comparisons and transfer-gain evidence.

This producer deliberately does not read the fine-tuning seed-0 comparison
columns written by ``summarize_from_scratch_reports.py``.  It recomputes both
methods from their five-seed run tables, verifies those recomputations against
the aggregate tables, and restricts scratch comparisons to the four groups for
which scratch runs exist: oxide/nitride at N=50 and N=500.

All runtime outputs are constrained to ``results/derived_evidence/protocol_1``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any


FAMILIES = ("oxide", "nitride")
ALL_NS = (10, 50, 100, 200, 500, 1000)
COMPARISON_NS = (50, 500)
SEEDS = (0, 1, 2, 3, 4)
MAE = "test_mae_eV_per_atom"
MEAN_MAE = "mean_test_mae_eV_per_atom"
STD_MAE = "std_test_mae_eV_per_atom"
DEFAULT_REPORT_ROOT = "results/derived_evidence/protocol_1"


def configure_plot_env() -> None:
    """Keep matplotlib caches outside the repository."""
    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temp_root / "xdg-cache"))
    os.environ.setdefault("FC_CACHEDIR", str(temp_root / "fontconfig"))
    for variable in ("MPLCONFIGDIR", "XDG_CACHE_HOME", "FC_CACHEDIR"):
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)


def load_dependencies():
    missing: list[str] = []
    try:
        import numpy as np  # pylint: disable=import-error,import-outside-toplevel
        import pandas as pd  # pylint: disable=import-error,import-outside-toplevel
    except Exception:  # noqa: BLE001
        np = None
        pd = None
        missing.extend(["numpy", "pandas"])

    configure_plot_env()
    try:
        import matplotlib  # pylint: disable=import-error,import-outside-toplevel

        matplotlib.use("Agg")
        matplotlib.rcParams["svg.hashsalt"] = "domain_shift-protocol_1-corrected-comparisons-v1"
        import matplotlib.pyplot as plt  # pylint: disable=import-error,import-outside-toplevel
    except Exception:  # noqa: BLE001
        plt = None
        missing.append("matplotlib")

    if missing:
        raise SystemExit("Missing reporting dependencies: " + ", ".join(sorted(set(missing))))
    return np, pd, plt


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repo_relative(repo: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


def resolve_input(repo: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (repo / path).resolve() if not path.is_absolute() else path.resolve()


def require_inside(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"{label} must be inside {root}: {path}") from exc


def validate_input_path(repo: Path, path: Path, label: str) -> None:
    """Reject excluded paper inputs even when a caller overrides a default."""
    require_inside(path, repo, label)
    relative = path.resolve().relative_to(repo.resolve())
    if "archived_submission_materials" in relative.parts:
        raise SystemExit(f"{label} points into forbidden Section 08: {relative}")
    if relative.parts and relative.parts[0] == repo.name:
        raise SystemExit(f"{label} points into the excluded nested checkout: {relative}")


def native(value: Any) -> Any:
    """Convert pandas/numpy scalar values into strict JSON values."""
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def records(df) -> list[dict[str, Any]]:
    return [{str(key): native(value) for key, value in row.items()} for row in df.to_dict("records")]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def require_columns(df, columns: set[str], label: str) -> None:
    missing = sorted(columns.difference(df.columns))
    if missing:
        raise SystemExit(f"{label} is missing columns: {', '.join(missing)}")


def normalize_keys(df, label: str, pd):
    result = df.copy()
    result["family"] = result["family"].astype(str).str.strip().str.lower()
    result["N"] = pd.to_numeric(result["N"], errors="raise").astype(int)
    if "seed" in result.columns:
        result["seed"] = pd.to_numeric(result["seed"], errors="raise").astype(int)
    return result


def require_finite(df, columns: list[str], label: str, np, pd) -> None:
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        bad = ~np.isfinite(numeric.to_numpy(dtype=float))
        if bad.any():
            locations = df.loc[bad, [item for item in ("family", "N", "seed") if item in df.columns]]
            raise SystemExit(f"{label} has non-finite {column} values at {locations.to_dict('records')}")


def require_exact_groups(df, expected: set[tuple[str, int]], label: str) -> None:
    actual = set(zip(df["family"], df["N"]))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise SystemExit(f"{label} group mismatch; missing={missing}, extra={extra}")


def require_seed_matrix(df, expected_groups: set[tuple[str, int]], label: str) -> None:
    duplicate = df.duplicated(["family", "N", "seed"], keep=False)
    if duplicate.any():
        raise SystemExit(
            f"{label} has duplicate run keys: "
            f"{df.loc[duplicate, ['family', 'N', 'seed']].to_dict('records')}"
        )
    require_exact_groups(df, expected_groups, label)
    expected_seeds = set(SEEDS)
    for (family, n_value), group in df.groupby(["family", "N"], sort=True):
        actual_seeds = set(int(value) for value in group["seed"])
        if actual_seeds != expected_seeds or len(group) != len(SEEDS):
            raise SystemExit(
                f"{label} {family} N={n_value} must contain exactly seeds {list(SEEDS)}; "
                f"found {sorted(actual_seeds)} across {len(group)} rows"
            )


def aggregate_runs(df, prefix: str):
    aggregate = (
        df.groupby(["family", "N"], as_index=False)
        .agg(
            runs=(MAE, "size"),
            mean=(MAE, "mean"),
            std=(MAE, lambda values: values.std(ddof=1)),
        )
        .sort_values(["family", "N"])
        .reset_index(drop=True)
    )
    return aggregate.rename(
        columns={
            "runs": f"{prefix}_runs",
            "mean": f"{prefix}_mean_test_mae_eV_per_atom",
            "std": f"{prefix}_std_test_mae_eV_per_atom",
        }
    )


def assert_close(left: float, right: float, label: str, *, atol: float = 1e-12) -> None:
    if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol):
        raise SystemExit(f"{label} mismatch: recomputed={left!r}, supplied={right!r}")


def verify_supplied_aggregate(recomputed, supplied, prefix: str, label: str) -> None:
    require_exact_groups(
        supplied,
        set(zip(recomputed["family"], recomputed["N"])),
        label,
    )
    if supplied.duplicated(["family", "N"]).any():
        raise SystemExit(f"{label} has duplicate family/N rows")
    indexed = supplied.set_index(["family", "N"])
    for row in recomputed.itertuples(index=False):
        key = (row.family, row.N)
        expected_runs = int(getattr(row, f"{prefix}_runs"))
        supplied_row = indexed.loc[key]
        if "runs" in supplied.columns and int(supplied_row["runs"]) != expected_runs:
            raise SystemExit(f"{label} {key} run-count mismatch")
        assert_close(
            getattr(row, f"{prefix}_mean_test_mae_eV_per_atom"),
            supplied_row[MEAN_MAE],
            f"{label} {key} mean (five-seed recomputation)",
        )
        assert_close(
            getattr(row, f"{prefix}_std_test_mae_eV_per_atom"),
            supplied_row[STD_MAE],
            f"{label} {key} sample std (ddof=1)",
        )


def verify_zero_shot(finetune_summary, scratch_summary, zero_df) -> dict[str, float]:
    if zero_df.duplicated(["family"]).any():
        raise SystemExit("zero-shot summary has duplicate family rows")
    zero = dict(zip(zero_df["family"], zero_df["mae_eV_per_atom"].astype(float)))
    if set(zero) != set(FAMILIES):
        raise SystemExit(f"zero-shot families must be {list(FAMILIES)}; found {sorted(zero)}")
    for family, value in zero.items():
        if not math.isfinite(value):
            raise SystemExit(f"zero-shot MAE is non-finite for {family}")
        for label, frame in (("fine-tune summary", finetune_summary), ("scratch summary", scratch_summary)):
            column = "zero_shot_mae_eV_per_atom"
            if column not in frame.columns:
                continue
            reported = frame.loc[frame["family"] == family, column].dropna().astype(float).unique()
            if len(reported) != 1:
                raise SystemExit(f"{label} has inconsistent zero-shot values for {family}: {reported.tolist()}")
            assert_close(value, reported[0], f"{label} zero-shot {family}")
    if "transfer_gain_vs_zero_shot" in finetune_summary.columns:
        for row in finetune_summary.itertuples(index=False):
            expected = zero[row.family] - float(getattr(row, MEAN_MAE))
            assert_close(
                expected,
                row.transfer_gain_vs_zero_shot,
                f"fine-tune summary zero-shot transfer gain {(row.family, row.N)}",
            )
    if "gain_vs_zero_shot" in scratch_summary.columns:
        for row in scratch_summary.itertuples(index=False):
            expected = zero[row.family] - float(getattr(row, MEAN_MAE))
            assert_close(
                expected,
                row.gain_vs_zero_shot,
                f"scratch summary zero-shot gain {(row.family, row.N)}",
            )
    return zero


def plot_family(group_df, family: str, png: Path, pdf: Path, plt) -> None:
    data = group_df[group_df["family"] == family].sort_values("N")
    x = data["N"].to_numpy(dtype=float)
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    axis.errorbar(
        x,
        data["finetune_mean_test_mae_eV_per_atom"],
        yerr=data["finetune_std_test_mae_eV_per_atom"],
        marker="s",
        linewidth=2,
        capsize=4,
        color="#1f77b4",
        label="Fine-tuned (5-seed mean ± sample SD)",
    )
    axis.errorbar(
        x,
        data["scratch_mean_test_mae_eV_per_atom"],
        yerr=data["scratch_std_test_mae_eV_per_atom"],
        marker="o",
        linewidth=2,
        capsize=4,
        color="#ff7f0e",
        label="From scratch (5-seed mean ± sample SD)",
    )
    zero = float(data["zero_shot_mae_eV_per_atom"].iloc[0])
    axis.axhline(zero, linestyle="--", color="#d62728", label=f"Zero-shot ({zero:.4f})")
    axis.set_xscale("log")
    axis.set_xticks(x, [str(int(value)) for value in x])
    axis.set_xlabel("Training size N")
    axis.set_ylabel("Test MAE (eV/atom)")
    axis.set_title(f"{family.capitalize()} protocol_1: Fine-tuning vs From Scratch")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(png, dpi=300, metadata={"Software": "DOMAIN_SHIFT corrected protocol_1 producer"})
    fig.savefig(pdf, metadata={"Creator": "DOMAIN_SHIFT corrected protocol_1 producer", "CreationDate": None})
    plt.close(fig)


def plot_transfer_benefit(comparison_df, png: Path, svg: Path, pdf: Path, plt, np) -> None:
    ordered = comparison_df.set_index(["family", "N"])
    x = np.arange(len(COMPARISON_NS), dtype=float)
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.2, 4.8))
    for index, (family, color) in enumerate((("oxide", "#4c78a8"), ("nitride", "#f58518"))):
        values = [float(ordered.loc[(family, n_value), "scratch_minus_finetune_mae_eV_per_atom"]) for n_value in COMPARISON_NS]
        positions = x + (index - 0.5) * width
        bars = axis.bar(positions, values, width, color=color, label=family.capitalize())
        axis.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=9)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, [f"N={n_value}" for n_value in COMPARISON_NS])
    axis.set_ylabel("Transfer benefit: scratch MAE − fine-tuned MAE (eV/atom)")
    axis.set_title("protocol_1 Benefit of Pretrained Initialization")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(png, dpi=300, metadata={"Software": "DOMAIN_SHIFT corrected protocol_1 producer"})
    fig.savefig(svg, metadata={"Creator": "DOMAIN_SHIFT corrected protocol_1 producer", "Date": None})
    # Matplotlib's SVG serializer emits harmless line-ending spaces. Normalize
    # them so repository whitespace validation remains deterministic.
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )
    fig.savefig(pdf, metadata={"Creator": "DOMAIN_SHIFT corrected protocol_1 producer", "CreationDate": None})
    plt.close(fig)


def input_metadata(repo: Path, path: Path) -> dict[str, Any]:
    return {
        "path": repo_relative(repo, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--finetune-runs",
        default=f"{DEFAULT_REPORT_ROOT}/Summaries/finetune/finetune_runs.csv",
    )
    parser.add_argument(
        "--finetune-summary",
        default=f"{DEFAULT_REPORT_ROOT}/Summaries/finetune/finetune_summary_by_N.csv",
    )
    parser.add_argument(
        "--scratch-runs",
        default=f"{DEFAULT_REPORT_ROOT}/Summaries/From Scratch/from_scratch_runs.csv",
    )
    parser.add_argument(
        "--scratch-summary",
        default=f"{DEFAULT_REPORT_ROOT}/Summaries/From Scratch/from_scratch_summary.csv",
    )
    parser.add_argument("--zero-shot-summary", default="results/zero_shot/zero_shot_summary.csv")
    parser.add_argument("--out-dir", default=f"{DEFAULT_REPORT_ROOT}/Comparison Plots")
    args = parser.parse_args()

    np, pd, plt = load_dependencies()
    repo = Path(args.repo_root).expanduser().resolve()
    report_root = (repo / DEFAULT_REPORT_ROOT).resolve()
    out_dir = resolve_input(repo, args.out_dir)
    require_inside(out_dir, report_root, "--out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = {
        "finetune_runs": resolve_input(repo, args.finetune_runs),
        "finetune_summary": resolve_input(repo, args.finetune_summary),
        "scratch_runs": resolve_input(repo, args.scratch_runs),
        "scratch_summary": resolve_input(repo, args.scratch_summary),
        "zero_shot_summary": resolve_input(repo, args.zero_shot_summary),
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing inputs: " + ", ".join(missing))
    for label, path in inputs.items():
        validate_input_path(repo, path, label)

    fine_runs = normalize_keys(pd.read_csv(inputs["finetune_runs"]), "fine-tune runs", pd)
    fine_summary = normalize_keys(pd.read_csv(inputs["finetune_summary"]), "fine-tune summary", pd)
    scratch_runs = normalize_keys(pd.read_csv(inputs["scratch_runs"]), "scratch runs", pd)
    scratch_summary = normalize_keys(pd.read_csv(inputs["scratch_summary"]), "scratch summary", pd)
    zero_df = pd.read_csv(inputs["zero_shot_summary"])
    zero_df["family"] = zero_df["family"].astype(str).str.strip().str.lower()

    require_columns(fine_runs, {"family", "N", "seed", MAE}, "fine-tune runs")
    require_columns(fine_summary, {"family", "N", "runs", MEAN_MAE, STD_MAE}, "fine-tune summary")
    require_columns(scratch_runs, {"family", "N", "seed", MAE}, "scratch runs")
    require_columns(scratch_summary, {"family", "N", "runs", MEAN_MAE, STD_MAE}, "scratch summary")
    require_columns(zero_df, {"family", "mae_eV_per_atom"}, "zero-shot summary")

    fine_groups = {(family, n_value) for family in FAMILIES for n_value in ALL_NS}
    scratch_groups = {(family, n_value) for family in FAMILIES for n_value in COMPARISON_NS}
    require_seed_matrix(fine_runs, fine_groups, "fine-tune runs")
    require_seed_matrix(scratch_runs, scratch_groups, "scratch runs")
    require_exact_groups(fine_summary, fine_groups, "fine-tune summary")
    require_exact_groups(scratch_summary, scratch_groups, "scratch summary")
    require_finite(fine_runs, [MAE], "fine-tune runs", np, pd)
    require_finite(scratch_runs, [MAE], "scratch runs", np, pd)
    require_finite(fine_summary, [MEAN_MAE, STD_MAE], "fine-tune summary", np, pd)
    require_finite(scratch_summary, [MEAN_MAE, STD_MAE], "scratch summary", np, pd)

    fine_agg = aggregate_runs(fine_runs, "finetune")
    scratch_agg = aggregate_runs(scratch_runs, "scratch")
    verify_supplied_aggregate(fine_agg, fine_summary, "finetune", "fine-tune summary")
    verify_supplied_aggregate(scratch_agg, scratch_summary, "scratch", "scratch summary")
    zero = verify_zero_shot(fine_summary, scratch_summary, zero_df)

    raw_seed = pd.concat(
        [
            fine_runs.merge(
                pd.DataFrame(sorted(scratch_groups), columns=["family", "N"]),
                on=["family", "N"],
                how="inner",
            )[["family", "N", "seed", MAE]].assign(method="finetune"),
            scratch_runs[["family", "N", "seed", MAE]].assign(method="from_scratch"),
        ],
        ignore_index=True,
    )
    raw_seed = raw_seed[["family", "N", "method", "seed", MAE]].sort_values(
        ["family", "N", "method", "seed"]
    ).reset_index(drop=True)
    if len(raw_seed) != 40:
        raise SystemExit(f"comparison seed table must have 40 rows; found {len(raw_seed)}")

    transfer = fine_agg.copy()
    transfer["zero_shot_mae_eV_per_atom"] = transfer["family"].map(zero)
    transfer["zero_shot_minus_finetune_mae_eV_per_atom"] = (
        transfer["zero_shot_mae_eV_per_atom"] - transfer["finetune_mean_test_mae_eV_per_atom"]
    )
    transfer = transfer.merge(scratch_agg, on=["family", "N"], how="left")
    transfer["scratch_supported"] = transfer["scratch_runs"].notna()
    transfer["scratch_minus_finetune_mae_eV_per_atom"] = (
        transfer["scratch_mean_test_mae_eV_per_atom"] - transfer["finetune_mean_test_mae_eV_per_atom"]
    )
    transfer = transfer.sort_values(["family", "N"]).reset_index(drop=True)
    if len(transfer) != 12 or int(transfer["scratch_supported"].sum()) != 4:
        raise SystemExit("transfer table must contain 12 fine-tune groups and exactly 4 scratch-supported groups")
    unsupported = ~transfer["scratch_supported"]
    scratch_columns = [
        "scratch_runs",
        "scratch_mean_test_mae_eV_per_atom",
        "scratch_std_test_mae_eV_per_atom",
        "scratch_minus_finetune_mae_eV_per_atom",
    ]
    if transfer.loc[unsupported, scratch_columns].notna().any().any():
        raise SystemExit("scratch values leaked into unsupported N groups")
    comparison = transfer.loc[transfer["scratch_supported"]].copy().reset_index(drop=True)

    # Version the from-scratch summary instead of overwriting the historical
    # file whose fine-tuning columns are seed-0-derived and stale. This v2 table
    # retains validated scratch aggregates while replacing dependent fields
    # with explicit five-seed fine-tuning values and formulas.
    scratch_summary_corrected = comparison[
        [
            "family",
            "N",
            "scratch_runs",
            "scratch_mean_test_mae_eV_per_atom",
            "scratch_std_test_mae_eV_per_atom",
            "finetune_runs",
            "finetune_mean_test_mae_eV_per_atom",
            "finetune_std_test_mae_eV_per_atom",
            "zero_shot_mae_eV_per_atom",
            "zero_shot_minus_finetune_mae_eV_per_atom",
            "scratch_minus_finetune_mae_eV_per_atom",
        ]
    ].copy()

    paths = {
        "seed_csv": out_dir / "protocol_1_corrected_comparison_seed_values.csv",
        "seed_json": out_dir / "protocol_1_corrected_comparison_seed_values.json",
        "comparison_csv": out_dir / "protocol_1_corrected_comparison_by_group.csv",
        "comparison_json": out_dir / "protocol_1_corrected_comparison_by_group.json",
        "transfer_csv": out_dir / "protocol_1_transfer_gain_table.csv",
        "transfer_json": out_dir / "protocol_1_transfer_gain_table.json",
        "scratch_corrected_csv": report_root
        / "Summaries/From Scratch/from_scratch_summary_corrected_five_seed.csv",
        "scratch_corrected_json": report_root
        / "Summaries/From Scratch/from_scratch_summary_corrected_five_seed.json",
        "oxide_png": out_dir / "Oxide Comparison Plot - protocol_1.png",
        "oxide_pdf": out_dir / "Oxide Comparison Plot - protocol_1.pdf",
        "nitride_png": out_dir / "Nitride Comparison Plot - protocol_1.png",
        "nitride_pdf": out_dir / "Nitride Comparison Plot - protocol_1.pdf",
        "benefit_png": out_dir / "FIG_TRANSFER_BENEFIT.png",
        "benefit_svg": out_dir / "FIG_TRANSFER_BENEFIT.svg",
        "benefit_pdf": out_dir / "FIG_TRANSFER_BENEFIT.pdf",
        "manifest": out_dir / "protocol_1_corrected_comparison_manifest.json",
    }

    raw_seed.to_csv(paths["seed_csv"], index=False)
    write_json(paths["seed_json"], {"schema_version": 1, "rows": records(raw_seed)})
    comparison.to_csv(paths["comparison_csv"], index=False)
    write_json(paths["comparison_json"], {"schema_version": 1, "rows": records(comparison)})
    transfer.to_csv(paths["transfer_csv"], index=False)
    write_json(paths["transfer_json"], {"schema_version": 1, "rows": records(transfer)})
    scratch_summary_corrected.to_csv(paths["scratch_corrected_csv"], index=False)
    write_json(
        paths["scratch_corrected_json"],
        {
            "schema_version": 1,
            "supersedes_for_finetune_dependent_fields": repo_relative(
                repo, inputs["scratch_summary"]
            ),
            "historical_seed0_fields_excluded": [
                "finetune_seed0_test_mae_eV_per_atom",
                "gain_vs_finetune_seed0",
            ],
            "rows": records(scratch_summary_corrected),
        },
    )

    plot_family(comparison, "oxide", paths["oxide_png"], paths["oxide_pdf"], plt)
    plot_family(comparison, "nitride", paths["nitride_png"], paths["nitride_pdf"], plt)
    plot_transfer_benefit(
        comparison,
        paths["benefit_png"],
        paths["benefit_svg"],
        paths["benefit_pdf"],
        plt,
        np,
    )

    output_metadata = {
        name: {
            "path": repo_relative(repo, path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for name, path in paths.items()
        if name != "manifest"
    }
    manifest = {
        "schema_version": 1,
        "producer": repo_relative(repo, Path(__file__).resolve()),
        "methodology": {
            "fine_tune_groups": "oxide/nitride at N=10,50,100,200,500,1000",
            "scratch_supported_groups": "oxide/nitride at N=50,500 only",
            "required_seeds_per_method_group": list(SEEDS),
            "aggregate_mean": "arithmetic mean across five run-level test MAEs",
            "aggregate_std": "sample standard deviation across five run-level test MAEs (ddof=1)",
            "zero_shot_transfer_gain": "zero_shot_mae - fine_tune_mean_mae",
            "pretraining_transfer_benefit": "scratch_mean_mae - fine_tune_mean_mae",
            "unsupported_scratch_policy": "scratch fields are blank/null outside N=50 and N=500",
            "ignored_stale_fields": [
                "from_scratch_summary.csv:finetune_seed0_test_mae_eV_per_atom",
                "from_scratch_summary.csv:gain_vs_finetune_seed0",
            ],
        },
        "validation": {
            "fine_tune_run_rows": len(fine_runs),
            "fine_tune_groups": len(fine_agg),
            "scratch_run_rows": len(scratch_runs),
            "scratch_groups": len(scratch_agg),
            "comparison_seed_rows": len(raw_seed),
            "transfer_table_rows": len(transfer),
            "scratch_supported_transfer_rows": int(transfer["scratch_supported"].sum()),
            "five_unique_seeds_per_method_group": True,
            "all_required_values_finite": True,
            "supplied_aggregates_match_run_recomputation": True,
            "sample_std_ddof": 1,
            "zero_shot_consistent_across_sources": True,
        },
        "inputs": {name: input_metadata(repo, path) for name, path in inputs.items()},
        "outputs": output_metadata,
    }
    write_json(paths["manifest"], manifest)
    print(json.dumps({"status": "PASS", "manifest": repo_relative(repo, paths["manifest"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
