from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path


def configure_plot_env() -> None:
    temp_root = Path(tempfile.gettempdir())
    os.environ.setdefault("MPLCONFIGDIR", str(temp_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(temp_root / "xdg-cache"))
    os.environ.setdefault("FC_CACHEDIR", str(temp_root / "fontconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["FC_CACHEDIR"]).mkdir(parents=True, exist_ok=True)


def load_reporting_dependencies():
    missing: list[str] = []
    try:
        import pandas as pd  # pylint: disable=import-error
    except Exception:  # noqa: BLE001
        pd = None
        missing.append("pandas")

    configure_plot_env()
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:  # noqa: BLE001
        plt = None
        missing.append("matplotlib")

    if missing:
        raise SystemExit(f"Missing reporting dependencies: {', '.join(missing)}.")
    return pd, plt


def filename_stem(family: str, set_number: int, n_value: int) -> str:
    return f"{family.capitalize()} Parity Plot - protocol {set_number}, N={n_value}"


def repo_relative(repo: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


def compute_metrics(df) -> tuple[float, float, float]:
    residual = df["prediction_mean"] - df["target"]
    mae = float(residual.abs().mean())
    rmse = float((residual.pow(2).mean()) ** 0.5)

    target_mean = float(df["target"].mean())
    ss_res = float(((df["target"] - df["prediction_mean"]) ** 2).sum())
    ss_tot = float(((df["target"] - target_mean) ** 2).sum())
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - (ss_res / ss_tot)
    return mae, rmse, r2


def load_seed_prediction_tables(repo: Path, results_root: str, family: str, n_value: int, run_subdir: str, seeds: list[int], pd):
    tables = []
    for seed in seeds:
        csv_path = (
            repo
            / results_root
            / family
            / f"N{n_value}_seed{seed}"
            / run_subdir
            / "prediction_results_test_set.csv"
        )
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path)
        frame = frame.rename(columns={"prediction": f"prediction_seed_{seed}"})
        tables.append((seed, csv_path, frame))
    return tables


def merge_seed_predictions(tables, pd):
    merged = None
    source_paths: list[str] = []
    seeds_used: list[int] = []
    for seed, path, frame in tables:
        source_paths.append(str(path))
        seeds_used.append(seed)
        keep = ["id", "target", f"prediction_seed_{seed}"]
        frame = frame[keep]
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on=["id", "target"], how="inner", validate="one_to_one")
    if merged is None:
        return None, [], []

    prediction_cols = [col for col in merged.columns if col.startswith("prediction_seed_")]
    merged["prediction_mean"] = merged[prediction_cols].mean(axis=1)
    merged["prediction_std"] = merged[prediction_cols].std(axis=1)
    return merged, seeds_used, source_paths


def plot_parity(df, family: str, set_number: int, n_value: int, seeds_used: list[int], out_png: Path, out_pdf: Path, plt) -> tuple[float, float, float]:
    mae, rmse, r2 = compute_metrics(df)

    x_min = float(min(df["target"].min(), df["prediction_mean"].min()))
    x_max = float(max(df["target"].max(), df["prediction_mean"].max()))
    pad = 0.05 * (x_max - x_min if x_max != x_min else 1.0)
    line_min = x_min - pad
    line_max = x_max + pad

    plt.figure(figsize=(5.6, 5.6))
    plt.scatter(
        df["target"],
        df["prediction_mean"],
        s=18,
        alpha=0.58,
        color="#1f77b4",
        edgecolors="none",
        label=f"Mean prediction across {len(seeds_used)} seeds",
    )
    plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="#d62728", linewidth=1.5, label="Ideal parity")
    plt.xlim(line_min, line_max)
    plt.ylim(line_min, line_max)
    plt.xlabel("True Formation Energy (eV/atom)")
    plt.ylabel("Predicted Formation Energy (eV/atom)")
    plt.title(filename_stem(family, set_number, n_value))
    plt.grid(alpha=0.25)
    stats_text = "\n".join(
        [
            f"Seeds: {', '.join(str(seed) for seed in seeds_used)}",
            f"MAE: {mae:.4f}",
            f"RMSE: {rmse:.4f}",
            f"R²: {r2:.4f}" if not math.isnan(r2) else "R²: nan",
        ]
    )
    plt.text(
        0.03,
        0.97,
        stats_text,
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    plt.legend(loc="lower right")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()
    return mae, rmse, r2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--run-subdir", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--out-dir")
    parser.add_argument("--set-number", type=int, required=True)
    parser.add_argument("--families", nargs="+", default=["oxide", "nitride"])
    parser.add_argument("--Ns", nargs="+", type=int, default=[10, 50, 100, 200, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    pd, plt = load_reporting_dependencies()
    repo = Path(args.repo_root).resolve()
    report_dir = (repo / args.report_dir).resolve()
    out_dir = (repo / args.out_dir).resolve() if args.out_dir else report_dir / "parity_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict] = []
    for family in args.families:
        for n_value in args.Ns:
            tables = load_seed_prediction_tables(repo, args.results_root, family, n_value, args.run_subdir, args.seeds, pd)
            merged, seeds_used, source_paths = merge_seed_predictions(tables, pd)
            if merged is None or not seeds_used:
                continue
            stem = filename_stem(family, args.set_number, n_value)
            out_png = out_dir / f"{stem}.png"
            out_pdf = out_dir / f"{stem}.pdf"
            mae, rmse, r2 = plot_parity(merged, family, args.set_number, n_value, seeds_used, out_png, out_pdf, plt)
            manifest_rows.append(
                {
                    "family": family,
                    "N": n_value,
                    "set_number": args.set_number,
                    "seeds_used": ",".join(str(seed) for seed in seeds_used),
                    "n_points": len(merged),
                    "mae_eV_per_atom": mae,
                    "rmse_eV_per_atom": rmse,
                    "r2": r2,
                    "png": repo_relative(repo, out_png),
                    "pdf": repo_relative(repo, out_pdf),
                    "source_prediction_csvs": ";".join(repo_relative(repo, Path(p)) for p in source_paths),
                }
            )

    if not manifest_rows:
        raise SystemExit("No parity plots were generated.")

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["family", "N"]).reset_index(drop=True)
    manifest_csv = out_dir / "parity_plot_manifest.csv"
    manifest_json = out_dir / "parity_plot_manifest.json"
    readme_path = out_dir / "readme.md"

    manifest_df.to_csv(manifest_csv, index=False)
    manifest_json.write_text(json.dumps(manifest_rows, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(
        "\n".join(
            [
                "# Parity Plots",
                "",
                "These parity plots compare ground-truth formation energy on the x-axis to predicted formation energy on the y-axis.",
                "",
                "Each plot is built from `prediction_results_test_set.csv` outputs produced from the best validation checkpoint for each seed.",
                "",
                "For each `{family, N, protocol}` combination, the plotted prediction is the mean test prediction across seeds `0..4`.",
                "",
                "- protocol_1: canonical fine-tuning (`epochs=50`, `batch_size=16`, `learning_rate=0.0001`)",
                "- protocol_2: ALIGNN-recommended fine-tuning (`epochs=300`, `batch_size=64`, `learning_rate=0.001`)",
                "- protocol_3: `epochs=100`, `batch_size=32`, `learning_rate=0.00005`",
                "",
                "See `parity_plot_manifest.csv` for file paths and summary metrics.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
