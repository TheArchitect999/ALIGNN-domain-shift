from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

DEFAULT_FAMILIES = ["oxide", "nitride"]
DEFAULT_NS = [50, 500]
DEFAULT_SEEDS = [0]
DEFAULT_ZERO_SHOT_SUMMARY = "results/zero_shot/zero_shot_summary.csv"
REPORTING_DEPENDENCY_HINT = (
    "from_scratch summarization requires pandas and matplotlib. "
    "Install the pinned environment from `environment/requirements-frozen.txt` "
    "or bootstrap it with `environment/setup_training.sh`."
)


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
        missing.append("pandas")
        pd = None

    configure_plot_env()
    try:
        import matplotlib.pyplot as plt  # pylint: disable=import-error
    except Exception:  # noqa: BLE001
        missing.append("matplotlib")
        plt = None

    if missing:
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing reporting dependencies: {missing_text}. {REPORTING_DEPENDENCY_HINT}")
    return pd, plt


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_relative(repo: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


def collect_from_scratch_rows(
    repo: Path,
    results_root: str,
    families: list[str],
    ns: list[int],
    seeds: list[int],
    run_subdir: str,
) -> list[dict]:
    rows: list[dict] = []
    for family in families:
        for n_value in ns:
            for seed in seeds:
                summary_path = repo / results_root / family / f"N{n_value}_seed{seed}" / run_subdir / "summary.json"
                if not summary_path.exists():
                    continue
                summary = load_json(summary_path)
                rows.append(
                    {
                        "family": family,
                        "N": n_value,
                        "seed": seed,
                        "n_train": summary["n_train"],
                        "n_val": summary["n_val"],
                        "n_test": summary["n_test"],
                        "epochs": summary["epochs"],
                        "batch_size": summary["batch_size"],
                        "learning_rate": summary["learning_rate"],
                        "test_mae_eV_per_atom": summary["test_mae_eV_per_atom"],
                        "train_mae_eV_per_atom": summary["train_mae_eV_per_atom"],
                        "val_best_l1": summary["val_best_l1"],
                        "summary_path": repo_relative(repo, summary_path),
                    }
                )
    return rows


def collect_finetune_seed0(
    repo: Path,
    finetune_results_root: str,
    families: list[str],
    ns: list[int],
    run_subdir: str,
    pd,
) -> "pd.DataFrame":
    rows: list[dict] = []
    for family in families:
        for n_value in ns:
            summary_path = repo / finetune_results_root / family / f"N{n_value}_seed0" / run_subdir / "summary.json"
            if not summary_path.exists():
                continue
            summary = load_json(summary_path)
            rows.append(
                {
                    "family": family,
                    "N": n_value,
                    "finetune_seed0_test_mae_eV_per_atom": summary["test_mae_eV_per_atom"],
                }
            )
    return pd.DataFrame(rows)


def collect_zero_shot(repo: Path, zero_shot_summary: str, families: list[str], pd) -> "pd.DataFrame":
    rows: list[dict] = []
    summary_path = repo / zero_shot_summary
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["family"] not in families:
                continue
            rows.append(
                {
                    "family": row["family"],
                    "zero_shot_mae_eV_per_atom": float(row["mae_eV_per_atom"]),
                    "zero_shot_summary_path": repo_relative(repo, summary_path),
                }
            )
    return pd.DataFrame(rows)


def plot_family_comparison(
    df: "pd.DataFrame",
    family: str,
    out_png: Path,
    out_pdf: Path,
    plt,
    title_label: str,
    plot_title_template: str | None,
) -> None:
    family_df = df[df["family"] == family].sort_values("N")
    if family_df.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        family_df["N"],
        family_df["mean_test_mae_eV_per_atom"],
        marker="o",
        linewidth=2,
        label="From scratch mean",
    )
    if (family_df["runs"] > 1).any():
        lower = family_df["mean_test_mae_eV_per_atom"] - family_df["std_test_mae_eV_per_atom"].fillna(0.0)
        upper = family_df["mean_test_mae_eV_per_atom"] + family_df["std_test_mae_eV_per_atom"].fillna(0.0)
        plt.fill_between(family_df["N"], lower, upper, alpha=0.2, label="From scratch ±1 std")
    if "finetune_seed0_test_mae_eV_per_atom" in family_df.columns:
        plt.plot(
            family_df["N"],
            family_df["finetune_seed0_test_mae_eV_per_atom"],
            marker="s",
            linewidth=2,
            label="Fine-tune (seed0)",
        )
    if family_df["zero_shot_mae_eV_per_atom"].notna().any():
        zero = float(family_df["zero_shot_mae_eV_per_atom"].dropna().iloc[0])
        plt.axhline(zero, linestyle="--", color="tab:red", label=f"Zero-shot ({zero:.4f})")

    plt.xscale("log")
    plt.xticks(family_df["N"], [str(v) for v in family_df["N"]])
    plt.xlabel("Training size N")
    plt.ylabel("Test MAE (eV/atom)")
    title = plot_title_template.format(family=family, Family=family.capitalize()) if plot_title_template else f"{family.capitalize()} {title_label}"
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default="results/protocol_2")
    parser.add_argument("--finetune-results-root", default="results/protocol_2")
    parser.add_argument("--zero-shot-root", default="results/zero_shot")
    parser.add_argument("--zero-shot-summary", default=DEFAULT_ZERO_SHOT_SUMMARY)
    parser.add_argument("--families", nargs="+", default=DEFAULT_FAMILIES)
    parser.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--run-subdir", default="train_alignn_from_scratch")
    parser.add_argument("--finetune-run-subdir", default="finetune_last2")
    parser.add_argument("--out-dir", default="results/summaries/protocol_2/From Scratch")
    parser.add_argument("--summary-dir")
    parser.add_argument("--plot-dir")
    parser.add_argument("--title-label", default="from_scratch From-Scratch Comparison")
    parser.add_argument("--plot-name-template")
    parser.add_argument("--plot-title-template")
    args = parser.parse_args()
    pd, plt = load_reporting_dependencies()

    repo = Path(args.repo_root).resolve()
    summary_dir = (repo / (args.summary_dir or args.out_dir)).resolve()
    plot_dir = (repo / (args.plot_dir or args.out_dir)).resolve()
    summary_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_from_scratch_rows(
        repo,
        args.results_root,
        args.families,
        args.Ns,
        args.seeds,
        args.run_subdir,
    )
    if not rows:
        raise SystemExit("No from-scratch summaries found.")

    runs_df = pd.DataFrame(rows).sort_values(["family", "N", "seed"]).reset_index(drop=True)
    finetune_df = collect_finetune_seed0(
        repo,
        args.finetune_results_root,
        args.families,
        args.Ns,
        args.finetune_run_subdir,
        pd,
    )
    zero_df = collect_zero_shot(repo, args.zero_shot_summary, args.families, pd)
    summary_df = (
        runs_df.groupby(["family", "N"], as_index=False)
        .agg(
            runs=("test_mae_eV_per_atom", "size"),
            n_train=("n_train", "first"),
            n_val=("n_val", "first"),
            n_test=("n_test", "first"),
            epochs=("epochs", "first"),
            batch_size=("batch_size", "first"),
            learning_rate=("learning_rate", "first"),
            mean_test_mae_eV_per_atom=("test_mae_eV_per_atom", "mean"),
            std_test_mae_eV_per_atom=("test_mae_eV_per_atom", "std"),
            mean_train_mae_eV_per_atom=("train_mae_eV_per_atom", "mean"),
            mean_val_best_l1=("val_best_l1", "mean"),
        )
        .sort_values(["family", "N"])
        .reset_index(drop=True)
    )
    merged = summary_df.merge(finetune_df, on=["family", "N"], how="left").merge(zero_df, on="family", how="left")
    merged["gain_vs_zero_shot"] = merged["zero_shot_mae_eV_per_atom"] - merged["mean_test_mae_eV_per_atom"]
    merged["gain_vs_finetune_seed0"] = (
        merged["finetune_seed0_test_mae_eV_per_atom"] - merged["mean_test_mae_eV_per_atom"]
    )

    runs_csv = summary_dir / "from_scratch_runs.csv"
    summary_csv = summary_dir / "from_scratch_summary.csv"
    manifest_json = summary_dir / "from_scratch_manifest.json"
    suite_json = summary_dir / "run_suite_summary.json"

    runs_df.to_csv(runs_csv, index=False)
    merged.to_csv(summary_csv, index=False)

    for family in args.families:
        stem = (
            args.plot_name_template.format(family=family, Family=family.capitalize())
            if args.plot_name_template
            else f"{family}_from_scratch_comparison"
        )
        plot_family_comparison(
            merged,
            family,
            plot_dir / f"{stem}.png",
            plot_dir / f"{stem}.pdf",
            plt,
            args.title_label,
            args.plot_title_template,
        )

    plot_paths = {
        family: {
            "png": repo_relative(
                repo,
                plot_dir
                / (
                    (
                        args.plot_name_template.format(family=family, Family=family.capitalize())
                        if args.plot_name_template
                        else f"{family}_from_scratch_comparison"
                    )
                    + ".png"
                ),
            ),
            "pdf": repo_relative(
                repo,
                plot_dir
                / (
                    (
                        args.plot_name_template.format(family=family, Family=family.capitalize())
                        if args.plot_name_template
                        else f"{family}_from_scratch_comparison"
                    )
                    + ".pdf"
                ),
            ),
        }
        for family in args.families
    }

    manifest = {
        "results_root": args.results_root,
        "finetune_results_root": args.finetune_results_root,
        "zero_shot_summary": args.zero_shot_summary,
        "run_subdir": args.run_subdir,
        "finetune_run_subdir": args.finetune_run_subdir,
        "runs_csv": repo_relative(repo, runs_csv),
        "summary_csv": repo_relative(repo, summary_csv),
        "seeds": args.seeds,
        "plots": plot_paths,
    }
    manifest_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    suite_summary = {
        "results_root": args.results_root,
        "run_subdir": args.run_subdir,
        "expected_runs": len(args.families) * len(args.Ns) * len(args.seeds),
        "actual_runs": len(runs_df),
        "complete": len(runs_df) == len(args.families) * len(args.Ns) * len(args.seeds),
        "report_dir": repo_relative(repo, summary_dir),
        "summary_csv": repo_relative(repo, summary_csv),
        "plot_dir": repo_relative(repo, plot_dir),
    }
    suite_json.write_text(json.dumps(suite_summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
