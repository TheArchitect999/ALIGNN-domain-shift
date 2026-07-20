from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("FC_CACHEDIR", "/tmp/fontconfig")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["FC_CACHEDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_NS = [10, 50, 100, 200, 500, 1000]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_TAG = "finetune_alignn_hyperparameters_new_script"
DEFAULT_ZERO_SHOT_SUMMARY = "results/zero_shot/zero_shot_summary.csv"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_finetune_rows(repo: Path, families: list[str], ns: list[int], seeds: list[int], run_subdir: str) -> list[dict]:
    rows: list[dict] = []
    for family in families:
        for n_value in ns:
            for seed in seeds:
                summary_path = repo / "results/protocol_1" / family / f"N{n_value}_seed{seed}" / run_subdir / "summary.json"
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
                        "device": summary.get("device", "unknown"),
                        "elapsed_seconds": summary.get("elapsed_seconds"),
                        "summary_path": str(summary_path.resolve()),
                    }
                )
    return rows


def collect_zero_shot_rows(repo: Path, families: list[str]) -> list[dict]:
    rows: list[dict] = []
    summary_path = repo / DEFAULT_ZERO_SHOT_SUMMARY
    with summary_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["family"] not in families:
                continue
            rows.append(
                {
                    "family": row["family"],
                    "zero_shot_mae_eV_per_atom": float(row["mae_eV_per_atom"]),
                    "summary_path": DEFAULT_ZERO_SHOT_SUMMARY,
                }
            )
    return rows


def write_latex_table(path: Path, frame: pd.DataFrame) -> None:
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Family & N & Runs & Train & Val & Mean test MAE & Std \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"{row.family.capitalize()} & {row.N} & {row.runs} & {row.n_train} & {row.n_val} & "
            f"{row.mean_test_mae_eV_per_atom:.6f} & {row.std_test_mae_eV_per_atom:.6f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_family_curve(
    family: str,
    summary_df: pd.DataFrame,
    zero_shot_mae: float | None,
    out_png: Path,
    out_pdf: Path,
) -> None:
    family_df = summary_df[summary_df["family"] == family].sort_values("N")
    if family_df.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        family_df["N"],
        family_df["mean_test_mae_eV_per_atom"],
        marker="o",
        linewidth=2,
        label="Fine-tune mean",
        color="#1f77b4",
    )
    if (family_df["runs"] > 1).any():
        lower = family_df["mean_test_mae_eV_per_atom"] - family_df["std_test_mae_eV_per_atom"].fillna(0.0)
        upper = family_df["mean_test_mae_eV_per_atom"] + family_df["std_test_mae_eV_per_atom"].fillna(0.0)
        plt.fill_between(family_df["N"], lower, upper, alpha=0.2, color="#1f77b4", label="±1 std")
    if zero_shot_mae is not None:
        plt.axhline(zero_shot_mae, linestyle="--", color="#d62728", label=f"Zero-shot ({zero_shot_mae:.4f})")

    plt.xscale("log")
    plt.xticks(family_df["N"], [str(n) for n in family_df["N"]])
    plt.xlabel("Fine-tuning size N")
    plt.ylabel("Test MAE (eV/atom)")
    plt.title(f"{family.capitalize()} fine-tuning learning curve ({DEFAULT_TAG})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--experiment-tag", default=DEFAULT_TAG)
    parser.add_argument("--run-subdir", default=None)
    parser.add_argument("--families", nargs="+", default=["oxide", "nitride"])
    parser.add_argument("--Ns", nargs="+", type=int, default=DEFAULT_NS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo = Path(args.repo_root).resolve()
    run_subdir = args.run_subdir or f"finetune_last2_{args.experiment_tag}"
    out_dir = (repo / (args.out_dir or f"results/derived_evidence/{args.experiment_tag}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    finetune_rows = collect_finetune_rows(repo, args.families, args.Ns, args.seeds, run_subdir)
    zero_shot_rows = collect_zero_shot_rows(repo, args.families)

    runs_df = pd.DataFrame(finetune_rows)
    zero_df = pd.DataFrame(zero_shot_rows)

    if runs_df.empty:
        raise SystemExit("No finetune fine-tuning summaries were found.")

    runs_df = runs_df.sort_values(["family", "N", "seed"]).reset_index(drop=True)
    summary_df = (
        runs_df.groupby(["family", "N"], as_index=False)
        .agg(
            runs=("test_mae_eV_per_atom", "size"),
            n_train=("n_train", "first"),
            n_val=("n_val", "first"),
            n_test=("n_test", "first"),
            mean_test_mae_eV_per_atom=("test_mae_eV_per_atom", "mean"),
            std_test_mae_eV_per_atom=("test_mae_eV_per_atom", "std"),
            mean_train_mae_eV_per_atom=("train_mae_eV_per_atom", "mean"),
            mean_val_best_l1=("val_best_l1", "mean"),
        )
        .sort_values(["family", "N"])
        .reset_index(drop=True)
    )
    wide_df = runs_df.pivot_table(index=["family", "N"], columns="seed", values="test_mae_eV_per_atom")
    wide_df = wide_df.reset_index()
    wide_df.columns = [
        "family" if col == "family" else "N" if col == "N" else f"seed_{col}"
        for col in wide_df.columns
    ]
    summary_with_zero = summary_df.merge(zero_df, on="family", how="left")
    summary_with_zero["transfer_gain_vs_zero_shot"] = (
        summary_with_zero["zero_shot_mae_eV_per_atom"] - summary_with_zero["mean_test_mae_eV_per_atom"]
    )

    runs_csv = out_dir / "finetune_runs.csv"
    summary_csv = out_dir / "finetune_summary_by_N.csv"
    wide_csv = out_dir / "finetune_summary_wide.csv"
    latex_table = out_dir / "finetune_summary_table.tex"
    manifest_json = out_dir / "finetune_summary_manifest.json"

    runs_df.to_csv(runs_csv, index=False)
    summary_with_zero.to_csv(summary_csv, index=False)
    wide_df.to_csv(wide_csv, index=False)
    write_latex_table(latex_table, summary_df)

    zero_lookup = {row["family"]: row["zero_shot_mae_eV_per_atom"] for row in zero_shot_rows}
    for family in args.families:
        plot_family_curve(
            family,
            summary_df,
            zero_lookup.get(family),
            out_dir / f"{family}_learning_curve.png",
            out_dir / f"{family}_learning_curve.pdf",
        )

    manifest = {
        "experiment_tag": args.experiment_tag,
        "run_subdir": run_subdir,
        "runs_csv": str(runs_csv),
        "summary_csv": str(summary_csv),
        "wide_csv": str(wide_csv),
        "canonical_zero_shot_summary": DEFAULT_ZERO_SHOT_SUMMARY,
        "latex_table": str(latex_table),
        "plots": {
            family: {
                "png": str((out_dir / f"{family}_learning_curve.png").resolve()),
                "pdf": str((out_dir / f"{family}_learning_curve.pdf").resolve()),
            }
            for family in args.families
        },
    }
    manifest_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
