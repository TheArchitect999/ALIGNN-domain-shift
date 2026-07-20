from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("FC_CACHEDIR", "/tmp/fontconfig")

import matplotlib.pyplot as plt
import numpy as np


def load_history(path: Path) -> np.ndarray:
    raw = json.loads(path.read_text())
    values: list[float] = []
    for row in raw:
        if isinstance(row, list):
            if not row:
                raise ValueError(f"Empty history row in {path}")
            values.append(float(row[0]))
        else:
            values.append(float(row))
    return np.asarray(values, dtype=float)


def relative_reduction(losses: np.ndarray) -> np.ndarray:
    if len(losses) == 0:
        return losses
    baseline = float(losses[0])
    if abs(baseline) < 1e-12:
        return np.zeros_like(losses)
    return 100.0 * (1.0 - losses / baseline)


def discover_runs(repo_root: Path, families: list[str], ns: list[int], seeds: list[int], run_subdir: str) -> list[dict]:
    runs: list[dict] = []
    for family in families:
        for n in ns:
            for seed in seeds:
                run_dir = repo_root / "results/protocol_1" / family / f"N{n}_seed{seed}" / run_subdir
                summary_path = run_dir / "summary.json"
                train_path = run_dir / "history_train.json"
                val_path = run_dir / "history_val.json"
                if not (summary_path.exists() and train_path.exists() and val_path.exists()):
                    continue
                summary = json.loads(summary_path.read_text())
                runs.append(
                    {
                        "family": family,
                        "N": n,
                        "seed": seed,
                        "run_dir": run_dir,
                        "summary": summary,
                        "history_train": load_history(train_path),
                        "history_val": load_history(val_path),
                    }
                )
    return runs


def plot_run(run: dict, out_dir: Path) -> dict:
    family = run["family"]
    n = run["N"]
    seed = run["seed"]
    summary = run["summary"]
    train = run["history_train"]
    val = run["history_val"]
    epochs = np.arange(1, len(train) + 1)
    best_epoch = int(np.argmin(val)) + 1
    best_val = float(np.min(val))

    train_reduction = relative_reduction(train)
    val_reduction = relative_reduction(val)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)

    axes[0].plot(epochs, train, marker="o", markersize=3.5, linewidth=1.8, label="Train L1 Loss")
    axes[0].plot(epochs, val, marker="o", markersize=3.5, linewidth=1.8, label="Validation L1 Loss")
    axes[0].axvline(best_epoch, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 Loss / MAE (eV per atom)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        epochs,
        train_reduction,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Train Error Reduction",
    )
    axes[1].plot(
        epochs,
        val_reduction,
        marker="o",
        markersize=3.5,
        linewidth=1.8,
        label="Validation Error Reduction",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_title("Relative Error Reduction")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Improvement From Epoch 1 (%)")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    title = (
        f"{family.capitalize()} | N={n} | seed={seed} | "
        f"test MAE={summary['test_mae_eV_per_atom']:.5f}"
    )
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        (
            f"Protocol: pretrained ALIGNN with partial fine-tuning only; "
            f"unfrozen groups = {', '.join(summary.get('unfrozen_groups', [])) or 'unknown'}; "
            f"best val L1 = {best_val:.5f} at epoch {best_epoch}"
        ),
        ha="center",
        fontsize=9,
    )

    family_dir = out_dir / family
    family_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{family}_N{n}_seed{seed}_training_curves"
    png_path = family_dir / f"{stem}.png"
    pdf_path = family_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "family": family,
        "N": n,
        "seed": seed,
        "epochs": len(train),
        "best_val_epoch": best_epoch,
        "best_val_l1": best_val,
        "test_mae_eV_per_atom": float(summary["test_mae_eV_per_atom"]),
        "png_path": str(png_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }


def plot_family_grid(runs: list[dict], family: str, out_dir: Path) -> dict:
    family_runs = [run for run in runs if run["family"] == family]
    if not family_runs:
        raise ValueError(f"No runs found for family={family}")

    ns = sorted({int(run["N"]) for run in family_runs})
    seeds = sorted({int(run["seed"]) for run in family_runs})
    lookup = {(int(run["N"]), int(run["seed"])): run for run in family_runs}

    fig, axes = plt.subplots(
        len(seeds),
        len(ns),
        figsize=(3.6 * len(ns), 2.8 * len(seeds)),
        sharex=True,
        sharey=False,
        constrained_layout=True,
    )
    if len(seeds) == 1 and len(ns) == 1:
        axes = np.array([[axes]])
    elif len(seeds) == 1:
        axes = np.array([axes])
    elif len(ns) == 1:
        axes = np.array([[ax] for ax in axes])

    for row_idx, seed in enumerate(seeds):
        for col_idx, n in enumerate(ns):
            ax = axes[row_idx, col_idx]
            run = lookup.get((n, seed))
            if run is None:
                ax.set_visible(False)
                continue
            train = run["history_train"]
            val = run["history_val"]
            epochs = np.arange(1, len(train) + 1)
            ax.plot(epochs, train, linewidth=1.4, label="Train", color="#1f77b4")
            ax.plot(epochs, val, linewidth=1.4, label="Val", color="#ff7f0e")
            best_epoch = int(np.argmin(val)) + 1
            ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title(f"N={n}, seed={seed}", fontsize=10)
            ax.grid(alpha=0.2)
            if row_idx == len(seeds) - 1:
                ax.set_xlabel("Epoch")
            if col_idx == 0:
                ax.set_ylabel("L1 Loss")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(
        f"{family.capitalize()} finetune Training-Curve Grid (three seeds x six N values)",
        fontsize=14,
        fontweight="bold",
    )

    png_path = out_dir / f"{family}_training_curve_grid.png"
    pdf_path = out_dir / f"{family}_training_curve_grid.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "family": family,
        "png_path": str(png_path.resolve()),
        "pdf_path": str(pdf_path.resolve()),
    }


def write_manifest(out_dir: Path, rows: list[dict], grid_rows: list[dict]) -> None:
    csv_path = out_dir / "training_curve_manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "family",
                "N",
                "seed",
                "epochs",
                "best_val_epoch",
                "best_val_l1",
                "test_mae_eV_per_atom",
                "png_path",
                "pdf_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    readme = out_dir / "readme.md"
    lines = [
        "# finetune Training Curves",
        "",
        "These figures were generated from the saved `history_train.json` and `history_val.json` files for every completed finetune fine-tuning run.",
        "",
        "Important note:",
        "This project is a regression task, not a classification task. That means there is no true training `accuracy` metric in the saved histories.",
        "Instead, each per-run figure contains:",
        "- left panel: train and validation L1 loss curves",
        "- right panel: train and validation relative error reduction from epoch 1",
        "",
        "Files in this folder:",
        "- one PNG and one PDF per fine-tuning run",
        "- one grid figure per family for quick browsing",
        "- `training_curve_manifest.csv` with file paths and run metadata",
    ]
    if grid_rows:
        lines.extend(["", "Family grid figures:"])
        for row in grid_rows:
            lines.append(f"- {row['family']}: `{row['png_path']}` and `{row['pdf_path']}`")
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "per_run_manifest_csv": str(csv_path.resolve()),
        "readme": str(readme.resolve()),
        "n_runs": len(rows),
        "family_grids": grid_rows,
    }
    (out_dir / "training_curve_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--families", nargs="+", default=["oxide", "nitride"])
    parser.add_argument("--Ns", nargs="+", type=int, default=[10, 50, 100, 200, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--experiment-tag", default="finetune_alignn_hyperparameters_new_script")
    parser.add_argument("--run-subdir", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    run_subdir = args.run_subdir or f"finetune_last2_{args.experiment_tag}"
    out_dir = (repo_root / (args.out_dir or f"results/derived_evidence/{args.experiment_tag}/training_curves")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(repo_root, args.families, args.Ns, args.seeds, run_subdir)
    if not runs:
        raise SystemExit("No completed fine-tuning runs with history files were found.")

    per_run_rows = [plot_run(run, out_dir) for run in runs]
    grid_rows = [plot_family_grid(runs, family, out_dir) for family in args.families]
    write_manifest(out_dir, per_run_rows, grid_rows)

    print(
        json.dumps(
            {
                "experiment_tag": args.experiment_tag,
                "run_subdir": run_subdir,
                "n_runs": len(per_run_rows),
                "out_dir": str(out_dir),
                "manifest_csv": str((out_dir / "training_curve_manifest.csv").resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
