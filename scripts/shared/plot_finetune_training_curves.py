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


def repo_relative(repo: Path, path: Path) -> str:
    return os.path.relpath(path.resolve(), repo.resolve())


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


def discover_runs(
    repo_root: Path,
    results_root: str,
    run_subdir: str,
    families: list[str],
    ns: list[int],
    seeds: list[int],
) -> list[dict]:
    runs: list[dict] = []
    for family in families:
        for n in ns:
            for seed in seeds:
                run_dir = repo_root / results_root / family / f"N{n}_seed{seed}" / run_subdir
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


def plot_run(repo_root: Path, run: dict, out_dir: Path, protocol_note: str) -> dict:
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

    title = f"{family.capitalize()} | N={n} | seed={seed} | test MAE={summary['test_mae_eV_per_atom']:.5f}"
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        -0.01,
        (
            f"Protocol: {protocol_note}; "
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
        "png_path": repo_relative(repo_root, png_path),
        "pdf_path": repo_relative(repo_root, pdf_path),
    }


def plot_family_grid(
    repo_root: Path,
    runs: list[dict],
    family: str,
    out_dir: Path,
    title_label: str,
) -> dict:
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
        f"{family.capitalize()} {title_label} Training-Curve Grid ({len(seeds)} seeds x {len(ns)} N values)",
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
        "png_path": repo_relative(repo_root, png_path),
        "pdf_path": repo_relative(repo_root, pdf_path),
    }


def write_manifest(
    repo_root: Path,
    out_dir: Path,
    rows: list[dict],
    grid_rows: list[dict],
    results_root: str,
    run_subdir: str,
) -> None:
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
        "These figures were generated from the saved `history_train.json` and `history_val.json` files for every completed finetune fine-tuning run in this namespace.",
        "",
        f"- results root: `{results_root}`",
        f"- run subdirectory: `{run_subdir}`",
        "",
        "Each per-run figure contains:",
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
        "results_root": results_root,
        "run_subdir": run_subdir,
        "per_run_manifest_csv": repo_relative(repo_root, csv_path),
        "readme": repo_relative(repo_root, readme),
        "n_runs": len(rows),
        "family_grids": grid_rows,
    }
    (out_dir / "training_curve_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--results-root", default="results/protocol_1")
    parser.add_argument("--run-subdir", default="finetune_last2")
    parser.add_argument("--families", nargs="+", default=["oxide", "nitride"])
    parser.add_argument("--Ns", nargs="+", type=int, default=[10, 50, 100, 200, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", default="results/derived_evidence/finetune/training_curves")
    parser.add_argument("--title-label", default="finetune")
    parser.add_argument(
        "--protocol-note",
        default="pretrained ALIGNN with partial fine-tuning only",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    out_dir = (repo_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(repo_root, args.results_root, args.run_subdir, args.families, args.Ns, args.seeds)
    if not runs:
        raise SystemExit("No completed fine-tuning runs with history files were found.")

    per_run_rows = [plot_run(repo_root, run, out_dir, args.protocol_note) for run in runs]
    grid_rows = [plot_family_grid(repo_root, runs, family, out_dir, args.title_label) for family in args.families]
    write_manifest(repo_root, out_dir, per_run_rows, grid_rows, args.results_root, args.run_subdir)

    print(
        json.dumps(
            {
                "results_root": args.results_root,
                "run_subdir": args.run_subdir,
                "n_runs": len(per_run_rows),
                "out_dir": repo_relative(repo_root, out_dir),
                "manifest_csv": repo_relative(repo_root, out_dir / "training_curve_manifest.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
