#!/usr/bin/env python3
"""Regenerate the 24 run grids and two robustness surfaces in the supplement.

The producer reads only the released run histories, predictions, summaries, and
zero-shot prediction tables.  It preserves the plotting geometry used by the
archived producer while using the public ``Protocol 1``--``Protocol 3`` names in
reviewer-visible titles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/alignn-domain-shift-supplement-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/alignn-domain-shift-supplement-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("oxide", "nitride")
PROTOCOLS = (1, 2, 3)
FINE_TUNE_SIZES = (10, 50, 100, 200, 500, 1000)
SCRATCH_SIZES = (50, 500)
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class Run:
    protocol: int
    experiment_type: str
    family: str
    n_value: int
    seed: int
    test_mae: float
    best_epoch: int
    best_val_l1: float
    train_history: tuple[float, ...]
    val_history: tuple[float, ...]
    targets: tuple[float, ...]
    predictions: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/supplementary/figures"),
    )
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def history_values(path: Path) -> tuple[float, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Invalid history: {path}")
    values = tuple(float(row[0] if isinstance(row, list) else row) for row in payload)
    if not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError(f"Non-finite history: {path}")
    return values


def prediction_values(path: Path) -> tuple[tuple[float, ...], tuple[float, ...]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not {"target", "prediction"}.issubset(rows[0]):
        raise ValueError(f"Invalid prediction table: {path}")
    targets = tuple(float(row["target"]) for row in rows)
    predictions = tuple(float(row["prediction"]) for row in rows)
    return targets, predictions


def run_leaf(protocol: int, experiment_type: str) -> str:
    if experiment_type == "fine_tune":
        return "finetune_last2" if protocol in {1, 2} else "finetune_last2_epochs100_bs32_lr5e5"
    return (
        "train_alignn_from_scratch"
        if protocol in {1, 2}
        else "train_alignn_from_scratch_epochs100_bs32_lr5e5"
    )


def load_runs(root: Path) -> list[Run]:
    runs: list[Run] = []
    for protocol in PROTOCOLS:
        for experiment_type, sizes in (
            ("fine_tune", FINE_TUNE_SIZES),
            ("from_scratch", SCRATCH_SIZES),
        ):
            for family in FAMILIES:
                for n_value in sizes:
                    for seed in SEEDS:
                        directory = (
                            root
                            / f"results/protocol_{protocol}/{family}/N{n_value}_seed{seed}"
                            / run_leaf(protocol, experiment_type)
                        )
                        summary_path = directory / "summary.json"
                        train_path = directory / "history_train.json"
                        val_path = directory / "history_val.json"
                        prediction_path = directory / "prediction_results_test_set.csv"
                        for path in (summary_path, train_path, val_path, prediction_path):
                            if not path.is_file():
                                raise FileNotFoundError(path)

                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        train_history = history_values(train_path)
                        val_history = history_values(val_path)
                        targets, predictions = prediction_values(prediction_path)
                        errors = np.abs(np.asarray(targets) - np.asarray(predictions))
                        test_mae = float(errors.mean())
                        best_index = min(range(len(val_history)), key=val_history.__getitem__)
                        if val_history.count(val_history[best_index]) != 1:
                            raise ValueError(f"Non-unique validation minimum: {val_path}")
                        if not math.isclose(
                            float(summary["test_mae_eV_per_atom"]),
                            test_mae,
                            rel_tol=0.0,
                            abs_tol=1e-12,
                        ):
                            raise ValueError(f"MAE mismatch: {summary_path}")
                        runs.append(
                            Run(
                                protocol=protocol,
                                experiment_type=experiment_type,
                                family=family,
                                n_value=n_value,
                                seed=seed,
                                test_mae=test_mae,
                                best_epoch=best_index + 1,
                                best_val_l1=val_history[best_index],
                                train_history=train_history,
                                val_history=val_history,
                                targets=targets,
                                predictions=predictions,
                            )
                        )
    if len(runs) != 240:
        raise ValueError(f"Expected 240 released runs, found {len(runs)}")
    return runs


def save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def plot_training_grid(group: list[Run], output: Path) -> None:
    sizes = sorted({run.n_value for run in group})
    ncols = 3 if len(sizes) > 2 else 2
    nrows = math.ceil(len(sizes) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.2 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for panel, n_value in enumerate(sizes):
        axis = axes.flat[panel]
        for run in sorted(
            (item for item in group if item.n_value == n_value), key=lambda item: item.seed
        ):
            epochs = np.arange(1, len(run.val_history) + 1)
            color = colors[run.seed]
            axis.plot(epochs, run.train_history, color=color, alpha=0.35, lw=0.8, ls="--")
            axis.plot(
                epochs,
                run.val_history,
                color=color,
                alpha=0.9,
                lw=1.0,
                label=f"seed {run.seed}",
            )
            axis.scatter([run.best_epoch], [run.best_val_l1], color=color, s=18, zorder=3)
        axis.set_title(f"N = {n_value}", fontsize=10, weight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("L1 loss")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
    for panel in range(len(sizes), nrows * ncols):
        axes.flat[panel].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    method = "fine-tuning" if group[0].experiment_type == "fine_tune" else "from scratch"
    figure.suptitle(
        f"Protocol {group[0].protocol} · {group[0].family.title()} · {method} training curves\n"
        "solid = validation; dashed = training; marker = selected validation epoch",
        fontsize=12,
        weight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 0.94))
    save_figure(figure, output)


def plot_parity_grid(group: list[Run], output: Path) -> None:
    sizes = sorted({run.n_value for run in group})
    ncols = 3 if len(sizes) > 2 else 2
    nrows = math.ceil(len(sizes) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(11.5, 3.35 * nrows), squeeze=False)
    colors = plt.get_cmap("tab10").colors
    for panel, n_value in enumerate(sizes):
        axis = axes.flat[panel]
        local = [run for run in group if run.n_value == n_value]
        low = min(min(run.targets + run.predictions) for run in local)
        high = max(max(run.targets + run.predictions) for run in local)
        pad = max(0.05, 0.03 * (high - low))
        for run in sorted(local, key=lambda item: item.seed):
            axis.scatter(
                run.targets,
                run.predictions,
                s=4,
                alpha=0.12,
                color=colors[run.seed],
                rasterized=True,
            )
        axis.plot(
            [low - pad, high + pad],
            [low - pad, high + pad],
            color="#222222",
            lw=1.0,
            ls="--",
        )
        mean_mae = np.mean([run.test_mae for run in local])
        axis.set_title(f"N = {n_value}; mean MAE = {mean_mae:.4f}", fontsize=9.5, weight="bold")
        axis.set_xlim(low - pad, high + pad)
        axis.set_ylim(low - pad, high + pad)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("DFT target (eV/atom)")
        axis.set_ylabel("Prediction (eV/atom)")
        axis.grid(alpha=0.15)
    for panel in range(len(sizes), nrows * ncols):
        axes.flat[panel].axis("off")
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[seed],
            markeredgecolor="none",
            markersize=6,
            label=f"seed {seed}",
        )
        for seed in SEEDS
    ]
    figure.legend(handles=legend_handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    method = "fine-tuning" if group[0].experiment_type == "fine_tune" else "from scratch"
    figure.suptitle(
        f"Protocol {group[0].protocol} · {group[0].family.title()} · {method} parity grids\n"
        "All five seeds are overlaid; dashed line is ideal parity",
        fontsize=12,
        weight="bold",
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    save_figure(figure, output)


def zero_shot_mae(root: Path, family: str) -> float:
    targets, predictions = prediction_values(root / f"results/zero_shot/{family}/predictions.csv")
    return float(np.mean(np.abs(np.asarray(targets) - np.asarray(predictions))))


def aggregate_rows(runs: list[Run], protocol: int, family: str, experiment_type: str) -> list[dict[str, float]]:
    sizes = FINE_TUNE_SIZES if experiment_type == "fine_tune" else SCRATCH_SIZES
    rows: list[dict[str, float]] = []
    for n_value in sizes:
        local = [
            run
            for run in runs
            if run.protocol == protocol
            and run.family == family
            and run.experiment_type == experiment_type
            and run.n_value == n_value
        ]
        values = np.asarray([run.test_mae for run in local], dtype=float)
        rows.append(
            {
                "N": n_value,
                "mae_mean": float(values.mean()),
                "mae_sample_sd": float(values.std(ddof=1)),
            }
        )
    return rows


def plot_robustness_surface(root: Path, protocol: int, runs: list[Run], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=False)
    for axis, family in zip(axes, FAMILIES):
        fine_tune = aggregate_rows(runs, protocol, family, "fine_tune")
        scratch = aggregate_rows(runs, protocol, family, "from_scratch")
        x_values = np.arange(len(fine_tune))
        means = np.asarray([row["mae_mean"] for row in fine_tune])
        standard_deviations = np.asarray([row["mae_sample_sd"] for row in fine_tune])
        axis.errorbar(
            x_values,
            means,
            yerr=standard_deviations,
            color="#1f77b4",
            marker="o",
            lw=1.6,
            capsize=3,
            label="Fine-tune mean ± SD",
        )
        for index, row in enumerate(fine_tune):
            seed_values = [
                run.test_mae
                for run in runs
                if run.protocol == protocol
                and run.experiment_type == "fine_tune"
                and run.family == family
                and run.n_value == row["N"]
            ]
            axis.scatter(
                index + np.linspace(-0.10, 0.10, len(seed_values)),
                seed_values,
                color="#1f77b4",
                facecolors="white",
                s=24,
                zorder=3,
            )
        baseline = zero_shot_mae(root, family)
        axis.axhline(
            baseline,
            color="#2ca02c",
            ls="--",
            lw=1.4,
            label=f"Zero-shot ({baseline:.4f})",
        )
        for row in scratch:
            index = [item["N"] for item in fine_tune].index(row["N"])
            axis.errorbar(
                [index],
                [row["mae_mean"]],
                yerr=[row["mae_sample_sd"]],
                color="#d62728",
                marker="s",
                capsize=3,
            )
            seed_values = [
                run.test_mae
                for run in runs
                if run.protocol == protocol
                and run.experiment_type == "from_scratch"
                and run.family == family
                and run.n_value == row["N"]
            ]
            axis.scatter(
                index + np.linspace(-0.10, 0.10, len(seed_values)),
                seed_values,
                color="#d62728",
                facecolors="white",
                s=24,
                zorder=3,
            )
        axis.set_xticks(x_values, [str(row["N"]) for row in fine_tune])
        axis.set_yscale("log")
        axis.set_title(family.title(), weight="bold")
        axis.set_xlabel("Labelled budget N")
        axis.set_ylabel("Test MAE (eV/atom)")
        axis.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(
        plt.Line2D(
            [0],
            [0],
            color="#d62728",
            marker="s",
            lw=0,
            label="From scratch mean ± SD",
        )
    )
    labels.append("From scratch mean ± SD")
    figure.legend(handles, labels, ncol=3, loc="lower center", frameon=False, fontsize=8.5)
    figure.suptitle(
        f"Protocol {protocol} robustness: complete five-seed learning surface",
        weight="bold",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.94))
    save_figure(figure, output)


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_dir = resolve(root, args.output_dir).resolve()
    runs = load_runs(root)
    outputs: list[Path] = []
    for protocol in PROTOCOLS:
        for experiment_type in ("fine_tune", "from_scratch"):
            for family in FAMILIES:
                group = [
                    run
                    for run in runs
                    if run.protocol == protocol
                    and run.experiment_type == experiment_type
                    and run.family == family
                ]
                stem = f"s2_protocol_{protocol}_{experiment_type}_{family}"
                training = output_dir / f"{stem}_training_grid.png"
                parity = output_dir / f"{stem}_parity_grid.png"
                plot_training_grid(group, training)
                plot_parity_grid(group, parity)
                outputs.extend((training, parity))
    for protocol in (2, 3):
        output = output_dir / f"s3_protocol_{protocol}_robustness_learning_surface.png"
        plot_robustness_surface(root, protocol, runs, output)
        outputs.append(output)
    print(f"validated {len(runs)} released runs")
    print(f"wrote {len(outputs)} supplementary figures to {output_dir}")


if __name__ == "__main__":
    main()
