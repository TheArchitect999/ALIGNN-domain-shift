from __future__ import annotations

import argparse
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
        missing_text = ", ".join(missing)
        raise SystemExit(f"Missing reporting dependencies: {missing_text}.")
    return pd, plt


def render_title(title_template: str, family: str) -> str:
    return title_template.format(family=family, family_title=family.capitalize())


def render_basename(filename_template: str, family: str) -> str:
    return filename_template.format(family=family, family_title=family.capitalize())


def plot_family_comparison(df, family: str, out_png: Path, out_pdf: Path, plt, plot_title: str) -> None:
    family_df = df[df["family"] == family].sort_values("N")
    if family_df.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        family_df["N"],
        family_df["finetune_mean_test_mae_eV_per_atom"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Fine-tune mean",
    )
    plt.fill_between(
        family_df["N"],
        family_df["finetune_mean_test_mae_eV_per_atom"] - family_df["finetune_std_test_mae_eV_per_atom"].fillna(0.0),
        family_df["finetune_mean_test_mae_eV_per_atom"] + family_df["finetune_std_test_mae_eV_per_atom"].fillna(0.0),
        alpha=0.2,
        color="#1f77b4",
        label="Fine-tune ±1 std",
    )
    plt.plot(
        family_df["N"],
        family_df["from_scratch_mean_test_mae_eV_per_atom"],
        marker="o",
        linewidth=2,
        color="#ff7f0e",
        label="From scratch mean",
    )
    plt.fill_between(
        family_df["N"],
        family_df["from_scratch_mean_test_mae_eV_per_atom"] - family_df["from_scratch_std_test_mae_eV_per_atom"].fillna(0.0),
        family_df["from_scratch_mean_test_mae_eV_per_atom"] + family_df["from_scratch_std_test_mae_eV_per_atom"].fillna(0.0),
        alpha=0.2,
        color="#ff7f0e",
        label="From scratch ±1 std",
    )

    if family_df["zero_shot_mae_eV_per_atom"].notna().any():
        zero = float(family_df["zero_shot_mae_eV_per_atom"].dropna().iloc[0])
        plt.axhline(zero, linestyle="--", color="tab:red", label=f"Zero-shot ({zero:.4f})")

    plt.xscale("log")
    plt.xticks(family_df["N"], [str(v) for v in family_df["N"]])
    plt.xlabel("Training size N")
    plt.ylabel("Test MAE (eV/atom)")
    plt.title(plot_title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=200)
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune-summary", required=True)
    parser.add_argument("--from_scratch-summary", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-template", required=True)
    parser.add_argument("--filename-template", default="{family}_from_scratch_comparison")
    args = parser.parse_args()

    pd, plt = load_reporting_dependencies()
    finetune_df = pd.read_csv(args.finetune_summary)
    from_scratch_df = pd.read_csv(args.from_scratch_summary)

    merged = finetune_df.merge(
        from_scratch_df,
        on=["family", "N"],
        suffixes=("_finetune", "_from_scratch"),
        how="inner",
    )
    if merged.empty:
        raise SystemExit("No overlapping fine-tune/from-scratch summary rows found.")

    merged = merged.rename(
        columns={
            "mean_test_mae_eV_per_atom_finetune": "finetune_mean_test_mae_eV_per_atom",
            "std_test_mae_eV_per_atom_finetune": "finetune_std_test_mae_eV_per_atom",
            "mean_test_mae_eV_per_atom_from_scratch": "from_scratch_mean_test_mae_eV_per_atom",
            "std_test_mae_eV_per_atom_from_scratch": "from_scratch_std_test_mae_eV_per_atom",
        }
    )

    if "zero_shot_mae_eV_per_atom_from_scratch" in merged.columns:
        merged["zero_shot_mae_eV_per_atom"] = merged["zero_shot_mae_eV_per_atom_from_scratch"]
    elif "zero_shot_mae_eV_per_atom_finetune" in merged.columns:
        merged["zero_shot_mae_eV_per_atom"] = merged["zero_shot_mae_eV_per_atom_finetune"]
    else:
        merged["zero_shot_mae_eV_per_atom"] = None

    out_dir = Path(args.out_dir)
    for family in sorted(merged["family"].unique()):
        basename = render_basename(args.filename_template, family)
        plot_family_comparison(
            merged,
            family,
            out_dir / f"{basename}.png",
            out_dir / f"{basename}.pdf",
            plt,
            render_title(args.title_template, family),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
