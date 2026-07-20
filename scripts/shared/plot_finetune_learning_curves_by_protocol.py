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


def render_template(template: str, family: str) -> str:
    return template.format(family=family, family_title=family.capitalize())


def plot_family_curve(df, family: str, out_png: Path, out_pdf: Path, plt, plot_title: str) -> None:
    family_df = df[df["family"] == family].sort_values("N")
    if family_df.empty:
        return

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        family_df["N"],
        family_df["mean_test_mae_eV_per_atom"],
        marker="o",
        linewidth=2,
        label="Fine-Tune Mean",
        color="#1f77b4",
    )
    plt.fill_between(
        family_df["N"],
        family_df["mean_test_mae_eV_per_atom"] - family_df["std_test_mae_eV_per_atom"].fillna(0.0),
        family_df["mean_test_mae_eV_per_atom"] + family_df["std_test_mae_eV_per_atom"].fillna(0.0),
        alpha=0.2,
        color="#1f77b4",
        label="Fine-Tune ±1 Std",
    )
    if family_df["zero_shot_mae_eV_per_atom"].notna().any():
        zero = float(family_df["zero_shot_mae_eV_per_atom"].dropna().iloc[0])
        plt.axhline(zero, linestyle="--", color="#d62728", label=f"Zero-Shot ({zero:.4f})")

    plt.xscale("log")
    plt.xticks(family_df["N"], [str(n) for n in family_df["N"]])
    plt.xlabel("Fine-Tuning Size N")
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
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title-template", required=True)
    parser.add_argument("--filename-template", required=True)
    args = parser.parse_args()

    pd, plt = load_reporting_dependencies()
    df = pd.read_csv(args.summary_csv)
    if df.empty:
        raise SystemExit("No fine-tuning summary rows found.")

    out_dir = Path(args.out_dir)
    for family in sorted(df["family"].unique()):
        basename = render_template(args.filename_template, family)
        plot_family_curve(
            df,
            family,
            out_dir / f"{basename}.png",
            out_dir / f"{basename}.pdf",
            plt,
            render_template(args.title_template, family),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
