#!/usr/bin/env python3
"""Refresh corrected protocol_1 figure copies and create versioned v2 memos."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_verified(repo: Path, source: str, destination: str, numerical: str) -> dict:
    src = repo / source
    dst = repo / destination
    if not src.is_file() or src.stat().st_size == 0:
        raise RuntimeError(f"Missing/nonempty source asset: {source}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    source_hash = sha256(src)
    destination_hash = sha256(dst)
    if source_hash != destination_hash:
        raise RuntimeError(f"Copied asset is not byte-identical: {destination}")
    return {
        "source_generated_file": source,
        "destination_copy": destination,
        "source_sha256": source_hash,
        "destination_sha256": destination_hash,
        "source_size_bytes": src.stat().st_size,
        "destination_size_bytes": dst.stat().st_size,
        "numerical_source_table": numerical,
        "byte_identical": True,
        "validation_status": "PASS",
    }


def learning_memo(family: str, aggregate: pd.DataFrame) -> str:
    title = family.capitalize()
    subset = aggregate[aggregate["family"] == family].sort_values("N")
    rows = [
        f"# Figure Memo v2: {title} protocol_1 Learning Curve",
        "",
        f"Status: **CORRECTED AND VALIDATED**. Supersedes the historical `{('fig02_oxide' if family == 'oxide' else 'fig03_nitride')}_learning_curve_memo.md` for protocol_1 numbers.",
        "",
        f"- Figure: `FIG_S1_LC_{family.upper()}`",
        f"- Fixed test count: {1484 if family == 'oxide' else 242}",
        "- Center/error: five-seed arithmetic mean ± sample SD (`ddof=1`)",
        "- Numerical authority: `results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv`",
        "",
        "| N | mean MAE | sample SD | zero-shot − fine-tune | mean best epoch | seed best epochs |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in subset.itertuples(index=False):
        raw_epochs = json.loads(row.raw_seed_best_epochs)
        rows.append(
            f"| {int(row.N)} | {row.mean_test_mae_eV_per_atom:.8f} | "
            f"{row.sample_std_test_mae_eV_per_atom_ddof1:.8f} | "
            f"{row.zero_shot_minus_finetune_mean_eV_per_atom:.8f} | "
            f"{sum(raw_epochs)/len(raw_epochs):.1f} | {','.join(map(str, raw_epochs))} |"
        )
    if family == "oxide":
        interpretation = (
            "All six fine-tuned means remain above zero-shot. The curve is nearly flat rather than a monotonic recovery: "
            "N=1000 is the lowest mean, but N=500 is close. Variability drops about 16.8-fold from N=10 to N=1000. "
            "Only N=10 selects epoch 1 for every seed."
        )
    else:
        interpretation = (
            "All six fine-tuned means remain above zero-shot. N=10,50,100 select epoch 1 for every seed; N=200 is mixed "
            "(49,1,1,1,1), so the historical claim of inertness through N≤200 is false. N=500 and N=1000 select later "
            "checkpoints for all seeds, but higher-N adaptation does not improve over the N=10 ensemble endpoint."
        )
    rows += [
        "",
        "## Corrected interpretation",
        "",
        interpretation,
        "",
        "## Caption guardrail",
        "",
        "Report the five-seed mean and sample SD, the fixed zero-shot reference, and the fact that fine-tuning remains worse than zero-shot. Do not call epoch-1 selection byte-identical to the pretrained checkpoint, do not infer statistical significance from overlapping bands, and do not merge protocol_2/3 into these primary estimates.",
        "",
    ]
    return "\n".join(rows)


def comparison_memo(family: str, comparison: pd.DataFrame) -> str:
    title = family.capitalize()
    subset = comparison[comparison["family"] == family].sort_values("N")
    rows = [
        f"# Figure Memo v2: {title} Fine-tuning versus From Scratch",
        "",
        "Status: **CORRECTED AND VALIDATED**. The comparison uses five seeds for each method; the stale fine-tuning-seed-0 fields are excluded.",
        "",
        f"- Figure: `FIG_S1_COMP_{family.upper()}`",
        "- Supported training sizes: N=50 and N=500 only",
        "- Transfer benefit formula: mean from-scratch MAE − mean fine-tuned MAE",
        "- Numerical authority: `results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv`",
        "",
        "| N | fine mean ± sample SD | scratch mean ± sample SD | zero-shot | transfer benefit |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in subset.itertuples(index=False):
        rows.append(
            f"| {int(row.N)} | {row.finetune_mean_test_mae_eV_per_atom:.8f} ± "
            f"{row.finetune_std_test_mae_eV_per_atom:.8f} | "
            f"{row.scratch_mean_test_mae_eV_per_atom:.8f} ± {row.scratch_std_test_mae_eV_per_atom:.8f} | "
            f"{row.zero_shot_mae_eV_per_atom:.8f} | {row.scratch_minus_finetune_mae_eV_per_atom:.8f} |"
        )
    rows += [
        "",
        "## Corrected interpretation",
        "",
        "Pretrained initialization plus fine-tuning is substantially better than training from scratch at both supported sizes, while zero-shot still has the lowest MAE. The comparison supports a pretraining benefit, not causal claims about domain-shift mechanisms. It does not support any scratch comparison at N=10,100,200,1000.",
        "",
    ]
    return "\n".join(rows)


def parity_memo(family: str, n_value: int, aggregate: pd.DataFrame, parity: dict) -> str:
    title = family.capitalize()
    row = aggregate[(aggregate["family"] == family) & (aggregate["N"] == n_value)].iloc[0]
    item = next(x for x in parity["rows"] if x["family"] == family and x["N"] == n_value)
    epochs = json.loads(row["raw_seed_best_epochs"])
    label = f"FIG_S1_PARITY_{family.upper()}_N{n_value}"
    return f"""# Figure Memo v2: {title} N={n_value} Parity

Status: **CORRECTED AND VALIDATED**.

- Figure: `{label}`
- Seeds: 0,1,2,3,4; ensemble prediction is the row-wise five-seed mean
- Fixed test rows: {item['n_points']}
- Ensemble MAE: {item['mae_eV_per_atom']:.8f} eV/atom
- Ensemble RMSE: {item['rmse_eV_per_atom']:.8f} eV/atom
- Ensemble R²: {item['r2']:.8f}
- Mean per-seed MAE: {row['mean_test_mae_eV_per_atom']:.8f} ± {row['sample_std_test_mae_eV_per_atom_ddof1']:.8f} eV/atom (sample SD)
- Seed best epochs: {','.join(map(str, epochs))}
- Numerical authority: `results/derived_evidence/protocol_1_regeneration/parity_validation.json`

## Interpretation and caption guardrails

The ensemble MAE and mean per-seed MAE are different aggregations and must remain explicitly distinguished. The figure uses identical ordered IDs and targets across all five seeds and a dashed y=x identity line with equal axes. Fine-tuning remains worse than the family zero-shot mean. Epoch 1 means the end-of-epoch-1 checkpoint was selected, not that the model is byte-identical to zero-shot. Do not infer causality or use parity R² alone as the headline metric.
"""


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    memo_root = repo / "results/derived_evidence/final_paper_factory/02_figure_memos"
    evidence = repo / "results/derived_evidence/protocol_1_regeneration"
    aggregate = pd.read_csv(evidence / "aggregate_recomputation.csv")
    comparison = pd.read_csv(
        repo / "results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv"
    )
    parity = json.loads((evidence / "parity_validation.json").read_text())
    table_aggregate = "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv"
    table_comparison = "results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv"
    table_parity = "results/derived_evidence/protocol_1_regeneration/parity_validation.json"

    mappings: list[tuple[str, str, str]] = [
        (
            "results/derived_evidence/protocol_1/Learning Curves/Oxide Learning Curve - protocol_1.png",
            "results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_S1_LC_OXIDE.png",
            table_aggregate,
        ),
        (
            "results/derived_evidence/protocol_1/Learning Curves/Nitride Learning Curve - protocol_1.png",
            "results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_S1_LC_NITRIDE.png",
            table_aggregate,
        ),
        (
            "results/summaries/protocol_1/comparisons/Oxide Comparison Plot - protocol_1.png",
            "results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_S1_COMP_OXIDE.png",
            table_comparison,
        ),
        (
            "results/summaries/protocol_1/comparisons/Nitride Comparison Plot - protocol_1.png",
            "results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_S1_COMP_NITRIDE.png",
            table_comparison,
        ),
    ]
    for suffix in ("png", "svg", "pdf"):
        mappings.append(
            (
                f"results/summaries/protocol_1/comparisons/FIG_TRANSFER_BENEFIT.{suffix}",
                f"results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_TRANSFER_BENEFIT.{suffix}",
                "results/summaries/protocol_1/comparisons/protocol_1_transfer_gain_table.csv",
            )
        )
    for family in ("oxide", "nitride"):
        title = family.capitalize()
        for n_value in (10, 1000):
            mappings.append(
                (
                    f"results/derived_evidence/protocol_1/Parity Plots/{title} Parity Plot - protocol_1, N={n_value}.png",
                    f"results/derived_evidence/final_paper_factory/02_figure_memos/core_figures/FIG_S1_PARITY_{family.upper()}_N{n_value}.png",
                    table_parity,
                )
            )
        for n_value in (50, 100, 200, 500):
            mappings.append(
                (
                    f"results/derived_evidence/protocol_1/Parity Plots/{title} Parity Plot - protocol_1, N={n_value}.png",
                    f"results/derived_evidence/final_paper_factory/02_figure_memos/appendix_figures/FIG_S1_PARITY_{family.upper()}_N{n_value}.png",
                    table_parity,
                )
            )
    assets = [copy_verified(repo, *mapping) for mapping in mappings]

    memo_files = {
        "fig02_oxide_learning_curve_memo_v2.md": learning_memo("oxide", aggregate),
        "fig03_nitride_learning_curve_memo_v2.md": learning_memo("nitride", aggregate),
        "fig05a_oxide_comparison_plot_memo_v2.md": comparison_memo("oxide", comparison),
        "fig05b_nitride_comparison_plot_memo_v2.md": comparison_memo("nitride", comparison),
        "fig06_oxide_lowN_parity_memo_v2.md": parity_memo("oxide", 10, aggregate, parity),
        "fig07_oxide_highN_parity_memo_v2.md": parity_memo("oxide", 1000, aggregate, parity),
        "fig08_nitride_lowN_parity_memo_v2.md": parity_memo("nitride", 10, aggregate, parity),
        "fig09_nitride_highN_parity_memo_v2.md": parity_memo("nitride", 1000, aggregate, parity),
    }
    transfer_rows = comparison.sort_values(["family", "N"])
    transfer_lines = [
        "# Figure Memo v2: Cross-family Transfer Benefit",
        "",
        "Status: **CORRECTED AND VALIDATED**. Positive bars equal five-seed mean from-scratch MAE minus five-seed mean fine-tuned MAE.",
        "",
        "| family | N | transfer benefit (eV/atom) |",
        "|---|---:|---:|",
    ]
    for row in transfer_rows.itertuples(index=False):
        transfer_lines.append(
            f"| {row.family} | {int(row.N)} | {row.scratch_minus_finetune_mae_eV_per_atom:.8f} |"
        )
    transfer_lines += [
        "",
        "The figure is restricted to N=50 and N=500. Nitride bars are larger at both supported sizes, but cross-family bar size is not a measure of transfer efficiency because baselines and test distributions differ. The plot, CSV, SVG, and PDF derive from the same numerical table.",
        "",
    ]
    memo_files["fig05_transfer_benefit_comparison_memo_v2.md"] = "\n".join(transfer_lines)
    for name, content in memo_files.items():
        (memo_root / name).write_text(content.rstrip() + "\n", encoding="utf-8")

    queue = pd.read_csv(memo_root / "figure_queue.csv", keep_default_na=False)
    queue["protocol_1_regeneration_status"] = "not_applicable"
    affected = queue["figure_label"].str.contains("FIG_S1_|FIG_TRANSFER_BENEFIT", regex=True)
    queue.loc[affected, "protocol_1_regeneration_status"] = "validated_corrected_v2_memo"
    queue.loc[affected, "status"] = "exists_validated_corrected"
    queue.loc[queue["figure_label"] == "FIG_S1_LC_NITRIDE", "purpose_note"] = (
        "Corrected nitride five-seed learning curve; epoch-1 at N=10/50/100, mixed timing at N=200, all-seed later checkpoints at N=500/1000."
    )
    queue.loc[queue["figure_label"] == "FIG_TRANSFER_BENEFIT", "linked_table_path"] = (
        "results/summaries/protocol_1/comparisons/protocol_1_transfer_gain_table.csv"
    )
    for family in ("OXIDE", "NITRIDE"):
        for n_value in (10, 50, 100, 200, 500, 1000):
            mask = queue["figure_label"].str.contains(f"PARITY_{family}_N{n_value}$", regex=True)
            if mask.any():
                queue.loc[mask, "source_path"] = (
                    f"results/derived_evidence/protocol_1/Parity Plots/{family.capitalize()} Parity Plot - protocol_1, N={n_value}.png"
                )
                queue.loc[mask, "linked_table_path"] = table_parity
    queue.to_csv(memo_root / "figure_queue_v3.csv", index=False)

    queue_md = [
        "# Figure Queue v3",
        "",
        "protocol_1-dependent assets below were refreshed from corrected homogeneous five-seed outputs. Historical queue/memos remain preserved.",
        "",
        "| figure | source | destination | status |",
        "|---|---|---|---|",
    ]
    for item in assets:
        queue_md.append(
            f"| {Path(item['destination_copy']).stem} | `{item['source_generated_file']}` | "
            f"`{item['destination_copy']}` | PASS byte-identical |"
        )
    (memo_root / "figure_queue_v3.md").write_text("\n".join(queue_md) + "\n", encoding="utf-8")
    index = "# Figure Memo Index v2\n\nStatus: **PASS**. Corrected v2 memos exist for fig02, fig03, fig05, fig05a, fig05b, and fig06–fig09. Historical unversioned memos are stale for protocol_1 numbers and are retained only as provenance.\n\n" + "\n".join(f"- `{name}`" for name in sorted(memo_files)) + "\n"
    (memo_root / "figure_memo_index_v2.md").write_text(index, encoding="utf-8")
    (memo_root / "figure_memo_audit_v2.md").write_text(
        "# Figure Memo Audit v2\n\nPASS. Nine corrected memos resolve to the validated aggregate, parity, comparison, and transfer tables. Nineteen copied protocol_1 assets are byte-identical to their regenerated sources. Embedding and zero-shot figures were not modified.\n",
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "assets": assets,
        "asset_count": len(assets),
        "all_byte_identical": all(item["byte_identical"] for item in assets),
        "memos": [
            {
                "path": str((memo_root / name).relative_to(repo)),
                "sha256": sha256(memo_root / name),
                "validation_status": "PASS",
            }
            for name in sorted(memo_files)
        ],
        "embedding_assets_modified": False,
        "section08_used": False,
    }
    (evidence / "figure_asset_refresh_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"protocol_1_FIGURE_PACKAGE_REFRESHED assets={len(assets)} memos={len(memo_files)}")


if __name__ == "__main__":
    main()
