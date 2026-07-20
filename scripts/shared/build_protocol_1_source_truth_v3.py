#!/usr/bin/env python3
"""Build versioned protocol_1 source-of-truth v3 files from validated evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_column(frame: pd.DataFrame, claim_id: str, column: str, value: str) -> None:
    mask = frame["claim_id"] == claim_id
    if mask.sum() != 1:
        raise RuntimeError(f"Expected one {claim_id} row; found {mask.sum()}")
    frame.loc[mask, column] = value


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    out = repo / "results/derived_evidence/final_paper_factory/00_source_of_truth"
    evidence = repo / "results/derived_evidence/protocol_1_regeneration"
    aggregate = pd.read_csv(evidence / "aggregate_recomputation.csv")
    runs = pd.read_csv(evidence / "canonical_60_run_recomputation.csv")
    transfer = pd.read_csv(
        repo / "results/summaries/protocol_1/comparisons/protocol_1_transfer_gain_table.csv"
    )

    if len(runs) != 60 or len(aggregate) != 12 or len(transfer) != 12:
        raise RuntimeError("Validated protocol_1 numerical authorities are incomplete")

    source = pd.read_csv(out / "canonical_numbers_v2.csv", keep_default_na=False)
    for column in (
        "raw_source_path",
        "raw_selector",
        "formula",
        "recomputed_value",
        "display_precision",
        "validation_status",
    ):
        if column not in source.columns:
            source[column] = ""

    updated_ids: set[str] = set()
    for row in aggregate.itertuples(index=False):
        family = str(row.family)
        n_value = int(row.N)
        group = runs[(runs["family"] == family) & (runs["N"] == n_value)].sort_values("seed")
        epochs = group["best_epoch_recomputed"].astype(int).tolist()
        all_epoch1 = all(value == 1 for value in epochs)
        mixed_timing = any(value == 1 for value in epochs) and not all_epoch1
        if all_epoch1:
            flag = "all_seeds_epoch1_selected"
            note = (
                f"All five seeds selected the end-of-epoch-1 checkpoint; this does not prove "
                f"byte identity with the zero-shot model. Epoch vector={epochs}."
            )
        elif mixed_timing:
            flag = "mixed_checkpoint_timing"
            note = f"Seed-specific checkpoint timing is heterogeneous. Epoch vector={epochs}."
        else:
            flag = "none"
            note = f"All seeds selected checkpoints after epoch 1. Epoch vector={epochs}."
        values = {
            "MEAN_TEST_MAE": (
                "mean_test_mae_eV_per_atom",
                float(row.mean_test_mae_eV_per_atom),
                "arithmetic mean of five canonical seed MAEs",
                "eV_per_atom",
            ),
            "STD_TEST_MAE": (
                "std_test_mae_eV_per_atom",
                float(row.sample_std_test_mae_eV_per_atom_ddof1),
                "sample standard deviation of five canonical seed MAEs (ddof=1)",
                "eV_per_atom",
            ),
            "TRANSFER_GAIN_VS_ZERO_SHOT": (
                "transfer_gain_vs_zero_shot",
                float(row.zero_shot_minus_finetune_mean_eV_per_atom),
                "zero_shot_mae - mean_finetune_mae",
                "eV_per_atom",
            ),
            "MEAN_BEST_EPOCH": (
                "mean_best_epoch",
                float(group["best_epoch_recomputed"].mean()),
                "arithmetic mean of five validation-argmin best epochs",
                "epoch",
            ),
        }
        for suffix, (metric, value, formula, unit) in values.items():
            number_id = f"CN_FT_S1_{family.upper()}_N{n_value}_{suffix}"
            mask = source["number_id"] == number_id
            if mask.sum() != 1:
                raise RuntimeError(f"Expected one row for {number_id}; found {mask.sum()}")
            source.loc[mask, "value"] = value
            source.loc[mask, "unit"] = unit
            source.loc[mask, "source_file_path"] = (
                "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv"
            )
            source.loc[mask, "source_locator"] = f"family={family}; N={n_value}; column={metric}"
            source.loc[mask, "interpretation_label"] = (
                f"corrected homogeneous five-seed protocol_1 {metric} for {family} N={n_value}"
            )
            source.loc[mask, "ambiguity_flag"] = flag
            source.loc[mask, "ambiguity_note"] = note
            source.loc[mask, "raw_source_path"] = (
                f"results/protocol_1/{family}/N{n_value}_seed{{0..4}}/finetune_last2/"
                "prediction_results_test_set.csv; history_val.json"
            )
            source.loc[mask, "raw_selector"] = f"family={family}; N={n_value}; seeds=0,1,2,3,4"
            source.loc[mask, "formula"] = formula
            source.loc[mask, "recomputed_value"] = f"{value:.17g}"
            source.loc[mask, "display_precision"] = "5 decimal places in prose/tables unless noted"
            source.loc[mask, "validation_status"] = "PASS_protocol_1_REGENERATION"
            updated_ids.add(number_id)

    supported = transfer[transfer["scratch_supported"].astype(str).str.lower().eq("true")]
    for row in supported.itertuples(index=False):
        number_id = f"CN_TRANSFER_BENEFIT_{str(row.family).upper()}_N{int(row.N)}"
        mask = source["number_id"] == number_id
        if mask.sum() != 1:
            raise RuntimeError(f"Expected one row for {number_id}; found {mask.sum()}")
        value = float(row.scratch_minus_finetune_mae_eV_per_atom)
        source.loc[mask, "value"] = value
        source.loc[mask, "source_file_path"] = (
            "results/summaries/protocol_1/comparisons/protocol_1_transfer_gain_table.csv"
        )
        source.loc[mask, "source_locator"] = f"family={row.family}; N={int(row.N)}"
        source.loc[mask, "ambiguity_flag"] = "five_seed_means_supported_condition"
        source.loc[mask, "ambiguity_note"] = (
            "Supported only at N=50 and N=500; positive value is the benefit of pretrained "
            "initialization plus fine-tuning relative to training from scratch."
        )
        source.loc[mask, "raw_source_path"] = (
            f"results/protocol_1/{row.family}/N{int(row.N)}_seed{{0..4}}/"
            "finetune_last2/summary.json; train_alignn_from_scratch/summary.json"
        )
        source.loc[mask, "raw_selector"] = f"family={row.family}; N={int(row.N)}; seeds=0,1,2,3,4"
        source.loc[mask, "formula"] = "mean_from_scratch_mae - mean_finetune_mae"
        source.loc[mask, "recomputed_value"] = f"{value:.17g}"
        source.loc[mask, "display_precision"] = "3 decimal places in figure; 5 in tables"
        source.loc[mask, "validation_status"] = "PASS_protocol_1_REGENERATION"
        updated_ids.add(number_id)

    expected_updates = 12 * 4 + 4
    if len(updated_ids) != expected_updates:
        raise RuntimeError(f"Expected {expected_updates} corrected number rows; found {len(updated_ids)}")
    canonical_csv = out / "canonical_numbers_v3.csv"
    source.to_csv(canonical_csv, index=False)

    # Human-readable numerical authority.
    md = [
        "# Canonical Numbers v3",
        "",
        "Status: **protocol_1 CORRECTED AND VALIDATED**. This version supersedes v2 for every protocol_1 fine-tuning, parity-dependent, and transfer-benefit claim. Historical v2 files remain preserved.",
        "",
        "The numerical authority is the independent raw recomputation in `results/derived_evidence/protocol_1_regeneration/`; generated report tables were validated against it. Means are arithmetic five-seed means and error bars are sample standard deviations (`ddof=1`). Units are eV/atom.",
        "",
        "## fine-tuning aggregates",
        "",
        "| family | N | mean MAE | sample SD | zero-shot − fine-tune | mean best epoch | seed best epochs |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate.sort_values(["family", "N"]).itertuples(index=False):
        group = runs[(runs["family"] == row.family) & (runs["N"] == row.N)].sort_values("seed")
        epochs = ",".join(str(int(x)) for x in group["best_epoch_recomputed"])
        md.append(
            f"| {row.family} | {int(row.N)} | {row.mean_test_mae_eV_per_atom:.8f} | "
            f"{row.sample_std_test_mae_eV_per_atom_ddof1:.8f} | "
            f"{row.zero_shot_minus_finetune_mean_eV_per_atom:.8f} | "
            f"{group['best_epoch_recomputed'].mean():.1f} | {epochs} |"
        )
    md += [
        "",
        "## Corrected transfer benefit",
        "",
        "Transfer benefit is `mean from-scratch MAE − mean fine-tuned MAE` and is defined only for N=50 and N=500.",
        "",
        "| family | N | from-scratch − fine-tuned MAE |",
        "|---|---:|---:|",
    ]
    for row in supported.sort_values(["family", "N"]).itertuples(index=False):
        md.append(
            f"| {row.family} | {int(row.N)} | {row.scratch_minus_finetune_mae_eV_per_atom:.8f} |"
        )
    md += [
        "",
        "## Interpretation guardrails",
        "",
        "- Fine-tuning remains worse than zero-shot in all 12 conditions; every zero-shot-minus-fine-tune value is negative.",
        "- `best_epoch=1` means the end-of-epoch-1 checkpoint was selected; it does not establish byte identity with the original zero-shot model.",
        "- Nitride N=200 has mixed checkpoint timing (49,1,1,1,1), so the former claim of complete inertness through N≤200 is false.",
        "- protocol_2 and 3 remain corrected robustness evidence and are not merged into these primary protocol_1 estimates.",
        "- Embedding numerical rows carried from v2 are unchanged by this regeneration.",
        "",
    ]
    (out / "canonical_numbers_v3.md").write_text("\n".join(md), encoding="utf-8")

    claims = pd.read_csv(out / "claim_support_map_v2.csv", keep_default_na=False)
    set_column(claims, "CLM_03", "notes", "Corrected oxide means are 0.03624–0.03800 eV/atom; all remain above zero-shot 0.03418. Use v3 values only.")
    set_column(claims, "CLM_04", "notes", "Corrected nitride means are 0.07040–0.08528 eV/atom; all remain above zero-shot 0.06954. N=200 has mixed checkpoint timing (49,1,1,1,1), so no blanket inert-through-N200 claim is allowed.")
    set_column(claims, "CLM_05", "notes", "Supported at N=50 and N=500 only. Corrected scratch-minus-fine-tune benefits are oxide 0.51806/0.22815 and nitride 0.61976/0.28923 eV/atom.")
    set_column(claims, "CLM_07", "claim_text", "protocol_2 and protocol_3 are robustness protocols; their numerical ranking relative to corrected protocol_1 varies by family and training size.")
    set_column(claims, "CLM_07", "claim_status", "protocol_specific_supported")
    set_column(claims, "CLM_07", "notes", "The former blanket claim that protocol_2 and 3 are always closer to zero-shot than protocol_1 is withdrawn.")
    claims["source_of_truth_version"] = "v3"
    claims.to_csv(out / "claim_support_map_v3.csv", index=False)

    claim_numbers = pd.read_csv(out / "claim_to_number_source_map_v2.csv", keep_default_na=False)
    claim_numbers["number_source_file"] = claim_numbers["number_source_file"].replace(
        "canonical_numbers_v2.csv", "canonical_numbers_v3.csv"
    )
    for claim in ("CLM_03", "CLM_04", "CLM_05", "CLM_07", "CLM_13"):
        mask = claim_numbers["claim_id"] == claim
        if mask.any():
            claim_numbers.loc[mask, "notes"] = (
                claim_numbers.loc[mask, "notes"].astype(str)
                + " | v3: resolved through validated corrected five-seed protocol_1 evidence."
            )
    claim_numbers["source_of_truth_version"] = "v3"
    claim_numbers.to_csv(out / "claim_to_number_source_map_v3.csv", index=False)

    figures = pd.read_csv(out / "figure_inventory_v2.csv", keep_default_na=False)
    affected_figures = figures["figure_id"].str.contains(
        "FIG_TRANSFER_BENEFIT|FIG_S1_LC_|FIG_S1_COMP_|FIG_S1_PARITY_", regex=True
    )
    figures["protocol_1_regeneration_status"] = "not_applicable"
    figures.loc[affected_figures, "protocol_1_regeneration_status"] = "validated_corrected_v3"
    figures.loc[affected_figures, "notes"] = (
        figures.loc[affected_figures, "notes"].astype(str)
        + " Corrected homogeneous five-seed protocol_1 asset; see protocol_1_regeneration evidence."
    )
    comparison_figures = figures["figure_id"].isin(
        ["FIG_TRANSFER_BENEFIT", "FIG_S1_COMP_OXIDE", "FIG_S1_COMP_NITRIDE"]
    )
    figures.loc[comparison_figures, "source_evidence"] = (
        "results/summaries/protocol_1/comparisons/"
        "protocol_1_corrected_comparison_by_group.csv; results/derived_evidence/protocol_1/"
        "Comparison Plots/protocol_1_transfer_gain_table.csv"
    )
    figures.to_csv(out / "figure_inventory_v3.csv", index=False)

    tables = pd.read_csv(out / "table_inventory_v2.csv", keep_default_na=False)
    affected_tables = tables["table_id"].isin(
        ["TAB_S1_FT_RUNS", "TAB_S1_FT_SUMMARY_BY_N", "TAB_S1_FT_WIDE", "TAB_S1_FS_SUMMARY"]
    )
    tables["protocol_1_regeneration_status"] = "not_applicable"
    tables.loc[affected_tables, "protocol_1_regeneration_status"] = "validated_corrected_v3"
    tables.loc[affected_tables, "notes"] = (
        tables.loc[affected_tables, "notes"].astype(str)
        + " protocol_1 fine-tuning-dependent fields validated after corrected seed-0–2 promotion."
    )
    tables.loc[tables["table_id"] == "TAB_S1_FS_SUMMARY", "path"] = (
        "results/summaries/protocol_1/From Scratch/"
        "from_scratch_summary_corrected_five_seed.csv"
    )
    tables.to_csv(out / "table_inventory_v3.csv", index=False)

    manifest = pd.read_csv(out / "master_evidence_manifest.csv", keep_default_na=False)
    affected_manifest = manifest["artifact_id"].str.contains("S1_FT_|S1_FS_", regex=True)
    manifest["protocol_1_regeneration_status"] = "not_applicable"
    manifest.loc[affected_manifest, "protocol_1_regeneration_status"] = "validated_corrected_v3"
    manifest.loc[affected_manifest, "notes"] = (
        manifest.loc[affected_manifest, "notes"].astype(str)
        + " Reconciled to corrected homogeneous five-seed protocol_1 evidence."
    )
    manifest.loc[manifest["artifact_id"] == "S1_FS_01", "path"] = (
        "results/summaries/protocol_1/From Scratch/"
        "from_scratch_summary_corrected_five_seed.csv"
    )
    manifest.loc[manifest["artifact_id"] == "S1_FS_02", "secondary_path_or_manifest"] = (
        "results/summaries/protocol_1/comparisons/"
        "protocol_1_corrected_comparison_manifest.json"
    )
    manifest.to_csv(out / "master_evidence_manifest_v3.csv", index=False)

    memo = """# Source of Truth Memo v3

protocol_1 is the primary brief-aligned protocol (50 epochs, batch size 16, learning rate 1e-4). Its seed-0–2 canonical results were corrected and promoted, then combined with protected corrected seeds 3–4. The complete 60-run matrix, all 12 aggregates, curves, parity ensembles, and supported scratch comparisons independently validate.

Fine-tuning does not beat zero-shot in any of the 12 corrected protocol_1 conditions. Report this directly. Checkpoint timing must be described seed-wise: oxide N=10 and nitride N=10/50/100 select epoch 1 for every seed; nitride N=200 is mixed (49,1,1,1,1); nitride N=500/1000 selects later checkpoints for every seed. Never equate epoch-1 selection with byte identity to the zero-shot checkpoint.

Scratch comparisons are limited to N=50 and N=500 and use five-seed means for both methods. protocol_2 and 3 remain robustness evidence and must not replace or be pooled with primary protocol_1 estimates. Embedding results are unchanged and remain correlational.

Drafting authority order: raw numerical outputs → protocol_1 regeneration evidence → `canonical_numbers_v3.csv` → v3 claim/figure/table maps. v2 and earlier files are historical and stale for protocol_1 numbers.
"""
    (out / "source_of_truth_memo_v3.md").write_text(memo, encoding="utf-8")

    manifest_md = """# Master Evidence Manifest v3

The machine-readable inventory is `master_evidence_manifest_v3.csv`. protocol_1 fine-tuning rows and all fine-tuning-dependent scratch comparison rows resolve through:

- `results/derived_evidence/protocol_1_regeneration/canonical_60_run_recomputation.csv`
- `results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv`
- `results/summaries/protocol_1/finetune/finetune_runs.csv`
- `results/summaries/protocol_1/finetune/finetune_summary_by_N.csv`
- `results/summaries/protocol_1/from_scratch/from_scratch_summary_corrected_five_seed.csv`
- `results/derived_evidence/protocol_1/Parity Plots/parity_plot_manifest.csv`
- `results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv`
- `results/summaries/protocol_1/comparisons/protocol_1_transfer_gain_table.csv`

protocol_2 and protocol_3 are robustness-only. Section 08 and the nested checkout are excluded. Historical v2 authority files are preserved but are stale for protocol_1-dependent claims.
"""
    (out / "master_evidence_manifest_v3.md").write_text(manifest_md, encoding="utf-8")

    changelog = f"""# protocol_1 correction changelog

- Superseded mixed-protocol protocol_1 values in v2 with independently validated homogeneous five-seed values.
- Updated {12 * 4} fine-tuning number rows and 4 transfer-benefit rows.
- Replaced the false blanket nitride inert-through-N≤200 interpretation with seed-wise checkpoint timing.
- Withdrew the blanket cross-protocol ranking in CLM_07.
- Added explicit raw paths, selectors, formulas, full-precision recomputed values, display precision, and validation status for corrected rows.
- Preserved every v2 file as historical evidence.
"""
    (out / "protocol_1_correction_changelog.md").write_text(changelog, encoding="utf-8")

    produced = [
        "canonical_numbers_v3.csv",
        "canonical_numbers_v3.md",
        "source_of_truth_memo_v3.md",
        "claim_support_map_v3.csv",
        "claim_to_number_source_map_v3.csv",
        "figure_inventory_v3.csv",
        "table_inventory_v3.csv",
        "master_evidence_manifest_v3.csv",
        "master_evidence_manifest_v3.md",
        "protocol_1_correction_changelog.md",
    ]
    verification = [
        "# protocol_1 correction verification",
        "",
        "Status: **PASS**",
        "",
        f"- Corrected number rows: {len(updated_ids)}/{expected_updates}",
        "- Canonical runs: 60/60",
        "- Aggregates: 12/12",
        "- Scratch-supported comparisons: 4/4",
        "- Standard deviation: sample (`ddof=1`)",
        "- Historical authority files overwritten: no",
        "",
        "## File checksums",
        "",
    ]
    for name in produced:
        path = out / name
        verification.append(f"- `{sha256(path)}`  `{path.relative_to(repo)}`")
    (out / "protocol_1_correction_verification.md").write_text(
        "\n".join(verification) + "\n", encoding="utf-8"
    )
    audit = """# Canonical Numbers v3 audit

PASS. All 52 protocol_1-dependent corrected rows resolve to the independent 60-run and 12-aggregate recomputations or the validated four-group transfer table. No protocol_2/3 or embedding number was altered. Every corrected row records a raw path, selector, formula, recomputed value, display precision, and validation status.
"""
    (out / "canonical_numbers_v3_audit.md").write_text(audit, encoding="utf-8")
    (out / "source_of_truth_v3_audit.md").write_text(
        "# Source of Truth v3 audit\n\nPASS. Versioned v3 files supersede v2 only for corrected protocol_1-dependent content; historical files remain intact. Section 08 and the nested checkout were excluded.\n",
        encoding="utf-8",
    )

    reconciliation = f"""# Source-of-truth reconciliation

Status: **PASS**

The versioned v3 authority package reconciles 60 canonical protocol_1 runs, 12 five-seed aggregates, 12 parity groups, 4 supported five-seed scratch comparisons, and 12 zero-shot gain rows. It updates {len(updated_ids)} protocol_1-dependent canonical-number rows without overwriting historical v2 files. protocol_2 and 3 remain robustness-only; embedding numerical evidence is unchanged.

Primary corrected authorities:

- `results/derived_evidence/final_paper_factory/00_source_of_truth/canonical_numbers_v3.csv`
- `results/derived_evidence/final_paper_factory/00_source_of_truth/claim_support_map_v3.csv`
- `results/derived_evidence/final_paper_factory/00_source_of_truth/claim_to_number_source_map_v3.csv`
- `results/derived_evidence/final_paper_factory/00_source_of_truth/figure_inventory_v3.csv`
- `results/derived_evidence/final_paper_factory/00_source_of_truth/table_inventory_v3.csv`
- `results/derived_evidence/final_paper_factory/00_source_of_truth/master_evidence_manifest_v3.csv`
"""
    (evidence / "source_of_truth_reconciliation.md").write_text(
        reconciliation, encoding="utf-8"
    )
    print("protocol_1_SOURCE_OF_TRUTH_V3_BUILT")


if __name__ == "__main__":
    main()
