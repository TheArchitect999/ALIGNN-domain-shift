#!/usr/bin/env python3
"""Create corrected protocol_1 analysis packets and sentence-level claim impact map."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


KNOWN_RANGES = {
    "results/derived_evidence/final_paper_factory/03_section_inputs/oxide_results_packet.md": [(40, 53), (63, 66), (77, 78), (149, 157)],
    "results/derived_evidence/final_paper_factory/03_section_inputs/nitride_results_packet.md": [(10, 10), (41, 54), (64, 67), (78, 79), (153, 161), (212, 213)],
    "results/derived_evidence/final_paper_factory/03_section_inputs/joint_comparison_packet.md": [(43, 61), (69, 83), (91, 96), (133, 145)],
    "results/derived_evidence/final_paper_factory/03_section_inputs/combined_paper_results_III_and_IV_draft_v3.md": [(29, 35), (41, 53), (57, 61), (127, 128)],
    "results/derived_evidence/final_paper_factory/04_drafts/analysis_stage_12_full_manuscripts/oxide_polished_v3.md": [(11, 11), (88, 110), (119, 128), (149, 151), (160, 160), (166, 166), (172, 172), (190, 190), (204, 206)],
    "results/derived_evidence/final_paper_factory/04_drafts/analysis_stage_12_full_manuscripts/nitride_polished_v3.md": [(11, 11), (89, 91), (104, 123), (157, 176), (194, 204), (222, 244)],
    "results/derived_evidence/final_paper_factory/04_drafts/analysis_stage_12_full_manuscripts/combined_paper_polished_v4.md": [(11, 11), (100, 136), (155, 157), (168, 201), (235, 254), (283, 309), (378, 400), (416, 440)],
}
TEMPLATE_FILES = (
    "results/derived_evidence/final_paper_factory/06_template_ready/oxide_template_ready_v3.md",
    "results/derived_evidence/final_paper_factory/06_template_ready/nitride_template_ready_v3.md",
    "results/derived_evidence/final_paper_factory/06_template_ready/combined_template_ready_v3.md",
)

OLD_TOKENS = re.compile(
    r"0\.(?:0417|0523|0465|0457|0430|04169|0874|1173|1722|1392|0977|0907|"
    r"0111|0053|0199|0135|5038|2214|5741|2706|0391|0383|0828|0829|1203|1220|"
    r"0179|0478|1027|0697|0281|0211)|"
    r"mean_best_epoch|best epoch|best-epoch|epoch 1|inert|N\s*(?:<=|≤)\s*200|"
    r"adaptation (?:begins|starts|onset)|from.scratch|transfer benefit|parity|protocol_1",
    re.IGNORECASE,
)


def selected_lines(path: Path, ranges: list[tuple[int, int]] | None) -> list[tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = []
    if ranges:
        numbers = {number for start, end in ranges for number in range(start, end + 1)}
        for number in sorted(numbers):
            if number <= len(lines):
                text = lines[number - 1].strip()
                if text and not re.fullmatch(r"\|?[-:| ]+\|?", text):
                    selected.append((number, text))
    else:
        for number, text in enumerate(lines, 1):
            stripped = text.strip()
            if stripped and OLD_TOKENS.search(stripped) and not re.fullmatch(r"\|?[-:| ]+\|?", stripped):
                selected.append((number, stripped))
    return selected


def impact_for(text: str) -> tuple[str, str, str, str]:
    lower = text.lower()
    source = (
        "aggregate_recomputation.csv; parity_validation.json; "
        "protocol_1_corrected_comparison_by_group.csv; protocol_1_transfer_gain_table.csv"
    )
    if "embedding" in lower or "representation" in lower or "mahalanobis" in lower:
        return (
            "Embedding numerical evidence is unchanged; any linkage to protocol_1 error remains correlational.",
            "correlational",
            "Retain only with correlational wording and replace any attached protocol_1 error number from v3 authority.",
            source,
        )
    if "inert" in lower or "n≤200" in lower or "n <= 200" in lower or "n<=200" in lower or "adaptation begins" in lower or "adaptation onset" in lower:
        return (
            "Nitride is all-seed epoch-1 only at N=10,50,100; N=200 is mixed (49,1,1,1,1); N=500/1000 are all-seed later-checkpoint regimes.",
            "unsupported",
            "Withdraw blanket inert-through-N200/onset-at-N500 wording; replace with seed-wise checkpoint timing.",
            source,
        )
    if "epoch 1" in lower or "best epoch" in lower or "mean_best_epoch" in lower or "checkpoint" in lower:
        return (
            "Use corrected seed epoch vectors. Epoch 1 means end-of-epoch-1 selection, not byte identity with zero-shot and not one optimizer step.",
            "protocol-specific",
            "Replace old means/vectors and tighten checkpoint semantics.",
            source,
        )
    if "scratch" in lower or "transfer benefit" in lower:
        return (
            "Five-seed scratch−fine benefits: oxide N50=0.51805641, N500=0.22814823; nitride N50=0.61976317, N500=0.28922690 eV/atom.",
            "protocol-specific",
            "Replace seed-0-derived or old benefit values; restrict comparison to N=50 and N=500.",
            source,
        )
    if "parity" in lower or "r²" in lower or "rmse" in lower:
        return (
            "Use the corrected 12-group five-seed ensemble metrics in parity_validation.json and distinguish ensemble MAE from mean per-seed MAE.",
            "protocol-specific",
            "Replace every old parity MAE/RMSE/R² and any endpoint trend interpretation.",
            source,
        )
    if "zero-shot" in lower or "fine-tun" in lower or "mae" in lower or re.search(r"0\.\d+", lower):
        return (
            "Use canonical_numbers_v3.csv. Fine-tuning remains above zero-shot for all 12 groups; corrected oxide means span 0.03597–0.03800 and nitride means 0.07040–0.08528 eV/atom.",
            "robust",
            "Preserve the broad conclusion only; replace all protocol_1 means, SDs, gaps, extrema, cross-family differences, and trend language.",
            source,
        )
    return (
        "Sentence is protocol_1-dependent and must be rechecked against v3 evidence before reuse.",
        "protocol-specific",
        "Revalidate or rewrite in Section 3; do not reuse v2 wording as authority.",
        source,
    )


def numeric_packet(family: str, aggregate: pd.DataFrame, parity: dict, comparison: pd.DataFrame) -> str:
    subset = aggregate[aggregate["family"] == family].sort_values("N")
    rows = [
        f"# {family.capitalize()} Results Packet v2 — Corrected protocol_1",
        "",
        "This is a versioned numerical/interpretive input for later manuscript rewriting. It does not rewrite the polished manuscript.",
        "",
        "| N | mean MAE | sample SD | zero-shot − fine | best epochs | ensemble parity MAE |",
        "|---:|---:|---:|---:|---|---:|",
    ]
    for row in subset.itertuples(index=False):
        item = next(x for x in parity["rows"] if x["family"] == family and x["N"] == int(row.N))
        rows.append(
            f"| {int(row.N)} | {row.mean_test_mae_eV_per_atom:.8f} | "
            f"{row.sample_std_test_mae_eV_per_atom_ddof1:.8f} | "
            f"{row.zero_shot_minus_finetune_mean_eV_per_atom:.8f} | "
            f"{','.join(map(str, json.loads(row.raw_seed_best_epochs)))} | {item['mae_eV_per_atom']:.8f} |"
        )
    rows += [
        "",
        "## Supported scratch comparisons",
        "",
        "| N | fine mean | scratch mean | scratch − fine |",
        "|---:|---:|---:|---:|",
    ]
    for row in comparison[comparison["family"] == family].sort_values("N").itertuples(index=False):
        rows.append(
            f"| {int(row.N)} | {row.finetune_mean_test_mae_eV_per_atom:.8f} | "
            f"{row.scratch_mean_test_mae_eV_per_atom:.8f} | "
            f"{row.scratch_minus_finetune_mae_eV_per_atom:.8f} |"
        )
    rows += [
        "",
        "## Drafting guardrails",
        "",
        "Fine-tuning remains worse than zero-shot at every N. Means and sample SDs are based on exactly five seeds. Parity ensemble MAE is a different aggregation from mean per-seed MAE. Scratch conclusions are limited to N=50 and N=500. Checkpoint claims must use the raw seed vector, and epoch-1 selection must not be called byte-identical to zero-shot.",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    evidence = repo / "results/derived_evidence/protocol_1_regeneration"
    section = repo / "results/derived_evidence/final_paper_factory/03_section_inputs"
    aggregate = pd.read_csv(evidence / "aggregate_recomputation.csv")
    parity = json.loads((evidence / "parity_validation.json").read_text())
    comparison = pd.read_csv(
        repo / "results/summaries/protocol_1/comparisons/protocol_1_corrected_comparison_by_group.csv"
    )

    impacts = []
    for relative, ranges in KNOWN_RANGES.items():
        path = repo / relative
        for line, text in selected_lines(path, ranges):
            corrected, status, action, source = impact_for(text)
            impacts.append(
                {
                    "file": relative,
                    "section_or_line": f"line {line}",
                    "existing_sentence_or_identifier": text,
                    "old_numerical_claim": text if re.search(r"\d", text) else "qualitative protocol_1 interpretation",
                    "corrected_numerical_value_or_interpretation": corrected,
                    "claim_status": status,
                    "required_action_in_section_3": action,
                    "source_evidence": source,
                    "old_protocol_1_claim_status": "STALE_PENDING_SECTION3_REWRITE",
                }
            )
    for relative in TEMPLATE_FILES:
        path = repo / relative
        for line, text in selected_lines(path, None):
            corrected, status, action, source = impact_for(text)
            impacts.append(
                {
                    "file": relative,
                    "section_or_line": f"line {line}",
                    "existing_sentence_or_identifier": text,
                    "old_numerical_claim": text if re.search(r"\d", text) else "qualitative protocol_1 interpretation",
                    "corrected_numerical_value_or_interpretation": corrected,
                    "claim_status": status,
                    "required_action_in_section_3": action,
                    "source_evidence": source,
                    "old_protocol_1_claim_status": "STALE_PENDING_SECTION3_REWRITE",
                }
            )
    impact_df = pd.DataFrame(impacts).drop_duplicates(
        ["file", "section_or_line", "existing_sentence_or_identifier"]
    )
    impact_df.to_csv(evidence / "manuscript_claim_impact.csv", index=False)

    md = [
        "# Manuscript Claim Impact — Corrected protocol_1",
        "",
        "Status: **AUDIT COMPLETE; MANUSCRIPT REWRITE DEFERRED TO SECTION 3**.",
        "",
        f"This sentence/line-level quarantine map contains {len(impact_df)} affected entries across the four analysis/draft inputs, the three latest polished manuscripts, and the three template-ready mirrors. Every old protocol_1 numerical claim in these files is stale until rewritten from v3 authority. No polished or template-ready manuscript was modified.",
        "",
        "High-impact corrections: nitride is not uniformly inert through N=200; nitride N=200 has best epochs 49,1,1,1,1. All old means, sample SDs, zero-shot gaps, transfer benefits, parity metrics, cross-family gaps, extrema, and stability narratives must be replaced. The broad conclusions that fine-tuning remains worse than zero-shot and that pretrained initialization beats scratch at the four supported conditions survive.",
        "",
        "The complete sentence text and actions are in `manuscript_claim_impact.csv`. Summary by file:",
        "",
        "| file | affected entries | required Section 3 disposition |",
        "|---|---:|---|",
    ]
    for file, group in impact_df.groupby("file", sort=True):
        md.append(f"| `{file}` | {len(group)} | rewrite/revalidate every listed entry from v3 evidence |")
    md += [
        "",
        "## Status taxonomy",
        "",
        "- `robust`: broad conclusion survives, but numerical wording is stale.",
        "- `protocol-specific`: retain only with the corrected protocol_1 protocol and formulas stated.",
        "- `correlational`: embedding/error association must remain non-causal.",
        "- `unsupported`: withdraw the old interpretation.",
        "",
        "## Required Section 3 actions",
        "",
        "1. Rewrite the latest combined, oxide, and nitride manuscripts from `canonical_numbers_v3.csv` and the corrected packets.",
        "2. Synchronize template-ready mirrors only after polished manuscripts validate.",
        "3. Preserve the distinction between ensemble-prediction MAE and mean per-seed MAE.",
        "4. Restrict scratch claims to N=50/500 and never substitute fine-tuning seed 0 for a five-seed mean.",
        "5. Treat protocol_2/3 as protocol-specific robustness evidence and embedding associations as correlational.",
        "",
    ]
    (evidence / "manuscript_claim_impact.md").write_text("\n".join(md), encoding="utf-8")

    (section / "oxide_results_packet_v2_protocol_1_corrected.md").write_text(
        numeric_packet("oxide", aggregate, parity, comparison), encoding="utf-8"
    )
    (section / "nitride_results_packet_v2_protocol_1_corrected.md").write_text(
        numeric_packet("nitride", aggregate, parity, comparison), encoding="utf-8"
    )

    oxide = aggregate[aggregate["family"] == "oxide"].set_index("N")
    nitride = aggregate[aggregate["family"] == "nitride"].set_index("N")
    joint = [
        "# Joint Comparison Packet v2 — Corrected protocol_1",
        "",
        "| N | oxide mean MAE | nitride mean MAE | nitride − oxide |",
        "|---:|---:|---:|---:|",
    ]
    for n_value in (10, 50, 100, 200, 500, 1000):
        o = float(oxide.loc[n_value, "mean_test_mae_eV_per_atom"])
        n = float(nitride.loc[n_value, "mean_test_mae_eV_per_atom"])
        joint.append(f"| {n_value} | {o:.8f} | {n:.8f} | {n-o:.8f} |")
    joint += [
        "",
        "Nitride remains worse than oxide at every paired N; the largest corrected gap occurs at N=200. This is descriptive and protocol-specific, not a causal measure of domain-shift efficiency. Scratch-minus-fine-tune benefit is larger for nitride at N=50 and N=500, but differing family baselines prevent interpreting that as superior transfer efficiency.",
        "",
    ]
    (section / "joint_comparison_packet_v2_protocol_1_corrected.md").write_text(
        "\n".join(joint), encoding="utf-8"
    )
    combined = """# Combined Results III/IV protocol_1 Correction Packet v1

This packet supersedes protocol_1 numbers and interpretations in `combined_paper_results_III_and_IV_draft_v3.md` without rewriting that historical draft. Use the corrected oxide, nitride, and joint packets plus `manuscript_claim_impact.csv` during Section 3.

Required narrative corrections:

- Fine-tuning remains above zero-shot for both families in all 12 conditions.
- The largest corrected nitride-minus-oxide mean-MAE gap occurs at N=200, not N=100.
- Nitride checkpoint timing is all-seed epoch-1 at N=10/50/100, mixed at N=200 (49,1,1,1,1), and all-seed later at N=500/1000.
- Corrected parity endpoints do not support the old high-N-improvement/reproducibility narrative.
- Five-seed scratch-minus-fine-tune benefits are 0.51806 and 0.22815 eV/atom for oxide and 0.61976 and 0.28923 for nitride at N=50 and N=500.
- Embedding evidence is unchanged and remains correlational; it must not be used to causally explain corrected protocol_1 errors.
"""
    (section / "combined_paper_results_III_and_IV_protocol_1_correction_packet_v1.md").write_text(
        combined, encoding="utf-8"
    )
    print(f"protocol_1_CLAIM_IMPACT_BUILT entries={len(impact_df)}")


if __name__ == "__main__":
    main()
