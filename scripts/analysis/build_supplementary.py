#!/usr/bin/env python3
"""Historical deterministic transformation used for the released supplement.

The one-time transformation consumes a frozen baseline retained in the full-history
archive, so it is preserved here as provenance rather than a clean-checkout entry
point. Regenerate public grids with
``scripts/figures/regenerate_supplementary_grids.py`` and validate the released
package with ``scripts/analysis/validate_supplementary.py``.
"""
import csv
import pathlib
import re

BASE = pathlib.Path("paper/supplementary")
SRC = BASE / "baseline" / "domain_shift_supplementary_materials.baseline.md"
OUT = BASE / "domain_shift_supplementary_materials.md"
DATA = BASE / "data"

raw = SRC.read_text(encoding="utf-8")

# --- split baseline into top-level (## ) sections --------------------------
parts = re.split(r"(?m)^## ", raw)
secs = {}
for p in parts[1:]:
    nl = p.index("\n")
    secs[p[:nl].strip()] = p[nl + 1:]


def get(prefix):
    for h, b in secs.items():
        if h.startswith(prefix):
            return b
    raise KeyError(prefix)


# original bodies (verbatim)
b_old_s1 = get("S1.")   # per-seed 240        -> new S5
b_old_s2 = get("S2.")   # grids               -> new S6
b_old_s3 = get("S3.")   # protocol_2/3 robustness   -> new S9
b_old_s4 = get("S4.")   # embedding/distance  -> new S11
b_old_s5 = get("S5.")   # pure-oxide          -> new S13
b_old_s6 = get("S6.")   # duplicate/split     -> new S1
b_old_s7 = get("S7.")   # provenance          -> new S2
b_old_s8 = get("S8.")   # distance-error      -> new S12
trace_conv = secs["Traceability convention"].rstrip()


def renum(body, old, new):
    """Remap '### S{old}.x' subsection headers to '### S{new}.x' (scoped)."""
    return re.sub(r"(?m)^### S%s\." % re.escape(old), "### S%s." % new, body)


b_new_s5 = renum(b_old_s1, "1", "5").strip()
b_new_s6 = b_old_s2.strip()                        # no subsections
b_new_s9 = renum(b_old_s3, "3", "9").strip()
b_new_s11 = renum(b_old_s4, "4", "11").strip()
b_new_s13 = renum(b_old_s5, "5", "13").strip()
b_new_s1 = renum(b_old_s6, "6", "1").strip()
b_new_s2 = b_old_s7.strip()                        # no subsections
b_new_s12 = b_old_s8.strip()                        # no subsections


# --- promoted-analysis table renderers (values straight from imported CSV) --
def read(name):
    with open(DATA / name, newline="") as f:
        return list(csv.DictReader(f))


def mdtable(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "| " + " | ".join(["---"] * len(headers)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


FAM = {"oxide": "Oxide", "nitride": "Nitride"}


def a1_table():
    rows = read("a1_protocol_1_effect_sizes.csv")
    rows.sort(key=lambda r: (r["family"] != "oxide", int(r["N"])))
    body = []
    for r in rows:
        N = "1,000" if r["N"] == "1000" else r["N"]
        body.append([FAM[r["family"]], N, r["mean_ft_mae_eV_per_atom"],
                     r["zero_shot_mae_eV_per_atom"],
                     "+" + r["delta_adapt_eV_per_atom"], r["r_adapt_percent"],
                     r["seeds_worse_than_zs"],
                     "A1-%s-%s" % (r["family"][:2].upper(), r["N"])])
    return mdtable(["Family", "N", "Fine-tuned MAE (mean)", "Zero-shot MAE",
                    "Δadapt (eV/atom)", "Radapt (%)", "Seeds worse than ZS",
                    "Trace"], body)


def a2_tables():
    r = {row["family"]: row for row in read("a2_residual_decomposition.csv")}
    dist = mdtable(
        ["Family", "n", "Signed mean error", "Median AE", "RMSE",
         "p90 AE", "p95 AE", "Worst-decile n", "Worst-decile AE share", "Trace"],
        [[FAM[f], r[f]["n"], r[f]["signed_mean_error_eV_per_atom"],
          r[f]["median_ae"], r[f]["rmse"], r[f]["p90_ae"], r[f]["p95_ae"],
          r[f]["n_top_decile"], r[f]["worst_decile_ae_share"],
          "A2-%s-DIST" % f[:2].upper()] for f in ("oxide", "nitride")])
    quart = mdtable(
        ["Family", "Q1 upper bound", "Q2 upper bound", "Q3 upper bound",
         "MAE Q1", "MAE Q2", "MAE Q3", "MAE Q4", "Trace"],
        [[FAM[f], r[f]["target_q1_bound"], r[f]["target_q2_bound"],
          r[f]["target_q3_bound"], r[f]["mae_target_q1"], r[f]["mae_target_q2"],
          r[f]["mae_target_q3"], r[f]["mae_target_q4"],
          "A2-%s-QUART" % f[:2].upper()] for f in ("oxide", "nitride")])
    return dist, quart


def a3_tables():
    het = read("a3_structure_heterogeneity.csv")
    het.sort(key=lambda r: (r["family"] != "oxide", int(r["N"])))
    het_t = mdtable(
        ["Family", "N", "Per-seed frac. improved (mean)",
         "Per-seed frac. improved (SD)", "Seed-mean-AE frac. improved", "Trace"],
        [[FAM[r["family"]], "1,000" if r["N"] == "1000" else r["N"],
          r["frac_improved_perseed_mean"], r["frac_improved_perseed_sd"],
          r["frac_improved_seedmean_ae"],
          "A3-%s-%s" % (r["family"][:2].upper(), r["N"])] for r in het])
    ter = read("a3_tertile_profile.csv")
    ter.sort(key=lambda r: (r["family"] != "oxide", int(r["N"]),
                            int(r["zs_ae_tertile"])))
    ter_t = mdtable(
        ["Family", "N", "ZS-AE tertile", "n structures",
         "Per-seed frac. improved (mean)", "Mean ΔAE (FT − ZS, eV/atom)", "Trace"],
        [[FAM[r["family"]], "1,000" if r["N"] == "1000" else r["N"],
          r["zs_ae_tertile"], r["n_structures"],
          r["frac_improved_perseed_mean"], r["mean_delta_ae_ft_minus_zs"],
          "A3-%s-%s-T%s" % (r["family"][:2].upper(), r["N"], r["zs_ae_tertile"])]
         for r in ter])
    return het_t, ter_t


def a4_table():
    rows = read("a4_learning_dynamics.csv")
    rows.sort(key=lambda r: (r["family"] != "oxide", int(r["N"])))
    body = [[FAM[r["family"]], "1,000" if r["N"] == "1000" else r["N"],
             r["median_rel_epoch1_to_best_val_improvement"],
             r["median_final_minus_best_val_l1"],
             r["boundary_selections_epoch50_of5"], r["best_epoch_sd"],
             "A4-%s-%s" % (r["family"][:2].upper(), r["N"])] for r in rows]
    return mdtable(
        ["Family", "N", "Median rel. epoch-1→best val improvement",
         "Median (final − best) val L1", "Boundary(50) selections (of 5)",
         "Best-epoch SD", "Trace"], body)


def a5_table():
    rows = read("a5_cross_protocol_effects.csv")
    rows.sort(key=lambda r: (int(r["set"][-1]), r["family"] != "oxide",
                             int(r["N"])))
    body = [[r["set"], FAM[r["family"]],
             "1,000" if r["N"] == "1000" else r["N"],
             r["mean_ft_mae_eV_per_atom"], r["zero_shot_mae_eV_per_atom"],
             "+" + r["delta_adapt_eV_per_atom"], r["r_adapt_percent"],
             "A5-%s-%s-%s" % (r["set"][-1], r["family"][:2].upper(), r["N"])]
            for r in rows]
    return mdtable(
        ["Set", "Family", "N", "Fine-tuned MAE (mean)", "Zero-shot MAE",
         "Δadapt (eV/atom)", "Radapt (%)", "Trace"], body)


a1 = a1_table()
a2_dist, a2_quart = a2_tables()
a3_het, a3_ter = a3_tables()
a4 = a4_table()
a5 = a5_table()

# --- assemble the restructured document ------------------------------------
DOC = f"""# Supplementary Materials

## Is Fine-Tuning Worth It? Reusing a Pretrained AI Model to Predict the Stability of New Materials

Faizan Ahmed; Muhammad Ali Bin Sarwar; Dr. Burhan Saifaddin (corresponding author)

This document is the governed supplement accompanying the manuscript. It is organized to follow the manuscript's own progression, so that each reported result can be traced to its full backing here: data and checkpoint provenance behind the Methods, then effect-size definitions, then the results arc — the zero-shot residual structure, the protocol_1 adaptation surface and its per-seed and structure-level detail, cross-protocol robustness, and finally frozen-representation geometry and the distance–error association. It is generated only from canonical tracked results and validated evidence packages, and it adds the deepened analyses reported in the accompanying analysis (adaptation effect sizes, residual decomposition, structure-level heterogeneity, learning dynamics, and cross-protocol effect magnitudes) as fully tabulated sections. The atomic traceability table is `../evidence_manifest.csv`; full machine-readable tables are under `data/`. protocol_2 and 3 are robustness evidence and are never pooled with canonical protocol_1. Paths embedded inside historical run summaries are not followed. Any path matching the permanently forbidden Section 08 pattern is excluded from this and future paper inputs.

### Map from the manuscript to this supplement

| Manuscript element | Supplement section(s) |
| --- | --- |
| Methods — data, families, provenance (Section 3.1, Table 1) | S1, S2 |
| Methods — adaptation effect-size definitions (Section 3.2) | S3 |
| Results — zero-shot gap and residual structure (Section 4.1) | S4 |
| Results — protocol_1 adaptation surface, effect sizes, Table 3 (Section 4.2) | S3, S5, S6 |
| Results — structure-level heterogeneity (Section 4.2) | S7 |
| Results — learning dynamics (Section 4.2) | S8 |
| Results — value of pretrained initialization, Table 4 (Section 4.3) | S5 |
| Results — cross-protocol robustness and effect ranges (Section 4.4) | S9, S10 |
| Results — representation geometry and distance–error (Section 4.5) | S11, S12, S13 |

## Traceability convention

{trace_conv}

## S1. Dataset composition, duplicate resolution, and split audit

This section documents the dataset construction that underlies the manuscript's Methods (Section 3.1) and Table 1: how the source catalog was deduplicated and how the global train/validation/test split was formed, together with the family definitions and the canonical-run integrity checks. It backs the membership and integrity claims on which every downstream result depends.

{b_new_s1}

## S2. Checkpoint provenance and exact-overlap methodology

This section gives the provenance of the evaluated checkpoint (file hashes and the official-archive match) and the exact identifier-overlap methodology behind the membership-verification statements in the manuscript's Methods (Section 3.1). Zero exact overlap prevents direct training-item leakage but is not interpreted as a statistical distribution label.

{b_new_s2}

## S3. Adaptation effect sizes and seed consistency

This section provides the full effect-size backing for the manuscript's Section 3.2 definitions and the restructured Table 3 in Section 4.2. For each protocol_1 condition we report the absolute adaptation change Δadapt = MAE_FT − MAE_ZS, where a positive value means the tested fine-tuning increased the fixed-test MAE (was worse), and its relative form Radapt = 100 · Δadapt / MAE_ZS. MAE_FT is the five-seed mean fine-tuned MAE (reducible from S5); MAE_ZS is the family zero-shot fixed-test MAE. The final column counts, of the five seeds, how many finished above the family zero-shot baseline. Across the 12 protocol_1 conditions every seed in every condition finished worse than its family zero-shot baseline — 60 seed-level runs in total — so the non-improving adaptation surface holds at the seed level and not only in the condition means.

{a1}

## S4. Zero-shot residual-distribution decomposition

This section decomposes the zero-shot error distribution behind the manuscript's Section 4.1, showing that the chemical-family gap is heavy-tailed rather than a uniform inflation. The first table reports, per family, the signed mean error (bias), the median absolute error, the RMSE, the 90th and 95th absolute-error percentiles, and the number and total-error share carried by the worst decile of structures. The second table reports the mean absolute error within quartiles of the target formation-energy distribution, with the quartile upper bounds; difficulty concentrates in the highest-formation-energy quartile in both families.

{a2_dist}

Absolute-error concentration by target formation-energy quartile (bounds in eV/atom; MAE in eV/atom):

{a2_quart}

## S5. Per-seed results for all 240 repository training runs

This section lists every per-seed run behind the manuscript's Section 4.2 protocol_1 curves (Table 3) and the Section 4.3 from-scratch comparison (Table 4). All aggregate means and sample standard deviations reported in the manuscript, and the effect sizes in S3, reduce from these rows.

{b_new_s5}

## S6. Full training-curve and parity grids

This section provides the complete per-seed training-curve and parity grids underlying the Section 4.2 learning curves. The validation-selected epoch marked in each training panel is the basis for the checkpoint-timing discussion summarized in S8.

{b_new_s6}

## S7. Structure-level heterogeneity and tertile profile

This section backs the manuscript's Section 4.2 statement that aggregate degradation conceals substantial structure-level heterogeneity. The first table reports, at the engaged budgets (N = 500 and N = 1,000), the fraction of fixed-test structures whose absolute error improved under fine-tuning, computed per seed (mean and SD across seeds) and from the seed-mean prediction. The second table profiles those improvements within tertiles of zero-shot absolute error, together with the mean signed change in absolute error (fine-tuned minus zero-shot). The tertile contrast is descriptive and partly reflects regression to the mean; it should not be read as evidence that fine-tuning targets difficult structures.

{a3_het}

Improvement and mean absolute-error change within zero-shot-error tertiles (tertile 1 = easiest, tertile 3 = hardest):

{a3_ter}

## S8. Learning-dynamics summary

This section backs the manuscript's Section 4.2 learning-dynamics statements. For each condition it reports the median relative validation improvement from the end of epoch 1 to the validation-selected epoch, the median gap between the final-epoch and best validation L1 (a check that training did not simply overfit the tiny validation set), the number of seeds (of five) that selected the last available epoch (the schedule boundary), and the across-seed standard deviation of the selected epoch. Non-zero relative improvement and wide selected-epoch dispersion indicate genuine optimizer engagement even where the fixed-test error did not improve.

{a4}

## S9. Complete protocol_2 and protocol_3 robustness tables and curves

This section reports the complete protocol_2 and protocol_3 robustness surfaces behind the manuscript's Section 4.4 cross-protocol result. protocol_2 and 3 vary only the optimization schedule; they are protocol-sensitivity evidence and are never pooled with protocol_1.

{b_new_s9}

## S10. Cross-protocol effect magnitudes

This section quantifies the manuscript's Section 4.4 "36 of 36" result with effect sizes rather than bare means. For every condition across all three protocols it reports Δadapt = MAE_FT − MAE_ZS (positive = worse) and Radapt = 100 · Δadapt / MAE_ZS, using the same definitions as S3. All 36 condition means lie above their family zero-shot baseline; the per-family Δadapt and Radapt ranges quoted in the manuscript are read directly from this table (the protocol_1 rows coincide with S3).

{a5}

## S11. Alternative embedding layers, projection sensitivity, and distance metrics

This section provides the frozen-representation family-separation metrics, the projection-parameter sensitivity, and the full multi-layer, multi-distance grids underlying the manuscript's Section 4.5 representation discussion. All inferential metrics are computed in the raw 256-dimensional space; projection panels are descriptive.

{b_new_s11}

## S12. Canonical distance–error recomputation and three-distance robustness

This section is the canonical distance–error recomputation behind the manuscript's Section 4.5 association result. The three prespecified raw-space distances (five-nearest-oxide, regularized Mahalanobis, and oxide-centroid) substantiate the manuscript's direction-stability statement: each yields a positive Spearman correlation with a positive confidence interval and a corrected q-value, as tabulated below (the same three distances also appear across embedding layers in S11.4). All statements remain correlational and protocol-specific.

{b_new_s12}

## S13. Pure-oxide (oxynitride-excluded) sensitivity

This section backs the manuscript's Section 4.5 pure-oxide robustness note and the Section 5 family-definition discussion. Removing oxynitrides from the oxide reference side leaves the zero-shot ratio and the frozen-embedding support flags qualitatively unchanged; the inclusive analysis remains primary.

{b_new_s13}

## Supplementary data and reproducibility files

The repository URL is https://github.com/TheArchitect999/ALIGNN-domain-shift. This restructured document (sections S1–S13), the 240-run table, the promoted A1–A5 analysis tables, the full numeric embedding, distance, and sensitivity tables, the generated figure manifest, the atomic evidence manifest, the validation report, and the checksums are all stored under `paper/supplementary/`. The released tables are governed by `paper/evidence_manifest.csv`; regenerate the public grids with `scripts/figures/regenerate_supplementary_grids.py` and validate the package with `scripts/analysis/validate_supplementary.py`.
"""

OUT.write_text(DOC, encoding="utf-8")
print("wrote", OUT)
print("length (chars):", len(DOC))
print("sections:", len(re.findall(r"(?m)^## S\d+\. ", DOC)))
