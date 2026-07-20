#!/usr/bin/env python3
"""Validated analysis — analysis_stage_3 independent validator.

Re-derives headline values from the raw sources via a second code path and
compares them against the producer's tables. Stdlib only. Writes
analysis_validation.json; exit 1 on any failure (blocks analysis_stage_4).
"""
import csv
import json
import math
import os
import sys

REPO = "."
OUT = os.path.join(REPO, "results/summaries")
TAB = os.path.join(OUT, "tables")
RESULTS = []


def result(name, ok, detail=""):
    RESULTS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:250]})


def rp(p):
    return os.path.join(REPO, p)


def load(path):
    return list(csv.DictReader(open(path)))


def q_linear(xs, q):
    s = sorted(xs)
    pos = q * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] if lo == hi else s[lo] + (pos - lo) * (s[hi] - s[lo])


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main():
    # ---- A1: full independent recompute --------------------------------
    agg = load(rp("results/derived_evidence/protocol_1_regeneration/"
                  "aggregate_recomputation.csv"))
    a1 = {(r["family"], r["N"]): r for r in load(os.path.join(TAB, "a1_protocol_1_effect_sizes.csv"))}
    ok, worse_total = True, 0
    for r in agg:
        key = (r["family"], r["N"])
        mean_mae = float(r["mean_test_mae_eV_per_atom"])
        zs = float(r["zero_shot_mae_eV_per_atom"])
        delta = mean_mae - zs
        radapt = 100 * delta / zs
        worse = sum(1 for m in json.loads(r["raw_seed_test_mae_eV_per_atom"]) if m > zs)
        worse_total += worse
        t = a1[key]
        ok &= close(t["delta_adapt_eV_per_atom"], round(delta, 5), 1e-8)
        ok &= close(t["r_adapt_percent"], round(radapt, 1), 0.051)
        ok &= t["seeds_worse_than_zs"] == f"{worse}/5"
    result("A1_recompute_12_conditions", ok)
    result("A1_seed_total_60", worse_total == 60, worse_total)

    # ---- A2: nitride headline recompute ---------------------------------
    zs_ni = load(rp("results/zero_shot/nitride/predictions.csv"))
    errs = [float(r["prediction"]) - float(r["target"]) for r in zs_ni]
    aes = [abs(e) for e in errs]
    bias = sum(errs) / len(errs)
    med = q_linear(aes, 0.5)
    p90 = q_linear(aes, 0.90)
    n_top = math.ceil(0.10 * len(aes))
    share = sum(sorted(aes)[-n_top:]) / sum(aes)
    t = next(r for r in load(os.path.join(TAB, "a2_residual_decomposition.csv"))
             if r["family"] == "nitride")
    result("A2_nitride_bias", close(t["signed_mean_error_eV_per_atom"], round(bias, 5), 1e-8), bias)
    result("A2_nitride_median", close(t["median_ae"], round(med, 5), 1e-8), med)
    result("A2_nitride_p90", close(t["p90_ae"], round(p90, 5), 1e-8), p90)
    result("A2_nitride_topdecile", close(t["worst_decile_ae_share"], round(share, 4), 1e-8), share)
    result("A2_counts", len(zs_ni) == 242, len(zs_ni))

    # ---- A3: nitride N=1000 primary fraction recompute ------------------
    zsd = {r["jid"]: (float(r["target"]), float(r["abs_error"])) for r in zs_ni}
    fracs = []
    join_ok = True
    for s in range(5):
        preds = load(rp(f"results/protocol_1/nitride/N1000_seed{s}/"
                        "finetune_last2/prediction_results_test_set.csv"))
        join_ok &= {r["id"] for r in preds} == set(zsd)
        join_ok &= all(abs(float(r["target"]) - zsd[r["id"]][0]) <= 1e-4 for r in preds)
        n_imp = sum(1 for r in preds
                    if abs(float(r["prediction"]) - float(r["target"])) < zsd[r["id"]][1])
        fracs.append(n_imp / len(preds))
    mean_frac = sum(fracs) / len(fracs)
    t = next(r for r in load(os.path.join(TAB, "a3_structure_heterogeneity.csv"))
             if r["family"] == "nitride" and r["N"] == "1000")
    result("A3_join_nitride_N1000", join_ok)
    result("A3_nitride_N1000_frac",
           close(t["frac_improved_perseed_mean"], round(mean_frac, 4), 1e-8), mean_frac)

    # ---- A4: oxide N=50 recompute ---------------------------------------
    rels, boundary = [], 0
    for s in range(5):
        base = rp(f"results/protocol_1/oxide/N50_seed{s}/finetune_last2")
        h = json.load(open(base + "/history_val.json"))
        v = [row[0] for row in h]
        be = int(json.load(open(base + "/summary.json"))["best_epoch"])
        boundary += be == 50
        rels.append((v[0] - min(v)) / v[0])
    med_rel = q_linear(rels, 0.5)
    t = next(r for r in load(os.path.join(TAB, "a4_learning_dynamics.csv"))
             if r["family"] == "oxide" and r["N"] == "50")
    result("A4_oxide_N50_medrel",
           close(t["median_rel_epoch1_to_best_val_improvement"], round(med_rel, 4), 1e-8), med_rel)
    result("A4_oxide_N50_boundary", t["boundary_selections_epoch50_of5"] == str(boundary))

    # ---- A5: all 36 deltas recompute ------------------------------------
    a5 = load(os.path.join(TAB, "a5_cross_protocol_effects.csv"))
    result("A5_row_count", len(a5) == 36, len(a5))
    # tolerance 1.05e-5: the table's delta is rounded from full-precision
    # sources, while this recompute differences two already-rounded columns
    ok = all(close(r["delta_adapt_eV_per_atom"],
                   round(float(r["mean_ft_mae_eV_per_atom"])
                         - float(r["zero_shot_mae_eV_per_atom"]), 5), 1.05e-5) for r in a5)
    result("A5_internal_consistency_within_rounding", ok)
    src_ok = True
    for k in (2, 3):
        src = load(rp(f"results/summaries/protocol_{k}/finetune/"
                      "finetune_summary_by_N.csv"))
        for r in src:
            row = next(x for x in a5 if x["set"] == f"Set{k}"
                       and x["family"] == r["family"] and x["N"] == r["N"])
            src_ok &= close(row["mean_ft_mae_eV_per_atom"],
                            round(float(r["mean_test_mae_eV_per_atom"]), 5), 1e-8)
    result("A5_source_match_sets23", src_ok)
    result("A5_all_positive", all(float(r["delta_adapt_eV_per_atom"]) > 0 for r in a5))

    # ---- A6: byte-equality against sources ------------------------------
    a6 = load(os.path.join(TAB, "a6_promoted_robustness.csv"))
    dist = load(rp("results/derived_evidence/distance_error_recompute/"
                   "distance_error_statistics.csv"))
    ok = True
    for row in a6:
        if row["item"].startswith(("embedding_auc", "materiality")):
            continue
        match = [r for r in dist if r["distance_metric"] == row["item"]
                 and r["statistic"] == row["statistic"]
                 and r["value"] == row["value"] and r["ci_low"] == row["ci_low"]]
        ok &= len(match) >= 1
    result("A6_byte_equality_distance_rows", ok)
    knn = next(r for r in a6 if r["item"] == "oxide_knn5_mean_distance"
               and r["statistic"] == "spearman_correlation")
    result("A6_knn5_matches_paper_value", abs(float(knn["value"]) - 0.3460) < 1e-3,
           knn["value"])

    fails = [r for r in RESULTS if not r["pass"]]
    verdict = "RR1_V3_ANALYSES_VALIDATED" if not fails else "RR1_V3_VALIDATION_FAILED"
    json.dump({"verdict": verdict, "checks_passed": len(RESULTS) - len(fails),
               "checks_total": len(RESULTS), "results": RESULTS},
              open(os.path.join(OUT, "analysis_validation.json"), "w"), indent=1)
    print(verdict, f"({len(RESULTS) - len(fails)}/{len(RESULTS)})")
    for f in fails:
        print("FAIL:", f)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
