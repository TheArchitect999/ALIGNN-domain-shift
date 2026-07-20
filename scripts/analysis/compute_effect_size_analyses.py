#!/usr/bin/env python3
"""Validated analysis — analysis_stage_3 producer.

Computes the prespecified analyses A1-A5 and extracts A6 exactly as defined in
results/summaries/analysis_protocol.md.
Stdlib only; deterministic; no new inferential tests.

Outputs (under results/summaries/):
  tables/a1_protocol_1_effect_sizes.csv
  tables/a2_residual_decomposition.csv
  tables/a3_structure_heterogeneity.csv
  tables/a3_tertile_profile.csv
  tables/a4_learning_dynamics.csv
  tables/a5_cross_protocol_effects.csv
  tables/a6_promoted_robustness.csv
  analysis_report.md
  analysis_evidence_manifest.csv
  analysis_evidence.sha256
"""
import csv
import hashlib
import json
import math
import os
import statistics
import sys

REPO = "."
OUT = os.path.join(REPO, "results/summaries")
TAB = os.path.join(OUT, "tables")

ZS_PATH = "results/zero_shot/{fam}/predictions.csv"
AGG_PATH = "results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv"
RUN_DIR = "results/protocol_1/{fam}/N{n}_seed{s}/finetune_last2"
SETSUM = "results/summaries/protocol_{k}/finetune/finetune_summary_by_N.csv"
DIST_STATS = "results/derived_evidence/distance_error_recompute/distance_error_statistics.csv"
EMB_SENS = ("results/derived_evidence/provenance_dataset_closure/"
            "2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv")
MATERIALITY = ("results/derived_evidence/provenance_dataset_closure/"
               "2_3C_oxynitride_bootstrap/materiality_claim_impact.md")

FAMILIES = ["oxide", "nitride"]
NS = [10, 50, 100, 200, 500, 1000]
SEEDS = [0, 1, 2, 3, 4]
EXPECTED_TEST_N = {"oxide": 1484, "nitride": 242}
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append({"check": name, "pass": bool(ok), "detail": str(detail)[:200]})
    if not ok:
        print(f"CHECK FAILED: {name}: {detail}", file=sys.stderr)


def rp(path):
    return os.path.join(REPO, path)


def quantile_linear(sorted_x, q):
    """NumPy-default 'linear' interpolation quantile on a pre-sorted list."""
    n = len(sorted_x)
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_x[int(pos)]
    return sorted_x[lo] + (pos - lo) * (sorted_x[hi] - sorted_x[lo])


def read_zero_shot(fam):
    rows = {}
    with open(rp(ZS_PATH.format(fam=fam))) as fh:
        for r in csv.DictReader(fh):
            rows[r["jid"]] = (float(r["target"]), float(r["prediction"]),
                              float(r["abs_error"]))
    check(f"zs_count_{fam}", len(rows) == EXPECTED_TEST_N[fam],
          f"{len(rows)} vs {EXPECTED_TEST_N[fam]}")
    return rows


def read_run_predictions(fam, n, s):
    path = rp(RUN_DIR.format(fam=fam, n=n, s=s)) + "/prediction_results_test_set.csv"
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[r["id"]] = (float(r["target"]), float(r["prediction"]))
    return out


def read_run_summary(fam, n, s):
    with open(rp(RUN_DIR.format(fam=fam, n=n, s=s)) + "/summary.json") as fh:
        return json.load(fh)


def read_run_val_history(fam, n, s):
    with open(rp(RUN_DIR.format(fam=fam, n=n, s=s)) + "/history_val.json") as fh:
        return json.load(fh)


def wcsv(path, header, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def main():
    os.makedirs(TAB, exist_ok=True)
    manifest = []

    zs = {fam: read_zero_shot(fam) for fam in FAMILIES}

    # ---------------- A1: protocol_1 effect sizes and seed consistency ----------
    agg = list(csv.DictReader(open(rp(AGG_PATH))))
    check("a1_agg_rows", len(agg) == 12, len(agg))
    a1_rows, seeds_worse_total = [], 0
    a1_lookup = {}
    for r in agg:
        fam, n = r["family"], int(r["N"])
        mean_mae = float(r["mean_test_mae_eV_per_atom"])
        zs_mae = float(r["zero_shot_mae_eV_per_atom"])
        seed_maes = json.loads(r["raw_seed_test_mae_eV_per_atom"])
        delta = mean_mae - zs_mae
        r_adapt = 100.0 * delta / zs_mae
        worse = sum(1 for m in seed_maes if m > zs_mae)
        seeds_worse_total += worse
        a1_rows.append([fam, n, f"{mean_mae:.5f}", f"{zs_mae:.5f}",
                        f"{delta:.5f}", f"{r_adapt:.1f}", f"{worse}/5"])
        a1_lookup[(fam, n)] = (delta, r_adapt, worse)
    wcsv(os.path.join(TAB, "a1_protocol_1_effect_sizes.csv"),
         ["family", "N", "mean_ft_mae_eV_per_atom", "zero_shot_mae_eV_per_atom",
          "delta_adapt_eV_per_atom", "r_adapt_percent", "seeds_worse_than_zs"],
         a1_rows)
    check("a1_all_positive", all(float(x[4]) > 0 for x in a1_rows))
    manifest.append(["RR1-A1-TABLE", "protocol_1 effect sizes + seed consistency",
                     AGG_PATH, "all 12 family x N rows", "eV/atom; %", "table"])
    manifest.append(["RR1-A1-SEEDS-TOTAL",
                     f"protocol_1 seed-level runs worse than family zero-shot: {seeds_worse_total}/60",
                     AGG_PATH, "raw_seed_test_mae vs zero_shot", "count",
                     f"{seeds_worse_total}/60"])

    # ---------------- A2: zero-shot residual decomposition -----------------
    a2_rows = []
    a2_detail = {}
    for fam in FAMILIES:
        data = zs[fam]
        errs = [p - t for (t, p, _) in data.values()]
        aes = sorted(a for (_, _, a) in data.values())
        n = len(aes)
        bias = statistics.fmean(errs)
        med = quantile_linear(aes, 0.5)
        rmse = math.sqrt(statistics.fmean([e * e for e in errs]))
        p90 = quantile_linear(aes, 0.90)
        p95 = quantile_linear(aes, 0.95)
        n_top = math.ceil(0.10 * n)
        top_share = sum(aes[-n_top:]) / sum(aes)
        targets = sorted(t for (t, _, _) in data.values())
        b1, b2, b3 = (quantile_linear(targets, q) for q in (0.25, 0.5, 0.75))
        qmae = {1: [], 2: [], 3: [], 4: []}
        for (t, _, a) in data.values():
            qi = 1 if t <= b1 else 2 if t <= b2 else 3 if t <= b3 else 4
            qmae[qi].append(a)
        qvals = [statistics.fmean(qmae[i]) for i in (1, 2, 3, 4)]
        a2_rows.append([fam, n, f"{bias:.5f}", f"{med:.5f}", f"{rmse:.5f}",
                        f"{p90:.5f}", f"{p95:.5f}", n_top, f"{top_share:.4f}",
                        f"{b1:.4f}", f"{b2:.4f}", f"{b3:.4f}",
                        *[f"{v:.5f}" for v in qvals]])
        a2_detail[fam] = dict(bias=bias, med=med, rmse=rmse, p90=p90, p95=p95,
                              top_share=top_share, qvals=qvals, n_top=n_top)
    wcsv(os.path.join(TAB, "a2_residual_decomposition.csv"),
         ["family", "n", "signed_mean_error_eV_per_atom", "median_ae", "rmse",
          "p90_ae", "p95_ae", "n_top_decile", "worst_decile_ae_share",
          "target_q1_bound", "target_q2_bound", "target_q3_bound",
          "mae_target_q1", "mae_target_q2", "mae_target_q3", "mae_target_q4"],
         a2_rows)
    manifest.append(["RR1-A2-TABLE", "Zero-shot residual decomposition",
                     ZS_PATH.format(fam="{oxide,nitride}"),
                     "per-family point statistics (descriptive)", "eV/atom", "table"])

    # ---------------- A3: structure-level heterogeneity --------------------
    a3_rows, a3_tert_rows = [], []
    for fam in FAMILIES:
        zsd = zs[fam]
        aes_sorted = sorted(a for (_, _, a) in zsd.values())
        t1 = quantile_linear(aes_sorted, 1 / 3)
        t2 = quantile_linear(aes_sorted, 2 / 3)
        tert_of = {j: (1 if a <= t1 else 2 if a <= t2 else 3)
                   for j, (_, _, a) in zsd.items()}
        for n in (500, 1000):
            per_seed_frac, ft_ae_by_jid = [], {j: [] for j in zsd}
            per_seed_frac_tert = {1: [], 2: [], 3: []}
            for s in SEEDS:
                preds = read_run_predictions(fam, n, s)
                check(f"a3_join_{fam}_{n}_{s}",
                      set(preds) == set(zsd), f"{len(preds)} ids")
                tmax = max(abs(preds[j][0] - zsd[j][0]) for j in zsd)
                check(f"a3_target_{fam}_{n}_{s}", tmax <= 1e-4, tmax)
                improved = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
                n_imp = 0
                for j, (t, p) in preds.items():
                    ae = abs(p - t)
                    ft_ae_by_jid[j].append(ae)
                    better = ae < zsd[j][2]
                    n_imp += better
                    ti = tert_of[j]
                    improved[ti][0] += better
                    improved[ti][1] += 1
                per_seed_frac.append(n_imp / len(preds))
                for ti in (1, 2, 3):
                    per_seed_frac_tert[ti].append(improved[ti][0] / improved[ti][1])
            frac_mean = statistics.fmean(per_seed_frac)
            frac_sd = statistics.stdev(per_seed_frac)
            mean_ae = {j: statistics.fmean(v) for j, v in ft_ae_by_jid.items()}
            frac_seedmean = sum(1 for j in zsd if mean_ae[j] < zsd[j][2]) / len(zsd)
            a3_rows.append([fam, n, f"{frac_mean:.4f}", f"{frac_sd:.4f}",
                            f"{frac_seedmean:.4f}"])
            for ti in (1, 2, 3):
                jids = [j for j in zsd if tert_of[j] == ti]
                dmean = statistics.fmean([mean_ae[j] - zsd[j][2] for j in jids])
                a3_tert_rows.append(
                    [fam, n, ti, len(jids),
                     f"{statistics.fmean(per_seed_frac_tert[ti]):.4f}",
                     f"{dmean:.5f}"])
    wcsv(os.path.join(TAB, "a3_structure_heterogeneity.csv"),
         ["family", "N", "frac_improved_perseed_mean", "frac_improved_perseed_sd",
          "frac_improved_seedmean_ae"], a3_rows)
    wcsv(os.path.join(TAB, "a3_tertile_profile.csv"),
         ["family", "N", "zs_ae_tertile", "n_structures",
          "frac_improved_perseed_mean", "mean_delta_ae_ft_minus_zs"], a3_tert_rows)
    manifest.append(["RR1-A3-TABLE", "Per-structure FT vs ZS heterogeneity (N=500/1000)",
                     RUN_DIR.format(fam="{fam}", n="{500,1000}", s="{0-4}")
                     + "/prediction_results_test_set.csv",
                     "paired by JID against canonical zero-shot", "fractions; eV/atom",
                     "tables"])

    # ---------------- A4: learning dynamics --------------------------------
    hist_col = 0
    a4_run = []
    mismatch0 = 0
    for fam in FAMILIES:
        for n in NS:
            for s in SEEDS:
                h = read_run_val_history(fam, n, s)
                summ = read_run_summary(fam, n, s)
                be = int(summ["best_epoch"])
                v = [row[hist_col] for row in h]
                argmin = min(range(len(v)), key=lambda i: v[i]) + 1
                if argmin != be:
                    mismatch0 += 1
                v1, vbest, vfin = v[0], min(v), v[-1]
                a4_run.append([fam, n, s, be, v1, vbest, vfin])
    check("a4_bestepoch_consistency", mismatch0 == 0,
          f"{mismatch0}/60 argmin mismatches on column {hist_col}")
    a4_rows = []
    for fam in FAMILIES:
        for n in NS:
            runs = [r for r in a4_run if r[0] == fam and r[1] == n]
            rel = [(r[4] - r[5]) / r[4] for r in runs]
            gap = [r[6] - r[5] for r in runs]
            bes = [r[3] for r in runs]
            a4_rows.append([fam, n,
                            f"{quantile_linear(sorted(rel), 0.5):.4f}",
                            f"{quantile_linear(sorted(gap), 0.5):.5f}",
                            sum(1 for b in bes if b == 50),
                            f"{statistics.stdev(bes):.1f}"])
    wcsv(os.path.join(TAB, "a4_learning_dynamics.csv"),
         ["family", "N", "median_rel_epoch1_to_best_val_improvement",
          "median_final_minus_best_val_l1", "boundary_selections_epoch50_of5",
          "best_epoch_sd"], a4_rows)
    manifest.append(["RR1-A4-TABLE", "protocol_1 validation-dynamics summaries",
                     RUN_DIR.format(fam="{fam}", n="{N}", s="{s}")
                     + "/history_val.json + summary.json",
                     "60 runs; column 0 of history_val", "relative; L1; counts",
                     "table"])

    # ---------------- A5: cross-protocol effect magnitudes -----------------
    a5_rows = []
    for r in agg:
        fam, n = r["family"], int(r["N"])
        mean_mae = float(r["mean_test_mae_eV_per_atom"])
        zs_mae = float(r["zero_shot_mae_eV_per_atom"])
        d = mean_mae - zs_mae
        a5_rows.append(["protocol_1", fam, n, f"{mean_mae:.5f}", f"{zs_mae:.5f}",
                        f"{d:.5f}", f"{100 * d / zs_mae:.1f}"])
    for k in (2, 3):
        for r in csv.DictReader(open(rp(SETSUM.format(k=k)))):
            fam, n = r["family"], int(r["N"])
            mean_mae = float(r["mean_test_mae_eV_per_atom"])
            zs_mae = float(r["zero_shot_mae_eV_per_atom"])
            d = mean_mae - zs_mae
            a5_rows.append([f"Set{k}", fam, n, f"{mean_mae:.5f}", f"{zs_mae:.5f}",
                            f"{d:.5f}", f"{100 * d / zs_mae:.1f}"])
    check("a5_36_rows", len(a5_rows) == 36, len(a5_rows))
    check("a5_all_positive", all(float(r[5]) > 0 for r in a5_rows))
    wcsv(os.path.join(TAB, "a5_cross_protocol_effects.csv"),
         ["set", "family", "N", "mean_ft_mae_eV_per_atom",
          "zero_shot_mae_eV_per_atom", "delta_adapt_eV_per_atom",
          "r_adapt_percent"], a5_rows)
    a5_summary = {}
    for fam in FAMILIES:
        ds = [float(r[5]) for r in a5_rows if r[1] == fam]
        rs = [float(r[6]) for r in a5_rows if r[1] == fam]
        a5_summary[fam] = dict(dmin=min(ds), dmax=max(ds), rmin=min(rs),
                               rmax=max(rs), pos=sum(1 for d in ds if d > 0))
        manifest.append([f"RR1-A5-RANGE-{fam.upper()}",
                         f"{fam} delta_adapt range {min(ds):.5f}..{max(ds):.5f} eV/atom "
                         f"({min(rs):.1f}%..{max(rs):.1f}%), positive {a5_summary[fam]['pos']}/18",
                         "protocol_1 aggregates + protocol_2/3 summaries", "18 conditions",
                         "eV/atom; %", "summary"])

    # ---------------- A6: promotion extraction -----------------------------
    dist_rows = list(csv.DictReader(open(rp(DIST_STATS))))
    qcol = next((c for c in dist_rows[0] if "bh_fdr" in c.lower()), None)
    sources = sorted({r["embedding_source"] for r in dist_rows})
    sel_source = None
    for src in sources:
        for r in dist_rows:
            if (r["embedding_source"] == src
                    and r["distance_metric"] == "oxide_knn5_mean_distance"
                    and r["statistic"] == "spearman_correlation"
                    and abs(float(r["value"]) - 0.3460) < 1e-3):
                sel_source = src
    check("a6_selected_source", sel_source is not None, f"sources={sources}")
    a6_rows = []
    for r in dist_rows:
        if r["embedding_source"] != sel_source:
            continue
        if (r["statistic"] == "spearman_correlation"
                or (r["distance_metric"] == "oxide_knn5_mean_distance"
                    and r["statistic"] == "hard_minus_easy_mean_distance")):
            a6_rows.append([r["distance_metric"], r["statistic"], r["value"],
                            r["ci_low"], r["ci_high"],
                            r.get(qcol, ""), DIST_STATS])
    emb = list(csv.DictReader(open(rp(EMB_SENS))))
    for r in emb:
        if (r["dataset"] == "fixed_test_set"
                and r["embedding_source"] == "last_alignn_pool"
                and r["metric_name"] == "logistic_regression_family_auc"):
            a6_rows.append(["embedding_auc:" + r["scenario"], r["metric_scope"],
                            r["value"], r["ci_low"], r["ci_high"], "", EMB_SENS])
    stable = "INTERPRETATION_STABLE" in open(rp(MATERIALITY)).read()
    check("a6_materiality_stable", stable)
    a6_rows.append(["materiality", "interpretation",
                    "INTERPRETATION_STABLE" if stable else "UNSTABLE",
                    "", "", "", MATERIALITY])
    wcsv(os.path.join(TAB, "a6_promoted_robustness.csv"),
         ["item", "statistic", "value", "ci_low", "ci_high", "bh_fdr_q",
          "source_path"], a6_rows)
    manifest.append(["RR1-A6-TABLE", "Promoted already-validated robustness rows "
                     f"(embedding source: {sel_source})", DIST_STATS + " ; " + EMB_SENS,
                     "verbatim extraction, no recompute", "as-source", "table"])

    # ---------------- report, manifest, checksums --------------------------
    fails = [c for c in CHECKS if not c["pass"]]
    ox, ni = a2_detail["oxide"], a2_detail["nitride"]
    report = f"""# Validated analysis — Analysis Report (analysis_stage_3 producer)

Producer checks: {len(CHECKS) - len(fails)}/{len(CHECKS)} passed.
{"ALL CHECKS PASSED." if not fails else "FAILED: " + json.dumps(fails)}

## Headlines for analysis_stage_4 writing

- **A1:** all 12 protocol_1 condition means worse than zero-shot; seed-level:
  {seeds_worse_total}/60 runs worse than family zero-shot. R_adapt ranges:
  oxide {min(float(r[5]) for r in a1_rows if r[0]=='oxide'):.1f}%..{max(float(r[5]) for r in a1_rows if r[0]=='oxide'):.1f}%,
  nitride {min(float(r[5]) for r in a1_rows if r[0]=='nitride'):.1f}%..{max(float(r[5]) for r in a1_rows if r[0]=='nitride'):.1f}%.
- **A2 oxide:** bias {ox['bias']:+.5f}, median AE {ox['med']:.5f}, RMSE {ox['rmse']:.5f},
  p90 {ox['p90']:.5f}, p95 {ox['p95']:.5f}, worst-decile share {ox['top_share']:.1%},
  target-quartile MAE {['%.4f' % v for v in ox['qvals']]}.
- **A2 nitride:** bias {ni['bias']:+.5f}, median AE {ni['med']:.5f}, RMSE {ni['rmse']:.5f},
  p90 {ni['p90']:.5f}, p95 {ni['p95']:.5f}, worst-decile share {ni['top_share']:.1%},
  target-quartile MAE {['%.4f' % v for v in ni['qvals']]}.
- **A3:** per-seed fraction improved (mean±SD) —
  {"; ".join(f"{r[0]} N={r[1]}: {r[2]}±{r[3]} (seed-mean {r[4]})" for r in a3_rows)}.
- **A4:** boundary(50) selections per condition: {[f"{r[0]}/N{r[1]}:{r[4]}" for r in a4_rows]};
  median rel val improvement at engaged nitride budgets:
  {[r[2] for r in a4_rows if r[0]=='nitride' and r[1] in (500,1000)]}.
- **A5:** oxide delta {a5_summary['oxide']['dmin']:.5f}..{a5_summary['oxide']['dmax']:.5f}
  ({a5_summary['oxide']['rmin']:.1f}%..{a5_summary['oxide']['rmax']:.1f}%), positive {a5_summary['oxide']['pos']}/18;
  nitride delta {a5_summary['nitride']['dmin']:.5f}..{a5_summary['nitride']['dmax']:.5f}
  ({a5_summary['nitride']['rmin']:.1f}%..{a5_summary['nitride']['rmax']:.1f}%), positive {a5_summary['nitride']['pos']}/18.
- **A6:** promoted rows in tables/a6_promoted_robustness.csv (source {sel_source}).

Tertile profile (A3) is in tables/a3_tertile_profile.csv; the manuscript
sentence using it MUST carry the regression-to-the-mean caveat.
"""
    open(os.path.join(OUT, "analysis_report.md"), "w").write(report)
    wcsv(os.path.join(OUT, "analysis_evidence_manifest.csv"),
         ["claim_id", "description", "source_path", "selector", "units", "kind"],
         manifest)
    with open(os.path.join(OUT, "producer_checks.json"), "w") as fh:
        json.dump(CHECKS, fh, indent=1)
    sha_lines = []
    for root, _, files in os.walk(OUT):
        for f in sorted(files):
            if f == "analysis_evidence.sha256":
                continue
            p = os.path.join(root, f)
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            sha_lines.append(f"{h}  {os.path.relpath(p, REPO)}")
    open(os.path.join(OUT, "analysis_evidence.sha256"), "w").write(
        "\n".join(sorted(sha_lines)) + "\n")
    print(report)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
