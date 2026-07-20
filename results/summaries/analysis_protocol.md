# Prespecified descriptive analysis protocol

This protocol was recorded on 2026-07-18 before the analyses below were
computed. Every input is released in this repository, and no additional model
training was performed for these summaries. The purpose is to define the
effect-size, residual, heterogeneity, learning-dynamics, and robustness analyses
reported in this directory and governed by `paper/evidence_manifest.csv`.

## Analysis principles

1. Every computed result is retained regardless of direction. Interpretations
   are revised if an analysis weakens an initial claim.
2. A1–A5 are descriptive analyses and introduce no new p-values. A6 reports
   statistics already computed under the registered bootstrap, permutation,
   and BH-FDR procedures.
3. Representation results are described as associations, not causal effects or
   evidence of a specific mechanism.
4. An epoch-1 selection means the end-of-epoch-1 checkpoint; it does not imply
   byte identity with the pretrained checkpoint or the absence of updates.
5. MAE values are rounded to five decimal places in tables and four in prose;
   percentages are rounded to one decimal place.

Per-structure error is prediction minus target, and absolute error is its
absolute value. Adaptation change is defined as
`delta_adapt = MAE_finetuned - MAE_zero_shot`; positive values therefore mean
that fine-tuning increased MAE. Relative adaptation change is
`100 * delta_adapt / MAE_zero_shot`.

## Verified inputs

| Input | Public path | Key fields |
| --- | --- | --- |
| Zero-shot per-structure predictions | `results/zero_shot/{oxide,nitride}/predictions.csv` | `jid`, `target`, `prediction`, `abs_error` |
| Protocol 1 fine-tuned predictions | `results/protocol_1/{family}/N{N}_seed{seed}/finetune_last2/prediction_results_test_set.csv` | `id`, `target`, `prediction` |
| Protocol 1 validation history | `results/protocol_1/{family}/N{N}_seed{seed}/finetune_last2/history_val.json` | validation L1 by epoch |
| Protocol 1 run summary | `results/protocol_1/{family}/N{N}_seed{seed}/finetune_last2/summary.json` | `best_epoch`, test MAE |
| Protocol 1 canonical aggregate | `results/derived_evidence/protocol_1_regeneration/aggregate_recomputation.csv` | seed MAEs, best epochs, mean, SD |
| Protocol 2 aggregate | `results/summaries/protocol_2/finetune/finetune_summary_by_N.csv` | mean MAE, mean best epoch, zero-shot MAE |
| Protocol 3 aggregate | `results/summaries/protocol_3/finetune/finetune_summary_by_N.csv` | mean MAE, mean best epoch, zero-shot MAE |
| Distance–error statistics | `results/derived_evidence/distance_error_recompute/distance_error_statistics.csv` | metric, statistic, interval, adjusted p-value |
| Embedding sensitivity | `results/derived_evidence/provenance_dataset_closure/2_3C_oxynitride_bootstrap/embedding_sensitivity_metrics.csv` | metric and scenario rows |

Fine-tuned `id` values are joined to zero-shot `jid` values. The fixed-test
identifier sets have complete join coverage, and matched targets agree within
`1e-4` eV/atom despite differing stored precision.

## Prespecified analyses

### A1 — Protocol 1 effect sizes and seed consistency

For each of the 12 family-by-training-size conditions, report absolute and
relative adaptation change. Count how many of the five seed MAEs exceed the
family zero-shot MAE, both per condition and across all 60 runs.

Output: `results/summaries/a1_protocol_1_effect_sizes.csv`.

### A2 — Zero-shot residual decomposition

For each family, report signed mean error, median absolute error, RMSE, p90 and
p95 absolute error, the share of total absolute error contributed by the worst
decile, and MAE within target-energy quartiles. These summaries are descriptive;
the 50,000-replicate structure-bootstrap intervals remain the uncertainty
statement for the family MAEs.

Output: `results/summaries/a2_residual_decomposition.csv`.

### A3 — Structure-level heterogeneity

Use Protocol 1 at `N = 500` and `N = 1,000` for both families. For each
condition, calculate the per-seed fraction of fixed-test structures whose
fine-tuned absolute error is lower than zero-shot error, then summarize the five
fractions by mean and sample SD. Also report the fraction improved by the
five-seed-mean prediction and descriptive profiles across zero-shot-error
tertiles. Tertile contrasts are subject to regression to the mean and are not a
targeting mechanism.

Outputs: `results/summaries/a3_structure_heterogeneity.csv` and
`results/summaries/a3_tertile_profile.csv`.

### A4 — Learning dynamics

Across the 60 Protocol 1 fine-tuning runs, report relative validation
improvement from epoch 1 to the selected minimum, the final-minus-best
validation gap, selection at the 50-epoch boundary, and within-condition sample
SD of selected epochs.

Output: `results/summaries/a4_learning_dynamics.csv`.

### A5 — Cross-protocol effect magnitudes

For all 36 protocol-by-family-by-training-size conditions, report absolute and
relative adaptation change. Summarize per-family ranges and sign-consistency
counts. Protocols are not pooled into an average; the comparison is evidence of
robustness across the three recorded optimization choices.

Output: `results/summaries/a5_cross_protocol_effects.csv`.

### A6 — Validated robustness statistics

Promote the already validated Spearman statistics for centroid, five-nearest-
neighbor, and Mahalanobis distances; the five-nearest-neighbor hard-minus-easy
contrast; and the pure-oxide embedding-sensitivity outcome. This step copies
registered values and source pointers without recomputing them.

Output: `results/summaries/a6_promoted_robustness.csv`.

## Deterministic implementation and validation

The producer is `scripts/analysis/compute_effect_size_analyses.py`; the
independent checker is `scripts/analysis/validate_effect_size_analyses.py`.
They verify expected row counts, complete identifier joins, target agreement,
all 12 Protocol 1 conditions, all 60 Protocol 1 runs, all 36 cross-protocol
conditions, and exact promotion of the A6 source values. The checker exits
nonzero on any mismatch.

The public evidence checker in
`scripts/analysis/verify_evidence_manifest.py` independently resolves every
governed selector and compares it with the released value in
`paper/evidence_manifest.csv`.
