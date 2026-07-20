# Validated analysis — Analysis Report (analysis stage 3 producer)

Producer checks: 49/49 passed.
ALL CHECKS PASSED.

## Headlines for analysis stage 4 writing

- **A1:** all 12 Protocol 1 condition means worse than zero-shot; seed-level:
  60/60 runs worse than family zero-shot. R_adapt ranges:
  oxide 5.2%..11.2%,
  nitride 1.2%..22.6%.
- **A2 oxide:** bias -0.00833, median AE 0.01545, RMSE 0.07002,
  p90 0.07865, p95 0.14216, worst-decile share 50.8%,
  target-quartile MAE ['0.0261', '0.0293', '0.0295', '0.0518'].
- **A2 nitride:** bias -0.01628, median AE 0.04194, RMSE 0.10956,
  p90 0.18068, p95 0.22724, worst-decile share 39.3%,
  target-quartile MAE ['0.0473', '0.0440', '0.0757', '0.1109'].
- **A3:** per-seed fraction improved (mean±SD) —
  oxide N=500: 0.4204±0.0086 (seed-mean 0.3821); oxide N=1000: 0.4276±0.0068 (seed-mean 0.4057); nitride N=500: 0.3736±0.0156 (seed-mean 0.3554); nitride N=1000: 0.3876±0.0254 (seed-mean 0.3471).
- **A4:** boundary(50) selections per condition: ['oxide/N10:0', 'oxide/N50:0', 'oxide/N100:0', 'oxide/N200:0', 'oxide/N500:1', 'oxide/N1000:0', 'nitride/N10:0', 'nitride/N50:0', 'nitride/N100:0', 'nitride/N200:0', 'nitride/N500:1', 'nitride/N1000:0'];
  median rel val improvement at engaged nitride budgets:
  ['0.6225', '0.7995'].
- **A5:** oxide delta 0.00130..0.00425
  (3.8%..12.4%), positive 18/18;
  nitride delta 0.00068..0.01574
  (1.0%..22.6%), positive 18/18.
- **A6:** promoted rows in tables/a6_promoted_robustness.csv (source last_alignn_pool).

Tertile profile (A3) is in tables/a3_tertile_profile.csv; the manuscript
sentence using it MUST carry the regression-to-the-mean caveat.
