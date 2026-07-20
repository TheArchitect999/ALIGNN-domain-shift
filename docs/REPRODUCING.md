# Reproducing the study

This guide follows the evidence pipeline from source data to the reported tables and figures. Run commands from the repository root. Model-training times are hardware-dependent; the recorded references below came from CUDA runs whose GPU model was not captured, so they are planning aids rather than performance claims.

## 1. Environment and public checkpoint

Create an isolated environment and install the recorded core research packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install alignn==2025.4.1 jarvis-tools==2026.3.10 numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2
python scripts/setup/fetch_pretrained.py
```

The fetcher downloads the official Figshare archive, extracts `checkpoint_300.pt`, and verifies SHA-256 `bce5cdafa06dc26ad8ddb3ceeb2bef7593c218dd66825e7cb5381c156317458f`. Its default destination is `models/pretrained/`.

`environment/requirements-frozen.txt` is the recorded dependency export used for the model runs; it is not a fully pinned cross-platform lock file. CUDA and PyTorch wheels are platform-specific, so use versions appropriate to the local driver when recreating a GPU environment.

## 2. Rebuild the family datasets

```bash
python scripts/setup/rebuild_family_datasets.py
```

The wrapper uses the committed official split manifest at `data/manifests/dft_3d_formation_energy_peratom_splits.csv`; it refuses to invent a replacement split. It downloads or reads JARVIS-DFT, deduplicates identifiers, exports structure files and manifests, materializes pool/test roots, and runs the independent family validator.

Expected validated counts:

| Family | Definition | All | Train | Validation | Fixed test | Pool |
|---|---|---:|---:|---:|---:|---:|
| Oxide | Contains O; O+N retained | 14,991 | 11,960 | 1,547 | 1,484 | 13,507 |
| Nitride | Contains N and no O | 2,288 | 1,837 | 209 | 242 | 2,046 |

To use a pre-downloaded JARVIS JSON payload:

```bash
python scripts/setup/rebuild_family_datasets.py --dataset-json /path/to/jdft_3d.json
```

A cached local rebuild took 22.37 seconds and validation took 0.12 seconds. The initial dataset download and writing 17,279 structure files will dominate on a fresh machine.

## 3. Reproduce one canonical fine-tuning run

First create the deterministic N = 50, seed 0 oxide training root:

```bash
python scripts/shared/prepare_baseline_finetune_dataset.py \
  --family oxide --N 50 --seed 0 \
  --repo-root . --results-root results/reproduction/protocol_1 \
  --link-mode symlink
```

Then run last-block-plus-head fine-tuning:

```bash
python scripts/shared/fine_tune_last_two_alignn_layers.py \
  --config configs/protocol_1/finetune/oxide_finetune_N50_seed0.finetune_last2.json \
  --output-dir results/reproduction/protocol_1/oxide/N50_seed0/finetune_last2 \
  --dataset-root results/reproduction/protocol_1/oxide/N50_seed0/dataset_root \
  --pretrained-checkpoint models/pretrained/checkpoint_300.pt \
  --pretrained-config configs/pretrained/config.json \
  --device cuda
```

The canonical output to compare against is `results/protocol_1/oxide/N50_seed0/finetune_last2/`. Compare `summary.json`, `history_val.json`, and `prediction_results_test_set.csv`; model checkpoints are intentionally external to the public tree.

## 4. Prepare the complete training matrix

The matrix is the Cartesian product of two families, six labelled budgets, five seeds, and three protocols: 180 fine-tuning runs. The 20 main-scope scratch comparisons use Protocol 1 at N = 50 and 500. An additional 40 Protocol 2/3 scratch runs are retained as supplementary robustness artifacts but are excluded from the main-scope comparison, giving 240 released training runs in total. Each tracked JSON in `configs/protocol_<n>/finetune/` is one immutable job specification.

For a scheduler or job array, use the single-run command above as the template and vary:

- `--config`: one tracked file under the selected protocol;
- `--dataset-root`: `<output namespace>/<family>/N<size>_seed<seed>/dataset_root`;
- `--output-dir`: the corresponding `finetune_last2` directory;
- family, labelled budget, and seed passed to `prepare_baseline_finetune_dataset.py`.

There is intentionally no scheduler-specific command that launches all 180 fine-tuning jobs: cluster launch syntax and resource requests are site-dependent. Generate one job per tracked configuration using this template, and preserve the protocol/family/budget/seed output layout so the aggregation step can discover the results.

For the matched random-initialization baseline, use the same prepared dataset root:

```bash
python scripts/shared/train_alignn_from_scratch.py \
  --config configs/protocol_1/from_scratch/oxide_from_scratch_N50_seed0.json \
  --output-dir results/reproduction/protocol_1/oxide/N50_seed0/train_alignn_from_scratch \
  --dataset-root results/reproduction/protocol_1/oxide/N50_seed0/dataset_root \
  --device cuda
```

Do not pool protocols: Protocol 1 is canonical; Protocols 2 and 3 are robustness checks.

## 5. Aggregate seed-level results

```bash
python scripts/shared/summarize_finetune_reports.py \
  --repo-root . \
  --results-root results/reproduction/protocol_1 \
  --zero-shot-summary results/zero_shot/zero_shot_summary.csv \
  --run-subdir finetune_last2 \
  --families oxide nitride \
  --Ns 10 50 100 200 500 1000 \
  --seeds 0 1 2 3 4 \
  --out-dir results/reproduction/summaries/protocol_1
```

The warm local aggregation took 1.11 seconds. The released reference aggregates are in `results/summaries/`.

## 6. Uncertainty and effect-size analyses

The zero-shot comparison uses 50,000 independent within-family structure-bootstrap resamples, seed 42, PCG64, and percentile intervals. The released zero-shot predictions are the numerical authorities:

- `results/zero_shot/oxide/predictions.csv`
- `results/zero_shot/nitride/predictions.csv`

Recompute the public zero-shot bootstrap directly from those prediction rows with:

```bash
python scripts/analysis/recompute_zero_shot_bootstrap.py
```

This writes `results/reproduction/zero_shot_bootstrap.csv`; compare its oxide, nitride, and ratio rows with the rows where `scenario=inclusive_oxide_vs_nitride` in `paper/supplementary/data/pure_oxide_zero_shot_bootstrap_summary.csv`. The adaptation, residual, heterogeneity, learning-dynamics, cross-protocol, and robustness tables are in `results/summaries/a1_*.csv` through `a6_*.csv`. The retained effect-size producer consumes regenerated audit inputs under `results/derived_evidence/` and writes analysis tables under `results/summaries/`; its independent validator checks those outputs. The released A1–A6 tables and evidence manifest are the public numerical authorities.

## 7. Frozen embeddings and distance–error analysis

The embedding pipeline is ordered. The first command performs the actual fixed-test extraction into a separate reproduction namespace:

```bash
python scripts/embedding_analysis/01_extract_structure_embeddings.py \
  --dataset-subset test \
  --output-dir results/reproduction/embeddings \
  --pretrained-checkpoint models/pretrained/checkpoint_300.pt \
  --pretrained-config configs/pretrained/config.json \
  --device cuda
```

Then run `02_build_embedding_metadata.py`, `03_plot_pca.py`, and `06_quantify_family_separation.py` with explicit input and output paths under `results/reproduction/embeddings/`; inspect each script's `--help` before selecting the fixed-test, balanced-pool, or oxide-reference scope. Script 02 also writes its human-readable design notes to `results/derived_evidence/embedding_analysis/`. Explicit paths keep the released `results/embeddings/` reference arrays immutable. Recorded extraction times were 47.36 seconds for 1,726 fixed-test structures, 97.14 seconds for the 4,092-structure balanced subset, and 405.68 seconds for the 13,507-structure oxide reference pool—about 9.2 minutes total.

For the publication-authority distance–error statistics, join the frozen arrays to the canonical per-structure zero-shot errors and run the independent validator:

```bash
python scripts/analysis/recompute_distance_error_canonical.py \
  --output-dir results/derived_evidence/distance_error_recompute --overwrite
python scripts/analysis/validate_distance_error_canonical.py \
  --output-dir results/derived_evidence/distance_error_recompute
```

The older `07_analyze_nitride_distance_vs_error.py` workflow is retained for exploratory provenance, but its extraction-time error column is not the numerical authority for Figure 4.

## 8. Figures and supplementary materials

The final figure package contains PNG, SVG, and plot-data CSV artifacts. `paper/figures/generation_manifest.json` records their current SHA-256 values and the neutral producer path. The small validated inputs needed for deterministic figure assembly are released under `results/derived_evidence/` and `results/summaries/protocol_1/comparisons/`, while dense public plot coordinates remain beside the figures. Rebuild the four main figures with:

```bash
python scripts/figures/generate_paper_figures.py \
  --repo-root . --output-dir results/reproduction/figures
```

Regenerate the 24 run grids and two robustness surfaces used by the supplement with:

```bash
python scripts/figures/regenerate_supplementary_grids.py
```

The curated supplementary source and render are `paper/supplementary/supplementary_materials.md` and `.pdf`; their machine-readable tables and figure inputs live beside them. The PDF itself is a release artifact rather than an output of the grid producer.

## 9. Verify against the reported numbers

Every row in `paper/evidence_manifest.csv` must resolve inside the repository, and each machine-readable selector must recover the displayed value. This read-only verifier checks both invariants:

```bash
python scripts/analysis/verify_evidence_manifest.py
```

Headline checks:

- oxide zero-shot MAE: 0.0342 eV/atom;
- nitride zero-shot MAE: 0.0695 eV/atom;
- nitride/oxide ratio: 2.03, 95% CI [1.70, 2.42];
- fine-tuning condition means below their family zero-shot baseline: 0/36.

## Recorded runtime reference

| Workload | Runs | Median per run | Sequential total |
|---|---:|---:|---:|
| Protocol 1 fine-tuning | 60 | 76.3 s | 1.61 h |
| Protocol 1 scratch | 20 | 97.7 s | 0.53 h |
| Protocol 2 fine-tuning | 60 | 131.4 s | 3.86 h |
| Protocol 2 scratch | 20 | 260.6 s | 1.49 h |
| Protocol 3 fine-tuning | 60 | 62.9 s | 1.71 h |
| Protocol 3 scratch | 20 | 124.3 s | 0.69 h |

All recorded model runs total about 9.88 sequential GPU-hours. Parallel execution changes wall-clock time but not the fixed configs, seeds, splits, or validation selectors.
