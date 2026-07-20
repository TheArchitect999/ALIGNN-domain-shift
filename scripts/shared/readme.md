# Shared Script Guide

This folder contains reusable training, evaluation, reporting, plotting, and validation scripts for the ALIGNN domain-shift experiments.

The filenames are intentionally descriptive. Most scripts follow this pattern:

- `Run_*`: launch a training workflow or experiment suite
- `Preflight_*`: check that the runtime has the expected packages, CUDA support, and input files before training
- `Check_*`: verify that expected outputs, summaries, plots, and manifests exist after a run
- `Generate_*`: build report bundles or derived artifacts
- `Summarize_*`: aggregate run-level results into CSV/JSON/LaTeX report files
- `Plot_*`: create figures from existing logs, summaries, or predictions

## Optimization protocols

The report and experiment scripts use these protocol numbers:

| Protocol | Meaning | Epochs | Batch size | Learning rate |
| --- | --- | ---: | ---: | ---: |
| `Protocol 1` | canonical setting | `50` | `16` | `0.0001` |
| `Protocol 2` | ALIGNN-recommended setting | `300` | `64` | `0.001` |
| `Protocol 3` | Additional low-learning-rate setting | `100` | `32` | `0.00005` |

## Main Training Engines

These are the core model-running scripts. Other pipeline/suite scripts call into these.

| Script | Purpose |
| --- | --- |
| `fine_tune_last_two_alignn_layers.py` | Fine-tunes only the last ALIGNN GCN block and the final fully connected layer while keeping the rest of the pretrained model frozen. This is the main fine-tuning engine. |
| `train_alignn_from_scratch.py` | Trains an ALIGNN model from random initialization using the configured architecture and data split. |
| `evaluate_alignn_zero_shot.py` | Evaluates the pretrained ALIGNN model without additional training. |
| `fine_tune_last_two_alignn_layers_candidate_hyperparameters.py` | Draft/legacy fine-tuning variant kept for provenance around early ALIGNN-hyperparameter experiments. |

## Fine-Tuning Pipelines And Suites

Use these when running fine-tuning experiments.

| Script | Purpose |
| --- | --- |
| `run_finetune_original_baseline_suite.py` | Original baseline fine-tuning suite for the `results/protocol_1/` namespace. |
| `run_finetune_protocol_2_5_seed_pipeline.sh` | Colab pipeline wrapper for 5-seed fine-tuning with Protocol 2. |
| `run_finetune_protocol_2_5_seed_suite.py` | Python suite for the 5-seed Protocol 2 fine-tuning runs. |
| `run_finetune_protocol_2_5_seed_smoke_test.sh` | Smoke-test wrapper for the Protocol 2 Colab fine-tuning workflow. |
| `run_finetune_protocol_3_pipeline.sh` | Pipeline wrapper for Protocol 3 fine-tuning. |
| `run_finetune_protocol_3_suite.py` | Python suite for Protocol 3 fine-tuning runs. |
| `run_finetune_candidate_hyperparameters_pipeline.sh` | Draft/legacy ALIGNN-hyperparameter fine-tuning pipeline. |
| `run_finetune_candidate_hyperparameters_suite.py` | Draft/legacy Python suite for the same experimental path. |
| `run_finetune_candidate_hyperparameters_smoke_test.sh` | Smoke-test wrapper for the draft ALIGNN-hyperparameter path. |

## From-Scratch Pipelines And Suites

Use these when running random-initialization from-scratch experiments.

| Script | Purpose |
| --- | --- |
| `run_from_scratch_suite.py` | Generic from-scratch run orchestrator used by hyperparameter-specific wrappers. |
| `run_from_scratch_protocol_2_pipeline.sh` | Pipeline wrapper for from-scratch training with Protocol 2. |
| `run_from_scratch_protocol_2_smoke_test.sh` | Smoke-test wrapper for the Protocol 2 from-scratch workflow. |
| `run_from_scratch_protocol_3_pipeline.sh` | Pipeline wrapper for from-scratch training with Protocol 3. |
| `run_from_scratch_protocol_3_suite.py` | Python suite wrapper for Protocol 3 from-scratch runs. |

## Preflight Checks

Preflight scripts are intended to be run before expensive Colab/A100 training jobs. They check environment and input readiness.

| Script | Purpose |
| --- | --- |
| `preflight_finetune_protocol_2_5_seed.py` | Preflight for the Protocol 2 Colab fine-tuning workflow. |
| `preflight_finetune_protocol_3.py` | Preflight for Protocol 3 fine-tuning. |
| `preflight_from_scratch_protocol_2.py` | Preflight for Protocol 2 from-scratch training. |
| `preflight_from_scratch_protocol_3.py` | Preflight for Protocol 3 from-scratch training. |
| `preflight_finetune_candidate_hyperparameters.py` | Preflight for the draft/legacy ALIGNN-hyperparameter fine-tuning workflow. |

## Status Checks

Status scripts verify that completed runs produced the expected summaries, plots, and manifests.

| Script | Purpose |
| --- | --- |
| `check_baseline_status.sh` | Checks the baseline outputs. |
| `check_finetune_original_baseline_status.sh` | Checks the original baseline fine-tuning outputs. |
| `check_finetune_imported_namespace_status.sh` | Generic checker for imported fine-tuning namespaces such as Protocol 1 and Protocol 2. |
| `check_finetune_protocol_2_5_seed_status.sh` | Checks the Colab 5-seed Protocol 2 fine-tuning outputs. |
| `check_finetune_protocol_3_status.sh` | Checks Protocol 3 fine-tuning outputs. |
| `check_from_scratch_imported_namespace_status.sh` | Generic checker for imported from-scratch namespaces such as Protocol 1 and Protocol 2. |
| `check_from_scratch_protocol_2_status.sh` | Checks Protocol 2 from-scratch outputs. |
| `check_from_scratch_protocol_3_status.sh` | Checks Protocol 3 from-scratch outputs. |
| `check_finetune_candidate_hyperparameters_status.sh` | Checks the draft/legacy ALIGNN-hyperparameter fine-tuning outputs. |

## Report Generation And Summaries

These scripts produce aggregate report artifacts under `results/derived_evidence/`.

| Script | Purpose |
| --- | --- |
| `generate_finetune_report_protocol_1.sh` | Builds fine-tuning report artifacts for Protocol 1. |
| `generate_finetune_report_protocol_2.sh` | Builds fine-tuning report artifacts for Protocol 2. |
| `generate_from_scratch_report_protocol_1.sh` | Builds from-scratch report artifacts for Protocol 1. |
| `generate_from_scratch_report_protocol_2.sh` | Builds from-scratch report artifacts for Protocol 2. |
| `generate_finetune_parity_plots.py` | Creates true-vs-predicted parity plots from best-checkpoint test predictions. |
| `summarize_finetune_reports.py` | Aggregates fine-tuning run summaries into report CSV/JSON/LaTeX files and learning curves. |
| `summarize_finetune_protocol_2_5_seed.py` | Historical summarizer for the Colab 5-seed Protocol 2 fine-tuning bundle. |
| `summarize_finetune_candidate_hyperparameters.py` | Draft/legacy fine-tuning summarizer. |
| `summarize_from_scratch_reports.py` | Aggregates from-scratch run summaries and creates comparison plots. |
| `summarize_from_scratch_zero_shot_only.py` | Variant from-scratch summarizer that compares only against zero-shot rather than fine-tuning. |

## Plotting Utilities

These create figures from existing results and summaries. They do not train models.

| Script | Purpose |
| --- | --- |
| `plot_finetune_learning_curves_by_protocol.py` | Regenerates named fine-tuning learning curves for protocol report folders. |
| `plot_finetune_training_curves.py` | Creates per-run fine-tuning training curves and family-level grids. |
| `plot_finetune_training_curves_protocol_2_5_seed.py` | Historical plotting helper for the Colab 5-seed Protocol 2 fine-tuning bundle. |
| `plot_finetune_training_curves_candidate_hyperparameters.py` | Draft/legacy fine-tuning training-curve plotter. |
| `plot_from_scratch_training_curves.py` | Creates per-run from-scratch training curves and family-level grids. |
| `plot_finetune_vs_from_scratch_comparison.py` | Creates 5-seed fine-tuning mean/std vs 5-seed from-scratch mean/std comparison plots. |

## Dataset, Config, And Utility Scripts

These support setup, diagnostics, and reproducibility.

| Script | Purpose |
| --- | --- |
| `prepare_baseline_finetune_dataset.py` | Prepares the baseline fine-tuning dataset layout. |
| `write_baseline_alignn_config.py` | Writes ALIGNN config files for early baseline experiments. |
| `inspect_alignn_model.py` | Inspects the pretrained ALIGNN model structure. |
| `save_pretrained_alignn_state_dict.py` | Extracts/saves pretrained ALIGNN state-dict information. |
| `Generate_Workspace_Inventory.py` | Creates an inventory of workspace artifacts. |
| `organize_protocol_reports.py` | One-time helper that reorganized report outputs into `Protocol 1/2/3` folders. |

## Recommended Usage Pattern

For new training on Colab, the safest order is:

1. Run the matching `Preflight_*` script.
2. Run the matching `Run_*_Pipeline.sh` wrapper.
3. Run the matching `Check_*_Status.sh` script.
4. Regenerate reports only if new results were added.

For local report work, prefer the `Generate_*`, `Summarize_*`, and `Plot_*` scripts rather than rerunning model training.
