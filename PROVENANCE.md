# Provenance

This repository is a curated public research release. To keep it browsable and
clonable, bulk and process artifacts were externalized or archived; this file
records where everything came from and how to recover or regenerate it.

## Full-history archive of record

The complete working repository — full git history and all bulk artifacts — is
preserved offline as the archive of record.

- Archived at commit **`026d01bec89fefbcc6fe84c3748aa5727383d9ec`**
  (tag `pre-restructure-archive-2026-07-20`).
- The archive holds the checksum-valid record: every `.sha256` evidence manifest
  in the project history verifies against it. The public tree pins this commit
  rather than trying to keep hundreds of path-specific checksums valid across the
  restructure.

## What is in this public tree

| Area | Contents |
|---|---|
| `paper/` | Final figures, the supplementary materials (S1-S13), and `evidence_manifest.csv` (number -> public source) |
| `results/` | Per-protocol run outputs (histories, summaries, test predictions), zero-shot predictions, aggregate summaries incl. the A1-A6 effect-size tables, and frozen embeddings |
| `configs/` | Per-protocol training configurations + the pretrained checkpoint config |
| `data/` | Split manifests, family definitions, and integrity diagnostics (structure files are regenerated, not shipped) |
| `scripts/` | Dataset, training, analysis, figure, and setup/fetch code |
| `docs/` | Reproduction and results guides |

## Externalized artifacts (Zenodo)

Reserved DOI **10.5281/zenodo.21398774** (record publication is deferred until
the public research release):

- `checkpoint_300.pt` - mirror of the public JARVIS-DFT ALIGNN v1 formation-energy
  pretrained checkpoint (Kamal Choudhary, Figshare record 17005681; original terms
  apply). `scripts/setup/fetch_pretrained.py` uses the official Figshare archive
  first and the deposited copy as a later mirror; both are SHA-256 verified.
- `dataset_structures.tar.gz` - dataset structure bundle.

## Regenerable (not shipped, not deposited)

- **Per-structure `.vasp` datasets:** the public rebuild materializes 17,279 unique
  family structures. The former working tree contained roughly 246,000 run-local
  copies and links of those structures; that duplicate-heavy layout is not shipped.
  Rebuild from JARVIS-DFT using the fixed manifests in `data/manifests/` (see
  `docs/REPRODUCING.md`).
- **Fine-tuned / from-scratch model checkpoints** (720 files, ~11 GB): regenerable
  from the committed configs, fixed seeds, and training scripts. A reproducibility
  rerun of Protocol 1 verified agreement with the canonical numerical results
  (archived provenance).

## Canonical numerical authorities

- `results/zero_shot/{oxide,nitride}/predictions.csv` - per-structure zero-shot errors.
- `results/summaries/a1..a6*.csv` - validated adaptation/effect-size analyses.
- `paper/supplementary/data/*.csv` - full supplementary tables.

## Path remapping

Directory names were made venue-neutral and space-free. Numbered optimization
variants are consistently named `protocol_N`; calendar-based experiment labels
were replaced by functional names such as `finetune`, `from_scratch`, and
`embedding_analysis`; supplementary materials moved under `paper/supplementary/`.
The complete source-to-public file map is `restructure_manifest.csv`, retained in
the archive alongside this release.

### Script rename map

| Archived basename or role | Public path |
|---|---|
| `generate_figures_r1.py` | `scripts/figures/generate_paper_figures.py` |
| `produce_revision_analyses.py` | `scripts/analysis/compute_effect_size_analyses.py` |
| `validate_revision_analyses.py` | `scripts/analysis/validate_effect_size_analyses.py` |
| `restructure_supplementary.py` | `scripts/analysis/build_supplementary.py` (historical transformation record; its frozen baseline remains in the archive) |
| `validate_supplementary_restructure.py` | `scripts/analysis/validate_supplementary.py` (rewritten as a clean-checkout package validator) |
| Legacy family-data builder, library, and validator | `scripts/dataset/{build_family_datasets.py,family_dataset_lib.py,validate_family_datasets.py}` |
| Experiment scripts containing calendar or numbered-variant labels | Lowercase snake-case basenames using `baseline` or `protocol_N` |

Submission-document and highlighted-review builders are archive-only and have no
public destination. Public run metadata uses repository-relative paths, and retained
analysis scripts write regenerated audit intermediates under
`results/derived_evidence/`. Personal workstation paths and the former submission
workspace namespace remain only in the full-history archive.
