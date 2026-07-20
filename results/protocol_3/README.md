# Results Protocol 3

Canonical namespace for the low-learning-rate hyperparameter setting split out from the old `results/` tree.

Hyperparameters:

- `epochs = 100`
- `batch_size = 32`
- `learning_rate = 0.00005`

Contents:

- `oxide/` and `nitride/`: family-specific run trees.
- `N*_seed*/finetune_last2_epochs100_bs32_lr5e5/`: partial fine-tuning runs using `model.eval()` with `fc.train()` and `gcn_layers[3].train()`.
- `N*_seed*/train_alignn_from_scratch_epochs100_bs32_lr5e5/`: from-scratch comparison runs for `N=50` and `N=500`.

Expected coverage:

- Fine-tuning: 60 runs (`2` families x `6` N values x `5` seeds).
- From-scratch: 20 runs (`2` families x `2` N values x `5` seeds).

Primary report bundle: `results/summaries/protocol_3/`.

Zero-shot baseline note:

Zero-shot summaries are not duplicated in this namespace. Use
`results/zero_shot/zero_shot_summary.csv` for the canonical zero-shot MAE table
and `results/zero_shot/{oxide,nitride}/predictions.csv` for
the full prediction outputs.

Dataset-root note:

Protocol 3 uses the same deterministic split definitions as the other protocols without duplicating run-local `dataset_root/` folders here. Recreate those inputs with `scripts/setup/rebuild_family_datasets.py`, then use the Protocol 3 runner scripts for a replay. All maintained references use the public, repository-relative paths documented in `docs/REPRODUCING.md`.
