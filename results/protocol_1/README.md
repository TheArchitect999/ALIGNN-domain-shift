# Results Protocol 1

Canonical namespace for the canonical hyperparameter setting.

Hyperparameters:

- `epochs = 50`
- `batch_size = 16`
- `learning_rate = 0.0001`

Contents:

- `oxide/` and `nitride/`: family-specific run trees.
- `N*_seed*/dataset_root/`: fixed run split materialized for ALIGNN.
- `N*_seed*/finetune_last2/`: partial fine-tuning runs using `model.eval()` with `fc.train()` and `gcn_layers[3].train()`.
- `N*_seed*/train_alignn_from_scratch/`: from-scratch comparison runs for `N=50` and `N=500`.

Expected coverage:

- Fine-tuning: 60 runs (`2` families x `6` N values x `5` seeds).
- From-scratch: 20 runs (`2` families x `2` N values x `5` seeds).

Primary report bundle: `results/summaries/protocol_1/`.

Zero-shot baseline note:

Zero-shot summaries are not duplicated in this namespace. Use
`results/zero_shot/zero_shot_summary.csv` for the canonical zero-shot MAE table
and `results/zero_shot/{oxide,nitride}/predictions.csv` for
the full prediction outputs.

This is the canonical error-linked namespace for the embedding-analysis phase because it matches the canonical hyperparameter setting most closely.
