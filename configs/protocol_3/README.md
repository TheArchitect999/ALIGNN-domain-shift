# `configs/protocol_3`

Canonical configs for `results/protocol_3`, the corrected low-
learning-rate experiment namespace.

- `finetune/`: configs matching `results/protocol_3/*/*/finetune_last2_epochs100_bs32_lr5e5/`
- `from_scratch/`: configs matching `results/protocol_3/*/*/train_alignn_from_scratch_epochs100_bs32_lr5e5/`

These configs correspond to:

- `epochs = 100`
- `batch_size = 32`
- `learning_rate = 0.00005`

The result subdirectories keep the `epochs100_bs32_lr5e5` suffix for provenance,
but the config folders use the same clean workflow names as Protocol 1 and 2.
