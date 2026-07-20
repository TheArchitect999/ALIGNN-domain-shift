# Pretrained ALIGNN Bundle

This directory contains the pretrained ALIGNN assets that the baseline and finetune
training scripts expect when they initialize from the shared baseline checkpoint.

## Files

- `checkpoint_300.pt`: pretrained model weights used as the starting point for
  zero-shot evaluation and fine-tuning experiments.
- `config.json`: model/config metadata paired with the checkpoint so the code can
  reconstruct the same architecture before loading the weights.

## Why This Directory Is Tracked

The checkpoint is part of the practical handoff between collaborators. A fresh
clone should include the same pretrained starting point rather than requiring a
manual side transfer.

For broader workspace context, see `readme.md` and
`docs/WORKSPACE_ARTIFACT_GUIDE.md`.
