# Hosted-notebook runbook

This compact runbook is for a fresh GPU-backed hosted notebook. It uses the same public commands as a local Linux environment and does not push changes or require remote GitHub write access. The commands do write downloaded and reproduced artifacts into the notebook filesystem.

## Setup

```bash
git clone https://github.com/TheArchitect999/ALIGNN-domain-shift.git
cd ALIGNN-domain-shift
python -m pip install alignn==2025.4.1 jarvis-tools==2026.3.10 numpy==1.26.4 pandas==2.3.3 scikit-learn==1.7.2
```

Choose storage before downloading the checkpoint or dataset. For an ephemeral session:

```bash
export ALIGNN_CKPT_DIR="$PWD/models/pretrained"
export JARVIS_CACHE_DIR="$PWD/cache/jarvis"
```

For a long session, mount persistent storage and use it instead. For example, after mounting Google Drive:

```bash
export PERSIST_ROOT="/content/drive/MyDrive/alignn-domain-shift-cache"
export ALIGNN_CKPT_DIR="$PERSIST_ROOT/models/pretrained"
export JARVIS_CACHE_DIR="$PERSIST_ROOT/jarvis"
```

Then fetch the checkpoint and rebuild the family datasets:

```bash
mkdir -p "$ALIGNN_CKPT_DIR" "$JARVIS_CACHE_DIR"
python scripts/setup/fetch_pretrained.py
python scripts/setup/rebuild_family_datasets.py --cache-dir "$JARVIS_CACHE_DIR"
```

## One smoke run

```bash
python scripts/shared/prepare_baseline_finetune_dataset.py \
  --family oxide --N 50 --seed 0 --repo-root . \
  --results-root results/reproduction/protocol_1 --link-mode symlink

python scripts/shared/fine_tune_last_two_alignn_layers.py \
  --config configs/protocol_1/finetune/oxide_finetune_N50_seed0.finetune_last2.json \
  --output-dir results/reproduction/protocol_1/oxide/N50_seed0/finetune_last2 \
  --dataset-root results/reproduction/protocol_1/oxide/N50_seed0/dataset_root \
  --pretrained-checkpoint "$ALIGNN_CKPT_DIR/checkpoint_300.pt" \
  --pretrained-config configs/pretrained/config.json --device cuda
```

The reference run is `results/protocol_1/oxide/N50_seed0/finetune_last2/`. Compare the reproduced `summary.json` and test-prediction CSV before scheduling larger matrices.

## Session safety

- Confirm a CUDA device is visible before training.
- Keep downloaded data and regenerated checkpoints outside Git.
- Do not place credentials in notebooks or shell history.
- Download results explicitly at session end; this runbook performs no automatic commits or pushes.
