from __future__ import annotations
import argparse, csv, json, os, shutil
from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def copy_or_link(src: Path, dst: Path, mode: str='copy'):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == 'hardlink':
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    if mode == 'symlink':
        try:
            dst.symlink_to(src.resolve())
            return
        except Exception:
            pass
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['oxide','nitride'], required=True)
    ap.add_argument('--N', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--results-root', default='results/protocol_1')
    ap.add_argument('--link-mode', choices=['copy','hardlink','symlink'], default='copy')
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    family_dir = repo / 'data' / args.family
    pool = load_csv(family_dir / 'manifests' / 'pool.csv')
    test = load_csv(family_dir / 'manifests' / 'test.csv')
    if args.N > len(pool):
        raise ValueError(f'N={args.N} exceeds pool size {len(pool)}')
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(pool))[:args.N]
    subset = pool.iloc[idx].copy().reset_index(drop=True)
    n_val = max(5, int(round(0.1 * args.N)))
    if n_val >= args.N:
        n_val = max(1, args.N // 5)
    val = subset.iloc[:n_val].copy().reset_index(drop=True)
    train = subset.iloc[n_val:].copy().reset_index(drop=True)
    root = repo / args.results_root / args.family / f'N{args.N}_seed{args.seed}' / 'dataset_root'
    root.mkdir(parents=True, exist_ok=True)
    structures_src = family_dir / 'structures'
    # exact order matters because keep_data_order=True and n_train/n_val/n_test are used
    ordered_rows = []
    for split_name, frame in [('train', train), ('val', val), ('test', test)]:
        for _, row in frame.iterrows():
            filename = row['filename']
            copy_or_link(structures_src / filename, root / filename, mode=args.link_mode)
            ordered_rows.append([filename, float(row['target'])])
    with (root / 'id_prop.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerows(ordered_rows)
    split_manifest = {
        'family': args.family,
        'N': args.N,
        'seed': args.seed,
        'n_train': int(len(train)),
        'n_val': int(len(val)),
        'n_test': int(len(test)),
        'train_jids': train['jid'].tolist(),
        'val_jids': val['jid'].tolist(),
        'test_jids': test['jid'].tolist(),
    }
    (root / 'split_manifest.json').write_text(json.dumps(split_manifest, indent=2))
    print(json.dumps({'dataset_root': str(root), **{k: split_manifest[k] for k in ['family','N','seed','n_train','n_val','n_test']}}, indent=2))

if __name__ == '__main__':
    main()
