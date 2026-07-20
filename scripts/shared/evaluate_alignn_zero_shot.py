from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["DGLBACKEND"] = "pytorch"
import argparse, csv, json
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error
from jarvis.core.atoms import Atoms
from alignn.pretrained import get_prediction
def coerce_prediction_to_float(x):
    import numpy as np

    # torch tensor
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    # list / tuple
    if isinstance(x, (list, tuple)):
        if len(x) == 1:
            x = x[0]
        else:
            raise ValueError(f"Expected one prediction, got sequence: {x}")

    # numpy scalar / array
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar prediction, got object with shape {arr.shape}: {x}")

    return float(arr.reshape(-1)[0])
def load_rows(manifest_csv: Path):
    with manifest_csv.open() as f:
        return list(csv.DictReader(f))

def update_zero_shot_summary(summary_csv: Path, row: dict) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["family", "model_name", "n_test", "mae_eV_per_atom", "predictions_csv"]
    rows = []
    if summary_csv.exists():
        with summary_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [existing for existing in rows if existing["family"] != row["family"]]
    rows.append(row)
    rows.sort(key=lambda item: item["family"])
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', choices=['oxide','nitride'], required=True)
    ap.add_argument('--repo-root', default='.')
    ap.add_argument('--model-name', default='configs/pretrained')
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()
    repo = Path(args.repo_root).resolve()
    family_dir = repo / 'data' / args.family
    manifest = family_dir / 'manifests' / 'test.csv'
    structures = family_dir / 'structures'
    outdir = Path(args.outdir) if args.outdir else repo / "results/protocol_1" / args.family / 'zero_shot'
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(manifest)
    results = []
    y_true, y_pred = [], []
    for i, row in enumerate(rows, 1):
        atoms = Atoms.from_poscar(str(structures / row['filename']))
        raw_pred = get_prediction(model_name=args.model_name, atoms=atoms)
        print("DEBUG raw_pred type:", type(raw_pred), "value:", raw_pred)
        pred = coerce_prediction_to_float(raw_pred)
        target = float(row['target'])
        results.append({
            'jid': row['jid'], 'filename': row['filename'], 'target': target,
            'prediction': pred, 'abs_error': abs(pred-target)
        })
        y_true.append(target); y_pred.append(pred)
        if i % 100 == 0:
            print(f'processed {i}/{len(rows)}')
    mae = float(mean_absolute_error(y_true, y_pred))
    pred_csv = outdir / 'predictions.csv'
    with pred_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    summary = {
        'family': args.family,
        'model_name': args.model_name,
        'n_test': len(rows),
        'mae_eV_per_atom': mae,
        'predictions_csv': os.path.relpath(pred_csv.resolve(), repo)
    }
    update_zero_shot_summary(repo / "results" / "zero_shot" / "zero_shot_summary.csv", summary)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
