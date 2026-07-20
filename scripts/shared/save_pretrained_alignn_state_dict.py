from __future__ import annotations
import argparse, json, torch
from pathlib import Path
from alignn.pretrained import get_figshare_model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-name', default='configs/pretrained')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    result = get_figshare_model(args.model_name)
    model, config = result if isinstance(result, tuple) else (result, None)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    meta = {'model_name': args.model_name, 'state_dict_path': str(out), 'config_returned': config is not None}
    (out.parent / (out.stem + '_meta.json')).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

if __name__ == '__main__':
    main()
