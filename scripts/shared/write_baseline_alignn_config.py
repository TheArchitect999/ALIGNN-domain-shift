from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_model_config(pretrained_config: Path | None) -> dict:
    if pretrained_config is None:
        return {
            "name": "alignn",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "output_features": 1,
        }
    raw = json.loads(pretrained_config.read_text())
    return dict(raw["model"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-train", type=int, required=True)
    parser.add_argument("--n-val", type=int, required=True)
    parser.add_argument("--n-test", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pretrained-config", default=None)
    args = parser.parse_args()

    from alignn.config import TrainingConfig
    from alignn.models.alignn import ALIGNNConfig

    pretrained_config = (
        Path(args.pretrained_config).resolve()
        if args.pretrained_config is not None
        else None
    )

    cfg = TrainingConfig()
    cfg.dataset = "user_data"
    cfg.target = "target"
    cfg.filename = "id_prop.csv"
    cfg.random_seed = args.seed
    cfg.keep_data_order = True
    cfg.n_train = args.n_train
    cfg.n_val = args.n_val
    cfg.n_test = args.n_test
    cfg.batch_size = args.batch_size
    cfg.epochs = args.epochs
    cfg.learning_rate = args.lr
    cfg.num_workers = 0
    cfg.pin_memory = False
    cfg.use_lmdb = True
    cfg.output_dir = "PLACEHOLDER_SET_BY_WRAPPER"
    cfg.model = ALIGNNConfig(**load_model_config(pretrained_config))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg.model_dump(), indent=2))
    print(out)


if __name__ == "__main__":
    main()
