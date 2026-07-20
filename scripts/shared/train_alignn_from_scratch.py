from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from jarvis.core.atoms import Atoms
from sklearn.metrics import mean_absolute_error

from alignn.config import TrainingConfig
from alignn.lmdb_dataset import get_torch_dataset
from alignn.models.alignn import ALIGNN, ALIGNNConfig


def coerce_target_shape(
    prediction: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if prediction.ndim == 0:
        prediction = prediction.view(1)
    if target.ndim == 0:
        target = target.view(1)
    if prediction.ndim == 2 and prediction.shape[-1] == 1 and target.ndim == 1:
        target = target.view(-1, 1)
    if prediction.ndim == 1 and target.ndim == 2 and target.shape[-1] == 1:
        prediction = prediction.view(-1, 1)
    return prediction, target


def extract_jid(file_id: str) -> str:
    name = Path(file_id).name
    if name.startswith("POSCAR-") and name.endswith(".vasp"):
        return name[len("POSCAR-") : -len(".vasp")]
    return Path(name).stem


def load_dataset_rows(dataset_root: Path) -> list[dict]:
    id_prop = dataset_root / "id_prop.csv"
    rows: list[dict] = []
    with id_prop.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for idx, row in enumerate(reader):
            if not row:
                continue
            if idx == 0:
                try:
                    float(row[1])
                except (IndexError, ValueError):
                    continue
            if len(row) < 2:
                raise ValueError(f"Expected at least 2 columns in {id_prop}, got: {row}")
            file_id = row[0].strip()
            target = float(row[1])
            atoms = Atoms.from_poscar(str(dataset_root / file_id))
            rows.append(
                {
                    "jid": extract_jid(file_id),
                    "file_id": file_id,
                    "target": target,
                    "atoms": atoms.to_dict(),
                }
            )
    return rows


def split_dataset(rows: list[dict], cfg: TrainingConfig) -> tuple[list[dict], list[dict], list[dict]]:
    expected = int(cfg.n_train) + int(cfg.n_val) + int(cfg.n_test)
    if len(rows) != expected:
        raise ValueError(
            f"Dataset size mismatch: expected {expected} rows from config, found {len(rows)}."
        )
    train_end = int(cfg.n_train)
    val_end = train_end + int(cfg.n_val)
    train_rows = rows[:train_end]
    val_rows = rows[train_end:val_end]
    test_rows = rows[val_end:]
    return train_rows, val_rows, test_rows


def write_split_metadata(
    output_dir: Path,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = {
        "id_train": [row["jid"] for row in train_rows],
        "id_val": [row["jid"] for row in val_rows],
        "id_test": [row["jid"] for row in test_rows],
    }
    (output_dir / "ids_train_val_test.json").write_text(json.dumps(ids, indent=2))

    all_targets = np.array([row["target"] for row in train_rows + val_rows + test_rows], dtype=float)
    mad = np.mean(np.abs(all_targets - np.mean(all_targets)))
    mad_text = "\n".join(
        [
            f"MAX val:{np.max(all_targets)}",
            f"MIN val:{np.min(all_targets)}",
            f"MAD val:{mad}",
        ]
    )
    (output_dir / "mad").write_text(mad_text + "\n")


def build_loaders(
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    cfg: TrainingConfig,
    output_dir: Path,
) -> tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    line_graph = bool(cfg.compute_line_graph)
    common = {
        "id_tag": cfg.id_tag,
        "target": cfg.target,
        "atom_features": cfg.atom_features,
        "neighbor_strategy": cfg.neighbor_strategy,
        "use_canonize": cfg.use_canonize,
        "name": "user_data",
        "line_graph": line_graph,
        "cutoff": cfg.cutoff,
        "cutoff_extra": cfg.cutoff_extra,
        "max_neighbors": cfg.max_neighbors,
        "classification": cfg.classification_threshold is not None,
        "output_dir": str(output_dir),
        "dtype": cfg.dtype,
    }
    train_data = get_torch_dataset(
        dataset=train_rows,
        tmp_name=str(output_dir / f"{cfg.filename}train_data"),
        **common,
    )
    val_data = get_torch_dataset(
        dataset=val_rows,
        tmp_name=str(output_dir / f"{cfg.filename}val_data"),
        **common,
    )
    test_data = get_torch_dataset(
        dataset=test_rows,
        tmp_name=str(output_dir / f"{cfg.filename}test_data"),
        **common,
    )

    collate_fn = train_data.collate_line_graph if line_graph else train_data.collate
    train_loader = GraphDataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    val_loader = GraphDataLoader(
        val_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    test_loader = GraphDataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )
    return train_loader, val_loader, test_loader


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward_graph_batch(
    model: torch.nn.Module,
    batch,
    device: torch.device,
    use_line_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_line_graph:
        graph, line_graph, lattice, target = batch
        outputs = model([graph.to(device), line_graph.to(device), lattice.to(device)])
    else:
        graph, lattice, target = batch
        outputs = model([graph.to(device), lattice.to(device)])
    prediction = outputs["out"] if isinstance(outputs, dict) else outputs
    prediction, target = coerce_target_shape(prediction, target.to(device))
    return prediction, target


def evaluate_loader(
    model: torch.nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    use_line_graph: bool,
) -> tuple[float, list[float], list[float], list[str]]:
    criterion = torch.nn.L1Loss()
    total_loss = 0.0
    count = 0
    ids: list[str] = []
    targets: list[float] = []
    predictions: list[float] = []

    model.eval()
    cursor = 0
    with torch.no_grad():
        for batch in loader:
            prediction, target = forward_graph_batch(model, batch, device, use_line_graph)
            loss = criterion(prediction, target)
            total_loss += float(loss.item())
            count += 1

            flat_pred = prediction.detach().cpu().numpy().reshape(-1).tolist()
            flat_target = target.detach().cpu().numpy().reshape(-1).tolist()
            predictions.extend(float(x) for x in flat_pred)
            targets.extend(float(x) for x in flat_target)
            batch_size = len(flat_pred)
            if hasattr(loader.dataset, "ids"):
                ids.extend(loader.dataset.ids[cursor : cursor + batch_size])
            cursor += batch_size

    mean_loss = total_loss / max(count, 1)
    return mean_loss, targets, predictions, ids


def write_prediction_csv(path: Path, ids: list[str], targets: list[float], predictions: list[float]) -> float:
    if len(ids) != len(targets):
        ids = [str(i) for i in range(len(targets))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target", "prediction"])
        for jid, target, prediction in zip(ids, targets, predictions):
            writer.writerow([jid, target, prediction])
    return float(mean_absolute_error(np.array(targets), np.array(predictions)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    cfg = TrainingConfig(**json.loads(Path(args.config).read_text()))
    output_dir = Path(args.output_dir).resolve()
    dataset_root = Path(args.dataset_root).resolve()

    output_dir.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(output_dir)
    (output_dir / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))

    set_random_seed(int(cfg.random_seed))
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")
    started_at = time.time()

    rows = load_dataset_rows(dataset_root)
    train_rows, val_rows, test_rows = split_dataset(rows, cfg)
    write_split_metadata(output_dir, train_rows, val_rows, test_rows)
    train_loader, val_loader, test_loader = build_loaders(train_rows, val_rows, test_rows, cfg, output_dir)

    model = ALIGNN(ALIGNNConfig(**cfg.model.model_dump()))
    model.to(device)

    total = sum(param.numel() for param in model.parameters())
    meta = {
        "dataset_root": str(dataset_root),
        "n_total_params": total,
        "train_protocol": "from_scratch",
    }
    (output_dir / "train_from_scratch_meta.json").write_text(json.dumps(meta, indent=2))

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
    )
    criterion = torch.nn.L1Loss()
    use_line_graph = bool(cfg.compute_line_graph)

    history_train: list[list[float]] = []
    history_val: list[list[float]] = []
    best_val = float("inf")

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for batch, _jid in zip(train_loader, train_loader.dataset.ids):
            optimizer.zero_grad()
            prediction, target = forward_graph_batch(model, batch, device, use_line_graph)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            batch_count += 1

        train_loss = running_loss / max(batch_count, 1)
        val_loss, _, _, _ = evaluate_loader(model, val_loader, device, use_line_graph)
        history_train.append([train_loss, train_loss, 0.0, 0.0, 0.0, 0.0])
        history_val.append([val_loss, val_loss, 0.0, 0.0, 0.0, 0.0])

        (output_dir / "history_train.json").write_text(json.dumps(history_train, indent=2))
        (output_dir / "history_val.json").write_text(json.dumps(history_val, indent=2))
        torch.save(model.state_dict(), output_dir / "current_model.pt")
        if val_loss <= best_val:
            best_val = val_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
        print(json.dumps({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss}))

    torch.save(model.state_dict(), output_dir / "last_model.pt")

    best_model = ALIGNN(ALIGNNConfig(**cfg.model.model_dump()))
    best_model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location="cpu"))
    for parameter in best_model.parameters():
        parameter.requires_grad = False
    best_model.to(device)

    test_loss, test_targets, test_predictions, test_ids = evaluate_loader(
        best_model, test_loader, device, use_line_graph
    )
    test_mae = write_prediction_csv(
        output_dir / "prediction_results_test_set.csv",
        test_ids,
        test_targets,
        test_predictions,
    )
    write_prediction_csv(
        output_dir / "pred_data_test_set.csv",
        test_ids,
        test_targets,
        test_predictions,
    )

    _, train_targets, train_predictions, train_ids = evaluate_loader(
        best_model, train_loader, device, use_line_graph
    )
    train_mae = write_prediction_csv(
        output_dir / "prediction_results_train_set.csv",
        train_ids,
        train_targets,
        train_predictions,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "device": str(device),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "epochs": cfg.epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "train_protocol": "from_scratch",
        "train_mae_eV_per_atom": train_mae,
        "val_best_l1": best_val,
        "test_l1": test_loss,
        "test_mae_eV_per_atom": test_mae,
        "prediction_csv": str((output_dir / "prediction_results_test_set.csv").resolve()),
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
