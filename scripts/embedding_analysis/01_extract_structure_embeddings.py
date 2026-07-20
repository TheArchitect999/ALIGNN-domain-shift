#!/usr/bin/env python
"""Extract structure-level embeddings from the local pretrained ALIGNN model.

This script is intentionally inference-only. It loads the checkpoint/config
already present in the repository, registers explicit forward hooks on the
ALIGNN model, and writes new embedding-analysis artifacts without touching
existing result folders.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_CHECKPOINT = "models/pretrained/checkpoint_300.pt"
DEFAULT_CONFIG = "configs/pretrained/config.json"
DEFAULT_OUTPUT_DIR = "results/embeddings"
DEFAULT_FAMILIES = ("oxide", "nitride")
DEFAULT_SOURCES = ("pre_head", "last_alignn_pool", "last_gcn_pool")


@dataclass(frozen=True)
class HookSpec:
    """Description of one embedding source."""

    name: str
    layer_name: str
    module_path: str
    description: str


@dataclass
class Runtime:
    """Lazy imports required only when ALIGNN/DGL execution is requested."""

    torch: Any
    TrainingConfig: Any
    get_torch_dataset: Any
    ALIGNN: Any
    ALIGNNConfig: Any
    GraphDataLoader: Any
    AvgPooling: Any
    Atoms: Any


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract fixed-split structure embeddings from the local pretrained "
            "ALIGNN formation-energy model."
        )
    )
    parser.add_argument(
        "--dataset-subset",
        choices=("test", "pool", "train", "val"),
        default="test",
        help="Dataset subset to extract. Start with the fixed test set.",
    )
    parser.add_argument(
        "--subset-metadata",
        default=None,
        help=(
            "Optional subset metadata CSV from results/embeddings/subsets. "
            "When set, rows are extracted in this file order instead of by family manifest."
        ),
    )
    parser.add_argument(
        "--subset-name",
        default=None,
        help="Subset name used in manifests when --subset-metadata is provided.",
    )
    parser.add_argument(
        "--embedding-folder-name",
        default=None,
        help=(
            "Embedding folder/manifest stem for --subset-metadata mode. "
            "Defaults to a stable name derived from --subset-name."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Embedding-analysis output root.",
    )
    parser.add_argument(
        "--hook-selection",
        choices=("all", "pre_head", "last_alignn_pool", "last_gcn_pool"),
        default="all",
        help="Embedding source to extract, or all supported sources.",
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=DEFAULT_FAMILIES,
        default=list(DEFAULT_FAMILIES),
        help="Families to process.",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="Repo-local pretrained ALIGNN checkpoint.",
    )
    parser.add_argument(
        "--pretrained-config",
        default=DEFAULT_CONFIG,
        help="Repo-local pretrained ALIGNN config.",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DGL dataloader workers. Default 0 keeps local runs predictable.",
    )
    parser.add_argument(
        "--limit-per-family",
        type=int,
        default=None,
        help="Optional development limit. Omit for the full subset.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect paths, model hooks, counts, and planned outputs without graph extraction.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Alias for --dry-run.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacement of existing embedding-analysis output files.",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild additive LMDB caches under the embedding-analysis output root.",
    )
    return parser.parse_args()


def selected_sources(selection: str) -> list[str]:
    if selection == "all":
        return list(DEFAULT_SOURCES)
    return [selection]


def import_runtime(output_dir: Path) -> Runtime:
    """Import ALIGNN/DGL after setting local writable cache locations."""

    matplotlib_cache = output_dir / "cache" / "matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    try:
        import torch
        from alignn.config import TrainingConfig
        from alignn.lmdb_dataset import get_torch_dataset
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
        from dgl.dataloading import GraphDataLoader
        from dgl.nn import AvgPooling
        from jarvis.core.atoms import Atoms
    except Exception as exc:  # pragma: no cover - environment-specific branch
        raise RuntimeError(
            "Failed to import ALIGNN/DGL runtime. Use the local training "
            "environment pinned by environment/requirements-frozen.txt "
            "(for example /path/to/python/environment/bin/python). "
            f"Original error: {exc}"
        ) from exc

    return Runtime(
        torch=torch,
        TrainingConfig=TrainingConfig,
        get_torch_dataset=get_torch_dataset,
        ALIGNN=ALIGNN,
        ALIGNNConfig=ALIGNNConfig,
        GraphDataLoader=GraphDataLoader,
        AvgPooling=AvgPooling,
        Atoms=Atoms,
    )


def set_random_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_training_config(runtime: Runtime, config_path: Path) -> tuple[Any, dict[str, Any]]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    cfg = runtime.TrainingConfig(**payload)
    return cfg, payload


def load_pretrained_model(runtime: Runtime, checkpoint_path: Path, config_path: Path) -> Any:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    model = runtime.ALIGNN(runtime.ALIGNNConfig(**payload["model"]))
    state = runtime.torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(model_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def resolve_hook_specs(model: Any, source_names: list[str]) -> dict[str, HookSpec]:
    """Resolve hook points and fail loudly when the model lacks one."""

    specs: dict[str, HookSpec] = {}

    if "pre_head" in source_names:
        if not hasattr(model, "readout"):
            raise RuntimeError("Requested pre_head, but model.readout does not exist.")
        if not hasattr(model, "fc"):
            raise RuntimeError("Requested pre_head, but model.fc regression head does not exist.")
        if int(getattr(model.config, "extra_features", 0) or 0) != 0:
            raise RuntimeError(
                "pre_head hook is defined for this project model with extra_features=0; "
                "the loaded config has extra_features != 0."
            )
        specs["pre_head"] = HookSpec(
            name="pre_head",
            layer_name="readout",
            module_path="model.readout",
            description=(
                "Avg-pooled crystal-graph node representation produced by "
                "model.readout immediately before model.fc."
            ),
        )

    if "last_alignn_pool" in source_names:
        if not hasattr(model, "alignn_layers") or len(model.alignn_layers) == 0:
            raise RuntimeError(
                "Requested last_alignn_pool, but model.alignn_layers is missing or empty."
            )
        index = len(model.alignn_layers) - 1
        specs["last_alignn_pool"] = HookSpec(
            name="last_alignn_pool",
            layer_name=f"alignn_layers.{index}",
            module_path=f"model.alignn_layers[{index}]",
            description=(
                "Avg-pooled node tensor x returned by the last ALIGNNConv block, "
                "before the gated GCN stack."
            ),
        )

    if "last_gcn_pool" in source_names:
        if not hasattr(model, "gcn_layers") or len(model.gcn_layers) == 0:
            raise RuntimeError(
                "Requested last_gcn_pool, but model.gcn_layers is missing or empty."
            )
        index = len(model.gcn_layers) - 1
        specs["last_gcn_pool"] = HookSpec(
            name="last_gcn_pool",
            layer_name=f"gcn_layers.{index}",
            module_path=f"model.gcn_layers[{index}]",
            description=(
                "Avg-pooled node tensor x returned by the last EdgeGatedGraphConv "
                "GCN block. In this config it should match pre_head."
            ),
        )

    missing = sorted(set(source_names) - set(specs))
    if missing:
        raise RuntimeError(f"Could not resolve requested hook sources: {missing}")
    return specs


def register_hooks(model: Any, specs: dict[str, HookSpec], capture: dict[str, Any]) -> list[Any]:
    handles: list[Any] = []

    if "pre_head" in specs:
        def readout_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            capture["pre_head"] = output.detach()

        handles.append(model.readout.register_forward_hook(readout_hook))

    if "last_alignn_pool" in specs:
        alignn_layer = model.alignn_layers[-1]

        def alignn_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 1:
                raise RuntimeError("last ALIGNN hook expected tuple output (x, y, z).")
            capture["last_alignn_nodes"] = output[0].detach()

        handles.append(alignn_layer.register_forward_hook(alignn_hook))

    if "last_gcn_pool" in specs:
        gcn_layer = model.gcn_layers[-1]

        def gcn_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, tuple) or len(output) < 1:
                raise RuntimeError("last GCN hook expected tuple output (x, y).")
            capture["last_gcn_nodes"] = output[0].detach()

        handles.append(gcn_layer.register_forward_hook(gcn_hook))

    return handles


def load_zero_shot_predictions(root: Path, family: str, subset: str) -> dict[str, dict[str, str]]:
    if subset != "test":
        return {}
    path = root / "results" / "zero_shot" / family / "predictions.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["jid"]: row for row in csv.DictReader(handle)}


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def subset_output_name(args: argparse.Namespace) -> str:
    if args.subset_metadata is None:
        return f"{args.dataset_subset}_set"
    if args.embedding_folder_name:
        return args.embedding_folder_name
    if args.subset_name == "balanced_pool_set":
        return "balanced_pool"
    if args.subset_name:
        return args.subset_name
    return Path(args.subset_metadata).stem


def load_family_rows(
    runtime: Runtime,
    root: Path,
    family: str,
    subset: str,
    target_key: str,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = root / "data" / family / "manifests" / f"{subset}.csv"
    structures_dir = root / "data" / family / "structures"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not structures_dir.exists():
        raise FileNotFoundError(f"Missing structures directory: {structures_dir}")

    zero_shot = load_zero_shot_predictions(root, family, subset)
    dataset_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if limit is not None and len(dataset_rows) >= limit:
                break
            jid = raw.get("jid", "").strip()
            filename = raw.get("filename", "").strip()
            if not jid:
                raise ValueError(f"Row without jid in {manifest_path}: {raw}")
            if not filename:
                raise ValueError(f"Row without filename in {manifest_path}: {raw}")
            structure_path = structures_dir / filename
            if not structure_path.exists():
                raise FileNotFoundError(f"Missing structure for {jid}: {structure_path}")

            target = as_float(raw.get("target"))
            if target is None:
                raise ValueError(f"Missing numeric target for {jid} in {manifest_path}")

            atoms = runtime.Atoms.from_poscar(str(structure_path))
            prediction_row = zero_shot.get(jid, {})
            prediction = as_float(prediction_row.get("prediction"))
            abs_error = as_float(prediction_row.get("abs_error"))
            if abs_error is None and prediction is not None:
                abs_error = abs(prediction - target)

            dataset_rows.append(
                {
                    "jid": jid,
                    target_key: target,
                    "atoms": atoms.to_dict(),
                }
            )
            records.append(
                {
                    "material_id": jid,
                    "family": family,
                    "split": raw.get("split", subset),
                    "formula": raw.get("formula", ""),
                    "target_formation_energy_peratom": target,
                    "pretrained_prediction": prediction,
                    "absolute_error": abs_error,
                    "filename": filename,
                    "structure_path": relpath(structure_path, root),
                    "source_manifest": relpath(manifest_path, root),
                    "n_atoms": raw.get("n_atoms", ""),
                    "is_oxide": raw.get("is_oxide", ""),
                    "is_nitride": raw.get("is_nitride", ""),
                    "is_oxynitride": raw.get("is_oxynitride", ""),
                }
            )

    return dataset_rows, records


def load_subset_rows(
    runtime: Runtime,
    root: Path,
    subset_metadata: Path,
    target_key: str,
    limit: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not subset_metadata.exists():
        raise FileNotFoundError(f"Missing subset metadata: {subset_metadata}")

    dataset_rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    missing_structures: list[str] = []
    missing_targets: list[str] = []
    families: dict[str, int] = {}
    splits: dict[str, int] = {}

    with subset_metadata.open("r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            if limit is not None and len(dataset_rows) >= limit:
                break
            jid = raw.get("material_id") or raw.get("structure_id") or ""
            jid = jid.strip()
            family = raw.get("family", "").strip()
            split = raw.get("split", "").strip()
            filename = raw.get("filename", "").strip()
            structure_text = raw.get("structure_path", "").strip()
            if not jid:
                raise ValueError(f"Subset row without material_id: {raw}")
            if not family:
                raise ValueError(f"Subset row without family for {jid}: {raw}")
            if not filename:
                raise ValueError(f"Subset row without filename for {jid}: {raw}")
            structure_path = (root / structure_text).resolve()
            if not structure_path.exists():
                missing_structures.append(f"{jid}: {structure_text}")
                continue

            target = as_float(raw.get("target_formation_energy_peratom"))
            if target is None:
                missing_targets.append(jid)
                continue

            families[family] = families.get(family, 0) + 1
            splits[split] = splits.get(split, 0) + 1
            atoms = runtime.Atoms.from_poscar(str(structure_path))
            dataset_rows.append(
                {
                    "jid": jid,
                    target_key: target,
                    "atoms": atoms.to_dict(),
                }
            )
            records.append(
                {
                    "material_id": jid,
                    "family": family,
                    "split": split,
                    "formula": raw.get("formula", ""),
                    "target_formation_energy_peratom": target,
                    "pretrained_prediction": None,
                    "absolute_error": None,
                    "filename": filename,
                    "structure_path": relpath(structure_path, root),
                    "source_manifest": raw.get("source_manifest", ""),
                    "n_atoms": raw.get("n_atoms", ""),
                    "is_oxide": raw.get("is_oxide", ""),
                    "is_nitride": raw.get("is_nitride", ""),
                    "is_oxynitride": raw.get("is_oxynitride", ""),
                }
            )

    if missing_structures:
        preview = "\n".join(missing_structures[:10])
        raise FileNotFoundError(
            f"Missing {len(missing_structures)} structure files from {subset_metadata}. "
            f"First records:\n{preview}"
        )
    if missing_targets:
        preview = ", ".join(missing_targets[:10])
        raise ValueError(
            f"Missing numeric targets for {len(missing_targets)} rows in {subset_metadata}. "
            f"First IDs: {preview}"
        )

    diagnostics = {
        "subset_metadata": relpath(subset_metadata, root),
        "loaded_rows": len(records),
        "missing_structure_files": 0,
        "missing_targets": 0,
        "counts_by_family": dict(sorted(families.items())),
        "counts_by_split": dict(sorted(splits.items())),
    }
    return dataset_rows, records, diagnostics


def make_loader(
    runtime: Runtime,
    cfg: Any,
    rows: list[dict[str, Any]],
    cache_dir: Path,
    family: str,
    subset: str,
    batch_size: int,
    num_workers: int,
    limit: int | None,
    rebuild_cache: bool,
) -> Any:
    line_graph = bool(cfg.compute_line_graph)
    cache_suffix = f"limit{limit}" if limit is not None else "full"
    tmp_name = cache_dir / f"{family}_{subset}_{cache_suffix}_lmdb"
    if rebuild_cache and tmp_name.exists():
        shutil.rmtree(tmp_name)

    dataset = runtime.get_torch_dataset(
        dataset=rows,
        id_tag=cfg.id_tag,
        target=cfg.target,
        atom_features=cfg.atom_features,
        neighbor_strategy=cfg.neighbor_strategy,
        use_canonize=cfg.use_canonize,
        name="domain_shift_embedding_analysis",
        line_graph=line_graph,
        cutoff=cfg.cutoff,
        cutoff_extra=cfg.cutoff_extra,
        max_neighbors=cfg.max_neighbors,
        classification=cfg.classification_threshold is not None,
        output_dir=str(cache_dir),
        tmp_name=str(tmp_name),
        read_existing=True,
        dtype=cfg.dtype,
    )
    if len(dataset) != len(rows):
        raise RuntimeError(
            f"Cache row-count mismatch for {tmp_name}: dataset has {len(dataset)}, "
            f"manifest has {len(rows)}. Re-run with --rebuild-cache."
        )

    collate_fn = dataset.collate_line_graph if line_graph else dataset.collate
    return runtime.GraphDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def move_batch_to_device(batch: Any, device: Any, use_line_graph: bool) -> tuple[Any, Any, Any]:
    if use_line_graph:
        graph, line_graph, lattice, target = batch
        graph = graph.to(device)
        line_graph = line_graph.to(device)
        lattice = lattice.to(device)
        target = target.to(device)
        model_input = [graph, line_graph, lattice]
        return graph, target, model_input

    graph, lattice, target = batch
    graph = graph.to(device)
    lattice = lattice.to(device)
    target = target.to(device)
    model_input = [graph, lattice]
    return graph, target, model_input


def flatten_prediction(prediction: Any) -> np.ndarray:
    return prediction.detach().cpu().reshape(-1).numpy().astype(np.float64)


def extract_embeddings(
    runtime: Runtime,
    model: Any,
    loader: Any,
    records: list[dict[str, Any]],
    specs: dict[str, HookSpec],
    device: Any,
    use_line_graph: bool,
) -> tuple[dict[str, np.ndarray], list[float], dict[str, float | None]]:
    model.to(device)
    model.eval()
    pooler = runtime.AvgPooling()
    capture: dict[str, Any] = {}
    handles = register_hooks(model, specs, capture)
    chunks: dict[str, list[np.ndarray]] = {name: [] for name in specs}
    predictions: list[float] = []
    source_diffs: dict[str, float | None] = {"max_abs_pre_head_minus_last_gcn_pool": None}

    cursor = 0
    try:
        with runtime.torch.no_grad():
            for batch in loader:
                capture.clear()
                graph, _target, model_input = move_batch_to_device(batch, device, use_line_graph)
                output = model(model_input)
                batch_predictions = flatten_prediction(output)
                predictions.extend(float(x) for x in batch_predictions)
                batch_size = len(batch_predictions)

                batch_arrays: dict[str, Any] = {}
                if "pre_head" in specs:
                    if "pre_head" not in capture:
                        raise RuntimeError("pre_head hook did not fire.")
                    batch_arrays["pre_head"] = capture["pre_head"]

                if "last_alignn_pool" in specs:
                    if "last_alignn_nodes" not in capture:
                        raise RuntimeError("last ALIGNN hook did not fire.")
                    batch_arrays["last_alignn_pool"] = pooler(graph, capture["last_alignn_nodes"])

                if "last_gcn_pool" in specs:
                    if "last_gcn_nodes" not in capture:
                        raise RuntimeError("last GCN hook did not fire.")
                    batch_arrays["last_gcn_pool"] = pooler(graph, capture["last_gcn_nodes"])

                for source_name, tensor in batch_arrays.items():
                    arr = tensor.detach().cpu().numpy().astype(np.float32)
                    if arr.shape[0] != batch_size:
                        raise RuntimeError(
                            f"{source_name} batch shape mismatch: {arr.shape[0]} "
                            f"embeddings for {batch_size} predictions."
                        )
                    chunks[source_name].append(arr)

                if {"pre_head", "last_gcn_pool"}.issubset(batch_arrays):
                    diff = (
                        batch_arrays["pre_head"] - batch_arrays["last_gcn_pool"]
                    ).detach().abs().max().item()
                    previous = source_diffs["max_abs_pre_head_minus_last_gcn_pool"]
                    source_diffs["max_abs_pre_head_minus_last_gcn_pool"] = (
                        diff if previous is None else max(previous, diff)
                    )

                cursor += batch_size
                if cursor > len(records):
                    raise RuntimeError("Loader produced more rows than metadata records.")
    finally:
        for handle in handles:
            handle.remove()

    if cursor != len(records):
        raise RuntimeError(
            f"Loader produced {cursor} rows, but metadata has {len(records)} rows."
        )

    arrays: dict[str, np.ndarray] = {}
    for source_name, parts in chunks.items():
        if not parts:
            raise RuntimeError(f"No embedding chunks were collected for {source_name}.")
        arrays[source_name] = np.concatenate(parts, axis=0)
    return arrays, predictions, source_diffs


def check_output_targets(paths: list[Path], allow_overwrite: bool) -> None:
    if allow_overwrite:
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing embedding-analysis outputs. "
            "Use --allow-overwrite to replace these files:\n"
            f"{joined}"
        )


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_optional_parquet(csv_rows: list[dict[str, Any]], parquet_path: Path) -> dict[str, Any]:
    try:
        import pandas as pd

        df = pd.DataFrame(csv_rows)
        df.to_parquet(parquet_path, index=False)
        return {"written": True, "path": str(parquet_path)}
    except Exception as exc:
        return {"written": False, "path": str(parquet_path), "reason": str(exc)}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_long_metadata(
    base_records: list[dict[str, Any]],
    specs: dict[str, HookSpec],
    prediction_by_id: dict[str, float],
    arrays: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(base_records):
        material_id = record["material_id"]
        for source_name, spec in specs.items():
            row = dict(record)
            row.update(
                {
                    "pretrained_prediction": prediction_by_id.get(
                        material_id, record.get("pretrained_prediction")
                    ),
                    "embedding_source": source_name,
                    "layer_name": spec.layer_name,
                    "module_path": spec.module_path,
                    "npz_key": source_name,
                    "embedding_index": index,
                    "embedding_dim": int(arrays[source_name].shape[1]),
                }
            )
            target = row.get("target_formation_energy_peratom")
            prediction = row.get("pretrained_prediction")
            if target is not None and prediction is not None:
                row["absolute_error"] = abs(float(prediction) - float(target))
            rows.append(row)
    return rows


def count_by_family(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        family = str(record["family"])
        counts[family] = counts.get(family, 0) + 1
    return counts


def runtime_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("alignn", "dgl", "torch", "jarvis-tools", "numpy", "pandas"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_embedding_dir_readme(
    root: Path,
    embeddings_dir: Path,
    report_summary: dict[str, Any],
    title: str,
    description: str,
) -> None:
    counts = report_summary["counts_by_family"]
    shapes = report_summary["embedding_shapes"]
    source_lines = "\n".join(
        f"- `{name}`: shape `{shape}`" for name, shape in shapes.items()
    )
    family_lines = "\n".join(f"- `{family}`: {count}" for family, count in counts.items())
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    (embeddings_dir / "readme.md").write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                description,
                "",
                "Counts by family:",
                family_lines,
                "",
                "Embedding matrices:",
                source_lines,
                "",
                f"NPZ: `{relpath(Path(report_summary['paths']['npz']), root)}`",
                f"Metadata CSV: `{relpath(Path(report_summary['paths']['metadata_csv']), root)}`",
                f"Manifest JSON: `{relpath(Path(report_summary['paths']['manifest_json']), root)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_readmes(
    root: Path,
    embeddings_dir: Path,
    manifests_dir: Path,
    report_summary: dict[str, Any],
) -> None:
    counts = report_summary["counts_by_family"]
    shapes = report_summary["embedding_shapes"]
    source_lines = "\n".join(
        f"- `{name}`: shape `{shape}`" for name, shape in shapes.items()
    )
    family_lines = "\n".join(f"- `{family}`: {count}" for family, count in counts.items())

    (root / "readme.md").write_text(
        "\n".join(
            [
                "# DOMAIN_SHIFT Embedding Analysis Artifacts",
                "",
                "These files were generated by `scripts/embedding_analysis/01_extract_structure_embeddings.py`.",
                "They are additive analysis artifacts and do not modify training or evaluation result folders.",
                "",
                "## Test-Set Extraction",
                "",
                "Counts by family:",
                family_lines,
                "",
                "Embedding matrices:",
                source_lines,
                "",
                f"NPZ: `{relpath(Path(report_summary['paths']['npz']), repo_root())}`",
                f"Metadata CSV: `{relpath(Path(report_summary['paths']['metadata_csv']), repo_root())}`",
                f"Manifest JSON: `{relpath(Path(report_summary['paths']['manifest_json']), repo_root())}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    (embeddings_dir / "readme.md").write_text(
        "\n".join(
            [
                "# Test-Set Structure Embeddings",
                "",
                "This folder contains fixed test-set structure embeddings from the local pretrained ALIGNN model.",
                "",
                "Files:",
                "- `structure_embeddings.npz`: compressed embedding matrices keyed by embedding source.",
                "- `structure_embedding_metadata.csv`: one row per structure per embedding source.",
                "- `structure_embedding_metadata.parquet`: written only when a parquet engine is installed.",
                "",
                "Embedding sources:",
                source_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )

    (manifests_dir / "readme.md").write_text(
        "\n".join(
            [
                "# Embedding Analysis Manifests",
                "",
                "Manifest files summarize input paths, hook choices, counts, shapes, and output paths.",
                "",
                "Primary file:",
                "- `test_set_embedding_manifest.json`",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_report_notes(
    root: Path,
    specs: dict[str, HookSpec],
    summary: dict[str, Any],
) -> None:
    report_dir = root / "results" / "derived_evidence" / "embedding_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)

    methods_lines = [
        "# embedding_analysis Embedding Extraction Methods Notes",
        "",
        "This phase extracts embeddings only. It does not retrain, fine-tune, download a model, or modify existing result folders.",
        "",
        "## Model Source",
        "",
        "- Checkpoint: `models/pretrained/checkpoint_300.pt`",
        "- Config: `configs/pretrained/config.json`",
        "- Target: `formation_energy_peratom`",
        "",
        "## Hook Points",
        "",
    ]
    for spec in specs.values():
        methods_lines.extend(
            [
                f"### `{spec.name}`",
                "",
                f"- Layer name: `{spec.layer_name}`",
                f"- Module path: `{spec.module_path}`",
                f"- Choice: {spec.description}",
                "",
            ]
        )
    methods_lines.extend(
        [
            "## Notes",
            "",
            "- `pre_head` is the output of `model.readout`; for this config with `extra_features=0`, it is immediately before `model.fc`.",
            "- `last_alignn_pool` pools the node tensor returned by the last ALIGNN block before the GCN stack.",
            "- `last_gcn_pool` pools the node tensor returned by the last GCN block. In this model it is the nearest equivalent to, and should numerically match, `pre_head`.",
            "- All pooling uses DGL `AvgPooling`, matching the model's own graph readout.",
            "",
        ]
    )
    (report_dir / "methods_notes.md").write_text(
        "\n".join(methods_lines), encoding="utf-8"
    )

    summary_lines = [
        "# Test-Set Embedding Extraction Summary",
        "",
        "Generated by `scripts/embedding_analysis/01_extract_structure_embeddings.py`.",
        "",
        "## Counts",
        "",
    ]
    for family, count in summary["counts_by_family"].items():
        summary_lines.append(f"- `{family}`: {count} structures")
    summary_lines.extend(["", "## Embedding Shapes", ""])
    for source, shape in summary["embedding_shapes"].items():
        summary_lines.append(f"- `{source}`: `{shape}`")
    summary_lines.extend(
        [
            "",
            "## Output Paths",
            "",
            f"- NPZ: `{relpath(Path(summary['paths']['npz']), root)}`",
            f"- Metadata CSV: `{relpath(Path(summary['paths']['metadata_csv']), root)}`",
            f"- Metadata parquet: `{relpath(Path(summary['paths']['metadata_parquet']), root)}`",
            f"- Manifest JSON: `{relpath(Path(summary['paths']['manifest_json']), root)}`",
            "",
            "## Hook Equivalence Check",
            "",
            f"- Max absolute `pre_head - last_gcn_pool`: `{summary['source_diffs'].get('max_abs_pre_head_minus_last_gcn_pool')}`",
            "",
            "## Parquet Status",
            "",
        ]
    )
    parquet_status = summary["parquet_status"]
    if parquet_status.get("written"):
        summary_lines.append(
            f"- Written: `{relpath(Path(parquet_status['path']), root)}`"
        )
    else:
        reason = str(parquet_status.get("reason", "unknown reason")).splitlines()[0]
        summary_lines.append(f"- Not written: {reason}")
    summary_lines.append("")
    (report_dir / "test_set_extraction_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )


def write_pool_extraction_summary(root: Path, output_root: Path) -> None:
    report_dir = root / "results" / "derived_evidence" / "embedding_analysis"
    manifest_paths = [
        output_root / "manifests" / "balanced_pool_embedding_manifest.json",
        output_root / "manifests" / "oxide_reference_pool_embedding_manifest.json",
    ]
    available = []
    for path in manifest_paths:
        if path.exists():
            available.append(json.loads(path.read_text(encoding="utf-8")))
    if not available:
        return

    lines = [
        "# Pool Embedding Extraction Summary",
        "",
        "Generated by `scripts/embedding_analysis/01_extract_structure_embeddings.py` using the same hook definitions as the fixed test-set extraction.",
        "",
        "No hook logic was changed for pool extraction.",
        "",
        "## Extracted Subsets",
        "",
    ]
    for payload in available:
        label = payload.get("subset_name") or payload.get("dataset_subset")
        lines.append(f"### `{label}`")
        lines.append("")
        lines.append(f"- Total structures: {payload['total_structures']}")
        for family, count in payload["counts_by_family"].items():
            lines.append(f"- `{family}`: {count}")
        lines.append("- Embedding shapes:")
        for source, shape in payload["embedding_shapes"].items():
            lines.append(f"  - `{source}`: `{shape}`")
        lines.append(
            f"- Max absolute `pre_head - last_gcn_pool`: `{payload['source_diffs'].get('max_abs_pre_head_minus_last_gcn_pool')}`"
        )
        lines.append(f"- NPZ: `{relpath(Path(payload['paths']['npz']), root)}`")
        lines.append(
            f"- Metadata CSV: `{relpath(Path(payload['paths']['metadata_csv']), root)}`"
        )
        lines.append(
            f"- Manifest JSON: `{relpath(Path(payload['paths']['manifest_json']), root)}`"
        )
        diagnostics = payload.get("diagnostics", {})
        lines.append(
            f"- Missing structure files: {diagnostics.get('missing_structure_files', 0)}"
        )
        lines.append(f"- Missing targets: {diagnostics.get('missing_targets', 0)}")
        lines.append("")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "pool_extraction_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def inspect_plan(
    root: Path,
    args: argparse.Namespace,
    cfg: Any,
    model: Any,
    specs: dict[str, HookSpec],
) -> None:
    print("Dry-run / inspection only.")
    print(f"Repository root: {root}")
    print(f"Checkpoint: {args.pretrained_checkpoint}")
    print(f"Config: {args.pretrained_config}")
    if args.subset_metadata:
        subset_metadata = (root / args.subset_metadata).resolve()
        print(f"Subset metadata: {relpath(subset_metadata, root)}")
        print(f"Subset name: {args.subset_name or Path(args.subset_metadata).stem}")
    else:
        print(f"Dataset subset: {args.dataset_subset}")
        print(f"Families: {', '.join(args.families)}")
    print(f"Target: {cfg.target}")
    print(f"Line graph: {cfg.compute_line_graph}")
    print(f"Hidden features: {getattr(model.config, 'hidden_features', 'unknown')}")
    print("Hook points:")
    for spec in specs.values():
        print(f"  - {spec.name}: {spec.module_path} ({spec.layer_name})")
    if args.subset_metadata:
        with (root / args.subset_metadata).open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if args.limit_per_family is not None:
            rows = rows[: args.limit_per_family]
        family_counts: dict[str, int] = {}
        split_counts: dict[str, int] = {}
        for row in rows:
            family_counts[row["family"]] = family_counts.get(row["family"], 0) + 1
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        print(f"Planned structures: {len(rows)}")
        print(f"Counts by family: {dict(sorted(family_counts.items()))}")
        print(f"Counts by split: {dict(sorted(split_counts.items()))}")
    else:
        for family in args.families:
            manifest = root / "data" / family / "manifests" / f"{args.dataset_subset}.csv"
            with manifest.open("r", encoding="utf-8", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            if args.limit_per_family is not None:
                count = min(count, args.limit_per_family)
            print(f"{family}: {count} planned structures from {relpath(manifest, root)}")
    output_root = (root / args.output_dir).resolve()
    print(f"Output folder: {relpath(output_root / 'embeddings' / subset_output_name(args), root)}")


def main() -> None:
    args = parse_args()
    if args.inspect_only:
        args.dry_run = True

    root = repo_root()
    output_root = (root / args.output_dir).resolve()
    output_name = subset_output_name(args)
    embeddings_dir = output_root / "embeddings" / output_name
    manifests_dir = output_root / "manifests"
    cache_dir = output_root / "cache"
    checkpoint_path = (root / args.pretrained_checkpoint).resolve()
    config_path = (root / args.pretrained_config).resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing local checkpoint: {checkpoint_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Missing local config: {config_path}")

    runtime = import_runtime(output_root)
    set_random_seed(runtime.torch, args.seed)
    cfg, config_payload = load_training_config(runtime, config_path)
    cfg.target = config_payload.get("target", cfg.target)
    cfg.id_tag = config_payload.get("id_tag", cfg.id_tag)
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    model = load_pretrained_model(runtime, checkpoint_path, config_path)
    source_names = selected_sources(args.hook_selection)
    specs = resolve_hook_specs(model, source_names)

    if args.dry_run:
        inspect_plan(root, args, cfg, model, specs)
        return

    embeddings_dir.mkdir(parents=True, exist_ok=True)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    npz_path = embeddings_dir / "structure_embeddings.npz"
    metadata_csv_path = embeddings_dir / "structure_embedding_metadata.csv"
    metadata_parquet_path = embeddings_dir / "structure_embedding_metadata.parquet"
    structure_records_csv_path = manifests_dir / f"{output_name}_structure_records.csv"
    manifest_json_path = manifests_dir / f"{output_name}_embedding_manifest.json"
    output_targets = [
        npz_path,
        metadata_csv_path,
        metadata_parquet_path,
        structure_records_csv_path,
        manifest_json_path,
        embeddings_dir / "readme.md",
    ]
    if args.subset_metadata is None:
        output_targets.extend(
            [
                output_root / "readme.md",
                manifests_dir / "readme.md",
                root / "results" / "derived_evidence" / "embedding_analysis" / "methods_notes.md",
                root / "results" / "derived_evidence" / "embedding_analysis" / "test_set_extraction_summary.md",
            ]
        )
    check_output_targets(output_targets, allow_overwrite=args.allow_overwrite)

    device = runtime.torch.device(args.device)
    if device.type == "cuda" and not runtime.torch.cuda.is_available():
        raise SystemExit("CUDA device requested but torch.cuda.is_available() is False.")

    start = time.time()
    all_records: list[dict[str, Any]] = []
    all_arrays: dict[str, list[np.ndarray]] = {name: [] for name in specs}
    prediction_by_id: dict[str, float] = {}
    family_counts: dict[str, int] = {}
    diff_values: list[float] = []
    diagnostics: dict[str, Any] = {}

    if args.subset_metadata:
        subset_metadata_path = (root / args.subset_metadata).resolve()
        print(
            f"Loading subset {args.subset_name or output_name} from {relpath(subset_metadata_path, root)}..."
        )
        rows, records, diagnostics = load_subset_rows(
            runtime=runtime,
            root=root,
            subset_metadata=subset_metadata_path,
            target_key=cfg.target,
            limit=args.limit_per_family,
        )
        print(f"subset rows: {len(rows)} structures")
        print(f"missing structure files: {diagnostics['missing_structure_files']}")
        print(f"missing targets: {diagnostics['missing_targets']}")
        loader = make_loader(
            runtime=runtime,
            cfg=cfg,
            rows=rows,
            cache_dir=cache_dir,
            family=output_name,
            subset="subset",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit_per_family,
            rebuild_cache=args.rebuild_cache,
        )
        arrays, predictions, source_diffs = extract_embeddings(
            runtime=runtime,
            model=model,
            loader=loader,
            records=records,
            specs=specs,
            device=device,
            use_line_graph=bool(cfg.compute_line_graph),
        )
        if len(predictions) != len(records):
            raise RuntimeError(
                f"{output_name}: got {len(predictions)} predictions for {len(records)} records."
            )
        for record, prediction in zip(records, predictions):
            prediction_by_id[record["material_id"]] = float(prediction)
        for source_name, array in arrays.items():
            print(f"{output_name} {source_name}: shape {array.shape}")
            all_arrays[source_name].append(array)
        if source_diffs["max_abs_pre_head_minus_last_gcn_pool"] is not None:
            diff_values.append(float(source_diffs["max_abs_pre_head_minus_last_gcn_pool"]))
        all_records.extend(records)
        family_counts = count_by_family(records)
    else:
        for family in args.families:
            print(f"Loading {family} {args.dataset_subset} structures...")
            rows, records = load_family_rows(
                runtime=runtime,
                root=root,
                family=family,
                subset=args.dataset_subset,
                target_key=cfg.target,
                limit=args.limit_per_family,
            )
            print(f"{family}: {len(rows)} structures")
            loader = make_loader(
                runtime=runtime,
                cfg=cfg,
                rows=rows,
                cache_dir=cache_dir,
                family=family,
                subset=args.dataset_subset,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                limit=args.limit_per_family,
                rebuild_cache=args.rebuild_cache,
            )
            arrays, predictions, source_diffs = extract_embeddings(
                runtime=runtime,
                model=model,
                loader=loader,
                records=records,
                specs=specs,
                device=device,
                use_line_graph=bool(cfg.compute_line_graph),
            )
            if len(predictions) != len(records):
                raise RuntimeError(
                    f"{family}: got {len(predictions)} predictions for {len(records)} records."
                )
            for record, prediction in zip(records, predictions):
                prediction_by_id[record["material_id"]] = float(prediction)
            for source_name, array in arrays.items():
                print(f"{family} {source_name}: shape {array.shape}")
                all_arrays[source_name].append(array)
            if source_diffs["max_abs_pre_head_minus_last_gcn_pool"] is not None:
                diff_values.append(float(source_diffs["max_abs_pre_head_minus_last_gcn_pool"]))
            all_records.extend(records)
            family_counts[family] = len(records)

    final_arrays = {
        source_name: np.concatenate(parts, axis=0)
        for source_name, parts in all_arrays.items()
    }
    np.savez_compressed(npz_path, **final_arrays)

    long_metadata = build_long_metadata(
        base_records=all_records,
        specs=specs,
        prediction_by_id=prediction_by_id,
        arrays=final_arrays,
    )
    metadata_fields = [
        "material_id",
        "family",
        "split",
        "formula",
        "target_formation_energy_peratom",
        "pretrained_prediction",
        "absolute_error",
        "embedding_source",
        "layer_name",
        "module_path",
        "npz_key",
        "embedding_index",
        "embedding_dim",
        "filename",
        "structure_path",
        "source_manifest",
        "n_atoms",
        "is_oxide",
        "is_nitride",
        "is_oxynitride",
    ]
    write_csv(metadata_csv_path, long_metadata, metadata_fields)
    parquet_status = write_optional_parquet(long_metadata, metadata_parquet_path)

    structure_fields = [
        "material_id",
        "family",
        "split",
        "formula",
        "target_formation_energy_peratom",
        "pretrained_prediction",
        "absolute_error",
        "filename",
        "structure_path",
        "source_manifest",
        "n_atoms",
        "is_oxide",
        "is_nitride",
        "is_oxynitride",
    ]
    structure_records = []
    for record in all_records:
        row = dict(record)
        prediction = prediction_by_id.get(row["material_id"])
        row["pretrained_prediction"] = prediction
        if prediction is not None and row["target_formation_energy_peratom"] is not None:
            row["absolute_error"] = abs(
                float(prediction) - float(row["target_formation_energy_peratom"])
            )
        structure_records.append(row)
    write_csv(structure_records_csv_path, structure_records, structure_fields)

    source_diff = max(diff_values) if diff_values else None
    summary = {
        "created_at_unix": time.time(),
        "elapsed_seconds": time.time() - start,
        "dataset_subset": args.dataset_subset,
        "subset_name": args.subset_name,
        "output_name": output_name,
        "families": list(args.families),
        "counts_by_family": family_counts,
        "total_structures": len(all_records),
        "embedding_sources": {
            name: {
                "layer_name": spec.layer_name,
                "module_path": spec.module_path,
                "description": spec.description,
            }
            for name, spec in specs.items()
        },
        "embedding_shapes": {
            name: list(array.shape) for name, array in final_arrays.items()
        },
        "source_diffs": {
            "max_abs_pre_head_minus_last_gcn_pool": source_diff,
        },
        "paths": {
            "npz": str(npz_path),
            "metadata_csv": str(metadata_csv_path),
            "metadata_parquet": str(metadata_parquet_path),
            "structure_records_csv": str(structure_records_csv_path),
            "manifest_json": str(manifest_json_path),
        },
        "inputs": {
            "checkpoint": relpath(checkpoint_path, root),
            "config": relpath(config_path, root),
            "subset_metadata": None
            if args.subset_metadata is None
            else relpath((root / args.subset_metadata).resolve(), root),
            "manifests": [
                f"data/{family}/manifests/{args.dataset_subset}.csv"
                for family in args.families
                if args.subset_metadata is None
            ],
            "zero_shot_predictions": [
                f"results/zero_shot/{family}/predictions.csv"
                for family in args.families
                if args.dataset_subset == "test"
            ],
        },
        "runtime_versions": runtime_versions(),
        "parquet_status": parquet_status,
        "diagnostics": diagnostics,
        "command": " ".join(sys.argv),
    }
    write_json(manifest_json_path, summary)
    if args.subset_metadata is None:
        write_readmes(output_root, embeddings_dir, manifests_dir, summary)
        write_report_notes(root, specs, summary)
    else:
        title = f"{args.subset_name or output_name} Structure Embeddings"
        write_embedding_dir_readme(
            root=root,
            embeddings_dir=embeddings_dir,
            report_summary=summary,
            title=title,
            description=(
                "This folder contains structure embeddings extracted with the "
                "same local pretrained ALIGNN hook definitions used for the fixed test set."
            ),
        )
        write_pool_extraction_summary(root, output_root)

    print("Extraction complete.")
    print(f"Counts by family: {family_counts}")
    for source_name, array in final_arrays.items():
        print(f"{source_name}: shape {array.shape}")
    print(f"NPZ: {relpath(npz_path, root)}")
    print(f"Metadata CSV: {relpath(metadata_csv_path, root)}")
    if parquet_status["written"]:
        print(f"Metadata parquet: {relpath(metadata_parquet_path, root)}")
    else:
        print(f"Metadata parquet skipped: {parquet_status['reason']}")
    print(f"Manifest: {relpath(manifest_json_path, root)}")


if __name__ == "__main__":
    main()
