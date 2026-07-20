from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path


def module_group(name: str) -> str:
    parts = name.split(".")
    if parts[0] in {"alignn_layers", "gcn_layers"} and len(parts) >= 2:
        return ".".join(parts[:2])
    return parts[0]


def import_alignn():
    try:
        from alignn.models.alignn import ALIGNN, ALIGNNConfig
    except ImportError as exc:
        raise SystemExit(
            "ALIGNN is not available in this Python environment.\n"
            "Activate the training environment first, for example:\n"
            "  conda activate domain_shift_train"
        ) from exc
    return ALIGNN, ALIGNNConfig


def load_model(config_path: Path):
    ALIGNN, ALIGNNConfig = import_alignn()
    payload = json.loads(config_path.read_text())
    model = ALIGNN(ALIGNNConfig(**payload["model"]))
    return model, payload


def maybe_load_checkpoint(model, checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is not available in this Python environment.\n"
            "Activate the training environment first, for example:\n"
            "  conda activate domain_shift_train"
        ) from exc

    state = torch.load(checkpoint_path, map_location="cpu")
    model_state = state["model"] if isinstance(state, dict) and "model" in state else state
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Checkpoint load summary")
    print(f"  path: {checkpoint_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  first missing key: {missing[0]}")
    if unexpected:
        print(f"  first unexpected key: {unexpected[0]}")
    print()


def count_parameters(module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def print_module_tree(module, max_depth: int, prefix: str = "", depth: int = 0) -> None:
    if max_depth >= 0 and depth > max_depth:
        return

    if depth == 0:
        print("Module tree")
        print("  [root] " + module.__class__.__name__ + f" | params={count_parameters(module):,}")

    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        indent = "  " * (depth + 1)
        print(
            f"{indent}- {child_prefix}: {child.__class__.__name__} "
            f"| params={count_parameters(child):,}"
        )
        print_module_tree(child, max_depth=max_depth, prefix=child_prefix, depth=depth + 1)


def ordered_parameter_groups(model) -> list[str]:
    groups: list[str] = []
    for name, _parameter in model.named_parameters():
        group = module_group(name)
        if group not in groups:
            groups.append(group)
    return groups


def parameter_count_by_group(model) -> OrderedDict[str, int]:
    counts: OrderedDict[str, int] = OrderedDict()
    for name, parameter in model.named_parameters():
        group = module_group(name)
        counts[group] = counts.get(group, 0) + parameter.numel()
    return counts


def print_parameter_groups(model, n_last_groups: int) -> None:
    counts = parameter_count_by_group(model)
    groups = list(counts.keys())
    chosen = groups[-n_last_groups:] if n_last_groups > 0 else []

    print("Parameter groups")
    for index, group in enumerate(groups):
        marker = " <== finetune wrapper unfreezes this group" if group in chosen else ""
        print(f"  [{index:02d}] {group:<20} params={counts[group]:>10,}{marker}")

    print()
    print(f"Last {n_last_groups} groups according to fine_tune_last_two_alignn_layers.py")
    print("  " + (", ".join(chosen) if chosen else "<none>"))
    print()


def print_named_parameters(model, limit: int) -> None:
    print(f"Named parameters (first {limit})")
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if index >= limit:
            break
        shape = tuple(parameter.shape)
        print(f"  {name:<60} shape={shape}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the ALIGNN model architecture used in this repo."
    )
    parser.add_argument(
        "--config",
        default="configs/pretrained/config.json",
        help="Path to the saved ALIGNN config JSON.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path to test loading weights.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=2,
        help="Max depth for the printed module tree. Use -1 for full recursion.",
    )
    parser.add_argument(
        "--n-last-groups",
        type=int,
        default=2,
        help="How many final parameter groups to highlight.",
    )
    parser.add_argument(
        "--show-named-parameters",
        action="store_true",
        help="Also print a sample of named parameter tensors.",
    )
    parser.add_argument(
        "--named-parameter-limit",
        type=int,
        default=40,
        help="How many named parameters to print when --show-named-parameters is set.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve() if args.checkpoint else None
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    if checkpoint_path is not None and not checkpoint_path.exists():
        raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")

    model, payload = load_model(config_path)

    print("ALIGNN model inspection")
    print(f"  config: {config_path}")
    print(f"  checkpoint: {checkpoint_path if checkpoint_path else '<not loaded>'}")
    print(f"  dataset: {payload.get('dataset')}")
    print(f"  target: {payload.get('target')}")
    print(f"  model name: {payload.get('model', {}).get('name')}")
    print(f"  total params: {count_parameters(model):,}")
    print()

    maybe_load_checkpoint(model, checkpoint_path)

    print("Model config")
    print(json.dumps(payload.get("model", {}), indent=2))
    print()

    print("Full model repr")
    print(model)
    print()

    print_module_tree(model, max_depth=args.tree_depth)
    print()
    print_parameter_groups(model, n_last_groups=args.n_last_groups)

    if args.show_named_parameters:
        print_named_parameters(model, limit=args.named_parameter_limit)


if __name__ == "__main__":
    main()
