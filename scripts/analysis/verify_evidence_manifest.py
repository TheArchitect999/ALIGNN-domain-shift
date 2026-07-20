#!/usr/bin/env python3
"""Verify public evidence paths, selectors, and displayed values."""

from __future__ import annotations

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=root)
    parser.add_argument("--manifest", type=Path, default=Path("paper/evidence_manifest.csv"))
    return parser.parse_args()


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def selected_rows(rows: list[dict[str, str]], where: dict[str, str]) -> list[dict[str, str]]:
    return [row for row in rows if all(row.get(key, "") == str(value) for key, value in where.items())]


def json_pointer(payload: Any, pointer: str) -> Any:
    value = payload
    if pointer in {"", "/"}:
        return value
    for token in pointer.lstrip("/").split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def value_matches(actual: Any, displayed: str) -> bool:
    actual_text = str(actual)
    if actual_text == displayed:
        return True
    try:
        expected_decimal = Decimal(displayed)
        actual_float = float(actual)
    except (InvalidOperation, TypeError, ValueError):
        return actual_text.strip() == displayed.strip()
    if not math.isfinite(actual_float):
        return False
    exponent = expected_decimal.as_tuple().exponent
    rounding_tolerance = float(Decimal("0.5") * (Decimal(10) ** exponent)) if exponent < 0 else 0.0
    tolerance = max(1e-12, rounding_tolerance)
    return math.isclose(actual_float, float(expected_decimal), rel_tol=1e-10, abs_tol=tolerance)


def resolve_value(source: Path, selector: dict[str, Any]) -> Any:
    selector_type = selector["type"]
    if selector_type.startswith("csv_"):
        rows = csv_rows(source)
        matches = selected_rows(rows, selector.get("where", {}))
        if selector_type == "csv_row_count":
            return len(rows)
        if selector_type == "csv_constant_column":
            expected_rows = int(selector["expected_rows"])
            if len(rows) != expected_rows:
                raise ValueError(f"expected {expected_rows} rows, found {len(rows)}")
            values = {row[selector["column"]] for row in rows}
            if len(values) != 1:
                raise ValueError(f"column {selector['column']} is not constant: {sorted(values)}")
            return next(iter(values))
        if len(matches) != 1:
            raise ValueError(f"selector matched {len(matches)} rows")
        row = matches[0]
        if selector_type == "csv_cell":
            return row[selector["column"]]
        if selector_type == "csv_formula" and selector.get("operation") == "subtract":
            left, right = selector["columns"]
            return float(row[left]) - float(row[right])
        raise ValueError(f"unsupported CSV selector: {selector_type}")

    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    value = json_pointer(payload, selector["pointer"])
    if selector_type == "json_pointer":
        return value
    if selector_type == "json_array_length":
        return len(value)
    raise ValueError(f"unsupported JSON selector: {selector_type}")


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    rows = csv_rows(manifest)
    failures: list[str] = []
    checked_values = 0

    for row in rows:
        evidence_id = row["id"]
        source = (root / row["source_path"]).resolve()
        if root not in source.parents and source != root:
            failures.append(f"{evidence_id}: unsafe source path {row['source_path']}")
            continue
        if not source.is_file():
            failures.append(f"{evidence_id}: missing source {row['source_path']}")
            continue
        selector_text = row["selector"].strip()
        if selector_text == "all rows":
            continue
        try:
            selector = json.loads(selector_text)
            actual = resolve_value(source, selector)
        except Exception as exc:  # report the evidence row rather than a long traceback
            failures.append(f"{evidence_id}: {exc}")
            continue
        checked_values += 1
        if not value_matches(actual, row["value_display"]):
            failures.append(
                f"{evidence_id}: displayed {row['value_display']!r} != resolved {actual!r}"
            )

    print(
        f"evidence rows={len(rows)}; selector/value checks={checked_values}; "
        f"failures={len(failures)}"
    )
    for failure in failures[:30]:
        print(f"  - {failure}")
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
