"""Small, dependency-free readers and validators for released artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError("cannot write an empty CSV")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in materialized:
        for field in row:
            if field not in fields:
                fields.append(field)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def group_actions(
    rows: list[dict[str, Any]], expected_queries: int | None = None
) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        dataset = str(row["dataset"]).strip().lower()
        query_id = str(row["query_id"]).strip()
        group = str(row.get("group") or f"{dataset}::{query_id}")
        # CSV exports preserve the trajectory depth as a numeric value, so it
        # may round-trip as either ``1`` or ``1.0``.
        depth = int(float(row["depth"]))
        if depth in grouped[group]:
            raise ValueError(f"duplicate action {group} depth={depth}")
        grouped[group][depth] = row
    if expected_queries is not None and len(grouped) != expected_queries:
        raise ValueError(f"expected {expected_queries} queries, found {len(grouped)}")
    if any(set(depths) != {1, 2} for depths in grouped.values()):
        raise ValueError("every query must contain exactly depth-1 and depth-2 actions")
    return dict(grouped)


def validate_paper_bank(rows: list[dict[str, Any]]) -> dict[str, int]:
    grouped = group_actions(rows, expected_queries=1800)
    counts = Counter(str(depths[1]["dataset"]).strip().lower() for depths in grouped.values())
    expected = Counter({"2wiki": 600, "hotpotqa": 600, "musique": 600})
    if counts != expected:
        raise ValueError(f"dataset counts differ from the paper contract: {counts}")
    return dict(counts)
