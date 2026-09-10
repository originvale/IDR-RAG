#!/usr/bin/env python
"""Create rights-safe action tables from local frozen trajectory banks.

This maintainer utility reads the original nested trajectory JSONL files but
writes only identifiers, F1 outcomes, the 32 numeric features, frozen policy
scores, and resource measurements. It intentionally excludes questions,
answers, reasoning text, passage text, API metadata, and local paths.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from idr_rag.features import FEATURE_NAMES, features
from idr_rag.io import read_csv, sha256, write_csv, write_json


def load_bank(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "complete":
                continue
            key = (str(row["dataset"]).strip().lower(), str(row["query_id"]).strip())
            if key in records:
                raise ValueError(f"duplicate trajectory {key}")
            records[key] = row
    if len(records) != 1800:
        raise ValueError(f"expected 1,800 complete trajectories, found {len(records)}")
    counts = Counter(dataset for dataset, _ in records)
    if counts != Counter({"2wiki": 600, "hotpotqa": 600, "musique": 600}):
        raise ValueError(f"unexpected dataset counts: {counts}")
    return records


def base_rows(bank: dict[tuple[str, str], dict[str, Any]], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (dataset, query_id), record in sorted(bank.items()):
        states = [record.get("initial_state") or {}, *(record.get("states") or [])]
        if len(states) != 3:
            raise ValueError(f"trajectory does not contain R0/R1/R2: {(dataset, query_id)}")
        quality = [float(state.get("f1") or 0.0) for state in states]
        for depth in (1, 2):
            values = features(states[depth - 1], depth)
            row: dict[str, Any] = {
                "split": split,
                "dataset": dataset,
                "query_id": query_id,
                "group": f"{dataset}::{query_id}",
                "depth": depth,
                "initial_f1": quality[0],
                "pre_f1": quality[depth - 1],
                "post_f1": quality[depth],
                "observed_delta": quality[depth] - quality[depth - 1],
            }
            row.update(dict(zip(FEATURE_NAMES, values)))
            rows.append(row)
    return rows


def keyed(path: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    rows = read_csv(path)
    result = {
        (str(row["dataset"]).lower(), str(row["query_id"]), int(row["depth"])): row
        for row in rows
    }
    if len(result) != 3600:
        raise ValueError(f"expected 3,600 score rows in {path}, found {len(result)}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--frozen-actions", type=Path, required=True)
    parser.add_argument("--mechanism-scores", type=Path, required=True)
    parser.add_argument("--adaptive-actions", type=Path, required=True)
    parser.add_argument("--query-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=False)
    development = base_rows(load_bank(args.development), "allocator-development")
    evaluation = base_rows(load_bank(args.evaluation), "independent-evaluation")
    frozen = keyed(args.frozen_actions)
    mechanism = keyed(args.mechanism_scores)
    adaptive = keyed(args.adaptive_actions)

    for row in evaluation:
        key = (str(row["dataset"]), str(row["query_id"]), int(row["depth"]))
        old = frozen[key]
        mech = mechanism[key]
        ext = adaptive[key]
        if abs(float(old["observed_delta"]) - float(row["observed_delta"])) > 1e-12:
            raise ValueError(f"outcome mismatch for {key}")
        row.update(
            {
                "mu": float(old["mu"]),
                "v_hat": float(old["v_hat"]),
                "dynamic_benefit": float(mech["DynamicBenefit"]),
                "adaptive_score": float(ext["adaptive_score"]),
                "historical_native_u": float(old["historical_native_u"]),
                "historical_raser": float(old["historical_raser"]),
                "historical_benefit": float(old["historical_benefit"]),
                "historical_direct_gain_et": float(old["historical_direct_gain_et"]),
                "historical_idr": float(old["historical_idr"]),
                "provider_tokens": float(old["provider_tokens"]),
                "latency_ms": float(old["latency_ms"]),
            }
        )

    write_csv(args.out / "development_actions_32d.csv", development)
    write_csv(args.out / "evaluation_actions_32d.csv", evaluation)
    manifest_target = args.out / "query_manifest.csv"
    manifest_target.write_bytes(args.query_manifest.read_bytes())
    inputs = {
        "development_trajectory_sha256": sha256(args.development),
        "evaluation_trajectory_sha256": sha256(args.evaluation),
        "frozen_actions_sha256": sha256(args.frozen_actions),
        "mechanism_scores_sha256": sha256(args.mechanism_scores),
        "adaptive_actions_sha256": sha256(args.adaptive_actions),
        "query_manifest_sha256": sha256(args.query_manifest),
    }
    outputs = {
        path.name: sha256(path)
        for path in sorted(args.out.iterdir())
        if path.is_file()
    }
    write_json(
        args.out / "ARTIFACT_MANIFEST.json",
        {
            "status": "PASS",
            "privacy_contract": "identifiers, numeric states/outcomes, frozen scores and resource measurements only; no question, answer, reasoning, passage text, API metadata or local path",
            "development_actions": len(development),
            "evaluation_actions": len(evaluation),
            "feature_count": len(FEATURE_NAMES),
            "inputs": inputs,
            "outputs": outputs,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
