#!/usr/bin/env python
"""Reproduce the eleven-policy main table without network or model calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from idr_rag.allocation import precedence_ok, summarize
from idr_rag.io import read_csv, write_csv
from idr_rag.paper import policy_orders


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", type=Path, default=Path("artifacts/evaluation_actions_32d.csv"))
    parser.add_argument("--expected", type=Path, default=Path("artifacts/expected_results.json"))
    parser.add_argument("--out", type=Path, default=Path("reproduced/main"))
    args = parser.parse_args()

    actions = read_csv(args.actions)
    orders = policy_orders(actions)
    rows = []
    for method, order in orders.items():
        if len(order) != 3600 or not precedence_ok(order):
            raise RuntimeError(f"invalid allocation order for {method}")
        rows.append({"method": method, **summarize(order)})

    expected = json.loads(args.expected.read_text(encoding="utf-8"))["main_table"]
    failures = []
    for row in rows:
        target = expected[row["method"]]
        for field in ("F1_at_10_pp", "F1_at_25_pp", "F1_at_50_pp", "AUBPC_0_50_pp"):
            error = abs(float(row[field]) - float(target[field]))
            if error > 5e-10:
                failures.append({"method": row["method"], "field": field, "error": error})

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "main_table.csv", rows)
    report = {"pass": not failures, "failures": failures, "methods": len(rows), "actions": len(actions), "network_calls": 0}
    (args.out / "REPRODUCTION_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        print(
            f"{row['method']:<30} "
            f"F1@10={row['F1_at_10_pp']:.3f} "
            f"F1@25={row['F1_at_25_pp']:.3f} "
            f"F1@50={row['F1_at_50_pp']:.3f} "
            f"AUBPC={row['AUBPC_0_50_pp']:.3f}"
        )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
