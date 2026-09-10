#!/usr/bin/env python
"""Verify that the released primary M-087 summary has the paper values."""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    path = Path(__file__).resolve().parents[1] / "artifacts" / "paper_results" / "primary_summary_m087.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "DPA_AUBPC": 48.24140663580247,
        "IDR_AUBPC": 55.72222484567901,
        "difference_pp": 7.480818209876539,
        "ci95_pp": [6.608820327932103, 8.359007484567897],
    }
    actual = {
        "DPA_AUBPC": summary["DPA_AUBPC"],
        "IDR_AUBPC": summary["frozen_IDR_AUBPC"],
        "difference_pp": summary["difference_pp"],
        "ci95_pp": summary["ci95_pp"],
    }
    if actual != expected:
        raise SystemExit(json.dumps({"pass": False, "expected": expected, "actual": actual}, indent=2))
    print(json.dumps({"pass": True, "source": "M-087", "api_calls": 0}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
