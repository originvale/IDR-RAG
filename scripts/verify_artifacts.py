#!/usr/bin/env python
"""Verify release checksums and artifact cardinalities."""

from __future__ import annotations

import json
from pathlib import Path

from idr_rag.io import read_csv, sha256, validate_paper_bank


def main() -> int:
    root = Path(__file__).resolve().parents[1] / "artifacts"
    manifest = json.loads((root / "PUBLIC_MANIFEST.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["sha256"].items():
        actual = sha256(root / name)
        if actual.lower() != str(expected).lower():
            errors.append({"file": name, "expected": expected, "actual": actual})
    development = read_csv(root / "development_actions_32d.csv")
    evaluation = read_csv(root / "evaluation_actions_32d.csv")
    validate_paper_bank(development)
    validate_paper_bank(evaluation)
    report = {"pass": not errors, "errors": errors, "development_actions": len(development), "evaluation_actions": len(evaluation), "privacy_contract": "numeric identifiers and derived values only"}
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
