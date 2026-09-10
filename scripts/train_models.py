#!/usr/bin/env python
"""Retrain the paper's IDR and Dynamic Benefit Probability estimators."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from idr_rag.io import read_csv
from idr_rag.model import (
    fit_dynamic_benefit,
    fit_idr,
    predict_dynamic_benefit,
    predict_idr,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, default=Path("artifacts/development_actions_32d.csv"))
    parser.add_argument("--evaluation", type=Path, default=Path("artifacts/evaluation_actions_32d.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    args = parser.parse_args()

    development = read_csv(args.development)
    evaluation = read_csv(args.evaluation)
    idr = fit_idr(development)
    dbp = fit_dynamic_benefit(development)
    idr_prediction = predict_idr(idr, evaluation)
    dbp_prediction = predict_dynamic_benefit(dbp, evaluation)
    frozen_idr = np.asarray([float(row["v_hat"]) for row in evaluation])
    frozen_dbp = np.asarray([float(row["dynamic_benefit"]) for row in evaluation])
    report = {
        "development_actions": len(development),
        "evaluation_actions": len(evaluation),
        "idr_max_abs_error": float(np.max(np.abs(idr_prediction - frozen_idr))),
        "dynamic_benefit_max_abs_error": float(np.max(np.abs(dbp_prediction - frozen_dbp))),
    }
    report["pass"] = report["idr_max_abs_error"] < 1e-10 and report["dynamic_benefit_max_abs_error"] < 1e-10
    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(idr, args.out / "idr_full32d.joblib")
    joblib.dump(dbp, args.out / "dynamic_benefit_full32d.joblib")
    (args.out / "RETRAIN_AUDIT.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
