"""Paper policy registry built from the released evaluation action ledger."""

from __future__ import annotations

from typing import Any

from .allocation import (
    dynamic_order,
    hash_tie,
    random_eligible_order,
    static_path_order,
)


def policy_orders(actions: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        "IAU": static_path_order(actions, "historical_native_u"),
        "RASER-inspired": static_path_order(actions, "historical_raser"),
        "Adaptive-RAG-style": dynamic_order(actions, "adaptive_score"),
        "Benefit Classifier": static_path_order(actions, "historical_benefit"),
        "Direct-Gain ExtraTrees": static_path_order(actions, "historical_direct_gain_et"),
        "Fixed Query Ranking": static_path_order(actions, "historical_idr"),
        "RandomEligible": random_eligible_order(actions),
        "Depth-Prior Allocation": dynamic_order(actions, "mu", tie_key=hash_tie),
        "Static Path Commitment": static_path_order(actions, "v_hat"),
        "Dynamic Benefit Probability": dynamic_order(actions, "dynamic_benefit"),
        "IDR": dynamic_order(actions, "v_hat"),
    }
