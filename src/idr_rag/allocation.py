"""Prefix-legal shared-budget allocation policies and metrics."""

from __future__ import annotations

import hashlib
import heapq
import random
import statistics
from typing import Any, Callable

from .io import group_actions

Action = dict[str, Any]
TieKey = Callable[[Action], tuple[Any, ...]]


def lexical_tie(action: Action) -> tuple[Any, ...]:
    return (
        str(action["dataset"]),
        str(action["query_id"]),
        int(float(action["depth"])),
    )


def query_hash(group: str) -> bytes:
    """The current DPA tie key: first 64 SHA-256 bits of dataset::query_id."""

    return hashlib.sha256(group.encode("utf-8")).digest()[:8]


def hash_tie(action: Action) -> tuple[Any, ...]:
    dataset = str(action["dataset"]).strip().lower()
    query_id = str(action["query_id"]).strip()
    group = str(action.get("group") or f"{dataset}::{query_id}")
    return (query_hash(group), int(float(action["depth"])))


def dynamic_order(
    actions: list[Action], score_field: str, tie_key: TieKey = lexical_tie
) -> list[Action]:
    """Allocate atomic actions, exposing depth 2 only after depth 1 executes."""

    grouped = group_actions(actions)
    heap: list[tuple[Any, ...]] = []
    for group, depths in grouped.items():
        action = depths[1]
        heapq.heappush(
            heap, (-float(action[score_field]), *tie_key(action), group, 1)
        )
    output: list[Action] = []
    while heap:
        *_, group, depth = heapq.heappop(heap)
        action = grouped[str(group)][int(depth)]
        output.append(action)
        if int(depth) == 1:
            successor = grouped[str(group)][2]
            heapq.heappush(
                heap,
                (-float(successor[score_field]), *tie_key(successor), group, 2),
            )
    return output


def static_path_order(
    actions: list[Action], score_field: str, tie_key: TieKey = lexical_tie
) -> list[Action]:
    """Rank from the first action and commit to each complete two-action path."""

    grouped = group_actions(actions)
    first = sorted(
        (depths[1] for depths in grouped.values()),
        key=lambda action: (-float(action[score_field]), *tie_key(action)),
    )
    return [
        action
        for head in first
        for action in (grouped[str(head["group"])][1], grouped[str(head["group"])][2])
    ]


def random_eligible_order(actions: list[Action], seed: int = 20260829) -> list[Action]:
    grouped = group_actions(actions)
    rng = random.Random(seed)
    eligible = [(group, 1) for group in sorted(grouped)]
    output: list[Action] = []
    while eligible:
        index = rng.randrange(len(eligible))
        group, depth = eligible.pop(index)
        output.append(grouped[group][depth])
        if depth == 1:
            eligible.append((group, 2))
    return output


def precedence_ok(order: list[Action]) -> bool:
    positions = {
        (str(action["group"]), int(float(action["depth"]))): index
        for index, action in enumerate(order)
    }
    groups = {str(action["group"]) for action in order}
    return all(positions[(group, 1)] < positions[(group, 2)] for group in groups)


def action_curve(
    order: list[Action], max_actions: int | None = None
) -> tuple[list[dict[str, float]], float]:
    grouped = group_actions(order)
    n_queries = len(grouped)
    total_actions = len(order)
    max_actions = n_queries if max_actions is None else int(max_actions)
    if max_actions > total_actions:
        raise ValueError("budget exceeds the action pool")
    baseline = statistics.mean(
        float(depths[1]["initial_f1"]) for depths in grouped.values()
    )
    curve = [{"actions": 0.0, "budget_fraction": 0.0, "mean_f1_pp": 100.0 * baseline}]
    cumulative = 0.0
    for used, action in enumerate(order[:max_actions], 1):
        cumulative += float(action["observed_delta"])
        curve.append(
            {
                "actions": float(used),
                "budget_fraction": used / total_actions,
                "mean_f1_pp": 100.0 * (baseline + cumulative / n_queries),
            }
        )
    area = sum(
        (right["budget_fraction"] - left["budget_fraction"])
        * (left["mean_f1_pp"] + right["mean_f1_pp"])
        / 2.0
        for left, right in zip(curve, curve[1:])
    )
    span = max_actions / total_actions
    return curve, area / span


def summarize(order: list[Action]) -> dict[str, float]:
    curve, area = action_curve(order)
    points = {int(row["actions"]): float(row["mean_f1_pp"]) for row in curve}
    return {
        "F1_at_10_pp": points[360],
        "F1_at_25_pp": points[900],
        "F1_at_50_pp": points[1800],
        "AUBPC_0_50_pp": area,
    }
