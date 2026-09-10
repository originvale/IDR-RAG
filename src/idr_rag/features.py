"""Frozen 32-dimensional pre-action retrieval--reasoning state.

The implementation mirrors the feature dictionary reported in Appendix B of
the accompanying manuscript. Gold answers and future trajectory states are
never read by this module.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

import numpy as np

TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

FEATURE_NAMES = [
    "depth",
    "resolved_count",
    "unresolved_count",
    "resolved_fraction",
    "evidence_support_count",
    "support_per_resolved",
    "answer_token_count",
    "answer_char_count",
    "reasoning_progress_token_count",
    "reasoning_progress_char_count",
    "next_information_need_token_count",
    "next_search_query_token_count",
    "cot_step_count",
    "cot_token_count",
    "passage_count",
    "unique_passage_ratio",
    "unique_title_ratio",
    "evidence_vocab_size",
    "retrieval_score_mean",
    "retrieval_score_max",
    "retrieval_score_std",
    "question_query_jaccard",
    "question_answer_jaccard",
    "answer_evidence_jaccard",
    "query_evidence_jaccard",
    "progress_evidence_jaccard",
    "need_evidence_jaccard",
    "support_evidence_jaccard",
    "answer_support_jaccard",
    "unresolved_query_jaccard",
    "resolved_evidence_jaccard",
    "unresolved_evidence_jaccard",
]


def _tokens(value: Any) -> set[str]:
    return {
        token.lower()
        for token in TOKEN_RE.findall(str(value or ""))
        if len(token) > 1
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def _support_text(items: list[Any]) -> str:
    parts: list[str] = []
    for item in items:
        if isinstance(item, dict):
            parts.extend(
                str(item.get(key) or "")
                for key in ("passage_id", "text", "claim", "support")
            )
        else:
            parts.append(str(item))
    return " ".join(parts)


def features(state: dict[str, Any], depth: int) -> list[float]:
    """Construct the frozen 32D state for the next legal action."""

    question = _tokens(state.get("question"))
    query = _tokens(state.get("current_query"))
    answer = _tokens(state.get("candidate_answer"))
    progress = _tokens(state.get("reasoning_progress"))
    need = _tokens(state.get("next_information_need"))
    resolved_values = [str(value) for value in state.get("resolved_subgoals") or []]
    unresolved_values = [str(value) for value in state.get("unresolved_subgoals") or []]
    resolved = _tokens(" ".join(resolved_values))
    unresolved = _tokens(" ".join(unresolved_values))
    supports = list(state.get("evidence_support") or [])
    support_tokens = _tokens(_support_text(supports))
    cot_values = list(state.get("cot_so_far") or [])
    cot = _tokens(" ".join(str(value) for value in cot_values))
    passages = [
        passage
        for passage in state.get("passages") or []
        if isinstance(passage, dict)
    ]
    passage_ids = [str(p.get("id") or p.get("passage_id") or "") for p in passages]
    titles = [str(p.get("title") or "") for p in passages]
    evidence = _tokens(
        " ".join(
            f"{p.get('title') or ''} {p.get('paragraph_text') or p.get('text') or ''}"
            for p in passages
        )
    )
    scores: list[float] = []
    for passage in passages:
        try:
            raw = passage.get("score")
            scores.append(float(raw if raw is not None else passage.get("retrieval_score") or 0.0))
        except (TypeError, ValueError):
            scores.append(0.0)

    resolved_count = len(resolved_values)
    unresolved_count = len(unresolved_values)
    total_subgoals = resolved_count + unresolved_count
    values = [
        float(depth),
        float(resolved_count),
        float(unresolved_count),
        resolved_count / total_subgoals if total_subgoals else 1.0,
        float(len(supports)),
        len(supports) / max(1, resolved_count),
        float(len(answer)),
        float(len(str(state.get("candidate_answer") or ""))),
        float(len(progress)),
        float(len(str(state.get("reasoning_progress") or ""))),
        float(len(need)),
        float(len(query)),
        float(len(cot_values)),
        float(len(cot)),
        float(len(passages)),
        len(set(passage_ids)) / len(passages) if passages else 0.0,
        len(set(titles)) / len(passages) if passages else 0.0,
        float(len(evidence)),
        _mean(scores),
        max(scores) if scores else 0.0,
        float(np.std(scores)) if scores else 0.0,
        _jaccard(question, query),
        _jaccard(question, answer),
        _jaccard(answer, evidence),
        _jaccard(query, evidence),
        _jaccard(progress, evidence),
        _jaccard(need, evidence),
        _jaccard(support_tokens, evidence),
        _jaccard(answer, support_tokens),
        _jaccard(unresolved, query),
        _jaccard(resolved, evidence),
        _jaccard(unresolved, evidence),
    ]
    if len(values) != 32 or not np.isfinite(np.asarray(values, dtype=float)).all():
        raise ValueError("state did not produce 32 finite features")
    return values
