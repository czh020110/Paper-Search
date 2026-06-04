from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .contracts import Paper


def evaluate_query(query: str, papers: list[Paper], golden_set_path: Path) -> dict[str, Any]:
    if not golden_set_path.exists():
        return {
            "dataset": "unknown",
            "included_relevance_levels": ["highly_relevant_papers", "partially_relevant_papers"],
            "precision": None,
            "recall": None,
            "f1": None,
            "expected_ids": [],
            "matched_ids": [],
            "missing_ids": [],
            "highly_relevant_ids": [],
            "partially_relevant_ids": [],
        }

    payload = cast(dict[str, Any], json.loads(golden_set_path.read_text(encoding="utf-8")))
    cases = cast(list[dict[str, Any]], payload.get("cases", []))
    matched_case = next((case for case in cases if case.get("query") == query), None)
    if not matched_case:
        return {
            "dataset": payload.get("dataset", "golden_set_v1"),
            "included_relevance_levels": ["highly_relevant_papers", "partially_relevant_papers"],
            "precision": None,
            "recall": None,
            "f1": None,
            "expected_ids": [],
            "matched_ids": [],
            "missing_ids": [],
            "highly_relevant_ids": [],
            "partially_relevant_ids": [],
        }

    highly_relevant_ids = {
        item for item in matched_case.get("highly_relevant_papers", []) if isinstance(item, str)
    }
    partially_relevant_ids = {
        item for item in matched_case.get("partially_relevant_papers", []) if isinstance(item, str)
    }
    expected_ids = highly_relevant_ids | partially_relevant_ids
    actual_ids = {paper.id for paper in papers}
    matched_ids = expected_ids & actual_ids

    precision = len(matched_ids) / len(actual_ids) if actual_ids else None
    recall = len(matched_ids) / len(expected_ids) if expected_ids else None
    f1 = _f1(precision, recall)

    return {
        "dataset": payload.get("dataset", "golden_set_v1"),
        "included_relevance_levels": ["highly_relevant_papers", "partially_relevant_papers"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "expected_ids": sorted(expected_ids),
        "matched_ids": sorted(matched_ids),
        "missing_ids": sorted(expected_ids - actual_ids),
        "highly_relevant_ids": sorted(highly_relevant_ids),
        "partially_relevant_ids": sorted(partially_relevant_ids),
    }


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
