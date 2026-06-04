from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from ..contracts import Paper, QueryPlan

VENUE_ALIASES = {
    "ieee/cvf conference on computer vision and pattern recognition": "CVPR",
    "cvpr": "CVPR",
    "neurips": "NeurIPS",
    "iclr": "ICLR",
}


def retrieve_mock_papers(query_plan: QueryPlan, fixtures_dir: Path) -> tuple[list[Paper], list[dict[str, object]]]:
    payload = cast(
        dict[str, Any],
        json.loads((fixtures_dir / "mock_api" / "papers.json").read_text(encoding="utf-8")),
    )
    raw_papers = cast(list[dict[str, Any]], payload.get("papers", []))
    all_papers = [_paper_from_dict(item) for item in raw_papers]

    selected = [paper for paper in all_papers if _matches_query_plan(paper, query_plan)]
    if not selected:
        selected = all_papers[:5]

    edges = [
        cast(dict[str, object], edge)
        for edge in cast(list[dict[str, Any]], payload.get("edges", []))
        if isinstance(edge, dict)
    ]
    return selected, edges


def _paper_from_dict(item: dict[str, Any]) -> Paper:
    return Paper(
        id=str(item["id"]),
        source_ids=_string_dict(item.get("source_ids")),
        title=str(item["title"]).strip(),
        abstract=_optional_str(item.get("abstract")),
        authors=_author_list(item.get("authors")),
        year=_optional_int(item.get("year")),
        venue=_normalize_venue(_optional_str(item.get("venue"))),
        publication_date=_optional_str(item.get("publication_date")),
        url=_optional_str(item.get("url")),
        open_access_pdf=_optional_dict(item.get("open_access_pdf")),
        fields=_string_list(item.get("fields")),
        topics=_string_list(item.get("topics")),
        citation_count=_optional_int(item.get("citation_count")),
        reference_count=_optional_int(item.get("reference_count")),
        source_api=str(item.get("source_api", "mock")),
        retrieved_at=_optional_str(item.get("retrieved_at")) or datetime.now(timezone.utc).isoformat(),
        raw=_optional_dict(item.get("raw")) or {},
        pool_status=str(item.get("pool_status", "seed")),
    )


def _matches_query_plan(paper: Paper, query_plan: QueryPlan) -> bool:
    normalized_title = _normalize_text(paper.title)
    normalized_haystack = _normalize_text(
        " ".join(filter(None, [paper.title, paper.abstract or "", paper.venue or "", " ".join(paper.topics)]))
    )

    if not _matches_year(paper, query_plan):
        return False

    if query_plan.intent_analysis.query_type == "navigational":
        return any(_normalize_text(sub_query) in normalized_title for sub_query in query_plan.sub_queries_for_retrieval)

    token_hits = 0
    for sub_query in query_plan.sub_queries_for_retrieval:
        for token in re.split(r"\s+", sub_query.lower()):
            normalized_token = _normalize_text(token)
            if len(normalized_token) < 2:
                continue
            if normalized_token in normalized_haystack:
                token_hits += 1

    if query_plan.intent_analysis.query_type == "metadata":
        venue_terms = [venue.lower() for venue in query_plan.ranking_signals.get("preferred_venues", [])]
        venue_match = any(venue in (paper.venue or "").lower() for venue in venue_terms)
        return venue_match or token_hits >= 1

    return token_hits >= 2


def _matches_year(paper: Paper, query_plan: QueryPlan) -> bool:
    year_filter = query_plan.hard_filters.get("year")
    if not year_filter or paper.year is None:
        return True
    expected_year = year_filter.get("value")
    return isinstance(expected_year, int) and paper.year >= expected_year


def _normalize_venue(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return VENUE_ALIASES.get(normalized, value.strip())


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", value.lower())


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _optional_dict(value: Any) -> dict[str, Any] | None:
    return cast(dict[str, Any], value) if isinstance(value, dict) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _author_list(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    authors: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        authors.append({
            "name": _optional_str(item.get("name")) or "",
            "id": _optional_str(item.get("id")),
        })
    return authors


def _string_dict(value: Any) -> dict[str, str | None]:
    if not isinstance(value, dict):
        return {}
    return {str(key): _optional_str(item) for key, item in value.items()}
