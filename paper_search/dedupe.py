from __future__ import annotations

import re
from typing import Iterable

from .contracts import Paper


def dedupe_papers(papers: Iterable[Paper]) -> list[Paper]:
    deduped: list[Paper] = []
    seen_to_index: dict[str, int] = {}

    for paper in papers:
        keys = _paper_keys(paper)
        matched_index = next((seen_to_index[key] for key in keys if key in seen_to_index), None)
        if matched_index is None:
            deduped.append(paper)
            index = len(deduped) - 1
            for key in keys:
                seen_to_index[key] = index
            continue

        merged = _merge_paper(deduped[matched_index], paper)
        deduped[matched_index] = merged
        for key in _paper_keys(merged):
            seen_to_index[key] = matched_index

    return deduped


def _paper_keys(paper: Paper) -> list[str]:
    source_ids = paper.source_ids
    keys = [
        _with_prefix("doi", source_ids.get("doi")),
        _with_prefix("arxiv", source_ids.get("arxiv")),
        _with_prefix("semantic_scholar", source_ids.get("semantic_scholar")),
        _with_prefix("openalex", source_ids.get("openalex")),
        _with_prefix("title", _normalize_title(paper.title)),
    ]
    return [key for key in keys if key]


def _merge_paper(primary: Paper, candidate: Paper) -> Paper:
    if len(candidate.abstract or "") > len(primary.abstract or ""):
        primary.abstract = candidate.abstract
    if len(candidate.authors) > len(primary.authors):
        primary.authors = candidate.authors
    if candidate.venue and not primary.venue:
        primary.venue = candidate.venue
    if candidate.url and not primary.url:
        primary.url = candidate.url
    if candidate.publication_date and not primary.publication_date:
        primary.publication_date = candidate.publication_date
    if candidate.citation_count and (primary.citation_count or 0) < candidate.citation_count:
        primary.citation_count = candidate.citation_count
    if candidate.reference_count and (primary.reference_count or 0) < candidate.reference_count:
        primary.reference_count = candidate.reference_count
    for key, value in candidate.source_ids.items():
        if value and not primary.source_ids.get(key):
            primary.source_ids[key] = value
    if len(candidate.fields) > len(primary.fields):
        primary.fields = candidate.fields
    if len(candidate.topics) > len(primary.topics):
        primary.topics = candidate.topics
    return primary


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9一-鿿]+", "", title.lower().strip())


def _with_prefix(prefix: str, value: str | None) -> str | None:
    if not value:
        return None
    return f"{prefix}:{value}"
