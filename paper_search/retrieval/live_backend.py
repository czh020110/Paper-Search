from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from ..contracts import Paper, QueryPlan
from .openalex import _paper_from_oa, search_works, search_works_by_title
from .semantic_scholar import search_papers, search_papers_by_title

logger = logging.getLogger(__name__)

S2_DELAY_SECONDS = 1.0

# Chinese tokens that should be stripped from API queries (APIs index English papers)
_CJK_PATTERN = re.compile(r"[一-鿿㐀-䶿]+")


def retrieve_live_papers(query_plan: QueryPlan) -> tuple[list[Paper], list[dict[str, Any]]]:
    query_type = query_plan.intent_analysis.query_type

    s2_callables: list[Callable[[], list[Paper]]] = []
    oa_callables: list[Callable[[], list[Paper]]] = []

    year_from = _extract_year_from(query_plan)

    if query_type == "navigational":
        for sub_query in query_plan.sub_queries_for_retrieval:
            s2_callables.append(lambda q=sub_query: search_papers_by_title(title=q, limit=10))
            oa_callables.append(lambda q=sub_query: search_works_by_title(title=q, per_page=10))
    else:
        # Build English-only search queries from semantic_queries
        english_queries = _build_english_queries(query_plan)

        for query in english_queries:
            s2_callables.append(lambda q=query, y=year_from: search_papers(query=q, year_from=y, limit=20))
            oa_callables.append(lambda q=query, y=year_from: search_works(query=q, year_from=y, per_page=25))

    all_papers: list[Paper] = []

    # S2: sequential with delay between requests (respect rate limits without API key)
    has_s2_key = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))
    for i, fn in enumerate(s2_callables):
        if i > 0 and not has_s2_key:
            time.sleep(S2_DELAY_SECONDS)
        try:
            result = fn()
            if isinstance(result, list):
                all_papers.extend(result)
                logger.info("S2 retrieved %d papers (task %d/%d)", len(result), i + 1, len(s2_callables))
        except Exception:
            logger.warning("S2 task %d/%d failed", i + 1, len(s2_callables), exc_info=True)

    # OA: parallel (polite pool with mailto is generous)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn): idx for idx, fn in enumerate(oa_callables)}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if isinstance(result, list):
                    all_papers.extend(result)
                    logger.info("OA retrieved %d papers (task %d/%d)", len(result), idx + 1, len(oa_callables))
            except Exception:
                logger.warning("OA task %d/%d failed", idx + 1, len(oa_callables), exc_info=True)

    edges = _build_edges(all_papers)
    return all_papers, edges


def _build_english_queries(query_plan: QueryPlan) -> list[str]:
    """Extract English-only search queries from semantic_queries, avoiding CJK tokens that APIs can't handle."""
    core = query_plan.semantic_queries.get("core_concepts", [])
    methods = query_plan.semantic_queries.get("methodologies", [])

    # Filter out CJK-only tokens
    english_core = [t for t in core if not _CJK_PATTERN.search(t)]
    english_methods = [t for t in methods if not _CJK_PATTERN.search(t)]

    if not english_core and not english_methods:
        # Fallback: strip CJK from sub_queries
        return [_strip_cjk(q) for q in query_plan.sub_queries_for_retrieval if _strip_cjk(q)]

    # Build query variants from English tokens, deduplicating tokens first
    queries: list[str] = []

    # Variant 1: core concepts (deduplicated)
    if english_core:
        unique_core = list(dict.fromkeys(english_core))[:6]
        queries.append(" ".join(unique_core))

    # Variant 2: core + methods (deduplicated, interleaved)
    if english_core and english_methods:
        combined = list(dict.fromkeys(english_core[:4] + english_methods[:2]))[:6]
        queries.append(" ".join(combined))

    # Variant 3: methods only
    if english_methods:
        unique_methods = list(dict.fromkeys(english_methods))[:4]
        queries.append(" ".join(unique_methods))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for q in queries:
        if q not in seen:
            seen.add(q)
            unique.append(q)

    return unique or [_strip_cjk(query_plan.original_query)]


def _strip_cjk(text: str) -> str:
    """Remove CJK characters from text."""
    return _CJK_PATTERN.sub("", text).strip()


def _extract_year_from(query_plan: QueryPlan) -> int | None:
    year_filter = query_plan.hard_filters.get("year")
    if isinstance(year_filter, dict):
        value = year_filter.get("value")
        if isinstance(value, int):
            return value
    return None


def _build_edges(papers: list[Paper]) -> list[dict[str, Any]]:
    return []