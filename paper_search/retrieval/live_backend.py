from __future__ import annotations

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any, Callable

from ..budget import BudgetController
from ..cache import CacheStore
from ..contracts import Paper, QueryPlan
from .arxiv import search_arxiv, search_arxiv_by_title
from .dblp import search_dblp, search_dblp_by_title
from .openalex import search_works, search_works_by_title
from .semantic_scholar import search_papers, search_papers_by_title

if TYPE_CHECKING:
    from ..schemas import OAFilterSchema, S2PayloadSchema

logger = logging.getLogger(__name__)

S2_DELAY_SECONDS = 1.0
# DBLP has no API key and strict rate-limiting (HTTP 429) — must run sequentially with delay.
# arXiv can handle moderate concurrency (5 queries tested OK), so runs in parallel.
DBLP_DELAY_SECONDS = 1.0

# Chinese tokens that should be stripped from API queries (APIs index English papers)
_CJK_PATTERN = re.compile(r"[一-鿿㐀-䶿]+")


def _is_source_enabled(name: str) -> bool:
    """Check if a search source is enabled via env var. Default: true."""
    return os.getenv(name, "true").lower() not in ("0", "false", "no")


def retrieve_live_papers(query_plan: QueryPlan, cache: CacheStore | None = None, budget: BudgetController | None = None) -> tuple[list[Paper], list[dict[str, Any]]]:
    query_type = query_plan.intent_analysis.query_type

    s2_callables: list[Callable[[], list[Paper]]] = []
    oa_callables: list[Callable[[], list[Paper]]] = []
    arxiv_callables: list[Callable[[], list[Paper]]] = []
    dblp_callables: list[Callable[[], list[Paper]]] = []

    year_from = _extract_year_from(query_plan)

    if query_type == "navigational":
        for sub_query in query_plan.sub_queries_for_retrieval:
            if _is_source_enabled("SEARCH_SOURCE_S2"):
                s2_callables.append(lambda q=sub_query: search_papers_by_title(title=q, limit=10, cache=cache))
            if _is_source_enabled("SEARCH_SOURCE_OA"):
                oa_callables.append(lambda q=sub_query: search_works_by_title(title=q, per_page=10, cache=cache))
            if _is_source_enabled("SEARCH_SOURCE_ARXIV"):
                arxiv_callables.append(lambda q=sub_query: search_arxiv_by_title(title=q, max_results=5))
            if _is_source_enabled("SEARCH_SOURCE_DBLP"):
                dblp_callables.append(lambda q=sub_query: search_dblp_by_title(title=q, max_results=5))
    else:
        # Prefer api_payload_translation when available, but strip CJK from query text
        # since S2/OA are English-focused APIs that return poor results for Chinese queries
        if _is_source_enabled("SEARCH_SOURCE_S2"):
            s2_payloads = query_plan.api_payload_translation.get("semantic_scholar", [])
            if s2_payloads:
                for entry in s2_payloads:
                    s2_p = _parse_s2_payload(entry, year_from)
                    if not s2_p:
                        continue
                    s2_callables.append(lambda p=s2_p: search_papers(query=p.query, s2_payload=p, cache=cache))
        if _is_source_enabled("SEARCH_SOURCE_OA"):
            oa_payloads = query_plan.api_payload_translation.get("openalex", [])
            if oa_payloads:
                for entry in oa_payloads:
                    oa_kw, oa_filter_obj, oa_sort, oa_pp = _parse_oa_payload(entry, year_from)
                    if not oa_kw and oa_filter_obj is None:
                        continue
                    has_any_filter = oa_filter_obj and any(v is not None for v in oa_filter_obj.model_dump().values())
                    if not oa_kw and not has_any_filter:
                        continue
                    oa_callables.append(
                        lambda kw=oa_kw, fi=oa_filter_obj, so=oa_sort, pp=oa_pp: search_works(
                            query=kw, oa_filter=fi, sort=so, per_page=pp, cache=cache
                        )
                    )

        # Fallback: build English queries from semantic_queries when api_payload_translation is empty
        if _is_source_enabled("SEARCH_SOURCE_S2") and not s2_callables:
            english_queries = _build_english_queries(query_plan)
            for query in english_queries:
                s2_callables.append(lambda q=query, y=year_from: search_papers(query=q, year_from=y, limit=20, cache=cache))
        if _is_source_enabled("SEARCH_SOURCE_OA") and not oa_callables:
            english_queries = _build_english_queries(query_plan)
            from ..schemas import OAFilterSchema
            fallback_filter = OAFilterSchema(publication_year=f">{year_from - 1}") if year_from else None
            for query in english_queries:
                oa_callables.append(lambda q=query, fi=fallback_filter: search_works(query=q, oa_filter=fi, per_page=25, cache=cache))

    # arXiv: semantic search engine — sub_queries_for_retrieval serves as intent summary
    if _is_source_enabled("SEARCH_SOURCE_ARXIV") and not arxiv_callables:
        for query in (query_plan.sub_queries_for_retrieval or [query_plan.original_query]):
            arxiv_callables.append(lambda q=query: search_arxiv(query=q, max_results=20))
    # DBLP: title-based exact match — use short keywords (≤3 words) from core_concepts
    # Long sentence queries ("LLM jailbreak attack methods 2026") return 0 hits.
    if _is_source_enabled("SEARCH_SOURCE_DBLP") and not dblp_callables:
        core = query_plan.semantic_queries.get("core_concepts", [])
        short_kw: list[str] = []
        for kw in core:
            wc = len(kw.split())
            if wc <= 3 and kw not in short_kw:
                short_kw.append(kw)
        if not short_kw:
            # Fallback: tokenize sub_queries into 1-2 word pairs
            for sq in (query_plan.sub_queries_for_retrieval or []):
                tokens = sq.split()[:4]  # first 4 words → up to 2 bigrams
                for i in range(len(tokens) - 1):
                    pair = f"{tokens[i]} {tokens[i + 1]}"
                    if pair not in short_kw:
                        short_kw.append(pair)
                if len(short_kw) >= 5:
                    break
        for kw in short_kw[:5]:  # limit to 5 queries for rate-limited free API
            dblp_callables.append(lambda q=kw: search_dblp(query=q, max_results=20))

    all_papers: list[Paper] = []

    # S2 rate-limits: without an API key ~1 rps → sequential with delay;
    # with a key it rises to ~100 rps → we can fan out in parallel.
    has_s2_key = bool(os.getenv("SEMANTIC_SCHOLAR_API_KEY"))

    # Submit S2 + OA + arXiv + DBLP tasks into the same thread-pool.
    max_workers = max(4, len(s2_callables) + len(oa_callables) + len(arxiv_callables) + len(dblp_callables))
    if not has_s2_key:
        # No S2 key → run S2 sequentially in one thread to stay under 1 rps.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            s2_future = executor.submit(_run_s2_sequential, s2_callables, budget, has_s2_key) if s2_callables else None
            oa_futures = [executor.submit(fn) for fn in oa_callables]
            ax_futures = [executor.submit(fn) for fn in arxiv_callables]
            db_future = executor.submit(_run_free_api_sequential, dblp_callables, "DBLP", len(dblp_callables), DBLP_DELAY_SECONDS, budget) if dblp_callables else None
            if s2_future:
                s2_result = s2_future.result()
                if isinstance(s2_result, list):
                    all_papers.extend(s2_result)
            if db_future:
                try:
                    r = db_future.result()
                    if isinstance(r, list):
                        all_papers.extend(r)
                except Exception:
                    pass
            for futures_list in (oa_futures, ax_futures):
                for future in as_completed(futures_list):
                    try:
                        result = future.result()
                        if isinstance(result, list):
                            all_papers.extend(result)
                    except Exception:
                        pass
    else:
        # S2 key available → all tasks run in parallel.
        def _call_one(fn: Callable[[], list[Paper]], label: str, idx: int, total: int) -> list[Paper]:
            if budget and not budget.can_call_api():
                logger.info("%s skipped task %d/%d (budget exhausted)", label, idx + 1, total)
                return []
            try:
                r = fn()
                if isinstance(r, list):
                    if budget and label in ("S2", "OA"):
                        budget.increment_api_calls()
                    logger.info("%s retrieved %d papers (task %d/%d)", label, len(r), idx + 1, total)
                    return r
            except Exception:
                logger.warning("%s task %d/%d failed", label, idx + 1, total, exc_info=True)
            return []

        s2_total = len(s2_callables)
        oa_total = len(oa_callables)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[Any, None] = {}
            for i, fn in enumerate(s2_callables):
                futures[executor.submit(_call_one, fn, "S2", i, s2_total)] = None
            for i, fn in enumerate(oa_callables):
                futures[executor.submit(_call_one, fn, "OA", i, oa_total)] = None
            for i, fn in enumerate(arxiv_callables):
                futures[executor.submit(_call_one, fn, "arXiv", i, len(arxiv_callables))] = None
            if dblp_callables:
                futures[executor.submit(_run_free_api_sequential, dblp_callables, "DBLP", len(dblp_callables), DBLP_DELAY_SECONDS, budget)] = None
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, list):
                        all_papers.extend(result)
                except Exception:
                    pass

    edges = _build_edges(all_papers)
    return all_papers, edges


def _run_free_api_sequential(
    callables: list[Callable[[], list[Paper]]],
    label: str,
    total: int,
    delay: float,
    budget: BudgetController | None,
) -> list[Paper]:
    """Run free-API tasks sequentially with *delay* between requests to avoid rate-limiting."""
    papers: list[Paper] = []
    for i, fn in enumerate(callables):
        if budget and not budget.can_call_api():
            logger.info("%s skipped task %d/%d (budget exhausted)", label, i + 1, total)
            break
        if i > 0:
            time.sleep(delay)
        try:
            r = fn()
            if isinstance(r, list):
                papers.extend(r)
                logger.info("%s retrieved %d papers (task %d/%d)", label, len(r), i + 1, total)
        except Exception:
            logger.warning("%s task %d/%d failed", label, i + 1, total, exc_info=True)
    return papers


def _run_s2_sequential(
    callables: list[Callable[[], list[Paper]]],
    budget: BudgetController | None,
    has_key: bool,
) -> list[Paper]:
    """Run S2 tasks sequentially with delay (for no-API-key rate limit)."""
    s2_papers: list[Paper] = []
    for i, fn in enumerate(callables):
        if budget and not budget.can_call_api():
            break
        if i > 0 and not has_key:
            time.sleep(S2_DELAY_SECONDS)
        try:
            result = fn()
            if isinstance(result, list):
                s2_papers.extend(result)
                if budget:
                    budget.increment_api_calls()
                logger.info("S2 retrieved %d papers (task %d/%d)", len(result), i + 1, len(callables))
        except Exception:
            logger.warning("S2 task %d/%d failed", i + 1, len(callables), exc_info=True)
    return s2_papers


def _build_english_queries(query_plan: QueryPlan) -> list[str]:
    """Extract English-only search queries from semantic_queries, avoiding CJK tokens that APIs can't handle."""
    core = query_plan.semantic_queries.get("core_concepts", [])
    methods = query_plan.semantic_queries.get("methodologies", [])

    # Filter out CJK-only tokens
    english_core = [t for t in core if not _CJK_PATTERN.search(t)]
    english_methods = [t for t in methods if not _CJK_PATTERN.search(t)]

    if not english_core:
        # Fallback: strip CJK from sub_queries — methods alone produce garbage queries
        return [_strip_cjk(q) for q in query_plan.sub_queries_for_retrieval if _strip_cjk(q)]

    # Build query variants from English tokens, deduplicating tokens first
    queries: list[str] = []

    # Variant 1: core concepts (deduplicated)
    unique_core = list(dict.fromkeys(english_core))[:6]
    queries.append(" ".join(unique_core))

    # Variant 2: core + methods (deduplicated, interleaved)
    if english_methods:
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


def _has_alpha(text: str) -> bool:
    """Return True if *text* contains at least one alphabetic character [a-zA-Z]."""
    import re as _re
    return bool(_re.search(r"[a-zA-Z]", text))


def _extract_year_from(query_plan: QueryPlan) -> int | None:
    year_filter = query_plan.hard_filters.get("year")
    if isinstance(year_filter, dict):
        value = year_filter.get("value")
        if isinstance(value, int):
            return value
    return None


def _parse_oa_payload(
    entry: dict[str, Any], default_year_from: int | None
) -> tuple[str | None, "OAFilterSchema | None", str | None, int]:
    """Parse an OA payload entry from api_payload_translation.

    Supports both the new structured format (filter as dict with OAFilterSchema
    fields) and the legacy format (filter as a raw string like
    'publication_year:>2021').

    Returns (search_query, oa_filter, sort, per_page).
    """
    from ..schemas import OAFilterSchema

    # search keyword
    q_raw = entry.get("search", "")
    q = _strip_cjk(q_raw) if q_raw else None
    if q and not _has_alpha(q):
        q = None

    # sort
    sort = entry.get("sort")

    # per_page
    per_page = entry.get("per_page", 25)
    if not isinstance(per_page, int) or per_page < 1:
        per_page = 25

    # filter — structured dict or legacy string
    raw_filter = entry.get("filter")
    if isinstance(raw_filter, dict):
        # New structured format — build OAFilterSchema from dict
        try:
            oa_filter = OAFilterSchema(**raw_filter)
        except Exception:
            logger.warning("OA payload filter parse failed, using fallback year filter")
            oa_filter = OAFilterSchema(publication_year=_oa_normalize_year(f">{default_year_from - 1}") if default_year_from else None)
    elif isinstance(raw_filter, str) and raw_filter:
        # Legacy format — extract publication_year expression directly from raw filter string
        # e.g. "publication_year:>2021" → publication_year=">2021"
        import re as _re
        year_match = _re.search(r"publication_year:([^,]+)", raw_filter)
        oa_filter = OAFilterSchema(publication_year=year_match.group(1) if year_match else None)
    else:
        # No filter provided — use default year_from
        oa_filter = OAFilterSchema(publication_year=_oa_normalize_year(f">{default_year_from - 1}") if default_year_from else None)

    # Normalize publication_year from S2-style formats to OA-compatible format
    if oa_filter and oa_filter.publication_year:
        oa_filter.publication_year = _oa_normalize_year(oa_filter.publication_year)

    # OA publication_year only accepts a single year or closed range (e.g. "2022", "2020-2024").
    # Open ranges like ">2021" are invalid for publication_year — must use from_publication_date.
    if oa_filter and oa_filter.publication_year:
        import re as _re
        if _re.match(r"^>\d{4}$", oa_filter.publication_year):
            # Convert ">2021" → from_publication_date="2022-01-01"
            year_val = int(oa_filter.publication_year[1:]) + 1
            if not oa_filter.from_publication_date:
                oa_filter.from_publication_date = f"{year_val}-01-01"
            oa_filter.publication_year = None

    return q, oa_filter, sort, per_page


def _oa_normalize_year(raw: str) -> str:
    """Normalize a year expression to OpenAlex-compatible format.

    LLM may output S2-style or generic year expressions that OA does not understand.
    OA ``publication_year`` only accepts a single year (``2022``) or a closed range
    (``2022-2026``).  Open ranges (``>=``) must use ``from_publication_date`` instead.

    Conversion examples:
      "2022-"  → ">2021"  (delegated to from_publication_date downstream)
      ">=2022" → ">2021"  (delegated to from_publication_date downstream)
      ">2021"  → ">2021"  (already OA-compatible for from_publication_date)
      "2020-2024" → "2020-2024"  (already OA-compatible for publication_year)
    """
    import re as _re
    raw = raw.strip()
    # S2 open range: "2022-"
    m = _re.match(r"^(\d{4})-$", raw)
    if m:
        return f">{int(m.group(1)) - 1}"
    # >= syntax: ">=2022" → ">2021"
    m = _re.match(r"^>=(\d{4})$", raw)
    if m:
        return f">{int(m.group(1)) - 1}"
    return raw


def _parse_s2_payload(entry: dict[str, Any], default_year_from: int | None) -> "S2PayloadSchema | None":
    """Parse an S2 payload entry from api_payload_translation.

    Supports both the new structured format (with S2PayloadSchema fields)
    and the legacy format (only query + year).

    Returns S2PayloadSchema or None if query is unusable.
    """
    from ..schemas import S2PayloadSchema

    q_raw = entry.get("query", "")
    q = _strip_cjk(q_raw) if q_raw else ""
    if not q or not _has_alpha(q):
        return None

    # year: prefer explicit year field, then fallback to default_year_from
    year = entry.get("year")
    if not year and default_year_from is not None:
        year = f"{default_year_from}-"

    try:
        raw_limit = entry.get("limit", 20)
        limit = raw_limit if isinstance(raw_limit, int) and 1 <= raw_limit <= 1000 else 20
        return S2PayloadSchema(
            query=q,
            year=year,
            venue=entry.get("venue"),
            fields_of_study=entry.get("fields_of_study"),
            publication_types=entry.get("publication_types"),
            min_citation_count=entry.get("min_citation_count"),
            open_access_pdf=entry.get("open_access_pdf"),
            sort=entry.get("sort"),
            limit=limit,
        )
    except Exception as exc:
        logger.warning("S2 payload parse failed (%s), using query + year fallback", exc)
        return S2PayloadSchema(query=q, year=year)


def _build_edges(papers: list[Paper]) -> list[dict[str, Any]]:
    # Citation edges will be populated during the snowball phase (S-005).
    # The initial retrieval stage only produces seed papers; edges are built
    # when references/citations are fetched and filtered for relevance.
    _ = papers  # kept for future snowball-phase edge extraction
    return []