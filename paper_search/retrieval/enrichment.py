"""Field enrichment: supplement missing venue/abstract by cross-source title search."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from ..contracts import Paper
from ..dedupe import normalize_title

if TYPE_CHECKING:
    from ..budget import BudgetController
    from ..cache import CacheStore

logger = logging.getLogger(__name__)


def enrich_papers(
    papers: list[Paper],
    cache: CacheStore | None = None,
    budget: BudgetController | None = None,
    max_concurrent: int = 4,
) -> list[Paper]:
    """Supplement missing venue/abstract by searching S2 and OA by title.

    Only papers with ``not paper.venue or not paper.abstract`` are queried;
    all other papers are left untouched.
    """
    needs_enrichment = [p for p in papers if not p.venue or not p.abstract]
    if not needs_enrichment:
        return papers

    logger.info(
        "Enriching %d papers with missing fields (out of %d total)",
        len(needs_enrichment),
        len(papers),
        extra={"stage": "enrichment", "papers_to_enrich": len(needs_enrichment)},
    )

    def _enrich_one(paper: Paper) -> Paper:
        return _enrich_paper(paper, cache=cache, budget=budget)

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {executor.submit(_enrich_one, p): p for p in needs_enrichment}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                logger.warning("Enrichment failed for a paper", exc_info=True)

    return papers


def _enrich_paper(
    paper: Paper,
    cache: CacheStore | None = None,
    budget: BudgetController | None = None,
) -> Paper:
    """Try to fill missing fields for a single paper via S2 then OA."""
    # Try S2 first — usually the most complete for abstract/venue
    if budget is None or budget.can_call_api():
        _try_enrich_from_s2(paper, cache=cache)
        if budget:
            budget.increment_api_calls()

    # If still missing fields, try OA
    if (not paper.venue or not paper.abstract) and (budget is None or budget.can_call_api()):
        _try_enrich_from_oa(paper, cache=cache)
        if budget:
            budget.increment_api_calls()

    return paper


def _try_enrich_from_s2(paper: Paper, cache: CacheStore | None = None) -> None:
    """Search S2 by title and backfill missing fields if a match is found."""
    from .semantic_scholar import search_papers_by_title

    try:
        results = search_papers_by_title(paper.title, limit=3, cache=cache)
    except Exception:
        logger.debug("S2 enrichment search failed for: %s", paper.title[:60])
        return

    match = _find_title_match(paper.title, results)
    if match:
        _backfill(paper, match, source_label="semantic_scholar")


def _try_enrich_from_oa(paper: Paper, cache: CacheStore | None = None) -> None:
    """Search OA by DOI or exact title and backfill missing fields if a match is found."""
    from .openalex import get_work_by_doi, search_works_by_exact_title, search_works_by_title

    doi = paper.source_ids.get("doi")
    if doi:
        try:
            by_doi = get_work_by_doi(doi, cache=cache)
        except Exception:
            logger.debug("OA enrichment DOI lookup failed for: %s", paper.title[:60])
        else:
            if by_doi is not None:
                _backfill(paper, by_doi, source_label="openalex")
                return

    try:
        results = search_works_by_exact_title(paper.title, per_page=3, cache=cache)
    except Exception:
        logger.debug("OA exact-title enrichment failed for: %s", paper.title[:60])
        results = []

    match = _find_title_match(paper.title, results)
    if match:
        _backfill(paper, match, source_label="openalex")
        return

    try:
        fallback_results = search_works_by_title(paper.title, per_page=3, cache=cache)
    except Exception:
        logger.debug("OA fallback title search failed for: %s", paper.title[:60])
        return

    fallback_match = _find_title_match(paper.title, fallback_results)
    if fallback_match:
        _backfill(paper, fallback_match, source_label="openalex")


def _find_title_match(original_title: str, candidates: list[Paper]) -> Paper | None:
    """Find a candidate whose normalized title matches the original."""
    norm_original = normalize_title(original_title)
    for candidate in candidates:
        if normalize_title(candidate.title) == norm_original:
            return candidate
    return None


def _backfill(paper: Paper, source: Paper, source_label: str) -> None:
    """Backfill missing fields from *source* into *paper* (in-place)."""
    if not paper.venue and source.venue:
        paper.venue = source.venue
    if not paper.abstract and source.abstract:
        paper.abstract = source.abstract
    elif paper.abstract and source.abstract and len(source.abstract) > len(paper.abstract):
        paper.abstract = source.abstract
    if not paper.authors and source.authors:
        paper.authors = source.authors
    if not paper.citation_count and source.citation_count:
        paper.citation_count = source.citation_count
    if not paper.reference_count and source.reference_count:
        paper.reference_count = source.reference_count
    if not paper.publication_date and source.publication_date:
        paper.publication_date = source.publication_date
    if not paper.url and source.url:
        paper.url = source.url
    # Merge source_ids
    for key, value in source.source_ids.items():
        if value and not paper.source_ids.get(key):
            paper.source_ids[key] = value
    # Track enrichment source
    if source_label not in paper.sources:
        paper.sources.append(source_label)
