from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import urlencode

import httpx

from ..cache import CacheStore
from ..contracts import Paper
from ..errors import classify_httpx_error, is_retryable
from .shared import VENUE_ALIASES

logger = logging.getLogger(__name__)

OA_BASE_URL = "https://api.openalex.org"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0


def search_works(query: str, year_from: int | None = None, per_page: int = 25, cache: CacheStore | None = None) -> list[Paper]:
    papers: list[Paper] = []
    cursor = "*"
    while cursor:
        params: dict[str, Any] = {
            "search": query,
            "per_page": per_page,
            "cursor": cursor,
        }
        if year_from is not None:
            params["filter"] = f"publication_year:>{year_from - 1}"

        mailto = os.getenv("OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto

        url = f"{OA_BASE_URL}/works?{urlencode(params, doseq=True)}"
        payload = _request_with_retry(url, cache=cache)
        raw_works = cast(list[dict[str, Any]], payload.get("results") or [])
        papers.extend(_paper_from_oa(item) for item in raw_works)
        cursor = cast(str | None, payload.get("meta", {}).get("cursor"))
        if not cursor or len(raw_works) == 0:
            break
    return papers


def search_works_by_title(title: str, per_page: int = 10, cache: CacheStore | None = None) -> list[Paper]:
    papers: list[Paper] = []
    cursor = "*"
    while cursor:
        params: dict[str, Any] = {
            "search": title,
            "per_page": per_page,
            "cursor": cursor,
        }

        mailto = os.getenv("OPENALEX_MAILTO")
        if mailto:
            params["mailto"] = mailto

        url = f"{OA_BASE_URL}/works?{urlencode(params, doseq=True)}"
        payload = _request_with_retry(url, cache=cache)
        raw_works = cast(list[dict[str, Any]], payload.get("results") or [])
        papers.extend(_paper_from_oa(item) for item in raw_works)
        cursor = cast(str | None, payload.get("meta", {}).get("cursor"))
        if not cursor or len(raw_works) == 0:
            break
    return papers


def _request_with_retry(url: str, cache: CacheStore | None = None) -> dict[str, Any]:
    if cache is not None:
        cached_body = cache.get("api_responses", url)
        if cached_body is not None:
            logger.info("OA cache hit: %s", url[:80])
            return cached_body

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(url, timeout=30.0)
            if response.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                logger.warning("OA rate limited (429), retrying in %.1fs (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            response.raise_for_status()
            body = response.json()
            if cache is not None:
                cache.put("api_responses", url, data=body)
            return body
        except httpx.HTTPStatusError as e:
            classified = classify_httpx_error(e)
            if not is_retryable(classified) or attempt == MAX_RETRIES:
                logger.error("OA request failed (not retryable or max retries): %s", classified)
                return {}
            wait = RETRY_BACKOFF * attempt
            logger.warning("OA HTTP error %d (%s), retrying in %.1fs", e.response.status_code, type(classified).__name__, wait)
            time.sleep(wait)
        except httpx.RequestError as e:
            classified = classify_httpx_error(e)
            if not is_retryable(classified) or attempt == MAX_RETRIES:
                logger.error("OA request error (not retryable or max retries): %s", classified)
                return {}
            wait = RETRY_BACKOFF * attempt
            logger.warning("OA request error (%s), retrying in %.1fs: %s", type(classified).__name__, wait, e)
            time.sleep(wait)
    return {}


def _paper_from_oa(item: dict[str, Any]) -> Paper:
    ids = cast(dict[str, Any], item.get("ids") or {})
    source_ids: dict[str, str | None] = {
        "openalex": _optional_str(ids.get("openalex")),
        "doi": _normalize_doi(_optional_str(ids.get("doi"))),
        "arxiv": _optional_str(ids.get("arxiv")),
        "semantic_scholar": None,
    }

    authors = _oa_author_list(item.get("authorships"))
    venue = _oa_venue(item.get("primary_location"))

    open_access_pdf = None
    best_oa = _best_oa_location(item.get("open_access"), item.get("oa_locations"))
    if best_oa:
        open_access_pdf = best_oa

    topics = _oa_topics(item.get("topics"))
    concepts = _oa_concepts(item.get("concepts"))

    return Paper(
        id=f"openalex:{ids.get('openalex', '')}",
        source_ids=source_ids,
        title=_optional_str(item.get("title")) or _optional_str(item.get("display_name")) or "",
        abstract=_reconstruct_abstract(item.get("abstract_inverted_index")),
        authors=authors,
        year=_optional_int(item.get("publication_year")),
        venue=venue,
        publication_date=_optional_str(item.get("publication_date")),
        url=_optional_str(item.get("doi")) or _optional_str(ids.get("openalex")),
        open_access_pdf=open_access_pdf,
        fields=concepts,
        topics=topics,
        citation_count=_optional_int(item.get("cited_by_count")),
        reference_count=_optional_int(item.get("referenced_works_count")),
        source_api="openalex",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        raw=item,
        pool_status="seed",
    )


def _reconstruct_abstract(inverted_index: Any) -> str | None:
    if not isinstance(inverted_index, dict):
        return None
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    word_positions.append((pos, word))
    if not word_positions:
        return None
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def _oa_author_list(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    authors: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if not isinstance(author, dict):
            continue
        authors.append({
            "name": _optional_str(author.get("display_name")) or "",
            "id": _optional_str(author.get("id")),
        })
    return authors


def _oa_venue(primary_location: Any) -> str | None:
    if not isinstance(primary_location, dict):
        return None
    source = primary_location.get("source")
    if not isinstance(source, dict):
        return None
    venue_raw = _optional_str(source.get("display_name"))
    return _normalize_venue(venue_raw)


def _best_oa_location(open_access: Any, oa_locations: Any) -> dict[str, Any] | None:
    if isinstance(oa_locations, list):
        for loc in oa_locations:
            if isinstance(loc, dict) and loc.get("pdf_url"):
                return {
                    "url": _optional_str(loc.get("pdf_url")),
                    "status": _optional_str(loc.get("oa_status")) or "open",
                }
    if isinstance(open_access, dict):
        oa_url = _optional_str(open_access.get("oa_url"))
        if oa_url:
            return {"url": oa_url, "status": _optional_str(open_access.get("oa_status")) or "open"}
    return None


def _oa_topics(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_optional_str(item.get("display_name")) or "" for item in value if isinstance(item, dict) and item.get("display_name")]


def _oa_concepts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [_optional_str(item.get("display_name")) or "" for item in value if isinstance(item, dict) and item.get("display_name")]


def _normalize_venue(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return VENUE_ALIASES.get(normalized, value.strip())


def _normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None