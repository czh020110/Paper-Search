from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, cast
from urllib.parse import urlencode

import httpx

from ..contracts import Paper
from .shared import VENUE_ALIASES

logger = logging.getLogger(__name__)

S2_BASE_URL = "https://api.semanticscholar.org/graph/v1"

S2_FIELDS = "paperId,title,abstract,authors,year,venue,publicationDate,url,openAccessPdf,fieldsOfStudy,citationCount,referenceCount,externalIds"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds, multiplied by attempt number


def search_papers(query: str, year_from: int | None = None, limit: int = 20) -> list[Paper]:
    params: dict[str, Any] = {
        "query": query,
        "fields": S2_FIELDS,
        "limit": limit,
    }
    if year_from is not None:
        params["year"] = f"{year_from}-"

    headers = _build_headers()
    url = f"{S2_BASE_URL}/paper/search?{urlencode(params)}"
    payload = _request_with_retry(url, headers)
    raw_papers = cast(list[dict[str, Any]], payload.get("data") or [])
    return [_paper_from_s2(item) for item in raw_papers]


def search_papers_by_title(title: str, limit: int = 10) -> list[Paper]:
    params: dict[str, Any] = {
        "query": title,
        "fields": S2_FIELDS,
        "limit": limit,
    }
    headers = _build_headers()
    url = f"{S2_BASE_URL}/paper/search?{urlencode(params)}"
    payload = _request_with_retry(url, headers)
    raw_papers = cast(list[dict[str, Any]], payload.get("data") or [])
    return [_paper_from_s2(item) for item in raw_papers]


def _request_with_retry(url: str, headers: dict[str, str]) -> dict[str, Any]:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(url, headers=headers, timeout=30.0)
            if response.status_code == 429:
                wait = RETRY_BACKOFF * attempt
                logger.warning("S2 rate limited (429), retrying in %.1fs (attempt %d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if attempt == MAX_RETRIES:
                logger.error("S2 request failed after %d retries: %s", MAX_RETRIES, e)
                return {}
            wait = RETRY_BACKOFF * attempt
            logger.warning("S2 HTTP error %d, retrying in %.1fs", e.response.status_code, wait)
            time.sleep(wait)
        except httpx.RequestError as e:
            if attempt == MAX_RETRIES:
                logger.error("S2 request error after %d retries: %s", MAX_RETRIES, e)
                return {}
            wait = RETRY_BACKOFF * attempt
            logger.warning("S2 request error, retrying in %.1fs: %s", wait, e)
            time.sleep(wait)
    return {}


def _paper_from_s2(item: dict[str, Any]) -> Paper:
    external_ids = cast(dict[str, Any], item.get("externalIds") or {})
    source_ids: dict[str, str | None] = {
        "semantic_scholar": _optional_str(item.get("paperId")),
        "doi": _optional_str(external_ids.get("DOI")),
        "arxiv": _optional_str(external_ids.get("ArXiv")),
        "openalex": _optional_str(external_ids.get("OpenAlex")),
    }

    authors = _s2_author_list(item.get("authors"))
    venue_raw = _optional_str(item.get("venue"))
    venue = _normalize_venue(venue_raw)

    open_access_pdf = None
    pdf_info = item.get("openAccessPdf")
    if isinstance(pdf_info, dict):
        open_access_pdf = {
            "url": _optional_str(pdf_info.get("url")),
            "status": _optional_str(pdf_info.get("status")),
        }

    return Paper(
        id=f"semantic_scholar:{item.get('paperId', '')}",
        source_ids=source_ids,
        title=_optional_str(item.get("title")) or "",
        abstract=_optional_str(item.get("abstract")),
        authors=authors,
        year=_optional_int(item.get("year")),
        venue=venue,
        publication_date=_optional_str(item.get("publicationDate")),
        url=_optional_str(item.get("url")),
        open_access_pdf=open_access_pdf,
        fields=_string_list(item.get("fieldsOfStudy")),
        topics=[],
        citation_count=_optional_int(item.get("citationCount")),
        reference_count=_optional_int(item.get("referenceCount")),
        source_api="semantic_scholar",
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        raw=item,
        pool_status="seed",
    )


def _build_headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def _normalize_venue(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    return VENUE_ALIASES.get(normalized, value.strip())


def _s2_author_list(value: Any) -> list[dict[str, str | None]]:
    if not isinstance(value, list):
        return []
    authors: list[dict[str, str | None]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        authors.append({
            "name": _optional_str(item.get("name")) or "",
            "id": _optional_str(item.get("authorId")),
        })
    return authors


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]