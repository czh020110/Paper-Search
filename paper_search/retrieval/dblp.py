"""DBLP API client for paper search.

Uses the public DBLP API (no key required).
API docs: https://dblp.org/faq/13501473.html
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from ..contracts import Paper
from ..errors import ApiRateLimitError, classify_httpx_error, is_retryable
from .shared import get_shared_http_client

logger = logging.getLogger(__name__)

DBLP_BASE_URL = "https://dblp.org/search/publ/api"

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0
DBLP_TIMEOUT = 20.0
DBLP_DEFAULT_RETRY_AFTER = 10.0


def search_dblp(query: str, max_results: int = 20) -> list[Paper]:
    """Search DBLP publications via the public JSON API."""
    params: dict[str, Any] = {
        "q": query,
        "format": "json",
        "h": max_results,
    }
    url = f"{DBLP_BASE_URL}?{urlencode(params)}"
    payload = _request_with_retry(url)
    return _parse_response(payload, source_api="dblp")


def search_dblp_by_title(title: str, max_results: int = 5) -> list[Paper]:
    """Search DBLP by title field."""
    params: dict[str, Any] = {
        "q": title,
        "format": "json",
        "h": max_results,
    }
    url = f"{DBLP_BASE_URL}?{urlencode(params)}"
    payload = _request_with_retry(url)
    return _parse_response(payload, source_api="dblp")


def _request_with_retry(url: str) -> dict[str, Any]:
    """Fetch JSON response with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            client = get_shared_http_client()
            response = client.get(url, timeout=DBLP_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            classified = classify_httpx_error(e)
            if attempt == MAX_RETRIES or not is_retryable(classified):
                logger.warning("DBLP request failed after %d attempts: %s", MAX_RETRIES, classified)
                return {}

            if isinstance(classified, ApiRateLimitError):
                retry_after = classified.retry_after or DBLP_DEFAULT_RETRY_AFTER
                wait = max(retry_after, RETRY_BACKOFF * attempt)
            else:
                wait = RETRY_BACKOFF * attempt

            logger.warning("DBLP retry %d/%d after error: %s", attempt, MAX_RETRIES, classified)
            time.sleep(wait)
    return {}


def _parse_response(data: dict[str, Any], source_api: str = "dblp") -> list[Paper]:
    """Parse DBLP JSON response into Paper objects."""
    papers: list[Paper] = []
    try:
        hits = data.get("result", {}).get("hits", {})
        total = int(hits.get("@total", 0))
        if total == 0:
            return []
        hit_list = hits.get("hit", [])
        if not isinstance(hit_list, list):
            hit_list = [hit_list]
    except Exception as e:
        logger.warning("DBLP response parse error: %s", e)
        return []

    for hit in hit_list:
        try:
            info = hit.get("info", {})
            if not info:
                continue
            paper = _hit_to_paper(info, source_api)
            if paper:
                papers.append(paper)
        except Exception as e:
            logger.warning("DBLP hit parse error: %s", e)
            continue

    return papers


def _hit_to_paper(info: dict[str, Any], source_api: str) -> Paper | None:
    """Convert a DBLP hit info dict to a Paper dataclass."""
    title = _clean_str(info.get("title", ""))

    # Authors: can be a single string or a list
    authors_raw = info.get("authors", {})
    author_list = authors_raw.get("author") if isinstance(authors_raw, dict) else []
    if isinstance(author_list, str):
        author_list = [author_list]
    if not isinstance(author_list, list):
        author_list = []
    authors: list[dict[str, str | None]] = []
    for a in author_list:
        name = _clean_str(a) if isinstance(a, str) else _clean_str(a.get("text", ""))
        if name:
            authors.append({"name": name, "id": None})

    # Year
    year_raw = info.get("year")
    year: int | None = None
    if year_raw:
        try:
            year = int(str(year_raw).strip())
        except (ValueError, TypeError):
            pass

    # Venue
    venue = _clean_str(info.get("venue", "")) or None

    # DOI → construct URL
    doi = _clean_str(info.get("doi", ""))
    doi_url = f"https://doi.org/{doi}" if doi else None

    # URL from DBLP
    dblp_url = _clean_str(info.get("url", ""))

    # Key: e.g. "conf/cvpr/2023" — useful for identifying paper
    key = _clean_str(info.get("key", ""))

    fields: list[str] = []
    publ_type = _clean_str(info.get("type", ""))
    if publ_type:
        fields.append(publ_type)

    source_ids: dict[str, str | None] = {
        "doi": doi or None,
        "arxiv": None,
        "semantic_scholar": None,
        "openalex": None,
    }

    return Paper(
        id=f"dblp:{key}" if key else f"dblp:{doi or title[:40]}",
        source_ids=source_ids,
        title=title or "Untitled",
        abstract=None,
        authors=authors,
        year=year,
        venue=venue,
        publication_date=None,
        url=doi_url or dblp_url,
        open_access_pdf=None,
        fields=fields,
        topics=[],
        citation_count=None,
        reference_count=None,
        source_api=source_api,
        sources=[source_api],
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        raw=info,
        pool_status="seed",
    )


def _clean_str(value: Any) -> str:
    """Clean a string value, stripping whitespace and extra spaces."""
    if not value:
        return ""
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text