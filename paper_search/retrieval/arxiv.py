"""arXiv API client for paper search.

Uses the public arXiv API (no key required).
API docs: https://info.arxiv.org/help/api/index.html
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import httpx

from ..contracts import Paper

logger = logging.getLogger(__name__)

ARXIV_BASE_URL = "https://export.arxiv.org/api/query"

# arXiv returns Atom XML with a specific namespace
_ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

MAX_RETRIES = 3
RETRY_BACKOFF = 1.0


def search_arxiv(query: str, max_results: int = 20, year_from: int | None = None) -> list[Paper]:
    """Search arXiv papers via the public API.

    Returns a list of Paper objects parsed from the Atom XML response.
    """
    params: dict[str, Any] = {
        "search_query": f"all:{query}",
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    url = f"{ARXIV_BASE_URL}?{urlencode(params)}"
    payload = _request_with_retry(url)
    return _parse_atom_response(payload, source_api="arxiv")


def search_arxiv_by_title(title: str, max_results: int = 5) -> list[Paper]:
    """Search arXiv by title (ti: field)."""
    params: dict[str, Any] = {
        "search_query": f"ti:{title}",
        "max_results": max_results,
    }
    url = f"{ARXIV_BASE_URL}?{urlencode(params)}"
    payload = _request_with_retry(url)
    return _parse_atom_response(payload, source_api="arxiv")


def _request_with_retry(url: str) -> str:
    """Fetch the Atom XML response with retry logic."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = httpx.get(url, timeout=30.0)
            response.raise_for_status()
            return response.text
        except (httpx.HTTPStatusError, httpx.RequestError) as e:
            if attempt == MAX_RETRIES:
                logger.warning("arXiv request failed after %d attempts: %s", MAX_RETRIES, e)
                return ""
            wait = RETRY_BACKOFF * attempt
            logger.warning("arXiv retry %d/%d after error: %s", attempt, MAX_RETRIES, e)
            import time
            time.sleep(wait)
    return ""


def _parse_atom_response(xml_text: str, source_api: str = "arxiv") -> list[Paper]:
    """Parse arXiv Atom XML into Paper objects."""
    if not xml_text:
        return []

    papers: list[Paper] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("arXiv XML parse error: %s", e)
        return []

    for entry in root.findall("atom:entry", _ARXIV_NS):
        try:
            paper = _entry_to_paper(entry, source_api)
            if paper:
                papers.append(paper)
        except Exception as e:
            logger.warning("arXiv entry parse error: %s", e)
            continue

    return papers


def _entry_to_paper(entry: ET.Element, source_api: str) -> Paper | None:
    """Convert an Atom entry element to a Paper dataclass."""
    # ID: use arXiv ID
    entry_id_el = entry.find("atom:id", _ARXIV_NS)
    full_id = entry_id_el.text if entry_id_el is not None else ""
    # Extract arXiv ID from URL like "http://arxiv.org/abs/2301.12345v1"
    arxiv_id = full_id.strip().rsplit("/", 1)[-1] if full_id else str(uuid.uuid4())[:8]

    # Title
    title_el = entry.find("atom:title", _ARXIV_NS)
    title = _clean_xml_text(title_el.text if title_el is not None else "")

    # Abstract
    abstract_el = entry.find("atom:summary", _ARXIV_NS)
    abstract = _clean_xml_text(abstract_el.text if abstract_el is not None else "")

    # Authors
    authors: list[dict[str, str | None]] = []
    for author_el in entry.findall("atom:author", _ARXIV_NS):
        name_el = author_el.find("atom:name", _ARXIV_NS)
        name = _clean_xml_text(name_el.text if name_el is not None else "")
        if name:
            authors.append({"name": name, "id": None})

    # Published date → year
    published_el = entry.find("atom:published", _ARXIV_NS)
    year: int | None = None
    publication_date: str | None = None
    if published_el is not None and published_el.text:
        publication_date = published_el.text[:10]
        match = re.match(r"(\d{4})", published_el.text)
        if match:
            year = int(match.group(1))

    # Venue (from arXiv journal_ref if available)
    journal_ref_el = entry.find("arxiv:journal_ref", _ARXIV_NS)
    venue = _clean_xml_text(journal_ref_el.text if journal_ref_el is not None else "") or None

    # URL
    url = f"https://arxiv.org/abs/{arxiv_id}"

    # Categories / fields
    fields: list[str] = []
    for cat_el in entry.findall("arxiv:primary_category", _ARXIV_NS):
        term = cat_el.get("term", "")
        if term:
            fields.append(term)

    # Link to PDF
    open_access_pdf: dict[str, Any] | None = {
        "url": f"https://arxiv.org/pdf/{arxiv_id}",
        "status": "open",
    }

    # Count references (not available from arXiv API directly)
    citation_count: int | None = None
    reference_count: int | None = None

    source_ids: dict[str, str | None] = {
        "arxiv": arxiv_id,
        "doi": None,
        "semantic_scholar": None,
        "openalex": None,
    }

    return Paper(
        id=f"arxiv:{arxiv_id}",
        source_ids=source_ids,
        title=title or "Untitled",
        abstract=abstract[:2000] if abstract else None,
        authors=authors,
        year=year,
        venue=venue,
        publication_date=publication_date,
        url=url,
        open_access_pdf=open_access_pdf,
        fields=fields,
        topics=[],
        citation_count=citation_count,
        reference_count=reference_count,
        source_api=source_api,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        raw={},
        pool_status="seed",
    )


def _clean_xml_text(text: str | None) -> str:
    """Clean XML text by stripping whitespace and normalizing newlines."""
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()