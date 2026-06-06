"""Embedding client — provider-agnostic wrapper for text vectorization.

Currently supports DashScope (``text-embedding-v4``).  The provider is
selected via ``EMBEDDING_PROVIDER`` environment variable.

Returns ``None`` when the provider is not configured or the call fails,
so callers can gracefully degrade.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_CONCURRENCY = 30


def get_embedding(text: str) -> list[float] | None:
    """Return a single embedding vector for *text*."""
    provider = os.getenv("EMBEDDING_PROVIDER", "")
    if not provider:
        logger.debug("No EMBEDDING_PROVIDER configured")
        return None

    if provider == "dashscope":
        return _dashscope_embed(text)
    else:
        logger.warning("Unknown EMBEDDING_PROVIDER: %s", provider)
        return None


def batch_embeddings(texts: list[str]) -> list[list[float] | None]:
    """Return embedding vectors for a batch of texts via concurrent single-text calls."""
    provider = os.getenv("EMBEDDING_PROVIDER", "")
    if not provider:
        return [None] * len(texts)

    if provider == "dashscope":
        return _dashscope_concurrent_embed(texts)
    else:
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# DashScope provider
# ---------------------------------------------------------------------------


def _dashscope_embed(text: str) -> list[float] | None:
    import dashscope
    from http import HTTPStatus

    api_key = os.getenv("EMBEDDING_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    if not api_key:
        logger.warning("EMBEDDING_API_KEY not configured")
        return None

    try:
        resp = dashscope.TextEmbedding.call(
            model=model,
            input=text,
            api_key=api_key,
        )
        if resp.status_code == HTTPStatus.OK:
            embeddings = resp.output.get("embeddings", [])
            if embeddings:
                return list(embeddings[0].get("embedding", []))
        logger.error("DashScope embedding failed: %s - %s", resp.status_code, resp.message)
    except Exception:
        logger.error("DashScope embedding call failed", exc_info=True)
    return None


def _dashscope_concurrent_embed(texts: list[str]) -> list[list[float] | None]:
    """Embed all texts concurrently (each as a single-text API call).

    Uses a ThreadPoolExecutor with *EMBEDDING_CONCURRENCY* workers to
    maximise throughput while staying within API rate limits.
    """
    results: list[list[float] | None] = [None] * len(texts)

    def _embed_one(index: int, text: str) -> tuple[int, list[float] | None]:
        return index, _dashscope_embed(text)

    with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
        futures = {executor.submit(_embed_one, i, t): i for i, t in enumerate(texts)}
        for future in as_completed(futures):
            try:
                idx, emb = future.result()
                results[idx] = emb
            except Exception:
                logger.warning("Embedding task failed for index %d", futures[future], exc_info=True)

    ok = sum(1 for r in results if r is not None)
    if ok < len(texts):
        logger.info("Embedding: %d/%d succeeded", ok, len(texts))
    return results
