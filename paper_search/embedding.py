"""Embedding client — provider-agnostic wrapper for text vectorization.

Currently supports DashScope (``text-embedding-v4``).  The provider is
selected via ``EMBEDDING_PROVIDER`` environment variable.

Returns ``None`` when the provider is not configured or the call fails,
so callers can gracefully degrade.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


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
    """Return embedding vectors for a batch of texts."""
    provider = os.getenv("EMBEDDING_PROVIDER", "")
    if not provider:
        return [None] * len(texts)

    if provider == "dashscope":
        return _dashscope_batch_embed(texts)
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


def _dashscope_batch_embed(texts: list[str]) -> list[list[float] | None]:
    """DashScope supports batched input natively."""
    import dashscope
    from http import HTTPStatus

    api_key = os.getenv("EMBEDDING_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    if not api_key:
        return [None] * len(texts)

    results: list[list[float] | None] = [None] * len(texts)
    chunk_size = 10  # text-embedding-v4 batch limit
    for offset in range(0, len(texts), chunk_size):
        chunk = texts[offset: offset + chunk_size]
        try:
            resp = dashscope.TextEmbedding.call(
                model=model,
                input=chunk,
                api_key=api_key,
            )
            if resp.status_code == HTTPStatus.OK:
                embeddings = resp.output.get("embeddings", [])
                for i, emb in enumerate(embeddings):
                    results[offset + i] = list(emb.get("embedding", []))
            else:
                logger.error("DashScope batch embedding failed: %s - %s", resp.status_code, resp.message)
        except Exception:
            logger.error("DashScope batch embedding failed for chunk %d", offset, exc_info=True)

    return results
