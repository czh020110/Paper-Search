"""Embedding client — provider-agnostic wrapper for text vectorization.

Supports DashScope and SiliconFlow providers, selected via
``EMBEDDING_PROVIDER`` environment variable.

Returns ``None`` when the provider is not configured or the call fails,
so callers can gracefully degrade.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

logger = logging.getLogger(__name__)

# DashScope text-embedding-v4 free tier: 30 QPS, ~60 QPM burst.
# We cap concurrent workers at 10 and enforce a max submission rate of
# *EMBEDDING_RPS_LIMIT* calls/sec via a token-bucket rate limiter.
EMBEDDING_CONCURRENCY = int(os.getenv("EMBEDDING_CONCURRENCY", "10"))
EMBEDDING_RPS_LIMIT = int(os.getenv("EMBEDDING_RPS_LIMIT", "15"))
EMBEDDING_ENABLED = os.getenv("EMBEDDING_ENABLED", "true").lower() not in ("0", "false", "no")


def get_embedding(text: str) -> list[float] | None:
    """Return a single embedding vector for *text*."""
    if not EMBEDDING_ENABLED:
        return None
    provider = os.getenv("EMBEDDING_PROVIDER", "")
    if not provider:
        logger.debug("No EMBEDDING_PROVIDER configured")
        return None

    if provider == "dashscope":
        return _dashscope_embed(text)
    elif provider == "siliconflow":
        return _siliconflow_embed(text)
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
    elif provider == "siliconflow":
        return _siliconflow_concurrent_embed(texts)
    else:
        return [None] * len(texts)


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


class _RateLimiter:
    """Token-bucket rate limiter for concurrent API calls.

    Multiple threads can call :meth:`acquire` concurrently — each acquires
    one token, blocking if necessary to stay within the configured rate.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._tokens = rate  # start full
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until one token is available, then consume it."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self._rate, self._tokens + elapsed * self._rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self._rate
            time.sleep(wait)


_embedding_limiter = _RateLimiter(EMBEDDING_RPS_LIMIT)


# ---------------------------------------------------------------------------
# DashScope provider
# ---------------------------------------------------------------------------


def _is_multimodal_model(model: str) -> bool:
    """Return True if *model* is a DashScope multi-modal embedding model.

    Multi-modal models (e.g. ``tongyi-embedding-vision-plus-*``) must be
    called via ``MultiModalEmbedding.call`` with ``input=[{'text': ...}]``.
    Plain text models use ``TextEmbedding.call`` with ``input=str``.
    """
    return model.startswith("tongyi-")


def _dashscope_embed(text: str) -> list[float] | None:
    import dashscope
    from http import HTTPStatus

    api_key = os.getenv("EMBEDDING_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")
    if not api_key:
        logger.warning("EMBEDDING_API_KEY not configured")
        return None

    try:
        if _is_multimodal_model(model):
            resp = dashscope.MultiModalEmbedding.call(
                model=model,
                input=[{"text": text}],
                api_key=api_key,
            )
        else:
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
    """Embed all texts concurrently, rate-limited to *EMBEDDING_RPS_LIMIT*.

    Each call to ``executor.submit`` is gated by a token-bucket rate
    limiter so the submission rate never exceeds the API quota.
    """
    results: list[list[float] | None] = [None] * len(texts)

    def _embed_one(index: int, text: str) -> tuple[int, list[float] | None]:
        return index, _dashscope_embed(text)

    with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
        futures: dict[Any, int] = {}
        for i, t in enumerate(texts):
            _embedding_limiter.acquire()
            futures[executor.submit(_embed_one, i, t)] = i
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


# ---------------------------------------------------------------------------
# SiliconFlow provider  (OpenAI-compatible API)
# ---------------------------------------------------------------------------

_SILICONFLOW_EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"


def _siliconflow_embed(text: str) -> list[float] | None:
    import httpx

    api_key = os.getenv("EMBEDDING_API_KEY")
    model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
    if not api_key:
        logger.warning("EMBEDDING_API_KEY not configured")
        return None

    try:
        resp = httpx.post(
            _SILICONFLOW_EMBED_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={"model": model, "input": text},
            timeout=30.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data.get("data", [])
            if items:
                return list(items[0].get("embedding", []))
        logger.error("SiliconFlow embedding failed: %s %s", resp.status_code, resp.text[:200])
    except Exception:
        logger.error("SiliconFlow embedding call failed", exc_info=True)
    return None


def _siliconflow_concurrent_embed(texts: list[str]) -> list[list[float] | None]:
    """Embed all texts concurrently, rate-limited to *EMBEDDING_RPS_LIMIT*."""
    results: list[list[float] | None] = [None] * len(texts)

    def _embed_one(index: int, text: str) -> tuple[int, list[float] | None]:
        return index, _siliconflow_embed(text)

    with ThreadPoolExecutor(max_workers=EMBEDDING_CONCURRENCY) as executor:
        futures: dict[Any, int] = {}
        for i, t in enumerate(texts):
            _embedding_limiter.acquire()
            futures[executor.submit(_embed_one, i, t)] = i
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
