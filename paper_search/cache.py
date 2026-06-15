from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

CACHE_DOMAINS = ("paper_metadata", "citations", "embeddings", "llm_judgments", "api_responses")

# TTL defaults per domain (seconds)
DEFAULT_TTL: dict[str, int | None] = {
    "paper_metadata": 86400 * 7,  # 7 days — paper metadata rarely changes
    "citations": 86400 * 3,  # 3 days
    "embeddings": None,  # no expiry — invalidated by version only
    "llm_judgments": None,  # no expiry — invalidated by prompt version only
    "api_responses": 86400,  # 1 day — API responses can change
}


class CacheStore:
    """File-based cache organized by domain subdirectories under *cache_dir*.

    Each entry is stored as ``{domain}/{sha256(key)}.json`` with a sidecar
    ``{domain}/{sha256(key)}.meta.json`` holding TTL and version metadata.

    Cache keys are derived from domain + arbitrary string parts via SHA-256.
    They never include API keys, auth headers, or secrets.
    """

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()
        for domain in CACHE_DOMAINS:
            (cache_dir / domain).mkdir(parents=True, exist_ok=True)

    def _make_hash(self, domain: str, *parts: str) -> str:
        raw = domain + ":" + ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, domain: str, *parts: str) -> Any | None:
        """Read a cached value. Returns ``None`` on miss or expired entry."""
        h = self._make_hash(domain, *parts)
        data_path = self.cache_dir / domain / f"{h}.json"
        meta_path = self.cache_dir / domain / f"{h}.meta.json"

        with self._lock:
            if not data_path.exists():
                self._misses += 1
                return None

            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                expires_at = meta.get("expires_at")
                if expires_at is not None and time.time() > expires_at:
                    data_path.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    self._misses += 1
                    logger.info("Cache entry expired: %s/%s", domain, h[:12])
                    return None

            self._hits += 1
            return json.loads(data_path.read_text(encoding="utf-8"))

    def put(
        self,
        domain: str,
        *parts: str,
        data: Any,
        ttl_seconds: int | None = None,
        version: str | None = None,
    ) -> None:
        """Write a cached value with optional TTL and version metadata."""
        # --- 写入前的基本校验 ---
        if not isinstance(data, dict) or not data:
            logger.warning(
                "Cache put skipped: data is %s (expected non-empty dict), domain=%s",
                type(data).__name__, domain,
            )
            return
        if isinstance(data.get("status_code"), int) and data["status_code"] != 200:
            logger.warning(
                "Cache put skipped: status_code=%s (expected 200), domain=%s",
                data["status_code"], domain,
            )
            return

        h = self._make_hash(domain, *parts)
        data_path = self.cache_dir / domain / f"{h}.json"
        meta_path = self.cache_dir / domain / f"{h}.meta.json"

        if ttl_seconds is None:
            ttl_seconds = DEFAULT_TTL.get(domain)

        meta: dict[str, Any] = {}
        if ttl_seconds is not None:
            meta["expires_at"] = time.time() + ttl_seconds
        if version is not None:
            meta["version"] = version

        with self._lock:
            data_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            meta_path.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    def invalidate(self, domain: str, version: str) -> int:
        """Delete all entries in *domain* whose version does not match *version*."""
        removed = 0
        domain_dir = self.cache_dir / domain
        if not domain_dir.is_dir():
            return removed
        with self._lock:
            for meta_file in domain_dir.glob("*.meta.json"):
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                if meta.get("version") != version:
                    h = meta_file.name[:-len(".meta.json")]
                    data_file = domain_dir / f"{h}.json"
                    data_file.unlink(missing_ok=True)
                    meta_file.unlink(missing_ok=True)
                    removed += 1
        return removed

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses}

    def hit_rate(self) -> float:
        with self._lock:
            total = self._hits + self._misses
            return self._hits / total if total > 0 else 0.0


class _CachedResponse:
    """Minimal httpx.Response-like object backed by cached JSON data."""

    def __init__(self, body: Any) -> None:
        self._body = body
        self.status_code = 200

    def json(self) -> Any:
        return self._body

    def raise_for_status(self) -> None:
        pass  # cached responses are always successful


def cached_get(
    cache: CacheStore,
    url: str,
    headers: dict[str, str] | None = None,
    timeout: float = 30.0,
    ttl_seconds: int | None = None,
) -> httpx.Response | _CachedResponse:
    """Attempt to fetch *url* from *cache* first; on miss, call ``httpx.get`` and cache the result.

    The cache key is derived from the URL only — headers and auth are
    intentionally excluded from the key to prevent secrets from leaking
    into cache filenames.
    """
    cached_body = cache.get("api_responses", url)
    if cached_body is not None:
        logger.info("Cache hit: %s", url[:80])
        return _CachedResponse(cached_body)

    logger.info("Cache miss: %s", url[:80])
    response = httpx.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()

    body = response.json()
    cache.put("api_responses", url, data=body, ttl_seconds=ttl_seconds)
    return response