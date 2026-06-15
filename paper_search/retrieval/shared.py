"""Shared constants and utilities for retrieval backends."""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import httpx

from ..cache import CacheStore
from ..errors import classify_httpx_error, is_retryable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 会议/期刊别名映射
# ---------------------------------------------------------------------------

VENUE_ALIASES = {
    "ieee/cvf conference on computer vision and pattern recognition": "CVPR",
    "cvpr": "CVPR",
    "neurips": "NeurIPS",
    "iclr": "ICLR",
    "acl": "ACL",
    "emnlp": "EMNLP",
    # OA returns "arXiv (Cornell University)" as venue
    "arxiv (cornell university)": "arXiv",
}


# ---------------------------------------------------------------------------
# 通用类型转换工具函数
# 从 semantic_scholar.py / openalex.py / mock_backend.py 抽取而来，统一导出。
# ---------------------------------------------------------------------------


def optional_str(value: object) -> str | None:
    """Return *value* if it is a ``str``, otherwise ``None``."""
    return value if isinstance(value, str) else None


def optional_int(value: object) -> int | None:
    """Return *value* if it is an ``int``, otherwise ``None``."""
    return value if isinstance(value, int) else None


def string_list(value: object) -> list[str]:
    """Return a list of ``str`` items from *value*; non-list or non-str items are dropped."""
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


# ---------------------------------------------------------------------------
# 向后兼容别名 — 原文件内以 _optional_str / _optional_int / _string_list 形式
# 被内部引用，改为从 shared 导入后仍保持下划线前缀以避免 API 变更。
# ---------------------------------------------------------------------------

_optional_str = optional_str
_optional_int = optional_int
_string_list = string_list


# ---------------------------------------------------------------------------
# 共享 httpx.Client 单例 — 连接池复用，避免每次请求新建连接
# ---------------------------------------------------------------------------

_shared_client: httpx.Client | None = None
_shared_client_lock = threading.Lock()


def get_shared_http_client() -> httpx.Client:
    """返回全局共享的 httpx.Client 单例。

    线程安全的懒初始化，连接池上限 50，超时 30 秒。
    所有 retrieval backend 应使用此函数获取 client，而非直接调用 ``httpx.get``。
    """
    global _shared_client
    if _shared_client is None:
        with _shared_client_lock:
            if _shared_client is None:
                _shared_client = httpx.Client(
                    limits=httpx.Limits(max_connections=50),
                    timeout=httpx.Timeout(30.0),
                )
    return _shared_client


# ---------------------------------------------------------------------------
# 通用 HTTP 请求重试逻辑
# 从 semantic_scholar.py / openalex.py 抽取而来，统一重试 + 错误分类 + 缓存。
# 各 backend 仍保留本地 _request_with_retry 薄包装以保持调用签名不变，
# 但核心逻辑集中在此处，避免重复代码。
# ---------------------------------------------------------------------------


def request_with_retry(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    cache: CacheStore | None = None,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    timeout: float = 30.0,
    label: str = "API",
) -> dict[str, Any]:
    """带重试和缓存的通用 HTTP GET 请求。

    统一处理 429 限流、分类错误重试、缓存命中逻辑。
    返回 JSON dict；请求失败时返回空 dict。

    Parameters
    ----------
    url : str
        请求 URL
    headers : dict | None
        请求头（如 S2 的 x-api-key）
    cache : CacheStore | None
        可选缓存，命中时直接返回
    max_retries : int
        最大重试次数
    retry_backoff : float
        重试退避基数（秒），实际等待 = retry_backoff * attempt
    timeout : float
        请求超时时间（秒）
    label : str
        日志标签，如 "S2"、"OA"，便于区分来源
    """
    if cache is not None:
        cached_body = cache.get("api_responses", url)
        if cached_body is not None:
            logger.info("%s cache hit: %s", label, url[:80])
            return cached_body

    for attempt in range(1, max_retries + 1):
        try:
            client = get_shared_http_client()
            response = client.get(url, headers=headers, timeout=timeout)
            if response.status_code == 429:
                wait = retry_backoff * attempt
                logger.warning(
                    "%s rate limited (429), retrying in %.1fs (attempt %d/%d)",
                    label, wait, attempt, max_retries,
                )
                time.sleep(wait)
                continue
            response.raise_for_status()
            body = response.json()
            if cache is not None:
                cache.put("api_responses", url, data=body)
            return body
        except httpx.HTTPStatusError as e:
            classified = classify_httpx_error(e)
            if not is_retryable(classified) or attempt == max_retries:
                logger.error(
                    "%s request failed (not retryable or max retries): %s",
                    label, classified,
                )
                return {}
            wait = retry_backoff * attempt
            logger.warning(
                "%s HTTP error %d (%s), retrying in %.1fs",
                label, e.response.status_code, type(classified).__name__, wait,
            )
            time.sleep(wait)
        except httpx.RequestError as e:
            classified = classify_httpx_error(e)
            if not is_retryable(classified) or attempt == max_retries:
                logger.error(
                    "%s request error (not retryable or max retries): %s",
                    label, classified,
                )
                return {}
            wait = retry_backoff * attempt
            logger.warning(
                "%s request error (%s), retrying in %.1fs: %s",
                label, type(classified).__name__, wait, e,
            )
            time.sleep(wait)
    return {}
