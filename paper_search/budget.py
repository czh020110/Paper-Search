from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BudgetLimits:
    max_api_calls: int | None = None
    max_tokens: int | None = None
    max_rounds: int | None = None
    max_pool_size: int | None = None
    max_batch_judgments: int | None = None


class BudgetController:
    """Enforces hard limits on API calls, token usage, iteration rounds, and pool size.

    When a limit is reached, the corresponding ``can_*`` method returns ``False``
    and logs the reason via the structured logger.  Callers decide whether to
    abort, degrade, or skip remaining work.

    修复：所有 check-then-act 操作加 threading.Lock 保护，防止并发场景下竞态条件
    """

    def __init__(self, limits: BudgetLimits) -> None:
        self.limits = limits
        self.api_calls_used: int = 0
        self.tokens_used: int = 0
        self.current_round: int = 0
        # 修复：添加互斥锁保护并发访问共享状态
        self._lock = threading.Lock()

    def can_call_api(self) -> bool:
        with self._lock:
            if self.limits.max_api_calls is None:
                return True
            if self.api_calls_used >= self.limits.max_api_calls:
                logger.warning(
                    "API call budget exceeded",
                    extra={"stage": "budget", "budget_reason": "max_api_calls", "limit": self.limits.max_api_calls, "used": self.api_calls_used},
                )
                return False
            return True

    def increment_api_calls(self) -> None:
        with self._lock:
            self.api_calls_used += 1

    def can_use_tokens(self, estimated: int) -> bool:
        with self._lock:
            if self.limits.max_tokens is None:
                return True
            if self.tokens_used + estimated > self.limits.max_tokens:
                logger.warning(
                    "Token budget would be exceeded",
                    extra={"stage": "budget", "budget_reason": "max_tokens", "limit": self.limits.max_tokens, "used": self.tokens_used, "estimated": estimated},
                )
                return False
            return True

    def record_tokens(self, used: int) -> None:
        with self._lock:
            self.tokens_used += used

    def can_expand_pool(self, current_size: int, additions: int) -> bool:
        with self._lock:
            if self.limits.max_pool_size is None:
                return True
            if current_size + additions > self.limits.max_pool_size:
                logger.warning(
                    "Pool size cap would be exceeded",
                    extra={"stage": "budget", "budget_reason": "max_pool_size", "limit": self.limits.max_pool_size, "current": current_size, "additions": additions},
                )
                return False
            return True

    def can_start_round(self) -> bool:
        with self._lock:
            if self.limits.max_rounds is None:
                return True
            if self.current_round >= self.limits.max_rounds:
                logger.warning(
                    "Round budget exceeded",
                    extra={"stage": "budget", "budget_reason": "max_rounds", "limit": self.limits.max_rounds, "current": self.current_round},
                )
                return False
            return True

    def increment_round(self) -> None:
        with self._lock:
            self.current_round += 1

    def usage_summary(self) -> dict[str, Any]:
        return {
            "api_calls_used": self.api_calls_used,
            "tokens_used": self.tokens_used,
            "current_round": self.current_round,
            "limits": {
                "max_api_calls": self.limits.max_api_calls,
                "max_tokens": self.limits.max_tokens,
                "max_rounds": self.limits.max_rounds,
                "max_pool_size": self.limits.max_pool_size,
                "max_batch_judgments": self.limits.max_batch_judgments,
            },
        }