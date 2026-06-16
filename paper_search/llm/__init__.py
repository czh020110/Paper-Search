from __future__ import annotations

import logging
import os
import threading
from typing import Any, Literal

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 修复：Token 用量追踪 callback — 修复 token_usage 全 0 的问题
# ---------------------------------------------------------------------------


class TokenUsageCallback(BaseCallbackHandler):
    """LangChain callback handler that tracks prompt/completion token usage.

    修复 token_usage 全 0：所有 LLM 调用自动通过此 callback 记录 token 用量，
    pipeline 阶段汇总到 stage_metrics。
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Extract token usage from LLM response metadata."""
        try:
            # LangChain ChatOpenAI 返回 response.llm_output 或 generations[].message
            llm_output = getattr(response, "llm_output", None)
            if llm_output and isinstance(llm_output, dict):
                usage = llm_output.get("token_usage", {})
                with self._lock:
                    self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                    self.completion_tokens += usage.get("completion_tokens", 0) or 0
                return

            # Fallback: 从 generations 中提取 response_metadata
            generations = getattr(response, "generations", [])
            for gen_list in generations:
                for gen in gen_list:
                    meta = getattr(gen, "generation_info", None) or {}
                    if not meta:
                        msg = getattr(gen, "message", None)
                        if msg:
                            meta = getattr(msg, "response_metadata", {}) or {}
                    usage = meta.get("token_usage", {}) or meta.get("usage", {})
                    if usage:
                        with self._lock:
                            self.prompt_tokens += usage.get("prompt_tokens", 0) or 0
                            self.completion_tokens += usage.get("completion_tokens", 0) or 0
                        return
        except Exception:
            logger.debug("TokenUsageCallback: failed to extract token usage", exc_info=True)

    def get_usage(self) -> dict[str, int]:
        """Return current token usage summary."""
        with self._lock:
            return {
                "prompt": self.prompt_tokens,
                "completion": self.completion_tokens,
                "total": self.prompt_tokens + self.completion_tokens,
            }

    def reset(self) -> None:
        """Reset token counters."""
        with self._lock:
            self.prompt_tokens = 0
            self.completion_tokens = 0


# 模块级全局 callback 实例，所有 LLM 调用共享
_token_callback = TokenUsageCallback()


def get_token_callback() -> TokenUsageCallback:
    """Return the global token usage callback instance."""
    return _token_callback

# Thinking budget presets — provider-agnostic.
# Each provider maps these to its native parameter.
ThinkingLevel = Literal["off", "none", "minimal", "low", "medium", "high", "xhigh"]

# OpenAI reasoning_effort mapping
_OPENAI_REASONING_MAP: dict[str, str | None] = {
    "off": None,       # 默认 — don't pass the parameter
    "none": "none",    # 关闭
    "minimal": "minimal",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
}

# DashScope enable_thinking mapping (None = don't pass, False = off, True = on)
_DASHSCOPE_THINKING_MAP: dict[str, bool | None] = {
    "off": None,       # 默认 — don't pass the parameter
    "none": False,     # 关闭
    "minimal": False,
    "low": True,
    "medium": True,
    "high": True,
    "xhigh": True,
}

# SiliconFlow thinking mapping (same as DashScope — enable_thinking via extra_body)
_SILICONFLOW_THINKING_MAP: dict[str, bool | None] = {
    "off": None,
    "none": False,
    "minimal": False,
    "low": True,
    "medium": True,
    "high": True,
    "xhigh": True,
}

# ZhipuAI thinking mapping (supports thinking via extra_body for GLM models)
_ZHIPUAI_THINKING_MAP: dict[str, bool | None] = {
    "off": None,
    "none": False,
    "minimal": False,
    "low": True,
    "medium": True,
    "high": True,
    "xhigh": True,
}


def _resolve_provider() -> str:
    """Return the current LLM provider: ``openai``, ``dashscope``, ``siliconflow``, or ``zhipuai``."""
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _resolve_api_key(provider: str) -> str | None:
    if provider == "dashscope":
        return os.getenv("DASHSCOPE_API_KEY")
    if provider == "siliconflow":
        return os.getenv("SILICONFLOW_API_KEY")
    if provider == "zhipuai":
        return os.getenv("ZHIPUAI_API_KEY")
    # openai (default)
    return os.getenv("OPENAI_API_KEY")


def is_llm_key_configured() -> bool:
    """Check whether the API key for the current LLM_PROVIDER is set."""
    provider = _resolve_provider()
    return bool(_resolve_api_key(provider))


def _resolve_base_url(provider: str) -> str:
    # OPENAI_BASE_URL only applies to the openai provider.
    # DashScope / SiliconFlow / ZhipuAI use hardcoded default endpoints.
    if provider == "openai":
        return os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    _DEFAULTS = {
        "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "siliconflow": "https://api.siliconflow.cn/v1",
        "zhipuai": "https://open.bigmodel.cn/api/paas/v4",
    }
    return _DEFAULTS.get(provider, "https://api.openai.com/v1")


def _resolve_model(fast: bool = False) -> str:
    if fast:
        return os.getenv("LLM_FAST_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"
    return os.getenv("LLM_MODEL") or os.getenv("LLM_FAST_MODEL") or "gpt-4o-mini"


def _get_thinking_level() -> ThinkingLevel:
    raw = os.getenv("LLM_THINKING", "none").lower()
    if raw in ("off", "none", "minimal", "low", "medium", "high", "xhigh"):
        return raw  # type: ignore[return-value]
    return "none"


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Return a LangChain ChatOpenAI instance configured from environment.

    Uses ``LLM_PROVIDER`` to switch between backends.  All providers use
    ``ChatOpenAI`` — only *base_url*, *api_key*, and thinking params differ.

    Supported providers: openai, dashscope, siliconflow, zhipuai.

    ``LLM_THINKING`` controls reasoning/thinking depth:
      - OpenAI: reasoning_effort parameter
      - DashScope / SiliconFlow / ZhipuAI: enable_thinking via extra_body
    """
    from langchain_openai import ChatOpenAI

    provider = _resolve_provider()
    api_key = _resolve_api_key(provider)
    api_base = _resolve_base_url(provider)

    if model is None:
        model = _resolve_model(fast=False)

    thinking: ThinkingLevel = _get_thinking_level()

    # Providers that use enable_thinking via extra_body (DashScope / SiliconFlow / ZhipuAI)
    _THINKING_MAPS: dict[str, dict[str, bool | None]] = {
        "dashscope": _DASHSCOPE_THINKING_MAP,
        "siliconflow": _SILICONFLOW_THINKING_MAP,
        "zhipuai": _ZHIPUAI_THINKING_MAP,
    }

    if provider in _THINKING_MAPS:
        enable_thinking = _THINKING_MAPS[provider].get(thinking)
        kwargs_ds: dict[str, Any] = dict(
            model=model,
            temperature=temperature,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=api_base,
            callbacks=[_token_callback],  # 修复：注入 token 用量追踪 callback
        )
        if enable_thinking is not None:
            kwargs_ds["extra_body"] = {"enable_thinking": enable_thinking}
        return ChatOpenAI(**kwargs_ds)  # type: ignore[call-arg]

    # OpenAI provider (default)
    reasoning = _OPENAI_REASONING_MAP.get(thinking)
    kwargs: dict[str, Any] = dict(
        model=model,
        temperature=temperature,
        api_key=api_key,  # type: ignore[arg-type]
        base_url=api_base,
        callbacks=[_token_callback],  # 修复：注入 token 用量追踪 callback
    )
    if reasoning is not None:
        kwargs["reasoning_effort"] = reasoning
    return ChatOpenAI(**kwargs)  # type: ignore[call-arg]


def get_fast_llm(temperature: float = 0.0) -> Any:
    """Return the fast/lightweight model configured via ``LLM_FAST_MODEL``.

    Falls back to :func:`get_llm` if no dedicated fast model is configured.
    """
    fast_model = os.getenv("LLM_FAST_MODEL")
    if fast_model:
        return get_llm(model=fast_model, temperature=temperature)
    return get_llm(temperature=temperature)


def get_structured_output_llm(temperature: float = 0.0) -> Any:
    """Return LLM with thinking **disabled** — for ``with_structured_output`` calls.

    DashScope / SiliconFlow / ZhipuAI's thinking mode is incompatible with
    ``tool_choice=required`` which LangChain sets implicitly when using
    ``method="function_calling"``.  This helper temporarily forces
    ``LLM_THINKING=none`` so the structured-output call succeeds.
    """
    saved = os.environ.get("LLM_THINKING")
    os.environ["LLM_THINKING"] = "none"
    try:
        return get_fast_llm(temperature=temperature)
    finally:
        if saved is None:
            os.environ.pop("LLM_THINKING", None)
        else:
            os.environ["LLM_THINKING"] = saved
