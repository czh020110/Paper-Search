from __future__ import annotations

import os
from typing import Any, Literal

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
        return os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if provider == "siliconflow":
        return os.getenv("LLM_API_KEY") or os.getenv("SILICONFLOW_API_KEY")
    if provider == "zhipuai":
        return os.getenv("LLM_API_KEY") or os.getenv("ZHIPUAI_API_KEY")
    return os.getenv("LLM_API_KEY")


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
