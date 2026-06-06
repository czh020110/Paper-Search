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


def _resolve_provider() -> str:
    """Return the current LLM provider: ``openai`` or ``dashscope``."""
    return os.getenv("LLM_PROVIDER", "openai").lower()


def _resolve_api_key(provider: str) -> str | None:
    if provider == "dashscope":
        return os.getenv("LLM_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    return os.getenv("LLM_API_KEY")


def _resolve_base_url(provider: str) -> str:
    base = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_API_BASE")
    if base:
        return base
    if provider == "dashscope":
        return "https://dashscope.aliyuncs.com/compatible-mode/v1"
    return "https://api.openai.com/v1"


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

    Uses ``LLM_PROVIDER`` to switch between backends.  Both ``openai`` and
    ``dashscope`` use the same ``ChatOpenAI`` client — only the *base_url*
    and *api_key* differ (DashScope exposes an OpenAI-compatible endpoint).

    When ``LLM_PROVIDER=dashscope``, ``LLM_API_KEY`` (or ``DASHSCOPE_API_KEY``)
    is used and the base URL defaults to the DashScope compatible endpoint.

    ``LLM_THINKING`` controls reasoning/thinking depth:
      - ``off`` / ``minimal`` / ``low`` / ``medium`` / ``high`` for OpenAI
      - ``off`` → enable_thinking=False, ``on`` → enable_thinking=True for DashScope
    """
    from langchain_openai import ChatOpenAI

    provider = _resolve_provider()
    api_key = _resolve_api_key(provider)
    api_base = _resolve_base_url(provider)

    if model is None:
        model = _resolve_model(fast=False)

    thinking: ThinkingLevel = _get_thinking_level()

    if provider == "dashscope":
        enable_thinking = _DASHSCOPE_THINKING_MAP.get(thinking)
        kwargs_ds: dict[str, Any] = dict(
            model=model,
            temperature=temperature,
            api_key=api_key,  # type: ignore[arg-type]
            base_url=api_base,
        )
        if enable_thinking is not None:
            kwargs_ds["extra_body"] = {"enable_thinking": enable_thinking}
        return ChatOpenAI(**kwargs_ds)  # type: ignore[call-arg]

    # OpenAI provider
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
