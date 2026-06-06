from __future__ import annotations

import os
from typing import Any


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Return a LangChain ChatOpenAI instance configured from environment.

    Uses ``LLM_API_BASE``, ``LLM_API_KEY`` from environment.  *model*
    defaults to ``LLM_MODEL``, falling back to ``LLM_FAST_MODEL``.

    The API is expected to be OpenAI-compatible (e.g. DeepSeek, Qwen via DashScope, etc.).

    The langchain_openai import is lazy so that CLI / web-server startup is
    fast — the import cost is only paid the first time get_llm() is called.
    """
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    if model is None:
        model = os.getenv("LLM_MODEL") or os.getenv("LLM_FAST_MODEL") or "gpt-4o-mini"

    return ChatOpenAI(  # type: ignore[call-arg]
        model=model,
        temperature=temperature,
        api_key=api_key,  # type: ignore[arg-type]
        base_url=api_base,
    )


def get_fast_llm(temperature: float = 0.0) -> Any:
    """Return the fast/lightweight model configured via ``LLM_FAST_MODEL``."""
    fast_model = os.getenv("LLM_FAST_MODEL")
    if fast_model:
        return get_llm(model=fast_model, temperature=temperature)
    return get_llm(temperature=temperature)
