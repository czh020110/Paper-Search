from __future__ import annotations

import logging
import re
from typing import Any

from .contracts import IntentAnalysis, QueryPlan

logger = logging.getLogger(__name__)

VENUE_KEYWORDS = ["cvpr", "neurips", "iclr", "acl", "emnlp"]


def build_query_plan(query: str, use_llm: bool = False) -> QueryPlan:
    """Convert a natural language query into a structured QueryPlan.

    When *use_llm* is True, delegates to an LLM via LangChain.  Raises
    ``RuntimeError`` if the LLM API key for the current provider is not
    configured so the caller never silently falls back to a lower-quality path.

    When *use_llm* is False, uses the rule-based path (for tests/mock).
    """
    if use_llm:
        from .llm import is_llm_key_configured, _resolve_provider
        if not is_llm_key_configured():
            provider = _resolve_provider()
            raise RuntimeError(
                f"LLM query understanding requested but API key for provider '{provider}' is not set. "
                "Set the corresponding key in .env.local or run with --backend mock for offline testing."
            )
        return _build_query_plan_llm(query)

    return _build_query_plan_rules(query)


def _build_query_plan_llm(query: str) -> QueryPlan:
    """Use LangChain + LLM with ``with_structured_output`` to convert query to QueryPlan.

    Internal function — always called through :func:`build_query_plan` which
    pre-checks the API key.  Raises ``RuntimeError`` when the LLM call or
    structured-output parsing fails so the caller knows exactly what went wrong
    instead of silently using a low-quality path.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from .llm import get_structured_output_llm
    from .prompts import QUERY_UNDERSTANDING_PROMPT
    from .schemas import QueryPlanSchema

    llm = get_structured_output_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(QueryPlanSchema, method="function_calling")
    prompt_text = QUERY_UNDERSTANDING_PROMPT.replace("{query}", query)

    try:
        result: QueryPlanSchema = structured_llm.invoke([
            SystemMessage(content=prompt_text),
            HumanMessage(content="Analyze this query and output the structured result."),
        ])
    except Exception as e:
        raise RuntimeError(
            f"LLM query understanding failed: {e}. "
            "Check your LLM API key and provider configuration. "
            "Use --backend mock for offline testing."
        ) from e

    return result.to_query_plan(original_query=query)


def _build_query_plan_rules(query: str) -> QueryPlan:
    normalized = query.lower()
    years = [int(year) for year in re.findall(r"20\d{2}", query)]

    if _looks_navigational(query, normalized):
        query_type = "navigational"
    elif _looks_metadata(normalized, years):
        query_type = "metadata"
    else:
        query_type = "semantic"

    year_value = years[0] if years else 2010
    hard_filters: dict[str, Any] = {
        "year": {
            "operator": ">=",
            "value": year_value,
            "relaxed_window": [year_value, max(year_value, 2026)],
        }
    }

    ranking_signals = {
        "preferred_venues": ["CVPR", "NeurIPS", "ICLR"],
        "venue_match_mode": "fuzzy_match_and_bonus",
        "venue_as_hard_filter": False,
    }

    semantic_queries = {
        "core_concepts": _extract_keywords(query),
        "methodologies": _extract_methodologies(query),
    }
    sub_queries = _build_sub_queries(query, query_type, semantic_queries)

    api_payload_translation = {
        "semantic_scholar": [
            {"query": sub_query, "year": f"{year_value}-", "limit": 20}
            for sub_query in sub_queries
        ],
        "openalex": [
            {"search": sub_query, "filter": {"publication_year": f">{year_value - 1}"}}
            for sub_query in sub_queries
        ],
    }

    return QueryPlan(
        original_query=query,
        intent_analysis=IntentAnalysis(
            domain="Academic Search",
            query_type=query_type,
            boundary_note=f"检索阶段先按 >={year_value} 处理，最终结果阶段再收口",
        ),
        hard_filters=hard_filters,
        ranking_signals=ranking_signals,
        semantic_queries=semantic_queries,
        sub_queries_for_retrieval=sub_queries,
        api_payload_translation=api_payload_translation,
        query_expansion_policy={
            "enabled": True,
            "seed_paper_driven": True,
            "extract_terms_from": ["title", "abstract", "keywords"],
            "max_rounds": 2,
        },
    )


def _extract_keywords(query: str) -> list[str]:
    """Tokenize and expand keywords for rule-based query understanding.

    Only used when ``use_llm=False`` (mock/testing).  The live pipeline
    always uses the LLM path which produces English keywords
    via LLM translation — no hardcoded mapping is involved in production.
    """
    tokens = [token.strip() for token in re.split(r"[，,、\s]+", query) if token.strip()]
    expanded = list(tokens)

    keyword_map = {
        "大模型": ["large language model", "llm"],
        "幻觉": ["hallucination"],
        "强化学习": ["reinforcement learning", "rl"],
        "多模态": ["multimodal", "vision-language model"],
        "工具": ["tool use"],
        "检索": ["retrieval"],
        "CVPR": ["cvpr"],
    }
    lowered = query.lower()
    for key, aliases in keyword_map.items():
        if key in query or key.lower() in lowered:
            expanded.extend(aliases)

    deduped: list[str] = []
    for token in expanded:
        if token not in deduped:
            deduped.append(token)
    return deduped[:10] if deduped else [query]


def _extract_methodologies(query: str) -> list[str]:
    """Extract methodology hints for rule-based query understanding.

    Only used when ``use_llm=False`` (mock/testing).
    """
    candidates: list[str] = []
    normalized = query.lower()
    if "强化学习" in query or "reinforcement" in normalized or "rl" in normalized:
        candidates.extend(["reinforcement learning", "RL"])
    if "llm" in normalized or "大模型" in query:
        candidates.append("large language model")
    if "tool" in normalized or "工具" in query:
        candidates.append("tool use")
    return candidates or ["literature retrieval"]


def _build_sub_queries(query: str, query_type: str, semantic_queries: dict[str, list[str]]) -> list[str]:
    if query_type == "navigational":
        quoted = re.findall(r'["“”《](.+?)["“”》]', query)
        return [*quoted, query] if quoted else [query]

    if query_type == "metadata":
        venue_terms = [term for term in semantic_queries["core_concepts"] if term.lower() in VENUE_KEYWORDS]
        base = " ".join(semantic_queries["core_concepts"][:5]) or query
        return [base, f"{base} {' '.join(venue_terms)}".strip()]

    core = semantic_queries["core_concepts"]
    methods = semantic_queries["methodologies"]
    variants = [
        " ".join(core[:4]),
        " ".join((core[:3] + methods[:2])[:5]),
        query,
    ]
    return [variant for variant in variants if variant.strip()]


def _looks_navigational(query: str, normalized: str) -> bool:
    has_quote = bool(re.search(r'["“”《].+?["“”》]', query))
    return has_quote or any(token in normalized for token in ["哪篇", "哪一篇", "题目", "title", "exact title", "论文《"])


def _looks_metadata(normalized: str, years: list[int]) -> bool:
    structured_terms = ["作者", "author", "venue", "会议", "期刊", "年份", "发表于"]
    has_structured_signal = any(token in normalized for token in structured_terms)
    thematic_terms = [
        "关于",
        "方法",
        "方向",
        "综述",
        "幻觉",
        "强化学习",
        "多模态",
        "大模型",
        "hallucination",
        "reinforcement",
        "multimodal",
        "retrieval",
        "tool",
    ]
    has_thematic_signal = any(token in normalized for token in thematic_terms)
    return bool(years) and has_structured_signal and not has_thematic_signal
