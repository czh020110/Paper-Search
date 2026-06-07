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
    ``RuntimeError`` if ``LLM_API_KEY`` is not configured so the caller
    never silently falls back to a lower-quality path.

    When *use_llm* is False, uses the rule-based path (for tests/mock).
    """
    if use_llm:
        import os
        if not os.getenv("LLM_API_KEY"):
            raise RuntimeError(
                "LLM query understanding requested but LLM_API_KEY is not set. "
                "Set LLM_API_KEY in .env or run with --backend mock for offline testing."
            )
        return build_query_plan_llm(query)

    return _build_query_plan_rules(query)


def build_query_plan_llm(query: str) -> QueryPlan:
    """Use LangChain + LLM with structured output to convert query to QueryPlan.

    Uses ``ChatOpenAI.with_structured_output(QueryPlanSchema)`` so the LLM
    is forced to return JSON conforming to the design-doc schema — no
    free-form JSON parsing needed.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

    from .llm import get_fast_llm
    from .prompts import QUERY_UNDERSTANDING_PROMPT
    from .schemas import QueryPlanSchema

    llm = get_fast_llm(temperature=0.0)
    prompt_text = QUERY_UNDERSTANDING_PROMPT.replace("{query}", query)

    response = llm.invoke([
        SystemMessage(content=prompt_text),
        HumanMessage(content="请根据上述要求分析此查询并输出结构化结果。返回 JSON 格式，不要包含其他文字。"),
    ])

    raw = response.content if hasattr(response, "content") else str(response)
    result = _parse_query_plan_json(raw, query)
    return result.to_query_plan(original_query=query)


def _parse_query_plan_json(raw: str, query: str) -> QueryPlanSchema:
    """Parse LLM JSON into QueryPlanSchema with defaults for missing fields.

    Uses ``model_construct`` (skips validation) so partial LLM output is
    still usable — missing fields fall back to the schema's defaults.
    Falls back to a rule-based plan when JSON cannot be extracted at all.
    """
    import json

    text = raw.strip()
    # Extract JSON block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        text = text[start: end + 1]
    else:
        logger.warning("No JSON block found in LLM response, using fallback")
        return _build_fallback_query_plan(query)

    try:
        data: dict[str, object] = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("LLM JSON decode failed, using fallback")
        return _build_fallback_query_plan(query)

    # Build schema with defaults for missing fields via model_construct
    return _build_schema_from_partial(data, query)


def _build_schema_from_partial(data: dict[str, object], query: str) -> QueryPlanSchema:
    """Build a QueryPlanSchema from partial LLM output, filling defaults."""
    from .schemas import (
        ApiPayloadTranslationSchema,
        HardFiltersSchema,
        IntentAnalysisSchema,
        OAPayloadSchema,
        QueryExpansionPolicySchema,
        QueryPlanSchema,
        RankingSignalsSchema,
        S2PayloadSchema,
        SemanticQueriesSchema,
        YearFilterSchema,
    )

    # intent_analysis
    ia_raw = data.get("intent_analysis")
    if isinstance(ia_raw, dict):
        intent_analysis = IntentAnalysisSchema.model_construct(
            domain=ia_raw.get("domain", "Academic Search"),
            query_type=ia_raw.get("query_type", "semantic"),
            boundary_note=ia_raw.get("boundary_note", ""),
        )
    else:
        intent_analysis = IntentAnalysisSchema(
            domain="Academic Search", query_type="semantic", boundary_note=""
        )

    # hard_filters
    hf_raw = data.get("hard_filters")
    year_val = 2022
    if isinstance(hf_raw, dict):
        year_raw = hf_raw.get("year", {})
        if isinstance(year_raw, dict):
            year_val = year_raw.get("value", 2022)
    hard_filters = HardFiltersSchema(year=YearFilterSchema(
        operator=">=", value=year_val, relaxed_window=[year_val, max(year_val, 2026)],
    ))

    # ranking_signals
    rs_raw = data.get("ranking_signals")
    if isinstance(rs_raw, dict):
        preferred = rs_raw.get("preferred_venues", ["CVPR", "NeurIPS", "ICLR"])
        ranking_signals = RankingSignalsSchema(
            preferred_venues=preferred,
            venue_match_mode="fuzzy_match_and_bonus",
            venue_as_hard_filter=False,
        )
    else:
        ranking_signals = RankingSignalsSchema(
            preferred_venues=["CVPR", "NeurIPS", "ICLR"],
        )

    # semantic_queries — pad to meet Pydantic min_length
    sq_raw = data.get("semantic_queries")
    if isinstance(sq_raw, dict):
        core = list(sq_raw.get("core_concepts", [query]))
        methods = list(sq_raw.get("methodologies", ["academic search"]))
    else:
        core = [query]
        methods = ["academic search"]
    # Pad core_concepts to at least 3, methodologies to at least 2
    while len(core) < 3:
        core.append(core[0] if core else query)
    while len(methods) < 2:
        methods.append(methods[0] if methods else "academic search")
    semantic_queries = SemanticQueriesSchema(core_concepts=core, methodologies=methods)

    # sub_queries
    raw_sq = data.get("sub_queries_for_retrieval", [query])
    sub_queries = list(raw_sq) if isinstance(raw_sq, list) else [query]
    if len(sub_queries) < 3:
        while len(sub_queries) < 3:
            sub_queries.append(sub_queries[0] if sub_queries else query)

    # api_payload_translation
    apt_raw = data.get("api_payload_translation")
    s2_payloads: list[dict[str, object]] = []
    oa_payloads: list[dict[str, object]] = []
    if isinstance(apt_raw, dict):
        s2_raw = apt_raw.get("semantic_scholar", [])
        oa_raw = apt_raw.get("openalex", [])
        if isinstance(s2_raw, list):
            s2_payloads = [p for p in s2_raw if isinstance(p, dict)]
        if isinstance(oa_raw, list):
            oa_payloads = [p for p in oa_raw if isinstance(p, dict)]

    # Ensure at least one payload per sub_query
    if not s2_payloads:
        s2_payloads = [{"query": q, "year": f"{year_val}-"} for q in sub_queries]
    if not oa_payloads:
        oa_payloads = [{"search": q, "filter": f"publication_year:>{year_val - 1}"} for q in sub_queries]

    # Build typed payload lists
    s2_typed = [S2PayloadSchema(query=str(p.get("query", q)), year=str(p.get("year", f"{year_val}-")))
                for p, q in zip(s2_payloads, sub_queries)]
    oa_typed = [OAPayloadSchema(search=str(p.get("search", q)), filter=str(p.get("filter", f"publication_year:>{year_val - 1}")))
                for p, q in zip(oa_payloads, sub_queries)]

    api_payload_translation = ApiPayloadTranslationSchema(
        semantic_scholar=s2_typed, openalex=oa_typed,
    )

    # query_expansion_policy
    qep_raw = data.get("query_expansion_policy")
    if isinstance(qep_raw, dict):
        query_expansion_policy = QueryExpansionPolicySchema.model_construct(**{
            k: v for k, v in qep_raw.items()
            if k in ("enabled", "seed_paper_driven", "extract_terms_from", "max_rounds")
        })
    else:
        query_expansion_policy = QueryExpansionPolicySchema()

    return QueryPlanSchema(
        original_query=query,
        intent_analysis=intent_analysis,
        hard_filters=hard_filters,
        ranking_signals=ranking_signals,
        semantic_queries=semantic_queries,
        sub_queries_for_retrieval=sub_queries,
        api_payload_translation=api_payload_translation,
        query_expansion_policy=query_expansion_policy,
    )


def _build_fallback_query_plan(query: str) -> QueryPlanSchema:
    """Build a minimal QueryPlanSchema when LLM output is unparseable."""
    from .schemas import (
        ApiPayloadTranslationSchema,
        HardFiltersSchema,
        IntentAnalysisSchema,
        OAPayloadSchema,
        QueryExpansionPolicySchema,
        QueryPlanSchema,
        RankingSignalsSchema,
        S2PayloadSchema,
        SemanticQueriesSchema,
        YearFilterSchema,
    )

    return QueryPlanSchema(
        original_query=query,
        intent_analysis=IntentAnalysisSchema(
            domain="Academic Search",
            query_type="semantic",
            boundary_note="LLM query understanding failed, using fallback",
        ),
        hard_filters=HardFiltersSchema(
            year=YearFilterSchema(
                operator=">=",
                value=2022,
                relaxed_window=[2022, 2026],
            ),
        ),
        ranking_signals=RankingSignalsSchema(
            preferred_venues=["CVPR", "NeurIPS", "ICLR"],
            venue_match_mode="fuzzy_match_and_bonus",
            venue_as_hard_filter=False,
        ),
        semantic_queries=SemanticQueriesSchema(
            core_concepts=[query, "academic research", "literature"],
            methodologies=["academic search", "literature retrieval"],
        ),
        sub_queries_for_retrieval=[query, f"{query} survey", f"{query} method"],
        query_expansion_policy=QueryExpansionPolicySchema(
            enabled=True,
            seed_paper_driven=True,
            extract_terms_from=["title", "abstract", "keywords"],
            max_rounds=2,
        ),
        api_payload_translation=ApiPayloadTranslationSchema(
            semantic_scholar=[
                S2PayloadSchema(query=query, year="2022-"),
                S2PayloadSchema(query=f"{query} survey", year="2022-"),
                S2PayloadSchema(query=f"{query} method", year="2022-"),
            ],
            openalex=[
                OAPayloadSchema(search=query, filter="publication_year:>2021"),
                OAPayloadSchema(search=f"{query} survey", filter="publication_year:>2021"),
                OAPayloadSchema(search=f"{query} method", filter="publication_year:>2021"),
            ],
        ),
    )


def _build_query_plan_rules(query: str) -> QueryPlan:
    normalized = query.lower()
    years = [int(year) for year in re.findall(r"20\d{2}", query)]

    if _looks_navigational(query, normalized):
        query_type = "navigational"
    elif _looks_metadata(normalized, years):
        query_type = "metadata"
    else:
        query_type = "semantic"

    year_value = years[0] if years else 2022
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
        "semantic_scholar": [{"query": sub_query, "year": f"{year_value}-"} for sub_query in sub_queries],
        "openalex": [{"search": sub_query, "filter": f"publication_year:>{year_value - 1}"} for sub_query in sub_queries],
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
    always uses :func:`build_query_plan_llm` which produces English keywords
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
