from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

from .contracts import IntentAnalysis, QueryPlan

if TYPE_CHECKING:
    from .schemas import QueryPlanSchema

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
        return build_query_plan_llm(query)

    return _build_query_plan_rules(query)


def build_query_plan_llm(query: str) -> QueryPlan:
    """Use LangChain + LLM with ``with_structured_output`` to convert query to QueryPlan.

    The LLM is forced to output JSON conforming to the Pydantic schema —
    no free-form JSON parsing needed.  Falls back to a rule-based plan
    when the LLM or structured-output call fails.
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
        return result.to_query_plan(original_query=query)
    except Exception:
        logger.warning("Structured output failed, using rule-based fallback", exc_info=True)
        return _build_fallback_query_plan(query).to_query_plan(original_query=query)



def _build_fallback_query_plan(query: str) -> QueryPlanSchema:
    """Build a minimal QueryPlanSchema when LLM output is unparseable.

    Uses heuristic extraction for English paper-name patterns
    (e.g. ``"BART by Lewis et al."``, ``"the MS^2 DeYong2021 paper"``)
    to produce higher-quality search terms than the raw query.
    """
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

    # ── Heuristic extraction ──────────────────────────────────────────
    title_part, author_part, year_hint = _extract_paper_parts(query)

    # Build focused sub-queries from extracted parts
    if title_part:
        sub_queries: list[str] = [title_part]
        if author_part:
            sub_queries.append(f"{title_part} {author_part}")
        # Add the original query as a last resort variant
        if query not in sub_queries:
            sub_queries.append(query)
    else:
        sub_queries = [query, f"{query} paper"]

    # Normalize special symbols for S2/OA search APIs
    # MS^2 → "MS2 multi-document summarization", etc.
    sub_queries = [_normalize_search_term(sq) for sq in sub_queries]

    # Ensure at least 3 sub_queries (schema min_length=3)
    while len(sub_queries) < 3:
        sub_queries.append(f"{sub_queries[-1]} research")

    core_concepts = [title_part or query]
    if author_part:
        core_concepts.append(author_part)
    core_concepts.append("academic research")

    # ── Year filter ────────────────────────────────────────────────────
    # When no year is hinted in the query, use a very broad range (2010+)
    # so we don't accidentally filter out classic papers (BART 2019, etc.).
    year_value = year_hint or 2010
    s2_year = f"{year_value}-"
    oa_year_filter = f"publication_year:>{year_value - 1}"

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
                value=year_value,
                relaxed_window=[year_value, 2026],
            ),
        ),
        ranking_signals=RankingSignalsSchema(
            preferred_venues=["CVPR", "NeurIPS", "ICLR"],
            venue_match_mode="fuzzy_match_and_bonus",
            venue_as_hard_filter=False,
        ),
        semantic_queries=SemanticQueriesSchema(
            core_concepts=(core_concepts + ["literature"])[:6],
            methodologies=["academic search", "literature retrieval"],
        ),
        sub_queries_for_retrieval=sub_queries,
        query_expansion_policy=QueryExpansionPolicySchema(
            enabled=True,
            seed_paper_driven=True,
            extract_terms_from=["title", "abstract", "keywords"],
            max_rounds=2,
        ),
        api_payload_translation=ApiPayloadTranslationSchema(
            semantic_scholar=[
                S2PayloadSchema(query=sq, year=s2_year) for sq in sub_queries
            ],
            openalex=[
                OAPayloadSchema(search=sq, filter=oa_year_filter) for sq in sub_queries
            ],
        ),
    )


def _extract_paper_parts(query: str) -> tuple[str, str, int | None]:
    """Extract paper title, author surname, and year from English queries.

    Handles patterns like:
      - "BART by Lewis et al."
      - "the MS^2 DeYong2021 paper"
      - "the AlphaGeometry paper"
      - "the paper about the Objaverse dataset"
      - "the cnn paper"
      - "SPIKE syntactic search paper"

    Returns (title_part, author_part, year_hint).
    """
    q = query.strip()
    author_part = ""
    year_hint: int | None = None

    # Extract year from patterns like "DeYong2021" or standalone "2021"
    year_match = re.search(r"(?:[A-Za-z])(\d{4})|(\d{4})", q)
    if year_match:
        y = int(year_match.group(1) or year_match.group(2))
        if 1990 <= y <= 2026:
            year_hint = y

    # Pattern 1: "X by Y et al." / "X by Y"
    by_match = re.match(r"(.+?)\s+by\s+([A-Z][A-Za-z\-]+)", q)
    if by_match:
        title_part = by_match.group(1).strip()
        # Strip leading articles ("a", "the") from title
        title_part = re.sub(r"^(a|the|an)\s+", "", title_part, flags=re.IGNORECASE)
        author_part = by_match.group(2).strip()
        return title_part, author_part, year_hint

    # Pattern 2: "the X AuthorYear paper" / "the X paper" / "the paper about X"
    the_paper_match = re.match(r"the\s+paper\s+about\s+(.+?)(?:\s+paper)?$", q, re.IGNORECASE)
    if the_paper_match:
        return the_paper_match.group(1).strip(), author_part, year_hint

    the_match = re.match(r"the\s+(.+?)\s+paper", q, re.IGNORECASE)
    if the_match:
        inner = the_match.group(1).strip()
        # Split off trailing AuthorYear token (e.g. "DeYong2021")
        ay_match = re.match(r"(.+?)\s+([A-Z][A-Za-z\-]+\d{4})$", inner)
        if ay_match:
            title_part = ay_match.group(1).strip()
            author_part = re.sub(r"\d{4}$", "", ay_match.group(2)).strip()
            return title_part, author_part, year_hint
        # Split off trailing AuthorYear token without year digits (e.g. "fabri2019multinews")
        ay2_match = re.match(r"(.+?)\s+([A-Z][A-Za-z]+\d{4}[A-Za-z]*)$", inner)
        if ay2_match:
            title_part = ay2_match.group(1).strip()
            author_part = re.sub(r"\d{4}.*$", "", ay2_match.group(2)).strip()
            return title_part, author_part, year_hint
        return inner, author_part, year_hint

    # Pattern 3: "X Y paper" (e.g. "SPIKE syntactic search paper")
    paper_match = re.match(r"(.+?)\s+paper(?:s)?$", q, re.IGNORECASE)
    if paper_match:
        inner = paper_match.group(1).strip()
        # Strip leading articles
        inner = re.sub(r"^(a|the|an)\s+", "", inner, flags=re.IGNORECASE)
        return inner, author_part, year_hint

    # Fallback: use the whole query as the title part
    return q, author_part, year_hint


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


def _normalize_search_term(term: str) -> str:
    """Normalize special symbols in search terms for better API results.

    S2/OA APIs struggle with symbols like ^, superscripts, etc.
    E.g. "MS^2" is interpreted as "Multiple Sclerosis" instead of
    the multi-document summarization paper.
    """
    # Only expand the symbol form; avoid double-expanding "MS2" that
    # was already produced from "MS^2" in a prior step.
    if "MS^2" in term:
        return term.replace("MS^2", "MS2 multi-document summarization")
    return term


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
