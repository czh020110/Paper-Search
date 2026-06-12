"""Pydantic v2 models for LangChain ``with_structured_output()``.

These models serve TWO purposes when used with ``method="function_calling"``:

1. **Hard constraints**: ``Literal`` types, ``min_length``, required fields — the
   LLM CANNOT output data that violates these.  Validation fails loudly.
2. **Content guidance**: ``Field(description=...)`` — the LLM reads these
   descriptions and uses them to decide *what* to put in each field.

The prompt should only handle cross-field rules that Pydantic cannot express
(e.g. "all queries must be in English").  Everything else belongs here.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentAnalysisSchema(BaseModel):
    domain: str = Field(
        description=(
            "Specific academic domain inferred from the query. "
            "Be PRECISE and NARROW — combine relevant subfields. "
            "Good: 'Computer Vision / Natural Language Processing (Vision-Language Models, Hallucination Mitigation)'. "
            "Bad: 'Academic Search' or 'Computer Science'. "
            "If the query mentions a specific application area (e.g. medical, legal), include it."
        ),
    )
    query_type: Literal["navigational", "semantic", "metadata"] = Field(
        description=(
            "navigational: user is looking for a SPECIFIC known paper (mentions title, author, exact paper name). "
            "semantic: user wants papers about a TOPIC or METHOD — the most common type. "
            "metadata: user gives structured constraints (year/venue/author) WITHOUT strong topic keywords."
        ),
    )
    boundary_note: str = Field(
        default="",
        description=(
            "MUST be non-empty whenever the query contains temporal expressions "
            "('2022年后', 'recent', 'last 5 years', 'since 2020', etc.) or other "
            "ambiguous boundaries.  Explain how the boundary is interpreted for "
            "retrieval and note that final filtering happens later. "
            "Example: '\"2022年后\" interpreted as >=2022 in retrieval with a "
            "relaxed_window to catch edge cases; final result set will re-verify.' "
            "Only leave empty if the query has genuinely no ambiguous boundary."
        ),
    )


class YearFilterSchema(BaseModel):
    operator: str = Field(
        default=">=",
        description="Year comparison operator: '>=', '>', '=', or '<='.",
    )
    value: int = Field(
        description="Year threshold extracted from the query, e.g. 2022 for '2022年后'.",
    )
    relaxed_window: list[int] = Field(
        min_length=2,
        max_length=2,
        description=(
            "Two-element window [lower, upper] that relaxes the year filter to "
            "avoid missing edge-case papers (e.g. published online in Dec 2021 "
            "but officially in 2022).  For '>=2022', use [2021, 2026]. "
            "For '2020-2024', use [2019, 2025].  Always wider than the literal constraint."
        ),
    )


class HardFiltersSchema(BaseModel):
    year: YearFilterSchema


class RankingSignalsSchema(BaseModel):
    preferred_venues: list[str] = Field(
        min_length=1,
        description=(
            "Flat list of venue names and ALL known aliases found in the query. "
            "For each venue, include: the short name, the full official name, "
            "common abbreviation variants, and workshop/symposium variants. "
            "Example for CVPR: ['CVPR', 'IEEE/CVF Conference on Computer Vision "
            "and Pattern Recognition', 'Computer Vision and Pattern Recognition', "
            "'CVPR Workshop']. "
            "If no venue is mentioned in the query, use a sensible default set "
            "of top-tier venues in the inferred domain, e.g. ['CVPR', 'ICCV', 'ECCV'] "
            "for computer vision."
        ),
    )
    venue_match_mode: Literal["fuzzy_match_and_bonus"] = Field(
        default="fuzzy_match_and_bonus",
        description="Always 'fuzzy_match_and_bonus' — venues boost ranking, never filter.",
    )
    venue_as_hard_filter: bool = Field(
        default=False,
        description="MUST be false.  Venue is a ranking signal, NEVER a hard filter.",
    )


class SemanticQueriesSchema(BaseModel):
    core_concepts: list[str] = Field(
        min_length=3,
        description=(
            "Topic/concept keywords in English.  Each item should be a SINGLE "
            "atomic concept, not a compound phrase.  "
            "Good: 'large language model', 'hallucination', 'object hallucination', "
            "'factuality', 'grounding'.  "
            "Bad: 'large language model hallucination control' (too compound).  "
            "Split compound ideas into separate entries.  Include synonyms and "
            "related sub-concepts the user might not have explicitly mentioned."
        ),
    )
    methodologies: list[str] = Field(
        min_length=2,
        description=(
            "Method/technique keywords in English.  Each item should be a SINGLE "
            "method name or technique, not a compound phrase.  "
            "Example for '强化学习': ['reinforcement learning', 'RL', 'RLHF', "
            "'RLAIF', 'reward modeling', 'preference optimization', 'PPO', 'DPO']. "
            "Include abbreviations, sub-techniques, and closely related methods."
        ),
    )


class S2PayloadSchema(BaseModel):
    query: str = Field(
        description=(
            "Search query for Semantic Scholar. Supports AND/OR/NOT, "
            "exact phrases with quotes, prefix matching. "
            "Prefer short, focused keyword combinations."
        ),
    )
    year: str | None = Field(
        default=None,
        description=(
            "Year or range filter, e.g. '2022-', '2020-2024', '2019'. "
            "Maps to S2 year parameter."
        ),
    )
    venue: str | None = Field(
        default=None,
        description=(
            "Comma-separated venue names, e.g. 'CVPR,NeurIPS'. "
            "Maps to S2 venue parameter."
        ),
    )
    fields_of_study: str | None = Field(
        default=None,
        description=(
            "Comma-separated fields of study, e.g. 'Computer Science,Artificial Intelligence'. "
            "Maps to S2 fieldsOfStudy parameter."
        ),
    )
    publication_types: str | None = Field(
        default=None,
        description=(
            "Comma-separated publication types, e.g. 'JournalArticle,Conference'. "
            "Maps to S2 publicationTypes parameter."
        ),
    )
    min_citation_count: int | None = Field(
        default=None,
        description="Minimum citation count filter. Maps to S2 minCitationCount parameter.",
    )
    open_access_pdf: bool | None = Field(
        default=None,
        description="If true, only return papers with open access PDFs. Maps to S2 openAccessPdf parameter.",
    )
    sort: str | None = Field(
        default=None,
        description=(
            "Sort order, e.g. 'citationCount:desc', 'publicationDate:desc'. "
            "Maps to S2 sort parameter."
        ),
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Maximum number of results (max 1000).",
    )


class OAFilterSchema(BaseModel):
    """OpenAlex filter fields — all optional, combined with AND semantics."""

    publication_year: str | None = Field(
        default=None,
        description=(
            "Year filter, e.g. '>2021', '2022', '2020-2024'. "
            "Maps to filter=publication_year:..."
        ),
    )
    from_publication_date: str | None = Field(
        default=None,
        description=(
            "Date lower bound (YYYY-MM-DD), e.g. '2022-01-01'. "
            "Maps to filter=from_publication_date:..."
        ),
    )
    to_publication_date: str | None = Field(
        default=None,
        description="Date upper bound (YYYY-MM-DD). Maps to filter=to_publication_date:...",
    )
    primary_location_source_id: str | None = Field(
        default=None,
        description=(
            "Venue/source OpenAlex ID, e.g. 'C121332964' for Nature. "
            "Maps to filter=primary_location.source.id:..."
        ),
    )
    authorships_author_id: str | None = Field(
        default=None,
        description=(
            "Author OpenAlex ID, e.g. 'A5023888391'. "
            "Maps to filter=authorships.author.id:..."
        ),
    )
    topics_id: str | None = Field(
        default=None,
        description="Topic OpenAlex ID, e.g. 'T10438'. Maps to filter=topics.id:...",
    )
    type: str | None = Field(
        default=None,
        description=(
            "Work type: 'article', 'book-chapter', 'dissertation', 'preprint', etc. "
            "Maps to filter=type:..."
        ),
    )
    is_oa: bool | None = Field(
        default=None,
        description="Whether the work is open access. Maps to filter=is_oa:true/false.",
    )
    has_abstract: bool | None = Field(
        default=None,
        description="Whether the work has an abstract. Maps to filter=has_abstract:true.",
    )
    language: str | None = Field(
        default=None,
        description="Language code, e.g. 'en'. Maps to filter=language:...",
    )
    cited_by_count: str | None = Field(
        default=None,
        description="Citation count filter, e.g. '>50'. Maps to filter=cited_by_count:...",
    )


class OAPayloadSchema(BaseModel):
    search: str | None = Field(
        default=None,
        description=(
            "Full-text search query (BM25 keyword matching). Words are ANDed by default. "
            "Use quotes for exact phrases: '\"object hallucination\"'. "
            "Prefer short, focused keyword combinations over long sentences. "
            "Leave null if using filter-based search only."
        ),
    )
    filter: OAFilterSchema = Field(
        default_factory=OAFilterSchema,
        description=(
            "Structured filter fields for OpenAlex. All non-null fields are "
            "combined with AND. Use these for precise filtering by venue, "
            "author, year, type, etc."
        ),
    )
    sort: str | None = Field(
        default=None,
        description=(
            "Sort order, e.g. 'relevance_score:desc' (default when search is used), "
            "'cited_by_count:desc', 'publication_date:desc'."
        ),
    )
    per_page: int = Field(
        default=25,
        ge=1,
        le=200,
        description="Number of results per page (max 200).",
    )


class ApiPayloadTranslationSchema(BaseModel):
    semantic_scholar: list[S2PayloadSchema] = Field(
        min_length=1,
        description=(
            "One payload per sub_query in sub_queries_for_retrieval. "
            "Each sub_query MUST have a corresponding entry here."
        ),
    )
    openalex: list[OAPayloadSchema] = Field(
        min_length=1,
        description=(
            "One payload per sub_query in sub_queries_for_retrieval. "
            "Each sub_query MUST have a corresponding entry here."
        ),
    )


class QueryExpansionPolicySchema(BaseModel):
    enabled: bool = Field(default=True)
    seed_paper_driven: bool = Field(default=True)
    extract_terms_from: list[str] = Field(
        default_factory=lambda: ["title", "abstract", "keywords"],
    )
    max_rounds: int = Field(default=2, ge=1, le=5)


class QueryPlanSchema(BaseModel):
    """Pydantic model for LLM structured output.

    Used with ``ChatOpenAI.with_structured_output(QueryPlanSchema, method="function_calling")``.
    The LLM is forced to output JSON matching this schema.  Field descriptions
    guide content quality; type constraints and ``min_length`` enforce structure.
    """

    original_query: str = Field(
        description="The user's original query, preserved VERBATIM (including Chinese characters, punctuation, etc.)."
    )
    intent_analysis: IntentAnalysisSchema
    hard_filters: HardFiltersSchema
    ranking_signals: RankingSignalsSchema
    semantic_queries: SemanticQueriesSchema
    sub_queries_for_retrieval: list[str] = Field(
        min_length=3,
        max_length=5,
        description=(
            "3-5 English search queries formed by combining core_concepts and "
            "methodologies.  Each query should cover a DIFFERENT angle or "
            "combination — don't just rephrase the same idea.  "
            "ALL characters must be ASCII/English — NO Chinese, Japanese, or "
            "Korean characters.  Include venue names in queries where the user "
            "specified a venue."
        ),
    )
    query_expansion_policy: QueryExpansionPolicySchema
    api_payload_translation: ApiPayloadTranslationSchema = Field(
        description=(
            "Translate EVERY sub_query from sub_queries_for_retrieval into "
            "API-specific payloads.  semantic_scholar and openalex arrays MUST "
            "have the same length as sub_queries_for_retrieval — one entry per sub_query."
        ),
    )

    def to_query_plan(self, original_query: str = "") -> "QueryPlan":
        """Convert to the internal :class:`~paper_search.contracts.QueryPlan` dataclass.

        If *original_query* is provided, it overrides the LLM-filled value
        to guarantee the exact user input is preserved.
        """
        from .contracts import IntentAnalysis, QueryPlan

        ia = self.intent_analysis
        return QueryPlan(
            original_query=original_query or self.original_query,
            intent_analysis=IntentAnalysis(
                domain=ia.domain,
                query_type=ia.query_type,
                boundary_note=ia.boundary_note,
            ),
            hard_filters=self.hard_filters.model_dump(),
            ranking_signals=self.ranking_signals.model_dump(),
            semantic_queries=self.semantic_queries.model_dump(),  # type: ignore[arg-type]
            sub_queries_for_retrieval=self.sub_queries_for_retrieval,
            api_payload_translation=self.api_payload_translation.model_dump(),  # type: ignore[arg-type]
            query_expansion_policy=self.query_expansion_policy.model_dump(),
        )


# ── Per-paper judge verdict (used with with_structured_output) ──

class JudgeVerdict(BaseModel):
    """Single-paper relevance verdict from LLM judge."""

    paper_id: str = Field(description="The paper ID being judged (as provided).")
    relevance: Literal["高度相关", "部分相关", "不相关"] = Field(
        description="Relevance level: 高度相关 (highly relevant), 部分相关 (partially relevant), or 不相关 (not relevant)."
    )
    reason: str = Field(description="One sentence explaining the relevance decision.")
    contribution: str = Field(description="One sentence describing the paper's main contribution.")


# ── Keyword list for query evolution (used with with_structured_output) ──

class KeywordList(BaseModel):
    """List of search keywords extracted by LLM for query evolution."""

    keywords: list[str] = Field(
        min_length=1,
        max_length=10,
        description="English search keywords or short phrases (1-4 words each).",
    )
