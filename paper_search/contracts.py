from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class IntentAnalysis:
    domain: str
    query_type: str
    boundary_note: str | None = None


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    intent_analysis: IntentAnalysis
    hard_filters: dict[str, Any]
    ranking_signals: dict[str, Any]
    semantic_queries: dict[str, list[str]]
    sub_queries_for_retrieval: list[str]
    api_payload_translation: dict[str, list[dict[str, Any]]]
    query_expansion_policy: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Paper:
    id: str
    source_ids: dict[str, str | None]
    title: str
    abstract: str | None
    authors: list[dict[str, str | None]] = field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    publication_date: str | None = None
    url: str | None = None
    open_access_pdf: dict[str, Any] | None = None
    fields: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    citation_count: int | None = None
    reference_count: int | None = None
    source_api: str = "mock"
    retrieved_at: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    pool_status: str = "seed"
    reranker_score: float | None = None
    llm_relevance: str | None = None
    reason: str | None = None
    contribution: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CitationEdge:
    source_paper_id: str
    target_paper_id: str
    edge_type: str
    discovered_round: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GraphOutput:
    query: str
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentRecord:
    run_id: str
    timestamp: str
    query: str
    config_snapshot: dict[str, Any]
    dataset: str
    stage_metrics: dict[str, Any]
    output_files: dict[str, str]
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
