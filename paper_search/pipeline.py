from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .budget import BudgetController
from .cache import CacheStore
from .config import Settings
from .contracts import ExperimentRecord, GraphOutput, Paper, QueryPlan
from .dedupe import dedupe_papers
from .evaluation import evaluate_query
from .logging_config import setup_logging
from .pool import CandidatePool
from .query_understanding import build_query_plan
from .retrieval.live_backend import retrieve_live_papers
from .retrieval.mock_backend import retrieve_mock_papers
from .screening.coarse import coarse_score
from .screening.judge import judge
from .screening.reranker import rerank
from .snowball import run_snowball
from .writers import write_outputs

logger = logging.getLogger(__name__)


def run_pipeline(query: str, backend: str, output_root: Path, settings: Settings) -> dict[str, object]:
    if backend not in ("mock", "live"):
        raise ValueError(f"Unsupported backend: {backend}. Choose 'mock' or 'live'.")

    timestamp = datetime.now(timezone.utc)
    run_id = f"run_{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"

    setup_logging(run_id, run_dir / "logs", settings.log_level)
    logger.info("Pipeline started", extra={"run_id": run_id, "stage": "pipeline_start"})

    budget = BudgetController(settings.budget_limits())
    # Use LLM query understanding for live backend when API key is available
    query_plan = build_query_plan(query, use_llm=(backend == "live"))

    if backend == "mock":
        retrieved_papers, raw_edges = retrieve_mock_papers(query_plan, fixtures_dir)
        cache_hit_rate = 0.0
    else:
        cache = CacheStore(settings.cache_dir)
        retrieved_papers, raw_edges = retrieve_live_papers(query_plan, cache=cache, budget=budget)
        cache_hit_rate = cache.hit_rate()

    deduped_papers = _rank_papers(dedupe_papers(retrieved_papers))

    pool = CandidatePool()
    for paper in deduped_papers:
        pool.add(paper, status="seed")

    # Screening chain: snowball → coarse → rerank → judge → selected
    # Snowball expansion (S-005): query evolution + selective citation expansion.
    # Skip in mock mode — the snowball node functions are stubs and running
    # LangGraph's StateGraph.invoke() with mock data is pure overhead that
    # was causing a GC death spiral during test discovery.
    if backend == "live":
        pool, snowball_edges = run_snowball(pool, query_plan, budget, cache=cache)
    else:
        snowball_edges: list[dict[str, object]] = []

    # Coarse screening (S-006): BM25 + Embedding + Structure triple scoring
    coarse_score(pool, query_plan)

    # Medium screening (S-007): Reranker high-precision re-ranking
    rerank(pool, query_plan)

    # Fine screening (S-008): LLM batch relevance judgment
    judge(pool, query_plan)

    # Merge snowball edges with any existing raw edges
    all_edges = _merge_edges(raw_edges, snowball_edges)
    selected_papers = pool.by_status("selected")
    evaluation = evaluate_query(query, selected_papers, fixtures_dir / "golden_set.json")
    graph_output = GraphOutput(
        query=query,
        nodes=[_paper_node(paper) for paper in selected_papers],
        edges=_filter_edges(all_edges, {paper.id for paper in selected_papers}),
    )
    stage_metrics = _build_stage_metrics(
        query_plan=query_plan,
        retrieved_count=len(retrieved_papers),
        deduped_count=len(deduped_papers),
        selected_count=len(selected_papers),
        graph_output=graph_output,
        evaluation=evaluation,
        backend=backend,
        cache_hit_rate=cache_hit_rate,
        pool_summary=pool.summary(),
        budget_summary=budget.usage_summary(),
    )

    output_files = _compute_output_file_paths(run_dir)
    experiment_record = ExperimentRecord(
        run_id=run_id,
        timestamp=timestamp.isoformat(),
        query=query,
        config_snapshot=settings.config_snapshot(),
        dataset=str(evaluation.get("dataset", "golden_set_v1")),
        stage_metrics=stage_metrics,
        output_files=output_files,
    )
    write_outputs(run_dir, query_plan, selected_papers, graph_output, experiment_record)

    return {
        "run_id": run_id,
        "query_plan": query_plan.to_dict(),
        "papers": [paper.to_dict() for paper in selected_papers],
        "output_files": output_files,
        "evaluation": evaluation,
    }


def _compute_output_file_paths(run_dir: Path) -> dict[str, str]:
    return {
        "markdown": str(run_dir / "result.md"),
        "graph": str(run_dir / "graph.json"),
        "experiment": str(run_dir / "experiment.json"),
        "log": str(run_dir / "logs"),
    }


def _paper_node(paper: Paper) -> dict[str, object]:
    return {
        "id": paper.id,
        "title": paper.title,
        "year": paper.year,
        "venue": paper.venue,
        "relevance": paper.llm_relevance or "highly_relevant",
        "source_api": paper.source_api,
        "pool_status": paper.pool_status,
    }


def _filter_edges(raw_edges: list[dict[str, object]], node_ids: set[str]) -> list[dict[str, object]]:
    filtered: list[dict[str, object]] = []
    for edge in raw_edges:
        source_id = edge.get("source_paper_id")
        target_id = edge.get("target_paper_id")
        edge_type = edge.get("edge_type")
        discovered_round = edge.get("discovered_round")
        if isinstance(source_id, str) and isinstance(target_id, str) and source_id in node_ids and target_id in node_ids:
            filtered.append({
                "source": source_id,
                "target": target_id,
                "type": edge_type if isinstance(edge_type, str) else "citation",
                "discovered_round": discovered_round if isinstance(discovered_round, int) else 0,
            })
    return filtered


def _rank_papers(papers: list[Paper]) -> list[Paper]:
    return sorted(
        papers,
        key=lambda paper: (paper.year or 0, paper.citation_count or 0, paper.title.lower()),
        reverse=True,
    )


def _merge_edges(raw_edges: list[dict[str, object]], snowball_edges: list[dict[str, object]]) -> list[dict[str, object]]:
    """Merge edges from initial retrieval and snowball phases, deduplicating by (source, target, type)."""
    seen: set[tuple[str, str, str]] = set()
    merged: list[dict[str, object]] = []
    for edge in raw_edges + snowball_edges:
        key = (
            str(edge.get("source_paper_id", edge.get("source", ""))),
            str(edge.get("target_paper_id", edge.get("target", ""))),
            str(edge.get("edge_type", edge.get("type", "citation"))),
        )
        if key not in seen:
            seen.add(key)
            merged.append(edge)
    return merged


def _build_stage_metrics(
    query_plan: QueryPlan,
    retrieved_count: int,
    deduped_count: int,
    selected_count: int,
    graph_output: GraphOutput,
    evaluation: dict[str, Any],
    backend: str,
    cache_hit_rate: float = 0.0,
    pool_summary: dict[str, int] | None = None,
    budget_summary: dict[str, Any] | None = None,
) -> dict[str, object]:
    api_calls = budget_summary.get("api_calls_used", 0) if budget_summary else 0
    return {
        "query_understanding": {
            "latency_ms": 0,
            "sub_queries_count": len(query_plan.sub_queries_for_retrieval),
            "api_calls": 0,
        },
        "initial_retrieval": {
            "latency_ms": 0,
            "seed_pool_size": deduped_count,
            "api_calls": api_calls,
            "cache_hit_rate": cache_hit_rate,
            "raw_result_count": retrieved_count,
            "average_hits_per_query": retrieved_count / max(len(query_plan.sub_queries_for_retrieval), 1),
            "minimum_recall": evaluation.get("recall"),
            "backend": backend,
        },
        "result_format": {
            "latency_ms": 0,
            "output_count": selected_count,
            "graph_node_count": len(graph_output.nodes),
            "graph_edge_count": len(graph_output.edges),
        },
        "overall": {
            "total_latency_ms": 0,
            "total_api_calls": api_calls,
            "total_token_usage": {"prompt": 0, "completion": 0, "total": 0},
            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1": evaluation.get("f1"),
            "cache_hit_rate": cache_hit_rate,
            "pool_status_summary": pool_summary or {},
            "budget": budget_summary or {},
        },
    }