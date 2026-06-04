from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .config import Settings
from .contracts import ExperimentRecord, GraphOutput, Paper, QueryPlan
from .dedupe import dedupe_papers
from .evaluation import evaluate_query
from .query_understanding import build_query_plan
from .retrieval.mock_backend import retrieve_mock_papers
from .writers import write_outputs


def run_pipeline(query: str, backend: str, output_root: Path, settings: Settings) -> dict[str, object]:
    if backend != "mock":
        raise ValueError(f"Unsupported backend: {backend}")

    timestamp = datetime.now(timezone.utc)
    run_id = f"run_{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"

    query_plan = build_query_plan(query)
    retrieved_papers, raw_edges = retrieve_mock_papers(query_plan, fixtures_dir)
    deduped_papers = _rank_papers(dedupe_papers(retrieved_papers))
    for paper in deduped_papers:
        paper.pool_status = "selected"

    evaluation = evaluate_query(query, deduped_papers, fixtures_dir / "golden_set.json")
    graph_output = GraphOutput(
        query=query,
        nodes=[_paper_node(paper) for paper in deduped_papers],
        edges=_filter_edges(raw_edges, {paper.id for paper in deduped_papers}),
    )
    stage_metrics = _build_stage_metrics(
        query_plan=query_plan,
        retrieved_count=len(retrieved_papers),
        deduped_count=len(deduped_papers),
        selected_count=len(deduped_papers),
        graph_output=graph_output,
        evaluation=evaluation,
    )

    placeholder_record = ExperimentRecord(
        run_id=run_id,
        timestamp=timestamp.isoformat(),
        query=query,
        config_snapshot=settings.config_snapshot(),
        dataset=str(evaluation.get("dataset", "golden_set_v1")),
        stage_metrics=stage_metrics,
        output_files={},
    )
    output_files = write_outputs(run_dir, query_plan, deduped_papers, graph_output, placeholder_record)
    experiment_record = ExperimentRecord(
        run_id=run_id,
        timestamp=timestamp.isoformat(),
        query=query,
        config_snapshot=settings.config_snapshot(),
        dataset=str(evaluation.get("dataset", "golden_set_v1")),
        stage_metrics=stage_metrics,
        output_files=output_files,
    )
    write_outputs(run_dir, query_plan, deduped_papers, graph_output, experiment_record)

    return {
        "run_id": run_id,
        "query_plan": query_plan.to_dict(),
        "papers": [paper.to_dict() for paper in deduped_papers],
        "output_files": output_files,
        "evaluation": evaluation,
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


def _build_stage_metrics(
    query_plan: QueryPlan,
    retrieved_count: int,
    deduped_count: int,
    selected_count: int,
    graph_output: GraphOutput,
    evaluation: dict[str, Any],
) -> dict[str, object]:
    api_calls = len(query_plan.sub_queries_for_retrieval) * 2
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
            "cache_hit_rate": 0.0,
            "raw_result_count": retrieved_count,
            "average_hits_per_query": retrieved_count / max(len(query_plan.sub_queries_for_retrieval), 1),
            "minimum_recall": evaluation.get("recall"),
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
            "cache_hit_rate": 0.0,
        },
    }
