from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
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
from .retrieval.enrichment import enrich_papers
from .retrieval.live_backend import retrieve_live_papers
from .retrieval.mock_backend import retrieve_mock_papers
from .screening.coarse import coarse_score
from .screening.judge import judge
from .screening.reranker import rerank
from .snowball import run_snowball
from .writers import write_outputs

logger = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    backend: str,
    output_root: Path,
    settings: Settings,
    on_stage: Callable[[str, str], None] | None = None,
) -> dict[str, Any]:
    if backend not in ("mock", "live"):
        raise ValueError(f"Unsupported backend: {backend}. Choose 'mock' or 'live'.")

    timestamp = datetime.now(timezone.utc)
    run_id = f"run_{timestamp.strftime('%Y%m%d%H%M%S')}_{uuid4().hex[:8]}"
    run_dir = output_root / run_id
    fixtures_dir = Path(__file__).resolve().parent.parent / "fixtures"

    setup_logging(run_id, run_dir / "logs", settings.log_level)
    logger.info("Pipeline started", extra={"run_id": run_id, "stage": "pipeline_start"})

    budget = BudgetController(settings.budget_limits())

    timings: dict[str, float] = {}

    t0 = time.time()
    query_plan = build_query_plan(query, use_llm=(backend == "live"))
    timings["query_understanding"] = (time.time() - t0) * 1000
    _notify(on_stage, "query_understanding", "done")

    if backend == "mock":
        t0 = time.time()
        retrieved_papers, raw_edges = retrieve_mock_papers(query_plan, fixtures_dir)
        timings["initial_retrieval"] = (time.time() - t0) * 1000
        cache_hit_rate = 0.0
    else:
        t0 = time.time()
        cache = CacheStore(settings.cache_dir)
        retrieved_papers, raw_edges = retrieve_live_papers(query_plan, cache=cache, budget=budget)
        timings["initial_retrieval"] = (time.time() - t0) * 1000
        cache_hit_rate = cache.hit_rate()

    _notify(on_stage, "initial_retrieval", "done")

    deduped_papers = _rank_papers(dedupe_papers(retrieved_papers))

    # Enrich papers with missing venue/abstract via cross-source title search
    if backend == "live":
        t0 = time.time()
        deduped_papers = enrich_papers(deduped_papers, cache=cache, budget=budget)
        timings["enrichment"] = (time.time() - t0) * 1000
    _notify(on_stage, "enrichment", "done")

    pool = CandidatePool()
    for paper in deduped_papers:
        pool.add(paper, status="seed")

    # Screening chain: coarse → rerank → judge → snowball → selected
    # Snowball runs AFTER judge so it has relevance-scored papers to
    # extract keywords from and expand citations from.
    # Skip in mock mode — mock data doesn't have real paper IDs for
    # API-based citation expansion and running the graph is pure overhead.

    # Coarse screening (S-006): BM25 + Embedding + Structure triple scoring
    t0 = time.time()
    coarse_score(pool, query_plan)
    timings["coarse"] = (time.time() - t0) * 1000
    _notify(on_stage, "coarse", "done")

    # Medium screening (S-007): Reranker high-precision re-ranking
    t0 = time.time()
    rerank(pool, query_plan)
    timings["rerank"] = (time.time() - t0) * 1000
    _notify(on_stage, "rerank", "done")

    # Fine screening (S-008): LLM concurrent per-paper relevance judgment
    t0 = time.time()
    judge(pool, query_plan)
    timings["judge"] = (time.time() - t0) * 1000
    _notify(on_stage, "judge", "done")

    # Snowball expansion (S-005): query evolution + selective citation expansion.
    # Runs after judge so it has llm_judged papers to extract keywords from
    # and expand citations from.
    if backend == "live":
        t0 = time.time()
        pool, snowball_edges = run_snowball(pool, query_plan, budget, cache=cache)
        timings["snowball"] = (time.time() - t0) * 1000
    else:
        snowball_edges: list[dict[str, object]] = []
    _notify(on_stage, "snowball", "done")

    # Merge snowball edges with any existing raw edges
    all_edges = _merge_edges(raw_edges, snowball_edges)
    selected_papers = _rank_by_relevance(pool.by_status("selected"))
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
        timings=timings,
        # 修复：汇总 token 用量到 stage_metrics（此前硬编码为全 0）
        token_usage=None,  # 使用默认值，在 _build_stage_metrics 内部从全局 callback 读取
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
        "query_plan": str(run_dir / "query_plan.json"),
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
        "sources": paper.sources,
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
    """排序候选论文：使用加权综合分 0.3×年份归一化 + 0.7×引用归一化。

    修复：原排序使用 (year, citation_count) 字典序，年份权重过大。
    改为加权综合分排序，年份和引用数分别归一化到 [0,1] 后按权重融合。
    """
    if not papers:
        return papers

    # 提取年份和引用数范围
    years = [p.year for p in papers if p.year is not None]
    citations = [p.citation_count for p in papers if p.citation_count is not None]

    year_min = min(years) if years else 0
    year_max = max(years) if years else 0
    year_range = year_max - year_min if year_max > year_min else 1

    cit_min = min(citations) if citations else 0
    cit_max = max(citations) if citations else 0
    cit_range = cit_max - cit_min if cit_max > cit_min else 1

    def _score(paper: Paper) -> float:
        # 年份归一化：越新分数越高
        year_norm = (paper.year - year_min) / year_range if paper.year is not None else 0.0
        # 引用归一化：引用越多分数越高
        cit_norm = (paper.citation_count - cit_min) / cit_range if paper.citation_count is not None else 0.0
        return 0.3 * year_norm + 0.7 * cit_norm

    return sorted(papers, key=_score, reverse=True)


def _rank_by_relevance(papers: list[Paper]) -> list[Paper]:
    """Sort selected papers: 高度相关 first, then by reranker_score desc."""
    def _sort_key(p: Paper) -> tuple[int, float]:
        rel_order = 0 if (p.llm_relevance or "").startswith("高度") else 1
        score = -(p.reranker_score or 0.0)
        return (rel_order, score)
    return sorted(papers, key=_sort_key)


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
    timings: dict[str, float] | None = None,
    token_usage: dict[str, int] | None = None,
) -> dict[str, object]:
    api_calls = budget_summary.get("api_calls_used", 0) if budget_summary else 0
    t = timings or {}
    total_ms = sum(t.values())
    # 修复：从全局 TokenUsageCallback 读取实际 token 用量，而非硬编码全 0
    if token_usage is None:
        try:
            from .llm import get_token_callback
            token_usage = get_token_callback().get_usage()
        except Exception:
            token_usage = {"prompt": 0, "completion": 0, "total": 0}
    return {
        "query_understanding": {
            "latency_ms": round(t.get("query_understanding", 0), 1),
            "sub_queries_count": len(query_plan.sub_queries_for_retrieval),
            "api_calls": 0,
        },
        "initial_retrieval": {
            "latency_ms": round(t.get("initial_retrieval", 0), 1),
            "seed_pool_size": deduped_count,
            "api_calls": api_calls,
            "cache_hit_rate": cache_hit_rate,
            "raw_result_count": retrieved_count,
            "average_hits_per_query": retrieved_count / max(len(query_plan.sub_queries_for_retrieval), 1),
            "minimum_recall": evaluation.get("recall"),
            "backend": backend,
        },
        "snowball": {"latency_ms": round(t.get("snowball", 0), 1)},
        "coarse": {"latency_ms": round(t.get("coarse", 0), 1)},
        "rerank": {"latency_ms": round(t.get("rerank", 0), 1)},
        "judge": {"latency_ms": round(t.get("judge", 0), 1)},
        "result_format": {
            "latency_ms": round(t.get("result_format", 0), 1),
            "output_count": selected_count,
            "graph_node_count": len(graph_output.nodes),
            "graph_edge_count": len(graph_output.edges),
        },
        "overall": {
            "total_latency_ms": round(total_ms, 1),
            "total_api_calls": api_calls,
            # 修复：使用实际追踪的 token 用量替代硬编码全 0
            "total_token_usage": token_usage,
            "precision": evaluation.get("precision"),
            "recall": evaluation.get("recall"),
            "f1": evaluation.get("f1"),
            "cache_hit_rate": cache_hit_rate,
            "pool_status_summary": pool_summary or {},
            "budget": budget_summary or {},
            "stage_timings": {k: round(v, 1) for k, v in t.items()},
        },
    }

def _notify(on_stage: Callable[[str, str], None] | None, stage: str, status: str) -> None:
    """Call the optional progress callback if provided."""
    if on_stage is not None:
        try:
            on_stage(stage, status)
        except Exception:
            pass
