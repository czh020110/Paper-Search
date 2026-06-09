"""Medium screening: DashScope qwen3-rerank for high-precision re-ranking.

Per design 4.6: takes coarse-scored candidates, sends (query, document) pairs
to DashScope's qwen3-rerank API, then keeps papers above the relative threshold
with a Top-K fallback.

Falls back gracefully when the reranker provider is not configured.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

TOP_K_FALLBACK = int(os.getenv("RERANKER_TOP_K_FALLBACK", "30"))
RERANKER_RELATIVE_THRESHOLD_FACTOR = float(os.getenv("RERANKER_RELATIVE_THRESHOLD_FACTOR", "0.7"))


def rerank(pool: CandidatePool, query_plan: QueryPlan) -> list[Paper]:
    """Re-rank coarse-scored candidates with DashScope qwen3-rerank.

    Transitions papers from ``rough_scored`` to ``reranked``.  Papers above
    the relative threshold (or Top-K fallback) are kept; others excluded.

    When no reranker provider is configured, papers pass through unchanged.
    """
    papers = pool.by_status("rough_scored")
    if not papers:
        logger.info("No rough-scored papers to rerank")
        return papers

    provider = os.getenv("RERANKER_PROVIDER", "")
    reranker_enabled = os.getenv("RERANKER_ENABLED", "false").lower() not in ("0", "false", "no")
    if not provider or not reranker_enabled:
        reason = "RERANKER_PROVIDER not set" if not provider else "RERANKER_ENABLED=false"
        logger.info("Reranker skipped (%s), passing %d papers through", reason, len(papers))
        _fallback_pass_through(pool, papers)
        return pool.by_status("reranked")

    logger.info("Reranking %d papers via %s", len(papers), provider)

    # Build query text with English keywords for better matching
    query_text = query_plan.original_query
    core = [c for c in query_plan.semantic_queries.get("core_concepts", []) if c.isascii()]
    if core:
        query_text = query_text + " " + " ".join(core[:8])

    # Prepare documents for reranker
    docs = [
        {"id": p.id, "text": f"{p.title} {(p.abstract or '')[:500]}"}
        for p in papers
    ]

    results = _call_rerank_api(query_text, docs)

    # Fetch scores from reranker response
    score_map: dict[str, float] = {}
    for r in results:
        score_map[r["id"]] = float(r.get("score", 0.0))

    # Store scores and determine threshold
    score_list = [score_map.get(p.id, 0.0) for p in papers]
    for paper, score in zip(papers, score_list):
        paper.reranker_score = score

    mean_score = sum(score_list) / max(len(score_list), 1)
    threshold = max(mean_score * RERANKER_RELATIVE_THRESHOLD_FACTOR, 0.01)

    scored = sorted(zip(papers, score_list), key=lambda x: x[1], reverse=True)
    kept = 0
    for i, (paper, score) in enumerate(scored):
        if i < TOP_K_FALLBACK or score >= threshold:
            pool.transition(paper.id, "reranked", reason=f"rerank_score={score:.3f}")
            kept += 1
        else:
            pool.transition(paper.id, "excluded", reason=f"rerank_score={score:.3f}_below_threshold")

    logger.info(
        "Reranking done: %d kept, %d excluded (threshold=%.3f, top_k=%d)",
        kept, len(papers) - kept, threshold, TOP_K_FALLBACK,
    )
    return pool.by_status("reranked")


def _call_rerank_api(query: str, documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Call DashScope qwen3-rerank via the official SDK. Falls back gracefully on failure."""
    api_key = os.getenv("RERANKER_API_KEY")
    model = os.getenv("RERANKER_MODEL", "qwen3-rerank")

    if not api_key:
        logger.warning("RERANKER_API_KEY not configured, falling back")
        return [{"id": d["id"], "score": 0.0} for d in documents]

    import dashscope
    from http import HTTPStatus

    # Set key both as env var (for first import) and module attr (for already-imported)
    os.environ["DASHSCOPE_API_KEY"] = api_key
    dashscope.api_key = api_key

    try:
        resp = dashscope.TextReRank.call(
            model=model,
            query=query,
            documents=[d["text"] for d in documents],
            top_n=min(len(documents), 100),
            return_documents=False,
            instruct="Given an academic research query, retrieve relevant papers that match the search intent.",
        )

        if resp.status_code == HTTPStatus.OK:
            results: list[dict[str, Any]] = []
            seen: set[str] = set()
            out: dict[str, Any] = resp.output if resp.output is not None else {}
            raw_results: list[dict[str, Any]] = out.get("results", [])  # type: ignore[assignment]
            for item in raw_results:
                idx = item.get("index", -1)
                score = item.get("relevance_score", 0.0)
                if 0 <= idx < len(documents):
                    did = documents[idx]["id"]
                    results.append({"id": did, "score": float(score)})
                    seen.add(did)

            for d in documents:
                if d["id"] not in seen:
                    results.append({"id": d["id"], "score": 0.0})
            return results
        else:
            logger.error("Reranker API returned %s: %s", resp.status_code, resp.message)
    except Exception:
        logger.error("Reranker API call failed", exc_info=True)

    return [{"id": d["id"], "score": 0.0} for d in documents]


def _fallback_pass_through(pool: CandidatePool, papers: list[Paper]) -> None:
    """Pass-through when no reranker is configured."""
    for paper in papers:
        paper.reranker_score = 0.0
        pool.transition(paper.id, "reranked", reason="no_reranker_provider")
