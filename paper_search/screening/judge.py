"""Fine screening: LLM batch relevance judgment via LangChain structured output.

Per design 4.7: batch-calls LLM to classify papers as ``高度相关 / 部分相关 /
不相关``, plus reason and contribution per paper.  Results are written into
the paper fields and candidate pool state is updated.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

JUDGE_BATCH_SIZE = 5


def judge(pool: CandidatePool, query_plan: QueryPlan, batch_size: int = JUDGE_BATCH_SIZE) -> list[Paper]:
    """Run LLM batch relevance judgment on reranked papers.

    Transitions papers from ``reranked`` to ``llm_judged``, then to
    ``selected`` (高度相关/部分相关) or ``excluded`` (不相关).

    When no LLM is configured (no ``LLM_API_KEY``), passes through: all
    papers are treated as ``高度相关`` and transitioned to ``selected``.
    """
    papers = pool.by_status("reranked")
    if not papers:
        logger.info("No reranked papers to judge")
        return papers

    if not os.getenv("LLM_API_KEY"):
        logger.info("No LLM API key configured, passing all %d papers as highly relevant", len(papers))
        pool.transition_all("reranked", "llm_judged", reason="judge_pass_through")
        pool.transition_all("llm_judged", "selected", reason="no_llm_default_include")
        return pool.by_status("selected")

    logger.info("LLM batch judging %d papers (batch_size=%d)", len(papers), batch_size)

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start: batch_start + batch_size]
        _judge_batch(pool, query_plan, batch)

    selected = pool.by_status("selected")
    excluded = pool.by_status("excluded")
    logger.info("LLM judgment done: %d selected, %d excluded", len(selected), len(excluded))
    return selected


def _judge_batch(pool: CandidatePool, query_plan: QueryPlan, batch: list[Paper]) -> None:
    papers_json = json.dumps(
        [
            {
                "paper_id": paper.id,
                "title": paper.title,
                "abstract": (paper.abstract or "")[:500],
                "venue": paper.venue or "未知",
                "year": paper.year or "",
            }
            for paper in batch
        ],
        ensure_ascii=False,
    )

    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.output_parsers import JsonOutputParser

    from ..llm import get_fast_llm
    from ..prompts import JUDGE_PROMPT

    prompt_text = JUDGE_PROMPT.replace("{query}", query_plan.original_query).replace("{papers_json}", papers_json)
    llm = get_fast_llm(temperature=0.0)
    parser = JsonOutputParser()

    messages = [
        SystemMessage(content=prompt_text),
        HumanMessage(content="请评估上述论文并返回 JSON 数组"),
    ]

    try:
        response = llm.invoke(messages)
        results = parser.parse(response.content)  # type: ignore[arg-type]

        if not isinstance(results, list):
            logger.error("LLM judge returned non-array: %s", type(results).__name__)
            _pass_through_batch(pool, batch)
            return

        judged_ids: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            paper_id = item.get("paper_id", "")
            relevance = item.get("relevance", "不相关")
            reason = item.get("reason", "")
            contribution = item.get("contribution", "")

            paper = pool.get(paper_id)
            if paper is None:
                logger.warning("LLM judge returned unknown paper_id: %s", paper_id)
                continue

            paper.llm_relevance = relevance
            paper.reason = reason
            paper.contribution = contribution
            pool.transition(paper_id, "llm_judged", reason="llm_judge_complete")
            judged_ids.add(paper_id)

            if relevance in ("高度相关", "部分相关"):
                pool.transition(paper_id, "selected", reason=f"llm_{relevance}")
            else:
                pool.transition(paper_id, "excluded", reason="llm_not_relevant")

        # Papers LLM didn't mention → pass through as selected
        for paper in batch:
            if paper.id not in judged_ids:
                paper.llm_relevance = "高度相关"
                paper.reason = "LLM 未返回判定，默认纳入"
                pool.transition(paper.id, "llm_judged", reason="judge_fallback")
                pool.transition(paper.id, "selected", reason="judge_fallback")

    except Exception:
        logger.error("LLM judge batch failed", exc_info=True)
        _pass_through_batch(pool, batch)


def _pass_through_batch(pool: CandidatePool, batch: list[Paper]) -> None:
    """Pass through a batch when LLM judgment fails."""
    for paper in batch:
        paper.llm_relevance = "高度相关"
        paper.reason = "LLM 判定失败，默认纳入"
        pool.transition(paper.id, "llm_judged", reason="judge_error")
        pool.transition(paper.id, "selected", reason="judge_error")
