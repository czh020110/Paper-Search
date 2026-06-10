"""Fine screening: LLM concurrent per-paper relevance judgment.

Per design 4.7: concurrently calls LLM for each paper to classify as
``高度相关 / 部分相关 / 不相关`` with reason and contribution.
Up to 5000 papers are processed in one wave; beyond that, results are
batched across multiple waves.
"""

from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

JUDGE_CONCURRENCY = int(os.getenv("JUDGE_CONCURRENCY", "100"))
JUDGE_WAVE_CAP = int(os.getenv("JUDGE_WAVE_CAP", "5000"))


def judge(pool: CandidatePool, query_plan: QueryPlan) -> list[Paper]:
    """Run LLM concurrent per-paper relevance judgment on reranked papers.

    Each paper gets its own LLM call.  Up to *JUDGE_WAVE_CAP* papers
    are submitted in one wave; any remainder is deferred to the next.

    Transitions papers from ``reranked`` → ``llm_judged`` → ``selected``
    (高度相关/部分相关) or ``excluded`` (不相关).
    """
    papers = pool.by_status("reranked")
    if not papers:
        logger.info("No reranked papers to judge")
        return papers

    from ..llm import is_llm_key_configured
    if not is_llm_key_configured():
        logger.info("No LLM API key configured for current provider, passing all %d papers as highly relevant", len(papers))
        pool.transition_all("reranked", "llm_judged", reason="judge_pass_through")
        pool.transition_all("llm_judged", "selected", reason="no_llm_default_include")
        return pool.by_status("selected")

    logger.info("LLM judging %d papers (concurrent, max %d per wave)", len(papers), JUDGE_CONCURRENCY)

    # Process in waves of JUDGE_WAVE_CAP
    for offset in range(0, len(papers), JUDGE_WAVE_CAP):
        wave = papers[offset: offset + JUDGE_WAVE_CAP]
        _judge_wave(pool, query_plan, wave)

    selected = pool.by_status("selected")
    excluded = pool.by_status("excluded")
    logger.info("LLM judgment done: %d selected, %d excluded", len(selected), len(excluded))
    return selected


def _judge_wave(pool: CandidatePool, query_plan: QueryPlan, papers: list[Paper]) -> None:
    """Submit all papers in one wave concurrently (up to JUDGE_CONCURRENCY at a time)."""
    futures: dict[Any, Paper] = {}
    with ThreadPoolExecutor(max_workers=JUDGE_CONCURRENCY) as executor:
        for paper in papers:
            future = executor.submit(_judge_single, query_plan, paper)
            futures[future] = paper

        for future in as_completed(futures):
            paper = futures[future]
            try:
                result = future.result()
                if result is not None:
                    paper.llm_relevance = result["relevance"]
                    paper.reason = result["reason"]
                    paper.contribution = result["contribution"]
                    pool.transition(paper.id, "llm_judged", reason="llm_judge_complete")
                    if result["relevance"] in ("高度相关", "部分相关"):
                        pool.transition(paper.id, "selected", reason=f'llm_{result["relevance"]}')
                    else:
                        pool.transition(paper.id, "excluded", reason="llm_not_relevant")
                else:
                    # LLM failed for this paper → pass through
                    paper.llm_relevance = "高度相关"
                    paper.reason = "LLM 判定失败，默认纳入"
                    pool.transition(paper.id, "llm_judged", reason="judge_error")
                    pool.transition(paper.id, "selected", reason="judge_error")
            except Exception:
                logger.warning("Judge failed for paper %s", paper.id, exc_info=True)
                paper.llm_relevance = "高度相关"
                paper.reason = "LLM 判定异常，默认纳入"
                pool.transition(paper.id, "llm_judged", reason="judge_error")
                pool.transition(paper.id, "selected", reason="judge_error")


def _judge_single(query_plan: QueryPlan, paper: Paper) -> dict[str, str] | None:
    """Score a single paper via LLM with structured output."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ..llm import get_fast_llm
    from ..prompts import JUDGE_SINGLE_PROMPT
    from ..schemas import JudgeVerdict

    prompt_text = (
        JUDGE_SINGLE_PROMPT
        .replace("{query}", query_plan.original_query)
        .replace("{paper_id}", paper.id)
        .replace("{title}", paper.title)
        .replace("{abstract}", (paper.abstract or "")[:500])
        .replace("{venue}", paper.venue or "未知")
        .replace("{year}", str(paper.year or ""))
    )
    llm = get_fast_llm(temperature=0.0)
    structured_llm = llm.with_structured_output(JudgeVerdict, method="function_calling")

    try:
        result: JudgeVerdict = structured_llm.invoke([
            SystemMessage(content=prompt_text),
            HumanMessage(content="请评估这篇论文并返回 JSON 对象"),
        ])
        return {
            "relevance": result.relevance,
            "reason": result.reason,
            "contribution": result.contribution,
        }
    except Exception:
        logger.warning("Single-paper judge failed for %s", paper.id, exc_info=True)
    return None
