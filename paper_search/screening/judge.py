"""Fine screening: LLM batch relevance judgment via LangChain structured output.

Per design 4.7: batch-calls LLM to classify papers as ``高度相关 / 部分相关 /
不相关``, plus reason and contribution per paper.  Results are written into
the paper fields and candidate pool state is updated.
"""

from __future__ import annotations

import logging

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

JUDGE_BATCH_SIZE = 5


def judge(pool: CandidatePool, query_plan: QueryPlan, batch_size: int = JUDGE_BATCH_SIZE) -> list[Paper]:
    """Run LLM batch relevance judgment on reranked papers.

    Transitions papers from ``reranked`` to ``llm_judged``, then to
    ``selected`` or ``excluded`` based on LLM classification.

    When no LLM is configured (no ``LLM_API_KEY``), passes through: all
    papers are treated as ``高度相关`` and transitioned to ``selected``.
    """
    papers = pool.by_status("reranked")
    if not papers:
        logger.info("No reranked papers to judge")
        return papers

    import os
    if not os.getenv("LLM_API_KEY"):
        logger.info("No LLM API key configured, passing all %d papers as highly relevant", len(papers))
        pool.transition_all("reranked", "llm_judged", reason="judge_pass_through")
        pool.transition_all("llm_judged", "selected", reason="no_llm_default_include")
        return pool.by_status("selected")

    # TODO: Full LangChain structured output batch judgment
    #   1. Batch papers into groups of batch_size
    #   2. Call LLM with system prompt from prompts registry (精筛 prompt)
    #   3. Parse structured JSON output
    #   4. Populate paper.llm_relevance, paper.reason, paper.contribution
    #   5. Transition: llm_judged → selected (highly_relevant) | excluded
    logger.info("LLM judging %d papers (batch=%d, passthrough mode)", len(papers), batch_size)
    pool.transition_all("reranked", "llm_judged", reason="judge_batch_pending")
    pool.transition_all("llm_judged", "selected", reason="llm_judge_pass_through")

    return pool.by_status("selected")
