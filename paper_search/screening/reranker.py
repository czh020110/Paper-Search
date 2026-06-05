"""Medium screening: Reranker high-precision re-ranking with relative threshold + Top-K fallback.

Per design 4.6: takes coarse-scored candidates, computes reranker_score via pairwise comparison
of the original query against paper titles/abstracts, then keeps high-scoring papers.
"""

from __future__ import annotations

import logging

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)


def rerank(pool: CandidatePool, query_plan: QueryPlan) -> list[Paper]:
    """Re-rank coarse-scored candidates with a reranker model.

    Transitions papers from ``rough_scored`` to ``reranked``.  When no
    reranker model is configured, papers pass through unchanged.
    """
    papers = pool.by_status("rough_scored")
    if not papers:
        logger.info("No rough-scored papers to rerank")
        return papers

    # TODO: Integrate actual reranker model (bge-reranker-v2-m3 or equivalent)
    #   1. (query, paper.title + abstract) pairwise scoring
    #   2. Relative threshold + Top-K fallback
    #   3. Transition kept papers to "reranked", excluded to "excluded"
    logger.info("Reranking %d papers (reranker model not yet integrated, passing through)", len(papers))
    pool.transition_all("rough_scored", "reranked", reason="reranker_pass_through")

    return pool.by_status("reranked")
