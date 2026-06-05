"""Coarse screening: BM25 + Embedding + structural signal triple scoring and truncation.

Per design 4.5: candidate pool > 80 triggers triple scoring; ≤ 80 skips.
"""

from __future__ import annotations

import logging

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

POOL_SKIP_THRESHOLD = 80


def coarse_score(pool: CandidatePool, query_plan: QueryPlan) -> list[Paper]:
    """Run coarse triple-scoring on seed and expanded papers and truncate the pool.

    When the pool has ≤ *POOL_SKIP_THRESHOLD* papers, the stage is skipped
    (papers transition from ``seed``/``expanded`` to ``rough_scored``).

    Returns the papers after coarse screening.
    """
    papers = pool.by_status("seed") + pool.by_status("expanded")
    if not papers:
        logger.info("No seed or expanded papers to coarsely score")
        return papers

    if len(papers) <= POOL_SKIP_THRESHOLD:
        logger.info("Pool size %d ≤ %d, skipping coarse scoring", len(papers), POOL_SKIP_THRESHOLD)
        pool.transition_all("seed", "rough_scored", reason="pool_skip")
        pool.transition_all("expanded", "rough_scored", reason="pool_skip")
        return pool.by_status("rough_scored")

    # TODO: Implement actual triple scoring when models are available
    logger.info("Coarse scoring %d papers (triple scoring not yet implemented, passing through)", len(papers))
    pool.transition_all("seed", "rough_scored", reason="coarse_pass_through")
    pool.transition_all("expanded", "rough_scored", reason="coarse_pass_through")

    return pool.by_status("rough_scored")
