from __future__ import annotations

import logging

from .contracts import Paper

logger = logging.getLogger(__name__)

POOL_STATUSES = ("discovered", "seed", "expanded", "rough_scored", "reranked", "llm_judged", "selected", "excluded")

VALID_TRANSITIONS: dict[str, set[str]] = {
    "discovered": {"seed", "excluded"},
    "seed": {"expanded", "rough_scored", "selected", "excluded"},
    "expanded": {"rough_scored", "excluded"},
    "rough_scored": {"reranked", "selected", "excluded"},
    "reranked": {"llm_judged", "selected", "excluded"},
    "llm_judged": {"selected", "excluded"},
    "selected": set(),
    "excluded": set(),
}


class InvalidTransition(Exception):
    """Raised when a candidate pool status transition is not allowed."""


class CandidatePool:
    """State machine governing the lifecycle of candidate papers.

    Each paper progresses through ``discovered → seed → expanded →
    rough_scored → reranked → llm_judged → selected/excluded``.  Transitions
    are validated against :data:`VALID_TRANSITIONS`.
    """

    def __init__(self) -> None:
        self._papers: dict[str, Paper] = {}

    def add(self, paper: Paper, status: str = "discovered") -> None:
        if status not in POOL_STATUSES:
            raise InvalidTransition(f"Invalid initial status: {status}")
        paper.pool_status = status
        self._papers[paper.id] = paper
        logger.info(
            "Paper added to pool",
            extra={"stage": "pool_add", "paper_id": paper.id, "status": status},
        )

    def transition(self, paper_id: str, new_status: str, reason: str | None = None) -> None:
        paper = self._papers.get(paper_id)
        if paper is None:
            raise KeyError(f"Paper {paper_id} not found in pool")
        current = paper.pool_status
        if new_status not in POOL_STATUSES:
            raise InvalidTransition(f"Invalid target status: {new_status}")
        if new_status not in VALID_TRANSITIONS.get(current, set()):
            raise InvalidTransition(f"Cannot transition {paper_id} from {current} to {new_status}")
        paper.pool_status = new_status
        logger.info(
            "Pool status transition",
            extra={"stage": "pool_transition", "paper_id": paper_id, "from": current, "to": new_status, "budget_reason": reason},
        )

    def transition_all(self, current_status: str, new_status: str, reason: str | None = None) -> int:
        """Bulk-transition all papers with *current_status* to *new_status*.

        Returns the number of papers transitioned.
        """
        count = 0
        for paper_id in list(self._papers.keys()):
            paper = self._papers[paper_id]
            if paper.pool_status == current_status:
                self.transition(paper_id, new_status, reason=reason)
                count += 1
        return count

    def get(self, paper_id: str) -> Paper | None:
        return self._papers.get(paper_id)

    def by_status(self, status: str) -> list[Paper]:
        return [p for p in self._papers.values() if p.pool_status == status]

    def all_papers(self) -> list[Paper]:
        return list(self._papers.values())

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for paper in self._papers.values():
            counts[paper.pool_status] = counts.get(paper.pool_status, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self._papers)
