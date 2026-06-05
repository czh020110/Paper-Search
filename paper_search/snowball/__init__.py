"""Snowball module with LangGraph StateGraph.

Per design 4.4: LLM-driven query evolution + selective citation expansion
in an iterative loop managed by a LangGraph state graph.

Nodes:
  - query_evolution : extract new search terms from high-scoring papers
  - re_search : call retrieval APIs with new queries
  - expand_citations : selective references/citations expansion
  - check_convergence : evaluate stop conditions

Conditional edges exit when:
  - overlap > 70%
  - no new keywords for one round
  - max rounds reached
  - budget exhausted

Note: the compiled LangGraph graph is cached at module level via
``_get_snowball_graph()`` to avoid recompiling the StateGraph on every
call — compiling is expensive and repeated compilation was the root cause
of a 100 GB+ GC death spiral during test discovery.
"""

from __future__ import annotations

import logging
from typing import Any, TypedDict

from ..budget import BudgetController
from ..cache import CacheStore
from ..contracts import QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

# Module-level compiled graph singleton — compiled once, reused across calls.
_compiled_graph: Any | None = None


class SnowballState(TypedDict, total=False):
    """LangGraph state for the snowball iteration.

    Uses only list/dict containers to avoid confusing LangGraph's state-diff
    machinery (mutable sets can trigger excessive deep-copy behaviour in the
    runtime).
    """

    pool: CandidatePool
    query_plan: QueryPlan
    budget: BudgetController
    cache: CacheStore | None
    current_round: int
    max_rounds: int
    new_keywords: list[str]
    overlap_ratio: float
    edges: list[dict[str, Any]]
    converged: bool


def _query_evolution(state: SnowballState) -> SnowballState:
    """Extract new search terms from high-scoring papers in the pool."""
    logger.info("Snowball round %d: query evolution", state.get("current_round", 1))

    # TODO: Use LLM to extract new keywords from highly-scored paper titles/abstracts
    # For now, mark as no new keywords (triggers convergence on next check)
    state["new_keywords"] = []
    return state


def _re_search(state: SnowballState) -> SnowballState:
    """Call retrieval APIs with new keywords to expand the candidate pool."""
    new_kw = state.get("new_keywords", [])
    if not new_kw:
        logger.info("No new keywords, skipping re-search")
        return state

    # TODO: Call S2/OA APIs with new keyword queries
    # Papers added to pool with status="expanded"
    logger.info("Re-search with %d new keywords", len(new_kw))
    return state


def _expand_citations(state: SnowballState) -> SnowballState:
    """Selectively expand references/citations for high-relevance papers."""
    # TODO: Fetch references & citations via S2/OA APIs
    # Only for papers judged as 高度相关 or 部分相关
    # Record edges in state["edges"]
    logger.info("Selective citation expansion")
    return state


def _check_convergence(state: SnowballState) -> SnowballState:
    """Evaluate convergence conditions and set converged flag."""
    budget = state.get("budget")
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)
    overlap = state.get("overlap_ratio", 0.0)
    new_kw = state.get("new_keywords", [])

    converged = False
    reason = ""

    if current_round >= max_rounds:
        converged = True
        reason = "max_rounds"
    elif overlap > 0.70:
        converged = True
        reason = f"overlap_{overlap:.0%}"
    elif not new_kw:
        converged = True
        reason = "no_new_keywords"
    elif budget and not budget.can_start_round():
        converged = True
        reason = "budget_exhausted"

    if converged:
        logger.info("Snowball converged: %s", reason)

    state["converged"] = converged
    return state


def _continue_snowball(state: SnowballState) -> str:
    """Return the next node name after check_convergence."""
    if state.get("converged", False):
        return "__end__"
    return "query_evolution"


def _build_snowball_graph() -> Any:
    """Build and compile a new LangGraph StateGraph for the snowball loop.

    Prefer ``_get_snowball_graph()`` unless you explicitly need a fresh graph.
    """
    try:
        from langgraph.graph import END, StateGraph

        graph = StateGraph(SnowballState)

        graph.add_node("query_evolution", _query_evolution)
        graph.add_node("re_search", _re_search)
        graph.add_node("expand_citations", _expand_citations)
        graph.add_node("check_convergence", _check_convergence)

        graph.set_entry_point("query_evolution")
        graph.add_edge("query_evolution", "re_search")
        graph.add_edge("re_search", "expand_citations")
        graph.add_edge("expand_citations", "check_convergence")

        graph.add_conditional_edges(
            "check_convergence",
            _continue_snowball,
            {"query_evolution": "query_evolution", "__end__": END},
        )

        return graph.compile()
    except ImportError:
        logger.warning("LangGraph not available, snowball graph not compiled")
        return None


def _get_snowball_graph() -> Any:
    """Return the module-level compiled snowball graph, compiling on first call."""
    global _compiled_graph
    if _compiled_graph is None:
        logger.debug("Compiling snowball graph (first call)")
        _compiled_graph = _build_snowball_graph()
    return _compiled_graph


def _reset_snowball_graph() -> None:
    """Release the cached compiled graph (useful for test teardown)."""
    global _compiled_graph
    _compiled_graph = None


# Backward-compatible public alias for any external callers.
build_snowball_graph = _build_snowball_graph


def run_snowball(
    pool: CandidatePool,
    query_plan: QueryPlan,
    budget: BudgetController,
    cache: CacheStore | None = None,
    max_rounds: int = 3,
) -> tuple[CandidatePool, list[dict[str, Any]]]:
    """Run the snowball iteration loop using the LangGraph state graph.

    Returns the updated candidate pool and collected citation edges.
    """
    graph = _get_snowball_graph()
    if graph is None:
        logger.warning("Snowball graph unavailable, skipping snowball")
        return pool, []

    initial_state: SnowballState = {
        "pool": pool,
        "query_plan": query_plan,
        "budget": budget,
        "cache": cache,
        "current_round": 0,
        "max_rounds": max_rounds,
        "new_keywords": [],
        "overlap_ratio": 0.0,
        "edges": [],
        "converged": False,
    }

    final_state = graph.invoke(initial_state)
    return final_state["pool"], final_state.get("edges", [])
