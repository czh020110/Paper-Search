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

import json
import logging
from typing import Any, TypedDict

from ..budget import BudgetController
from ..cache import CacheStore
from ..contracts import QueryPlan
from ..dedupe import dedupe_papers
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

# Module-level compiled graph singleton — compiled once, reused across calls.
_compiled_graph: Any | None = None

# Max papers to use for LLM keyword extraction
_QUERY_EVOLUTION_TOP_K = 5
# Max number of new keywords to extract per snowball round
_QUERY_EVOLUTION_MAX_KEYWORDS = 5
# Max papers to expand citations for (per round)
_EXPAND_CITATIONS_TOP_K = 3


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
    seen_ids: set[str]  # paper IDs already in pool before current round


def _query_evolution(state: SnowballState) -> SnowballState:
    """Extract new search terms from high-scoring papers using LLM."""
    logger.info("Snowball round %d: query evolution", state.get("current_round", 1))
    state["new_keywords"] = []

    pool = state.get("pool")
    query_plan = state.get("query_plan")
    if not pool or not query_plan:
        return state

    # Pick top papers: prefer llm_judged → highly_relevant, fallback to reranked
    papers = pool.by_status("llm_judged")
    if not papers:
        papers = pool.by_status("reranked")
    if not papers:
        papers = pool.by_status("rough_scored")
    if not papers:
        # Fallback: use raw seed papers (coarse/rerank/judge haven't run yet)
        papers = pool.by_status("seed")
    if not papers:
        logger.info("No papers in pool for query evolution")
        return state

    # Take top K and build context
    top = papers[:_QUERY_EVOLUTION_TOP_K]
    existing_kw = set(query_plan.sub_queries_for_retrieval)
    # Also collect keywords from previous snowball rounds
    prev_rounds = state.get("current_round", 0)
    all_new = state.get("new_keywords", [])
    existing_kw.update(all_new)

    papers_text = "\n".join(
        f"{i + 1}. {p.title} - {(p.abstract or '')[:300]}"
        for i, p in enumerate(top)
    )
    existing_str = ", ".join(sorted(existing_kw)) if existing_kw else "(none so far)"

    prompt = f"""You are an academic search expert. Given the user's original research query and a list of highly relevant papers found so far, extract NEW English search keywords or short phrases that could help discover additional relevant papers not yet covered.

Original research query: {query_plan.original_query}

Top relevant papers found so far:
{papers_text}

Existing keywords already searched:
{existing_str}

Extract {_QUERY_EVOLUTION_MAX_KEYWORDS} new search keywords or short phrases (in English, 1-4 words each) that represent aspects NOT yet covered by existing keywords. Focus on specific methods, techniques, or concepts from the papers above.

Return ONLY a JSON array of strings, nothing else. Example: ["keyword one", "keyword two", "keyword three"]"""

    try:
        from ..llm import get_fast_llm
        from langchain_core.messages import HumanMessage

        llm = get_fast_llm(temperature=0.0)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip() if hasattr(response, "content") else str(response).strip()

        # Parse JSON array from LLM response
        keywords = _parse_json_array(raw)
        if keywords:
            # Deduplicate against existing keywords
            new_unique = [kw for kw in keywords if kw.lower() not in {k.lower() for k in existing_kw}]
            new_unique = new_unique[:_QUERY_EVOLUTION_MAX_KEYWORDS]
            state["new_keywords"] = new_unique
            logger.info("Query evolution: extracted %d new keywords: %s", len(new_unique), new_unique)
        else:
            logger.info("Query evolution: no new keywords extracted (LLM returned empty)")
    except Exception:
        logger.warning("Query evolution LLM call failed", exc_info=True)

    return state


def _re_search(state: SnowballState) -> SnowballState:
    """Call retrieval APIs with new keywords to expand the candidate pool."""
    new_kw = state.get("new_keywords", [])
    if not new_kw:
        logger.info("No new keywords, skipping re-search")
        return state

    pool = state.get("pool")
    cache = state.get("cache")
    budget = state.get("budget")
    if not pool:
        return state

    # Record existing IDs for overlap computation later
    all_before = {p.id for p in pool.all_papers()}
    state["seen_ids"] = all_before
    logger.info("Re-search with %d new keywords (pool before: %d papers)", len(new_kw), len(all_before))

    from ..retrieval.semantic_scholar import search_papers
    from ..retrieval.openalex import search_works

    new_papers: list[Any] = []
    for keyword in new_kw:
        if budget and not budget.can_call_api():
            logger.info("Budget exhausted, stopping re-search")
            break

        # Search S2
        try:
            s2_results = search_papers(query=keyword, limit=20, cache=cache)
            if s2_results:
                if budget:
                    budget.increment_api_calls()
                new_papers.extend(s2_results)
        except Exception:
            logger.warning("S2 re-search failed for keyword: %s", keyword, exc_info=True)

        if budget and not budget.can_call_api():
            break

        # Search OA
        try:
            oa_results = search_works(query=keyword, per_page=25, cache=cache)
            if oa_results:
                if budget:
                    budget.increment_api_calls()
                new_papers.extend(oa_results)
        except Exception:
            logger.warning("OA re-search failed for keyword: %s", keyword, exc_info=True)

    if not new_papers:
        logger.info("Re-search returned no papers")
        return state

    # Deduplicate against each other and against existing pool
    deduped = dedupe_papers(new_papers)
    fresh = [p for p in deduped if p.id not in all_before]

    # Add to pool with "expanded" status
    count_added = 0
    for paper in fresh:
        try:
            pool.add(paper, status="expanded")
            count_added += 1
        except Exception:
            pass  # skip duplicates that got past dedup

    # Compute overlap ratio
    total_found = len(deduped)
    overlap = (total_found - len(fresh)) / max(total_found, 1)
    state["overlap_ratio"] = round(overlap, 4)

    logger.info(
        "Re-search done: %d new papers added, %d total found, overlap=%.1f%%",
        count_added, total_found, overlap * 100,
    )
    return state


def _expand_citations(state: SnowballState) -> SnowballState:
    """Selectively expand references/citations for high-relevance papers."""
    logger.info("Selective citation expansion")

    pool = state.get("pool")
    cache = state.get("cache")
    current_round = state.get("current_round", 1)
    if not pool:
        return state

    # Pick top-judged papers (高度相关 first, then 部分相关)
    judged = pool.by_status("llm_judged")
    target_papers = [
        p for p in judged
        if (p.llm_relevance or "").startswith("高度")
    ]
    if not target_papers:
        target_papers = [
            p for p in judged
            if (p.llm_relevance or "").startswith("部分")
        ]
    if not target_papers:
        logger.info("No judged papers to expand citations from")
        return state

    top_targets = target_papers[:_EXPAND_CITATIONS_TOP_K]
    all_before = {p.id for p in pool.all_papers()}
    edges = list(state.get("edges", []))

    from ..retrieval.semantic_scholar import get_paper_citations, get_paper_references
    from ..retrieval.openalex import get_work_citations, get_work_references

    for paper in top_targets:
        s2_id = paper.source_ids.get("semantic_scholar")
        oa_id = paper.source_ids.get("openalex")

        # Fetch references from S2
        if s2_id:
            for source_name, fetcher in [("refs", get_paper_references), ("cites", get_paper_citations)]:
                try:
                    refs = fetcher(s2_id, cache=cache, limit=20)
                    for ref in refs:
                        if ref.id not in all_before:
                            try:
                                pool.add(ref, status="expanded")
                                all_before.add(ref.id)
                            except Exception:
                                pass
                        edges.append({
                            "source_paper_id": paper.id,
                            "target_paper_id": ref.id,
                            "edge_type": "reference" if source_name == "refs" else "citation",
                            "discovered_round": current_round,
                        })
                except Exception:
                    logger.warning("S2 %s failed for %s", source_name, s2_id, exc_info=True)

        # Fetch citations from OA
        if oa_id:
            # OA ID is a full URL like "https://openalex.org/W..." → extract short ID
            oa_short = oa_id.strip().rstrip("/").rsplit("/", 1)[-1]
            for source_name, fetcher in [("refs", get_work_references), ("cites", get_work_citations)]:
                try:
                    refs = fetcher(oa_short, cache=cache, per_page=20)
                    for ref in refs:
                        if ref.id not in all_before:
                            try:
                                pool.add(ref, status="expanded")
                                all_before.add(ref.id)
                            except Exception:
                                pass
                        edges.append({
                            "source_paper_id": paper.id,
                            "target_paper_id": ref.id,
                            "edge_type": "reference" if source_name == "refs" else "citation",
                            "discovered_round": current_round,
                        })
                except Exception:
                    logger.warning("OA %s failed for %s", source_name, oa_short, exc_info=True)

    state["edges"] = edges
    logger.info("Citation expansion done: %d edges recorded", len(edges))
    return state


def _check_convergence(state: SnowballState) -> SnowballState:
    """Evaluate convergence conditions and set converged flag."""
    budget = state.get("budget")
    current_round = state.get("current_round", 0)
    max_rounds = state.get("max_rounds", 3)
    overlap = state.get("overlap_ratio", 0.0)
    new_kw = state.get("new_keywords", [])

    # Increment round counter on first convergence check of each iteration
    state["current_round"] = current_round + 1

    converged = False
    reason = ""

    if current_round + 1 >= max_rounds:
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
        logger.info("Snowball converged after round %d: %s", current_round + 1, reason)
    else:
        logger.info("Snowball continuing to round %d", current_round + 2)

    state["converged"] = converged
    state["overlap_ratio"] = overlap
    return state


def _parse_json_array(text: str) -> list[str]:
    """Parse a JSON array of strings from LLM response text.

    Tries direct ``json.loads`` first; if that fails, attempts to locate
    ``[...]`` brackets and parse the enclosed content.
    """
    text = text.strip()
    # Try direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list) and all(isinstance(item, str) for item in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: find [...] and parse just that portion
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, list):
                return [str(item) for item in parsed if isinstance(item, str)]
        except json.JSONDecodeError:
            pass

    return []


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
