"""Coarse screening: BM25 + Embedding + structure signal triple scoring and truncation.

Per design 4.5: candidate pool > COARSE_POOL_SKIP_THRESHOLD triggers scoring; ≤ skips.
路 A: BM25 sparse keyword matching
路 B: DashScope text-embedding-v4 cosine similarity (deferred when unavailable)
路 C: Structure signals (year proximity, venue match, citation count)

All tuning parameters are configurable via environment variables so they can be
adjusted without code changes.
"""

from __future__ import annotations

import logging
import math
import os
import re

from ..contracts import Paper, QueryPlan
from ..pool import CandidatePool

logger = logging.getLogger(__name__)

# -- Tuning knobs (all configurable via env vars) --------------------------------

POOL_SKIP_THRESHOLD = int(os.getenv("COARSE_POOL_SKIP_THRESHOLD", "40"))
EMBEDDING_MIN_SIMILARITY = float(os.getenv("COARSE_EMBEDDING_MIN_SIMILARITY", "0.35"))

# Triple-scoring weights (must sum roughly to 1.0)
WEIGHT_BM25 = float(os.getenv("COARSE_WEIGHT_BM25", "0.35"))
WEIGHT_EMBEDDING = float(os.getenv("COARSE_WEIGHT_EMBEDDING", "0.35"))
WEIGHT_STRUCTURE = float(os.getenv("COARSE_WEIGHT_STRUCTURE", "0.30"))

# Fallback weights when embedding is unavailable
WEIGHT_BM25_FALLBACK = float(os.getenv("COARSE_WEIGHT_BM25_FALLBACK", "0.60"))
WEIGHT_STRUCTURE_FALLBACK = float(os.getenv("COARSE_WEIGHT_STRUCTURE_FALLBACK", "0.40"))

# Relative threshold multiplier applied to the mean combined score
RELATIVE_THRESHOLD_FACTOR = float(os.getenv("COARSE_RELATIVE_THRESHOLD_FACTOR", "0.3"))


def coarse_score(pool: CandidatePool, query_plan: QueryPlan) -> list[Paper]:
    """Run coarse scoring on seed and expanded papers and truncate the pool.

    When the pool has ≤ *POOL_SKIP_THRESHOLD* papers, the stage is skipped
    (papers transition from ``seed``/``expanded`` to ``rough_scored``).

    Otherwise, BM25 + structure signals are computed and papers scoring
    below the relative threshold are excluded.
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

    # Build query tokens for BM25
    query_text = _build_query_text(query_plan)
    query_tokens = _tokenize(query_text)

    # Build corpus from paper titles + abstracts
    corpus = [_paper_text(p) for p in papers]
    corpus_tokens = [_tokenize(text) for text in corpus]

    # 路 A: BM25 sparse scoring
    bm25_scores = _bm25_score(query_tokens, corpus_tokens)

    # 路 B: Embedding cosine similarity (dashscope text-embedding-v4)
    embedding_scores = _embedding_scores(query_text, corpus)

    # 路 C: Structure signals (year proximity, venue match, citation count)
    structure_scores = [_structure_score(p, query_plan) for p in papers]

    # Weighted fusion — adjust weights based on what's available
    bm25_norm = _minmax_norm(bm25_scores)
    structure_norm = _minmax_norm(structure_scores)

    has_embedding = any(s is not None for s in embedding_scores)
    if has_embedding:
        # Replace None with 0.0 so we can still compute weighted average
        emb_present = [s if s is not None else 0.0 for s in embedding_scores]
        emb_norm = _minmax_norm(emb_present)
        combined = [WEIGHT_BM25 * b + WEIGHT_EMBEDDING * e + WEIGHT_STRUCTURE * s for b, e, s in zip(bm25_norm, emb_norm, structure_norm)]
        logger.info("Coarse triple scoring (BM25 + Embedding + Structure) on %d papers", len(papers))
    else:
        combined = [WEIGHT_BM25_FALLBACK * b + WEIGHT_STRUCTURE_FALLBACK * s for b, s in zip(bm25_norm, structure_norm)]
        logger.info("Coarse double scoring (BM25 + Structure, no embedding available) on %d papers", len(papers))

    # Relative threshold: keep papers above mean * 0.3
    mean_score = sum(combined) / max(len(combined), 1)
    threshold = mean_score * RELATIVE_THRESHOLD_FACTOR

    kept = 0
    emb_excluded = 0
    for i, (paper, score) in enumerate(zip(papers, combined)):
        # Embedding hard floor: papers with cosine similarity below the minimum
        # are semantically irrelevant — exclude regardless of other signals.
        emb_score = embedding_scores[i] if i < len(embedding_scores) else None
        if emb_score is not None and emb_score < EMBEDDING_MIN_SIMILARITY:
            pool.transition(paper.id, "excluded", reason=f"emb_sim={emb_score:.3f}_below_min")
            emb_excluded += 1
        elif score >= threshold:
            pool.transition(paper.id, "rough_scored", reason=f"coarse_score={score:.3f}")
            kept += 1
        else:
            pool.transition(paper.id, "excluded", reason=f"coarse_score={score:.3f}_below_threshold")

    total_excluded = len(papers) - kept
    logger.info(
        "Coarse scoring done: %d kept, %d excluded (emb=%d, combined=%d, threshold=%.3f)",
        kept, total_excluded, emb_excluded, total_excluded - emb_excluded, threshold,
    )
    return pool.by_status("rough_scored")


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> list[str]:
    """Simple English tokenizer: lowercase, split on whitespace/punctuation, filter short tokens."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 1]


def _bm25_score(query_tokens: list[str], corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Compute BM25 scores for a list of tokenized documents against query tokens."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        logger.warning("rank-bm25 not installed, using fallback TF scoring")
        return _fallback_tf_score(query_tokens, corpus_tokens)

    if not corpus_tokens or not query_tokens:
        return [0.0] * len(corpus_tokens)

    bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b)
    scores = bm25.get_scores(query_tokens)
    return [float(s) for s in scores]


def _fallback_tf_score(query_tokens: list[str], corpus_tokens: list[list[str]]) -> list[float]:
    """Simple TF-based scoring fallback when rank-bm25 is unavailable."""
    query_set = set(query_tokens)
    if not query_set:
        return [0.0] * len(corpus_tokens)
    scores: list[float] = []
    for doc in corpus_tokens:
        matches = sum(1 for t in doc if t in query_set)
        scores.append(float(matches) / max(len(doc), 1))
    return scores


# ---------------------------------------------------------------------------
# Structure signal helpers
# ---------------------------------------------------------------------------


def _structure_score(paper: Paper, query_plan: QueryPlan) -> float:
    """Compute a structure signal score for a paper.

    Factors: year proximity to query constraint, venue match bonus,
    citation count log-normalized.
    """
    score = 0.0

    # Year proximity: prefer papers close to the query year constraint
    year_constraint = _extract_year_constraint(query_plan)
    if paper.year is not None and year_constraint is not None:
        year_gap = abs(paper.year - year_constraint)
        score += max(0.0, 1.0 - year_gap * 0.1)  # decays by year distance

    # Citation count: log-normalize, cap at 5.0
    if paper.citation_count is not None and paper.citation_count > 0:
        score += min(5.0, math.log1p(paper.citation_count))

    # Venue match: bonus for matching preferred venues
    preferred = query_plan.ranking_signals.get("preferred_venues", [])
    venue = (paper.venue or "").lower()
    for pv in preferred:
        if pv.lower() in venue or venue in pv.lower():
            score += 2.0
            break

    return score


def _extract_year_constraint(query_plan: QueryPlan) -> int | None:
    hf = query_plan.hard_filters.get("year")
    if isinstance(hf, dict):
        val = hf.get("value")
        if isinstance(val, int):
            return val
    return None


def _build_query_text(query_plan: QueryPlan) -> str:
    """Build a search text from the query plan for BM25 scoring."""
    parts: list[str] = [query_plan.original_query]
    core = query_plan.semantic_queries.get("core_concepts", [])
    methods = query_plan.semantic_queries.get("methodologies", [])
    parts.extend(core)
    parts.extend(methods)
    return " ".join(parts)


def _paper_text(paper: Paper) -> str:
    return f"{paper.title} {(paper.abstract or '')}"


def _minmax_norm(values: list[float]) -> list[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx == mn:
        return [0.5] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


# ---------------------------------------------------------------------------
# Embedding helpers (路 B)
# ---------------------------------------------------------------------------


def _embedding_scores(query_text: str, corpus: list[str]) -> list[float | None]:
    """Compute cosine similarity between query embedding and each paper embedding.

    Returns a list parallel to *corpus*; ``None`` entries indicate embedding
    was unavailable (caller should fall back to zero-weight for that paper).
    """
    try:
        from ..embedding import batch_embeddings, get_embedding

        # Get query embedding
        query_emb = get_embedding(query_text[:1000])
        if query_emb is None:
            logger.debug("Query embedding unavailable, skipping 路 B")
            return [None] * len(corpus)

        # Get paper embeddings in batch
        paper_texts = [text[:1000] for text in corpus]  # 1000 char limit
        paper_embs = batch_embeddings(paper_texts)

        scores: list[float | None] = []
        for paper_emb in paper_embs:
            if paper_emb is not None:
                scores.append(_cosine_similarity(query_emb, paper_emb))
            else:
                scores.append(None)
        return scores
    except ImportError:
        logger.debug("dashscope not installed, skipping 路 B")
        return [None] * len(corpus)
    except Exception:
        logger.warning("Embedding scoring failed, skipping 路 B", exc_info=True)
        return [None] * len(corpus)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
