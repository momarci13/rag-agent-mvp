"""Failure-driven KB expansion.

When RAG retrieval quality falls below a threshold the system automatically
searches for gap-filling papers across all configured sources and ingests
the top results so subsequent retrievals are better-grounded.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.hybrid import LiteHybridRAG
    from agents.llm import OllamaLLM

logger = logging.getLogger(__name__)

RETRIEVAL_QUALITY_THRESHOLD = 0.25  # mean hybrid score below which expansion fires


def score_retrieval_quality(docs: list[dict]) -> float:
    """Mean hybrid retrieval score across returned docs. Returns 0.0 if empty."""
    if not docs:
        return 0.0
    return sum(d.get("score", 0.0) for d in docs) / len(docs)


def expand_kb_on_weak_retrieval(
    query: str,
    docs: list[dict],
    rag: "LiteHybridRAG",
    llm: "OllamaLLM",
    task_id: str,
    n_papers: int = 5,
    threshold: float = RETRIEVAL_QUALITY_THRESHOLD,
) -> tuple[list[dict], bool]:
    """Expand the KB when retrieval quality is too low.

    If mean score < threshold:
      1. Run multi_source_search for gap-filling papers
      2. Auto-ingest hop=0 results into RAG
      3. Save all candidates to pending_sources.json for user review
      4. Re-retrieve and return the improved doc set

    Returns (docs, expanded) where expanded=True signals that new
    papers were ingested. Always returns at least the original docs
    on failure so callers never get an empty context.
    """
    quality = score_retrieval_quality(docs)
    if quality >= threshold:
        return docs, False

    logger.info(
        "[KB_EXPAND] score=%.3f < threshold=%.3f for '%s...' — triggering expansion",
        quality, threshold, query[:60],
    )

    try:
        from tools.source_search import multi_source_search, save_pending_sources
        results = multi_source_search(query, n_papers=n_papers, llm=llm)
        hop0 = [r for r in results if r.hop == 0]
        if hop0:
            added = rag.ingest_papers(hop0)
            logger.info("[KB_EXPAND] Ingested %d papers (%d chunks)", len(hop0), added)
        if results:
            save_pending_sources(task_id, results)
        new_docs = rag.retrieve(query, k=max(len(docs), 6))
        new_quality = score_retrieval_quality(new_docs)
        logger.info("[KB_EXPAND] Quality %.3f → %.3f", quality, new_quality)
        return new_docs if new_docs else docs, True
    except Exception as exc:
        logger.warning("[KB_EXPAND] Expansion failed, using original docs: %s", exc)
        return docs, False
