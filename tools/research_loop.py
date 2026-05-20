"""Iterative autonomous research loop.

SEARCH → RETRIEVE → GAP-DETECT → EXPAND → REPORT

Runs without human intervention: detects knowledge gaps after each retrieval,
searches for gap-filling papers, expands the citation DAG, and generates a
final structured report after convergence or exhausting iterations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:
    from agents.llm import OllamaLLM
    from rag.hybrid import LiteHybridRAG
    from tools.citation_dag import CitationDAG
    from tools.auto_report import ResearchReport

logger = logging.getLogger(__name__)


@dataclass
class LoopIteration:
    iteration: int
    keywords_used: list[str] = field(default_factory=list)
    papers_found: int = 0
    kb_before: int = 0
    kb_after: int = 0
    gaps_detected: list[str] = field(default_factory=list)
    quality_score: float = 0.0


@dataclass
class AutonomousResearchResult:
    topic: str
    task_id: str
    iterations: list[LoopIteration] = field(default_factory=list)
    total_papers_ingested: int = 0
    final_report: "ResearchReport | None" = None
    citation_dag_nodes: int = 0


def detect_knowledge_gaps(
    query: str,
    docs: list[dict],
    llm: "OllamaLLM",
) -> list[str]:
    """Ask the LLM which subtopics are missing from retrieved docs.

    Returns up to 3 short gap-search queries.
    """
    if not docs:
        return []
    snippets = "\n".join(f"- {d.get('text', '')[:200]}" for d in docs[:8])
    msgs = [
        {
            "role": "system",
            "content": (
                "Given a research topic and retrieved paper snippets, identify "
                "important subtopics NOT covered. "
                "Output JSON: {\"gaps\": [\"gap_query_1\", ...]}. "
                "Each gap is a short search query (≤8 words). Return 1-3 gaps."
            ),
        },
        {
            "role": "user",
            "content": f"Topic: {query}\n\nRetrieved snippets:\n{snippets}",
        },
    ]
    try:
        result = llm.chat_json(msgs, temperature=0.0)
        gaps = [str(g) for g in (result.get("gaps") or [])[:3] if g]
        logger.info("[LOOP] Detected %d knowledge gaps: %s", len(gaps), gaps)
        return gaps
    except Exception as exc:
        logger.warning("[LOOP] Gap detection failed: %s", exc)
        return []


def autonomous_research_loop(
    topic: str,
    llm: "OllamaLLM",
    rag: "LiteHybridRAG",
    citation_dag: "CitationDAG",
    task_id: str | None = None,
    n_iterations: int = 3,
    n_papers_per_iter: int = 6,
    quality_threshold: float = 0.4,
) -> AutonomousResearchResult:
    """Iterative research loop: SEARCH → RETRIEVE → GAP-DETECT → EXPAND → REPORT.

    Args:
        topic: Research question or topic string.
        llm: LLM instance for query expansion, gap detection, report synthesis.
        rag: RAG knowledge base (modified in-place as papers are ingested).
        citation_dag: Persistent citation graph (expanded in-place).
        task_id: Caller-supplied ID; auto-generated if None.
        n_iterations: Maximum loop iterations before stopping.
        n_papers_per_iter: Papers fetched per search query per iteration.
        quality_threshold: Retrieval quality below which KB expansion triggers.

    Returns:
        AutonomousResearchResult with per-iteration stats, final report, DAG info.
    """
    from tools.source_search import multi_source_search, save_pending_sources
    from tools.kb_expansion import score_retrieval_quality
    from tools.auto_report import generate_research_report
    from tools.semantic_memory import store_finding
    from agents.graph import RunState

    if task_id is None:
        task_id = str(uuid4())

    result = AutonomousResearchResult(topic=topic, task_id=task_id)
    all_papers: list = []
    seen_ids: set[str] = set()
    prev_quality = 0.0

    for i in range(n_iterations):
        logger.info("[LOOP iter=%d/%d] topic=%s", i + 1, n_iterations, topic[:60])
        loop_iter = LoopIteration(iteration=i + 1)

        # ── 1. Retrieve from current KB ───────────────────────────────────────
        loop_iter.kb_before = len(rag)
        docs = rag.retrieve(topic, k=8, llm=llm)
        quality = score_retrieval_quality(docs)
        loop_iter.quality_score = quality

        # ── 2. Detect gaps ────────────────────────────────────────────────────
        gaps = detect_knowledge_gaps(topic, docs, llm)
        loop_iter.gaps_detected = gaps

        # ── 3. Search for topic + gap papers ─────────────────────────────────
        search_queries = [topic] + gaps
        iter_papers: list = []
        for sq in search_queries[:3]:
            try:
                found = multi_source_search(sq, n_papers=n_papers_per_iter, llm=llm)
                iter_papers.extend(found)
            except Exception as exc:
                logger.warning("[LOOP iter=%d] Search failed for %r: %s", i + 1, sq, exc)

        new_papers = [p for p in iter_papers if p.id not in seen_ids]
        for p in new_papers:
            seen_ids.add(p.id)
        loop_iter.papers_found = len(new_papers)

        # ── 4. Ingest into KB and save pending list for user review ──────────
        hop0 = [p for p in new_papers if p.hop == 0]
        if hop0:
            try:
                rag.ingest_papers(hop0)
            except Exception as exc:
                logger.warning("[LOOP iter=%d] Ingest failed: %s", i + 1, exc)
        try:
            save_pending_sources(task_id, new_papers)
        except Exception:
            pass
        all_papers.extend(new_papers)

        # ── 5. Expand citation DAG from new OpenAlex seeds ───────────────────
        oa_seeds = [p.id.split("::", 1)[-1] for p in new_papers[:5] if p.source == "openalex"]
        if oa_seeds and citation_dag is not None:
            try:
                citation_dag.expand(oa_seeds, max_hops=2, n_per_hop=3)
            except Exception as exc:
                logger.debug("[LOOP iter=%d] DAG expansion failed: %s", i + 1, exc)

        loop_iter.kb_after = len(rag)
        logger.info(
            "[LOOP iter=%d] quality=%.3f gaps=%d new_papers=%d kb=%d→%d",
            i + 1, quality, len(gaps), len(new_papers),
            loop_iter.kb_before, loop_iter.kb_after,
        )
        result.iterations.append(loop_iter)

        # ── 6. Early stopping on convergence ─────────────────────────────────
        if i > 0 and (quality - prev_quality) < 0.05:
            logger.info(
                "[LOOP] Converged at iteration %d (Δquality=%.3f)", i + 1, quality - prev_quality
            )
            break
        prev_quality = quality

    result.total_papers_ingested = len(all_papers)
    result.citation_dag_nodes = citation_dag.G.number_of_nodes() if citation_dag else 0

    # ── 7. Generate structured research report ────────────────────────────────
    try:
        state = RunState(task=topic, task_id=task_id)
        key_results = [
            f"Iter {it.iteration}: {it.papers_found} papers, quality={it.quality_score:.2f}"
            for it in result.iterations
        ]
        state.artifacts.append({
            "type": "ds",
            "report": {
                "narrative": {
                    "conclusions": (
                        f"Autonomous research completed {len(result.iterations)} iteration(s). "
                        f"{len(all_papers)} papers discovered."
                    ),
                    "key_results": key_results,
                    "methodology": "Iterative multi-source search with LLM gap detection.",
                }
            },
        })
        report = generate_research_report(state, all_papers, llm)
        result.final_report = report
        logger.info("[LOOP] Research report written for task %s", task_id)
    except Exception as exc:
        logger.warning("[LOOP] Report generation failed: %s", exc)

    # ── 8. Store summary in semantic memory for cross-task recall ─────────────
    if result.final_report and result.final_report.executive_summary:
        try:
            store_finding(result.final_report.executive_summary, task_id, topic, rag)
        except Exception as exc:
            logger.debug("[LOOP] Semantic memory store failed: %s", exc)

    return result
