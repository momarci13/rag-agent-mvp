"""Autonomous literature acquisition with persistent deduplication tracking.

Step 1 of the research expansion: automatically search arXiv, track ingested
paper IDs to avoid re-fetching, detect domain from keywords, and ingest into RAG.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tools.scholar import search_arxiv, ArxivPaper, extract_keywords

if TYPE_CHECKING:
    from rag.hybrid import LiteHybridRAG

REGISTRY_PATH = Path(__file__).parent.parent / "output" / "literature" / "registry.json"

_ML_CATS = ["cs.LG", "cs.AI", "stat.ML", "stat.AP", "math.ST"]
_FINANCE_CATS = ["q-fin", "q-fin.PM", "stat.ML", "math.ST"]
_ALL_CATS = ["cs.LG", "q-fin", "stat.ML", "cs.AI", "stat.AP", "math.ST", "q-fin.PM"]

_ML_TOKENS = {
    "neural", "deep", "transformer", "lstm", "gru", "gradient", "classification",
    "regression", "clustering", "forest", "xgboost", "attention", "embedding",
    "learning", "train", "predict", "convolutional", "autoencoder", "diffusion",
}
_FINANCE_TOKENS = {
    "portfolio", "trading", "strategy", "backtest", "sharpe", "volatility",
    "equity", "market", "factor", "alpha", "risk", "hedge", "return", "price",
    "momentum", "arbitrage", "derivative", "option", "futures", "fixed",
}


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"papers": {}, "topics": {}}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, default=str), encoding="utf-8")


def _detect_domain(keywords: list[str]) -> list[str]:
    kw = {k.lower() for k in keywords}
    is_ml = bool(kw & _ML_TOKENS)
    is_fin = bool(kw & _FINANCE_TOKENS)
    if is_fin and is_ml:
        return _ALL_CATS
    if is_ml:
        return _ML_CATS
    if is_fin:
        return _FINANCE_CATS
    return _ALL_CATS


def acquire_literature(
    topic: str,
    n_papers: int = 8,
    skip_known: bool = True,
    rag=None,
) -> tuple[list[ArxivPaper], list[str]]:
    """Search arXiv for topic, track ingested IDs, ingest into RAG if provided.

    Returns (new_papers, skipped_arxiv_ids).
    """
    registry = _load_registry()
    known_ids = set(registry["papers"].keys())

    keywords = extract_keywords(topic, max_keywords=7)
    if not keywords:
        return [], []

    categories = _detect_domain(keywords)
    query = " ".join(keywords[:5])
    print(f"[LITERATURE] Searching '{query}' in {categories}")

    candidates = search_arxiv(query, n=n_papers * 2, category=categories)

    new_papers: list[ArxivPaper] = []
    skipped: list[str] = []
    for paper in candidates:
        if skip_known and paper.arxiv_id in known_ids:
            skipped.append(paper.arxiv_id)
        else:
            new_papers.append(paper)
            registry["papers"][paper.arxiv_id] = {
                "title": paper.title,
                "published": paper.published,
                "topics": [topic[:80]],
                "ingested": False,
            }

    new_papers = new_papers[:n_papers]

    if rag is not None and new_papers:
        chunks = rag.ingest_papers(new_papers)
        for p in new_papers:
            if p.arxiv_id in registry["papers"]:
                registry["papers"][p.arxiv_id]["ingested"] = True
        print(f"[LITERATURE] Ingested {len(new_papers)} papers ({chunks} chunks)")

    registry["topics"][topic[:80]] = {
        "keywords": keywords,
        "categories": categories,
        "new": [p.arxiv_id for p in new_papers],
        "skipped": skipped,
    }
    _save_registry(registry)
    print(f"[LITERATURE] {len(new_papers)} new, {len(skipped)} already in registry")
    return new_papers, skipped


def literature_context(papers: list[ArxivPaper]) -> str:
    """Format paper list as LLM context string."""
    if not papers:
        return ""
    lines = [f"## Retrieved Literature ({len(papers)} papers)\n"]
    for i, p in enumerate(papers, 1):
        authors = ", ".join(p.authors[:2]) + (" et al." if len(p.authors) > 2 else "")
        lines.append(f"{i}. **{p.title}** — {authors} ({p.published[:4]})")
        lines.append(f"   arXiv:{p.arxiv_id}")
        lines.append(f"   {p.summary[:220]}...\n")
    return "\n".join(lines)


def registry_stats() -> dict:
    """Return summary stats of the literature registry."""
    reg = _load_registry()
    papers = reg.get("papers", {})
    return {
        "total_tracked": len(papers),
        "ingested": sum(1 for p in papers.values() if p.get("ingested")),
        "topics_searched": len(reg.get("topics", {})),
    }
