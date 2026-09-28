"""Multi-source academic paper search with citation-graph traversal.

Sources:
  - arXiv (free, no key)
  - OpenAlex (free, no key, 250M works, has citation graph)
  - Semantic Scholar (free, no key, CS/finance/ML coverage)

Usage:
    results = multi_source_search("volatility clustering GARCH", n_papers=5, llm=llm)
    save_pending_sources(task_id, results)
    ...
    approved = load_pending_sources(task_id)
    rag.ingest_papers([r for r in approved if r.id in approved_ids])
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from math import log
from pathlib import Path
from typing import Any

import requests

TASKS_DIR = Path(__file__).parent.parent / "output" / "tasks"
_SS_API = "https://api.semanticscholar.org/graph/v1/paper/search"
_SS_FIELDS = "title,authors,year,abstract,citationCount,externalIds,openAccessPdf,url"


# ── Unified result type ───────────────────────────────────────────────────────

@dataclass
class SourceResult:
    """Unified paper record across all sources. Compatible with rag.ingest_papers()."""
    id: str                      # "{source}::{native_id}"
    source: str                  # "arxiv" | "openalex" | "semantic_scholar"
    title: str
    authors: list[str] = field(default_factory=list)
    year: str = ""
    abstract: str = ""
    url: str = ""
    citation_count: int = 0
    relevance_score: float = 0.0
    hop: int = 0                 # 0=direct search hit, 1=via citation traversal

    # ── rag.ingest_papers() interface ────────────────────────────────────────
    @property
    def arxiv_id(self) -> str:
        return self.id.split("::", 1)[-1]

    @property
    def published(self) -> str:
        return self.year

    @property
    def summary(self) -> str:
        return self.abstract

    def to_markdown(self) -> str:
        author_str = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_str += " et al."
        source_label = {"arxiv": "arXiv", "openalex": "OpenAlex",
                        "semantic_scholar": "Semantic Scholar"}.get(self.source, self.source)
        hop_label = " [via citation graph]" if self.hop > 0 else ""
        return (
            f"# {self.title}\n\n"
            f"**Authors:** {author_str}  \n"
            f"**Year:** {self.year}  \n"
            f"**Source:** {source_label}{hop_label}  \n"
            f"**Citations:** {self.citation_count}  \n"
            f"**URL:** {self.url}\n\n"
            f"## Abstract\n\n{self.abstract}\n\n"
            f"---\n*Retrieved via {source_label} multi-source search.*\n"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SourceResult":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ── Per-source adapters ───────────────────────────────────────────────────────

def _search_arxiv(keywords: list[str], n: int) -> list[SourceResult]:
    from tools.scholar import search_arxiv, DEFAULT_SEARCH_CATEGORIES
    results = []
    for query in [" ".join(keywords[:5]), " ".join(keywords[:3])]:
        papers = search_arxiv(query=query, n=n, category=DEFAULT_SEARCH_CATEGORIES)
        if papers:
            for i, p in enumerate(papers):
                results.append(SourceResult(
                    id=f"arxiv::{p.arxiv_id}",
                    source="arxiv",
                    title=p.title,
                    authors=p.authors,
                    year=p.published[:4] if p.published else "",
                    abstract=p.summary,
                    url=p.url,
                    citation_count=0,
                    relevance_score=1.0 - i * 0.05,
                ))
            break
    return results


def _search_openalex(keywords: list[str], n: int) -> list[SourceResult]:
    from tools import openalex
    query = " ".join(keywords[:5])
    try:
        works = openalex.search(query, limit=n, work_type="article")
    except Exception as e:
        print(f"[SOURCE] OpenAlex search failed: {e}")
        return []
    results = []
    for i, w in enumerate(works):
        oa_id = (w.get("id") or "").rsplit("/", 1)[-1]
        if not oa_id or not w.get("title"):
            continue
        doi = w.get("doi") or ""
        url = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{oa_id}"
        results.append(SourceResult(
            id=f"openalex::{oa_id}",
            source="openalex",
            title=w.get("title") or "",
            authors=w.get("authors") or [],
            year=str(w.get("year") or ""),
            abstract="",  # OpenAlex doesn't return abstract in search results
            url=url,
            citation_count=w.get("cited_by_count") or 0,
            relevance_score=1.0 - i * 0.05,
        ))
    return results


def _search_semantic_scholar(keywords: list[str], n: int) -> list[SourceResult]:
    query = " ".join(keywords[:5])
    params = {"query": query, "fields": _SS_FIELDS, "limit": n}
    try:
        resp = requests.get(_SS_API, params=params, timeout=12)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[SOURCE] Semantic Scholar search failed: {e}")
        return []

    results = []
    for i, item in enumerate((data.get("data") or [])[:n]):
        pid = item.get("paperId") or ""
        title = item.get("title") or ""
        if not pid or not title:
            continue
        authors = [a.get("name") or "" for a in (item.get("authors") or [])[:6] if a.get("name")]
        year = str(item.get("year") or "")
        abstract = item.get("abstract") or ""
        ext_ids = item.get("externalIds") or {}
        if ext_ids.get("ArXiv"):
            url = f"https://arxiv.org/abs/{ext_ids['ArXiv']}"
        elif ext_ids.get("DOI"):
            url = f"https://doi.org/{ext_ids['DOI']}"
        else:
            url = f"https://semanticscholar.org/paper/{pid}"
        results.append(SourceResult(
            id=f"semantic_scholar::{pid}",
            source="semantic_scholar",
            title=title,
            authors=authors,
            year=year,
            abstract=abstract,
            url=url,
            citation_count=item.get("citationCount") or 0,
            relevance_score=1.0 - i * 0.05,
        ))
    return results


# ── Citation-graph hop ────────────────────────────────────────────────────────

def _citation_hop(seeds: list[SourceResult], n_per_seed: int = 5) -> list[SourceResult]:
    """Fetch 1-hop references/citations for openalex seed papers."""
    from tools import openalex
    hop_results: list[SourceResult] = []
    for seed in seeds:
        if seed.source != "openalex":
            continue
        oa_id = seed.id.split("::", 1)[-1]
        for fetcher, label in [(openalex.get_refs, "refs"), (openalex.get_cites, "cites")]:
            try:
                works = fetcher(oa_id, limit=n_per_seed)
                for w in works:
                    hop_id = (w.get("id") or "").rsplit("/", 1)[-1]
                    if not hop_id or not w.get("title"):
                        continue
                    doi = w.get("doi") or ""
                    url = f"https://doi.org/{doi}" if doi else f"https://openalex.org/{hop_id}"
                    hop_results.append(SourceResult(
                        id=f"openalex::{hop_id}",
                        source="openalex",
                        title=w.get("title") or "",
                        authors=w.get("authors") or [],
                        year=str(w.get("year") or ""),
                        abstract="",
                        url=url,
                        citation_count=w.get("cited_by_count") or 0,
                        relevance_score=0.6,
                        hop=1,
                    ))
            except Exception:
                pass
    return hop_results


# ── Deduplication & scoring ───────────────────────────────────────────────────

def _normalise_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def _deduplicate(results: list[SourceResult]) -> list[SourceResult]:
    seen: dict[str, SourceResult] = {}
    for r in results:
        key = _normalise_title(r.title)
        if not key:
            continue
        if key not in seen:
            seen[key] = r
        else:
            # Keep whichever has more info (lower hop, higher citations)
            existing = seen[key]
            if r.hop < existing.hop or r.citation_count > existing.citation_count:
                seen[key] = r
    return list(seen.values())


def _score(r: SourceResult) -> float:
    citation_boost = log(1 + r.citation_count) / 10.0
    hop_factor = 1.2 if r.hop == 0 else 1.0
    return (r.relevance_score + citation_boost) * hop_factor


# ── Public API ────────────────────────────────────────────────────────────────

def multi_source_search(
    task_description: str,
    n_papers: int = 5,
    llm=None,
) -> list[SourceResult]:
    """Search arXiv, OpenAlex, and Semantic Scholar; traverse 1-hop citations.

    Returns up to n_papers * 3 ranked SourceResult objects.
    hop=0 papers were found directly; hop=1 were found via citation graph.
    """
    from tools.scholar import extract_keywords
    keywords = extract_keywords(task_description, llm=llm)
    if not keywords:
        keywords = task_description.lower().split()[:5]

    print(f"[SOURCE] Searching with keywords: {keywords}")

    # Search all three sources
    arxiv_results = _search_arxiv(keywords, n_papers)
    oa_results = _search_openalex(keywords, n_papers)
    ss_results = _search_semantic_scholar(keywords, n_papers)

    print(f"[SOURCE] Found {len(arxiv_results)} arXiv, {len(oa_results)} OpenAlex, "
          f"{len(ss_results)} Semantic Scholar")

    all_results = arxiv_results + oa_results + ss_results
    deduped = _deduplicate(all_results)

    # Citation hop on top-3 OpenAlex seeds
    oa_seeds = sorted([r for r in deduped if r.source == "openalex"],
                      key=_score, reverse=True)[:3]
    if oa_seeds:
        hop_results = _citation_hop(oa_seeds, n_per_seed=4)
        print(f"[SOURCE] Citation hop added {len(hop_results)} candidates")
        all_results = deduped + hop_results
        deduped = _deduplicate(all_results)

    # Score and sort
    ranked = sorted(deduped, key=_score, reverse=True)
    limit = n_papers * 3
    final = ranked[:limit]
    print(f"[SOURCE] Returning {len(final)} ranked results (hop=0: "
          f"{sum(1 for r in final if r.hop == 0)}, "
          f"hop=1: {sum(1 for r in final if r.hop == 1)})")

    # Persist citation edges in the persistent DAG (best-effort)
    try:
        from tools.citation_dag import CitationDAG
        oa_seeds = [r.id.split("::", 1)[-1] for r in final[:5] if r.source == "openalex"]
        if oa_seeds:
            dag = CitationDAG()
            dag.expand(oa_seeds, max_hops=2, n_per_hop=3)
    except Exception:
        pass

    return final


# ── Pending-source persistence ─────────────────────────────────────────────────

def save_pending_sources(task_id: str, sources: list[SourceResult]) -> None:
    task_dir = TASKS_DIR / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in sources]
    (task_dir / "pending_sources.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def load_pending_sources(task_id: str) -> list[SourceResult]:
    path = TASKS_DIR / task_id / "pending_sources.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [SourceResult.from_dict(d) for d in data]
    except Exception:
        return []


def clear_pending_sources(task_id: str) -> None:
    path = TASKS_DIR / task_id / "pending_sources.json"
    if path.exists():
        path.unlink()
