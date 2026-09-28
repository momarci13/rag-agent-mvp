"""Persistent N-hop citation graph built on top of OpenAlex.

Extends the 1-hop traversal in source_search.py into a BFS-expandable
directed graph that persists across sessions and supports PageRank scoring
and novel-paper detection.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from rag.hybrid import LiteHybridRAG

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).parent.parent / "output" / "citation_dag.json"


@dataclass
class CitationNode:
    oa_id: str         # OpenAlex work ID, e.g. "W2741809807"
    title: str = ""
    year: str = ""
    citation_count: int = 0
    authors: list[str] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.authors is None:
            self.authors = []


class CitationDAG:
    """NetworkX DiGraph of citation relationships persisted as JSON.

    Node attrs : title, year, citation_count, authors
    Edge attrs : relation ("cites" | "cited_by"), hop (distance from seed)
    """

    def __init__(self, path: str | Path = _DEFAULT_PATH):
        self.path = Path(path)
        self.G: nx.DiGraph = nx.DiGraph()
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.G = nx.node_link_graph(data)
            logger.debug("[CITATION_DAG] Loaded %d nodes from %s", self.G.number_of_nodes(), self.path)
        except Exception as exc:
            logger.warning("[CITATION_DAG] Failed to load from %s: %s", self.path, exc)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = nx.node_link_data(self.G)
            self.path.write_text(json.dumps(data, default=str), encoding="utf-8")
            logger.debug("[CITATION_DAG] Saved %d nodes to %s", self.G.number_of_nodes(), self.path)
        except Exception as exc:
            logger.warning("[CITATION_DAG] Save failed: %s", exc)

    # ── Graph population ──────────────────────────────────────────────────────

    def _add_node(self, oa_id: str, **attrs) -> None:
        if not self.G.has_node(oa_id):
            self.G.add_node(oa_id, **attrs)
        else:
            # Update attrs if we now have more info
            for k, v in attrs.items():
                if v and not self.G.nodes[oa_id].get(k):
                    self.G.nodes[oa_id][k] = v

    def _add_work(self, work: dict, hop: int = 0) -> str | None:
        """Register an OpenAlex work dict as a graph node. Returns node ID."""
        raw_id = (work.get("id") or "").rsplit("/", 1)[-1]
        if not raw_id:
            return None
        oa_id = raw_id if raw_id.startswith("W") else f"W{raw_id}"
        self._add_node(
            oa_id,
            title=work.get("title") or "",
            year=str(work.get("year") or ""),
            citation_count=int(work.get("cited_by_count") or 0),
            authors=[a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or [])[:5]],
            hop=hop,
        )
        return oa_id

    def expand(
        self,
        seed_ids: list[str],
        max_hops: int = 2,
        n_per_hop: int = 5,
    ) -> list[str]:
        """BFS citation traversal up to max_hops from each seed ID.

        seed_ids should be OpenAlex IDs (with or without "W" prefix).
        Returns list of new node IDs added in this expansion.
        """
        from tools import openalex

        new_ids: list[str] = []
        visited: set[str] = set(self.G.nodes)
        frontier = [(sid if sid.startswith("W") else f"W{sid}", 0) for sid in seed_ids]

        while frontier:
            current_id, hop = frontier.pop(0)
            if hop >= max_hops:
                continue

            for fetcher, relation in [
                (openalex.get_refs, "cites"),
                (openalex.get_cites, "cited_by"),
            ]:
                try:
                    works = fetcher(current_id, limit=n_per_hop)
                except Exception as exc:
                    logger.debug("[CITATION_DAG] %s(%s) failed: %s", fetcher.__name__, current_id, exc)
                    continue

                for work in works:
                    nid = self._add_work(work, hop=hop + 1)
                    if nid is None:
                        continue
                    # Add edge
                    if relation == "cites":
                        self.G.add_edge(current_id, nid, relation=relation, hop=hop + 1)
                    else:
                        self.G.add_edge(nid, current_id, relation=relation, hop=hop + 1)

                    if nid not in visited:
                        visited.add(nid)
                        new_ids.append(nid)
                        if hop + 1 < max_hops:
                            frontier.append((nid, hop + 1))

        logger.info(
            "[CITATION_DAG] Expanded from %d seeds → %d new nodes (max_hops=%d)",
            len(seed_ids), len(new_ids), max_hops,
        )
        self.save()
        return new_ids

    # ── Analytics ────────────────────────────────────────────────────────────

    def pagerank_scores(self, alpha: float = 0.85) -> dict[str, float]:
        """PageRank scores for all nodes. Higher = more cited within the DAG."""
        if self.G.number_of_nodes() == 0:
            return {}
        try:
            return nx.pagerank(self.G, alpha=alpha)
        except Exception as exc:
            logger.warning("[CITATION_DAG] PageRank failed: %s", exc)
            return {}

    def novel_papers(self, rag: "LiteHybridRAG", distance_threshold: float = 0.5) -> list[str]:
        """Return DAG node IDs not yet well-represented in the RAG KB.

        A node is "novel" if no existing RAG chunk has a dense score ≥
        (1 - distance_threshold) against the node's title query.
        """
        novel: list[str] = []
        if not rag._ids:
            return list(self.G.nodes)[:20]

        for nid, attrs in self.G.nodes(data=True):
            title = (attrs.get("title") or "").strip()
            if not title:
                continue
            docs = rag.retrieve(title, k=1)
            if not docs or docs[0].get("score", 0) < (1.0 - distance_threshold):
                novel.append(nid)
        return novel

    def top_by_pagerank(self, n: int = 10) -> list[tuple[str, float]]:
        """Return top-n nodes by PageRank score."""
        scores = self.pagerank_scores()
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def summary(self) -> str:
        return (
            f"CitationDAG: {self.G.number_of_nodes()} papers, "
            f"{self.G.number_of_edges()} citation edges"
        )
