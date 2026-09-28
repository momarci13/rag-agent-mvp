"""Persistent research knowledge graph (NetworkX + JSON).

Step 5: Connects concepts, papers, experiments, and findings across sessions.
Ingests completed RunState artifacts automatically after each research run.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

try:
    import networkx as nx
except ImportError as e:
    raise ImportError("networkx is required: pip install networkx") from e

GRAPH_PATH = Path(__file__).parent.parent / "output" / "knowledge_graph.json"

NodeType = Literal["concept", "paper", "experiment", "finding", "task"]


class ResearchKnowledgeGraph:
    """Directed graph connecting research artifacts across sessions."""

    def __init__(self, path: Path = GRAPH_PATH):
        self.path = path
        self.G: nx.DiGraph = nx.DiGraph()
        self._load()

    # ---------- add nodes ----------

    def add_concept(self, name: str, definition: str = "", domain: str = "") -> str:
        nid = f"concept::{name.lower().replace(' ', '_')}"
        if nid not in self.G:
            self.G.add_node(nid, type="concept", name=name, definition=definition,
                            domain=domain, created_at=_now())
        return nid

    def add_paper(self, arxiv_id: str, title: str, authors: list[str],
                  year: str, summary: str = "") -> str:
        nid = f"paper::{arxiv_id}"
        self.G.add_node(nid, type="paper", arxiv_id=arxiv_id, title=title,
                        authors=", ".join(authors[:3]),
                        year=year, summary=summary[:300], created_at=_now())
        return nid

    def add_experiment(self, experiment_id: str, name: str,
                       model_type: str, metrics: dict) -> str:
        nid = f"experiment::{experiment_id}"
        self.G.add_node(nid, type="experiment", experiment_id=experiment_id,
                        name=name, model_type=model_type,
                        metrics=json.dumps(metrics, default=str), created_at=_now())
        return nid

    def add_finding(self, text: str, source_task_id: str, confidence: float = 0.5) -> str:
        nid = f"finding::{str(uuid4())[:8]}"
        self.G.add_node(nid, type="finding", text=text[:500],
                        source_task_id=source_task_id,
                        confidence=confidence, created_at=_now())
        return nid

    def add_task_node(self, task_id: str, task: str, task_type: str,
                      accepted: bool, created_at: str) -> str:
        nid = f"task::{task_id}"
        self.G.add_node(nid, type="task", task_id=task_id, task=task[:200],
                        task_type=task_type, accepted=accepted, created_at=created_at)
        return nid

    def add_edge(self, from_id: str, to_id: str, relation: str, weight: float = 1.0) -> None:
        self.G.add_edge(from_id, to_id, relation=relation, weight=weight)

    # ---------- query ----------

    def find_related(self, node_id: str, depth: int = 2) -> list[dict]:
        if node_id not in self.G:
            return []
        reachable = nx.single_source_shortest_path_length(
            self.G.to_undirected(), node_id, cutoff=depth
        )
        result = []
        for nid, dist in reachable.items():
            if nid == node_id:
                continue
            attrs = dict(self.G.nodes[nid])
            attrs["node_id"] = nid
            attrs["distance"] = dist
            result.append(attrs)
        return sorted(result, key=lambda x: x["distance"])

    def find_by_type(self, node_type: NodeType) -> list[dict]:
        return [
            {"node_id": n, **dict(self.G.nodes[n])}
            for n in self.G.nodes
            if self.G.nodes[n].get("type") == node_type
        ]

    def prior_findings_for(self, topic_keywords: list[str], top_n: int = 5) -> list[dict]:
        """Retrieve findings whose text overlaps with topic keywords."""
        findings = self.find_by_type("finding")
        kw = {k.lower() for k in topic_keywords}

        def relevance(f: dict) -> int:
            text_words = set(f.get("text", "").lower().split())
            return len(text_words & kw)

        scored = sorted(findings, key=relevance, reverse=True)
        return [f for f in scored if relevance(f) > 0][:top_n]

    def summarize(self) -> str:
        counts: dict[str, int] = {}
        for n in self.G.nodes:
            t = self.G.nodes[n].get("type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        lines = [
            f"Knowledge Graph: {self.G.number_of_nodes()} nodes, "
            f"{self.G.number_of_edges()} edges"
        ]
        for t, c in sorted(counts.items()):
            lines.append(f"  {t}: {c}")
        return "\n".join(lines)

    # ---------- auto-ingest ----------

    def ingest_run_state(self, state: Any, papers: list | None = None) -> None:
        """Extract and link nodes from a completed RunState, then persist."""
        created = (
            state.created_at.isoformat()
            if hasattr(state.created_at, "isoformat")
            else str(state.created_at)
        )
        task_nid = self.add_task_node(
            state.task_id, state.task, state.task_type, state.accepted, created
        )

        # Key concepts from task text
        for word in state.task.lower().split():
            clean = word.strip(".,;:!?()")
            if len(clean) > 5 and clean.isalpha():
                cnid = self.add_concept(clean)
                self.add_edge(task_nid, cnid, "mentions")

        # Papers
        if papers:
            for p in papers:
                pnid = self.add_paper(
                    p.arxiv_id, p.title, p.authors,
                    p.published.split("-")[0], p.summary
                )
                self.add_edge(task_nid, pnid, "cites")

        # Findings from accepted artifacts
        if state.accepted and state.artifacts:
            for art in state.artifacts:
                if art.get("type") == "ds":
                    stdout = art["payload"].get("stdout", "").strip()
                    if stdout:
                        fnid = self.add_finding(stdout[:400], state.task_id, confidence=0.7)
                        self.add_edge(task_nid, fnid, "produces")
                elif art.get("type") == "writing":
                    tex = art["payload"].get("tex", "")[:200].strip()
                    if tex:
                        fnid = self.add_finding(tex, state.task_id, confidence=0.6)
                        self.add_edge(task_nid, fnid, "produces")
                elif art.get("type") == "quant":
                    bt = art["payload"].get("backtest") or {}
                    if isinstance(bt, dict) and bt.get("sharpe"):
                        summary = f"Sharpe={bt['sharpe']:.3f} CAGR={bt.get('cagr',0):.3f}"
                        fnid = self.add_finding(summary, state.task_id, confidence=0.8)
                        self.add_edge(task_nid, fnid, "produces")

        self._save()
        print(f"[KG] Graph now has {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")

    # ---------- persistence ----------

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self.G)
        self.path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def _load(self) -> None:
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.G = nx.node_link_graph(data)
            except Exception as e:
                print(f"[KG] Failed to load graph ({e}), starting fresh")
                self.G = nx.DiGraph()


def _now() -> str:
    return datetime.utcnow().isoformat()
