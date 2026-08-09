"""Retrieval metrics and logging utilities for RAG system evaluation.

Tracks precision@k, recall@k, and other IR metrics to evaluate retrieval quality.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class RetrievalMetrics:
    """Track retrieval performance metrics."""

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.output_dir / "retrieval_metrics.json"
        self.queries: list[dict] = []

    def log_query(
        self,
        query: str,
        retrieved_docs: list[dict],
        relevant_doc_ids: set[str] | None = None,
        metadata_filters: dict | None = None,
        query_type: str | None = None,
    ) -> dict:
        """Log a retrieval query and compute metrics.

        Args:
            query: the search query
            retrieved_docs: list of {"id", "text", "meta", "score"} from RAG
            relevant_doc_ids: set of doc IDs that are relevant (for evaluation)
            metadata_filters: any metadata filters applied
            query_type: optional domain or query intent label
        
        Returns:
            dict with computed metrics (precision@k, etc)
        """
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "query_type": query_type,
            "num_retrieved": len(retrieved_docs),
            "metadata_filters": metadata_filters,
            "retrieved_ids": [d["id"] for d in retrieved_docs],
            "retrieved_scores": [d.get("score", 0.0) for d in retrieved_docs],
        }

        # If we have ground truth, compute precision/recall
        if relevant_doc_ids:
            relevant_doc_ids = set(relevant_doc_ids)
            retrieved_ids = set(d["id"] for d in retrieved_docs)
            true_positives = len(retrieved_ids & relevant_doc_ids)
            false_positives = len(retrieved_ids - relevant_doc_ids)
            false_negatives = len(relevant_doc_ids - retrieved_ids)

            precision = (
                true_positives / len(retrieved_ids)
                if retrieved_ids
                else 0.0
            )
            recall = (
                true_positives / len(relevant_doc_ids)
                if relevant_doc_ids
                else 0.0
            )
            f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            metrics.update({
                "relevant_docs_count": len(relevant_doc_ids),
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            })

        self.queries.append(metrics)
        return metrics

    def compute_averages(self) -> dict:
        """Compute average metrics across all logged queries."""
        if not self.queries:
            return {}

        # Queries with relevance judgments
        evaluated = [q for q in self.queries if "precision" in q]
        if not evaluated:
            return {"num_queries": len(self.queries), "note": "no relevance judgments"}

        avg_precision = sum(q["precision"] for q in evaluated) / len(evaluated)
        avg_recall = sum(q["recall"] for q in evaluated) / len(evaluated)
        avg_f1 = sum(q["f1"] for q in evaluated) / len(evaluated)
        avg_retrieved = sum(q["num_retrieved"] for q in self.queries) / len(
            self.queries
        )

        return {
            "num_queries": len(self.queries),
            "num_evaluated": len(evaluated),
            "avg_precision": avg_precision,
            "avg_recall": avg_recall,
            "avg_f1": avg_f1,
            "avg_retrieved_count": avg_retrieved,
        }

    def save(self, filename: str | None = None) -> Path:
        """Save metrics to JSON and return the written path.

        ``filename`` is retained for compatibility with the original Phase 1
        smoke test; callers that omit it use ``retrieval_metrics.json``.
        """
        target = self.output_dir / filename if filename else self.metrics_file
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "queries": self.queries,
            "summary": self.compute_averages(),
        }
        with open(target, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"Metrics saved to {target}")
        return target

    def summary(self) -> dict:
        """Alias for compute_averages to support legacy metric interfaces."""
        return self.compute_averages()

    def print_summary(self) -> None:
        """Print summary of metrics to console."""
        summary = self.compute_averages()
        print("\n--- Retrieval Metrics Summary ---")
        print(f"Queries logged: {summary.get('num_queries', 0)}")
        if "avg_precision" in summary:
            print(f"Average Precision: {summary['avg_precision']:.3f}")
            print(f"Average Recall: {summary['avg_recall']:.3f}")
            print(f"Average F1: {summary['avg_f1']:.3f}")
        print(f"Avg docs retrieved: {summary.get('avg_retrieved_count', 0):.1f}")
