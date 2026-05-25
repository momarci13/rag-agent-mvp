"""
Tiered retrieval: Multi-strategy fallback retrieval when primary queries fail.
Improves robustness on sloppy instructions by trying alternative search strategies.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from a retrieval attempt."""

    documents: list[dict] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    tier_used: int = 1
    mean_score: float = 0.0
    threshold_met: bool = False
    timing_ms: float = 0.0
    queries_tried: list[str] = field(default_factory=list)

    @property
    def quality(self) -> float:
        """Return mean score as quality metric."""
        return self.mean_score


class TieredRetrieval:
    """
    Multi-strategy retrieval with automatic fallback.

    Tries progressively different retrieval strategies when quality is low:
    - Tier 1: Fast hybrid search (dense + BM25)
    - Tier 2: Title/metadata only search
    - Tier 3: Concept expansion using domain hierarchy
    - Tier 4: KB expansion (existing slow mechanism)
    """

    # Configuration (should match rag config)
    TIER_CONFIG = {
        1: {
            "name": "Hybrid (dense + BM25)",
            "timeout_s": 2.0,
            "threshold": 0.35,
            "search_full_text": True,
            "use_expansion": True,
        },
        2: {
            "name": "Title & Metadata Only",
            "timeout_s": 1.0,
            "threshold": 0.25,
            "search_full_text": False,
            "use_expansion": False,
        },
        3: {
            "name": "Concept Expansion",
            "timeout_s": 2.0,
            "threshold": 0.20,
            "search_full_text": True,
            "use_expansion": True,
            "use_concept_hierarchy": True,
        },
        4: {
            "name": "KB Expansion",
            "timeout_s": 30.0,
            "threshold": 0.15,
            "search_full_text": True,
            "trigger_kb_expansion": True,
        },
    }

    def __init__(self, base_rag, domain_concepts_path: str = None):
        """
        Initialize tiered retrieval wrapper.

        Args:
            base_rag: Existing LiteHybridRAG instance
            domain_concepts_path: Path to domain_concepts.yaml
        """
        self.base_rag = base_rag
        self.domain_concepts = {}

        if domain_concepts_path:
            try:
                with open(domain_concepts_path, 'r') as f:
                    self.domain_concepts = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning(f"Failed to load domain concepts: {e}")

    def retrieve(
        self,
        queries: list[str],
        k: int = 6,
        llm=None,
    ) -> RetrievalResult:
        """
        Retrieve documents using tiered fallback strategy.

        Args:
            queries: List of query strings (from QueryGenerator)
            k: Number of documents to retrieve per tier
            llm: Optional LLM for adaptive strategies

        Returns:
            RetrievalResult with documents and quality metrics
        """

        start_time = time.time()
        all_queries_tried = []

        # Tier 1: Fast hybrid search
        logger.info(f"[TIER1] Starting hybrid search with {len(queries)} queries")
        result = self._try_tier(
            tier=1,
            queries=queries,
            k=k,
            llm=llm,
        )
        result.queries_tried = queries
        all_queries_tried.extend(queries)

        if result.threshold_met:
            result.timing_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[TIER1 SUCCESS] Found {len(result.documents)} docs, "
                f"mean_score={result.mean_score:.3f}, {result.timing_ms:.0f}ms"
            )
            return result

        # Tier 2: Title & metadata only
        logger.info("[TIER2] Tier 1 failed, trying metadata-only search")
        result = self._try_tier(
            tier=2,
            queries=queries,
            k=k,
            llm=llm,
        )
        result.queries_tried = all_queries_tried + queries

        if result.threshold_met:
            result.timing_ms = (time.time() - start_time) * 1000
            logger.info(
                f"[TIER2 SUCCESS] Found {len(result.documents)} docs, "
                f"mean_score={result.mean_score:.3f}, {result.timing_ms:.0f}ms"
            )
            return result

        # Tier 3: Concept expansion
        expanded_queries = self._expand_queries_with_concepts(queries)
        if expanded_queries != queries:
            logger.info(
                f"[TIER3] Tier 2 failed, expanding queries with domain concepts: "
                f"{expanded_queries}"
            )
            result = self._try_tier(
                tier=3,
                queries=expanded_queries,
                k=k,
                llm=llm,
            )
            result.queries_tried = all_queries_tried + expanded_queries

            if result.threshold_met:
                result.timing_ms = (time.time() - start_time) * 1000
                logger.info(
                    f"[TIER3 SUCCESS] Found {len(result.documents)} docs, "
                    f"mean_score={result.mean_score:.3f}, {result.timing_ms:.0f}ms"
                )
                return result
        else:
            logger.info("[TIER3] No concept expansion available, skipping")

        # Tier 4: KB expansion (slow but comprehensive)
        logger.info("[TIER4] All fast tiers failed, triggering KB expansion")
        result = self._try_tier_kb_expansion(
            queries=queries,
            k=k,
            llm=llm,
        )

        result.timing_ms = (time.time() - start_time) * 1000
        logger.warning(
            f"[TIER4 FALLBACK] KB expansion triggered, "
            f"found {len(result.documents)} docs, {result.timing_ms:.0f}ms"
        )

        return result

    def _try_tier(
        self,
        tier: int,
        queries: list[str],
        k: int,
        llm=None,
    ) -> RetrievalResult:
        """Try a single tier of retrieval."""

        config = self.TIER_CONFIG[tier]
        tier_name = config["name"]

        start = time.time()
        timeout = config["timeout_s"]
        threshold = config["threshold"]

        try:
            # Collect results from all queries in this tier
            all_docs = []
            all_scores = []

            for query in queries:
                if time.time() - start > timeout:
                    logger.warning(f"[{tier_name}] Timeout exceeded ({timeout}s)")
                    break

                try:
                    if tier == 1:
                        # Tier 1: Standard hybrid search with expansion
                        docs, scores = self.base_rag.retrieve(query, k=k, llm=llm)
                    elif tier == 2:
                        # Tier 2: Title-only search (metadata fallback)
                        docs, scores = self._retrieve_metadata_only(query, k=k)
                    elif tier == 3:
                        # Tier 3: Full text search without expansion
                        docs, scores = self.base_rag.retrieve(query, k=k, llm=None)
                    else:
                        continue

                    all_docs.extend(docs)
                    all_scores.extend(scores)

                except Exception as e:
                    logger.debug(f"[{tier_name}] Query failed: {e}")
                    continue

            # Deduplicate docs by ID, keep highest score
            doc_map = {}
            for doc, score in zip(all_docs, all_scores):
                doc_id = doc.get("id") or doc.get("source", "")
                if doc_id not in doc_map or score > doc_map[doc_id][1]:
                    doc_map[doc_id] = (doc, score)

            final_docs = [d[0] for d in doc_map.values()]
            final_scores = [d[1] for d in doc_map.values()]

            # Sort by score descending, take top k
            sorted_pairs = sorted(zip(final_docs, final_scores), key=lambda x: x[1], reverse=True)
            final_docs = [p[0] for p in sorted_pairs[:k]]
            final_scores = [p[1] for p in sorted_pairs[:k]]

            mean_score = sum(final_scores) / len(final_scores) if final_scores else 0.0

            return RetrievalResult(
                documents=final_docs,
                scores=final_scores,
                tier_used=tier,
                mean_score=mean_score,
                threshold_met=mean_score >= threshold,
                timing_ms=(time.time() - start) * 1000,
            )

        except Exception as e:
            logger.error(f"[{tier_name}] Tier failed: {e}")
            return RetrievalResult(tier_used=tier, threshold_met=False)

    def _retrieve_metadata_only(self, query: str, k: int) -> tuple[list, list]:
        """
        Retrieve by searching only document metadata (titles, keywords).
        Fallback for when embeddings don't match.
        """
        if not hasattr(self.base_rag, "collection") or not self.base_rag.collection:
            return [], []

        try:
            # Query Chroma's metadata: titles and keywords
            results = self.base_rag.collection.query(
                query_embeddings=None,  # Don't use embeddings
                where_document={"$contains": query},  # Full-text search in metadata
                n_results=k,
            )

            if not results or not results.get("ids"):
                return [], []

            docs = []
            scores = []

            for doc_id, metadata, doc_text in zip(
                results.get("ids", []),
                results.get("metadatas", []),
                results.get("documents", []),
            ):
                # Simple TF-IDF-like score: how many query terms match metadata
                query_terms = set(query.lower().split())
                metadata_text = " ".join([str(v) for v in metadata.values()]).lower()
                matches = sum(1 for term in query_terms if term in metadata_text)
                score = matches / len(query_terms) if query_terms else 0.0

                docs.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": metadata,
                })
                scores.append(score)

            return docs, scores

        except Exception as e:
            logger.debug(f"Metadata search failed: {e}")
            return [], []

    def _expand_queries_with_concepts(self, queries: list[str]) -> list[str]:
        """Expand queries using domain concept hierarchy."""
        if not self.domain_concepts:
            return queries

        expanded = []
        for query in queries:
            expanded.append(query)  # Always include original

            # Try to find concepts in query and add related terms
            for domain, concepts in self.domain_concepts.items():
                if isinstance(concepts, dict):
                    for concept_name, concept_data in concepts.items():
                        if isinstance(concept_data, dict):
                            canonical = concept_data.get("canonical", "")
                            synonyms = concept_data.get("synonyms", [])

                            if canonical and canonical.lower() in query.lower():
                                # Found a known concept, add synonyms
                                for syn in synonyms[:2]:  # Top 2 synonyms
                                    expanded.append(query.replace(canonical, syn))

        return list(set(expanded))  # Remove duplicates

    def _try_tier_kb_expansion(
        self,
        queries: list[str],
        k: int,
        llm=None,
    ) -> RetrievalResult:
        """
        Trigger KB expansion using existing mechanism.
        This is the slow path but most comprehensive.
        """
        try:
            if hasattr(self.base_rag, "expand_kb_on_weak_retrieval"):
                # Call existing expansion logic
                self.base_rag.expand_kb_on_weak_retrieval(
                    query=" ".join(queries),
                    retrieval_quality=0.0,
                    llm=llm,
                )

                # Re-retrieve after expansion
                docs, scores = self.base_rag.retrieve(queries[0], k=k, llm=llm)
                mean_score = sum(scores) / len(scores) if scores else 0.0

                return RetrievalResult(
                    documents=docs,
                    scores=scores,
                    tier_used=4,
                    mean_score=mean_score,
                    threshold_met=mean_score >= 0.15,
                )
        except Exception as e:
            logger.error(f"KB expansion failed: {e}")

        return RetrievalResult(tier_used=4, threshold_met=False)
