"""
RAG integration: Orchestrates intent classification, query generation, and tiered retrieval.
Provides a unified interface for robust instruction handling.
"""

import logging
from pathlib import Path
from typing import Optional

from agents.intent_classifier import IntentClassifier, IntentClassification
from rag.query_generator import QueryGenerator, GeneratedQueries
from rag.tiered_retrieval import TieredRetrieval, RetrievalResult
from rag.retrieval_diagnostics import RetrievalDiagnostics
from rag.hybrid import LiteHybridRAG

logger = logging.getLogger(__name__)


class RobustRAGPipeline:
    """
    End-to-end RAG pipeline with robustness to sloppy instructions.

    Flow:
    1. Intent Classification: Parse sloppy instruction → structured intent
    2. Query Generation: Intent → multiple canonical query forms
    3. Tiered Retrieval: Retrieve with automatic fallback strategies
    4. Diagnostics: Log failures for offline analysis
    """

    def __init__(
        self,
        base_rag: LiteHybridRAG,
        domain_concepts_path: str = "data/domain_concepts.yaml",
        output_dir: str = "output",
    ):
        """Initialize robust RAG pipeline."""
        self.base_rag = base_rag
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.query_generator = QueryGenerator()
        self.tiered_retrieval = TieredRetrieval(
            base_rag=base_rag,
            domain_concepts_path=domain_concepts_path,
        )
        self.diagnostics = RetrievalDiagnostics(
            log_path=str(self.output_dir / "retrieval_failures.jsonl")
        )

        logger.info("[RobustRAG] Pipeline initialized")

    def retrieve(
        self,
        instruction: str,
        k: int = 6,
        llm=None,
        task_id: str = "",
    ) -> dict:
        """
        End-to-end retrieval from raw instruction to documents.

        Args:
            instruction: Raw user instruction (can be sloppy)
            k: Number of documents to retrieve
            llm: Optional LLM for adaptive strategies
            task_id: Task ID for diagnostics tracking

        Returns:
            Dict with:
                - documents: Retrieved documents
                - intent: Classified intent
                - queries: Generated query forms
                - retrieval_quality: Mean score
                - tier_used: Which tier succeeded
                - needs_clarification: Whether user confirmation is needed
        """

        # 1. Classify instruction intent
        intent = self.intent_classifier.classify(instruction)
        logger.info(
            f"[Intent] domain={intent.primary_domain}, "
            f"task={intent.task_intent}, confidence={intent.confidence:.0%}"
        )

        # 2. Generate query forms
        queries = self.query_generator.generate(
            primary_domain=intent.primary_domain,
            task_intent=intent.task_intent,
            primary_keywords=intent.primary_keywords,
            entity_names=intent.entity_names,
            original_instruction=instruction,
        )
        logger.info(f"[Queries] Generated {len(queries.all_queries)} query variants")

        # 3. Tiered retrieval
        retrieval_result = self.tiered_retrieval.retrieve(
            queries=queries.all_queries,
            k=k,
            llm=llm,
        )

        # 4. Log diagnostics if retrieval quality is low
        if retrieval_result.quality < 0.25:
            failure_type = (
                "no_docs" if len(retrieval_result.documents) == 0
                else "low_score"
            )
            self.diagnostics.log_failure(
                query=queries.primary_query,
                tier_reached=retrieval_result.tier_used,
                mean_score=retrieval_result.mean_score,
                num_docs_found=len(retrieval_result.documents),
                failure_type=failure_type,
                task_id=task_id,
                domain=intent.primary_domain,
                keywords=intent.primary_keywords,
                notes=f"Confidence: {intent.confidence:.1%}",
            )

        return {
            "documents": retrieval_result.documents,
            "scores": retrieval_result.scores,
            "intent": intent,
            "queries": queries,
            "retrieval_quality": retrieval_result.mean_score,
            "tier_used": retrieval_result.tier_used,
            "needs_clarification": intent.clarification_needed,
            "clarification_suggestion": intent.clarification_suggestion,
            "timing_ms": retrieval_result.timing_ms,
        }

    def get_user_confirmation_prompt(self, intent: IntentClassification) -> str:
        """Generate a user-facing prompt to confirm detected intent."""
        lines = [
            "**Please confirm the task type before proceeding:**\n",
            f"📋 Domain: `{intent.primary_domain}`",
            f"🎯 Task: `{intent.task_intent}`",
            f"📊 Confidence: {intent.confidence:.0%}\n",
        ]

        if intent.entity_names:
            lines.append(f"🏷️  Entities detected: {', '.join(intent.entity_names)}\n")

        if intent.primary_keywords:
            lines.append(f"🔑 Key concepts: {', '.join(intent.primary_keywords)}\n")

        if intent.clarification_needed:
            lines.append(f"⚠️  {intent.clarification_suggestion}\n")

        lines.extend([
            "**Proceed with this interpretation?** (yes/no/revise)",
        ])

        return "\n".join(lines)

    def get_diagnostics_summary(self) -> str:
        """Get human-readable diagnostics of retrieval patterns."""
        return self.diagnostics.get_failure_summary()
