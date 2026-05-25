"""
Query generator: Converts structured intent into multiple canonical query forms.
Produces diverse query variants for better retrieval coverage.
"""

from dataclasses import dataclass, field
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GeneratedQueries:
    """Multiple query forms generated from a single intent."""

    primary_query: str  # Core concept in canonical domain terminology
    alternative_queries: list[str] = field(default_factory=list)  # 2-3 reformulations
    entity_queries: list[str] = field(default_factory=list)  # Entity-specific searches
    relationship_queries: list[str] = field(default_factory=list)  # Concept relationship searches
    all_queries: list[str] = field(default_factory=list)  # Flattened list for retrieval

    def __post_init__(self):
        """Flatten all queries for convenient iteration."""
        self.all_queries = (
            [self.primary_query] +
            self.alternative_queries +
            self.entity_queries +
            self.relationship_queries
        )


class QueryGenerator:
    """Generates multiple query forms from classified intent."""

    # Domain-specific query templates
    DOMAIN_TEMPLATES = {
        "finance": {
            "comparative_analysis": "comparison of {concept} in {domain}",
            "explanation": "how {concept} affects {domain}",
            "strategy_design": "{concept} trading strategy",
            "prediction": "{concept} forecasting model",
            "optimization": "optimal {concept} for portfolio",
        },
        "statistics": {
            "comparative_analysis": "differences in {concept}",
            "explanation": "interpretation of {concept}",
            "hypothesis_formation": "hypothesis testing with {concept}",
            "data_analysis": "statistical analysis of {concept}",
            "optimization": "optimal {concept} estimation",
        },
        "ml": {
            "comparative_analysis": "comparison of {concept} models",
            "explanation": "understanding {concept} in machine learning",
            "strategy_design": "{concept} architecture",
            "prediction": "{concept} for prediction",
            "optimization": "hyperparameter tuning for {concept}",
        },
        "writing": {
            "literature_review": "literature review of {concept}",
            "explanation": "research on {concept}",
            "synthesis": "synthesis of {concept}",
            "hypothesis_formation": "research hypothesis about {concept}",
        },
    }

    # Synonym mappings for common concepts (augments domain_concepts.yaml)
    CONCEPT_SYNONYMS = {
        "volatility": ["vol", "risk", "variance", "dispersion"],
        "momentum": ["trend", "runup", "acceleration"],
        "return": ["performance", "profit", "yield"],
        "risk": ["downside", "loss", "uncertainty"],
        "correlation": ["dependence", "co-movement", "relationship"],
        "prediction": ["forecast", "projection", "estimate"],
        "analysis": ["examination", "investigation", "study"],
        "strategy": ["approach", "method", "system", "algorithm"],
        "optimization": ["tuning", "improvement", "maximization"],
    }

    # Relationship patterns for multi-concept queries
    RELATIONSHIP_PATTERNS = {
        "trade_off": "{concept_a} and {concept_b} trade-off",
        "interaction": "interaction between {concept_a} and {concept_b}",
        "causality": "does {concept_a} cause {concept_b}",
        "correlation": "{concept_a} correlation with {concept_b}",
        "vs": "{concept_a} versus {concept_b}",
    }

    def __init__(self, domain_concepts: dict = None):
        """
        Initialize with optional domain concepts mapping.

        Args:
            domain_concepts: YAML-loaded domain_concepts dict
        """
        self.domain_concepts = domain_concepts or {}

    def generate(
        self,
        primary_domain: str,
        task_intent: str,
        primary_keywords: list[str],
        entity_names: list[str],
        original_instruction: str,
    ) -> GeneratedQueries:
        """
        Generate multiple query forms from intent components.

        Args:
            primary_domain: Detected domain (finance, ml, etc.)
            task_intent: Task type (comparative_analysis, etc.)
            primary_keywords: Main concepts mentioned
            entity_names: Specific entities (stock symbols, etc.)
            original_instruction: Original user instruction for reference

        Returns:
            GeneratedQueries with primary + alternative + entity + relationship queries
        """

        # 1. Generate primary query (canonical form)
        if primary_keywords:
            primary_concept = primary_keywords[0]
        else:
            primary_concept = self._extract_main_concept(original_instruction)

        template = self.DOMAIN_TEMPLATES.get(primary_domain, {}).get(task_intent)
        if template:
            primary_query = template.format(concept=primary_concept, domain=primary_domain)
        else:
            primary_query = f"{task_intent}: {primary_concept}"

        # 2. Generate alternative queries (reformulations)
        alternative_queries = []
        for keyword in primary_keywords[:2]:  # Top 2 keywords
            alt1 = self._create_broad_form(keyword, primary_domain)
            alt2 = self._create_narrow_form(keyword, primary_domain, task_intent)
            alternative_queries.extend([alt1, alt2])

        # 3. Generate entity queries (specific names/symbols)
        entity_queries = []
        for entity in entity_names[:3]:  # Top 3 entities
            entity_queries.append(f"{entity} {primary_keywords[0] if primary_keywords else 'analysis'}")

        # 4. Generate relationship queries (multi-concept)
        relationship_queries = []
        if len(primary_keywords) >= 2:
            relationship_queries = self._generate_relationship_queries(
                primary_keywords[:2], task_intent
            )

        return GeneratedQueries(
            primary_query=primary_query,
            alternative_queries=list(set(alternative_queries)),  # Remove duplicates
            entity_queries=list(set(entity_queries)),
            relationship_queries=list(set(relationship_queries)),
        )

    def _extract_main_concept(self, text: str) -> str:
        """Extract main concept from unstructured text."""
        # Simple heuristic: first noun or keyword-like word
        words = text.lower().split()
        for word in words:
            # Skip common words
            if len(word) > 3 and word.isalpha():
                return word.capitalize()
        return "analysis"

    def _create_broad_form(self, concept: str, domain: str) -> str:
        """Create a broader query form for concept."""
        return f"{concept} {domain}"

    def _create_narrow_form(self, concept: str, domain: str, task_intent: str) -> str:
        """Create a narrower, more specific query form."""
        return f"{task_intent} of {concept} in {domain}"

    def _generate_relationship_queries(self, concepts: list[str], task_intent: str) -> list[str]:
        """Generate queries expressing relationships between multiple concepts."""
        if len(concepts) < 2:
            return []

        queries = []
        concept_a, concept_b = concepts[0], concepts[1]

        # Use relationship patterns appropriate for task intent
        if "comparative" in task_intent:
            queries.append(
                self.RELATIONSHIP_PATTERNS["vs"].format(concept_a=concept_a, concept_b=concept_b)
            )
        elif "correlation" in task_intent or "hypothesis" in task_intent:
            queries.append(
                self.RELATIONSHIP_PATTERNS["correlation"].format(
                    concept_a=concept_a, concept_b=concept_b
                )
            )

        # Always add trade-off pattern (common in finance)
        queries.append(
            self.RELATIONSHIP_PATTERNS["trade_off"].format(concept_a=concept_a, concept_b=concept_b)
        )

        return queries

    @staticmethod
    def expand_with_synonyms(query: str, synonyms: dict = None) -> list[str]:
        """
        Expand a query by substituting synonyms.

        Args:
            query: Original query string
            synonyms: Dict mapping terms to synonym lists

        Returns:
            List of queries with synonyms substituted
        """
        if not synonyms:
            return [query]

        expanded = [query]
        for term, synonym_list in synonyms.items():
            if term in query.lower():
                for synonym in synonym_list[:2]:  # Use top 2 synonyms
                    expanded.append(query.replace(term, synonym))

        return list(set(expanded))  # Remove duplicates

    @staticmethod
    def format_for_log(queries: GeneratedQueries) -> str:
        """Format query variants for logging."""
        lines = [
            f"Primary query: '{queries.primary_query}'",
            f"Alternatives: {queries.alternative_queries}",
        ]
        if queries.entity_queries:
            lines.append(f"Entity-specific: {queries.entity_queries}")
        if queries.relationship_queries:
            lines.append(f"Relationships: {queries.relationship_queries}")

        return "\n  ".join(lines)
