"""Query expansion for RAG retrieval.

Provides synonym expansion, related term discovery, and multi-query generation
to improve retrieval quality beyond single-pass queries.
Supports LLM-based reformulation when an OllamaLLM instance is available.
"""
from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from agents.llm import OllamaLLM

logger = logging.getLogger(__name__)

try:
    import nltk
    from nltk.corpus import wordnet
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    _WORDNET_AVAILABLE = True
except ImportError:
    _WORDNET_AVAILABLE = False


class QueryExpander:
    """Expand queries with synonyms, related terms, and domain-specific expansions."""

    def __init__(self):
        # Domain-specific term mappings for quant/trading/finance
        self.domain_expansions = {
            "momentum": ["trend following", "momentum strategy", "price momentum", "momentum investing"],
            "value": ["value investing", "fundamental value", "undervalued", "value premium"],
            "volatility": ["volatility", "vol", "variance", "risk", "standard deviation"],
            "sharpe": ["sharpe ratio", "risk-adjusted return", "sharpe", "performance measure"],
            "backtest": ["backtesting", "historical simulation", "strategy testing", "performance testing"],
            "portfolio": ["portfolio", "asset allocation", "investment portfolio", "portfolio optimization"],
            "risk": ["risk", "volatility", "drawdown", "var", "value at risk"],
            "alpha": ["alpha", "excess return", "abnormal return", "alpha generation"],
            "beta": ["beta", "market beta", "systematic risk", "market sensitivity"],
            "correlation": ["correlation", "correlation coefficient", "dependency", "relationship"],
            "optimization": ["optimization", "portfolio optimization", "mean variance", "efficient frontier"],
            "machine learning": ["ml", "machine learning", "statistical learning", "predictive modeling"],
            "factor": ["factor", "risk factor", "factor model", "multi-factor"],
            "trading": ["trading", "trade execution", "trading strategy", "algorithmic trading"],
        }

    def expand_query(self, query: str, domain: str = "general", max_expansions: int = 3) -> list[str]:
        """Expand a query into multiple related queries.

        Args:
            query: original search query
            domain: domain context ("ds", "quant", "writing", "general")
            max_expansions: maximum number of expanded queries to return

        Returns:
            list of expanded queries including original
        """
        expansions = [query]  # Always include original

        # Clean and tokenize query
        clean_query = query.lower().strip()
        tokens = re.findall(r'\b\w+\b', clean_query)

        # Domain-specific expansions
        domain_terms = []
        for token in tokens:
            if token in self.domain_expansions:
                domain_terms.extend(self.domain_expansions[token][:2])  # Limit per term

        if domain_terms:
            # Create domain-aware expansion
            domain_expansion = f"{query} {' '.join(domain_terms[:3])}"
            expansions.append(domain_expansion.strip())

        # Synonym-based expansions (if WordNet available)
        if _WORDNET_AVAILABLE:
            synonym_terms = []
            for token in tokens:
                if len(token) > 3:  # Skip short words
                    synonyms = self._get_synonyms(token)
                    synonym_terms.extend(synonyms[:2])  # Limit synonyms per word

            if synonym_terms:
                synonym_expansion = f"{query} {' '.join(synonym_terms[:3])}"
                expansions.append(synonym_expansion.strip())

        # Query reformulation for different aspects
        if domain == "quant":
            # Add mathematical/statistical perspectives
            math_terms = ["mathematical", "statistical", "quantitative", "model"]
            expansions.append(f"{query} {' '.join(math_terms[:2])}")
        elif domain == "ds":
            # Add data science perspectives
            ds_terms = ["data", "analysis", "modeling", "prediction"]
            expansions.append(f"{query} {' '.join(ds_terms[:2])}")
        elif domain == "writing":
            # Add academic/research perspectives
            writing_terms = ["research", "literature", "methodology", "findings"]
            expansions.append(f"{query} {' '.join(writing_terms[:2])}")

        # Remove duplicates and limit
        seen = set()
        unique_expansions = []
        for exp in expansions:
            if exp not in seen and len(unique_expansions) < max_expansions + 1:
                seen.add(exp)
                unique_expansions.append(exp)

        return unique_expansions

    def _get_synonyms(self, word: str) -> list[str]:
        """Get synonyms for a word using WordNet."""
        if not _WORDNET_AVAILABLE:
            return []

        synonyms = set()
        try:
            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().lower().replace('_', ' ')
                    if synonym != word and len(synonym) > 2:
                        synonyms.add(synonym)
        except Exception:
            pass

        return list(synonyms)[:5]  # Limit to 5 synonyms

    def expand_with_related_terms(self, query: str, related_terms: list[str]) -> list[str]:
        """Expand query with manually provided related terms.

        Args:
            query: original query
            related_terms: list of related terms to add

        Returns:
            list of expanded queries
        """
        expansions = [query]

        if related_terms:
            # Add related terms as separate query
            related_query = f"{query} {' '.join(related_terms)}"
            expansions.append(related_query)

            # Also try individual related terms
            for term in related_terms[:2]:  # Limit to 2
                expansions.append(f"{query} {term}")

        return expansions


def expand_query_with_llm(
    query: str,
    llm: "OllamaLLM",
    n_variants: int = 3,
) -> list[str]:
    """Generate n_variants academically-diverse reformulations via LLM JSON call.

    Returns new queries only (original excluded). Falls back to [] on failure.
    """
    msgs = [
        {
            "role": "system",
            "content": (
                f"Generate {n_variants} alternative academic search queries for the query below. "
                "Each variant should target a different aspect or use different terminology. "
                "Keep each query 3–8 words. "
                f'Respond with JSON only: {{"queries": ["...", "...", "..."]}}'
            ),
        },
        {"role": "user", "content": query[:300]},
    ]
    try:
        result = llm.chat_json(msgs, temperature=0.0)
        variants = result.get("queries") or []
        seen = {query.lower()}
        cleaned = [v for v in variants if isinstance(v, str) and v.lower() not in seen]
        return cleaned[:n_variants]
    except Exception as exc:
        logger.warning("[QUERY_EXPANSION] LLM expansion failed: %s", exc)
        return []


def expand_quant_query(query: str) -> list[str]:
    """Convenience function for quantitative finance queries."""
    expander = QueryExpander()
    return expander.expand_query(query, domain="quant")


def expand_ds_query(query: str) -> list[str]:
    """Convenience function for data science queries."""
    expander = QueryExpander()
    return expander.expand_query(query, domain="ds")


def expand_writing_query(query: str) -> list[str]:
    """Convenience function for academic writing queries."""
    expander = QueryExpander()
    return expander.expand_query(query, domain="writing")


# Example usage:
if __name__ == "__main__":
    expander = QueryExpander()

    # Test quant query
    quant_queries = expander.expand_query("momentum strategy", domain="quant")
    print("Quant expansions:", quant_queries)

    # Test DS query
    ds_queries = expander.expand_query("machine learning model", domain="ds")
    print("DS expansions:", ds_queries)

    # Test writing query
    writing_queries = expander.expand_query("research methodology", domain="writing")
    print("Writing expansions:", writing_queries)