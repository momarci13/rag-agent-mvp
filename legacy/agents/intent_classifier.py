"""
Intent classifier: Parses sloppy user instructions into structured intents.
Handles ambiguous phrasing, domain mixing, and informal terminology.
"""

from dataclasses import dataclass, field
from typing import Optional
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntentClassification:
    """Structured intent extracted from user instruction."""

    primary_domain: str  # finance, statistics, data_eng, ml, writing, general
    secondary_domains: list[str] = field(default_factory=list)  # Additional domains detected
    task_intent: str = ""  # comparative_analysis, explanation, strategy_design, etc.
    confidence: float = 0.5  # 0.0 - 1.0
    clarification_needed: bool = False  # True if ambiguous
    clarification_suggestion: str = ""  # What to clarify
    primary_keywords: list[str] = field(default_factory=list)  # Main concepts mentioned
    entity_names: list[str] = field(default_factory=list)  # Specific symbols/names (BTC, AAPL, etc.)
    instruction_type: str = ""  # question, analysis_request, research_request, synthesis_request


class IntentClassifier:
    """Classifies unstructured user instructions into structured intents."""

    # Define known task intents and their keywords/patterns
    TASK_INTENTS = {
        "comparative_analysis": ["compare", "versus", "vs", "difference between", "how different", "relative"],
        "explanation": ["explain", "why", "how does", "what is", "interpret", "understand", "mechanism"],
        "strategy_design": ["design", "strategy", "build", "create", "backtest", "trading"],
        "literature_review": ["review", "survey", "literature", "systematic", "gaps in"],
        "hypothesis_formation": ["hypothesize", "test if", "hypothesis", "conjecture", "suppose"],
        "data_analysis": ["analyze", "analysis", "explore", "statistics", "correlation", "regression"],
        "optimization": ["optimize", "improve", "best", "maximize", "minimize", "tuning"],
        "prediction": ["predict", "forecast", "estimate", "future", "projection"],
        "synthesis": ["synthesize", "summarize", "consolidate", "integrate", "combine"],
    }

    # Domain-specific keywords for classification
    DOMAIN_KEYWORDS = {
        "finance": [
            "stock", "equity", "bond", "crypto", "trading", "portfolio", "sharpe",
            "volatility", "momentum", "price", "market", "financial", "investment",
            "returns", "strategy", "backtest", "derivative", "hedging", "carry"
        ],
        "statistics": [
            "regression", "correlation", "hypothesis", "test", "distribution", "normal",
            "p_value", "significant", "variance", "mean", "median", "standard_deviation",
            "coefficient", "statistical", "inference"
        ],
        "data_eng": [
            "data", "timeseries", "time_series", "dataset", "ingest", "pipeline",
            "transform", "sql", "database", "streaming", "batch", "etl", "preprocessing"
        ],
        "ml": [
            "machine learning", "ml", "neural network", "model", "train", "deep learning",
            "algorithm", "prediction", "classification", "regression", "tensorflow",
            "pytorch", "xgboost", "ensemble", "feature"
        ],
        "writing": [
            "paper", "write", "research", "publication", "report", "article", "document",
            "academic", "abstract", "introduction", "literature", "methodology"
        ],
        "general": []  # catch-all
    }

    def __init__(self, llm=None):
        """Initialize with optional LLM for sophisticated classification."""
        self.llm = llm

    def classify(self, instruction: str) -> IntentClassification:
        """
        Classify a user instruction into a structured intent.

        Args:
            instruction: Raw user instruction (can be sloppy/informal)

        Returns:
            IntentClassification object
        """

        lower_instr = instruction.lower()

        # 1. Detect domains by keyword matching
        domain_scores = self._score_domains(lower_instr)
        primary_domain = max(domain_scores, key=domain_scores.get)
        secondary_domains = [d for d in domain_scores
                           if d != primary_domain and domain_scores[d] > 0.3]

        # 2. Detect task intent
        task_intent = self._detect_task_intent(lower_instr)

        # 3. Extract keywords and entities
        primary_keywords = self._extract_keywords(lower_instr, primary_domain)
        entity_names = self._extract_entities(instruction)  # Use original, not lowercased

        # 4. Determine instruction type
        instruction_type = self._classify_instruction_type(lower_instr)

        # 5. Check for ambiguity
        clarification_needed = False
        clarification_suggestion = ""

        if len(secondary_domains) > 1:
            # Multiple domains detected — ambiguous
            clarification_needed = True
            clarification_suggestion = (
                f"I detected multiple domains: {', '.join([primary_domain] + secondary_domains[:2])}. "
                f"Which is the main focus?"
            )
        elif domain_scores[primary_domain] < 0.4:
            # Weak domain signal
            clarification_needed = True
            clarification_suggestion = (
                f"Your instruction seems general. Are you looking for {primary_domain}? "
                f"Can you be more specific about what you need?"
            )

        # 6. Determine confidence
        confidence = domain_scores[primary_domain]
        if clarification_needed:
            confidence *= 0.7  # Lower confidence if ambiguous

        return IntentClassification(
            primary_domain=primary_domain,
            secondary_domains=secondary_domains,
            task_intent=task_intent,
            confidence=confidence,
            clarification_needed=clarification_needed,
            clarification_suggestion=clarification_suggestion,
            primary_keywords=primary_keywords,
            entity_names=entity_names,
            instruction_type=instruction_type,
        )

    def _score_domains(self, text: str) -> dict[str, float]:
        """Score how likely each domain matches the text."""
        scores = {}

        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text)
            # Normalize by keyword list length to avoid bias toward domains with many keywords
            score = matches / max(len(keywords), 1) if keywords else 0.0
            scores[domain] = min(score, 1.0)  # Cap at 1.0

        # If no strong signal, default to general
        if max(scores.values()) == 0.0:
            scores["general"] = 0.1

        return scores

    def _detect_task_intent(self, text: str) -> str:
        """Detect task intent from keywords and patterns."""
        for intent, keywords in self.TASK_INTENTS.items():
            if any(kw in text for kw in keywords):
                return intent
        return "data_analysis"  # default

    def _extract_keywords(self, text: str, domain: str) -> list[str]:
        """Extract primary keywords relevant to domain."""
        keywords = []

        # Domain-specific keywords
        domain_kws = self.DOMAIN_KEYWORDS.get(domain, [])
        for kw in domain_kws:
            if kw in text:
                keywords.append(kw)

        # Task intent keywords
        for intent, intent_kws in self.TASK_INTENTS.items():
            for kw in intent_kws:
                if kw in text and kw not in keywords:
                    keywords.append(kw)

        return list(set(keywords))[:5]  # Return top 5

    def _extract_entities(self, text: str) -> list[str]:
        """Extract specific entities (stock symbols, concepts, etc.)."""
        import re
        entities = []

        # 1. Common stock symbols: sequences of 1-5 uppercase letters
        # Split on non-alphanumeric and look for all-caps sequences
        tokens = re.split(r'[^A-Za-z]+', text)
        for token in tokens:
            if token and 1 <= len(token) <= 5 and token.isupper():
                entities.append(token)

        # 2. Cryptocurrency names (case-insensitive)
        crypto_mapping = {
            "bitcoin": "Bitcoin",
            "ethereum": "Ethereum",
            "btc": "BTC",
            "eth": "ETH",
        }
        for crypto_lower, crypto_proper in crypto_mapping.items():
            if crypto_lower in text.lower():
                entities.append(crypto_proper)

        # 3. Known company names (when capitalized)
        company_keywords = ["Apple", "Tesla", "Amazon", "Google", "Meta", "Microsoft", "Netflix"]
        for company in company_keywords:
            if company in text:
                entities.append(company)

        # Remove duplicates, keep unique entries
        entities = list(set(entities))

        # Filter out very common words that might get through
        common_words = {"THE", "AND", "OR", "FOR", "WITH", "FROM", "TO", "BY", "IN", "IF", "A", "I"}
        entities = [e for e in entities if e.upper() not in common_words]

        return entities[:5]  # Return top 5

    def _classify_instruction_type(self, text: str) -> str:
        """Classify whether instruction is question, request, etc."""
        if "?" in text:
            return "question"
        elif any(word in text for word in ["analyze", "analyze", "compute", "calculate"]):
            return "analysis_request"
        elif any(word in text for word in ["research", "investigate", "explore"]):
            return "research_request"
        elif any(word in text for word in ["synthesize", "summarize", "combine", "integrate"]):
            return "synthesis_request"
        else:
            return "general_request"

    @staticmethod
    def format_for_user(classification: IntentClassification) -> str:
        """Format classification output for user display."""
        lines = [
            f"📋 **Detected Intent**",
            f"   Domain: {classification.primary_domain}",
            f"   Task: {classification.task_intent}",
            f"   Confidence: {classification.confidence:.0%}",
        ]

        if classification.secondary_domains:
            lines.append(f"   Also detected: {', '.join(classification.secondary_domains)}")

        if classification.entity_names:
            lines.append(f"   Entities: {', '.join(classification.entity_names)}")

        if classification.clarification_needed:
            lines.append(f"\n⚠️ {classification.clarification_suggestion}")

        return "\n".join(lines)
