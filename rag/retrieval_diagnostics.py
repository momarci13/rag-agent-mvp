"""
Retrieval diagnostics: Track and analyze retrieval failure patterns.
Helps identify KB gaps and improve retrieval strategy over time.
"""

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalFailure:
    """Record of a retrieval failure for analysis."""

    timestamp: str
    query: str
    tier_reached: int
    mean_score: float
    num_docs_found: int
    failure_type: str  # "no_docs", "low_score", "false_positive", "timeout"
    task_id: str = ""
    domain: str = ""
    keywords: list[str] = None
    notes: str = ""

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class RetrievalDiagnostics:
    """
    Logs and analyzes retrieval failures.
    Helps identify patterns (e.g., all crypto queries fail, bias against certain domains).
    """

    def __init__(self, log_path: str = "output/retrieval_failures.jsonl"):
        """Initialize diagnostics with log file path."""
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_failure(
        self,
        query: str,
        tier_reached: int,
        mean_score: float,
        num_docs_found: int,
        failure_type: str,
        task_id: str = "",
        domain: str = "",
        keywords: list[str] = None,
        notes: str = "",
    ):
        """Log a retrieval failure."""

        failure = RetrievalFailure(
            timestamp=datetime.utcnow().isoformat(),
            query=query,
            tier_reached=tier_reached,
            mean_score=mean_score,
            num_docs_found=num_docs_found,
            failure_type=failure_type,
            task_id=task_id,
            domain=domain,
            keywords=keywords or [],
            notes=notes,
        )

        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(asdict(failure)) + "\n")
        except Exception as e:
            logger.error(f"Failed to log retrieval failure: {e}")

    def analyze_failures(self, limit: int = 100) -> dict:
        """
        Analyze logged failures to identify patterns.

        Returns:
            Dict with pattern analysis
        """
        if not self.log_path.exists():
            return {"no_data": True}

        failures = []
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    if line.strip():
                        failures.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read failure logs: {e}")
            return {"error": str(e)}

        if not failures:
            return {"total_failures": 0}

        # Analyze patterns
        by_failure_type = {}
        by_domain = {}
        by_keyword = {}
        tier_distribution = {}

        for failure in failures[-limit:]:  # Last N failures
            # By failure type
            ft = failure["failure_type"]
            by_failure_type[ft] = by_failure_type.get(ft, 0) + 1

            # By domain
            domain = failure.get("domain", "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(failure["mean_score"])

            # By keyword
            for kw in failure.get("keywords", []):
                if kw not in by_keyword:
                    by_keyword[kw] = []
                by_keyword[kw].append(failure["mean_score"])

            # Tier distribution
            tier = failure["tier_reached"]
            tier_distribution[tier] = tier_distribution.get(tier, 0) + 1

        # Aggregate by domain: avg score per domain
        domain_stats = {}
        for domain, scores in by_domain.items():
            domain_stats[domain] = {
                "failures": len(scores),
                "avg_score": sum(scores) / len(scores),
                "min_score": min(scores),
                "max_score": max(scores),
            }

        # Identify problematic keywords
        keyword_stats = {}
        for kw, scores in by_keyword.items():
            if len(scores) >= 2:  # Only if seen multiple times
                keyword_stats[kw] = {
                    "failures": len(scores),
                    "avg_score": sum(scores) / len(scores),
                    "severity": "critical" if sum(scores) / len(scores) < 0.15 else "warning",
                }

        # Sort by problem severity
        keyword_stats = dict(sorted(
            keyword_stats.items(),
            key=lambda x: x[1]["avg_score"]
        ))

        return {
            "total_failures_analyzed": len(failures[-limit:]),
            "failure_types": by_failure_type,
            "tier_distribution": tier_distribution,
            "by_domain": domain_stats,
            "problematic_keywords": keyword_stats,
            "recommendations": self._generate_recommendations(domain_stats, keyword_stats),
        }

    @staticmethod
    def _generate_recommendations(domain_stats: dict, keyword_stats: dict) -> list[str]:
        """Generate recommendations based on failure analysis."""
        recommendations = []

        # Check for domains with high failure rates
        for domain, stats in domain_stats.items():
            if stats["avg_score"] < 0.20:
                recommendations.append(
                    f"Domain '{domain}': Avg score {stats['avg_score']:.3f} is very low. "
                    f"Consider adding more documents to KB or improving embedding model."
                )

        # Check for problematic keywords
        for keyword, stats in list(keyword_stats.items())[:5]:
            if stats["failures"] >= 3:
                recommendations.append(
                    f"Keyword '{keyword}': {stats['failures']} failures with avg score {stats['avg_score']:.3f}. "
                    f"Add domain concepts mapping or relevant documents."
                )

        if not recommendations:
            recommendations.append("Retrieval performance looks good overall!")

        return recommendations

    def get_failure_summary(self) -> str:
        """Get human-readable summary of failure patterns."""
        analysis = self.analyze_failures()

        if analysis.get("no_data"):
            return "No failure data yet."

        lines = [
            f"📊 Retrieval Failure Analysis (last 100)",
            f"Total failures: {analysis['total_failures_analyzed']}",
            "",
            "Failure types:",
        ]

        for ft, count in analysis.get("failure_types", {}).items():
            lines.append(f"  - {ft}: {count}")

        lines.append("\nBy domain:")
        for domain, stats in analysis.get("by_domain", {}).items():
            lines.append(
                f"  - {domain}: {stats['failures']} failures, avg_score={stats['avg_score']:.3f}"
            )

        if analysis.get("problematic_keywords"):
            lines.append("\nProblematic keywords (critical issues):")
            for kw, stats in list(analysis["problematic_keywords"].items())[:5]:
                lines.append(f"  - {kw}: {stats['failures']} failures, score={stats['avg_score']:.3f}")

        if analysis.get("recommendations"):
            lines.append("\n💡 Recommendations:")
            for rec in analysis["recommendations"][:3]:
                lines.append(f"  • {rec}")

        return "\n".join(lines)
