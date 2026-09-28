"""Problem decoder role for parsing user tasks into structured analysis requirements.

Uses LLM + RAG to understand data needs, geographies, time ranges, and analysis types.
"""
from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .llm import HostedLLM
from .roles import build_messages


class DataRequirement(BaseModel):
    """Structured data requirement extracted from user task."""
    symbols: List[str] = Field(description="List of data symbols/identifiers")
    sources: List[str] = Field(description="Preferred data sources (yahoo, ksh, mnb, eurostat)")
    start_date: str = Field(description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(default=None, description="End date in YYYY-MM-DD format, None for latest")
    frequency: str = Field(description="Data frequency (daily, monthly, quarterly, annual)")
    data_types: List[str] = Field(description="Types of data needed (price, economic, fundamental)")


class AnalysisRequirement(BaseModel):
    """Structured analysis requirement."""
    analysis_type: str = Field(description="Type of analysis (correlation, regression, time_series, etc.)")
    variables: List[str] = Field(description="Key variables to analyze")
    hypotheses: List[str] = Field(description="Hypotheses to test")
    visualizations: List[str] = Field(description="Required plot types")
    statistical_tests: List[str] = Field(description="Statistical tests to perform")


class ProblemDecoding(BaseModel):
    """Complete structured decoding of user problem."""
    task_type: str = Field(description="Overall task category (data_analysis, trading, research)")
    data_requirements: List[DataRequirement] = Field(description="Data fetching requirements")
    analysis_requirements: AnalysisRequirement = Field(description="Analysis specifications")
    report_requirements: Dict = Field(description="Report structure and content needs")
    confidence: float = Field(description="Confidence score in the decoding (0-1)")


def decode_problem(llm: HostedLLM, task: str, rag_context: Optional[List[Dict]] = None) -> ProblemDecoding:
    """Decode user problem into structured requirements using LLM + RAG.

    Args:
        llm: LLM instance for decoding
        task: Natural language task description
        rag_context: Optional retrieved context from KB

    Returns:
        Structured ProblemDecoding with data and analysis requirements
    """
    system_prompt = """You are a problem decoder for quantitative analysis tasks.

Your job is to parse natural language descriptions of data analysis problems and extract:
1. Data requirements (symbols, sources, dates, frequency, types)
2. Analysis requirements (type, variables, hypotheses, visualizations, tests)
3. Report requirements (structure, content)

Available data sources:
- yahoo: Stock prices, ETFs, indices (AAPL, SPY, ^VIX)
- ksh: Hungarian economic statistics (HU_CPI, HU_GDP, HU_UNEMPLOYMENT)
- mnb: Hungarian financial data (HUF/EUR exchange rates, interest rates)
- eurostat: EU-wide economic data (EU_INFLATION, EU_GDP)

Analysis types: correlation, regression, time_series, forecasting, hypothesis_testing, descriptive

Be specific about symbols and map common terms to data sources.
Return valid JSON matching the ProblemDecoding schema.
"""

    context_text = ""
    if rag_context:
        context_text = "\n\nRelevant context:\n" + "\n".join([f"- {doc.get('text', '')[:200]}..." for doc in rag_context])

    user_prompt = f"""Decode this analysis task into structured requirements:

Task: {task}

{context_text}

Return JSON with:
- task_type: "data_analysis"
- data_requirements: array of objects with symbols, sources, start_date, end_date, frequency, data_types
- analysis_requirements: object with analysis_type, variables, hypotheses, visualizations, statistical_tests
- report_requirements: object with sections, include_plots, include_data
- confidence: 0.0-1.0"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.chat(messages, temperature=0.1)

    # Parse JSON response
    try:
        import json
        parsed = json.loads(response)
        return ProblemDecoding(**parsed)
    except Exception as e:
        # Fallback: create basic decoding
        return ProblemDecoding(
            task_type="data_analysis",
            data_requirements=[DataRequirement(
                symbols=["SPY"],
                sources=["yahoo"],
                start_date="2020-01-01",
                frequency="daily",
                data_types=["price"]
            )],
            analysis_requirements=AnalysisRequirement(
                analysis_type="descriptive",
                variables=["price"],
                hypotheses=[],
                visualizations=["line_plot"],
                statistical_tests=["summary_stats"]
            ),
            report_requirements={"sections": ["introduction", "analysis", "conclusion"], "include_plots": True, "include_data": True},
            confidence=0.5
        )


def validate_requirements(decoding: ProblemDecoding) -> List[str]:
    """Validate decoded requirements for feasibility.

    Returns list of validation warnings/errors.
    """
    warnings = []

    # Check data requirements
    for req in decoding.data_requirements:
        if not req.symbols:
            warnings.append("No symbols specified in data requirements")
        if req.start_date > (req.end_date or "2030-01-01"):
            warnings.append("Start date after end date")
        if req.frequency not in ["daily", "weekly", "monthly", "quarterly", "annual"]:
            warnings.append(f"Unsupported frequency: {req.frequency}")

    # Check analysis requirements
    if decoding.analysis_requirements.analysis_type not in [
        "correlation", "regression", "time_series", "forecasting",
        "hypothesis_testing", "descriptive", "clustering", "classification"
    ]:
        warnings.append(f"Unsupported analysis type: {decoding.analysis_requirements.analysis_type}")

    return warnings
