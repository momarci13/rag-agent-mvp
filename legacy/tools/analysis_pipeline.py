"""Composable analysis pipeline for data science tasks.

Allows building DAGs of analysis steps that can be executed sequentially,
with intermediate results passed between steps.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class PipelineContext:
    """Context passed between pipeline steps."""
    data: Dict[str, pd.DataFrame] = None  # symbol -> DataFrame
    variables: Dict[str, Any] = None      # named variables/results
    metadata: Dict[str, Any] = None       # step metadata
    artifacts: List[Dict] = None          # plots, files, etc.

    def __post_init__(self):
        if self.data is None:
            self.data = {}
        if self.variables is None:
            self.variables = {}
        if self.metadata is None:
            self.metadata = {}
        if self.artifacts is None:
            self.artifacts = []


class PipelineStep(ABC):
    """Abstract base class for pipeline steps."""

    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute this step, modifying context in place.

        Args:
            context: Current pipeline context

        Returns:
            Modified context (can be same object)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Step name for logging."""
        pass


class FetchStep(PipelineStep):
    """Fetch data from sources."""

    def __init__(self, symbols: List[str], sources: List[str], start_date: str, end_date: Optional[str] = None):
        self.symbols = symbols
        self.sources = sources
        self.start_date = start_date
        self.end_date = end_date

    def execute(self, context: PipelineContext) -> PipelineContext:
        from .data import MultiSourceFetcher
        fetcher = MultiSourceFetcher()
        data = fetcher.fetch(self.symbols, self.start_date, self.end_date or "2030-01-01")
        context.data.update(data)
        context.metadata[f"{self.name}_fetched"] = list(data.keys())
        return context

    @property
    def name(self) -> str:
        return "fetch"


class PreprocessStep(PipelineStep):
    """Preprocess data (cleaning, alignment, feature engineering)."""

    def __init__(self, operations: List[str]):
        self.operations = operations

    def execute(self, context: PipelineContext) -> PipelineContext:
        for symbol, df in context.data.items():
            df = df.copy()
            for op in self.operations:
                if op == "drop_na":
                    df = df.dropna()
                elif op == "fill_forward":
                    df = df.fillna(method='ffill')
                elif op == "add_returns":
                    df['returns'] = df['close'].pct_change()
                elif op == "add_log_returns":
                    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
                # Add more operations as needed
            context.data[symbol] = df
        context.metadata[f"{self.name}_operations"] = self.operations
        return context

    @property
    def name(self) -> str:
        return "preprocess"


class StatisticalTestStep(PipelineStep):
    """Perform statistical tests."""

    def __init__(self, tests: List[str], variables: List[str]):
        self.tests = tests
        self.variables = variables

    def execute(self, context: PipelineContext) -> PipelineContext:
        import scipy.stats as stats
        results = {}

        # Simple correlation test
        if "correlation" in self.tests and len(self.variables) >= 2:
            var1, var2 = self.variables[:2]
            if var1 in context.data and var2 in context.data:
                df1 = context.data[var1]
                df2 = context.data[var2]
                # Align dates
                common_idx = df1.index.intersection(df2.index)
                if len(common_idx) > 10:
                    corr, p_value = stats.pearsonr(df1.loc[common_idx, 'close'], df2.loc[common_idx, 'close'])
                    results["correlation"] = {"coefficient": corr, "p_value": p_value}

        context.variables.update(results)
        context.metadata[f"{self.name}_results"] = list(results.keys())
        return context

    @property
    def name(self) -> str:
        return "statistical_test"


class VisualizationStep(PipelineStep):
    """Generate plots and visualizations."""

    def __init__(self, plot_types: List[str], variables: List[str]):
        self.plot_types = plot_types
        self.variables = variables

    def execute(self, context: PipelineContext) -> PipelineContext:
        import matplotlib.pyplot as plt
        import io
        import base64

        plots = []
        for plot_type in self.plot_types:
            if plot_type == "line_plot" and self.variables:
                plt.figure(figsize=(10, 6))
                for var in self.variables:
                    if var in context.data:
                        plt.plot(context.data[var].index, context.data[var]['close'], label=var)
                plt.legend()
                plt.title("Price Time Series")
                plt.xlabel("Date")
                plt.ylabel("Price")

                # Save to base64
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                img_base64 = base64.b64encode(buf.read()).decode('utf-8')
                plots.append({
                    "type": "line_plot",
                    "title": "Price Time Series",
                    "image_base64": img_base64
                })
                plt.close()

        context.artifacts.extend(plots)
        context.metadata[f"{self.name}_plots"] = len(plots)
        return context

    @property
    def name(self) -> str:
        return "visualization"


class AnalysisPipeline:
    """Composable analysis pipeline."""

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def execute(self, initial_context: Optional[PipelineContext] = None) -> PipelineContext:
        """Execute the pipeline."""
        context = initial_context or PipelineContext()

        for step in self.steps:
            print(f"[PIPELINE] Executing {step.name}")
            try:
                context = step.execute(context)
            except Exception as e:
                print(f"[PIPELINE] Step {step.name} failed: {e}")
                context.metadata[f"{step.name}_error"] = str(e)

        return context

    @classmethod
    def from_requirements(cls, decoding: 'ProblemDecoding') -> 'AnalysisPipeline':
        """Build pipeline from problem decoding."""
        steps = []

        # Add fetch steps
        for req in decoding.data_requirements:
            steps.append(FetchStep(
                symbols=req.symbols,
                sources=req.sources,
                start_date=req.start_date,
                end_date=req.end_date
            ))

        # Add preprocess step
        steps.append(PreprocessStep(["drop_na", "add_returns"]))

        # Add analysis steps based on requirements
        analysis_req = decoding.analysis_requirements
        if "correlation" in analysis_req.analysis_type.lower():
            steps.append(StatisticalTestStep(
                tests=["correlation"],
                variables=analysis_req.variables
            ))

        # Add visualization
        if analysis_req.visualizations:
            steps.append(VisualizationStep(
                plot_types=analysis_req.visualizations,
                variables=analysis_req.variables
            ))

        return cls(steps)


# Convenience functions
def build_basic_pipeline(symbols: List[str], start_date: str) -> AnalysisPipeline:
    """Build a basic fetch + preprocess + visualize pipeline."""
    steps = [
        FetchStep(symbols, ["yahoo"], start_date),
        PreprocessStep(["drop_na", "add_returns"]),
        VisualizationStep(["line_plot"], symbols)
    ]
    return AnalysisPipeline(steps)