"""Report builder for analysis results with embedded plots and dataset export.

Generates LaTeX reports with matplotlib plots and exports datasets as CSV/JSON.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

import pandas as pd


@dataclass
class ReportSection:
    """A section in the report."""
    title: str
    content: str
    plots: List[Dict] = None  # List of plot dicts with image_base64
    code_blocks: List[str] = None

    def __post_init__(self):
        if self.plots is None:
            self.plots = []
        if self.code_blocks is None:
            self.code_blocks = []


@dataclass
class AnalysisReport:
    """Complete analysis report."""
    title: str
    sections: List[ReportSection]
    datasets: Dict[str, pd.DataFrame] = None  # Named datasets to export
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = {}
        if self.metadata is None:
            self.metadata = {}


class ReportBuilder:
    """Builds LaTeX reports with embedded plots and dataset export."""

    def __init__(self, output_dir: str = "output/report"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_latex_report(self, report: AnalysisReport) -> str:
        """Build LaTeX report content."""
        latex_parts = [
            r"\documentclass{article}",
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage{listings}",
            r"\usepackage[margin=1in]{geometry}",
            r"\title{" + report.title + "}",
            r"\author{Auto-generated Analysis Report}",
            r"\date{\today}",
            r"\begin{document}",
            r"\maketitle",
            r"\tableofcontents",
            r"\newpage"
        ]

        for section in report.sections:
            latex_parts.append(r"\section{" + section.title + "}")
            latex_parts.append(section.content)

            # Add plots
            for i, plot in enumerate(section.plots):
                plot_path = self._save_plot_image(plot, f"{section.title.lower().replace(' ', '_')}_plot_{i}")
                latex_parts.append(r"\begin{figure}[H]")
                latex_parts.append(r"\centering")
                latex_parts.append(r"\includegraphics[width=0.8\textwidth]{" + str(plot_path) + "}")
                latex_parts.append(r"\caption{" + plot.get("title", "Analysis Plot") + "}")
                latex_parts.append(r"\end{figure}")

            # Add code blocks
            for code in section.code_blocks:
                latex_parts.append(r"\begin{lstlisting}[language=Python]")
                latex_parts.append(code)
                latex_parts.append(r"\end{lstlisting}")

        latex_parts.extend([
            r"\end{document}"
        ])

        return "\n".join(latex_parts)

    def _save_plot_image(self, plot: Dict, filename: str) -> Path:
        """Save base64 plot image to file."""
        import base64
        import io
        from PIL import Image

        img_data = base64.b64decode(plot["image_base64"])
        img = Image.open(io.BytesIO(img_data))

        output_path = self.output_dir / f"{filename}.png"
        img.save(output_path)
        return output_path

    def export_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, str]:
        """Export datasets as CSV and JSON."""
        exported = {}

        for name, df in datasets.items():
            # CSV export
            csv_path = self.output_dir / f"{name}.csv"
            df.to_csv(csv_path)
            exported[f"{name}_csv"] = str(csv_path)

            # JSON export
            json_path = self.output_dir / f"{name}.json"
            df.to_json(json_path, orient="records", date_format="iso")
            exported[f"{name}_json"] = str(json_path)

        return exported

    def build_report(self, pipeline_context: 'PipelineContext', decoding: 'ProblemDecoding') -> AnalysisReport:
        """Build report from pipeline results."""
        sections = []

        # Introduction
        intro_content = f"""
This report presents the analysis of: {decoding.task_type}

Data sources used: {', '.join([req.sources[0] if req.sources else 'unknown' for req in decoding.data_requirements])}
Time period: {decoding.data_requirements[0].start_date if decoding.data_requirements else 'Not specified'} to present
"""
        sections.append(ReportSection("Introduction", intro_content))

        # Data Summary
        data_summary = "Datasets analyzed:\n"
        for symbol, df in pipeline_context.data.items():
            data_summary += f"- {symbol}: {len(df)} observations from {df.index.min()} to {df.index.max()}\n"
        sections.append(ReportSection("Data Summary", data_summary))

        # Analysis Results
        results_content = "Key findings:\n"
        for key, value in pipeline_context.variables.items():
            results_content += f"- {key}: {value}\n"
        sections.append(ReportSection("Analysis Results", results_content, plots=pipeline_context.artifacts))

        # Methodology
        methodology = """
Analysis pipeline:
1. Data fetching from specified sources
2. Preprocessing (missing value handling, feature engineering)
3. Statistical analysis and testing
4. Visualization generation
"""
        sections.append(ReportSection("Methodology", methodology))

        return AnalysisReport(
            title=f"Analysis Report: {decoding.task_type}",
            sections=sections,
            datasets=pipeline_context.data,
            metadata={
                "decoding_confidence": decoding.confidence,
                "pipeline_steps": list(pipeline_context.metadata.keys())
            }
        )

    def generate_report(self, pipeline_context: 'PipelineContext', decoding: 'ProblemDecoding') -> Dict[str, Any]:
        """Generate complete report with LaTeX, plots, and datasets."""
        report = self.build_report(pipeline_context, decoding)

        # Generate LaTeX
        latex_content = self.build_latex_report(report)
        latex_path = self.output_dir / "report.tex"
        latex_path.write_text(latex_content)

        # Export datasets
        dataset_paths = self.export_datasets(report.datasets)

        # Try to compile PDF
        pdf_path = None
        try:
            from pathlib import Path as _Path
            from .tex import build_latex_artifact
            _bib_path = _Path(__file__).parent.parent / "data" / "papers" / "refs.bib"
            _bib = _bib_path.read_text(encoding="utf-8") if _bib_path.exists() else ""
            tex_p, pdf_p, dropped = build_latex_artifact(latex_content, _bib, str(self.output_dir))
            if pdf_p:
                pdf_path = pdf_p
        except Exception as e:
            print(f"PDF compilation failed: {e}")

        return {
            "latex_path": str(latex_path),
            "pdf_path": pdf_path,
            "dataset_paths": dataset_paths,
            "plots_saved": len([a for a in pipeline_context.artifacts if "image_base64" in a]),
            "sections": len(report.sections)
        }


# Convenience function
def generate_analysis_report(pipeline_context: 'PipelineContext', decoding: 'ProblemDecoding', output_dir: str = "output/report") -> Dict[str, Any]:
    """Generate a complete analysis report."""
    builder = ReportBuilder(output_dir)
    return builder.generate_report(pipeline_context, decoding)