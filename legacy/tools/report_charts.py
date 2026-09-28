"""Shared chart-rendering helper for the risk-validation report/deck builders.

Avoids duplicating the base64/PNG plot logic that already exists in
tools/report.py::ReportBuilder for the LaTeX report path.
"""
from __future__ import annotations

import io

import matplotlib

matplotlib.use("Agg")  # headless: no display needed on a server process
import matplotlib.pyplot as plt


def render_metric_chart(data: dict[str, float], title: str) -> bytes:
    """Render a simple labeled bar chart of numeric metrics and return PNG bytes."""

    labels = list(data.keys())
    values = [float(v) for v in data.values()]

    fig, ax = plt.subplots(figsize=(6.0, 3.2), dpi=150)
    ax.bar(labels, values, color="#4472C4")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()
