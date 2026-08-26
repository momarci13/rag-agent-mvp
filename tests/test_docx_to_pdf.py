"""Tests for tools/docx_to_pdf.py's LibreOffice-headless conversion path.

Skipped when the `soffice` binary isn't available on this machine, matching
tools/tex.py's shutil.which-gated engine fallback style.
"""

import shutil

import pytest

from agents.risk_schemas import ValidationReport
from tools.docx_report import ValidationWordReportBuilder
from tools.docx_to_pdf import convert_docx_to_pdf

_SOFFICE_AVAILABLE = shutil.which("soffice") is not None or shutil.which("soffice.exe") is not None


def test_unknown_engine_raises_value_error(tmp_path):
    with pytest.raises(ValueError):
        convert_docx_to_pdf(tmp_path / "x.docx", tmp_path, engine="not_a_real_engine")


def test_missing_docx2pdf_returns_none_without_raising(tmp_path):
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
    )
    docx_path = ValidationWordReportBuilder().build(report, tmp_path / "report.docx")
    # docx2pdf is not a declared dependency; if it happens to be installed in
    # this environment the conversion may succeed instead of returning None,
    # so only assert it doesn't raise.
    convert_docx_to_pdf(docx_path, tmp_path, engine="docx2pdf")


@pytest.mark.skipif(not _SOFFICE_AVAILABLE, reason="LibreOffice (soffice) is not installed on this machine")
def test_libreoffice_conversion_produces_a_pdf(tmp_path):
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
    )
    docx_path = ValidationWordReportBuilder().build(report, tmp_path / "report.docx")
    pdf_path = convert_docx_to_pdf(docx_path, tmp_path, engine="libreoffice")
    assert pdf_path is not None
    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 0


def test_libreoffice_conversion_returns_none_when_soffice_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.docx_to_pdf._find_libreoffice", lambda: None)
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
    )
    docx_path = ValidationWordReportBuilder().build(report, tmp_path / "report.docx")
    result = convert_docx_to_pdf(docx_path, tmp_path, engine="libreoffice")
    assert result is None
