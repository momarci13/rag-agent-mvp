"""DOCX -> PDF conversion for the risk-validation report pipeline.

The PDF is generated as a derived artifact of the DOCX (not a fourth
independently-coded template), so the Word and PDF outputs never drift
apart. Default engine is LibreOffice headless (free, scriptable, no
interactive Word/COM state needed from a server process); ``docx2pdf``
(Windows COM automation via a local MS Word install) is supported as an
optional alternative. See configs/config.yaml's ``risk_validation.pdf_via``.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

_LIBREOFFICE_CANDIDATES = ("soffice", "soffice.exe")


def _find_libreoffice() -> str | None:
    for name in _LIBREOFFICE_CANDIDATES:
        path = shutil.which(name)
        if path:
            return path
    return None


def convert_docx_to_pdf(
    docx_path: str | Path,
    output_dir: str | Path,
    *,
    engine: str = "libreoffice",
) -> Path | None:
    """Convert a .docx file to PDF. Returns the PDF path, or None on failure."""

    docx_path = Path(docx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_pdf = output_dir / (docx_path.stem + ".pdf")

    if engine == "libreoffice":
        soffice = _find_libreoffice()
        if soffice is None:
            return None
        try:
            subprocess.run(
                [
                    soffice, "--headless", "--norestore",
                    "--convert-to", "pdf", "--outdir", str(output_dir), str(docx_path),
                ],
                check=True,
                capture_output=True,
                timeout=120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
        return expected_pdf if expected_pdf.exists() else None

    if engine == "docx2pdf":
        try:
            from docx2pdf import convert
        except ImportError:
            return None
        try:
            convert(str(docx_path), str(expected_pdf))
        except Exception:
            return None
        return expected_pdf if expected_pdf.exists() else None

    raise ValueError(f"Unknown PDF conversion engine: {engine!r}")
