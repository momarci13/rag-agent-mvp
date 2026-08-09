"""LaTeX build + citation validator.

Citation validation is important because language models can invent BibTeX
keys. We scrub any \\cite{key} whose key isn't in refs.bib.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


CITE_RE = re.compile(r"\\cite[tp]?\{([^}]+)\}")


def extract_cite_keys(tex: str) -> set[str]:
    keys = set()
    for m in CITE_RE.finditer(tex):
        for k in m.group(1).split(","):
            keys.add(k.strip())
    return keys


def validate_citations(tex: str, available_keys: set[str]) -> tuple[str, list[str]]:
    """Remove any \\cite{k} where k not in available_keys. Return (clean_tex, dropped)."""
    dropped: list[str] = []

    def _fix(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        good = [k for k in keys if k in available_keys]
        bad = [k for k in keys if k not in available_keys]
        dropped.extend(bad)
        if not good:
            return "[CITATION NEEDED]"
        return f"\\cite{{{','.join(good)}}}"

    return CITE_RE.sub(_fix, tex), dropped


def build_pdf(
    tex_source: str,
    output_dir: str | Path,
    bib_source: str = "",
    engine: str = "tectonic",
) -> Path | None:
    """Compile tex_source -> PDF. Returns path to PDF or None on failure."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tex_path = out / "main.tex"
    tex_path.write_text(tex_source, encoding="utf-8")
    if bib_source:
        (out / "refs.bib").write_text(bib_source, encoding="utf-8")

    if shutil.which(engine) is None:
        # fallback
        if engine == "tectonic" and shutil.which("pdflatex"):
            engine = "pdflatex"
        else:
            return None

    try:
        if engine == "tectonic":
            subprocess.run(
                ["tectonic", str(tex_path), "--outdir", str(out)],
                check=True, capture_output=True, text=True, timeout=120,
            )
        else:
            # pdflatex needs two passes + bibtex
            for _ in range(2):
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory",
                     str(out), str(tex_path)],
                    capture_output=True, text=True, timeout=120,
                )
            if bib_source:
                subprocess.run(
                    ["bibtex", str(out / "main")],
                    capture_output=True, text=True, timeout=60,
                )
                subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory",
                     str(out), str(tex_path)],
                    capture_output=True, text=True, timeout=120,
                )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
        return None

    pdf = out / "main.pdf"
    return pdf if pdf.exists() else None


def build_latex_artifact(
    tex: str,
    bib: str,
    out_dir: str | Path = "./output/paper",
) -> tuple[Path, Path | None, list[str]]:
    """End-to-end: validate, compile. Returns (tex_path, pdf_path, dropped_keys)."""
    out = Path(out_dir)
    # Extract keys from bib
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,]+)\s*,", bib))
    clean_tex, dropped = validate_citations(tex, bib_keys)
    pdf = build_pdf(clean_tex, out, bib_source=bib)
    tex_path = out / "main.tex"
    return tex_path, pdf, dropped
