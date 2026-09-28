#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Creating Python environment..."
[ -d .venv ] || python3 -m venv .venv
PYTHON=".venv/bin/python"

echo "[2/3] Installing dependencies..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo "[3/3] Checking configuration..."
[ -n "${OPENAI_API_KEY:-}" ] || echo "  Set OPENAI_API_KEY (billed OpenAI account)."
[ -n "${STUDIO_API_TOKEN:-}" ] || echo "  Set STUDIO_API_TOKEN to enable starting projects and sign-off."
command -v soffice >/dev/null || echo "  LibreOffice (soffice) not found: documents will be DOCX only."

echo "Setup complete. Run: .venv/bin/uvicorn server:app  (then open http://127.0.0.1:8000)"
