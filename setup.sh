#!/usr/bin/env bash
set -euo pipefail

echo "[1/3] Creating Python environment..."
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

PYTHON=".venv/bin/python"
echo "[2/3] Installing dependencies (no local model runtime)..."
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r requirements.txt

echo "[3/3] Checking OpenRouter..."
BASE_URL="${OPENROUTER_BASE_URL:-https://openrouter.ai/api/v1}"
BASE_URL="${BASE_URL%/}"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  echo "  Set OPENROUTER_API_KEY to use hosted free models."
elif curl -fsS --max-time 8 -H "Authorization: Bearer ${OPENROUTER_API_KEY}" "$BASE_URL/key" >/dev/null; then
  echo "  OpenRouter is reachable at $BASE_URL"
else
  echo "  OpenRouter is not reachable or authorized at $BASE_URL"
fi

echo "Setup complete. Run: .venv/bin/python run.py --healthcheck"
