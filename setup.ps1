$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating Python environment..."
if (-Not (Test-Path ".venv")) { python -m venv .venv }
$python = Join-Path $PWD ".venv\Scripts\python.exe"

Write-Host "[2/3] Installing dependencies..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "[3/3] Checking configuration..."
if (-Not $env:OPENAI_API_KEY) { Write-Host "  Set OPENAI_API_KEY (billed OpenAI account)." -ForegroundColor Yellow }
if (-Not $env:STUDIO_API_TOKEN) { Write-Host "  Set STUDIO_API_TOKEN to enable starting projects and sign-off." -ForegroundColor Yellow }
if (-Not (Get-Command soffice -ErrorAction SilentlyContinue)) {
    Write-Host "  LibreOffice (soffice) not found: documents will be DOCX only, or set studio.pdf_via to docx2pdf and pip install docx2pdf." -ForegroundColor Yellow
}

Write-Host "Setup complete. Run: .\.venv\Scripts\uvicorn.exe server:app  (then open http://127.0.0.1:8000)"
