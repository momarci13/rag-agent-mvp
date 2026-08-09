$ErrorActionPreference = "Stop"

Write-Host "[1/3] Creating Python environment..."
if (-Not (Test-Path ".venv")) {
    python -m venv .venv
}

$python = Join-Path $PWD ".venv\Scripts\python.exe"
Write-Host "[2/3] Installing dependencies (no local model runtime)..."
& $python -m pip install --upgrade pip
& $python -m pip install -r requirements.txt

Write-Host "[3/3] Checking OpenRouter..."
$baseUrl = if ($env:OPENROUTER_BASE_URL) {
    $env:OPENROUTER_BASE_URL.TrimEnd("/")
} else {
    "https://openrouter.ai/api/v1"
}
if (-Not $env:OPENROUTER_API_KEY) {
    Write-Host "  Set OPENROUTER_API_KEY to use hosted free models." -ForegroundColor Yellow
} else {
    $headers = @{ Authorization = "Bearer $($env:OPENROUTER_API_KEY)" }
    try {
        $null = Invoke-WebRequest -Uri "$baseUrl/key" -Headers $headers -UseBasicParsing -TimeoutSec 8
        Write-Host "  OpenRouter is reachable at $baseUrl" -ForegroundColor Green
    } catch {
        Write-Host "  OpenRouter is not reachable or authorized at $baseUrl" -ForegroundColor Yellow
    }
}

Write-Host "Setup complete. Run: .\.venv\Scripts\python.exe run.py --healthcheck"
