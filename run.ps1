# Start backend (FastAPI) and frontend (Vite) in parallel for local dev on Windows.
# Usage: .\run.ps1   — Ctrl+C in each spawned window to stop, or close the windows.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path ".venv")) {
    Write-Host "Creating Python virtualenv..."
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install -q -r backend\requirements.txt

Write-Host "Starting FastAPI on :8000..."
Start-Process -FilePath ".venv\Scripts\python.exe" `
    -ArgumentList "-m", "uvicorn", "backend.main:app", "--reload", "--port", "8000" `
    -WorkingDirectory $root

if (-not (Test-Path "frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..."
    Push-Location frontend
    npm install
    Pop-Location
}

Write-Host "Starting Vite on :5173..."
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "npm run dev" `
    -WorkingDirectory (Join-Path $root "frontend")

Write-Host ""
Write-Host "Backend:  http://localhost:8000/docs"
Write-Host "Frontend: http://localhost:5173"
