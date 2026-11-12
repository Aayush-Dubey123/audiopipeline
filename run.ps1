Write-Host "========================================================" -ForegroundColor Cyan
Write-Host " Starting AudioPipeline Backend and Frontend..." -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

$Root = $PSScriptRoot
if (-not $Root) { $Root = Get-Location }

# 1. Detect Python Executable (single project environment: backend\.venv)
$PythonExe = "$Root\backend\.venv\Scripts\python.exe"
if (-not (Test-Path $PythonExe)) {
    $PythonExe = "python"
}

# 2. Start Backend Server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root'; Write-Host '--- AudioPipeline Backend (FastAPI) ---' -ForegroundColor Green; & '$PythonExe' -m uvicorn core.apis.api:app --app-dir backend --host 127.0.0.1 --port 8000"

# 2. Start Frontend Server
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$Root\frontend'; Write-Host '--- AudioPipeline Frontend (Vite/React) ---' -ForegroundColor Blue; npm run dev"

Write-Host "`nBoth services launched in separate windows!" -ForegroundColor Green
Write-Host "Frontend UI:  http://localhost:3000" -ForegroundColor Yellow
Write-Host "Backend API:  http://127.0.0.1:8000" -ForegroundColor Yellow
Write-Host "API Docs:     http://127.0.0.1:8000/docs`n" -ForegroundColor Yellow

