@echo off
echo Starting AudioPipeline Backend and Frontend...

rem Start Backend Server
start "AudioPipeline Backend" cmd /k "cd /d %~dp0 && .venv\Scripts\python.exe -m uvicorn core.apis.api:app --app-dir backend --host 127.0.0.1 --port 8000"

rem Start Frontend Server
start "AudioPipeline Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================================
echo  AudioPipeline Services Launched!
echo ========================================================
echo  Frontend UI:  http://localhost:3000
echo  Backend API:  http://127.0.0.1:8000
echo  API Docs:     http://127.0.0.1:8000/docs
echo ========================================================
echo.

