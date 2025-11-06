@echo off
echo.
echo ========================================
echo   STARTING ANYA AI ASSISTANT
echo ========================================
echo.

echo [1/3] Checking Python...
python --version
if errorlevel 1 (
    echo ERROR: Python not found!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing dependencies...
pip install fastapi uvicorn pydantic -q

echo.
echo [3/3] Starting API Server...
start "Anya API" cmd /k "python anya_api.py"

timeout /t 3 /nobreak > nul

echo.
echo Opening Web Interface...
start "" "frontend/index.html"

echo.
echo ========================================
echo   ANYA IS RUNNING!
echo ========================================
echo.
echo   API:     http://localhost:8000
echo   Docs:    http://localhost:8000/docs
echo   UI:      frontend/index.html
echo.
echo Close this window or press Ctrl+C to stop
echo.
pause
