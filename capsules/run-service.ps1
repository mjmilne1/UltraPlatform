# Run Capsule Service
# Location: C:\Users\mjmil\UltraReinforcementLearning\UltraPlatform\capsules

$venvPath = Join-Path (Get-Location) "venv\Scripts\Activate.ps1"
& $venvPath

Write-Host "Starting Capsules Service..." -ForegroundColor Cyan
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop" -ForegroundColor Yellow
Write-Host ""

python src\services\capsule_service\app.py
