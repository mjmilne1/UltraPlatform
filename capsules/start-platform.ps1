# Master Startup Script - Wire Up Complete Platform
$TargetDir = "C:\Users\mjmil\UltraReinforcementLearning\UltraPlatform\capsules"

Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║                    CAPSULES PLATFORM - MASTER STARTUP                     ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

cd $TargetDir

Write-Host "Checking system status..." -ForegroundColor Yellow

$apiRunning = $false
$dashboardRunning = $false

try {
    $null = Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get -TimeoutSec 2
    $apiRunning = $true
    Write-Host "  ✓ API already running" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ API not running" -ForegroundColor Yellow
}

try {
    $null = Invoke-WebRequest -Uri "http://localhost:3001/" -TimeoutSec 2
    $dashboardRunning = $true
    Write-Host "  ✓ Dashboard already running" -ForegroundColor Green
} catch {
    Write-Host "  ⚠ Dashboard not running" -ForegroundColor Yellow
}

Write-Host ""

if (-not $apiRunning) {
    Write-Host "Starting Flask API..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$TargetDir'; .\venv\Scripts\Activate.ps1; python src\services\capsule_service\app.py"
    Write-Host "  ✓ API starting" -ForegroundColor Green
    Start-Sleep -Seconds 5
}

if (-not $dashboardRunning) {
    Write-Host "Starting React Dashboard..." -ForegroundColor Cyan
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$TargetDir\dashboard'; npm start"
    Write-Host "  ✓ Dashboard starting" -ForegroundColor Green
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "Waiting for services..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "Testing connections..." -ForegroundColor Cyan

try {
    $apiResponse = Invoke-RestMethod -Uri "http://localhost:8000/" -Method Get
    Write-Host "  ✓ API: $($apiResponse.service) v$($apiResponse.version)" -ForegroundColor Green
} catch {
    Write-Host "  ✗ API not responding" -ForegroundColor Red
}

try {
    $capsules = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/capsules" -Method Get
    Write-Host "  ✓ Database: $($capsules.Count) capsules" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Database error" -ForegroundColor Red
}

Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║                    ✓ PLATFORM READY                                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green

Write-Host "Access Points:" -ForegroundColor Cyan
Write-Host "  Dashboard:  http://localhost:3001" -ForegroundColor White
Write-Host "  API:        http://localhost:8000" -ForegroundColor White
Write-Host ""

Start-Process "http://localhost:3001"
