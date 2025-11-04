# UltraPlatform.ps1
# Unified launcher for UltraLedger + UltraCMA

Write-Host "`n" -NoNewline
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    ULTRA PLATFORM                           ║" -ForegroundColor Cyan
Write-Host "║         UltraLedger + UltraCMA Unified Architecture        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "  🏗️  Architecture: Event-Sourced CQRS with Bitemporal Model" -ForegroundColor Gray
Write-Host "  📚  Ledger: github.com/mjmilne1/UltraLedger" -ForegroundColor Gray
Write-Host "  💳  CMA: github.com/mjmilne1/UltraCMA" -ForegroundColor Gray
Write-Host ""

# Load infrastructure
Import-Module .\src\UltraLedger\UltraLedger.Core.psm1 -Force
Initialize-UltraLedger | Out-Null

# Load application
Import-Module .\src\UltraCMA\UltraCMA.Core.psm1 -Force

Write-Host "`n✅ Platform Ready!" -ForegroundColor Green
Write-Host ""
Write-Host "Commands:" -ForegroundColor Yellow
Write-Host "  Start-CMADemo      - Run full demo" -ForegroundColor Gray
Write-Host "  New-CMACustomer    - Create customer" -ForegroundColor Gray
Write-Host "  New-CMAPayment     - Process payment" -ForegroundColor Gray
Write-Host "  Get-CMAMetrics     - View metrics" -ForegroundColor Gray
Write-Host "  Get-UltraEvents    - Query event store" -ForegroundColor Gray
Write-Host ""
