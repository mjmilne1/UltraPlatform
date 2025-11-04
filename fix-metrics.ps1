function Get-CMAMetrics {
    # Fixed volume calculation
    $totalVolume = 0
    if ($Global:CMAPayments.Count -gt 0) {
        foreach ($payment in $Global:CMAPayments) {
            $totalVolume += $payment.Amount
        }
    }
    
    $assets = 0
    $liabilities = 0
    
    foreach ($account in $Global:UltraAccounts.Values) {
        if ($account.AccountType -eq "Asset") {
            $assets += $account.Balance
        }
        if ($account.AccountType -eq "Liability") {
            $liabilities += $account.Balance
        }
    }
    
    Write-Host "`n?? CMA Platform Metrics" -ForegroundColor Cyan
    Write-Host "------------------------------" -ForegroundColor Cyan
    Write-Host "Customers:     $($Global:CMACustomers.Count)" -ForegroundColor Gray
    Write-Host "Accounts:      $($Global:CMAAccounts.Count)" -ForegroundColor Gray
    Write-Host "Payments:      $($Global:CMAPayments.Count)" -ForegroundColor Gray
    Write-Host "Total Volume:  $totalVolume AUD" -ForegroundColor Gray
    Write-Host ""
    Write-Host "Balance Sheet:" -ForegroundColor Yellow
    Write-Host "  Assets:      $assets AUD" -ForegroundColor Gray
    Write-Host "  Liabilities: $liabilities AUD" -ForegroundColor Gray
    Write-Host "  Balanced:    $(if ([Math]::Abs($assets - $liabilities) -lt 0.01) {'? Yes'} else {'? No'})" -ForegroundColor Gray
    
    return @{
        Customers = $Global:CMACustomers.Count
        Accounts = $Global:CMAAccounts.Count
        Payments = $Global:CMAPayments.Count
        Volume = $totalVolume
        Assets = $assets
        Liabilities = $liabilities
        Balanced = ([Math]::Abs($assets - $liabilities) -lt 0.01)
    }
}
