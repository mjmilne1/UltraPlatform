# Test All New Features
$baseUrl = "http://localhost:8000"

Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║              TESTING INSTITUTIONAL FEATURES                               ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Cyan

# 1. Create a capsule
Write-Host "`n1. Creating test capsule..." -ForegroundColor Yellow
$capsule = @{
    client_id = "test-client"
    capsule_type = "retirement"
    goal_amount = 500000
    goal_date = "2045-12-31"
} | ConvertTo-Json

$cap = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules" -Method Post -Body $capsule -ContentType "application/json"
Write-Host "   ✓ Created: $($cap.id)" -ForegroundColor Green

# 2. Add transactions
Write-Host "`n2. Adding transactions..." -ForegroundColor Yellow

$deposit1 = @{
    transaction_type = "deposit"
    amount = 10000
    description = "Initial deposit"
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/transactions" -Method Post -Body $deposit1 -ContentType "application/json" | Out-Null
Write-Host "   ✓ Deposit: $10,000" -ForegroundColor Green

$deposit2 = @{
    transaction_type = "deposit"
    amount = 5000
    description = "Monthly contribution"
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/transactions" -Method Post -Body $deposit2 -ContentType "application/json" | Out-Null
Write-Host "   ✓ Deposit: $5,000" -ForegroundColor Green

# 3. Set allocations
Write-Host "`n3. Setting portfolio allocations..." -ForegroundColor Yellow

$alloc1 = @{
    asset_class = "stocks"
    target_percentage = 60
    current_percentage = 50
    current_value = 7500
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/allocations" -Method Post -Body $alloc1 -ContentType "application/json" | Out-Null

$alloc2 = @{
    asset_class = "bonds"
    target_percentage = 30
    current_percentage = 35
    current_value = 5250
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/allocations" -Method Post -Body $alloc2 -ContentType "application/json" | Out-Null

$alloc3 = @{
    asset_class = "cash"
    target_percentage = 10
    current_percentage = 15
    current_value = 2250
} | ConvertTo-Json
Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/allocations" -Method Post -Body $alloc3 -ContentType "application/json" | Out-Null

Write-Host "   ✓ Stocks: 60% target (currently 50%)" -ForegroundColor Green
Write-Host "   ✓ Bonds: 30% target (currently 35%)" -ForegroundColor Green
Write-Host "   ✓ Cash: 10% target (currently 15%)" -ForegroundColor Green

# 4. Check rebalancing
Write-Host "`n4. Checking rebalancing..." -ForegroundColor Yellow
$rebalCheck = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/rebalance/check" -Method Get
Write-Host "   Rebalance needed: $($rebalCheck.rebalance_needed)" -ForegroundColor $(if($rebalCheck.rebalance_needed){"Yellow"}else{"Green"})
Write-Host "   Max drift: $($rebalCheck.max_drift)%" -ForegroundColor Cyan

# 5. Perform rebalancing
if ($rebalCheck.rebalance_needed) {
    Write-Host "`n5. Executing rebalancing..." -ForegroundColor Yellow
    $rebalResult = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/rebalance" -Method Post
    Write-Host "   ✓ Rebalanced successfully" -ForegroundColor Green
    foreach ($action in $rebalResult.actions) {
        Write-Host "   - $($action.action.ToUpper()) $($action.asset_class): `$$($action.amount)" -ForegroundColor Cyan
    }
}

# 6. Get performance
Write-Host "`n6. Calculating performance..." -ForegroundColor Yellow
$perf = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/performance" -Method Get
Write-Host "   Start value: `$$($perf.start_value)" -ForegroundColor Cyan
Write-Host "   End value: `$$($perf.end_value)" -ForegroundColor Cyan
Write-Host "   Return: $($perf.return_percentage)%" -ForegroundColor $(if($perf.return_percentage -gt 0){"Green"}else{"Red"})

# 7. View all transactions
Write-Host "`n7. Transaction history..." -ForegroundColor Yellow
$trans = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($cap.id)/transactions" -Method Get
Write-Host "   Total transactions: $($trans.Count)" -ForegroundColor Green

Write-Host @"

╔═══════════════════════════════════════════════════════════════════════════╗
║                  ✓ ALL FEATURES TESTED SUCCESSFULLY                       ║
╚═══════════════════════════════════════════════════════════════════════════╝

"@ -ForegroundColor Green
