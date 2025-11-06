# Test Capsule Service API
# Make sure service is running first!

$baseUrl = "http://localhost:8000"

Write-Host "Testing Capsules API..." -ForegroundColor Cyan
Write-Host ""

# Test 1: Health check
Write-Host "1. Health Check" -ForegroundColor Yellow
$response = Invoke-RestMethod -Uri "$baseUrl/health" -Method Get
Write-Host "   Status: $($response.status)" -ForegroundColor Green

# Test 2: Create capsule
Write-Host "2. Create Capsule" -ForegroundColor Yellow
$capsule = @{
    client_id = "client-123"
    capsule_type = "retirement"
    goal_amount = 1000000
    goal_date = "2050-12-31"
} | ConvertTo-Json

$newCapsule = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules" -Method Post -Body $capsule -ContentType "application/json"
Write-Host "   Created: $($newCapsule.id)" -ForegroundColor Green

# Test 3: Get capsule
Write-Host "3. Get Capsule" -ForegroundColor Yellow
$retrieved = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules/$($newCapsule.id)" -Method Get
Write-Host "   Type: $($retrieved.capsule_type)" -ForegroundColor Green

# Test 4: List capsules
Write-Host "4. List Capsules" -ForegroundColor Yellow
$allCapsules = Invoke-RestMethod -Uri "$baseUrl/api/v1/capsules" -Method Get
Write-Host "   Total: $($allCapsules.Count)" -ForegroundColor Green

Write-Host ""
Write-Host "✓ All tests passed!" -ForegroundColor Green
