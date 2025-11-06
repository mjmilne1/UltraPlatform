# Test Docker Deployment Locally

Write-Host "Testing Docker Deployment..." -ForegroundColor Cyan
Write-Host ""

# Build image
Write-Host "1. Building Docker image..." -ForegroundColor Yellow
docker build -t capsules-platform:local .
Write-Host "   ✓ Built" -ForegroundColor Green

# Start services
Write-Host "2. Starting services with docker-compose..." -ForegroundColor Yellow
docker-compose up -d
Write-Host "   ✓ Services started" -ForegroundColor Green

# Wait for services to be ready
Write-Host "3. Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Test health endpoint
Write-Host "4. Testing health endpoint..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost/health" -Method Get
    if ($response.status -eq "healthy") {
        Write-Host "   ✓ Health check passed" -ForegroundColor Green
    }
} catch {
    Write-Host "   ✗ Health check failed" -ForegroundColor Red
}

# Test API
Write-Host "5. Testing API..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost/" -Method Get
    Write-Host "   ✓ API responding" -ForegroundColor Green
    Write-Host "   Service: $($response.service)" -ForegroundColor Cyan
    Write-Host "   Version: $($response.version)" -ForegroundColor Cyan
} catch {
    Write-Host "   ✗ API test failed" -ForegroundColor Red
}

Write-Host ""
Write-Host "Services running:" -ForegroundColor Green
Write-Host "  API:      http://localhost" -ForegroundColor White
Write-Host "  Direct:   http://localhost:8000" -ForegroundColor White
Write-Host "  Postgres: localhost:5432" -ForegroundColor White
Write-Host ""
Write-Host "View logs:   docker-compose logs -f" -ForegroundColor Cyan
Write-Host "Stop:        docker-compose down" -ForegroundColor Cyan
Write-Host "Clean:       docker-compose down -v" -ForegroundColor Cyan
