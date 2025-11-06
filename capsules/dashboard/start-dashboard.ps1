# Start Capsules Dashboard
Write-Host "Starting Capsules Dashboard..." -ForegroundColor Cyan
Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Yellow
npm install

Write-Host ""
Write-Host "Starting development server..." -ForegroundColor Yellow
Write-Host ""
Write-Host "Dashboard will open at: http://localhost:3000" -ForegroundColor Green
Write-Host "API should be running at: http://localhost:8000" -ForegroundColor Green
Write-Host ""

npm start
