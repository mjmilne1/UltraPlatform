# Capsule Service

Flask-based REST API for capsule management.

## Start the Service
```powershell
cd capsules
.\run-service.ps1
```

Service will start at: **http://localhost:8000**

## Test the API

In another terminal:
```powershell
cd capsules
.\test-service.ps1
```

## API Endpoints

- **GET /**  - Service info
- **GET /health** - Health check
- **GET /api/v1/capsules** - List capsules
- **GET /api/v1/capsules/{id}** - Get capsule
- **POST /api/v1/capsules** - Create capsule
- **PUT /api/v1/capsules/{id}** - Update capsule
- **DELETE /api/v1/capsules/{id}** - Delete capsule

## Example: Create Capsule
```powershell
$capsule = @{
    client_id = "client-123"
    capsule_type = "retirement"
    goal_amount = 1000000
    goal_date = "2050-12-31"
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://localhost:8000/api/v1/capsules" `
    -Method Post `
    -Body $capsule `
    -ContentType "application/json"
```
