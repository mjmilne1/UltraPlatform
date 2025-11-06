# AWS Deployment Script for Capsules Platform
# Deploys to AWS ECS with RDS PostgreSQL

# Prerequisites:
# - AWS CLI installed and configured
# - Docker installed
# - AWS account with appropriate permissions

# Configuration
$AWS_REGION = "us-east-1"
$ECR_REPO = "capsules-platform"
$ECS_CLUSTER = "capsules-cluster"
$ECS_SERVICE = "capsules-service"
$TASK_FAMILY = "capsules-task"

Write-Host "AWS Deployment for Capsules Platform" -ForegroundColor Cyan
Write-Host ""

# 1. Build Docker image
Write-Host "1. Building Docker image..." -ForegroundColor Yellow
docker build -t capsules-platform:latest .
Write-Host "   ✓ Image built" -ForegroundColor Green

# 2. Create ECR repository (if not exists)
Write-Host "2. Setting up ECR repository..." -ForegroundColor Yellow
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>$null
if ($LASTEXITCODE -ne 0) {
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION
    Write-Host "   ✓ ECR repository created" -ForegroundColor Green
} else {
    Write-Host "   ✓ ECR repository exists" -ForegroundColor Green
}

# 3. Get ECR login
Write-Host "3. Logging into ECR..." -ForegroundColor Yellow
$ECR_URI = (aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION --query 'repositories[0].repositoryUri' --output text)
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI.Split('/')[0]
Write-Host "   ✓ Logged in" -ForegroundColor Green

# 4. Tag and push image
Write-Host "4. Pushing to ECR..." -ForegroundColor Yellow
docker tag capsules-platform:latest "${ECR_URI}:latest"
docker push "${ECR_URI}:latest"
Write-Host "   ✓ Image pushed" -ForegroundColor Green

# 5. Create/Update ECS Task Definition
Write-Host "5. Updating ECS task definition..." -ForegroundColor Yellow
$TASK_DEF = @"
{
  "family": "$TASK_FAMILY",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "containerDefinitions": [
    {
      "name": "capsules-api",
      "image": "${ECR_URI}:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/capsules",
          "awslogs-region": "$AWS_REGION",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import requests; requests.get('http://localhost:8000/health')\""],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
"@

$TASK_DEF | Out-File -FilePath "task-definition.json" -Encoding utf8
aws ecs register-task-definition --cli-input-json file://task-definition.json --region $AWS_REGION
Write-Host "   ✓ Task definition updated" -ForegroundColor Green

# 6. Update ECS service
Write-Host "6. Updating ECS service..." -ForegroundColor Yellow
aws ecs update-service --cluster $ECS_CLUSTER --service $ECS_SERVICE --task-definition $TASK_FAMILY --force-new-deployment --region $AWS_REGION
Write-Host "   ✓ Service updated" -ForegroundColor Green

Write-Host ""
Write-Host "✓ Deployment complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Set up RDS PostgreSQL database" -ForegroundColor White
Write-Host "  2. Configure load balancer" -ForegroundColor White
Write-Host "  3. Set up SSL certificate" -ForegroundColor White
Write-Host "  4. Configure environment variables in ECS" -ForegroundColor White
