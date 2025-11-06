# ANYA DEPLOYMENT GUIDE
======================

## Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (EKS, GKE, AKS, or local)
- kubectl CLI
- Helm (optional)

## Quick Start - Local Development

### 1. Using Docker Compose
```bash
# Set environment variables
export OPENAI_API_KEY=your_key_here
export POSTGRES_PASSWORD=secure_password

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f anya-api

# Access API
curl http://localhost:8000/health

# Stop services
docker-compose down
```

### 2. Build Docker Image
```bash
# Build image
docker build -t anya:latest .

# Run container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_key \
  anya:latest

# Test
curl http://localhost:8000/health
```

## Production Deployment - Kubernetes

### 1. Create Namespace
```bash
kubectl apply -f deployment/kubernetes/namespace.yaml
```

### 2. Create Secrets
```bash
kubectl create secret generic anya-secrets \
  --from-literal=openai-api-key=YOUR_KEY \
  --from-literal=postgres-password=YOUR_PASSWORD \
  --from-literal=jwt-secret=YOUR_SECRET \
  -n anya
```

### 3. Apply Configuration
```bash
kubectl apply -f deployment/kubernetes/configmap.yaml
```

### 4. Deploy Application
```bash
kubectl apply -f deployment/kubernetes/deployment.yaml
```

### 5. Create Ingress
```bash
kubectl apply -f deployment/kubernetes/ingress.yaml
```

### 6. Verify Deployment
```bash
# Check pods
kubectl get pods -n anya

# Check services
kubectl get svc -n anya

# Check deployment status
kubectl rollout status deployment/anya-api -n anya

# View logs
kubectl logs -f deployment/anya-api -n anya
```

## Health Checks
```bash
# Liveness probe
curl http://your-domain/health

# Readiness probe
curl http://your-domain/health

# Metrics
curl http://your-domain/metrics
```

## Monitoring

### Prometheus
```bash
# Port forward Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n anya

# Access: http://localhost:9090
```

### Grafana
```bash
# Port forward Grafana
kubectl port-forward svc/grafana 3000:3000 -n anya

# Access: http://localhost:3000
# Default credentials: admin/admin
```

## Scaling

### Manual Scaling
```bash
kubectl scale deployment/anya-api --replicas=5 -n anya
```

### Auto-scaling (HPA)
```bash
# Already configured in deployment.yaml
# Scales between 3-10 pods based on CPU/memory
kubectl get hpa -n anya
```

## Rolling Updates
```bash
# Update image
kubectl set image deployment/anya-api \
  anya-api=anya:v2.0.0 -n anya

# Monitor rollout
kubectl rollout status deployment/anya-api -n anya

# Rollback if needed
kubectl rollout undo deployment/anya-api -n anya
```

## Troubleshooting

### Check Pod Status
```bash
kubectl describe pod <pod-name> -n anya
```

### View Logs
```bash
kubectl logs <pod-name> -n anya --tail=100
```

### Execute Commands in Pod
```bash
kubectl exec -it <pod-name> -n anya -- /bin/bash
```

### Check Events
```bash
kubectl get events -n anya --sort-by='.lastTimestamp'
```

## CI/CD Pipeline

The GitHub Actions pipeline automatically:

1. Runs tests on PR
2. Scans for security vulnerabilities
3. Builds Docker image on merge to main
4. Deploys to Kubernetes cluster
5. Runs smoke tests

## Environment Variables

Required:
- OPENAI_API_KEY: OpenAI API key
- POSTGRES_PASSWORD: Database password
- JWT_SECRET: JWT signing secret

Optional:
- ENVIRONMENT: production/staging/development
- LOG_LEVEL: info/debug/warning/error
- REDIS_HOST: Redis hostname
- POSTGRES_HOST: PostgreSQL hostname

## Security Best Practices

1. Never commit secrets to version control
2. Use Kubernetes secrets or sealed-secrets
3. Enable network policies
4. Use RBAC for access control
5. Regular security scans
6. Keep images updated

## Backup & Disaster Recovery

### Database Backup
```bash
kubectl exec -n anya anya-postgres-0 -- \
  pg_dump -U anya anya > backup.sql
```

### Restore Database
```bash
kubectl exec -i -n anya anya-postgres-0 -- \
  psql -U anya anya < backup.sql
```

## Performance Tuning

1. Adjust resource requests/limits
2. Configure HPA thresholds
3. Enable caching
4. Optimize database queries
5. Use CDN for static assets

## Support

For issues or questions:
- GitHub Issues: [your-repo]/issues
- Documentation: [your-docs-url]
- Email: support@example.com
