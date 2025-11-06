# ============================================================================
# ANYA MAKEFILE - COMMON COMMANDS
# ============================================================================

.PHONY: help build run test deploy clean

help:
@echo "Anya Deployment Commands:"
@echo "  make build         - Build Docker image"
@echo "  make run           - Run with Docker Compose"
@echo "  make test          - Run tests"
@echo "  make deploy-k8s    - Deploy to Kubernetes"
@echo "  make clean         - Clean up containers"
@echo "  make logs          - View logs"

build:
docker build -t anya:latest .

run:
docker-compose up -d

test:
pytest modules/anya/tests/ -v

deploy-k8s:
kubectl apply -f deployment/kubernetes/namespace.yaml
kubectl apply -f deployment/kubernetes/configmap.yaml
kubectl apply -f deployment/kubernetes/deployment.yaml

clean:
docker-compose down -v
docker system prune -f

logs:
docker-compose logs -f anya-api

shell:
docker-compose exec anya-api /bin/bash
