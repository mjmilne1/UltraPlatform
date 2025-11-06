"""
Ultra Platform - Enterprise Infrastructure Framework

Institutional-grade infrastructure management:
- Multi-region AWS cloud architecture
- Microservices orchestration (Kubernetes/EKS)
- Multi-database architecture
- Performance monitoring
- CI/CD automation

Performance Targets:
- API Response: p95 < 200ms, p99 < 500ms
- Database Query: p95 < 100ms
- Concurrent Users: 100,000+
- Throughput: 10,000+ TPS

Version: 1.0.0
"""

from .infrastructure_engine import (
    InfrastructureFramework,
    CloudInfrastructureManager,
    MicroservicesOrchestrator,
    DatabaseArchitecture,
    PerformanceMonitor,
    CICDPipeline,
    CloudProvider,
    DeploymentEnvironment,
    ServiceStatus,
    DatabaseType,
    AWSRegion,
    MicroserviceConfig,
    DatabaseConfig,
    PerformanceReport
)

__version__ = "1.0.0"

__all__ = [
    "InfrastructureFramework",
    "CloudInfrastructureManager",
    "MicroservicesOrchestrator",
    "DatabaseArchitecture",
    "PerformanceMonitor",
    "CICDPipeline",
    "CloudProvider",
    "DeploymentEnvironment",
    "ServiceStatus",
    "DatabaseType",
    "AWSRegion",
    "MicroserviceConfig",
    "DatabaseConfig",
    "PerformanceReport"
]
