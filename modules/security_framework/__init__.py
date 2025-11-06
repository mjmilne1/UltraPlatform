"""
Ultra Platform - Enterprise Security and Data Protection Framework

BUSINESS CRITICAL - Military-grade security implementation:
- Multi-layer security architecture
- AES-256 encryption at rest, TLS 1.3 in transit
- IAM with MFA, biometric, SSO
- RBAC and ABAC authorization
- 24/7 SOC with real-time threat detection
- Business continuity (RTO < 2h, RPO < 15min)

Based on: Section 8 - Security and Data Protection
Security Standards:
- OWASP Top 10 compliance
- SOC 2 Type II
- ISO 27001
- Australian Privacy Principles (APPs)
- GDPR alignment

Version: 1.0.0
"""

from .security_engine import (
    # Core framework
    SecurityFramework,
    EncryptionManager,
    AuthenticationManager,
    AuthorizationManager,
    SecurityMonitor,
    BusinessContinuityManager,
    
    # Data classes
    UserIdentity,
    AccessToken,
    SecurityEvent,
    SecurityIncident,
    EncryptionKey,
    
    # Enums
    SecurityLevel,
    AuthenticationMethod,
    IncidentSeverity,
    AccessDecision
)

__version__ = "1.0.0"

__all__ = [
    # Core framework
    "SecurityFramework",
    "EncryptionManager",
    "AuthenticationManager",
    "AuthorizationManager",
    "SecurityMonitor",
    "BusinessContinuityManager",
    
    # Data classes
    "UserIdentity",
    "AccessToken",
    "SecurityEvent",
    "SecurityIncident",
    "EncryptionKey",
    
    # Enums
    "SecurityLevel",
    "AuthenticationMethod",
    "IncidentSeverity",
    "AccessDecision"
]
