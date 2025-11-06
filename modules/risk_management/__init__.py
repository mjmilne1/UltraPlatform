"""
Ultra Platform - Enterprise Risk Management Framework

Business-critical risk management:
- Risk identification and assessment
- Market risk analysis (VaR, stress testing)
- Operational risk controls
- Incident response (<15 min detection, <4hr resolution)
- Real-time risk monitoring
- Comprehensive reporting

Standards:
- ISO 31000 (Risk Management)
- COSO ERM Framework
- Basel III/IV
- APRA CPS 220

Version: 1.0.0
"""

from .risk_engine import (
    RiskManagementFramework,
    RiskIdentificationEngine,
    MarketRiskAnalyzer,
    RiskControlManager,
    IncidentResponseManager,
    RiskReportingEngine,
    RiskCategory,
    RiskLevel,
    RiskStatus,
    ControlEffectiveness,
    IncidentSeverity,
    RiskIdentification,
    RiskControl,
    RiskIncident,
    MarketRiskMetrics,
    RiskReport
)

__version__ = "1.0.0"

__all__ = [
    "RiskManagementFramework",
    "RiskIdentificationEngine",
    "MarketRiskAnalyzer",
    "RiskControlManager",
    "IncidentResponseManager",
    "RiskReportingEngine",
    "RiskCategory",
    "RiskLevel",
    "RiskStatus",
    "ControlEffectiveness",
    "IncidentSeverity",
    "RiskIdentification",
    "RiskControl",
    "RiskIncident",
    "MarketRiskMetrics",
    "RiskReport"
]
