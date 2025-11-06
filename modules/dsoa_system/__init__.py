"""
Ultra Platform - Dynamic Statement of Advice (DSOA) System

Institutional-grade dynamic advisory system aligned with Ultra Platform standards.
"""

from .dsoa_core import (
    DSOASystem,
    ClientProfile,
    FinancialGoal,
    PortfolioAllocation,
    AdvisoryRecommendation,
    MarketConditions,
    GoalOptimizer,
    RealTimeDecisionEngine,
    ComplianceEngine,
    MCPAdvisoryServer,
    GoalType,
    RiskProfile,
    MarketRegime,
    AdvisoryAction,
    ComplianceStatus
)

__version__ = "1.0.0"

__all__ = [
    "DSOASystem",
    "ClientProfile",
    "FinancialGoal",
    "PortfolioAllocation",
    "AdvisoryRecommendation",
    "MarketConditions",
    "GoalOptimizer",
    "RealTimeDecisionEngine",
    "ComplianceEngine",
    "MCPAdvisoryServer",
    "GoalType",
    "RiskProfile",
    "MarketRegime",
    "AdvisoryAction",
    "ComplianceStatus"
]
