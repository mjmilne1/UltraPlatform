"""
Ultra Platform - Real-Time Portfolio Management System

Institutional-grade portfolio management with:
- Dynamic rebalancing (<2 second target)
- Drift detection (<100ms target)
- Tax optimization (75-125 bps annually)
- Risk management (VaR, CVaR, stress testing)
- Performance analytics (Sharpe, Sortino, attribution)
- Transaction management (<10 bps cost target)

Based on: 4. Real-Time Portfolio Management specification
Version: 1.0.0
"""

from .rebalancing_engine import (
    # Core classes
    RebalancingEngine,
    DriftDetector,
    TaxOptimizer,
    RiskManager,
    PerformanceAnalytics,
    TransactionManager,
    
    # Data classes
    Position,
    TaxLot,
    RebalanceTrigger,
    RebalanceProposal,
    
    # Enums
    RebalanceTriggerType,
    TaxLotMethod,
    ExecutionAlgorithm
)

__version__ = "1.0.0"

__all__ = [
    # Core classes
    "RebalancingEngine",
    "DriftDetector",
    "TaxOptimizer",
    "RiskManager",
    "PerformanceAnalytics",
    "TransactionManager",
    
    # Data classes
    "Position",
    "TaxLot",
    "RebalanceTrigger",
    "RebalanceProposal",
    
    # Enums
    "RebalanceTriggerType",
    "TaxLotMethod",
    "ExecutionAlgorithm"
]
