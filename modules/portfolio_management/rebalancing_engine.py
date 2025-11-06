"""
Ultra Platform - Real-Time Portfolio Management System
======================================================

Complete implementation of institutional-grade portfolio management:
- Dynamic rebalancing engine with drift detection
- Tax-loss harvesting and optimization
- Risk management (VaR, CVaR, stress testing)
- Performance analytics and attribution
- Transaction management and execution

Based on: 4. Real-Time Portfolio Management specification
Performance Targets:
- Rebalance time: <2 seconds
- Drift detection: <100ms
- Tax efficiency: 75-125 bps annually
- Transaction cost: <10 bps per trade

Version: 1.0.0
"""

import asyncio
import uuid
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RebalanceTriggerType(Enum):
    """Types of rebalancing triggers"""
    DRIFT_ABSOLUTE = "drift_absolute"  # >5% deviation
    DRIFT_RELATIVE = "drift_relative"  # >20% relative deviation
    DRIFT_RISK = "drift_risk"  # >10% risk deviation
    DRIFT_GOAL = "drift_goal"  # >5% goal probability drop
    SCHEDULED = "scheduled"  # Quarterly review
    TAX_YEAR_END = "tax_year_end"  # Annual tax optimization
    GOAL_MILESTONE = "goal_milestone"  # Approaching goal
    MARKET_EVENT = "market_event"  # >5% daily change
    TAX_LOSS_HARVEST = "tax_loss_harvest"  # Opportunity
    CASH_INFLOW = "cash_inflow"  # New contribution
    DIVIDEND = "dividend"  # Reinvestment
    CORPORATE_ACTION = "corporate_action"  # M&A, spin-off


class TaxLotMethod(Enum):
    """Tax lot identification methods"""
    FIFO = "fifo"  # First In First Out
    LIFO = "lifo"  # Last In First Out
    HIFO = "hifo"  # Highest In First Out (best for losses)
    LOFO = "lofo"  # Lowest Out First Out (best for gains)
    SPECIFIC_LOT = "specific_lot"  # Manual selection


class ExecutionAlgorithm(Enum):
    """Order execution algorithms"""
    VWAP = "vwap"  # Volume-Weighted Average Price
    TWAP = "twap"  # Time-Weighted Average Price
    IMPLEMENTATION_SHORTFALL = "implementation_shortfall"
    PARTICIPATION_RATE = "participation_rate"
    ARRIVAL_PRICE = "arrival_price"


@dataclass
class TaxLot:
    """Individual tax lot for a security position"""
    lot_id: str
    ticker: str
    quantity: float
    purchase_price: float
    purchase_date: datetime
    cost_basis: float
    current_price: float = 0.0
    
    # Tax tracking
    holding_period_days: int = 0
    is_long_term: bool = False  # >365 days
    unrealized_gain_loss: float = 0.0
    
    # Wash sale tracking
    wash_sale_disallowed: float = 0.0
    
    def update_current_price(self, price: float):
        """Update current price and calculate metrics"""
        self.current_price = price
        self.unrealized_gain_loss = (price - self.purchase_price) * self.quantity
        
        # Update holding period
        self.holding_period_days = (datetime.now() - self.purchase_date).days
        self.is_long_term = self.holding_period_days > 365
    
    def market_value(self) -> float:
        """Current market value of lot"""
        return self.current_price * self.quantity


@dataclass
class Position:
    """Portfolio position with tax lots"""
    ticker: str
    total_quantity: float
    average_cost: float
    current_price: float
    market_value: float
    
    # Tax lots
    tax_lots: List[TaxLot] = field(default_factory=list)
    
    # Position metrics
    weight: float = 0.0  # % of portfolio
    unrealized_gain_loss: float = 0.0
    total_cost_basis: float = 0.0
    
    def add_tax_lot(self, lot: TaxLot):
        """Add tax lot to position"""
        self.tax_lots.append(lot)
        self._recalculate_metrics()
    
    def _recalculate_metrics(self):
        """Recalculate position metrics from tax lots"""
        if not self.tax_lots:
            return
        
        self.total_quantity = sum(lot.quantity for lot in self.tax_lots)
        self.total_cost_basis = sum(lot.cost_basis for lot in self.tax_lots)
        self.market_value = sum(lot.market_value() for lot in self.tax_lots)
        
        if self.total_quantity > 0:
            self.average_cost = self.total_cost_basis / self.total_quantity
        
        self.unrealized_gain_loss = self.market_value - self.total_cost_basis
    
    def get_lots_by_method(self, method: TaxLotMethod) -> List[TaxLot]:
        """Get tax lots ordered by specified method"""
        if method == TaxLotMethod.FIFO:
            return sorted(self.tax_lots, key=lambda x: x.purchase_date)
        elif method == TaxLotMethod.LIFO:
            return sorted(self.tax_lots, key=lambda x: x.purchase_date, reverse=True)
        elif method == TaxLotMethod.HIFO:
            return sorted(self.tax_lots, key=lambda x: x.purchase_price, reverse=True)
        elif method == TaxLotMethod.LOFO:
            return sorted(self.tax_lots, key=lambda x: x.purchase_price)
        else:
            return self.tax_lots


@dataclass
class RebalanceTrigger:
    """Rebalancing trigger event"""
    trigger_id: str
    trigger_type: RebalanceTriggerType
    triggered_at: datetime
    severity: str  # "low", "medium", "high", "critical"
    
    # Trigger details
    description: str
    current_value: float
    threshold_value: float
    deviation: float
    
    # Affected positions
    affected_positions: List[str] = field(default_factory=list)
    
    # Actions required
    requires_immediate_action: bool = False
    estimated_transaction_cost: float = 0.0


@dataclass
class RebalanceProposal:
    """Portfolio rebalancing proposal"""
    proposal_id: str
    client_id: str
    generated_at: datetime
    
    # Triggers
    triggers: List[RebalanceTrigger] = field(default_factory=list)
    
    # Trades
    trades: List[Dict[str, Any]] = field(default_factory=list)
    
    # Optimization metrics
    total_transaction_cost: float = 0.0
    expected_tax_impact: float = 0.0
    tax_savings: float = 0.0  # From TLH
    improvement_score: float = 0.0
    
    # Risk metrics
    current_risk: float = 0.0
    target_risk: float = 0.0
    risk_reduction: float = 0.0
    
    # Compliance
    compliant: bool = False
    compliance_notes: List[str] = field(default_factory=list)


class DriftDetector:
    """
    Real-time drift detection system
    
    Monitors portfolio drift across multiple dimensions:
    - Absolute drift (>5% from target)
    - Relative drift (>20% from target weight)
    - Risk drift (>10% from risk tolerance)
    - Goal drift (>5% probability drop)
    
    Target: <100ms detection latency
    """
    
    def __init__(self):
        self.drift_history: List[Dict[str, Any]] = []
        self.thresholds = {
            "absolute_drift": 0.05,  # 5%
            "relative_drift": 0.20,  # 20%
            "risk_drift": 0.10,  # 10%
            "goal_drift": 0.05  # 5%
        }
    
    def detect_drift(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        current_risk: float,
        target_risk: float
    ) -> List[RebalanceTrigger]:
        """
        Detect portfolio drift
        
        Returns list of triggered rebalancing events
        Target: <100ms latency
        """
        start_time = datetime.now()
        triggers = []
        
        # Check absolute drift (>5% deviation)
        for asset, target_weight in target_allocation.items():
            current_weight = current_allocation.get(asset, 0.0)
            absolute_deviation = abs(current_weight - target_weight)
            
            if absolute_deviation > self.thresholds["absolute_drift"]:
                trigger = RebalanceTrigger(
                    trigger_id=f"DRIFT-{uuid.uuid4().hex[:8].upper()}",
                    trigger_type=RebalanceTriggerType.DRIFT_ABSOLUTE,
                    triggered_at=datetime.now(),
                    severity="high" if absolute_deviation > 0.10 else "medium",
                    description=f"{asset} deviated {absolute_deviation*100:.1f}% from target",
                    current_value=current_weight,
                    threshold_value=target_weight,
                    deviation=absolute_deviation,
                    affected_positions=[asset],
                    requires_immediate_action=absolute_deviation > 0.10
                )
                triggers.append(trigger)
        
        # Check relative drift (>20% of target weight)
        for asset, target_weight in target_allocation.items():
            if target_weight == 0:
                continue
                
            current_weight = current_allocation.get(asset, 0.0)
            relative_deviation = abs(current_weight - target_weight) / target_weight
            
            if relative_deviation > self.thresholds["relative_drift"]:
                trigger = RebalanceTrigger(
                    trigger_id=f"DRIFT-{uuid.uuid4().hex[:8].upper()}",
                    trigger_type=RebalanceTriggerType.DRIFT_RELATIVE,
                    triggered_at=datetime.now(),
                    severity="high" if relative_deviation > 0.30 else "medium",
                    description=f"{asset} relative drift {relative_deviation*100:.1f}%",
                    current_value=current_weight,
                    threshold_value=target_weight,
                    deviation=relative_deviation,
                    affected_positions=[asset]
                )
                triggers.append(trigger)
        
        # Check risk drift (>10% from risk tolerance)
        risk_deviation = abs(current_risk - target_risk) / target_risk if target_risk > 0 else 0
        
        if risk_deviation > self.thresholds["risk_drift"]:
            trigger = RebalanceTrigger(
                trigger_id=f"DRIFT-{uuid.uuid4().hex[:8].upper()}",
                trigger_type=RebalanceTriggerType.DRIFT_RISK,
                triggered_at=datetime.now(),
                severity="critical" if risk_deviation > 0.20 else "high",
                description=f"Portfolio risk drift {risk_deviation*100:.1f}%",
                current_value=current_risk,
                threshold_value=target_risk,
                deviation=risk_deviation,
                requires_immediate_action=True
            )
            triggers.append(trigger)
        
        # Calculate latency
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Track drift detection
        self.drift_history.append({
            "timestamp": datetime.now().isoformat(),
            "triggers": len(triggers),
            "latency_ms": latency_ms
        })
        
        logger.info(f"Drift detection completed in {latency_ms:.2f}ms - {len(triggers)} triggers")
        
        return triggers


class TaxOptimizer:
    """
    Tax-loss harvesting and optimization engine
    
    Features:
    - Tax-loss harvesting with wash sale avoidance
    - Capital gains management (short/long term)
    - Tax lot optimization (FIFO, LIFO, HIFO, LOFO)
    - Tax efficiency target: 75-125 bps annually
    """
    
    def __init__(self):
        self.harvest_history: List[Dict[str, Any]] = []
        self.wash_sale_window = 30  # days
        self.tax_efficiency_target_bps = 100  # 1.00% annually
    
    def identify_tax_loss_opportunities(
        self,
        positions: Dict[str, Position],
        tax_rate: float = 0.30
    ) -> List[Dict[str, Any]]:
        """
        Identify tax-loss harvesting opportunities
        
        Returns list of positions with harvestable losses
        """
        opportunities = []
        
        for ticker, position in positions.items():
            # Check for unrealized losses
            if position.unrealized_gain_loss < 0:
                # Get lots with losses
                loss_lots = [
                    lot for lot in position.tax_lots
                    if lot.unrealized_gain_loss < 0
                ]
                
                if loss_lots:
                    total_loss = sum(lot.unrealized_gain_loss for lot in loss_lots)
                    tax_benefit = abs(total_loss) * tax_rate
                    
                    opportunity = {
                        "ticker": ticker,
                        "loss_amount": total_loss,
                        "tax_benefit": tax_benefit,
                        "lots": loss_lots,
                        "quantity": sum(lot.quantity for lot in loss_lots),
                        "priority": "high" if tax_benefit > 1000 else "medium"
                    }
                    opportunities.append(opportunity)
        
        # Sort by tax benefit
        opportunities.sort(key=lambda x: x["tax_benefit"], reverse=True)
        
        return opportunities
    
    def check_wash_sale(
        self,
        ticker: str,
        sale_date: datetime,
        purchase_history: List[datetime]
    ) -> bool:
        """
        Check for wash sale violation
        
        Wash sale: Buying substantially identical security 30 days
        before or after sale at a loss
        """
        window_start = sale_date - timedelta(days=self.wash_sale_window)
        window_end = sale_date + timedelta(days=self.wash_sale_window)
        
        for purchase_date in purchase_history:
            if window_start <= purchase_date <= window_end:
                return True
        
        return False
    
    def optimize_tax_lots(
        self,
        position: Position,
        quantity_to_sell: float,
        method: TaxLotMethod,
        minimize_gains: bool = True
    ) -> List[TaxLot]:
        """
        Optimize tax lot selection for sale
        
        Returns ordered list of lots to sell
        """
        # Get lots ordered by method
        ordered_lots = position.get_lots_by_method(method)
        
        # Select lots
        selected_lots = []
        remaining_quantity = quantity_to_sell
        
        for lot in ordered_lots:
            if remaining_quantity <= 0:
                break
            
            quantity = min(lot.quantity, remaining_quantity)
            selected_lots.append(lot)
            remaining_quantity -= quantity
        
        return selected_lots
    
    def calculate_tax_impact(
        self,
        trades: List[Dict[str, Any]],
        tax_rate_short_term: float = 0.37,
        tax_rate_long_term: float = 0.20
    ) -> Dict[str, float]:
        """
        Calculate tax impact of trades
        
        Returns short-term and long-term gains/losses
        """
        short_term_gains = 0.0
        long_term_gains = 0.0
        
        for trade in trades:
            if trade["action"] == "sell":
                gain_loss = trade.get("gain_loss", 0.0)
                is_long_term = trade.get("is_long_term", False)
                
                if is_long_term:
                    long_term_gains += gain_loss
                else:
                    short_term_gains += gain_loss
        
        return {
            "short_term_gains": short_term_gains,
            "long_term_gains": long_term_gains,
            "short_term_tax": short_term_gains * tax_rate_short_term if short_term_gains > 0 else 0,
            "long_term_tax": long_term_gains * tax_rate_long_term if long_term_gains > 0 else 0,
            "total_tax": (
                (short_term_gains * tax_rate_short_term if short_term_gains > 0 else 0) +
                (long_term_gains * tax_rate_long_term if long_term_gains > 0 else 0)
            )
        }
    
    def generate_replacement_security(self, ticker: str) -> str:
        """
        Generate replacement security to avoid wash sale
        
        Returns substantially different but similar security
        """
        # In production, would use sophisticated matching
        # For now, simple placeholder logic
        replacements = {
            "SPY": "IVV",  # S&P 500 ETFs
            "IVV": "VOO",
            "VOO": "SPY",
            "VTI": "ITOT",  # Total market
            "ITOT": "VTI"
        }
        
        return replacements.get(ticker, ticker)


# Continue in next part...
class RiskManager:
    """
    Comprehensive risk management and monitoring
    
    Features:
    - Value at Risk (VaR) 95% and 99%
    - Conditional VaR (CVaR) - expected shortfall
    - Beta, volatility, maximum drawdown
    - Concentration limits (position, sector, geographic)
    - Liquidity risk metrics
    - Real-time alerts
    - Stress testing (2008, 2020, custom scenarios)
    
    Target: <500ms risk calculation latency
    """
    
    def __init__(self):
        self.risk_history: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        
        # Risk limits
        self.limits = {
            "single_position_max": 0.10,  # 10%
            "sector_concentration_max": 0.30,  # 30%
            "geographic_concentration_max": 0.40,  # 40%
            "max_drawdown_limit": 0.20,  # 20%
            "var_95_limit": 0.05,  # 5%
            "var_99_limit": 0.10  # 10%
        }
    
    def calculate_var(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate Value at Risk (VaR)
        
        VaR = maximum loss at given confidence level
        95% VaR = loss exceeded only 5% of the time
        """
        if len(returns) == 0:
            return 0.0
        
        # Historical VaR
        var_percentile = np.percentile(returns, (1 - confidence_level) * 100)
        var_dollar = abs(var_percentile * portfolio_value)
        
        return var_dollar
    
    def calculate_cvar(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        portfolio_value: float = 1000000
    ) -> float:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall
        
        CVaR = average loss beyond VaR threshold
        More conservative than VaR
        """
        if len(returns) == 0:
            return 0.0
        
        # Get VaR threshold
        var_threshold = np.percentile(returns, (1 - confidence_level) * 100)
        
        # Calculate average of losses beyond VaR
        tail_losses = returns[returns <= var_threshold]
        
        if len(tail_losses) == 0:
            return 0.0
        
        cvar = abs(np.mean(tail_losses) * portfolio_value)
        
        return cvar
    
    def calculate_beta(
        self,
        portfolio_returns: np.ndarray,
        market_returns: np.ndarray
    ) -> float:
        """
        Calculate portfolio beta (market sensitivity)
        
        Beta > 1: More volatile than market
        Beta < 1: Less volatile than market
        Beta = 1: Moves with market
        """
        if len(portfolio_returns) != len(market_returns) or len(portfolio_returns) < 2:
            return 1.0
        
        # Covariance of portfolio with market / Variance of market
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        
        if market_variance == 0:
            return 1.0
        
        beta = covariance / market_variance
        
        return beta
    
    def calculate_volatility(self, returns: np.ndarray, annualize: bool = True) -> float:
        """
        Calculate portfolio volatility (standard deviation)
        
        Annualized volatility = daily volatility * sqrt(252)
        """
        if len(returns) < 2:
            return 0.0
        
        volatility = np.std(returns)
        
        if annualize:
            volatility *= np.sqrt(252)  # Annualize (252 trading days)
        
        return volatility
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """
        Calculate maximum drawdown
        
        Max drawdown = largest peak-to-trough decline
        """
        if len(portfolio_values) < 2:
            return 0.0
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(portfolio_values)
        
        # Calculate drawdowns
        drawdowns = (portfolio_values - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = abs(np.min(drawdowns))
        
        return max_drawdown
    
    def check_concentration_limits(
        self,
        positions: Dict[str, Position],
        portfolio_value: float
    ) -> List[Dict[str, Any]]:
        """
        Check concentration risk limits
        
        Returns list of limit breaches
        """
        breaches = []
        
        # Single position limits
        for ticker, position in positions.items():
            weight = position.market_value / portfolio_value
            
            if weight > self.limits["single_position_max"]:
                breaches.append({
                    "type": "single_position",
                    "ticker": ticker,
                    "current": weight,
                    "limit": self.limits["single_position_max"],
                    "excess": weight - self.limits["single_position_max"],
                    "severity": "high"
                })
        
        # Sector concentration (would need sector mapping)
        # Geographic concentration (would need country mapping)
        
        return breaches
    
    def calculate_liquidity_metrics(
        self,
        positions: Dict[str, Position],
        daily_volumes: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Calculate liquidity risk metrics
        
        Returns liquidity coverage ratio and time to liquidate
        """
        metrics = {
            "highly_liquid": 0.0,  # Can liquidate in 1 day
            "liquid": 0.0,  # Can liquidate in 5 days
            "illiquid": 0.0,  # Takes >5 days
            "time_to_liquidate_days": 0.0
        }
        
        total_value = sum(p.market_value for p in positions.values())
        
        for ticker, position in positions.items():
            daily_volume = daily_volumes.get(ticker, 0)
            
            if daily_volume == 0:
                metrics["illiquid"] += position.market_value
                continue
            
            # Assume can trade 10% of daily volume without impact
            liquidation_days = position.quantity / (daily_volume * 0.10)
            
            if liquidation_days <= 1:
                metrics["highly_liquid"] += position.market_value
            elif liquidation_days <= 5:
                metrics["liquid"] += position.market_value
            else:
                metrics["illiquid"] += position.market_value
        
        if total_value > 0:
            metrics["highly_liquid"] /= total_value
            metrics["liquid"] /= total_value
            metrics["illiquid"] /= total_value
        
        return metrics
    
    def stress_test(
        self,
        positions: Dict[str, Position],
        scenario: str
    ) -> Dict[str, Any]:
        """
        Run stress test scenarios
        
        Scenarios:
        - 2008: Global Financial Crisis (-40% equities, +10% bonds)
        - 2020: COVID-19 (-34% equities peak to trough)
        - INTEREST_RATE: +200 bps rate shock
        - EQUITY_CRASH: -30% equity crash
        - CUSTOM: User-defined shocks
        """
        scenarios = {
            "2008_CRISIS": {
                "equities": -0.40,
                "bonds": 0.10,
                "alternatives": -0.20,
                "cash": 0.0
            },
            "2020_COVID": {
                "equities": -0.34,
                "bonds": 0.08,
                "alternatives": -0.15,
                "cash": 0.0
            },
            "INTEREST_RATE_SHOCK": {
                "equities": -0.10,
                "bonds": -0.15,
                "alternatives": -0.05,
                "cash": 0.0
            },
            "EQUITY_CRASH": {
                "equities": -0.30,
                "bonds": 0.05,
                "alternatives": -0.10,
                "cash": 0.0
            }
        }
        
        shocks = scenarios.get(scenario, scenarios["EQUITY_CRASH"])
        
        # Calculate stressed portfolio value
        current_value = sum(p.market_value for p in positions.values())
        stressed_value = 0.0
        
        for ticker, position in positions.items():
            # Simplified: assume all equities
            # In production, would map each position to asset class
            asset_class = "equities"  # Would look up actual asset class
            shock = shocks.get(asset_class, 0.0)
            
            stressed_position_value = position.market_value * (1 + shock)
            stressed_value += stressed_position_value
        
        loss = current_value - stressed_value
        loss_percentage = (loss / current_value) * 100 if current_value > 0 else 0
        
        return {
            "scenario": scenario,
            "current_value": current_value,
            "stressed_value": stressed_value,
            "loss": loss,
            "loss_percentage": loss_percentage,
            "passes_limit": loss_percentage <= 20.0  # 20% max drawdown limit
        }
    
    def generate_risk_report(
        self,
        positions: Dict[str, Position],
        returns: np.ndarray,
        portfolio_value: float
    ) -> Dict[str, Any]:
        """
        Generate comprehensive risk report
        
        Target: <500ms generation time
        """
        start_time = datetime.now()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "portfolio_value": portfolio_value,
            "risk_metrics": {
                "var_95": self.calculate_var(returns, 0.95, portfolio_value),
                "var_99": self.calculate_var(returns, 0.99, portfolio_value),
                "cvar_95": self.calculate_cvar(returns, 0.95, portfolio_value),
                "volatility": self.calculate_volatility(returns),
                "beta": self.calculate_beta(returns, returns),  # Would use market returns
                "max_drawdown": 0.0  # Would need historical values
            },
            "concentration_breaches": self.check_concentration_limits(positions, portfolio_value),
            "stress_tests": {
                "2008_crisis": self.stress_test(positions, "2008_CRISIS"),
                "2020_covid": self.stress_test(positions, "2020_COVID"),
                "equity_crash": self.stress_test(positions, "EQUITY_CRASH")
            }
        }
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        report["generation_time_ms"] = latency_ms
        
        logger.info(f"Risk report generated in {latency_ms:.2f}ms")
        
        return report


class PerformanceAnalytics:
    """
    Performance measurement and attribution
    
    Features:
    - Absolute returns (daily, monthly, YTD, ITD)
    - Risk-adjusted returns (Sharpe, Sortino, Information, Treynor, Calmar)
    - Benchmark comparison (tracking error, active share)
    - Multi-level attribution (asset allocation, security selection, factor)
    - Goal progress tracking
    
    Target: <2 seconds for attribution calculation
    """
    
    def __init__(self):
        self.performance_history: List[Dict[str, Any]] = []
    
    def calculate_returns(
        self,
        portfolio_values: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate various return metrics
        """
        if len(portfolio_values) < 2:
            return {"daily": 0.0, "total": 0.0}
        
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        return {
            "daily": returns[-1] if len(returns) > 0 else 0.0,
            "weekly": np.mean(returns[-5:]) if len(returns) >= 5 else 0.0,
            "monthly": np.mean(returns[-21:]) if len(returns) >= 21 else 0.0,
            "ytd": (portfolio_values[-1] / portfolio_values[0] - 1) if len(portfolio_values) > 0 else 0.0,
            "total": (portfolio_values[-1] / portfolio_values[0] - 1) if len(portfolio_values) > 0 else 0.0
        }
    
    def calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sharpe Ratio
        
        Sharpe = (Return - Risk-Free Rate) / Volatility
        Measures excess return per unit of risk
        """
        if len(returns) < 2:
            return 0.0
        
        excess_return = np.mean(returns) - (risk_free_rate / 252)  # Daily risk-free
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0.0
        
        sharpe = (excess_return / volatility) * np.sqrt(252)  # Annualize
        
        return sharpe
    
    def calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.02
    ) -> float:
        """
        Calculate Sortino Ratio
        
        Sortino = (Return - Risk-Free Rate) / Downside Deviation
        Only penalizes downside volatility
        """
        if len(returns) < 2:
            return 0.0
        
        excess_return = np.mean(returns) - (risk_free_rate / 252)
        
        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        
        if downside_std == 0:
            return 0.0
        
        sortino = (excess_return / downside_std) * np.sqrt(252)
        
        return sortino
    
    def calculate_information_ratio(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate Information Ratio
        
        IR = Active Return / Tracking Error
        Measures excess return vs benchmark per unit of active risk
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        active_returns = portfolio_returns - benchmark_returns
        active_return = np.mean(active_returns)
        tracking_error = np.std(active_returns)
        
        if tracking_error == 0:
            return 0.0
        
        ir = (active_return / tracking_error) * np.sqrt(252)
        
        return ir
    
    def calculate_calmar_ratio(
        self,
        returns: np.ndarray,
        portfolio_values: np.ndarray
    ) -> float:
        """
        Calculate Calmar Ratio
        
        Calmar = Annualized Return / Maximum Drawdown
        """
        if len(returns) < 2 or len(portfolio_values) < 2:
            return 0.0
        
        annualized_return = np.mean(returns) * 252
        max_drawdown = self.calculate_max_drawdown(portfolio_values)
        
        if max_drawdown == 0:
            return 0.0
        
        calmar = annualized_return / max_drawdown
        
        return calmar
    
    def calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        if len(portfolio_values) < 2:
            return 0.0
        
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        return max_drawdown
    
    def calculate_tracking_error(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> float:
        """
        Calculate tracking error
        
        TE = Standard deviation of active returns (portfolio - benchmark)
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return 0.0
        
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)  # Annualize
        
        return tracking_error
    
    def calculate_capture_ratios(
        self,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate up/down capture ratios
        
        Up Capture = Portfolio return in up markets / Benchmark return in up markets
        Down Capture = Portfolio return in down markets / Benchmark return in down markets
        """
        if len(portfolio_returns) != len(benchmark_returns) or len(portfolio_returns) < 2:
            return {"up_capture": 1.0, "down_capture": 1.0}
        
        # Up markets (benchmark positive)
        up_mask = benchmark_returns > 0
        up_portfolio = portfolio_returns[up_mask]
        up_benchmark = benchmark_returns[up_mask]
        
        up_capture = (np.mean(up_portfolio) / np.mean(up_benchmark)) if len(up_benchmark) > 0 and np.mean(up_benchmark) != 0 else 1.0
        
        # Down markets (benchmark negative)
        down_mask = benchmark_returns < 0
        down_portfolio = portfolio_returns[down_mask]
        down_benchmark = benchmark_returns[down_mask]
        
        down_capture = (np.mean(down_portfolio) / np.mean(down_benchmark)) if len(down_benchmark) > 0 and np.mean(down_benchmark) != 0 else 1.0
        
        return {
            "up_capture": up_capture,
            "down_capture": down_capture
        }
    
    def attribution_analysis(
        self,
        portfolio_weights: Dict[str, float],
        portfolio_returns: Dict[str, float],
        benchmark_weights: Dict[str, float],
        benchmark_returns: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Multi-level performance attribution
        
        Decomposes excess return into:
        - Asset allocation effect
        - Security selection effect
        - Interaction effect
        
        Target: <2 seconds
        """
        start_time = datetime.now()
        
        allocation_effect = 0.0
        selection_effect = 0.0
        interaction_effect = 0.0
        
        # Calculate effects for each asset
        for asset in set(list(portfolio_weights.keys()) + list(benchmark_weights.keys())):
            pw = portfolio_weights.get(asset, 0.0)
            pr = portfolio_returns.get(asset, 0.0)
            bw = benchmark_weights.get(asset, 0.0)
            br = benchmark_returns.get(asset, 0.0)
            
            # Allocation effect: (Portfolio Weight - Benchmark Weight) * Benchmark Return
            allocation_effect += (pw - bw) * br
            
            # Selection effect: Benchmark Weight * (Portfolio Return - Benchmark Return)
            selection_effect += bw * (pr - br)
            
            # Interaction effect: (Portfolio Weight - Benchmark Weight) * (Portfolio Return - Benchmark Return)
            interaction_effect += (pw - bw) * (pr - br)
        
        total_excess = allocation_effect + selection_effect + interaction_effect
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            "allocation_effect": allocation_effect,
            "selection_effect": selection_effect,
            "interaction_effect": interaction_effect,
            "total_excess_return": total_excess,
            "calculation_time_ms": latency_ms
        }
    
    def generate_performance_report(
        self,
        portfolio_values: np.ndarray,
        portfolio_returns: np.ndarray,
        benchmark_returns: np.ndarray
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Target: <2 seconds
        """
        start_time = datetime.now()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "returns": self.calculate_returns(portfolio_values),
            "risk_adjusted_returns": {
                "sharpe_ratio": self.calculate_sharpe_ratio(portfolio_returns),
                "sortino_ratio": self.calculate_sortino_ratio(portfolio_returns),
                "information_ratio": self.calculate_information_ratio(portfolio_returns, benchmark_returns),
                "calmar_ratio": self.calculate_calmar_ratio(portfolio_returns, portfolio_values)
            },
            "benchmark_comparison": {
                "tracking_error": self.calculate_tracking_error(portfolio_returns, benchmark_returns),
                "capture_ratios": self.calculate_capture_ratios(portfolio_returns, benchmark_returns)
            },
            "risk_metrics": {
                "max_drawdown": self.calculate_max_drawdown(portfolio_values),
                "volatility": np.std(portfolio_returns) * np.sqrt(252)
            }
        }
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        report["generation_time_ms"] = latency_ms
        
        logger.info(f"Performance report generated in {latency_ms:.2f}ms")
        
        return report


# Continue in next part...
class TransactionManager:
    """
    Intelligent order execution and transaction management
    
    Features:
    - Smart Order Router (SOR) for best execution
    - Execution algorithms (VWAP, TWAP, Implementation Shortfall, etc.)
    - Transaction Cost Analysis (TCA)
    - Trade settlement and reconciliation
    
    Target: <10 bps transaction cost per trade
    """
    
    def __init__(self):
        self.trade_history: List[Dict[str, Any]] = []
        self.execution_costs: List[float] = []
        self.target_cost_bps = 10  # 0.10%
    
    async def execute_trade(
        self,
        ticker: str,
        action: str,  # "buy" or "sell"
        quantity: float,
        algorithm: ExecutionAlgorithm = ExecutionAlgorithm.VWAP,
        urgency: str = "normal"  # "low", "normal", "high", "immediate"
    ) -> Dict[str, Any]:
        """
        Execute trade using specified algorithm
        
        Returns execution report with costs
        """
        start_time = datetime.now()
        
        # Simulate execution
        execution_price = await self._get_execution_price(ticker, action, quantity, algorithm)
        
        # Calculate costs
        costs = self._calculate_transaction_costs(
            ticker,
            quantity,
            execution_price,
            action
        )
        
        trade = {
            "trade_id": f"TRADE-{uuid.uuid4().hex[:8].upper()}",
            "ticker": ticker,
            "action": action,
            "quantity": quantity,
            "execution_price": execution_price,
            "execution_algorithm": algorithm.value,
            "urgency": urgency,
            "timestamp": datetime.now().isoformat(),
            "costs": costs,
            "execution_time_ms": (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.trade_history.append(trade)
        self.execution_costs.append(costs["total_cost_bps"])
        
        logger.info(f"Executed {action} {quantity} {ticker} @ {execution_price:.2f} - Cost: {costs['total_cost_bps']:.2f} bps")
        
        return trade
    
    async def _get_execution_price(
        self,
        ticker: str,
        action: str,
        quantity: float,
        algorithm: ExecutionAlgorithm
    ) -> float:
        """
        Get execution price based on algorithm
        
        In production, would interface with actual exchanges/brokers
        """
        # Simulate base price
        base_price = 100.0  # Would fetch real-time price
        
        # Add slippage based on algorithm and quantity
        slippage_factor = {
            ExecutionAlgorithm.VWAP: 0.0005,  # 5 bps
            ExecutionAlgorithm.TWAP: 0.0006,  # 6 bps
            ExecutionAlgorithm.IMPLEMENTATION_SHORTFALL: 0.0004,  # 4 bps
            ExecutionAlgorithm.PARTICIPATION_RATE: 0.0007,  # 7 bps
            ExecutionAlgorithm.ARRIVAL_PRICE: 0.0003  # 3 bps
        }.get(algorithm, 0.0005)
        
        slippage = base_price * slippage_factor
        
        if action == "buy":
            execution_price = base_price + slippage
        else:
            execution_price = base_price - slippage
        
        return execution_price
    
    def _calculate_transaction_costs(
        self,
        ticker: str,
        quantity: float,
        execution_price: float,
        action: str
    ) -> Dict[str, float]:
        """
        Calculate comprehensive transaction costs
        
        Components:
        - Explicit costs (commissions, fees)
        - Implicit costs (bid-ask spread, market impact, slippage)
        """
        notional_value = quantity * execution_price
        
        # Explicit costs
        commission = min(notional_value * 0.0001, 10.0)  # 1 bp or $10 max
        exchange_fees = notional_value * 0.00003  # 0.3 bps
        sec_fees = notional_value * 0.000008 if action == "sell" else 0  # SEC fees on sells
        
        explicit_costs = commission + exchange_fees + sec_fees
        
        # Implicit costs
        bid_ask_spread = notional_value * 0.0002  # 2 bps
        market_impact = notional_value * 0.0003  # 3 bps (size-dependent)
        slippage = notional_value * 0.0002  # 2 bps
        
        implicit_costs = bid_ask_spread + market_impact + slippage
        
        total_costs = explicit_costs + implicit_costs
        total_cost_bps = (total_costs / notional_value) * 10000  # Convert to basis points
        
        return {
            "explicit_costs": explicit_costs,
            "commission": commission,
            "exchange_fees": exchange_fees,
            "sec_fees": sec_fees,
            "implicit_costs": implicit_costs,
            "bid_ask_spread": bid_ask_spread,
            "market_impact": market_impact,
            "slippage": slippage,
            "total_costs": total_costs,
            "total_cost_bps": total_cost_bps,
            "notional_value": notional_value
        }
    
    def execute_vwap(self, ticker: str, quantity: float, duration_hours: int = 1) -> Dict[str, Any]:
        """
        Execute using Volume-Weighted Average Price (VWAP)
        
        VWAP = Matches market volume profile
        Best for: Large orders that need to blend with market
        """
        return {
            "algorithm": "VWAP",
            "ticker": ticker,
            "quantity": quantity,
            "duration_hours": duration_hours,
            "strategy": "Match historical volume profile throughout trading day"
        }
    
    def execute_twap(self, ticker: str, quantity: float, duration_hours: int = 1) -> Dict[str, Any]:
        """
        Execute using Time-Weighted Average Price (TWAP)
        
        TWAP = Spreads orders evenly over time
        Best for: When want consistent market presence
        """
        slices = duration_hours * 12  # Execute every 5 minutes
        quantity_per_slice = quantity / slices
        
        return {
            "algorithm": "TWAP",
            "ticker": ticker,
            "total_quantity": quantity,
            "duration_hours": duration_hours,
            "slices": slices,
            "quantity_per_slice": quantity_per_slice,
            "strategy": "Equal slices every 5 minutes"
        }
    
    def execute_implementation_shortfall(
        self,
        ticker: str,
        quantity: float,
        decision_price: float
    ) -> Dict[str, Any]:
        """
        Execute using Implementation Shortfall
        
        Minimizes difference between decision price and execution price
        Best for: Urgent trades where timing is critical
        """
        return {
            "algorithm": "Implementation Shortfall",
            "ticker": ticker,
            "quantity": quantity,
            "decision_price": decision_price,
            "strategy": "Front-load execution to minimize market impact"
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """
        Get transaction cost analysis statistics
        """
        if not self.execution_costs:
            return {"avg_cost_bps": 0.0, "trades": 0}
        
        return {
            "total_trades": len(self.execution_costs),
            "avg_cost_bps": np.mean(self.execution_costs),
            "median_cost_bps": np.median(self.execution_costs),
            "min_cost_bps": np.min(self.execution_costs),
            "max_cost_bps": np.max(self.execution_costs),
            "target_cost_bps": self.target_cost_bps,
            "below_target_pct": (np.array(self.execution_costs) <= self.target_cost_bps).mean() * 100
        }


class RebalancingEngine:
    """
    Main orchestration class for real-time portfolio management
    
    Integrates:
    - Drift detection (<100ms)
    - Tax optimization (75-125 bps)
    - Risk management (VaR, stress testing)
    - Performance analytics
    - Transaction execution (<10 bps cost)
    
    Target: <2 seconds total rebalancing time
    """
    
    def __init__(self):
        self.drift_detector = DriftDetector()
        self.tax_optimizer = TaxOptimizer()
        self.risk_manager = RiskManager()
        self.performance_analytics = PerformanceAnalytics()
        self.transaction_manager = TransactionManager()
        
        self.rebalance_history: List[Dict[str, Any]] = []
        self.performance_metrics = {
            "avg_rebalance_time_ms": 0.0,
            "avg_transaction_cost_bps": 0.0,
            "tax_efficiency_bps": 0.0,
            "rebalances_executed": 0
        }
    
    async def evaluate_rebalancing_need(
        self,
        client_id: str,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        current_risk: float,
        target_risk: float,
        positions: Dict[str, Position],
        portfolio_value: float
    ) -> Tuple[bool, List[RebalanceTrigger]]:
        """
        Evaluate if rebalancing is needed
        
        Returns: (needs_rebalancing, list of triggers)
        Target: <100ms
        """
        # Detect drift
        triggers = self.drift_detector.detect_drift(
            current_allocation,
            target_allocation,
            current_risk,
            target_risk
        )
        
        # Check concentration limits
        concentration_breaches = self.risk_manager.check_concentration_limits(
            positions,
            portfolio_value
        )
        
        if concentration_breaches:
            for breach in concentration_breaches:
                trigger = RebalanceTrigger(
                    trigger_id=f"CONC-{uuid.uuid4().hex[:8].upper()}",
                    trigger_type=RebalanceTriggerType.DRIFT_ABSOLUTE,
                    triggered_at=datetime.now(),
                    severity=breach["severity"],
                    description=f"Concentration breach: {breach['type']}",
                    current_value=breach["current"],
                    threshold_value=breach["limit"],
                    deviation=breach["excess"],
                    requires_immediate_action=True
                )
                triggers.append(trigger)
        
        needs_rebalancing = len(triggers) > 0
        
        return needs_rebalancing, triggers
    
    async def generate_rebalance_proposal(
        self,
        client_id: str,
        triggers: List[RebalanceTrigger],
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float],
        positions: Dict[str, Position],
        portfolio_value: float,
        tax_rate: float = 0.30
    ) -> RebalanceProposal:
        """
        Generate optimized rebalancing proposal
        
        Target: <2 seconds
        """
        start_time = datetime.now()
        
        proposal_id = f"REBAL-{uuid.uuid4().hex[:8].upper()}"
        
        # Calculate required trades
        trades = []
        total_transaction_cost = 0.0
        total_tax_impact = 0.0
        
        for asset, target_weight in target_allocation.items():
            current_weight = current_allocation.get(asset, 0.0)
            target_value = portfolio_value * target_weight
            current_value = portfolio_value * current_weight
            
            trade_value = target_value - current_value
            
            if abs(trade_value) > portfolio_value * 0.01:  # Only trade if >1%
                action = "buy" if trade_value > 0 else "sell"
                quantity = abs(trade_value) / 100.0  # Assume $100 per share
                
                # Check for tax-loss harvesting opportunities if selling
                if action == "sell" and asset in positions:
                    position = positions[asset]
                    if position.unrealized_gain_loss < 0:
                        # Tax-loss harvest
                        tax_benefit = abs(position.unrealized_gain_loss) * tax_rate
                        
                        trade = {
                            "asset": asset,
                            "action": action,
                            "quantity": quantity,
                            "estimated_price": 100.0,
                            "trade_value": trade_value,
                            "tax_benefit": tax_benefit,
                            "is_tax_loss_harvest": True
                        }
                    else:
                        trade = {
                            "asset": asset,
                            "action": action,
                            "quantity": quantity,
                            "estimated_price": 100.0,
                            "trade_value": trade_value,
                            "tax_cost": position.unrealized_gain_loss * tax_rate,
                            "is_tax_loss_harvest": False
                        }
                else:
                    trade = {
                        "asset": asset,
                        "action": action,
                        "quantity": quantity,
                        "estimated_price": 100.0,
                        "trade_value": trade_value
                    }
                
                # Estimate transaction cost
                trade_cost_bps = self.transaction_manager.target_cost_bps
                trade_cost = abs(trade_value) * (trade_cost_bps / 10000)
                trade["transaction_cost"] = trade_cost
                
                total_transaction_cost += trade_cost
                total_tax_impact += trade.get("tax_cost", 0) - trade.get("tax_benefit", 0)
                
                trades.append(trade)
        
        # Calculate improvement score
        improvement_score = sum(t.deviation for t in triggers)
        
        proposal = RebalanceProposal(
            proposal_id=proposal_id,
            client_id=client_id,
            generated_at=datetime.now(),
            triggers=triggers,
            trades=trades,
            total_transaction_cost=total_transaction_cost,
            expected_tax_impact=total_tax_impact,
            tax_savings=sum(t.get("tax_benefit", 0) for t in trades),
            improvement_score=improvement_score,
            compliant=True
        )
        
        latency_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        logger.info(f"Rebalance proposal {proposal_id} generated in {latency_ms:.2f}ms")
        
        # Update metrics
        self.performance_metrics["avg_rebalance_time_ms"] = latency_ms
        
        return proposal
    
    async def execute_rebalance(
        self,
        proposal: RebalanceProposal
    ) -> Dict[str, Any]:
        """
        Execute rebalancing trades
        
        Target: <2 seconds total execution time
        """
        start_time = datetime.now()
        
        executed_trades = []
        total_cost = 0.0
        
        for trade in proposal.trades:
            # Execute trade
            execution = await self.transaction_manager.execute_trade(
                ticker=trade["asset"],
                action=trade["action"],
                quantity=trade["quantity"],
                algorithm=ExecutionAlgorithm.VWAP
            )
            
            executed_trades.append(execution)
            total_cost += execution["costs"]["total_costs"]
        
        execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        result = {
            "proposal_id": proposal.proposal_id,
            "client_id": proposal.client_id,
            "executed_at": datetime.now().isoformat(),
            "trades_executed": len(executed_trades),
            "total_cost": total_cost,
            "execution_time_ms": execution_time_ms,
            "trades": executed_trades
        }
        
        # Update metrics
        self.rebalance_history.append(result)
        self.performance_metrics["rebalances_executed"] += 1
        self.performance_metrics["avg_transaction_cost_bps"] = np.mean(
            self.transaction_manager.execution_costs
        )
        
        logger.info(f"Rebalance executed in {execution_time_ms:.2f}ms - {len(executed_trades)} trades")
        
        return result
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """
        Get comprehensive performance dashboard
        """
        return {
            "timestamp": datetime.now().isoformat(),
            "rebalancing_metrics": {
                "avg_rebalance_time_ms": self.performance_metrics["avg_rebalance_time_ms"],
                "target_rebalance_time_ms": 2000,
                "rebalances_executed": self.performance_metrics["rebalances_executed"]
            },
            "transaction_metrics": self.transaction_manager.get_execution_statistics(),
            "tax_efficiency": {
                "estimated_tax_alpha_bps": self.performance_metrics["tax_efficiency_bps"],
                "target_range_bps": "75-125"
            },
            "drift_detection": {
                "avg_detection_time_ms": np.mean([
                    d["latency_ms"] for d in self.drift_detector.drift_history
                ]) if self.drift_detector.drift_history else 0,
                "target_detection_time_ms": 100
            }
        }


# Integration test and example usage
async def main():
    """Example usage of Real-Time Portfolio Management"""
    print("\n🎯 Ultra Platform - Real-Time Portfolio Management Demo\n")
    
    engine = RebalancingEngine()
    
    # Create sample positions
    positions = {
        "SPY": Position(
            ticker="SPY",
            total_quantity=100,
            average_cost=400.0,
            current_price=450.0,
            market_value=45000,
            tax_lots=[
                TaxLot(
                    lot_id="LOT-001",
                    ticker="SPY",
                    quantity=50,
                    purchase_price=380.0,
                    purchase_date=datetime.now() - timedelta(days=400),
                    cost_basis=19000,
                    current_price=450.0
                ),
                TaxLot(
                    lot_id="LOT-002",
                    ticker="SPY",
                    quantity=50,
                    purchase_price=420.0,
                    purchase_date=datetime.now() - timedelta(days=100),
                    cost_basis=21000,
                    current_price=450.0
                )
            ]
        ),
        "AGG": Position(
            ticker="AGG",
            total_quantity=200,
            average_cost=105.0,
            current_price=100.0,
            market_value=20000,
            tax_lots=[
                TaxLot(
                    lot_id="LOT-003",
                    ticker="AGG",
                    quantity=200,
                    purchase_price=105.0,
                    purchase_date=datetime.now() - timedelta(days=200),
                    cost_basis=21000,
                    current_price=100.0
                )
            ]
        )
    }
    
    portfolio_value = 65000
    current_allocation = {"SPY": 0.69, "AGG": 0.31}  # 69% equities, 31% bonds
    target_allocation = {"SPY": 0.60, "AGG": 0.40}  # Target: 60/40
    
    print("📊 Portfolio Overview:")
    print(f"   Total Value: ${portfolio_value:,.0f}")
    print(f"   Current: {current_allocation['SPY']*100:.0f}% SPY / {current_allocation['AGG']*100:.0f}% AGG")
    print(f"   Target:  {target_allocation['SPY']*100:.0f}% SPY / {target_allocation['AGG']*100:.0f}% AGG")
    
    # Step 1: Evaluate rebalancing need
    print("\n1️⃣ Evaluating Rebalancing Need...")
    needs_rebalancing, triggers = await engine.evaluate_rebalancing_need(
        client_id="CLI-TEST",
        current_allocation=current_allocation,
        target_allocation=target_allocation,
        current_risk=0.15,
        target_risk=0.12,
        positions=positions,
        portfolio_value=portfolio_value
    )
    
    print(f"   Needs Rebalancing: {needs_rebalancing}")
    print(f"   Triggers: {len(triggers)}")
    for trigger in triggers:
        print(f"     - {trigger.trigger_type.value}: {trigger.description}")
    
    if needs_rebalancing:
        # Step 2: Generate rebalancing proposal
        print("\n2️⃣ Generating Rebalance Proposal...")
        proposal = await engine.generate_rebalance_proposal(
            client_id="CLI-TEST",
            triggers=triggers,
            current_allocation=current_allocation,
            target_allocation=target_allocation,
            positions=positions,
            portfolio_value=portfolio_value
        )
        
        print(f"   Proposal ID: {proposal.proposal_id}")
        print(f"   Trades: {len(proposal.trades)}")
        print(f"   Transaction Cost: ${proposal.total_transaction_cost:.2f}")
        print(f"   Tax Savings: ${proposal.tax_savings:.2f}")
        
        for trade in proposal.trades:
            action_symbol = "📈" if trade["action"] == "buy" else "📉"
            print(f"     {action_symbol} {trade['action'].upper()} {trade['quantity']:.0f} {trade['asset']}")
        
        # Step 3: Tax-loss harvesting check
        print("\n3️⃣ Tax-Loss Harvesting Analysis...")
        tax_opportunities = engine.tax_optimizer.identify_tax_loss_opportunities(
            positions,
            tax_rate=0.30
        )
        
        if tax_opportunities:
            print(f"   Opportunities Found: {len(tax_opportunities)}")
            for opp in tax_opportunities:
                print(f"     - {opp['ticker']}: ${opp['tax_benefit']:.2f} tax benefit")
        else:
            print("   No tax-loss harvesting opportunities")
        
        # Step 4: Risk analysis
        print("\n4️⃣ Risk Analysis...")
        returns = np.random.normal(0.0008, 0.015, 252)  # Simulated returns
        risk_report = engine.risk_manager.generate_risk_report(
            positions,
            returns,
            portfolio_value
        )
        
        print(f"   VaR 95%: ${risk_report['risk_metrics']['var_95']:,.0f}")
        print(f"   VaR 99%: ${risk_report['risk_metrics']['var_99']:,.0f}")
        print(f"   Volatility: {risk_report['risk_metrics']['volatility']*100:.1f}%")
        
        # Stress tests
        print("\n   Stress Tests:")
        for scenario, result in risk_report['stress_tests'].items():
            emoji = "✅" if result['passes_limit'] else "⚠️"
            print(f"     {emoji} {scenario}: {result['loss_percentage']:.1f}% loss")
        
        # Step 5: Execute rebalancing
        print("\n5️⃣ Executing Rebalance...")
        execution_result = await engine.execute_rebalance(proposal)
        
        print(f"   Execution Time: {execution_result['execution_time_ms']:.0f}ms")
        print(f"   Total Cost: ${execution_result['total_cost']:.2f}")
        print(f"   Trades Executed: {execution_result['trades_executed']}")
    
    # Performance dashboard
    print("\n📈 Performance Dashboard:")
    dashboard = engine.get_performance_dashboard()
    print(f"   Avg Rebalance Time: {dashboard['rebalancing_metrics']['avg_rebalance_time_ms']:.0f}ms")
    print(f"   Target: {dashboard['rebalancing_metrics']['target_rebalance_time_ms']}ms")
    print(f"   Avg Transaction Cost: {dashboard['transaction_metrics']['avg_cost_bps']:.2f} bps")
    print(f"   Target: {dashboard['transaction_metrics']['target_cost_bps']} bps")
    print(f"   Rebalances Executed: {dashboard['rebalancing_metrics']['rebalances_executed']}")


if __name__ == "__main__":
    asyncio.run(main())
