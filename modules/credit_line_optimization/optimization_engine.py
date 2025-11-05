from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict, deque
import statistics
import math
import hashlib
from abc import ABC, abstractmethod
from scipy import stats, optimize
from scipy.optimize import linprog, minimize
import warnings
warnings.filterwarnings('ignore')

class CustomerSegment(Enum):
    PRIME = 'prime'
    SUPER_PRIME = 'super_prime'
    NEAR_PRIME = 'near_prime'
    SUBPRIME = 'subprime'
    CORPORATE = 'corporate'
    SME = 'sme'
    INSTITUTIONAL = 'institutional'

class OptimizationObjective(Enum):
    MAX_PROFIT = 'max_profit'
    MIN_RISK = 'min_risk'
    MAX_SHARPE = 'max_sharpe_ratio'
    MAX_VOLUME = 'max_volume'
    BALANCED = 'balanced'
    REGULATORY = 'regulatory_compliant'

class LimitStrategy(Enum):
    CONSERVATIVE = 'conservative'
    MODERATE = 'moderate'
    AGGRESSIVE = 'aggressive'
    DYNAMIC = 'dynamic'
    RISK_BASED = 'risk_based'

class AdjustmentTrigger(Enum):
    UTILIZATION_CHANGE = 'utilization_change'
    PAYMENT_BEHAVIOR = 'payment_behavior'
    RISK_SCORE_CHANGE = 'risk_score_change'
    MARKET_CONDITIONS = 'market_conditions'
    REGULATORY_CHANGE = 'regulatory_change'
    TIME_BASED = 'time_based'

@dataclass
class CreditLineProfile:
    '''Credit line profile for a customer'''
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment: CustomerSegment = CustomerSegment.NEAR_PRIME
    current_limit: float = 0.0
    utilized_amount: float = 0.0
    available_credit: float = 0.0
    utilization_rate: float = 0.0
    risk_score: float = 0.0
    probability_of_default: float = 0.0
    loss_given_default: float = 0.0
    expected_loss: float = 0.0
    revenue_potential: float = 0.0
    profitability: float = 0.0
    
@dataclass
class OptimizationResult:
    '''Result of credit line optimization'''
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    customer_id: str = ''
    current_limit: float = 0.0
    recommended_limit: float = 0.0
    change_amount: float = 0.0
    change_percentage: float = 0.0
    expected_revenue: float = 0.0
    expected_loss: float = 0.0
    risk_adjusted_return: float = 0.0
    constraints_satisfied: bool = True
    recommendations: List[str] = field(default_factory=list)

class CreditLineOptimization:
    '''Comprehensive Credit Line Optimization System for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Credit Line Optimization'
        self.version = '2.0'
        
        # Core components
        self.limit_calculator = CreditLimitCalculator()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.dynamic_adjuster = DynamicLimitAdjuster()
        self.risk_pricer = RiskBasedPricer()
        self.exposure_manager = ExposureManager()
        self.capital_allocator = CapitalAllocator()
        self.profitability_analyzer = ProfitabilityAnalyzer()
        self.compliance_checker = ComplianceChecker()
        
        # Configuration
        self.optimization_params = self._initialize_params()
        self.risk_limits = self._initialize_risk_limits()
        self.pricing_matrix = self._initialize_pricing()
        
        print('💳 Credit Line Optimization System initialized')
    
    def optimize_credit_line(self, customer_data: Dict, objective: OptimizationObjective = OptimizationObjective.MAX_SHARPE):
        '''Optimize credit line for a customer'''
        print('CREDIT LINE OPTIMIZATION')
        print('='*80)
        print(f'Customer ID: {customer_data.get("customer_id", "unknown")}')
        print(f'Segment: {customer_data.get("segment", "unknown")}')
        print(f'Current Limit: ')
        print(f'Optimization Objective: {objective.value}')
        print()
        
        # Create customer profile
        profile = self._create_customer_profile(customer_data)
        
        # Step 1: Risk Assessment
        print('1️⃣ RISK ASSESSMENT')
        print('-'*40)
        risk_metrics = self._assess_customer_risk(profile, customer_data)
        print(f'  Risk Score: {risk_metrics["risk_score"]:.1f}/100')
        print(f'  Probability of Default: {risk_metrics["pd"]:.2%}')
        print(f'  Loss Given Default: {risk_metrics["lgd"]:.2%}')
        print(f'  Expected Loss: ')
        print(f'  Risk Category: {risk_metrics["risk_category"]}')
        
        # Step 2: Capacity Analysis
        print('\n2️⃣ CAPACITY ANALYSIS')
        print('-'*40)
        capacity = self.limit_calculator.calculate_capacity(customer_data)
        print(f'  Debt Service Capacity: ')
        print(f'  Free Cash Flow: ')
        print(f'  Debt-to-Income Ratio: {capacity["dti_ratio"]:.1%}')
        print(f'  Maximum Sustainable Debt: ')
        print(f'  Recommended Max Limit: ')
        
        # Step 3: Utilization Analysis
        print('\n3️⃣ UTILIZATION ANALYSIS')
        print('-'*40)
        utilization = self._analyze_utilization(customer_data)
        print(f'  Current Utilization: {utilization["current_util"]:.1%}')
        print(f'  Average Utilization (6M): {utilization["avg_util_6m"]:.1%}')
        print(f'  Peak Utilization (12M): {utilization["peak_util_12m"]:.1%}')
        print(f'  Utilization Trend: {utilization["trend"]}')
        print(f'  Unused Credit Cost: ')
        
        # Step 4: Profitability Analysis
        print('\n4️⃣ PROFITABILITY ANALYSIS')
        print('-'*40)
        profitability = self.profitability_analyzer.analyze(customer_data, profile)
        print(f'  Current Revenue: ')
        print(f'  Potential Revenue: ')
        print(f'  Expected Loss: ')
        print(f'  Net Profit: ')
        print(f'  ROA: {profitability["roa"]:.2%}')
        print(f'  Risk-Adjusted Return: {profitability["raroc"]:.2%}')
        
        # Step 5: Optimization
        print('\n5️⃣ LIMIT OPTIMIZATION')
        print('-'*40)
        optimization = self._run_optimization(
            profile, 
            risk_metrics, 
            capacity, 
            profitability,
            objective
        )
        print(f'  Current Limit: ')
        print(f'  Optimal Limit: ')
        print(f'  Change:  ({optimization["change_pct"]:+.1%})')
        print(f'  Expected Revenue Change: ')
        print(f'  Risk-Adjusted Return: {optimization["expected_raroc"]:.2%}')
        
        # Step 6: Stress Testing
        print('\n6️⃣ STRESS TESTING')
        print('-'*40)
        stress_test = self._stress_test_limit(optimization["optimal_limit"], customer_data)
        for scenario, result in stress_test.items():
            icon = '🟢' if result['passes'] else '🔴'
            print(f'  {icon} {scenario}: Loss  | Pass: {result["passes"]}')
        
        # Step 7: Pricing Optimization
        print('\n7️⃣ PRICING OPTIMIZATION')
        print('-'*40)
        pricing = self.risk_pricer.calculate_pricing(
            optimization["optimal_limit"], 
            risk_metrics["pd"],
            risk_metrics["lgd"]
        )
        print(f'  Base Rate: {pricing["base_rate"]:.2%}')
        print(f'  Risk Premium: {pricing["risk_premium"]:.2%}')
        print(f'  Final APR: {pricing["final_apr"]:.2%}')
        print(f'  Expected Yield: {pricing["expected_yield"]:.2%}')
        
        # Step 8: Compliance Check
        print('\n8️⃣ COMPLIANCE CHECK')
        print('-'*40)
        compliance = self.compliance_checker.check_limit(
            optimization["optimal_limit"],
            customer_data
        )
        print(f'  Regulatory Compliance: {"✅ Pass" if compliance["regulatory_pass"] else "❌ Fail"}')
        print(f'  Internal Policy: {"✅ Pass" if compliance["policy_pass"] else "❌ Fail"}')
        print(f'  Concentration Limit: {"✅ Pass" if compliance["concentration_pass"] else "❌ Fail"}')
        print(f'  Capital Adequacy: {"✅ Pass" if compliance["capital_pass"] else "❌ Fail"}')
        
        # Create optimization result
        result = OptimizationResult(
            customer_id=customer_data.get("customer_id", ""),
            current_limit=profile.current_limit,
            recommended_limit=optimization["optimal_limit"],
            change_amount=optimization["change_amount"],
            change_percentage=optimization["change_pct"],
            expected_revenue=optimization["expected_revenue"],
            expected_loss=risk_metrics["expected_loss"],
            risk_adjusted_return=optimization["expected_raroc"],
            constraints_satisfied=compliance["all_pass"],
            recommendations=self._generate_recommendations(
                optimization, 
                risk_metrics, 
                compliance
            )
        )
        
        return result
    
    def optimize_portfolio(self, portfolio_data: List[Dict], constraints: Dict = None):
        '''Optimize credit limits across portfolio'''
        print('\n📊 PORTFOLIO OPTIMIZATION')
        print('='*80)
        print(f'Portfolio Size: {len(portfolio_data)} customers')
        print(f'Total Current Exposure: ')
        
        # Run portfolio optimization
        portfolio_result = self.portfolio_optimizer.optimize(
            portfolio_data,
            constraints or self._get_default_constraints()
        )
        
        print(f'\nOptimization Results:')
        print(f'  Total Optimal Exposure: ')
        print(f'  Expected Portfolio Return: {portfolio_result["expected_return"]:.2%}')
        print(f'  Portfolio Risk (VaR): ')
        print(f'  Sharpe Ratio: {portfolio_result["sharpe_ratio"]:.2f}')
        
        return portfolio_result
    
    def _initialize_params(self):
        '''Initialize optimization parameters'''
        return {
            'min_limit': 1000,
            'max_limit': 1000000,
            'limit_increment': 500,
            'target_utilization': 0.30,
            'max_dti': 0.45,
            'min_profitability': 0.05,
            'confidence_level': 0.95
        }
    
    def _initialize_risk_limits(self):
        '''Initialize risk limits'''
        return {
            'max_pd': 0.10,
            'max_lgd': 0.60,
            'max_expected_loss_rate': 0.03,
            'max_concentration': 0.05,
            'min_coverage_ratio': 1.5
        }
    
    def _initialize_pricing(self):
        '''Initialize pricing matrix'''
        return {
            CustomerSegment.SUPER_PRIME: {'base_rate': 0.039, 'risk_premium': 0.01},
            CustomerSegment.PRIME: {'base_rate': 0.059, 'risk_premium': 0.02},
            CustomerSegment.NEAR_PRIME: {'base_rate': 0.099, 'risk_premium': 0.04},
            CustomerSegment.SUBPRIME: {'base_rate': 0.159, 'risk_premium': 0.08},
            CustomerSegment.CORPORATE: {'base_rate': 0.049, 'risk_premium': 0.015},
            CustomerSegment.SME: {'base_rate': 0.079, 'risk_premium': 0.03}
        }
    
    def _create_customer_profile(self, customer_data):
        '''Create customer profile'''
        return CreditLineProfile(
            customer_id=customer_data.get('customer_id', ''),
            segment=CustomerSegment(customer_data.get('segment', CustomerSegment.NEAR_PRIME.value)),
            current_limit=customer_data.get('current_limit', 0),
            utilized_amount=customer_data.get('utilized_amount', 0),
            available_credit=customer_data.get('current_limit', 0) - customer_data.get('utilized_amount', 0),
            utilization_rate=customer_data.get('utilized_amount', 0) / max(customer_data.get('current_limit', 1), 1),
            risk_score=customer_data.get('risk_score', 50),
            probability_of_default=customer_data.get('pd', 0.02),
            loss_given_default=customer_data.get('lgd', 0.40)
        )
    
    def _assess_customer_risk(self, profile, customer_data):
        '''Assess customer risk metrics'''
        pd = customer_data.get('pd', 0.02)
        lgd = customer_data.get('lgd', 0.40)
        ead = profile.current_limit
        
        expected_loss = pd * lgd * ead
        
        # Risk categorization
        if pd < 0.01:
            risk_category = "Low Risk"
        elif pd < 0.03:
            risk_category = "Medium Risk"
        elif pd < 0.05:
            risk_category = "High Risk"
        else:
            risk_category = "Very High Risk"
        
        return {
            'risk_score': profile.risk_score,
            'pd': pd,
            'lgd': lgd,
            'ead': ead,
            'expected_loss': expected_loss,
            'risk_category': risk_category,
            'unexpected_loss': expected_loss * 2.5  # Simplified
        }
    
    def _analyze_utilization(self, customer_data):
        '''Analyze utilization patterns'''
        current_util = customer_data.get('utilized_amount', 0) / max(customer_data.get('current_limit', 1), 1)
        
        # Historical utilization (simulated)
        historical_utils = customer_data.get('historical_utilization', 
                                            [current_util * np.random.uniform(0.7, 1.3) for _ in range(12)])
        
        avg_util_6m = np.mean(historical_utils[-6:]) if len(historical_utils) >= 6 else current_util
        peak_util_12m = max(historical_utils) if historical_utils else current_util
        
        # Trend analysis
        if len(historical_utils) >= 3:
            recent_trend = historical_utils[-1] - historical_utils[-3]
            if recent_trend > 0.1:
                trend = "Increasing"
            elif recent_trend < -0.1:
                trend = "Decreasing"
            else:
                trend = "Stable"
        else:
            trend = "Insufficient Data"
        
        # Unused credit cost
        unused_amount = customer_data.get('current_limit', 0) - customer_data.get('utilized_amount', 0)
        unused_cost = unused_amount * 0.001  # 10 bps cost of unused credit
        
        return {
            'current_util': current_util,
            'avg_util_6m': avg_util_6m,
            'peak_util_12m': peak_util_12m,
            'trend': trend,
            'unused_cost': unused_cost,
            'historical_utilization': historical_utils
        }
    
    def _run_optimization(self, profile, risk_metrics, capacity, profitability, objective):
        '''Run optimization algorithm'''
        current_limit = profile.current_limit
        
        # Define optimization bounds
        min_limit = max(self.optimization_params['min_limit'], current_limit * 0.5)
        max_limit = min(
            self.optimization_params['max_limit'],
            capacity['recommended_max'],
            current_limit * 2  # Max 100% increase
        )
        
        # Optimization based on objective
        if objective == OptimizationObjective.MAX_PROFIT:
            optimal_limit = self._optimize_for_profit(
                profile, risk_metrics, profitability, min_limit, max_limit
            )
        elif objective == OptimizationObjective.MIN_RISK:
            optimal_limit = self._optimize_for_risk(
                profile, risk_metrics, min_limit, max_limit
            )
        elif objective == OptimizationObjective.MAX_SHARPE:
            optimal_limit = self._optimize_for_sharpe(
                profile, risk_metrics, profitability, min_limit, max_limit
            )
        else:
            optimal_limit = self._balanced_optimization(
                profile, risk_metrics, profitability, capacity, min_limit, max_limit
            )
        
        # Round to increment
        increment = self.optimization_params['limit_increment']
        optimal_limit = round(optimal_limit / increment) * increment
        
        # Calculate changes
        change_amount = optimal_limit - current_limit
        change_pct = change_amount / current_limit if current_limit > 0 else 0
        
        # Expected outcomes
        expected_revenue = optimal_limit * 0.05  # 5% revenue assumption
        expected_loss = optimal_limit * risk_metrics['pd'] * risk_metrics['lgd']
        expected_profit = expected_revenue - expected_loss
        expected_raroc = expected_profit / optimal_limit if optimal_limit > 0 else 0
        
        return {
            'current_limit': current_limit,
            'optimal_limit': optimal_limit,
            'change_amount': change_amount,
            'change_pct': change_pct,
            'expected_revenue': expected_revenue,
            'revenue_change': expected_revenue - (current_limit * 0.05),
            'expected_loss': expected_loss,
            'expected_profit': expected_profit,
            'expected_raroc': expected_raroc
        }
    
    def _optimize_for_profit(self, profile, risk_metrics, profitability, min_limit, max_limit):
        '''Optimize for maximum profit'''
        def profit_function(limit):
            revenue = limit * 0.05 * (1 + profile.utilization_rate)
            loss = limit * risk_metrics['pd'] * risk_metrics['lgd']
            return -(revenue - loss)  # Negative for minimization
        
        result = optimize.minimize_scalar(
            profit_function,
            bounds=(min_limit, max_limit),
            method='bounded'
        )
        
        return result.x
    
    def _optimize_for_risk(self, profile, risk_metrics, min_limit, max_limit):
        '''Optimize for minimum risk'''
        # Lower limit for high-risk customers
        risk_factor = 1 - (risk_metrics['pd'] * 10)  # Scale PD
        risk_factor = max(0.3, min(1.0, risk_factor))
        
        optimal = profile.current_limit * risk_factor
        return max(min_limit, min(max_limit, optimal))
    
    def _optimize_for_sharpe(self, profile, risk_metrics, profitability, min_limit, max_limit):
        '''Optimize for maximum Sharpe ratio'''
        def sharpe_ratio(limit):
            expected_return = limit * 0.05 * (1 + profile.utilization_rate)
            risk = limit * risk_metrics['pd'] * risk_metrics['lgd']
            volatility = np.sqrt(risk) if risk > 0 else 1
            
            return -(expected_return - 0.02 * limit) / volatility  # Negative for minimization
        
        result = optimize.minimize_scalar(
            sharpe_ratio,
            bounds=(min_limit, max_limit),
            method='bounded'
        )
        
        return result.x
    
    def _balanced_optimization(self, profile, risk_metrics, profitability, capacity, min_limit, max_limit):
        '''Balanced optimization considering multiple factors'''
        # Weight factors
        weights = {
            'capacity': 0.3,
            'risk': 0.3,
            'profitability': 0.2,
            'utilization': 0.2
        }
        
        # Calculate component scores
        capacity_limit = capacity['recommended_max']
        risk_limit = profile.current_limit * (1 - risk_metrics['pd'] * 5)
        profit_limit = profile.current_limit * (1 + profitability['roa'] * 2)
        util_limit = profile.current_limit * (1 + profile.utilization_rate)
        
        # Weighted average
        optimal = (
            capacity_limit * weights['capacity'] +
            risk_limit * weights['risk'] +
            profit_limit * weights['profitability'] +
            util_limit * weights['utilization']
        )
        
        return max(min_limit, min(max_limit, optimal))
    
    def _stress_test_limit(self, optimal_limit, customer_data):
        '''Stress test the optimal limit'''
        scenarios = {
            'Base Case': {'pd_mult': 1.0, 'lgd_mult': 1.0},
            'Mild Recession': {'pd_mult': 2.0, 'lgd_mult': 1.2},
            'Severe Recession': {'pd_mult': 3.0, 'lgd_mult': 1.5},
            'Market Crash': {'pd_mult': 5.0, 'lgd_mult': 2.0}
        }
        
        results = {}
        base_pd = customer_data.get('pd', 0.02)
        base_lgd = customer_data.get('lgd', 0.40)
        
        for scenario, factors in scenarios.items():
            stressed_pd = min(1.0, base_pd * factors['pd_mult'])
            stressed_lgd = min(1.0, base_lgd * factors['lgd_mult'])
            
            loss = optimal_limit * stressed_pd * stressed_lgd
            max_acceptable_loss = optimal_limit * 0.05  # 5% loss threshold
            
            results[scenario] = {
                'loss': loss,
                'loss_rate': loss / optimal_limit if optimal_limit > 0 else 0,
                'passes': loss <= max_acceptable_loss
            }
        
        return results
    
    def _generate_recommendations(self, optimization, risk_metrics, compliance):
        '''Generate optimization recommendations'''
        recommendations = []
        
        # Limit change recommendations
        if optimization['change_pct'] > 0.5:
            recommendations.append(f"Large limit increase ({optimization['change_pct']:.0%}) - implement in stages")
        elif optimization['change_pct'] < -0.3:
            recommendations.append(f"Significant limit decrease recommended - monitor customer reaction")
        
        # Risk-based recommendations
        if risk_metrics['pd'] > 0.05:
            recommendations.append("High default risk - consider additional collateral or guarantees")
        
        if risk_metrics['risk_category'] == "Very High Risk":
            recommendations.append("Review account more frequently (monthly)")
        
        # Compliance recommendations
        if not compliance['all_pass']:
            if not compliance['regulatory_pass']:
                recommendations.append("Regulatory constraints prevent full optimization")
            if not compliance['concentration_pass']:
                recommendations.append("Reduce concentration risk through diversification")
        
        # Profitability recommendations
        if optimization['expected_raroc'] < 0.10:
            recommendations.append("Consider repricing to improve risk-adjusted returns")
        
        # General recommendations
        recommendations.append("Monitor utilization patterns for next 3 months")
        recommendations.append("Review credit line quarterly based on performance")
        
        return recommendations[:5]  # Top 5 recommendations
    
    def _get_default_constraints(self):
        '''Get default portfolio constraints'''
        return {
            'total_exposure_limit': 100000000,
            'single_obligor_limit': 5000000,
            'sector_concentration_limit': 0.25,
            'min_portfolio_return': 0.08,
            'max_portfolio_pd': 0.03,
            'regulatory_capital': 10000000
        }

class CreditLimitCalculator:
    '''Calculate credit limits based on capacity'''
    
        def calculate_capacity(self, customer_data):
        '''Calculate customer credit capacity'''
        # Income-based calculation
        annual_income = customer_data.get('annual_income', 100000)  # Default if missing
        monthly_income = annual_income / 12
        
        # Existing obligations
        existing_debt_payments = customer_data.get('monthly_debt_payments', 0)
        
        # Free cash flow
        monthly_expenses = customer_data.get('monthly_expenses', monthly_income * 0.6)
        free_cash_flow = max(0, monthly_income - monthly_expenses - existing_debt_payments)
        
        # Debt service capacity (using 28% DTI limit for new debt)
        max_total_debt_service = monthly_income * 0.28
        available_debt_service = max(0, max_total_debt_service - existing_debt_payments)
        
        # Maximum sustainable debt (assuming 5% monthly payment)
        if available_debt_service > 0:
            max_sustainable_debt = available_debt_service / 0.05
        else:
            max_sustainable_debt = 0
        
        # DTI ratio
        current_dti = existing_debt_payments / monthly_income if monthly_income > 0 else 0
        
        # Recommended maximum (conservative)
        recommended_max = min(
            max_sustainable_debt * 0.8 if max_sustainable_debt > 0 else annual_income * 0.2,
            annual_income * 0.5,
            max(free_cash_flow * 20, 10000)  # Minimum 10k
        )
        
        # Ensure recommended_max is reasonable
        recommended_max = max(10000, recommended_max)  # Minimum 10k
        
        return {
            'monthly_income': monthly_income,
            'debt_service_capacity': available_debt_service,
            'free_cash_flow': free_cash_flow,
            'max_sustainable_debt': max_sustainable_debt,
            'dti_ratio': current_dti,
            'recommended_max': recommended_max
        }

class PortfolioOptimizer:
    '''Optimize credit limits across portfolio'''
    
    def optimize(self, portfolio_data, constraints):
        '''Optimize portfolio allocation'''
        n_customers = len(portfolio_data)
        
        # Extract current state
        current_limits = np.array([c.get('current_limit', 0) for c in portfolio_data])
        pds = np.array([c.get('pd', 0.02) for c in portfolio_data])
        lgds = np.array([c.get('lgd', 0.40) for c in portfolio_data])
        returns = np.array([c.get('expected_return', 0.05) for c in portfolio_data])
        
        # Optimization objective: maximize risk-adjusted return
        def objective(limits):
            portfolio_return = np.sum(limits * returns)
            portfolio_risk = np.sqrt(np.sum((limits * pds * lgds) ** 2))
            
            if portfolio_risk > 0:
                sharpe = (portfolio_return - 0.02 * np.sum(limits)) / portfolio_risk
                return -sharpe  # Negative for minimization
            return 0
        
        # Constraints
        constraint_list = []
        
        # Total exposure constraint
        constraint_list.append({
            'type': 'ineq',
            'fun': lambda x: constraints['total_exposure_limit'] - np.sum(x)
        })
        
        # Single obligor limit
        for i in range(n_customers):
            constraint_list.append({
                'type': 'ineq',
                'fun': lambda x, idx=i: constraints['single_obligor_limit'] - x[idx]
            })
        
        # Minimum return constraint
        constraint_list.append({
            'type': 'ineq',
            'fun': lambda x: np.sum(x * returns) - constraints['min_portfolio_return'] * np.sum(x)
        })
        
        # Bounds for each customer (50% decrease to 100% increase)
        bounds = [(max(0, l * 0.5), min(l * 2, constraints['single_obligor_limit'])) 
                 for l in current_limits]
        
        # Initial guess
        x0 = current_limits
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': 1000}
        )
        
        optimal_limits = result.x if result.success else current_limits
        
        # Calculate portfolio metrics
        total_exposure = np.sum(optimal_limits)
        expected_return = np.sum(optimal_limits * returns) / total_exposure if total_exposure > 0 else 0
        portfolio_el = np.sum(optimal_limits * pds * lgds)
        portfolio_var = portfolio_el * 2.33  # 99% confidence
        
        sharpe_ratio = 0
        if portfolio_var > 0:
            excess_return = expected_return - 0.02
            portfolio_vol = np.sqrt(portfolio_var) / total_exposure if total_exposure > 0 else 1
            sharpe_ratio = excess_return / portfolio_vol
        
        # Create results
        optimized_customers = []
        for i, customer in enumerate(portfolio_data):
            customer_copy = customer.copy()
            customer_copy['optimal_limit'] = optimal_limits[i]
            customer_copy['limit_change'] = optimal_limits[i] - current_limits[i]
            optimized_customers.append(customer_copy)
        
        return {
            'success': result.success,
            'total_exposure': total_exposure,
            'expected_return': expected_return,
            'expected_loss': portfolio_el,
            'portfolio_var': portfolio_var,
            'sharpe_ratio': sharpe_ratio,
            'optimized_customers': optimized_customers
        }

class DynamicLimitAdjuster:
    '''Dynamically adjust credit limits'''
    
    def __init__(self):
        self.adjustment_rules = self._initialize_rules()
        self.trigger_thresholds = self._initialize_triggers()
    
    def _initialize_rules(self):
        '''Initialize adjustment rules'''
        return {
            'high_utilization': {
                'condition': lambda u: u > 0.8,
                'action': 'increase',
                'amount': 0.25
            },
            'low_utilization': {
                'condition': lambda u: u < 0.2,
                'action': 'decrease',
                'amount': 0.10
            },
            'improved_risk': {
                'condition': lambda r: r < 30,
                'action': 'increase',
                'amount': 0.15
            },
            'deteriorated_risk': {
                'condition': lambda r: r > 70,
                'action': 'decrease',
                'amount': 0.20
            }
        }
    
    def _initialize_triggers(self):
        '''Initialize trigger thresholds'''
        return {
            'utilization_change': 0.20,  # 20% change
            'risk_score_change': 10,  # 10 point change
            'payment_delay': 30,  # 30 days
            'review_period': 90  # 90 days
        }
    
    def check_adjustments(self, customer_profile, historical_data):
        '''Check if limit adjustments are needed'''
        adjustments = []
        
        # Check utilization triggers
        current_util = customer_profile.utilization_rate
        if self.adjustment_rules['high_utilization']['condition'](current_util):
            adjustments.append({
                'trigger': 'high_utilization',
                'action': 'increase',
                'percentage': self.adjustment_rules['high_utilization']['amount'],
                'reason': f'Utilization consistently above 80% ({current_util:.0%})'
            })
        elif self.adjustment_rules['low_utilization']['condition'](current_util):
            adjustments.append({
                'trigger': 'low_utilization',
                'action': 'decrease',
                'percentage': self.adjustment_rules['low_utilization']['amount'],
                'reason': f'Utilization consistently below 20% ({current_util:.0%})'
            })
        
        # Check risk score changes
        if 'risk_score_history' in historical_data:
            current_risk = customer_profile.risk_score
            prev_risk = historical_data['risk_score_history'][-1] if historical_data['risk_score_history'] else current_risk
            
            risk_change = current_risk - prev_risk
            
            if abs(risk_change) > self.trigger_thresholds['risk_score_change']:
                if risk_change < 0:  # Risk improved
                    adjustments.append({
                        'trigger': 'improved_risk',
                        'action': 'increase',
                        'percentage': 0.10,
                        'reason': f'Risk score improved by {abs(risk_change)} points'
                    })
                else:  # Risk deteriorated
                    adjustments.append({
                        'trigger': 'deteriorated_risk',
                        'action': 'decrease',
                        'percentage': 0.15,
                        'reason': f'Risk score deteriorated by {risk_change} points'
                    })
        
        return adjustments

class RiskBasedPricer:
    '''Risk-based pricing calculator'''
    
    def calculate_pricing(self, limit, pd, lgd):
        '''Calculate risk-based pricing'''
        # Base rate (risk-free rate)
        base_rate = 0.03  # 3% base rate
        
        # Risk premium calculation
        expected_loss_rate = pd * lgd
        unexpected_loss_rate = expected_loss_rate * 2.33  # 99% confidence
        
        # Risk premium components
        expected_loss_premium = expected_loss_rate
        unexpected_loss_premium = unexpected_loss_rate * 0.5  # Capital charge
        operational_cost = 0.01  # 1% operational cost
        profit_margin = 0.02  # 2% profit margin
        
        risk_premium = (
            expected_loss_premium +
            unexpected_loss_premium +
            operational_cost +
            profit_margin
        )
        
        # Final APR
        final_apr = base_rate + risk_premium
        
        # Competitive adjustment (cap at market rates)
        max_apr = 0.25  # 25% max APR
        final_apr = min(final_apr, max_apr)
        
        # Expected yield (accounting for defaults)
        expected_yield = final_apr * (1 - pd)
        
        return {
            'base_rate': base_rate,
            'risk_premium': risk_premium,
            'expected_loss_premium': expected_loss_premium,
            'unexpected_loss_premium': unexpected_loss_premium,
            'operational_cost': operational_cost,
            'profit_margin': profit_margin,
            'final_apr': final_apr,
            'expected_yield': expected_yield
        }

class ExposureManager:
    '''Manage credit exposure'''
    
    def calculate_exposure(self, customer_data):
        '''Calculate credit exposure metrics'''
        current_limit = customer_data.get('current_limit', 0)
        utilized = customer_data.get('utilized_amount', 0)
        
        # Current exposure
        current_exposure = utilized
        
        # Potential future exposure (credit conversion factor)
        unused_limit = current_limit - utilized
        ccf = 0.75  # 75% credit conversion factor for unused portion
        potential_exposure = unused_limit * ccf
        
        # Exposure at default
        ead = current_exposure + potential_exposure
        
        # Peak exposure (historical)
        historical_utilization = customer_data.get('historical_utilization', [utilized])
        peak_exposure = max(historical_utilization) if historical_utilization else utilized
        
        return {
            'current_exposure': current_exposure,
            'potential_exposure': potential_exposure,
            'ead': ead,
            'peak_exposure': peak_exposure,
            'unused_limit': unused_limit,
            'utilization_rate': utilized / current_limit if current_limit > 0 else 0
        }

class CapitalAllocator:
    '''Allocate capital for credit lines'''
    
    def allocate_capital(self, limit, pd, lgd, segment):
        '''Calculate capital allocation'''
        # Basel III standardized approach
        risk_weights = {
            CustomerSegment.SUPER_PRIME: 0.20,
            CustomerSegment.PRIME: 0.35,
            CustomerSegment.NEAR_PRIME: 0.75,
            CustomerSegment.SUBPRIME: 1.00,
            CustomerSegment.CORPORATE: 1.00,
            CustomerSegment.SME: 0.85
        }
        
        risk_weight = risk_weights.get(segment, 1.0)
        
        # Risk-weighted assets
        rwa = limit * risk_weight
        
        # Capital requirement (8% of RWA)
        regulatory_capital = rwa * 0.08
        
        # Economic capital (based on unexpected loss)
        expected_loss = limit * pd * lgd
        unexpected_loss = expected_loss * 2.33  # 99% confidence
        economic_capital = unexpected_loss
        
        # Buffer capital
        buffer_capital = regulatory_capital * 0.25  # 25% buffer
        
        # Total capital
        total_capital = max(regulatory_capital, economic_capital) + buffer_capital
        
        return {
            'risk_weighted_assets': rwa,
            'regulatory_capital': regulatory_capital,
            'economic_capital': economic_capital,
            'buffer_capital': buffer_capital,
            'total_capital': total_capital,
            'capital_ratio': total_capital / limit if limit > 0 else 0
        }

class ProfitabilityAnalyzer:
    '''Analyze profitability of credit lines'''
    
    def analyze(self, customer_data, profile):
        '''Analyze credit line profitability'''
        limit = profile.current_limit
        utilized = profile.utilized_amount
        
        # Revenue components
        interest_revenue = utilized * customer_data.get('interest_rate', 0.10)
        fee_revenue = limit * 0.005  # 50 bps annual fee
        transaction_revenue = customer_data.get('transaction_revenue', utilized * 0.02)
        
        total_revenue = interest_revenue + fee_revenue + transaction_revenue
        
        # Cost components
        funding_cost = utilized * 0.03  # 3% funding cost
        operational_cost = limit * 0.01  # 1% operational cost
        expected_loss = limit * profile.probability_of_default * profile.loss_given_default
        capital_cost = limit * 0.08 * 0.10  # 8% capital * 10% cost of capital
        
        total_cost = funding_cost + operational_cost + expected_loss + capital_cost
        
        # Profitability metrics
        net_profit = total_revenue - total_cost
        roa = net_profit / limit if limit > 0 else 0
        
        # Risk-adjusted metrics
        risk_adjusted_revenue = total_revenue - expected_loss
        raroc = net_profit / (limit * 0.08) if limit > 0 else 0  # Return on allocated capital
        
        # Potential revenue with optimal utilization
        optimal_utilization = 0.70  # 70% target utilization
        potential_revenue = limit * optimal_utilization * customer_data.get('interest_rate', 0.10) + fee_revenue
        
        return {
            'current_revenue': total_revenue,
            'interest_revenue': interest_revenue,
            'fee_revenue': fee_revenue,
            'transaction_revenue': transaction_revenue,
            'total_cost': total_cost,
            'funding_cost': funding_cost,
            'operational_cost': operational_cost,
            'expected_loss': expected_loss,
            'capital_cost': capital_cost,
            'net_profit': net_profit,
            'roa': roa,
            'risk_adjusted_revenue': risk_adjusted_revenue,
            'raroc': raroc,
            'potential_revenue': potential_revenue
        }

class ComplianceChecker:
    '''Check compliance for credit limits'''
    
    def check_limit(self, recommended_limit, customer_data):
        '''Check if recommended limit meets compliance'''
        checks = {
            'regulatory_pass': True,
            'policy_pass': True,
            'concentration_pass': True,
            'capital_pass': True,
            'all_pass': True
        }
        
        # Regulatory limits
        max_dti_limit = customer_data.get('annual_income', 0) * 0.45 / 12 / 0.05  # 45% DTI
        if recommended_limit > max_dti_limit:
            checks['regulatory_pass'] = False
        
        # Internal policy limits
        segment = customer_data.get('segment', CustomerSegment.NEAR_PRIME.value)
        policy_limits = {
            CustomerSegment.SUPER_PRIME.value: 500000,
            CustomerSegment.PRIME.value: 250000,
            CustomerSegment.NEAR_PRIME.value: 100000,
            CustomerSegment.SUBPRIME.value: 50000
        }
        
        if recommended_limit > policy_limits.get(segment, 100000):
            checks['policy_pass'] = False
        
        # Concentration limits
        portfolio_total = customer_data.get('portfolio_total', 100000000)
        if recommended_limit / portfolio_total > 0.05:  # 5% concentration
            checks['concentration_pass'] = False
        
        # Capital adequacy
        required_capital = recommended_limit * 0.08
        available_capital = customer_data.get('available_capital', 10000000)
        if required_capital > available_capital * 0.10:  # Max 10% of available capital
            checks['capital_pass'] = False
        
        # Overall pass/fail
        checks['all_pass'] = all([
            checks['regulatory_pass'],
            checks['policy_pass'],
            checks['concentration_pass'],
            checks['capital_pass']
        ])
        
        return checks

# Demonstrate system
if __name__ == '__main__':
    print('💳 CREDIT LINE OPTIMIZATION - ULTRAPLATFORM')
    print('='*80)
    
    # Create sample customer
    customer_data = {
        'customer_id': 'CUST_001',
        'segment': CustomerSegment.PRIME.value,
        'current_limit': 50000,
        'utilized_amount': 15000,
        'annual_income': 120000,
        'monthly_debt_payments': 2000,
        'monthly_expenses': 4000,
        
        # Risk metrics
        'risk_score': 65,
        'pd': 0.015,  # 1.5% probability of default
        'lgd': 0.35,  # 35% loss given default
        
        # Historical data
        'historical_utilization': [0.25, 0.30, 0.28, 0.35, 0.32, 0.30, 0.28, 0.31, 0.30, 0.29, 0.30, 0.30],
        'payment_history_score': 0.95,
        
        # Pricing
        'interest_rate': 0.089,  # 8.9% APR
        
        # Portfolio context
        'portfolio_total': 100000000,
        'available_capital': 10000000
    }
    
    # Initialize system
    optimizer = CreditLineOptimization()
    
    # Run optimization
    print('\n🎯 OPTIMIZING CREDIT LINE')
    print('='*80 + '\n')
    
    result = optimizer.optimize_credit_line(
        customer_data,
        objective=OptimizationObjective.MAX_SHARPE
    )
    
    # Show results
    print('\n' + '='*80)
    print('OPTIMIZATION RESULTS')
    print('='*80)
    print(f'Customer ID: {result.customer_id}')
    print(f'Current Limit: ')
    print(f'Recommended Limit: ')
    print(f'Change:  ({result.change_percentage:+.1%})')
    print(f'Expected Revenue: ')
    print(f'Expected Loss: ')
    print(f'Risk-Adjusted Return: {result.risk_adjusted_return:.2%}')
    print(f'Constraints Satisfied: {"✅ Yes" if result.constraints_satisfied else "❌ No"}')
    
    if result.recommendations:
        print('\n💡 RECOMMENDATIONS')
        print('-'*40)
        for rec in result.recommendations:
            print(f'  • {rec}')
    
    # Portfolio optimization demo
    portfolio = [customer_data.copy() for _ in range(5)]
    for i, cust in enumerate(portfolio):
        cust['customer_id'] = f'CUST_{i+1:03d}'
        cust['current_limit'] = np.random.uniform(20000, 100000)
        cust['pd'] = np.random.uniform(0.01, 0.05)
    
    portfolio_result = optimizer.optimize_portfolio(portfolio)
    
    print('\n✅ Credit Line Optimization Complete!')

