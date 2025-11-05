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
from scipy import stats
import warnings

class RiskType(Enum):
    MARKET_RISK = 'market_risk'
    CREDIT_RISK = 'credit_risk'
    LIQUIDITY_RISK = 'liquidity_risk'
    OPERATIONAL_RISK = 'operational_risk'
    COUNTERPARTY_RISK = 'counterparty_risk'
    CONCENTRATION_RISK = 'concentration_risk'
    SYSTEMIC_RISK = 'systemic_risk'
    MODEL_RISK = 'model_risk'
    REGULATORY_RISK = 'regulatory_risk'
    REPUTATIONAL_RISK = 'reputational_risk'

class RiskLevel(Enum):
    MINIMAL = 'minimal'
    LOW = 'low'
    MODERATE = 'moderate'
    HIGH = 'high'
    SEVERE = 'severe'
    CRITICAL = 'critical'

class RiskMetric(Enum):
    VALUE_AT_RISK = 'var'
    CONDITIONAL_VAR = 'cvar'
    STRESS_TEST = 'stress_test'
    BETA = 'beta'
    SHARPE_RATIO = 'sharpe_ratio'
    MAX_DRAWDOWN = 'max_drawdown'
    VOLATILITY = 'volatility'
    CORRELATION = 'correlation'
    EXPOSURE = 'exposure'

@dataclass
class RiskProfile:
    '''Risk profile for an entity'''
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity_type: str = ''  # portfolio, position, trader, strategy
    timestamp: datetime = field(default_factory=datetime.now)
    risk_scores: Dict[RiskType, float] = field(default_factory=dict)
    aggregate_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.MODERATE
    risk_metrics: Dict = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class RiskScoreResult:
    '''Result of risk scoring'''
    score_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    entity_id: str = ''
    risk_type: RiskType = RiskType.MARKET_RISK
    raw_score: float = 0.0
    normalized_score: float = 0.0  # 0-100 scale
    confidence: float = 0.0
    factors: Dict = field(default_factory=dict)
    threshold_breach: bool = False
    alert_level: str = 'none'

class RiskScoringEngine:
    '''Comprehensive Risk Scoring Engine for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Risk Scoring Engine'
        self.version = '2.0'
        
        # Initialize components
        self.market_risk_scorer = MarketRiskScorer()
        self.credit_risk_scorer = CreditRiskScorer()
        self.liquidity_risk_scorer = LiquidityRiskScorer()
        self.operational_risk_scorer = OperationalRiskScorer()
        self.counterparty_risk_scorer = CounterpartyRiskScorer()
        self.concentration_risk_scorer = ConcentrationRiskScorer()
        self.systemic_risk_scorer = SystemicRiskScorer()
        self.model_risk_scorer = ModelRiskScorer()
        
        # Risk aggregator and monitor
        self.risk_aggregator = RiskAggregator()
        self.risk_monitor = RiskMonitor()
        self.stress_tester = StressTester()
        self.risk_calculator = RiskCalculator()
        
        # Configuration
        self.risk_weights = self._initialize_risk_weights()
        self.risk_thresholds = self._initialize_risk_thresholds()
        
        print('🎯 Risk Scoring Engine initialized')
    
    def score_risk(self, entity_data: Dict, risk_types: List[RiskType] = None):
        '''Score risk for an entity across multiple dimensions'''
        print('RISK SCORING ENGINE')
        print('='*80)
        print(f'Entity Type: {entity_data.get("type", "unknown")}')
        print(f'Entity ID: {entity_data.get("id", "N/A")}')
        print()
        
        # Default to all risk types if none specified
        if risk_types is None:
            risk_types = list(RiskType)
        
        risk_scores = {}
        
        # Step 1: Market Risk
        if RiskType.MARKET_RISK in risk_types:
            print('1️⃣ MARKET RISK SCORING')
            print('-'*40)
            market_score = self.market_risk_scorer.score(entity_data)
            risk_scores[RiskType.MARKET_RISK] = market_score
            print(f'  VaR Score: {market_score.factors.get("var_score", 0):.1f}')
            print(f'  Volatility Score: {market_score.factors.get("volatility_score", 0):.1f}')
            print(f'  Beta Score: {market_score.factors.get("beta_score", 0):.1f}')
            print(f'  Drawdown Score: {market_score.factors.get("drawdown_score", 0):.1f}')
            print(f'  📊 Market Risk Score: {market_score.normalized_score:.1f}/100')
        
        # Step 2: Credit Risk
        if RiskType.CREDIT_RISK in risk_types:
            print('\n2️⃣ CREDIT RISK SCORING')
            print('-'*40)
            credit_score = self.credit_risk_scorer.score(entity_data)
            risk_scores[RiskType.CREDIT_RISK] = credit_score
            print(f'  Default Probability: {credit_score.factors.get("pd", 0):.2%}')
            print(f'  Loss Given Default: {credit_score.factors.get("lgd", 0):.2%}')
            print(f'  Exposure at Default: ')
            print(f'  Expected Loss: ')
            print(f'  📊 Credit Risk Score: {credit_score.normalized_score:.1f}/100')
        
        # Step 3: Liquidity Risk
        if RiskType.LIQUIDITY_RISK in risk_types:
            print('\n3️⃣ LIQUIDITY RISK SCORING')
            print('-'*40)
            liquidity_score = self.liquidity_risk_scorer.score(entity_data)
            risk_scores[RiskType.LIQUIDITY_RISK] = liquidity_score
            print(f'  Bid-Ask Spread: {liquidity_score.factors.get("spread", 0):.3f}')
            print(f'  Market Depth: {liquidity_score.factors.get("depth", 0):.1f}')
            print(f'  Liquidity Ratio: {liquidity_score.factors.get("liquidity_ratio", 0):.2f}')
            print(f'  Days to Liquidate: {liquidity_score.factors.get("liquidation_days", 0):.1f}')
            print(f'  📊 Liquidity Risk Score: {liquidity_score.normalized_score:.1f}/100')
        
        # Step 4: Operational Risk
        if RiskType.OPERATIONAL_RISK in risk_types:
            print('\n4️⃣ OPERATIONAL RISK SCORING')
            print('-'*40)
            operational_score = self.operational_risk_scorer.score(entity_data)
            risk_scores[RiskType.OPERATIONAL_RISK] = operational_score
            print(f'  Process Risk: {operational_score.factors.get("process_risk", 0):.1f}')
            print(f'  System Risk: {operational_score.factors.get("system_risk", 0):.1f}')
            print(f'  Human Risk: {operational_score.factors.get("human_risk", 0):.1f}')
            print(f'  External Risk: {operational_score.factors.get("external_risk", 0):.1f}')
            print(f'  📊 Operational Risk Score: {operational_score.normalized_score:.1f}/100')
        
        # Step 5: Counterparty Risk
        if RiskType.COUNTERPARTY_RISK in risk_types:
            print('\n5️⃣ COUNTERPARTY RISK SCORING')
            print('-'*40)
            counterparty_score = self.counterparty_risk_scorer.score(entity_data)
            risk_scores[RiskType.COUNTERPARTY_RISK] = counterparty_score
            print(f'  Counterparty Rating: {counterparty_score.factors.get("rating", "BBB")}')
            print(f'  Exposure: ')
            print(f'  Collateral Coverage: {counterparty_score.factors.get("collateral", 0):.1%}')
            print(f'  Settlement Risk: {counterparty_score.factors.get("settlement_risk", 0):.1f}')
            print(f'  📊 Counterparty Risk Score: {counterparty_score.normalized_score:.1f}/100')
        
        # Step 6: Concentration Risk
        if RiskType.CONCENTRATION_RISK in risk_types:
            print('\n6️⃣ CONCENTRATION RISK SCORING')
            print('-'*40)
            concentration_score = self.concentration_risk_scorer.score(entity_data)
            risk_scores[RiskType.CONCENTRATION_RISK] = concentration_score
            print(f'  Position Concentration: {concentration_score.factors.get("position_conc", 0):.1%}')
            print(f'  Sector Concentration: {concentration_score.factors.get("sector_conc", 0):.1%}')
            print(f'  Geographic Concentration: {concentration_score.factors.get("geo_conc", 0):.1%}')
            print(f'  Herfindahl Index: {concentration_score.factors.get("hhi", 0):.0f}')
            print(f'  📊 Concentration Risk Score: {concentration_score.normalized_score:.1f}/100')
        
        # Step 7: Aggregate Risk
        print('\n7️⃣ AGGREGATE RISK ASSESSMENT')
        print('-'*40)
        aggregate_result = self.risk_aggregator.aggregate_risks(risk_scores)
        print(f'  Total Risk Score: {aggregate_result["total_score"]:.1f}/100')
        print(f'  Risk Level: {aggregate_result["risk_level"].value.upper()}')
        print(f'  Confidence: {aggregate_result["confidence"]:.1%}')
        
        # Step 8: Stress Testing
        print('\n8️⃣ STRESS TEST RESULTS')
        print('-'*40)
        stress_results = self.stress_tester.run_stress_test(entity_data, risk_scores)
        for scenario, result in stress_results.items():
            icon = '🔴' if result['impact'] > 30 else '🟡' if result['impact'] > 15 else '🟢'
            print(f'  {icon} {scenario}: {result["impact"]:.1f}% impact')
        
        # Create risk profile
        risk_profile = RiskProfile(
            entity_id=entity_data.get('id', ''),
            entity_type=entity_data.get('type', ''),
            risk_scores={k: v.normalized_score for k, v in risk_scores.items()},
            aggregate_score=aggregate_result['total_score'],
            risk_level=aggregate_result['risk_level'],
            risk_metrics=self._calculate_risk_metrics(entity_data),
            violations=self._check_violations(risk_scores),
            recommendations=self._generate_recommendations(risk_scores, aggregate_result)
        )
        
        return risk_profile
    
    def _initialize_risk_weights(self):
        '''Initialize risk type weights'''
        return {
            RiskType.MARKET_RISK: 0.25,
            RiskType.CREDIT_RISK: 0.20,
            RiskType.LIQUIDITY_RISK: 0.15,
            RiskType.OPERATIONAL_RISK: 0.15,
            RiskType.COUNTERPARTY_RISK: 0.10,
            RiskType.CONCENTRATION_RISK: 0.10,
            RiskType.SYSTEMIC_RISK: 0.05
        }
    
    def _initialize_risk_thresholds(self):
        '''Initialize risk thresholds'''
        return {
            'minimal': (0, 20),
            'low': (20, 40),
            'moderate': (40, 60),
            'high': (60, 75),
            'severe': (75, 90),
            'critical': (90, 100)
        }
    
    def _calculate_risk_metrics(self, entity_data):
        '''Calculate key risk metrics'''
        return self.risk_calculator.calculate_metrics(entity_data)
    
    def _check_violations(self, risk_scores):
        '''Check for risk limit violations'''
        violations = []
        
        for risk_type, score in risk_scores.items():
            if score.normalized_score > 75:
                violations.append(f'{risk_type.value}: Score exceeds threshold ({score.normalized_score:.1f})')
            
            if score.threshold_breach:
                violations.append(f'{risk_type.value}: Threshold breach detected')
        
        return violations
    
    def _generate_recommendations(self, risk_scores, aggregate_result):
        '''Generate risk mitigation recommendations'''
        recommendations = []
        
        # High-level recommendations based on aggregate score
        if aggregate_result['total_score'] > 75:
            recommendations.append('URGENT: Immediate risk reduction required')
            recommendations.append('Consider closing high-risk positions')
        elif aggregate_result['total_score'] > 60:
            recommendations.append('Reduce overall portfolio risk exposure')
            recommendations.append('Review and tighten risk limits')
        
        # Specific recommendations for high-risk areas
        for risk_type, score in risk_scores.items():
            if score.normalized_score > 70:
                if risk_type == RiskType.MARKET_RISK:
                    recommendations.append('Hedge market exposure with derivatives')
                    recommendations.append('Reduce position sizes in volatile assets')
                elif risk_type == RiskType.CREDIT_RISK:
                    recommendations.append('Review counterparty creditworthiness')
                    recommendations.append('Increase collateral requirements')
                elif risk_type == RiskType.LIQUIDITY_RISK:
                    recommendations.append('Increase cash reserves')
                    recommendations.append('Improve asset liquidity profile')
                elif risk_type == RiskType.CONCENTRATION_RISK:
                    recommendations.append('Diversify portfolio holdings')
                    recommendations.append('Reduce largest position sizes')
        
        return recommendations[:5]  # Top 5 recommendations

class MarketRiskScorer:
    '''Score market risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate market risk score'''
        factors = {}
        
        # VaR scoring
        var = entity_data.get('var_95', 0)
        portfolio_value = entity_data.get('portfolio_value', 1000000)
        var_percentage = (var / portfolio_value) * 100 if portfolio_value > 0 else 0
        
        var_score = self._score_var(var_percentage)
        factors['var_score'] = var_score
        factors['var_percentage'] = var_percentage
        
        # Volatility scoring
        volatility = entity_data.get('volatility', 0.15)
        volatility_score = self._score_volatility(volatility)
        factors['volatility_score'] = volatility_score
        factors['volatility'] = volatility
        
        # Beta scoring
        beta = entity_data.get('beta', 1.0)
        beta_score = self._score_beta(beta)
        factors['beta_score'] = beta_score
        factors['beta'] = beta
        
        # Drawdown scoring
        max_drawdown = entity_data.get('max_drawdown', 0.10)
        drawdown_score = self._score_drawdown(max_drawdown)
        factors['drawdown_score'] = drawdown_score
        factors['max_drawdown'] = max_drawdown
        
        # Calculate weighted score
        raw_score = (
            var_score * 0.35 +
            volatility_score * 0.25 +
            beta_score * 0.20 +
            drawdown_score * 0.20
        )
        
        # Normalize to 0-100 scale
        normalized_score = min(100, max(0, raw_score))
        
        # Check threshold breach
        threshold_breach = normalized_score > 75
        
        # Determine alert level
        if normalized_score > 90:
            alert_level = 'critical'
        elif normalized_score > 75:
            alert_level = 'high'
        elif normalized_score > 60:
            alert_level = 'medium'
        else:
            alert_level = 'low'
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.MARKET_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.85,
            factors=factors,
            threshold_breach=threshold_breach,
            alert_level=alert_level
        )
    
    def _score_var(self, var_percentage):
        '''Score VaR as percentage of portfolio'''
        if var_percentage < 2:
            return 10
        elif var_percentage < 5:
            return 30
        elif var_percentage < 10:
            return 50
        elif var_percentage < 15:
            return 70
        elif var_percentage < 20:
            return 85
        else:
            return 100
    
    def _score_volatility(self, volatility):
        '''Score volatility'''
        # Annual volatility
        if volatility < 0.10:
            return 10
        elif volatility < 0.15:
            return 25
        elif volatility < 0.25:
            return 45
        elif volatility < 0.35:
            return 65
        elif volatility < 0.50:
            return 80
        else:
            return 100
    
    def _score_beta(self, beta):
        '''Score market beta'''
        abs_beta = abs(beta)
        if abs_beta < 0.5:
            return 20
        elif abs_beta < 1.0:
            return 40
        elif abs_beta < 1.5:
            return 60
        elif abs_beta < 2.0:
            return 75
        else:
            return 90
    
    def _score_drawdown(self, drawdown):
        '''Score maximum drawdown'''
        if drawdown < 0.05:
            return 10
        elif drawdown < 0.10:
            return 25
        elif drawdown < 0.20:
            return 45
        elif drawdown < 0.30:
            return 65
        elif drawdown < 0.40:
            return 80
        else:
            return 100

class CreditRiskScorer:
    '''Score credit risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate credit risk score'''
        factors = {}
        
        # Probability of Default
        pd = entity_data.get('probability_of_default', 0.02)
        pd_score = self._score_pd(pd)
        factors['pd'] = pd
        factors['pd_score'] = pd_score
        
        # Loss Given Default
        lgd = entity_data.get('loss_given_default', 0.40)
        lgd_score = self._score_lgd(lgd)
        factors['lgd'] = lgd
        factors['lgd_score'] = lgd_score
        
        # Exposure at Default
        ead = entity_data.get('exposure_at_default', 1000000)
        portfolio_value = entity_data.get('portfolio_value', 1000000)
        ead_percentage = (ead / portfolio_value) if portfolio_value > 0 else 0
        ead_score = self._score_ead(ead_percentage)
        factors['ead'] = ead
        factors['ead_score'] = ead_score
        
        # Expected Loss
        expected_loss = pd * lgd * ead
        factors['expected_loss'] = expected_loss
        
        # Credit rating
        rating = entity_data.get('credit_rating', 'BBB')
        rating_score = self._score_rating(rating)
        factors['rating'] = rating
        factors['rating_score'] = rating_score
        
        # Calculate weighted score
        raw_score = (
            pd_score * 0.30 +
            lgd_score * 0.20 +
            ead_score * 0.25 +
            rating_score * 0.25
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.CREDIT_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.80,
            factors=factors,
            threshold_breach=normalized_score > 70,
            alert_level='high' if normalized_score > 70 else 'medium'
        )
    
    def _score_pd(self, pd):
        '''Score probability of default'''
        if pd < 0.001:
            return 5
        elif pd < 0.005:
            return 15
        elif pd < 0.01:
            return 30
        elif pd < 0.03:
            return 50
        elif pd < 0.05:
            return 70
        elif pd < 0.10:
            return 85
        else:
            return 100
    
    def _score_lgd(self, lgd):
        '''Score loss given default'''
        if lgd < 0.20:
            return 20
        elif lgd < 0.40:
            return 40
        elif lgd < 0.60:
            return 60
        elif lgd < 0.80:
            return 80
        else:
            return 100
    
    def _score_ead(self, ead_percentage):
        '''Score exposure at default'''
        if ead_percentage < 0.10:
            return 10
        elif ead_percentage < 0.25:
            return 30
        elif ead_percentage < 0.50:
            return 50
        elif ead_percentage < 0.75:
            return 70
        else:
            return 90
    
    def _score_rating(self, rating):
        '''Score credit rating'''
        rating_scores = {
            'AAA': 5, 'AA+': 10, 'AA': 15, 'AA-': 20,
            'A+': 25, 'A': 30, 'A-': 35,
            'BBB+': 40, 'BBB': 50, 'BBB-': 60,
            'BB+': 70, 'BB': 75, 'BB-': 80,
            'B+': 85, 'B': 90, 'B-': 95,
            'CCC': 100, 'CC': 100, 'C': 100, 'D': 100
        }
        return rating_scores.get(rating, 50)

class LiquidityRiskScorer:
    '''Score liquidity risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate liquidity risk score'''
        factors = {}
        
        # Bid-ask spread
        spread = entity_data.get('bid_ask_spread', 0.001)
        spread_score = self._score_spread(spread)
        factors['spread'] = spread
        factors['spread_score'] = spread_score
        
        # Market depth
        depth = entity_data.get('market_depth', 100)
        depth_score = self._score_depth(depth)
        factors['depth'] = depth
        factors['depth_score'] = depth_score
        
        # Liquidity ratio
        liquidity_ratio = entity_data.get('liquidity_ratio', 0.20)
        liquidity_score = self._score_liquidity_ratio(liquidity_ratio)
        factors['liquidity_ratio'] = liquidity_ratio
        factors['liquidity_score'] = liquidity_score
        
        # Days to liquidate
        liquidation_days = entity_data.get('liquidation_days', 1)
        liquidation_score = self._score_liquidation_time(liquidation_days)
        factors['liquidation_days'] = liquidation_days
        factors['liquidation_score'] = liquidation_score
        
        # Calculate weighted score
        raw_score = (
            spread_score * 0.25 +
            depth_score * 0.25 +
            liquidity_score * 0.30 +
            liquidation_score * 0.20
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.LIQUIDITY_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.75,
            factors=factors,
            threshold_breach=normalized_score > 65,
            alert_level='high' if normalized_score > 65 else 'low'
        )
    
    def _score_spread(self, spread):
        '''Score bid-ask spread'''
        # As percentage
        spread_pct = spread * 100
        if spread_pct < 0.1:
            return 10
        elif spread_pct < 0.5:
            return 30
        elif spread_pct < 1.0:
            return 50
        elif spread_pct < 2.0:
            return 70
        else:
            return 90
    
    def _score_depth(self, depth):
        '''Score market depth'''
        if depth > 1000:
            return 10
        elif depth > 500:
            return 30
        elif depth > 100:
            return 50
        elif depth > 50:
            return 70
        else:
            return 90
    
    def _score_liquidity_ratio(self, ratio):
        '''Score liquidity ratio'''
        if ratio > 0.50:
            return 10
        elif ratio > 0.30:
            return 30
        elif ratio > 0.20:
            return 50
        elif ratio > 0.10:
            return 70
        else:
            return 90
    
    def _score_liquidation_time(self, days):
        '''Score time to liquidate'''
        if days < 1:
            return 10
        elif days < 3:
            return 30
        elif days < 7:
            return 50
        elif days < 14:
            return 70
        else:
            return 90

class OperationalRiskScorer:
    '''Score operational risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate operational risk score'''
        factors = {}
        
        # Process risk
        process_failures = entity_data.get('process_failures', 0)
        process_score = self._score_process_risk(process_failures)
        factors['process_risk'] = process_score
        factors['process_failures'] = process_failures
        
        # System risk
        system_downtime = entity_data.get('system_downtime_hours', 0)
        system_score = self._score_system_risk(system_downtime)
        factors['system_risk'] = system_score
        factors['system_downtime'] = system_downtime
        
        # Human risk
        human_errors = entity_data.get('human_errors', 0)
        human_score = self._score_human_risk(human_errors)
        factors['human_risk'] = human_score
        factors['human_errors'] = human_errors
        
        # External risk
        external_incidents = entity_data.get('external_incidents', 0)
        external_score = self._score_external_risk(external_incidents)
        factors['external_risk'] = external_score
        factors['external_incidents'] = external_incidents
        
        # Calculate weighted score
        raw_score = (
            process_score * 0.30 +
            system_score * 0.30 +
            human_score * 0.25 +
            external_score * 0.15
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.OPERATIONAL_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.70,
            factors=factors,
            threshold_breach=normalized_score > 60,
            alert_level='medium'
        )
    
    def _score_process_risk(self, failures):
        '''Score process failures'''
        if failures == 0:
            return 5
        elif failures < 2:
            return 20
        elif failures < 5:
            return 40
        elif failures < 10:
            return 60
        elif failures < 20:
            return 80
        else:
            return 100
    
    def _score_system_risk(self, downtime_hours):
        '''Score system downtime'''
        if downtime_hours == 0:
            return 5
        elif downtime_hours < 1:
            return 20
        elif downtime_hours < 4:
            return 40
        elif downtime_hours < 8:
            return 60
        elif downtime_hours < 24:
            return 80
        else:
            return 100
    
    def _score_human_risk(self, errors):
        '''Score human errors'''
        if errors == 0:
            return 5
        elif errors < 3:
            return 25
        elif errors < 10:
            return 45
        elif errors < 20:
            return 65
        else:
            return 85
    
    def _score_external_risk(self, incidents):
        '''Score external incidents'''
        if incidents == 0:
            return 10
        elif incidents < 2:
            return 30
        elif incidents < 5:
            return 50
        elif incidents < 10:
            return 70
        else:
            return 90

class CounterpartyRiskScorer:
    '''Score counterparty risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate counterparty risk score'''
        factors = {}
        
        # Counterparty rating
        rating = entity_data.get('counterparty_rating', 'BBB')
        rating_score = self._score_counterparty_rating(rating)
        factors['rating'] = rating
        factors['rating_score'] = rating_score
        
        # Exposure
        exposure = entity_data.get('counterparty_exposure', 0)
        portfolio_value = entity_data.get('portfolio_value', 1000000)
        exposure_pct = (exposure / portfolio_value) if portfolio_value > 0 else 0
        exposure_score = self._score_exposure(exposure_pct)
        factors['exposure'] = exposure
        factors['exposure_score'] = exposure_score
        
        # Collateral coverage
        collateral = entity_data.get('collateral_coverage', 0)
        collateral_score = self._score_collateral(collateral)
        factors['collateral'] = collateral
        factors['collateral_score'] = collateral_score
        
        # Settlement risk
        settlement_fails = entity_data.get('settlement_failures', 0)
        settlement_score = self._score_settlement_risk(settlement_fails)
        factors['settlement_risk'] = settlement_score
        factors['settlement_failures'] = settlement_fails
        
        # Calculate weighted score
        raw_score = (
            rating_score * 0.35 +
            exposure_score * 0.30 +
            collateral_score * 0.20 +
            settlement_score * 0.15
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.COUNTERPARTY_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.75,
            factors=factors,
            threshold_breach=normalized_score > 70,
            alert_level='high' if normalized_score > 70 else 'low'
        )
    
    def _score_counterparty_rating(self, rating):
        '''Score counterparty credit rating'''
        rating_scores = {
            'AAA': 5, 'AA': 10, 'A': 20,
            'BBB': 40, 'BB': 60, 'B': 80,
            'CCC': 95, 'CC': 100, 'D': 100
        }
        return rating_scores.get(rating, 50)
    
    def _score_exposure(self, exposure_pct):
        '''Score exposure as percentage of portfolio'''
        if exposure_pct < 0.05:
            return 10
        elif exposure_pct < 0.10:
            return 25
        elif exposure_pct < 0.20:
            return 45
        elif exposure_pct < 0.30:
            return 65
        else:
            return 85
    
    def _score_collateral(self, collateral_coverage):
        '''Score collateral coverage'''
        if collateral_coverage > 1.5:
            return 10
        elif collateral_coverage > 1.2:
            return 25
        elif collateral_coverage > 1.0:
            return 40
        elif collateral_coverage > 0.8:
            return 60
        elif collateral_coverage > 0.5:
            return 80
        else:
            return 100
    
    def _score_settlement_risk(self, failures):
        '''Score settlement failures'''
        if failures == 0:
            return 5
        elif failures < 2:
            return 25
        elif failures < 5:
            return 50
        elif failures < 10:
            return 75
        else:
            return 95

class ConcentrationRiskScorer:
    '''Score concentration risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate concentration risk score'''
        factors = {}
        
        # Position concentration
        largest_position = entity_data.get('largest_position_pct', 0)
        position_score = self._score_position_concentration(largest_position)
        factors['position_conc'] = largest_position
        factors['position_score'] = position_score
        
        # Sector concentration
        largest_sector = entity_data.get('largest_sector_pct', 0)
        sector_score = self._score_sector_concentration(largest_sector)
        factors['sector_conc'] = largest_sector
        factors['sector_score'] = sector_score
        
        # Geographic concentration
        largest_geo = entity_data.get('largest_geo_pct', 0)
        geo_score = self._score_geo_concentration(largest_geo)
        factors['geo_conc'] = largest_geo
        factors['geo_score'] = geo_score
        
        # Herfindahl-Hirschman Index
        hhi = entity_data.get('herfindahl_index', 0)
        hhi_score = self._score_hhi(hhi)
        factors['hhi'] = hhi
        factors['hhi_score'] = hhi_score
        
        # Calculate weighted score
        raw_score = (
            position_score * 0.35 +
            sector_score * 0.25 +
            geo_score * 0.20 +
            hhi_score * 0.20
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.CONCENTRATION_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.80,
            factors=factors,
            threshold_breach=normalized_score > 65,
            alert_level='medium'
        )
    
    def _score_position_concentration(self, concentration):
        '''Score single position concentration'''
        if concentration < 0.05:
            return 5
        elif concentration < 0.10:
            return 20
        elif concentration < 0.20:
            return 40
        elif concentration < 0.30:
            return 60
        elif concentration < 0.40:
            return 80
        else:
            return 100
    
    def _score_sector_concentration(self, concentration):
        '''Score sector concentration'''
        if concentration < 0.20:
            return 10
        elif concentration < 0.30:
            return 25
        elif concentration < 0.40:
            return 45
        elif concentration < 0.50:
            return 65
        else:
            return 85
    
    def _score_geo_concentration(self, concentration):
        '''Score geographic concentration'''
        if concentration < 0.30:
            return 10
        elif concentration < 0.50:
            return 30
        elif concentration < 0.70:
            return 50
        elif concentration < 0.85:
            return 70
        else:
            return 90
    
    def _score_hhi(self, hhi):
        '''Score Herfindahl-Hirschman Index'''
        if hhi < 1000:
            return 10  # Highly diversified
        elif hhi < 1500:
            return 30  # Moderately concentrated
        elif hhi < 2500:
            return 50  # Concentrated
        elif hhi < 5000:
            return 70  # Highly concentrated
        else:
            return 90  # Extremely concentrated

class SystemicRiskScorer:
    '''Score systemic risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate systemic risk score'''
        factors = {}
        
        # Market correlation
        market_correlation = entity_data.get('market_correlation', 0.5)
        correlation_score = market_correlation * 100
        factors['market_correlation'] = market_correlation
        factors['correlation_score'] = correlation_score
        
        # Interconnectedness
        interconnectedness = entity_data.get('interconnectedness', 0.3)
        interconnect_score = interconnectedness * 100
        factors['interconnectedness'] = interconnectedness
        factors['interconnect_score'] = interconnect_score
        
        # Size impact
        size_impact = entity_data.get('systemic_importance', 0.1)
        size_score = size_impact * 100
        factors['size_impact'] = size_impact
        factors['size_score'] = size_score
        
        # Contagion risk
        contagion_risk = entity_data.get('contagion_risk', 0.2)
        contagion_score = contagion_risk * 100
        factors['contagion_risk'] = contagion_risk
        factors['contagion_score'] = contagion_score
        
        # Calculate weighted score
        raw_score = (
            correlation_score * 0.30 +
            interconnect_score * 0.30 +
            size_score * 0.20 +
            contagion_score * 0.20
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.SYSTEMIC_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.65,
            factors=factors,
            threshold_breach=normalized_score > 70,
            alert_level='high' if normalized_score > 70 else 'low'
        )

class ModelRiskScorer:
    '''Score model risk'''
    
    def score(self, entity_data: Dict) -> RiskScoreResult:
        '''Calculate model risk score'''
        factors = {}
        
        # Model complexity
        complexity = entity_data.get('model_complexity', 0.5)
        complexity_score = complexity * 100
        factors['complexity'] = complexity
        factors['complexity_score'] = complexity_score
        
        # Model validation
        validation_score = entity_data.get('validation_score', 0.8)
        validation_risk = (1 - validation_score) * 100
        factors['validation_score'] = validation_score
        factors['validation_risk'] = validation_risk
        
        # Data quality
        data_quality = entity_data.get('data_quality', 0.9)
        data_risk = (1 - data_quality) * 100
        factors['data_quality'] = data_quality
        factors['data_risk'] = data_risk
        
        # Model age (months)
        model_age = entity_data.get('model_age_months', 6)
        age_score = min(100, model_age * 5)  # 5 points per month
        factors['model_age'] = model_age
        factors['age_score'] = age_score
        
        # Calculate weighted score
        raw_score = (
            complexity_score * 0.25 +
            validation_risk * 0.35 +
            data_risk * 0.25 +
            age_score * 0.15
        )
        
        normalized_score = min(100, max(0, raw_score))
        
        return RiskScoreResult(
            entity_id=entity_data.get('id', ''),
            risk_type=RiskType.MODEL_RISK,
            raw_score=raw_score,
            normalized_score=normalized_score,
            confidence=0.70,
            factors=factors,
            threshold_breach=normalized_score > 65,
            alert_level='medium'
        )

class RiskAggregator:
    '''Aggregate multiple risk scores'''
    
    def aggregate_risks(self, risk_scores: Dict[RiskType, RiskScoreResult]):
        '''Aggregate individual risk scores into overall assessment'''
        
        if not risk_scores:
            return {
                'total_score': 0,
                'risk_level': RiskLevel.MINIMAL,
                'confidence': 0,
                'components': {}
            }
        
        # Calculate weighted average
        weights = {
            RiskType.MARKET_RISK: 0.25,
            RiskType.CREDIT_RISK: 0.20,
            RiskType.LIQUIDITY_RISK: 0.15,
            RiskType.OPERATIONAL_RISK: 0.15,
            RiskType.COUNTERPARTY_RISK: 0.10,
            RiskType.CONCENTRATION_RISK: 0.10,
            RiskType.SYSTEMIC_RISK: 0.05
        }
        
        total_score = 0
        total_weight = 0
        total_confidence = 0
        
        for risk_type, score in risk_scores.items():
            weight = weights.get(risk_type, 0.1)
            total_score += score.normalized_score * weight
            total_weight += weight
            total_confidence += score.confidence * weight
        
        # Normalize
        if total_weight > 0:
            total_score = total_score / total_weight
            total_confidence = total_confidence / total_weight
        
        # Apply correlation adjustment
        correlation_factor = self._calculate_correlation_factor(risk_scores)
        adjusted_score = total_score * correlation_factor
        
        # Determine risk level
        risk_level = self._determine_risk_level(adjusted_score)
        
        return {
            'total_score': adjusted_score,
            'risk_level': risk_level,
            'confidence': total_confidence,
            'correlation_factor': correlation_factor,
            'components': {k: v.normalized_score for k, v in risk_scores.items()}
        }
    
    def _calculate_correlation_factor(self, risk_scores):
        '''Calculate correlation adjustment factor'''
        # If multiple risks are high, increase overall risk
        high_risk_count = sum(1 for score in risk_scores.values() 
                            if score.normalized_score > 70)
        
        if high_risk_count >= 3:
            return 1.15  # 15% increase
        elif high_risk_count >= 2:
            return 1.08  # 8% increase
        else:
            return 1.0
    
    def _determine_risk_level(self, score):
        '''Determine risk level from score'''
        if score < 20:
            return RiskLevel.MINIMAL
        elif score < 40:
            return RiskLevel.LOW
        elif score < 60:
            return RiskLevel.MODERATE
        elif score < 75:
            return RiskLevel.HIGH
        elif score < 90:
            return RiskLevel.SEVERE
        else:
            return RiskLevel.CRITICAL

class StressTester:
    '''Stress testing for risk scenarios'''
    
    def run_stress_test(self, entity_data: Dict, risk_scores: Dict):
        '''Run stress test scenarios'''
        scenarios = {
            'Market Crash': self._market_crash_scenario,
            'Credit Crisis': self._credit_crisis_scenario,
            'Liquidity Freeze': self._liquidity_freeze_scenario,
            'Operational Failure': self._operational_failure_scenario,
            'Black Swan Event': self._black_swan_scenario
        }
        
        results = {}
        
        for scenario_name, scenario_func in scenarios.items():
            impact = scenario_func(entity_data, risk_scores)
            results[scenario_name] = {
                'impact': impact,
                'severity': self._classify_impact(impact)
            }
        
        return results
    
    def _market_crash_scenario(self, entity_data, risk_scores):
        '''Market crash stress test'''
        market_score = risk_scores.get(RiskType.MARKET_RISK)
        if market_score:
            base_impact = market_score.normalized_score * 0.5
            leverage = entity_data.get('leverage', 1)
            return min(100, base_impact * leverage)
        return 25
    
    def _credit_crisis_scenario(self, entity_data, risk_scores):
        '''Credit crisis stress test'''
        credit_score = risk_scores.get(RiskType.CREDIT_RISK)
        if credit_score:
            return min(100, credit_score.normalized_score * 0.7)
        return 20
    
    def _liquidity_freeze_scenario(self, entity_data, risk_scores):
        '''Liquidity freeze stress test'''
        liquidity_score = risk_scores.get(RiskType.LIQUIDITY_RISK)
        if liquidity_score:
            return min(100, liquidity_score.normalized_score * 0.8)
        return 30
    
    def _operational_failure_scenario(self, entity_data, risk_scores):
        '''Operational failure stress test'''
        operational_score = risk_scores.get(RiskType.OPERATIONAL_RISK)
        if operational_score:
            return min(100, operational_score.normalized_score * 0.6)
        return 15
    
    def _black_swan_scenario(self, entity_data, risk_scores):
        '''Black swan event stress test'''
        # Extreme scenario - all risks materialize
        total_impact = 0
        for score in risk_scores.values():
            total_impact += score.normalized_score * 0.3
        return min(100, total_impact)
    
    def _classify_impact(self, impact):
        '''Classify stress test impact'''
        if impact < 15:
            return 'Low'
        elif impact < 30:
            return 'Medium'
        elif impact < 50:
            return 'High'
        else:
            return 'Severe'

class RiskCalculator:
    '''Calculate various risk metrics'''
    
    def calculate_metrics(self, entity_data: Dict):
        '''Calculate comprehensive risk metrics'''
        metrics = {}
        
        # Value at Risk
        portfolio_value = entity_data.get('portfolio_value', 1000000)
        volatility = entity_data.get('volatility', 0.15)
        
        metrics['var_95'] = self._calculate_var(portfolio_value, volatility, 0.95)
        metrics['var_99'] = self._calculate_var(portfolio_value, volatility, 0.99)
        
        # Conditional VaR
        metrics['cvar_95'] = metrics['var_95'] * 1.2
        metrics['cvar_99'] = metrics['var_99'] * 1.3
        
        # Sharpe Ratio
        returns = entity_data.get('returns', 0.08)
        risk_free = 0.02
        metrics['sharpe_ratio'] = (returns - risk_free) / volatility if volatility > 0 else 0
        
        # Sortino Ratio
        downside_vol = entity_data.get('downside_volatility', volatility * 0.7)
        metrics['sortino_ratio'] = (returns - risk_free) / downside_vol if downside_vol > 0 else 0
        
        # Maximum Drawdown
        metrics['max_drawdown'] = entity_data.get('max_drawdown', 0.10)
        
        # Beta
        metrics['beta'] = entity_data.get('beta', 1.0)
        
        # Information Ratio
        tracking_error = entity_data.get('tracking_error', 0.05)
        excess_return = returns - 0.07  # Benchmark return
        metrics['information_ratio'] = excess_return / tracking_error if tracking_error > 0 else 0
        
        return metrics
    
    def _calculate_var(self, portfolio_value, volatility, confidence):
        '''Calculate Value at Risk'''
        # Using parametric VaR
        z_scores = {0.95: 1.645, 0.99: 2.326}
        z = z_scores.get(confidence, 1.645)
        
        daily_volatility = volatility / np.sqrt(252)
        var = portfolio_value * z * daily_volatility
        
        return var

class RiskMonitor:
    '''Monitor risk levels and alerts'''
    
    def __init__(self):
        self.alert_history = deque(maxlen=1000)
        self.risk_limits = self._initialize_limits()
    
    def _initialize_limits(self):
        '''Initialize risk limits'''
        return {
            RiskType.MARKET_RISK: 70,
            RiskType.CREDIT_RISK: 65,
            RiskType.LIQUIDITY_RISK: 60,
            RiskType.OPERATIONAL_RISK: 55,
            RiskType.COUNTERPARTY_RISK: 70,
            RiskType.CONCENTRATION_RISK: 65,
            RiskType.SYSTEMIC_RISK: 75
        }
    
    def monitor_limits(self, risk_profile: RiskProfile):
        '''Monitor risk limits and generate alerts'''
        alerts = []
        
        for risk_type, score in risk_profile.risk_scores.items():
            limit = self.risk_limits.get(risk_type, 70)
            
            if score > limit:
                alert = {
                    'timestamp': datetime.now(),
                    'entity_id': risk_profile.entity_id,
                    'risk_type': risk_type,
                    'score': score,
                    'limit': limit,
                    'breach_amount': score - limit,
                    'severity': self._determine_severity(score, limit)
                }
                alerts.append(alert)
                self.alert_history.append(alert)
        
        return alerts
    
    def _determine_severity(self, score, limit):
        '''Determine alert severity'''
        breach_pct = ((score - limit) / limit) * 100
        
        if breach_pct > 50:
            return 'CRITICAL'
        elif breach_pct > 25:
            return 'HIGH'
        elif breach_pct > 10:
            return 'MEDIUM'
        else:
            return 'LOW'

# Demonstrate system
if __name__ == '__main__':
    print('🎯 RISK SCORING ENGINE - ULTRAPLATFORM')
    print('='*80)
    
    # Create sample entity data
    entity_data = {
        'id': 'PORT_001',
        'type': 'portfolio',
        'portfolio_value': 10000000,  # 
        'var_95': 200000,  #  VaR
        'volatility': 0.18,  # 18% annual volatility
        'beta': 1.2,
        'max_drawdown': 0.15,
        'probability_of_default': 0.02,
        'loss_given_default': 0.40,
        'exposure_at_default': 2000000,
        'credit_rating': 'BBB',
        'bid_ask_spread': 0.002,
        'market_depth': 500,
        'liquidity_ratio': 0.25,
        'liquidation_days': 3,
        'process_failures': 2,
        'system_downtime_hours': 1.5,
        'human_errors': 3,
        'external_incidents': 1,
        'counterparty_rating': 'A',
        'counterparty_exposure': 1500000,
        'collateral_coverage': 1.2,
        'settlement_failures': 0,
        'largest_position_pct': 0.15,
        'largest_sector_pct': 0.35,
        'largest_geo_pct': 0.60,
        'herfindahl_index': 1800,
        'leverage': 1.5,
        'returns': 0.12
    }
    
    # Initialize Risk Scoring Engine
    engine = RiskScoringEngine()
    
    # Score risks
    print('\n🎯 SCORING ENTITY RISK')
    print('='*80 + '\n')
    
    risk_profile = engine.score_risk(entity_data)
    
    # Show summary
    print('\n' + '='*80)
    print('RISK SCORING SUMMARY')
    print('='*80)
    print(f'Entity ID: {risk_profile.entity_id}')
    print(f'Aggregate Risk Score: {risk_profile.aggregate_score:.1f}/100')
    print(f'Risk Level: {risk_profile.risk_level.value.upper()}')
    
    # Show individual risk scores
    print('\n📊 RISK SCORES BY TYPE')
    print('-'*40)
    for risk_type, score in risk_profile.risk_scores.items():
        icon = '🔴' if score > 70 else '🟡' if score > 50 else '🟢'
        print(f'{icon} {risk_type.value}: {score:.1f}/100')
    
    # Show violations
    if risk_profile.violations:
        print('\n⚠️ RISK VIOLATIONS')
        print('-'*40)
        for violation in risk_profile.violations:
            print(f'  • {violation}')
    
    # Show recommendations
    if risk_profile.recommendations:
        print('\n💡 RECOMMENDATIONS')
        print('-'*40)
        for rec in risk_profile.recommendations:
            print(f'  • {rec}')
    
    print('\n✅ Risk Scoring Complete!')
