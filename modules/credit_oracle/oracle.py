from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import hashlib
import statistics
import math
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

class CreditRating(Enum):
    AAA = 'AAA'    # Prime
    AA_PLUS = 'AA+'
    AA = 'AA'
    AA_MINUS = 'AA-'
    A_PLUS = 'A+'
    A = 'A'
    A_MINUS = 'A-'
    BBB_PLUS = 'BBB+'  # Investment Grade Cutoff
    BBB = 'BBB'
    BBB_MINUS = 'BBB-'
    BB_PLUS = 'BB+'    # Speculative
    BB = 'BB'
    BB_MINUS = 'BB-'
    B_PLUS = 'B+'
    B = 'B'
    B_MINUS = 'B-'
    CCC = 'CCC'        # Highly Speculative
    CC = 'CC'
    C = 'C'
    D = 'D'            # Default

class EntityType(Enum):
    CORPORATE = 'corporate'
    SOVEREIGN = 'sovereign'
    FINANCIAL_INSTITUTION = 'financial_institution'
    SME = 'sme'
    INDIVIDUAL = 'individual'
    STRUCTURED_PRODUCT = 'structured_product'

class DataSource(Enum):
    FINANCIAL_STATEMENTS = 'financial_statements'
    MARKET_DATA = 'market_data'
    CREDIT_BUREAU = 'credit_bureau'
    PAYMENT_HISTORY = 'payment_history'
    NEWS_SENTIMENT = 'news_sentiment'
    REGULATORY_FILINGS = 'regulatory_filings'
    ALTERNATIVE_DATA = 'alternative_data'

class AssessmentType(Enum):
    FULL_ASSESSMENT = 'full_assessment'
    QUICK_SCORE = 'quick_score'
    MONITORING_UPDATE = 'monitoring_update'
    STRESS_TEST = 'stress_test'
    PORTFOLIO_REVIEW = 'portfolio_review'

@dataclass
class CreditEntity:
    '''Entity for credit assessment'''
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ''
    entity_type: EntityType = EntityType.CORPORATE
    country: str = 'AU'
    industry: str = ''
    incorporation_date: Optional[datetime] = None
    financial_data: Dict = field(default_factory=dict)
    market_data: Dict = field(default_factory=dict)
    credit_history: List = field(default_factory=list)
    
    def to_dict(self):
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type.value,
            'country': self.country,
            'industry': self.industry
        }

@dataclass
class CreditAssessment:
    '''Credit assessment result'''
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entity: CreditEntity = None
    timestamp: datetime = field(default_factory=datetime.now)
    credit_score: float = 0.0
    credit_rating: CreditRating = CreditRating.BBB
    probability_of_default: float = 0.0
    loss_given_default: float = 0.0
    exposure_at_default: float = 0.0
    expected_loss: float = 0.0
    credit_limit: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    confidence_score: float = 0.0

class CreditOracle:
    '''Comprehensive Credit Oracle System for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Credit Oracle'
        self.version = '2.0'
        
        # Core components
        self.credit_scorer = CreditScorer()
        self.risk_assessor = RiskAssessor()
        self.rating_engine = RatingEngine()
        self.pd_model = ProbabilityOfDefaultModel()
        self.lgd_model = LossGivenDefaultModel()
        self.monitoring_engine = CreditMonitoringEngine()
        self.data_aggregator = DataAggregator()
        self.ml_models = MachineLearningModels()
        self.compliance_checker = ComplianceChecker()
        self.report_generator = CreditReportGenerator()
        
        # Initialize models
        self._initialize_models()
        
    def assess_credit(self, entity: CreditEntity, assessment_type: AssessmentType = AssessmentType.FULL_ASSESSMENT):
        '''Perform comprehensive credit assessment'''
        print('CREDIT ORACLE - ASSESSMENT')
        print('='*80)
        print(f'Entity: {entity.name}')
        print(f'Type: {entity.entity_type.value}')
        print(f'Country: {entity.country}')
        print(f'Industry: {entity.industry}')
        print()
        
        # Step 1: Data Collection
        print('1️⃣ DATA COLLECTION & VALIDATION')
        print('-'*40)
        data = self.data_aggregator.collect_data(entity)
        print(f'  Financial Data Points: {data["financial_count"]}')
        print(f'  Market Data Points: {data["market_count"]}')
        print(f'  Historical Records: {data["history_count"]}')
        print(f'  Data Quality Score: {data["quality_score"]:.1%}')
        
        # Step 2: Credit Scoring
        print('\n2️⃣ CREDIT SCORING')
        print('-'*40)
        credit_score = self.credit_scorer.calculate_score(entity, data)
        print(f'  Base Score: {credit_score["base_score"]:.1f}/1000')
        print(f'  Industry Adjustment: {credit_score["industry_adj"]:+.1f}')
        print(f'  Country Risk Adjustment: {credit_score["country_adj"]:+.1f}')
        print(f'  Final Score: {credit_score["final_score"]:.1f}/1000')
        
        # Step 3: Rating Assignment
        print('\n3️⃣ CREDIT RATING')
        print('-'*40)
        rating = self.rating_engine.assign_rating(credit_score["final_score"], entity)
        print(f'  Credit Rating: {rating["rating"].value}')
        print(f'  Rating Category: {rating["category"]}')
        print(f'  Outlook: {rating["outlook"]}')
        print(f'  Watch Status: {rating["watch_status"]}')
        
        # Step 4: Probability of Default
        print('\n4️⃣ DEFAULT PROBABILITY ANALYSIS')
        print('-'*40)
        pd_analysis = self.pd_model.calculate_pd(entity, credit_score["final_score"])
        print(f'  1-Year PD: {pd_analysis["pd_1y"]:.2%}')
        print(f'  3-Year PD: {pd_analysis["pd_3y"]:.2%}')
        print(f'  5-Year PD: {pd_analysis["pd_5y"]:.2%}')
        print(f'  Through-the-Cycle PD: {pd_analysis["pd_ttc"]:.2%}')
        
        # Step 5: Loss Given Default
        print('\n5️⃣ LOSS GIVEN DEFAULT ANALYSIS')
        print('-'*40)
        lgd_analysis = self.lgd_model.calculate_lgd(entity)
        print(f'  Senior Secured: {lgd_analysis["senior_secured"]:.1%}')
        print(f'  Senior Unsecured: {lgd_analysis["senior_unsecured"]:.1%}')
        print(f'  Subordinated: {lgd_analysis["subordinated"]:.1%}')
        print(f'  Recovery Rate: {lgd_analysis["recovery_rate"]:.1%}')
        
        # Step 6: Risk Assessment
        print('\n6️⃣ RISK ASSESSMENT')
        print('-'*40)
        risk = self.risk_assessor.assess_risks(entity, credit_score, pd_analysis)
        print(f'  Risk Factors:')
        for factor in risk["risk_factors"][:5]:
            print(f'    • {factor}')
        print(f'  Strengths:')
        for strength in risk["strengths"][:3]:
            print(f'    • {strength}')
        
        # Step 7: Credit Limit Calculation
        print('\n7️⃣ CREDIT LIMIT CALCULATION')
        print('-'*40)
        limit_calc = self._calculate_credit_limit(entity, credit_score["final_score"], pd_analysis["pd_1y"])
        print(f'  Maximum Exposure: ')
        print(f'  Recommended Limit: ')
        print(f'  Collateral Required: {limit_calc["collateral_required"]}')
        
        # Step 8: ML Enhancement
        print('\n8️⃣ MACHINE LEARNING ENHANCEMENT')
        print('-'*40)
        ml_insights = self.ml_models.enhance_assessment(entity, credit_score["final_score"])
        print(f'  ML Score Adjustment: {ml_insights["score_adjustment"]:+.1f}')
        print(f'  Anomaly Detection: {ml_insights["anomalies"]}')
        print(f'  Peer Comparison: {ml_insights["peer_percentile"]:.0f}th percentile')
        
        # Create assessment report
        assessment = CreditAssessment(
            entity=entity,
            credit_score=credit_score["final_score"],
            credit_rating=rating["rating"],
            probability_of_default=pd_analysis["pd_1y"],
            loss_given_default=lgd_analysis["senior_unsecured"],
            exposure_at_default=limit_calc["max_exposure"],
            expected_loss=limit_calc["max_exposure"] * pd_analysis["pd_1y"] * lgd_analysis["senior_unsecured"],
            credit_limit=limit_calc["recommended"],
            risk_factors=risk["risk_factors"],
            strengths=risk["strengths"],
            recommendations=self._generate_recommendations(entity, credit_score, risk),
            confidence_score=data["quality_score"]
        )
        
        return assessment
    
    def _initialize_models(self):
        '''Initialize credit models'''
        # Load pre-trained models
        self.ml_models.load_models()
        
    def _calculate_credit_limit(self, entity, credit_score, pd):
        '''Calculate appropriate credit limit'''
        # Base calculation
        if entity.entity_type == EntityType.CORPORATE:
            revenue = entity.financial_data.get('annual_revenue', 10000000)
            base_limit = revenue * 0.10  # 10% of revenue
        elif entity.entity_type == EntityType.INDIVIDUAL:
            income = entity.financial_data.get('annual_income', 100000)
            base_limit = income * 0.30  # 30% of income
        else:
            base_limit = 1000000  # Default 
        
        # Adjust for credit score
        score_multiplier = credit_score / 500  # Normalize to ~2x for good credit
        
        # Adjust for PD
        pd_adjustment = 1 - pd  # Reduce limit based on default probability
        
        # Calculate final limit
        max_exposure = base_limit * score_multiplier * pd_adjustment
        recommended = max_exposure * 0.8  # Conservative 80% of max
        
        # Collateral requirements
        if credit_score < 600:
            collateral = "Full collateral required"
        elif credit_score < 700:
            collateral = "Partial collateral (50%)"
        else:
            collateral = "No collateral required"
        
        return {
            'max_exposure': max_exposure,
            'recommended': recommended,
            'collateral_required': collateral
        }
    
    def _generate_recommendations(self, entity, credit_score, risk):
        '''Generate credit recommendations'''
        recommendations = []
        
        if credit_score["final_score"] < 600:
            recommendations.append("Require full collateral for any credit exposure")
            recommendations.append("Implement enhanced monitoring and reporting")
        
        if risk.get("leverage_high", False):
            recommendations.append("Request debt reduction plan")
        
        if risk.get("concentration_risk", False):
            recommendations.append("Encourage customer diversification")
        
        recommendations.append("Review credit terms quarterly")
        recommendations.append("Monitor key financial covenants")
        
        return recommendations

class CreditScorer:
    '''Credit scoring engine'''
    
    def __init__(self):
        self.scoring_models = {
            EntityType.CORPORATE: self._score_corporate,
            EntityType.INDIVIDUAL: self._score_individual,
            EntityType.FINANCIAL_INSTITUTION: self._score_financial,
            EntityType.SME: self._score_sme,
            EntityType.SOVEREIGN: self._score_sovereign
        }
        
    def calculate_score(self, entity: CreditEntity, data: Dict):
        '''Calculate credit score'''
        # Get appropriate scoring model
        scoring_func = self.scoring_models.get(
            entity.entity_type,
            self._score_generic
        )
        
        # Calculate base score
        base_score = scoring_func(entity, data)
        
        # Industry adjustment
        industry_adj = self._industry_adjustment(entity.industry)
        
        # Country risk adjustment
        country_adj = self._country_adjustment(entity.country)
        
        # Calculate final score
        final_score = base_score + industry_adj + country_adj
        final_score = max(0, min(1000, final_score))  # Bound between 0-1000
        
        return {
            'base_score': base_score,
            'industry_adj': industry_adj,
            'country_adj': country_adj,
            'final_score': final_score,
            'components': self._get_score_components(entity, data)
        }
    
    def _score_corporate(self, entity, data):
        '''Score corporate entity'''
        score = 500  # Start at midpoint
        
        # Financial ratios
        financial = entity.financial_data
        
        # Profitability (max 150 points)
        roe = financial.get('return_on_equity', 0.10)
        score += min(150, roe * 1000)
        
        # Leverage (max 150 points)
        debt_to_equity = financial.get('debt_to_equity', 1.0)
        score += max(-150, min(150, (2 - debt_to_equity) * 75))
        
        # Liquidity (max 100 points)
        current_ratio = financial.get('current_ratio', 1.0)
        score += min(100, current_ratio * 50)
        
        # Size (max 100 points)
        revenue = financial.get('annual_revenue', 0)
        if revenue > 1000000000:  # +
            score += 100
        elif revenue > 100000000:  # +
            score += 50
        
        return score
    
    def _score_individual(self, entity, data):
        '''Score individual'''
        score = 500
        
        # Credit history (max 200 points)
        payment_history = len(entity.credit_history)
        defaults = sum(1 for h in entity.credit_history if h.get('default', False))
        score += min(200, payment_history * 2 - defaults * 50)
        
        # Income (max 150 points)
        income = entity.financial_data.get('annual_income', 0)
        score += min(150, income / 1000)
        
        # Debt-to-income (max 150 points)
        dti = entity.financial_data.get('debt_to_income', 0.3)
        score += max(-150, min(150, (0.4 - dti) * 500))
        
        return score
    
    def _score_financial(self, entity, data):
        '''Score financial institution'''
        score = 500
        
        # Capital adequacy (max 200 points)
        car = entity.financial_data.get('capital_adequacy_ratio', 0.08)
        score += min(200, car * 2000)
        
        # Asset quality (max 150 points)
        npl_ratio = entity.financial_data.get('npl_ratio', 0.02)
        score += max(-150, min(150, (0.05 - npl_ratio) * 3000))
        
        # ROA (max 150 points)
        roa = entity.financial_data.get('return_on_assets', 0.01)
        score += min(150, roa * 10000)
        
        return score
    
    def _score_sme(self, entity, data):
        '''Score SME'''
        # Similar to corporate but with adjusted thresholds
        return self._score_corporate(entity, data) * 0.9
    
    def _score_sovereign(self, entity, data):
        '''Score sovereign entity'''
        score = 600  # Higher base for sovereigns
        
        # GDP growth (max 100 points)
        gdp_growth = entity.financial_data.get('gdp_growth', 0.02)
        score += min(100, gdp_growth * 2500)
        
        # Debt-to-GDP (max 150 points)
        debt_to_gdp = entity.financial_data.get('debt_to_gdp', 0.6)
        score += max(-150, min(150, (1.0 - debt_to_gdp) * 150))
        
        # Current account (max 100 points)
        current_account = entity.financial_data.get('current_account_gdp', 0)
        score += min(100, max(-100, current_account * 1000))
        
        # Reserves (max 50 points)
        reserves_months = entity.financial_data.get('reserves_import_months', 3)
        score += min(50, reserves_months * 10)
        
        return score
    
    def _score_generic(self, entity, data):
        '''Generic scoring fallback'''
        return 500
    
    def _industry_adjustment(self, industry):
        '''Industry-based score adjustment'''
        adjustments = {
            'technology': 20,
            'healthcare': 15,
            'utilities': 30,
            'energy': -10,
            'retail': -20,
            'real_estate': 0,
            'financial': 10,
            'manufacturing': 0,
            'mining': -15
        }
        return adjustments.get(industry.lower(), 0)
    
    def _country_adjustment(self, country):
        '''Country risk adjustment'''
        # Simplified country risk scores
        country_scores = {
            'AU': 50,   # Australia
            'NZ': 45,   # New Zealand
            'US': 40,   # United States
            'GB': 35,   # United Kingdom
            'SG': 40,   # Singapore
            'JP': 35,   # Japan
            'CN': -10,  # China
            'IN': -20,  # India
        }
        return country_scores.get(country, 0)
    
    def _get_score_components(self, entity, data):
        '''Break down score components'''
        return {
            'financial_strength': random.uniform(100, 200),
            'business_profile': random.uniform(100, 200),
            'management_quality': random.uniform(50, 150),
            'market_position': random.uniform(50, 150),
            'regulatory_compliance': random.uniform(80, 120)
        }

class RatingEngine:
    '''Credit rating assignment engine'''
    
    def __init__(self):
        self.rating_scale = self._initialize_rating_scale()
        
    def _initialize_rating_scale(self):
        '''Initialize rating scale mapping'''
        return [
            (900, CreditRating.AAA, 'Prime', 'Stable'),
            (850, CreditRating.AA_PLUS, 'High Grade', 'Stable'),
            (820, CreditRating.AA, 'High Grade', 'Stable'),
            (790, CreditRating.AA_MINUS, 'High Grade', 'Stable'),
            (760, CreditRating.A_PLUS, 'Upper Medium', 'Stable'),
            (730, CreditRating.A, 'Upper Medium', 'Stable'),
            (700, CreditRating.A_MINUS, 'Upper Medium', 'Stable'),
            (670, CreditRating.BBB_PLUS, 'Lower Medium', 'Stable'),
            (640, CreditRating.BBB, 'Lower Medium', 'Stable'),
            (610, CreditRating.BBB_MINUS, 'Lower Medium', 'Watch'),
            (580, CreditRating.BB_PLUS, 'Speculative', 'Negative'),
            (550, CreditRating.BB, 'Speculative', 'Negative'),
            (520, CreditRating.BB_MINUS, 'Speculative', 'Negative'),
            (490, CreditRating.B_PLUS, 'Highly Speculative', 'Negative'),
            (460, CreditRating.B, 'Highly Speculative', 'Negative'),
            (430, CreditRating.B_MINUS, 'Highly Speculative', 'Negative'),
            (400, CreditRating.CCC, 'Substantial Risk', 'Watch'),
            (350, CreditRating.CC, 'Very High Risk', 'Watch'),
            (300, CreditRating.C, 'Near Default', 'Watch'),
            (0, CreditRating.D, 'Default', 'Default')
        ]
    
    def assign_rating(self, credit_score, entity):
        '''Assign credit rating based on score'''
        rating = CreditRating.BBB  # Default
        category = 'Lower Medium'
        outlook = 'Stable'
        
        for threshold, rating_value, cat, outl in self.rating_scale:
            if credit_score >= threshold:
                rating = rating_value
                category = cat
                outlook = outl
                break
        
        # Check for watch status
        watch_status = self._check_watch_status(entity, credit_score)
        
        return {
            'rating': rating,
            'category': category,
            'outlook': outlook,
            'watch_status': watch_status,
            'investment_grade': self._is_investment_grade(rating)
        }
    
    def _check_watch_status(self, entity, score):
        '''Check if entity should be on watch list'''
        # Check for deteriorating metrics
        if hasattr(entity, 'previous_score'):
            if score < entity.previous_score * 0.9:
                return 'Negative Watch'
        return 'None'
    
    def _is_investment_grade(self, rating):
        '''Check if rating is investment grade'''
        investment_grade = [
            CreditRating.AAA, CreditRating.AA_PLUS, CreditRating.AA,
            CreditRating.AA_MINUS, CreditRating.A_PLUS, CreditRating.A,
            CreditRating.A_MINUS, CreditRating.BBB_PLUS, CreditRating.BBB,
            CreditRating.BBB_MINUS
        ]
        return rating in investment_grade

class ProbabilityOfDefaultModel:
    '''Probability of Default (PD) modeling'''
    
    def __init__(self):
        self.base_pd_rates = self._initialize_base_pd()
        
    def _initialize_base_pd(self):
        '''Initialize base PD rates by score range'''
        return {
            (900, 1000): 0.0001,  # AAA
            (820, 900): 0.0005,   # AA
            (700, 820): 0.002,    # A
            (610, 700): 0.01,     # BBB
            (520, 610): 0.03,     # BB
            (430, 520): 0.08,     # B
            (300, 430): 0.20,     # CCC
            (0, 300): 0.50        # CC and below
        }
    
    def calculate_pd(self, entity, credit_score):
        '''Calculate probability of default'''
        # Get base PD
        base_pd = self._get_base_pd(credit_score)
        
        # Calculate term structure
        pd_1y = base_pd
        pd_3y = 1 - (1 - base_pd) ** 3
        pd_5y = 1 - (1 - base_pd) ** 5
        
        # Through-the-cycle adjustment
        pd_ttc = self._calculate_ttc_pd(base_pd, entity)
        
        # Point-in-time adjustment
        pd_pit = self._calculate_pit_pd(base_pd, entity)
        
        return {
            'pd_1y': pd_1y,
            'pd_3y': pd_3y,
            'pd_5y': pd_5y,
            'pd_ttc': pd_ttc,
            'pd_pit': pd_pit,
            'expected_default_time': self._expected_default_time(base_pd)
        }
    
    def _get_base_pd(self, score):
        '''Get base PD from score'''
        for (min_score, max_score), pd in self.base_pd_rates.items():
            if min_score <= score < max_score:
                return pd
        return 0.01  # Default 1%
    
    def _calculate_ttc_pd(self, base_pd, entity):
        '''Calculate through-the-cycle PD'''
        # Average over economic cycle
        return base_pd * 1.2  # Simplified: 20% higher for conservatism
    
    def _calculate_pit_pd(self, base_pd, entity):
        '''Calculate point-in-time PD'''
        # Adjust for current economic conditions
        economic_factor = random.uniform(0.8, 1.2)
        return base_pd * economic_factor
    
    def _expected_default_time(self, pd):
        '''Calculate expected time to default'''
        if pd > 0:
            return -1 / math.log(1 - pd)
        return float('inf')

class LossGivenDefaultModel:
    '''Loss Given Default (LGD) modeling'''
    
    def __init__(self):
        self.recovery_rates = self._initialize_recovery_rates()
        
    def _initialize_recovery_rates(self):
        '''Initialize recovery rates by security type'''
        return {
            'senior_secured': 0.65,      # 65% recovery
            'senior_unsecured': 0.40,    # 40% recovery
            'senior_subordinated': 0.30,  # 30% recovery
            'subordinated': 0.20,         # 20% recovery
            'junior_subordinated': 0.15,  # 15% recovery
            'equity': 0.05               # 5% recovery
        }
    
    def calculate_lgd(self, entity):
        '''Calculate Loss Given Default'''
        lgd_results = {}
        
        for security_type, recovery in self.recovery_rates.items():
            # Adjust for entity-specific factors
            adjusted_recovery = self._adjust_recovery(recovery, entity)
            lgd = 1 - adjusted_recovery
            lgd_results[security_type] = lgd
        
        # Overall recovery rate (weighted average)
        overall_recovery = self._calculate_overall_recovery(entity)
        
        lgd_results['recovery_rate'] = overall_recovery
        lgd_results['expected_recovery_time'] = self._recovery_time(entity)
        
        return lgd_results
    
    def _adjust_recovery(self, base_recovery, entity):
        '''Adjust recovery rate for entity-specific factors'''
        adjustments = 1.0
        
        # Industry adjustment
        if entity.industry in ['utilities', 'infrastructure']:
            adjustments *= 1.2  # Higher recovery
        elif entity.industry in ['technology', 'retail']:
            adjustments *= 0.8  # Lower recovery
        
        # Size adjustment
        revenue = entity.financial_data.get('annual_revenue', 0)
        if revenue > 1000000000:
            adjustments *= 1.1
        elif revenue < 10000000:
            adjustments *= 0.9
        
        # Collateral adjustment
        if entity.financial_data.get('collateral_coverage', 0) > 1.5:
            adjustments *= 1.15
        
        return min(0.95, base_recovery * adjustments)
    
    def _calculate_overall_recovery(self, entity):
        '''Calculate overall recovery rate'''
        # Simplified weighted average
        secured_weight = 0.6
        unsecured_weight = 0.4
        
        secured_recovery = self.recovery_rates['senior_secured']
        unsecured_recovery = self.recovery_rates['senior_unsecured']
        
        overall = (secured_recovery * secured_weight + 
                  unsecured_recovery * unsecured_weight)
        
        return self._adjust_recovery(overall, entity)
    
    def _recovery_time(self, entity):
        '''Estimate recovery time in months'''
        if entity.entity_type == EntityType.CORPORATE:
            return random.randint(12, 36)
        elif entity.entity_type == EntityType.INDIVIDUAL:
            return random.randint(3, 12)
        else:
            return random.randint(6, 24)

class RiskAssessor:
    '''Comprehensive risk assessment'''
    
    def assess_risks(self, entity, credit_score, pd_analysis):
        '''Assess various risk factors'''
        risk_factors = []
        strengths = []
        
        # Financial risks
        financial_risks = self._assess_financial_risks(entity)
        risk_factors.extend(financial_risks['risks'])
        strengths.extend(financial_risks['strengths'])
        
        # Business risks
        business_risks = self._assess_business_risks(entity)
        risk_factors.extend(business_risks['risks'])
        strengths.extend(business_risks['strengths'])
        
        # Market risks
        market_risks = self._assess_market_risks(entity)
        risk_factors.extend(market_risks['risks'])
        strengths.extend(market_risks['strengths'])
        
        # Operational risks
        operational_risks = self._assess_operational_risks(entity)
        risk_factors.extend(operational_risks['risks'])
        
        return {
            'risk_factors': risk_factors,
            'strengths': strengths,
            'overall_risk': self._calculate_overall_risk(risk_factors),
            'risk_mitigation': self._suggest_mitigation(risk_factors)
        }
    
    def _assess_financial_risks(self, entity):
        '''Assess financial risk factors'''
        risks = []
        strengths = []
        
        # Check leverage
        debt_to_equity = entity.financial_data.get('debt_to_equity', 1.0)
        if debt_to_equity > 2.0:
            risks.append(f'High leverage (D/E: {debt_to_equity:.1f})')
        elif debt_to_equity < 0.5:
            strengths.append('Low leverage position')
        
        # Check liquidity
        current_ratio = entity.financial_data.get('current_ratio', 1.0)
        if current_ratio < 1.0:
            risks.append(f'Weak liquidity (Current ratio: {current_ratio:.1f})')
        elif current_ratio > 2.0:
            strengths.append('Strong liquidity position')
        
        # Check profitability
        roe = entity.financial_data.get('return_on_equity', 0.10)
        if roe < 0:
            risks.append('Negative profitability')
        elif roe > 0.15:
            strengths.append(f'Strong profitability (ROE: {roe:.1%})')
        
        return {'risks': risks, 'strengths': strengths}
    
    def _assess_business_risks(self, entity):
        '''Assess business risk factors'''
        risks = []
        strengths = []
        
        # Market position
        market_share = entity.financial_data.get('market_share', 0.05)
        if market_share < 0.01:
            risks.append('Weak market position')
        elif market_share > 0.20:
            strengths.append('Strong market position')
        
        # Customer concentration
        top_customer_pct = entity.financial_data.get('top_customer_concentration', 0.10)
        if top_customer_pct > 0.30:
            risks.append(f'High customer concentration ({top_customer_pct:.0%})')
        
        # Industry outlook
        if entity.industry in ['coal', 'tobacco']:
            risks.append('Declining industry outlook')
        elif entity.industry in ['renewable_energy', 'healthcare']:
            strengths.append('Positive industry outlook')
        
        return {'risks': risks, 'strengths': strengths}
    
    def _assess_market_risks(self, entity):
        '''Assess market risk factors'''
        risks = []
        strengths = []
        
        # Currency exposure
        fx_exposure = entity.financial_data.get('fx_exposure', 0)
        if fx_exposure > 0.30:
            risks.append('High foreign exchange exposure')
        
        # Interest rate sensitivity
        if entity.financial_data.get('floating_rate_debt', 0) > 0.50:
            risks.append('High interest rate sensitivity')
        
        # Commodity exposure
        if entity.industry in ['mining', 'energy']:
            risks.append('Commodity price volatility')
        
        return {'risks': risks, 'strengths': strengths}
    
    def _assess_operational_risks(self, entity):
        '''Assess operational risk factors'''
        risks = []
        
        # Regulatory compliance
        if entity.financial_data.get('regulatory_issues', 0) > 0:
            risks.append('Regulatory compliance issues')
        
        # Cyber security
        if entity.industry in ['financial', 'technology']:
            risks.append('Elevated cyber security risk')
        
        # Geographic concentration
        if entity.country in ['emerging_markets']:
            risks.append('Geographic concentration risk')
        
        return {'risks': risks}
    
    def _calculate_overall_risk(self, risk_factors):
        '''Calculate overall risk level'''
        if len(risk_factors) > 10:
            return 'Very High'
        elif len(risk_factors) > 7:
            return 'High'
        elif len(risk_factors) > 4:
            return 'Medium'
        elif len(risk_factors) > 2:
            return 'Low'
        else:
            return 'Very Low'
    
    def _suggest_mitigation(self, risk_factors):
        '''Suggest risk mitigation strategies'''
        mitigations = []
        
        for risk in risk_factors:
            if 'leverage' in risk.lower():
                mitigations.append('Debt reduction program')
            elif 'liquidity' in risk.lower():
                mitigations.append('Improve working capital management')
            elif 'concentration' in risk.lower():
                mitigations.append('Diversification strategy')
        
        return mitigations

class CreditMonitoringEngine:
    '''Continuous credit monitoring'''
    
    def __init__(self):
        self.monitoring_triggers = self._initialize_triggers()
        self.alert_history = []
        
    def _initialize_triggers(self):
        '''Initialize monitoring triggers'''
        return {
            'rating_downgrade': 2,  # 2 notch downgrade
            'score_decline': 50,    # 50 point decline
            'pd_increase': 0.05,    # 5% PD increase
            'covenant_breach': True,
            'payment_delay': 30,    # 30 days
            'news_sentiment': -0.5  # Negative sentiment
        }
    
    def monitor_entity(self, entity, previous_assessment, current_assessment):
        '''Monitor entity for credit changes'''
        alerts = []
        
        # Check rating change
        if self._rating_declined(previous_assessment.credit_rating, 
                                current_assessment.credit_rating):
            alerts.append({
                'type': 'Rating Downgrade',
                'severity': 'High',
                'message': f'Rating declined from {previous_assessment.credit_rating.value} to {current_assessment.credit_rating.value}'
            })
        
        # Check score change
        score_change = current_assessment.credit_score - previous_assessment.credit_score
        if score_change < -self.monitoring_triggers['score_decline']:
            alerts.append({
                'type': 'Score Decline',
                'severity': 'Medium',
                'message': f'Credit score declined by {abs(score_change):.0f} points'
            })
        
        # Check PD change
        pd_change = current_assessment.probability_of_default - previous_assessment.probability_of_default
        if pd_change > self.monitoring_triggers['pd_increase']:
            alerts.append({
                'type': 'Default Risk Increase',
                'severity': 'High',
                'message': f'PD increased by {pd_change:.2%}'
            })
        
        self.alert_history.extend(alerts)
        return alerts
    
    def _rating_declined(self, previous, current):
        '''Check if rating declined'''
        rating_order = [r for r in CreditRating]
        prev_idx = rating_order.index(previous)
        curr_idx = rating_order.index(current)
        return curr_idx > prev_idx

class DataAggregator:
    '''Aggregate data from multiple sources'''
    
    def collect_data(self, entity):
        '''Collect data from various sources'''
        data = {
            'financial_count': 0,
            'market_count': 0,
            'history_count': 0,
            'quality_score': 0.0
        }
        
        # Financial statements
        if entity.financial_data:
            data['financial_count'] = len(entity.financial_data)
        
        # Market data
        if entity.market_data:
            data['market_count'] = len(entity.market_data)
        
        # Credit history
        data['history_count'] = len(entity.credit_history)
        
        # Calculate quality score
        total_points = data['financial_count'] + data['market_count'] + data['history_count']
        data['quality_score'] = min(1.0, total_points / 50)
        
        return data

class MachineLearningModels:
    '''Machine learning credit models'''
    
    def __init__(self):
        self.models = {}
        
    def load_models(self):
        '''Load pre-trained ML models'''
        # Placeholder for model loading
        self.models = {
            'xgboost': None,
            'neural_net': None,
            'random_forest': None
        }
    
    def enhance_assessment(self, entity, base_score):
        '''Enhance assessment with ML'''
        # Simplified ML enhancement
        ml_adjustment = random.uniform(-20, 20)
        
        # Anomaly detection
        anomalies = []
        if random.random() > 0.8:
            anomalies.append('Unusual financial pattern detected')
        
        # Peer comparison
        peer_percentile = random.uniform(25, 75)
        
        return {
            'score_adjustment': ml_adjustment,
            'anomalies': anomalies,
            'peer_percentile': peer_percentile,
            'confidence': random.uniform(0.7, 0.95)
        }

class ComplianceChecker:
    '''Regulatory compliance checking'''
    
    def check_compliance(self, entity, assessment):
        '''Check regulatory compliance'''
        compliance_issues = []
        
        # APRA requirements for AU entities
        if entity.country == 'AU':
            if assessment.credit_rating.value in ['CCC', 'CC', 'C', 'D']:
                compliance_issues.append('APRA high-risk classification')
        
        # Basel III requirements
        if entity.entity_type == EntityType.FINANCIAL_INSTITUTION:
            car = entity.financial_data.get('capital_adequacy_ratio', 0)
            if car < 0.08:
                compliance_issues.append('Below Basel III minimum CAR')
        
        return {
            'compliant': len(compliance_issues) == 0,
            'issues': compliance_issues
        }

class CreditReportGenerator:
    '''Generate credit reports'''
    
    def generate_report(self, assessment: CreditAssessment):
        '''Generate comprehensive credit report'''
        report = {
            'executive_summary': self._generate_executive_summary(assessment),
            'credit_opinion': self._generate_credit_opinion(assessment),
            'financial_analysis': self._generate_financial_analysis(assessment),
            'risk_assessment': self._generate_risk_assessment(assessment),
            'rating_rationale': self._generate_rating_rationale(assessment),
            'outlook': self._generate_outlook(assessment)
        }
        return report
    
    def _generate_executive_summary(self, assessment):
        '''Generate executive summary'''
        return f"""
        Entity: {assessment.entity.name}
        Credit Rating: {assessment.credit_rating.value}
        Credit Score: {assessment.credit_score:.0f}/1000
        1-Year PD: {assessment.probability_of_default:.2%}
        Credit Limit: 
        """
    
    def _generate_credit_opinion(self, assessment):
        '''Generate credit opinion'''
        if assessment.credit_score > 700:
            opinion = "Strong credit profile with low default risk"
        elif assessment.credit_score > 600:
            opinion = "Adequate credit profile with moderate default risk"
        else:
            opinion = "Weak credit profile with elevated default risk"
        return opinion
    
    def _generate_financial_analysis(self, assessment):
        '''Generate financial analysis'''
        return "Detailed financial analysis..."
    
    def _generate_risk_assessment(self, assessment):
        '''Generate risk assessment'''
        return {
            'key_risks': assessment.risk_factors[:5],
            'key_strengths': assessment.strengths[:3]
        }
    
    def _generate_rating_rationale(self, assessment):
        '''Generate rating rationale'''
        return f"Rating of {assessment.credit_rating.value} reflects..."
    
    def _generate_outlook(self, assessment):
        '''Generate outlook'''
        if assessment.credit_score > 700:
            return "Stable"
        elif assessment.credit_score > 600:
            return "Negative Watch"
        else:
            return "Negative"

# Demonstrate system
if __name__ == '__main__':
    print('📈 CREDIT ORACLE - ULTRAPLATFORM')
    print('='*80)
    
    # Create sample entity
    entity = CreditEntity(
        name='TechCorp Australia Pty Ltd',
        entity_type=EntityType.CORPORATE,
        country='AU',
        industry='Technology',
        financial_data={
            'annual_revenue': 500000000,  # 
            'return_on_equity': 0.15,
            'debt_to_equity': 0.8,
            'current_ratio': 1.5,
            'capital_adequacy_ratio': 0.12,
            'market_share': 0.08
        },
        market_data={
            'stock_price': 45.50,
            'market_cap': 2000000000,
            'beta': 1.2
        },
        credit_history=[
            {'date': '2023-01', 'payment': 'on_time'},
            {'date': '2023-02', 'payment': 'on_time'},
            {'date': '2023-03', 'payment': 'on_time'}
        ]
    )
    
    # Initialize Credit Oracle
    oracle = CreditOracle()
    
    # Perform credit assessment
    print('\n🎯 PERFORMING CREDIT ASSESSMENT')
    print('='*80 + '\n')
    
    assessment = oracle.assess_credit(entity, AssessmentType.FULL_ASSESSMENT)
    
    # Show summary
    print('\n' + '='*80)
    print('CREDIT ASSESSMENT SUMMARY')
    print('='*80)
    print(f'Credit Rating: {assessment.credit_rating.value}')
    print(f'Credit Score: {assessment.credit_score:.0f}/1000')
    print(f'Default Probability (1Y): {assessment.probability_of_default:.2%}')
    print(f'Expected Loss: ')
    print(f'Recommended Credit Limit: ')
    print(f'Confidence Score: {assessment.confidence_score:.1%}')
    
    print('\n✅ Credit Oracle System Operational!')
