from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import redis
import jwt

# FastAPI imports (will be installed separately)
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UltraPlatform Real-Time Decision API",
    description="Real-time credit and risk decision engine",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== ENUMS ====================

class DecisionType(str, Enum):
    CREDIT_APPROVAL = "credit_approval"
    LIMIT_INCREASE = "limit_increase"
    RISK_ASSESSMENT = "risk_assessment"
    PRICING_DECISION = "pricing_decision"
    FRAUD_CHECK = "fraud_check"
    PORTFOLIO_ACTION = "portfolio_action"
    COLLECTION_STRATEGY = "collection_strategy"

class DecisionOutcome(str, Enum):
    APPROVED = "approved"
    DECLINED = "declined"
    REFER = "refer"
    PENDING = "pending"
    CONDITIONAL = "conditional"

class RiskLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNACCEPTABLE = "unacceptable"

class ResponseTime(str, Enum):
    INSTANT = "instant"  # < 100ms
    FAST = "fast"       # < 500ms
    NORMAL = "normal"   # < 2s
    SLOW = "slow"       # > 2s

# ==================== MODELS ====================

class CustomerData(BaseModel):
    """Customer data model for API requests"""
    customer_id: str
    customer_type: str = "individual"
    
    # Financial information
    annual_income: float = Field(gt=0)
    monthly_income: Optional[float] = None
    employment_status: str = "employed"
    employment_duration_months: int = 12
    
    # Credit information
    credit_score: Optional[int] = Field(None, ge=300, le=850)
    existing_debt: float = Field(ge=0, default=0)
    monthly_debt_payments: float = Field(ge=0, default=0)
    
    # Request specifics
    requested_amount: float = Field(gt=0)
    requested_term_months: Optional[int] = Field(None, gt=0, le=360)
    purpose: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "customer_id": "CUST_001",
                "customer_type": "individual",
                "annual_income": 120000,
                "employment_status": "employed",
                "employment_duration_months": 36,
                "credit_score": 720,
                "existing_debt": 50000,
                "monthly_debt_payments": 2000,
                "requested_amount": 25000,
                "requested_term_months": 60,
                "purpose": "debt_consolidation"
            }
        }
    
    @validator('monthly_income', always=True)
    def set_monthly_income(cls, v, values):
        if v is None and 'annual_income' in values:
            return values['annual_income'] / 12
        return v

class DecisionRequest(BaseModel):
    """Main decision request model"""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType
    customer_data: CustomerData
    context: Optional[Dict[str, Any]] = {}
    urgency: str = "normal"
    bypass_cache: bool = False
    
    class Config:
        use_enum_values = True

class DecisionResponse(BaseModel):
    """Decision response model"""
    request_id: str
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.now)
    decision_type: DecisionType
    outcome: DecisionOutcome
    
    # Decision details
    approved_amount: Optional[float] = None
    approved_term: Optional[int] = None
    interest_rate: Optional[float] = None
    monthly_payment: Optional[float] = None
    
    # Risk assessment
    risk_score: float = Field(ge=0, le=100)
    risk_level: RiskLevel
    probability_of_default: Optional[float] = Field(None, ge=0, le=1)
    
    # Decision factors
    key_factors: List[Dict[str, Any]] = []
    conditions: List[str] = []
    decline_reasons: List[str] = []
    
    # Metadata
    processing_time_ms: float
    response_time_category: ResponseTime
    confidence_score: float = Field(ge=0, le=1)
    
    class Config:
        use_enum_values = True

class HealthCheckResponse(BaseModel):
    """Health check response"""
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str = "2.0.0"
    uptime_seconds: float
    total_requests: int
    cache_hit_rate: float
    average_response_time_ms: float

# ==================== DECISION ENGINES ====================

class DecisionEngine(ABC):
    """Abstract base class for decision engines"""
    
    @abstractmethod
    async def make_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        pass

class CreditApprovalEngine(DecisionEngine):
    """Credit approval decision engine"""
    
    def __init__(self):
        self.approval_rules = self._initialize_rules()
        self.risk_thresholds = self._initialize_thresholds()
        
    def _initialize_rules(self):
        return {
            'min_income': 30000,
            'max_dti': 0.45,
            'min_credit_score': 600,
            'min_employment_months': 6,
            'max_loan_to_income': 5.0
        }
    
    def _initialize_thresholds(self):
        return {
            'auto_approve': {'score': 750, 'dti': 0.30},
            'auto_decline': {'score': 550, 'dti': 0.50},
            'refer': {'score': 650, 'dti': 0.40}
        }
    
    async def make_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        """Make credit approval decision"""
        customer = request.customer_data
        
        # Calculate DTI
        total_monthly_debt = customer.monthly_debt_payments + (
            customer.requested_amount / (request.customer_data.requested_term_months or 60)
        )
        dti = total_monthly_debt / customer.monthly_income
        
        # Check hard rules
        decline_reasons = []
        
        if customer.annual_income < self.approval_rules['min_income']:
            decline_reasons.append(f"Income below minimum ({self.approval_rules['min_income']})")
        
        if dti > self.approval_rules['max_dti']:
            decline_reasons.append(f"DTI ratio too high ({dti:.1%})")
        
        if customer.credit_score and customer.credit_score < self.approval_rules['min_credit_score']:
            decline_reasons.append(f"Credit score below minimum ({self.approval_rules['min_credit_score']})")
        
        if customer.employment_duration_months < self.approval_rules['min_employment_months']:
            decline_reasons.append("Insufficient employment history")
        
        # Decision logic
        if decline_reasons:
            outcome = DecisionOutcome.DECLINED
            approved_amount = 0
        elif customer.credit_score >= self.risk_thresholds['auto_approve']['score'] and \
             dti <= self.risk_thresholds['auto_approve']['dti']:
            outcome = DecisionOutcome.APPROVED
            approved_amount = customer.requested_amount
        elif customer.credit_score <= self.risk_thresholds['auto_decline']['score'] or \
             dti >= self.risk_thresholds['auto_decline']['dti']:
            outcome = DecisionOutcome.DECLINED
            approved_amount = 0
            if not decline_reasons:
                decline_reasons.append("Risk profile exceeds acceptable limits")
        else:
            outcome = DecisionOutcome.REFER
            approved_amount = customer.requested_amount * 0.8  # Conditional approval
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(customer, dti)
        
        # Determine interest rate
        interest_rate = self._calculate_interest_rate(risk_score, customer.credit_score)
        
        # Calculate monthly payment
        if approved_amount > 0:
            term = customer.requested_term_months or 60
            monthly_rate = interest_rate / 12
            monthly_payment = approved_amount * (monthly_rate * (1 + monthly_rate)**term) / \
                            ((1 + monthly_rate)**term - 1)
        else:
            monthly_payment = 0
        
        return {
            'outcome': outcome,
            'approved_amount': approved_amount,
            'approved_term': customer.requested_term_months,
            'interest_rate': interest_rate,
            'monthly_payment': monthly_payment,
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'dti_ratio': dti,
            'decline_reasons': decline_reasons,
            'key_factors': [
                {'factor': 'DTI Ratio', 'value': f'{dti:.1%}', 'impact': 'high'},
                {'factor': 'Credit Score', 'value': customer.credit_score, 'impact': 'high'},
                {'factor': 'Income', 'value': f'', 'impact': 'medium'}
            ]
        }
    
    def _calculate_risk_score(self, customer, dti):
        """Calculate risk score (0-100, lower is better)"""
        base_score = 50
        
        # Credit score impact
        if customer.credit_score:
            credit_factor = (850 - customer.credit_score) / 5.5
            base_score += credit_factor
        
        # DTI impact
        dti_factor = dti * 100
        base_score += dti_factor
        
        # Employment stability
        if customer.employment_duration_months < 12:
            base_score += 10
        elif customer.employment_duration_months > 36:
            base_score -= 5
        
        return min(100, max(0, base_score))
    
    def _calculate_interest_rate(self, risk_score, credit_score):
        """Calculate interest rate based on risk"""
        base_rate = 0.035  # 3.5% base rate
        
        # Risk premium
        risk_premium = risk_score * 0.002  # 0.2% per 10 risk points
        
        # Credit score adjustment
        if credit_score:
            if credit_score >= 750:
                credit_adj = -0.01
            elif credit_score >= 700:
                credit_adj = 0
            elif credit_score >= 650:
                credit_adj = 0.02
            else:
                credit_adj = 0.04
        else:
            credit_adj = 0.02
        
        return base_rate + risk_premium + credit_adj
    
    def _get_risk_level(self, risk_score):
        """Determine risk level from score"""
        if risk_score < 20:
            return RiskLevel.VERY_LOW
        elif risk_score < 40:
            return RiskLevel.LOW
        elif risk_score < 60:
            return RiskLevel.MEDIUM
        elif risk_score < 75:
            return RiskLevel.HIGH
        elif risk_score < 90:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.UNACCEPTABLE

class RiskAssessmentEngine(DecisionEngine):
    """Risk assessment decision engine"""
    
    async def make_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        """Perform risk assessment"""
        customer = request.customer_data
        
        # Calculate various risk metrics
        pd = self._calculate_probability_of_default(customer)
        lgd = self._calculate_loss_given_default(customer)
        ead = customer.requested_amount
        expected_loss = pd * lgd * ead
        
        # Risk score calculation
        risk_score = pd * 100 * 2  # Simple transformation
        risk_level = self._determine_risk_level(pd)
        
        # Determine outcome
        if pd < 0.02:
            outcome = DecisionOutcome.APPROVED
        elif pd < 0.05:
            outcome = DecisionOutcome.CONDITIONAL
        elif pd < 0.10:
            outcome = DecisionOutcome.REFER
        else:
            outcome = DecisionOutcome.DECLINED
        
        return {
            'outcome': outcome,
            'risk_score': risk_score,
            'risk_level': risk_level,
            'probability_of_default': pd,
            'loss_given_default': lgd,
            'exposure_at_default': ead,
            'expected_loss': expected_loss,
            'key_factors': [
                {'factor': 'Default Probability', 'value': f'{pd:.2%}', 'impact': 'high'},
                {'factor': 'Expected Loss', 'value': f'', 'impact': 'high'},
                {'factor': 'LGD', 'value': f'{lgd:.1%}', 'impact': 'medium'}
            ]
        }
    
    def _calculate_probability_of_default(self, customer):
        """Calculate PD using simplified model"""
        base_pd = 0.02
        
        # Credit score adjustment
        if customer.credit_score:
            if customer.credit_score >= 750:
                pd_mult = 0.3
            elif customer.credit_score >= 700:
                pd_mult = 0.5
            elif customer.credit_score >= 650:
                pd_mult = 1.0
            elif customer.credit_score >= 600:
                pd_mult = 2.0
            else:
                pd_mult = 3.0
        else:
            pd_mult = 1.5
        
        # DTI adjustment
        dti = customer.monthly_debt_payments / customer.monthly_income
        if dti > 0.4:
            pd_mult *= 1.5
        elif dti > 0.3:
            pd_mult *= 1.2
        
        return min(1.0, base_pd * pd_mult)
    
    def _calculate_loss_given_default(self, customer):
        """Calculate LGD"""
        # Simplified LGD model
        if customer.customer_type == "secured":
            return 0.35
        else:
            return 0.65
    
    def _determine_risk_level(self, pd):
        """Determine risk level from PD"""
        if pd < 0.01:
            return RiskLevel.VERY_LOW
        elif pd < 0.03:
            return RiskLevel.LOW
        elif pd < 0.05:
            return RiskLevel.MEDIUM
        elif pd < 0.10:
            return RiskLevel.HIGH
        elif pd < 0.20:
            return RiskLevel.VERY_HIGH
        else:
            return RiskLevel.UNACCEPTABLE

class PricingEngine(DecisionEngine):
    """Pricing decision engine"""
    
    async def make_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        """Determine optimal pricing"""
        customer = request.customer_data
        
        # Risk-based pricing
        base_rate = 0.045
        
        # Risk adjustments
        risk_premium = self._calculate_risk_premium(customer)
        competitive_adj = self._competitive_adjustment(customer)
        volume_discount = self._volume_discount(customer.requested_amount)
        
        final_rate = base_rate + risk_premium + competitive_adj - volume_discount
        final_rate = max(0.029, min(0.25, final_rate))  # Floor and ceiling
        
        # Calculate monthly payment
        term = customer.requested_term_months or 60
        monthly_rate = final_rate / 12
        monthly_payment = customer.requested_amount * \
                         (monthly_rate * (1 + monthly_rate)**term) / \
                         ((1 + monthly_rate)**term - 1)
        
        # Total interest
        total_interest = (monthly_payment * term) - customer.requested_amount
        
        return {
            'outcome': DecisionOutcome.APPROVED,
            'base_rate': base_rate,
            'risk_premium': risk_premium,
            'final_rate': final_rate,
            'monthly_payment': monthly_payment,
            'total_interest': total_interest,
            'approved_amount': customer.requested_amount,
            'approved_term': term,
            'interest_rate': final_rate,
            'key_factors': [
                {'factor': 'Base Rate', 'value': f'{base_rate:.2%}', 'impact': 'medium'},
                {'factor': 'Risk Premium', 'value': f'{risk_premium:.2%}', 'impact': 'high'},
                {'factor': 'Final APR', 'value': f'{final_rate:.2%}', 'impact': 'high'}
            ]
        }
    
    def _calculate_risk_premium(self, customer):
        """Calculate risk-based premium"""
        premium = 0
        
        # Credit score based
        if customer.credit_score:
            if customer.credit_score < 650:
                premium += 0.04
            elif customer.credit_score < 700:
                premium += 0.02
            elif customer.credit_score < 750:
                premium += 0.01
        else:
            premium += 0.025
        
        # DTI based
        dti = customer.monthly_debt_payments / customer.monthly_income
        if dti > 0.4:
            premium += 0.03
        elif dti > 0.3:
            premium += 0.015
        
        return premium
    
    def _competitive_adjustment(self, customer):
        """Competitive market adjustment"""
        # Simplified competitive pricing
        if customer.credit_score and customer.credit_score > 750:
            return -0.005  # Discount for prime customers
        return 0
    
    def _volume_discount(self, amount):
        """Volume-based discount"""
        if amount > 100000:
            return 0.005
        elif amount > 50000:
            return 0.003
        return 0

class FraudDetectionEngine(DecisionEngine):
    """Fraud detection engine"""
    
    async def make_decision(self, request: DecisionRequest) -> Dict[str, Any]:
        """Detect potential fraud"""
        customer = request.customer_data
        
        fraud_score = 0
        fraud_signals = []
        
        # Check for fraud indicators
        if customer.employment_duration_months < 3:
            fraud_score += 20
            fraud_signals.append("Very recent employment")
        
        # Income to loan ratio check
        if customer.requested_amount > customer.annual_income * 0.5:
            fraud_score += 30
            fraud_signals.append("High loan to income ratio")
        
        # Velocity check (simplified)
        if request.context.get('recent_applications', 0) > 3:
            fraud_score += 40
            fraud_signals.append("Multiple recent applications")
        
        # Determine outcome
        if fraud_score < 30:
            outcome = DecisionOutcome.APPROVED
            risk_level = RiskLevel.LOW
        elif fraud_score < 60:
            outcome = DecisionOutcome.REFER
            risk_level = RiskLevel.MEDIUM
        else:
            outcome = DecisionOutcome.DECLINED
            risk_level = RiskLevel.HIGH
        
        return {
            'outcome': outcome,
            'fraud_score': fraud_score,
            'risk_level': risk_level,
            'fraud_signals': fraud_signals,
            'risk_score': fraud_score,
            'key_factors': [
                {'factor': 'Fraud Score', 'value': fraud_score, 'impact': 'high'},
                {'factor': 'Signals Detected', 'value': len(fraud_signals), 'impact': 'medium'}
            ]
        }

# ==================== MAIN DECISION ORCHESTRATOR ====================

class DecisionOrchestrator:
    """Orchestrates decision making across engines"""
    
    def __init__(self):
        self.engines = {
            DecisionType.CREDIT_APPROVAL: CreditApprovalEngine(),
            DecisionType.RISK_ASSESSMENT: RiskAssessmentEngine(),
            DecisionType.PRICING_DECISION: PricingEngine(),
            DecisionType.FRAUD_CHECK: FraudDetectionEngine()
        }
        self.cache = {}  # Simple in-memory cache
        self.metrics = {
            'total_requests': 0,
            'cache_hits': 0,
            'total_response_time': 0
        }
        self.start_time = time.time()
    
    async def process_decision(self, request: DecisionRequest) -> DecisionResponse:
        """Process decision request"""
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if not request.bypass_cache and cache_key in self.cache:
            self.metrics['cache_hits'] += 1
            cached_response = self.cache[cache_key]
            cached_response.processing_time_ms = (time.time() - start_time) * 1000
            return cached_response
        
        # Get appropriate engine
        engine = self.engines.get(request.decision_type)
        if not engine:
            # Default to credit approval
            engine = self.engines[DecisionType.CREDIT_APPROVAL]
        
        # Make decision
        try:
            decision_result = await engine.make_decision(request)
        except Exception as e:
            logger.error(f"Decision engine error: {e}")
            decision_result = {
                'outcome': DecisionOutcome.DECLINED,
                'risk_score': 100,
                'risk_level': RiskLevel.UNACCEPTABLE,
                'decline_reasons': ["System error occurred"]
            }
        
        # Process response time
        processing_time = (time.time() - start_time) * 1000
        
        # Determine response time category
        if processing_time < 100:
            response_category = ResponseTime.INSTANT
        elif processing_time < 500:
            response_category = ResponseTime.FAST
        elif processing_time < 2000:
            response_category = ResponseTime.NORMAL
        else:
            response_category = ResponseTime.SLOW
        
        # Create response
        response = DecisionResponse(
            request_id=request.request_id,
            decision_type=request.decision_type,
            outcome=decision_result.get('outcome', DecisionOutcome.DECLINED),
            approved_amount=decision_result.get('approved_amount'),
            approved_term=decision_result.get('approved_term'),
            interest_rate=decision_result.get('interest_rate'),
            monthly_payment=decision_result.get('monthly_payment'),
            risk_score=decision_result.get('risk_score', 50),
            risk_level=decision_result.get('risk_level', RiskLevel.MEDIUM),
            probability_of_default=decision_result.get('probability_of_default'),
            key_factors=decision_result.get('key_factors', []),
            conditions=decision_result.get('conditions', []),
            decline_reasons=decision_result.get('decline_reasons', []),
            processing_time_ms=processing_time,
            response_time_category=response_category,
            confidence_score=0.85  # Simplified confidence
        )
        
        # Cache response
        self.cache[cache_key] = response
        
        # Update metrics
        self.metrics['total_requests'] += 1
        self.metrics['total_response_time'] += processing_time
        
        return response
    
    def _get_cache_key(self, request: DecisionRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            'decision_type': request.decision_type,
            'customer_id': request.customer_data.customer_id,
            'requested_amount': request.customer_data.requested_amount,
            'credit_score': request.customer_data.credit_score,
            'annual_income': request.customer_data.annual_income
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get_metrics(self) -> Dict:
        """Get system metrics"""
        uptime = time.time() - self.start_time
        total_requests = max(1, self.metrics['total_requests'])
        
        return {
            'uptime_seconds': uptime,
            'total_requests': total_requests,
            'cache_hit_rate': self.metrics['cache_hits'] / total_requests,
            'average_response_time_ms': self.metrics['total_response_time'] / total_requests
        }

# ==================== API ENDPOINTS ====================

# Initialize orchestrator
orchestrator = DecisionOrchestrator()

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token"""
    token = credentials.credentials
    # Simplified token verification - implement proper JWT validation in production
    if not token or len(token) < 10:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )
    return token

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "UltraPlatform Real-Time Decision API",
        "version": "2.0.0",
        "documentation": "/api/docs"
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    metrics = orchestrator.get_metrics()
    
    return HealthCheckResponse(
        status="healthy",
        uptime_seconds=metrics['uptime_seconds'],
        total_requests=metrics['total_requests'],
        cache_hit_rate=metrics['cache_hit_rate'],
        average_response_time_ms=metrics['average_response_time_ms']
    )

@app.post("/decision", 
          response_model=DecisionResponse,
          tags=["Decision"],
          dependencies=[Depends(verify_token)])
@limiter.limit("100/minute")
async def make_decision(
    request: Request,
    decision_request: DecisionRequest
):
    """Make real-time decision"""
    try:
        response = await orchestrator.process_decision(decision_request)
        return response
    except Exception as e:
        logger.error(f"Decision processing error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@app.post("/batch-decision",
          response_model=List[DecisionResponse],
          tags=["Decision"],
          dependencies=[Depends(verify_token)])
@limiter.limit("10/minute")
async def batch_decision(
    request: Request,
    decision_requests: List[DecisionRequest]
):
    """Process batch of decisions"""
    if len(decision_requests) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 requests per batch"
        )
    
    responses = []
    for decision_request in decision_requests:
        response = await orchestrator.process_decision(decision_request)
        responses.append(response)
    
    return responses

@app.get("/metrics", tags=["Metrics"], dependencies=[Depends(verify_token)])
async def get_metrics():
    """Get API metrics"""
    return orchestrator.get_metrics()

@app.post("/simulate", 
          response_model=DecisionResponse,
          tags=["Simulation"],
          dependencies=[Depends(verify_token)])
async def simulate_decision(
    customer_id: str = "TEST_001",
    annual_income: float = 75000,
    credit_score: int = 700,
    requested_amount: float = 25000,
    decision_type: DecisionType = DecisionType.CREDIT_APPROVAL
):
    """Simulate a decision for testing"""
    # Create test request
    test_customer = CustomerData(
        customer_id=customer_id,
        annual_income=annual_income,
        credit_score=credit_score,
        requested_amount=requested_amount,
        monthly_debt_payments=annual_income / 12 * 0.2,  # 20% of monthly income
        existing_debt=requested_amount * 2
    )
    
    test_request = DecisionRequest(
        decision_type=decision_type,
        customer_data=test_customer,
        bypass_cache=True
    )
    
    response = await orchestrator.process_decision(test_request)
    return response

# ==================== STARTUP/SHUTDOWN ====================

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Starting UltraPlatform Real-Time Decision API")
    logger.info("Version: 2.0.0")
    logger.info("Documentation available at /api/docs")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Shutting down Real-Time Decision API")
    metrics = orchestrator.get_metrics()
    logger.info(f"Total requests processed: {metrics['total_requests']}")
    logger.info(f"Average response time: {metrics['average_response_time_ms']:.2f}ms")

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 REAL-TIME DECISION API - ULTRAPLATFORM")
    print("="*80)
    print("Starting API server...")
    print("\nAPI Documentation: http://localhost:8000/api/docs")
    print("Health Check: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop")
    print("="*80)
    
    # Run the API
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
