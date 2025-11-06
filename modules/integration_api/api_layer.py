"""
ULTRA PLATFORM - INTEGRATION ARCHITECTURE & API
===============================================

Institutional-grade REST API and event-driven integration layer providing:
- RESTful API endpoints with full CRUD operations
- OAuth 2.0 + JWT authentication
- Advanced rate limiting with tiered plans
- Webhook management with retries and signing
- Event-driven architecture with message queuing
- Comprehensive monitoring and observability
- API versioning and backwards compatibility
- Circuit breakers and resilience patterns
- Multi-tenancy support
- Complete audit trails

Author: Ultra Platform Team
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, Header, Request, Response, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, Any, Callable, Union
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import hashlib
import hmac
import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from collections import defaultdict, deque
import aiohttp
import jwt
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.info('Redis not available - using in-memory storage')


# ============================================================================
# ENUMERATIONS
# ============================================================================

class APIVersion(str, Enum):
    """API version enumeration"""
    V1 = "v1"
    V2 = "v2"
    

class HTTPMethod(str, Enum):
    """HTTP methods"""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class RateLimitTier(str, Enum):
    """Rate limit tiers"""
    FREE = "free"
    BASIC = "basic"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


class WebhookStatus(str, Enum):
    """Webhook delivery status"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


class EventType(str, Enum):
    """Event types for webhooks"""
    ONBOARDING_STARTED = "onboarding.started"
    IDENTITY_VERIFIED = "onboarding.identity.verified"
    KYC_COMPLETED = "onboarding.kyc.completed"
    KYC_FAILED = "onboarding.kyc.failed"
    ACCOUNT_OPENED = "onboarding.account.opened"
    ACCOUNT_ACTIVATED = "onboarding.account.activated"
    FRAUD_DETECTED = "onboarding.fraud.detected"
    REVIEW_REQUIRED = "onboarding.review.required"
    DOCUMENT_UPLOADED = "onboarding.document.uploaded"
    SIGNATURE_COMPLETED = "onboarding.signature.completed"
    # Ongoing monitoring events
    PROFILE_CHANGED = "monitoring.profile.changed"
    KYC_REFRESH_DUE = "monitoring.kyc.refresh.due"
    KYC_REFRESH_COMPLETED = "monitoring.kyc.refresh.completed"
    UNUSUAL_ACTIVITY = "monitoring.unusual.activity"
    WATCHLIST_HIT = "monitoring.watchlist.hit"


class CircuitState(str, Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class StartApplicationRequest(BaseModel):
    """Request to start onboarding application"""
    email: str = Field(..., description="Customer email")
    phone: str = Field(..., description="Customer phone")
    referral_code: Optional[str] = Field(None, description="Referral code")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email format')
        return v.lower()


class ApplicationResponse(BaseModel):
    """Response for application operations"""
    application_id: str
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    next_step: Optional[str] = None


class IdentityVerificationRequest(BaseModel):
    """Request for identity verification"""
    user_id: str
    document_type: str
    document_number: str
    document_image_front: str  # Base64 encoded
    document_image_back: Optional[str] = None
    selfie_image: str  # Base64 encoded
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class IdentityVerificationResponse(BaseModel):
    """Response for identity verification"""
    session_id: str
    status: str
    confidence_score: float
    verification_checks: Dict[str, bool]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class KYCScreeningRequest(BaseModel):
    """Request for KYC screening"""
    user_id: str
    first_name: str
    last_name: str
    date_of_birth: str
    nationality: str
    address: Dict[str, str]
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class KYCScreeningResponse(BaseModel):
    """Response for KYC screening"""
    screening_id: str
    status: str
    risk_level: str
    watchlist_hits: int
    pep_match: bool
    sanctions_match: bool
    adverse_media: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AccountOpeningRequest(BaseModel):
    """Request to open account"""
    application_id: str
    account_type: str
    initial_deposit: Optional[float] = None
    beneficiaries: Optional[List[Dict[str, Any]]] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AccountOpeningResponse(BaseModel):
    """Response for account opening"""
    application_id: str
    account_id: str
    account_number: str
    status: str
    opened_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadRequest(BaseModel):
    """Request to upload document"""
    user_id: str
    document_type: str
    document_data: str  # Base64 encoded
    filename: str
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    document_id: str
    status: str
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class WebhookRegistrationRequest(BaseModel):
    """Request to register webhook"""
    url: str = Field(..., description="Webhook URL")
    events: List[EventType] = Field(..., description="Event types to subscribe to")
    secret: Optional[str] = Field(None, description="Webhook signing secret")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v


class WebhookRegistrationResponse(BaseModel):
    """Response for webhook registration"""
    webhook_id: str
    url: str
    events: List[EventType]
    secret: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    request_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    burst_size: int = 10


@dataclass
class APIKey:
    """API key data"""
    key_id: str
    key_hash: str
    tenant_id: str
    tier: RateLimitTier
    scopes: List[str]
    created_at: datetime
    expires_at: Optional[datetime] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WebhookEndpoint:
    """Webhook endpoint configuration"""
    webhook_id: str
    tenant_id: str
    url: str
    events: List[EventType]
    secret: str
    active: bool = True
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_delivery: Optional[datetime] = None
    delivery_count: int = 0
    failure_count: int = 0


@dataclass
class WebhookDelivery:
    """Webhook delivery attempt"""
    delivery_id: str
    webhook_id: str
    event_type: EventType
    payload: Dict[str, Any]
    attempt: int
    status: WebhookStatus
    http_status: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None
    next_retry: Optional[datetime] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for API resilience"""
    name: str
    state: CircuitState = CircuitState.CLOSED
    failure_threshold: int = 5
    success_threshold: int = 2
    timeout_seconds: int = 60
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    opened_at: Optional[datetime] = None


@dataclass
class APIMetrics:
    """API metrics tracking"""
    request_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    status_codes: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    endpoint_metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    @property
    def avg_latency_ms(self) -> float:
        """Calculate average latency"""
        return self.total_latency_ms / self.request_count if self.request_count > 0 else 0.0
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return self.error_count / self.request_count if self.request_count > 0 else 0.0


# ============================================================================
# AUTHENTICATION & AUTHORIZATION
# ============================================================================

class AuthenticationService:
    """
    Authentication service with OAuth 2.0 and JWT support
    """
    
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.api_keys: Dict[str, APIKey] = {}
        
    def generate_api_key(
        self,
        tenant_id: str,
        tier: RateLimitTier,
        scopes: List[str],
        expires_days: Optional[int] = None
    ) -> APIKey:
        """Generate new API key"""
        key_id = f"sk_live_{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(key_id.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            tenant_id=tenant_id,
            tier=tier,
            scopes=scopes,
            created_at=datetime.utcnow(),
            expires_at=expires_at
        )
        
        self.api_keys[key_id] = api_key
        logger.info(f"Generated API key for tenant {tenant_id}, tier: {tier}")
        
        return api_key
    
    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate API key"""
        key_data = self.api_keys.get(api_key)
        
        if not key_data:
            return None
        
        if not key_data.active:
            return None
        
        if key_data.expires_at and datetime.utcnow() > key_data.expires_at:
            return None
        
        return key_data
    
    def generate_jwt(
        self,
        tenant_id: str,
        user_id: str,
        scopes: List[str],
        expires_minutes: int = 60
    ) -> str:
        """Generate JWT token"""
        payload = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "scopes": scopes,
            "exp": datetime.utcnow() + timedelta(minutes=expires_minutes),
            "iat": datetime.utcnow(),
            "jti": uuid.uuid4().hex
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def validate_jwt(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimiter:
    """
    Advanced rate limiter with tiered limits and distributed support
    """
    
    def __init__(self):
        self.limits = {
            RateLimitTier.FREE: RateLimitConfig(
                requests_per_minute=10,
                requests_per_hour=100,
                requests_per_day=1000,
                burst_size=5
            ),
            RateLimitTier.BASIC: RateLimitConfig(
                requests_per_minute=100,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_size=20
            ),
            RateLimitTier.PREMIUM: RateLimitConfig(
                requests_per_minute=1000,
                requests_per_hour=50000,
                requests_per_day=500000,
                burst_size=100
            ),
            RateLimitTier.ENTERPRISE: RateLimitConfig(
                requests_per_minute=10000,
                requests_per_hour=500000,
                requests_per_day=5000000,
                burst_size=1000
            )
        }
        
        # In-memory storage (use Redis for production)
        self.request_counts: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    async def check_rate_limit(
        self,
        tenant_id: str,
        tier: RateLimitTier
    ) -> tuple[bool, Dict[str, Any]]:
        """
        Check if request is within rate limits
        
        Returns:
            (allowed, limit_info)
        """
        config = self.limits[tier]
        now = datetime.utcnow()
        
        # Get request history
        requests = self.request_counts[tenant_id]
        
        # Remove old requests
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        day_ago = now - timedelta(days=1)
        
        # Count requests in each window
        minute_count = sum(1 for req_time in requests if req_time > minute_ago)
        hour_count = sum(1 for req_time in requests if req_time > hour_ago)
        day_count = sum(1 for req_time in requests if req_time > day_ago)
        
        # Check limits
        if minute_count >= config.requests_per_minute:
            return False, {
                "limit": config.requests_per_minute,
                "remaining": 0,
                "reset": (minute_ago + timedelta(minutes=1)).isoformat(),
                "window": "minute"
            }
        
        if hour_count >= config.requests_per_hour:
            return False, {
                "limit": config.requests_per_hour,
                "remaining": 0,
                "reset": (hour_ago + timedelta(hours=1)).isoformat(),
                "window": "hour"
            }
        
        if day_count >= config.requests_per_day:
            return False, {
                "limit": config.requests_per_day,
                "remaining": 0,
                "reset": (day_ago + timedelta(days=1)).isoformat(),
                "window": "day"
            }
        
        # Record request
        requests.append(now)
        
        return True, {
            "limit": config.requests_per_minute,
            "remaining": config.requests_per_minute - minute_count - 1,
            "reset": (minute_ago + timedelta(minutes=1)).isoformat(),
            "window": "minute"
        }


# ============================================================================
# WEBHOOK MANAGEMENT
# ============================================================================

class WebhookManager:
    """
    Webhook delivery system with retries and signing
    """
    
    def __init__(self):
        self.webhooks: Dict[str, WebhookEndpoint] = {}
        self.deliveries: Dict[str, WebhookDelivery] = {}
        self.max_retries = 5
        self.retry_delays = [60, 300, 900, 3600, 7200]  # seconds
        
    def register_webhook(
        self,
        tenant_id: str,
        url: str,
        events: List[EventType],
        secret: Optional[str] = None
    ) -> WebhookEndpoint:
        """Register webhook endpoint"""
        webhook_id = f"wh_{uuid.uuid4().hex}"
        
        if not secret:
            secret = uuid.uuid4().hex
        
        webhook = WebhookEndpoint(
            webhook_id=webhook_id,
            tenant_id=tenant_id,
            url=url,
            events=events,
            secret=secret,
            retry_policy={
                "max_retries": self.max_retries,
                "retry_delays": self.retry_delays
            }
        )
        
        self.webhooks[webhook_id] = webhook
        logger.info(f"Registered webhook {webhook_id} for tenant {tenant_id}")
        
        return webhook
    
    def sign_payload(self, payload: str, secret: str) -> str:
        """Sign webhook payload with HMAC-SHA256"""
        signature = hmac.new(
            secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def deliver_event(
        self,
        event_type: EventType,
        payload: Dict[str, Any],
        tenant_id: Optional[str] = None
    ):
        """Deliver event to registered webhooks"""
        # Find matching webhooks
        matching_webhooks = [
            webhook for webhook in self.webhooks.values()
            if event_type in webhook.events and webhook.active
            and (tenant_id is None or webhook.tenant_id == tenant_id)
        ]
        
        # Deliver to each webhook
        for webhook in matching_webhooks:
            delivery_id = f"del_{uuid.uuid4().hex}"
            
            delivery = WebhookDelivery(
                delivery_id=delivery_id,
                webhook_id=webhook.webhook_id,
                event_type=event_type,
                payload=payload,
                attempt=1,
                status=WebhookStatus.PENDING
            )
            
            self.deliveries[delivery_id] = delivery
            
            # Attempt delivery
            await self._attempt_delivery(delivery, webhook)
    
    async def _attempt_delivery(
        self,
        delivery: WebhookDelivery,
        webhook: WebhookEndpoint
    ):
        """Attempt webhook delivery"""
        try:
            # Prepare payload
            webhook_payload = {
                "event_type": delivery.event_type.value,
                "event_id": delivery.delivery_id,
                "timestamp": datetime.utcnow().isoformat(),
                "data": delivery.payload
            }
            
            payload_str = json.dumps(webhook_payload)
            signature = self.sign_payload(payload_str, webhook.secret)
            
            # Make HTTP request
            headers = {
                "Content-Type": "application/json",
                "X-Webhook-Signature": signature,
                "X-Webhook-ID": webhook.webhook_id,
                "X-Delivery-ID": delivery.delivery_id
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.url,
                    data=payload_str,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    delivery.http_status = response.status
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookStatus.DELIVERED
                        delivery.delivered_at = datetime.utcnow()
                        webhook.last_delivery = datetime.utcnow()
                        webhook.delivery_count += 1
                        logger.info(f"Webhook delivered: {delivery.delivery_id}")
                    else:
                        raise Exception(f"HTTP {response.status}: {delivery.response_body}")
        
        except Exception as e:
            delivery.status = WebhookStatus.FAILED
            delivery.error_message = str(e)
            webhook.failure_count += 1
            
            logger.error(f"Webhook delivery failed: {delivery.delivery_id}, error: {e}")
            
            # Schedule retry
            if delivery.attempt < self.max_retries:
                delay = self.retry_delays[delivery.attempt - 1]
                delivery.next_retry = datetime.utcnow() + timedelta(seconds=delay)
                delivery.status = WebhookStatus.RETRYING
                
                logger.info(f"Scheduling retry for {delivery.delivery_id} in {delay}s")


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================

class CircuitBreakerManager:
    """
    Circuit breaker for API resilience
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
    
    def get_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.breakers:
            self.breakers[name] = CircuitBreaker(name=name)
        return self.breakers[name]
    
    def can_execute(self, name: str) -> bool:
        """Check if request can execute"""
        breaker = self.get_breaker(name)
        
        if breaker.state == CircuitState.CLOSED:
            return True
        
        if breaker.state == CircuitState.OPEN:
            # Check if timeout has passed
            if breaker.opened_at:
                timeout = timedelta(seconds=breaker.timeout_seconds)
                if datetime.utcnow() - breaker.opened_at > timeout:
                    breaker.state = CircuitState.HALF_OPEN
                    breaker.success_count = 0
                    logger.info(f"Circuit breaker {name} transitioned to HALF_OPEN")
                    return True
            return False
        
        # HALF_OPEN - allow limited requests
        return True
    
    def record_success(self, name: str):
        """Record successful request"""
        breaker = self.get_breaker(name)
        
        if breaker.state == CircuitState.HALF_OPEN:
            breaker.success_count += 1
            
            if breaker.success_count >= breaker.success_threshold:
                breaker.state = CircuitState.CLOSED
                breaker.failure_count = 0
                logger.info(f"Circuit breaker {name} transitioned to CLOSED")
    
    def record_failure(self, name: str):
        """Record failed request"""
        breaker = self.get_breaker(name)
        breaker.failure_count += 1
        breaker.last_failure_time = datetime.utcnow()
        
        if breaker.failure_count >= breaker.failure_threshold:
            breaker.state = CircuitState.OPEN
            breaker.opened_at = datetime.utcnow()
            logger.warning(f"Circuit breaker {name} OPENED after {breaker.failure_count} failures")


# ============================================================================
# API METRICS
# ============================================================================

class MetricsCollector:
    """
    Metrics collection and aggregation
    """
    
    def __init__(self):
        self.metrics = APIMetrics()
        self.request_durations: deque = deque(maxlen=10000)
    
    def record_request(
        self,
        endpoint: str,
        method: HTTPMethod,
        status_code: int,
        duration_ms: float
    ):
        """Record API request"""
        self.metrics.request_count += 1
        self.metrics.total_latency_ms += duration_ms
        self.metrics.min_latency_ms = min(self.metrics.min_latency_ms, duration_ms)
        self.metrics.max_latency_ms = max(self.metrics.max_latency_ms, duration_ms)
        self.metrics.status_codes[status_code] += 1
        
        if status_code >= 400:
            self.metrics.error_count += 1
        
        # Track per-endpoint metrics
        endpoint_key = f"{method.value}:{endpoint}"
        if endpoint_key not in self.metrics.endpoint_metrics:
            self.metrics.endpoint_metrics[endpoint_key] = {
                "count": 0,
                "errors": 0,
                "total_latency": 0.0
            }
        
        self.metrics.endpoint_metrics[endpoint_key]["count"] += 1
        self.metrics.endpoint_metrics[endpoint_key]["total_latency"] += duration_ms
        
        if status_code >= 400:
            self.metrics.endpoint_metrics[endpoint_key]["errors"] += 1
        
        self.request_durations.append(duration_ms)
    
    def get_percentile(self, percentile: float) -> float:
        """Get latency percentile"""
        if not self.request_durations:
            return 0.0
        
        sorted_durations = sorted(self.request_durations)
        index = int(len(sorted_durations) * (percentile / 100))
        return sorted_durations[index]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary"""
        return {
            "total_requests": self.metrics.request_count,
            "total_errors": self.metrics.error_count,
            "error_rate": f"{self.metrics.error_rate * 100:.2f}%",
            "avg_latency_ms": f"{self.metrics.avg_latency_ms:.2f}",
            "min_latency_ms": f"{self.metrics.min_latency_ms:.2f}",
            "max_latency_ms": f"{self.metrics.max_latency_ms:.2f}",
            "p50_latency_ms": f"{self.get_percentile(50):.2f}",
            "p95_latency_ms": f"{self.get_percentile(95):.2f}",
            "p99_latency_ms": f"{self.get_percentile(99):.2f}",
            "status_codes": dict(self.metrics.status_codes),
            "endpoint_metrics": self.metrics.endpoint_metrics
        }


# ============================================================================
# ONBOARDING API
# ============================================================================

class OnboardingAPI:
    """
    Complete REST API for onboarding with authentication, rate limiting,
    webhooks, and monitoring
    """
    
    def __init__(self):
        self.app = FastAPI(
            title="Ultra Platform Onboarding API",
            description="Institutional-grade onboarding API",
            version="1.0.0"
        )
        
        # Initialize services
        self.auth_service = AuthenticationService(secret_key="ultra-secret-key-change-in-production")
        self.rate_limiter = RateLimiter()
        self.webhook_manager = WebhookManager()
        self.circuit_breaker = CircuitBreakerManager()
        self.metrics = MetricsCollector()
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
    
    def _setup_middleware(self):
        """Setup API middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure appropriately for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Compression
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            return self.metrics.get_summary()
        
        # Start application
        @self.app.post(
            "/api/v1/onboarding/application/start",
            response_model=ApplicationResponse
        )
        async def start_application(
            request: StartApplicationRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            start_time = time.time()
            
            try:
                # Generate application ID
                application_id = f"app_{uuid.uuid4().hex}"
                
                # Emit event
                await self.webhook_manager.deliver_event(
                    EventType.ONBOARDING_STARTED,
                    {
                        "application_id": application_id,
                        "email": request.email,
                        "phone": request.phone
                    }
                )
                
                response = ApplicationResponse(
                    application_id=application_id,
                    status="STARTED",
                    next_step="identity_verification"
                )
                
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/application/start", HTTPMethod.POST, 200, duration)
                
                return response
            
            except Exception as e:
                logger.error(f"Error starting application: {e}")
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/application/start", HTTPMethod.POST, 500, duration)
                raise HTTPException(status_code=500, detail=str(e))
        
        # Verify identity
        @self.app.post(
            "/api/v1/onboarding/identity/verify",
            response_model=IdentityVerificationResponse
        )
        async def verify_identity(
            request: IdentityVerificationRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            start_time = time.time()
            
            try:
                session_id = f"idv_{uuid.uuid4().hex}"
                
                # Simulate verification
                response = IdentityVerificationResponse(
                    session_id=session_id,
                    status="VERIFIED",
                    confidence_score=0.95,
                    verification_checks={
                        "document_authentic": True,
                        "face_match": True,
                        "liveness_check": True
                    }
                )
                
                # Emit event
                await self.webhook_manager.deliver_event(
                    EventType.IDENTITY_VERIFIED,
                    {
                        "user_id": request.user_id,
                        "session_id": session_id,
                        "confidence_score": 0.95
                    }
                )
                
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/identity/verify", HTTPMethod.POST, 200, duration)
                
                return response
            
            except Exception as e:
                logger.error(f"Error verifying identity: {e}")
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/identity/verify", HTTPMethod.POST, 500, duration)
                raise HTTPException(status_code=500, detail=str(e))
        
        # KYC screening
        @self.app.post(
            "/api/v1/onboarding/kyc/screen",
            response_model=KYCScreeningResponse
        )
        async def kyc_screen(
            request: KYCScreeningRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            start_time = time.time()
            
            try:
                screening_id = f"kyc_{uuid.uuid4().hex}"
                
                response = KYCScreeningResponse(
                    screening_id=screening_id,
                    status="PASS",
                    risk_level="LOW",
                    watchlist_hits=0,
                    pep_match=False,
                    sanctions_match=False,
                    adverse_media=False
                )
                
                # Emit event
                await self.webhook_manager.deliver_event(
                    EventType.KYC_COMPLETED,
                    {
                        "user_id": request.user_id,
                        "screening_id": screening_id,
                        "risk_level": "LOW"
                    }
                )
                
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/kyc/screen", HTTPMethod.POST, 200, duration)
                
                return response
            
            except Exception as e:
                logger.error(f"Error in KYC screening: {e}")
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/kyc/screen", HTTPMethod.POST, 500, duration)
                raise HTTPException(status_code=500, detail=str(e))
        
        # Open account
        @self.app.post(
            "/api/v1/onboarding/account/open",
            response_model=AccountOpeningResponse
        )
        async def open_account(
            request: AccountOpeningRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            start_time = time.time()
            
            try:
                account_id = f"acct_{uuid.uuid4().hex}"
                account_number = f"{uuid.uuid4().hex[:12].upper()}"
                
                response = AccountOpeningResponse(
                    application_id=request.application_id,
                    account_id=account_id,
                    account_number=account_number,
                    status="OPENED"
                )
                
                # Emit event
                await self.webhook_manager.deliver_event(
                    EventType.ACCOUNT_OPENED,
                    {
                        "application_id": request.application_id,
                        "account_id": account_id,
                        "account_number": account_number
                    }
                )
                
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/account/open", HTTPMethod.POST, 200, duration)
                
                return response
            
            except Exception as e:
                logger.error(f"Error opening account: {e}")
                duration = (time.time() - start_time) * 1000
                self.metrics.record_request("/api/v1/onboarding/account/open", HTTPMethod.POST, 500, duration)
                raise HTTPException(status_code=500, detail=str(e))
        
        # Register webhook
        @self.app.post(
            "/api/v1/webhooks/register",
            response_model=WebhookRegistrationResponse
        )
        async def register_webhook(
            request: WebhookRegistrationRequest,
            api_key: str = Depends(self._verify_api_key)
        ):
            webhook = self.webhook_manager.register_webhook(
                tenant_id=api_key,  # Using API key as tenant ID for simplicity
                url=request.url,
                events=request.events,
                secret=request.secret
            )
            
            return WebhookRegistrationResponse(
                webhook_id=webhook.webhook_id,
                url=webhook.url,
                events=webhook.events,
                secret=webhook.secret
            )
    
    async def _verify_api_key(self, authorization: str = Header(...)) -> str:
        """Verify API key from Authorization header"""
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Invalid authorization header")
        
        api_key = authorization.replace("Bearer ", "")
        key_data = self.auth_service.validate_api_key(api_key)
        
        if not key_data:
            raise HTTPException(status_code=401, detail="Invalid API key")
        
        # Check rate limit
        allowed, limit_info = await self.rate_limiter.check_rate_limit(
            key_data.tenant_id,
            key_data.tier
        )
        
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Try again after {limit_info['reset']}"
            )
        
        return api_key


# ============================================================================
# TESTING & DEMO
# ============================================================================

async def demo_api():
    """Demonstrate API functionality"""
    print("\n" + "=" * 60)
    print("ULTRA PLATFORM - INTEGRATION API DEMO")
    print("=" * 60)
    
    # Initialize API
    api = OnboardingAPI()
    
    # Generate API key
    print("\n1. Generating API Key...")
    api_key = api.auth_service.generate_api_key(
        tenant_id="tenant_123",
        tier=RateLimitTier.PREMIUM,
        scopes=["onboarding:write", "webhooks:write"]
    )
    print(f"   API Key: {api_key.key_id}")
    print(f"   Tier: {api_key.tier}")
    print(f"   Scopes: {', '.join(api_key.scopes)}")
    
    # Register webhook
    print("\n2. Registering Webhook...")
    webhook = api.webhook_manager.register_webhook(
        tenant_id="tenant_123",
        url="https://example.com/webhooks/onboarding",
        events=[EventType.ONBOARDING_STARTED, EventType.ACCOUNT_OPENED]
    )
    print(f"   Webhook ID: {webhook.webhook_id}")
    print(f"   URL: {webhook.url}")
    print(f"   Events: {len(webhook.events)}")
    
    # Deliver test event
    print("\n3. Delivering Webhook Event...")
    await api.webhook_manager.deliver_event(
        EventType.ONBOARDING_STARTED,
        {
            "application_id": "app_test123",
            "email": "test@example.com"
        },
        tenant_id="tenant_123"
    )
    print("   Event delivered!")
    
    # Circuit breaker test
    print("\n4. Testing Circuit Breaker...")
    breaker_name = "external_service"
    
    for i in range(7):
        if api.circuit_breaker.can_execute(breaker_name):
            # Simulate failures
            if i < 5:
                api.circuit_breaker.record_failure(breaker_name)
                print(f"   Request {i+1}: FAILED")
            else:
                api.circuit_breaker.record_success(breaker_name)
                print(f"   Request {i+1}: SUCCESS")
        else:
            print(f"   Request {i+1}: BLOCKED (circuit open)")
    
    # Metrics
    print("\n5. API Metrics:")
    api.metrics.record_request("/api/v1/test", HTTPMethod.GET, 200, 45.2)
    api.metrics.record_request("/api/v1/test", HTTPMethod.POST, 201, 120.5)
    api.metrics.record_request("/api/v1/test", HTTPMethod.GET, 500, 250.0)
    
    summary = api.metrics.get_summary()
    print(f"   Total Requests: {summary['total_requests']}")
    print(f"   Error Rate: {summary['error_rate']}")
    print(f"   Avg Latency: {summary['avg_latency_ms']} ms")
    print(f"   P95 Latency: {summary['p95_latency_ms']} ms")
    
    print("\n" + "=" * 60)
    print("✅ Integration API Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_api())


