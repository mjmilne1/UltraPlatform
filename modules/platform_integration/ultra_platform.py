"""
ULTRA PLATFORM - COMPLETE INTEGRATION LAYER
============================================

Main orchestration layer that wires all modules together through Data Mesh
architecture for seamless cross-domain data sharing and workflow execution.

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class WorkflowStage(str, Enum):
    """Onboarding workflow stages"""
    APPLICATION_STARTED = "application_started"
    IDENTITY_VERIFICATION = "identity_verification"
    KYC_SCREENING = "kyc_screening"
    FRAUD_CHECK = "fraud_check"
    ONGOING_MONITORING_SETUP = "ongoing_monitoring_setup"
    SUITABILITY_ASSESSMENT = "suitability_assessment"
    ACCOUNT_OPENING = "account_opening"
    COMPLIANCE_APPROVAL = "compliance_approval"
    ACCOUNT_ACTIVATED = "account_activated"


class DataDomain(str, Enum):
    """Data domains in the mesh"""
    IDENTITY = "identity"
    KYC = "kyc"
    FRAUD = "fraud"
    MONITORING = "monitoring"
    ACCOUNT = "account"
    COMPLIANCE = "compliance"
    AUDIT = "audit"
    PERFORMANCE = "performance"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CustomerOnboardingState:
    """Complete customer onboarding state"""
    customer_id: str
    application_id: str
    current_stage: WorkflowStage
    started_at: datetime
    completed_stages: List[WorkflowStage] = field(default_factory=list)
    data_products: Dict[DataDomain, Dict[str, Any]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def completion_percentage(self) -> float:
        """Calculate completion percentage"""
        total_stages = len(WorkflowStage)
        completed = len(self.completed_stages)
        return (completed / total_stages) * 100


@dataclass
class DataProduct:
    """Data product in the mesh"""
    domain: DataDomain
    product_id: str
    data: Dict[str, Any]
    version: str
    created_at: datetime
    quality_score: float


# ============================================================================
# ULTRA PLATFORM - MAIN ORCHESTRATOR
# ============================================================================

class UltraPlatform:
    """
    Main Ultra Platform orchestrator that integrates all modules
    through Data Mesh architecture
    """
    
    def __init__(self):
        logger.info("Initializing Ultra Platform...")
        
        # Import all modules
        self._initialize_modules()
        
        # Data Mesh hub for cross-domain communication
        self.data_mesh = self._initialize_data_mesh()
        
        # Active customer workflows
        self.active_workflows: Dict[str, CustomerOnboardingState] = {}
        
        logger.info("✅ Ultra Platform initialized successfully")
    
    def _initialize_modules(self):
        """Initialize all platform modules"""
        try:
            # Import audit and data mesh
            from modules.audit_reporting.audit_system import OnboardingAuditService
            from modules.audit_reporting.data_mesh_integration import DataMeshAuditService
            
            # Import performance
            from modules.performance.performance_system import (
                PerformanceMonitoringService,
                AutoScalingService,
                CacheManagementService,
                CapacityPlanningService
            )
            
            # Initialize services
            self.audit_service = OnboardingAuditService()
            self.mesh_audit = DataMeshAuditService()
            self.performance_monitor = PerformanceMonitoringService()
            self.auto_scaler = AutoScalingService(self.performance_monitor)
            self.cache_manager = CacheManagementService()
            self.capacity_planner = CapacityPlanningService(self.performance_monitor)
            
            logger.info("✅ All modules loaded")
            
        except ImportError as e:
            logger.warning(f"Some modules not available: {e}")
            # Create mock services for demo
            self.audit_service = None
            self.mesh_audit = None
            self.performance_monitor = None
    
    def _initialize_data_mesh(self) -> Dict[DataDomain, List[DataProduct]]:
        """Initialize Data Mesh with domain data products"""
        mesh = {domain: [] for domain in DataDomain}
        
        logger.info("✅ Data Mesh initialized")
        return mesh
    
    # ========================================================================
    # COMPLETE ONBOARDING WORKFLOW
    # ========================================================================
    
    async def start_onboarding(
        self,
        customer_data: Dict[str, Any]
    ) -> CustomerOnboardingState:
        """
        Start complete onboarding workflow
        
        This orchestrates the entire end-to-end onboarding process through
        all domains, using Data Mesh for cross-domain data sharing
        """
        import uuid
        
        customer_id = f"CUST_{uuid.uuid4().hex[:8].upper()}"
        application_id = f"APP_{uuid.uuid4().hex[:8].upper()}"
        
        logger.info(f"🚀 Starting onboarding for customer {customer_id}")
        
        # Create workflow state
        state = CustomerOnboardingState(
            customer_id=customer_id,
            application_id=application_id,
            current_stage=WorkflowStage.APPLICATION_STARTED,
            started_at=datetime.utcnow()
        )
        
        self.active_workflows[customer_id] = state
        
        # Publish to Data Mesh - Application domain
        await self._publish_to_mesh(
            DataDomain.ACCOUNT,
            {
                "event": "application_started",
                "customer_id": customer_id,
                "application_id": application_id,
                "email": customer_data.get("email"),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Log audit event
        if self.audit_service:
            from modules.audit_reporting.audit_system import EventType, EventCategory
            await self.audit_service.log_event(
                event_type=EventType.APPLICATION_STARTED,
                category=EventCategory.ACCOUNT,
                customer_id=customer_id,
                details={"application_id": application_id}
            )
        
        # Record performance metric
        if self.performance_monitor:
            from modules.performance.performance_system import ServiceType, MetricType
            await self.performance_monitor.record_metric(
                ServiceType.API_GATEWAY,
                MetricType.LATENCY,
                350,  # 350ms
                {"operation": "start_application"}
            )
        
        state.completed_stages.append(WorkflowStage.APPLICATION_STARTED)
        
        logger.info(f"✅ Application started: {application_id}")
        return state
    
    async def verify_identity(
        self,
        customer_id: str,
        identity_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Verify customer identity
        
        Identity data flows through Data Mesh to:
        - Fraud detection (risk assessment)
        - KYC (identity validation)
        - Ongoing monitoring (profile baseline)
        """
        logger.info(f"🔍 Verifying identity for {customer_id}")
        
        state = self.active_workflows.get(customer_id)
        if not state:
            raise ValueError(f"No active workflow for customer {customer_id}")
        
        # Simulate identity verification
        start_time = datetime.utcnow()
        
        # Perform verification (simulated)
        await asyncio.sleep(0.1)  # Simulate processing
        
        verification_result = {
            "session_id": f"IDV_{customer_id}",
            "status": "VERIFIED",
            "confidence_score": 0.95,
            "checks": {
                "document_authentic": True,
                "face_match": True,
                "liveness_check": True
            },
            "verified_at": datetime.utcnow().isoformat()
        }
        
        # Publish to Data Mesh - Identity domain
        await self._publish_to_mesh(
            DataDomain.IDENTITY,
            {
                "event": "identity_verified",
                "customer_id": customer_id,
                "verification_result": verification_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Store in state
        state.data_products[DataDomain.IDENTITY] = verification_result
        state.current_stage = WorkflowStage.IDENTITY_VERIFICATION
        state.completed_stages.append(WorkflowStage.IDENTITY_VERIFICATION)
        
        # Log audit
        if self.audit_service:
            from modules.audit_reporting.audit_system import EventType, EventCategory
            await self.audit_service.log_event(
                event_type=EventType.IDENTITY_VERIFIED,
                category=EventCategory.IDENTITY,
                customer_id=customer_id,
                details=verification_result
            )
        
        # Record performance
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if self.performance_monitor:
            from modules.performance.performance_system import ServiceType, MetricType
            await self.performance_monitor.record_metric(
                ServiceType.IDENTITY_VERIFICATION,
                MetricType.LATENCY,
                duration_ms
            )
        
        logger.info(f"✅ Identity verified for {customer_id}")
        return verification_result
    
    async def perform_kyc_screening(
        self,
        customer_id: str,
        kyc_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform KYC screening
        
        KYC results flow through Data Mesh to:
        - Compliance (regulatory review)
        - Risk management (risk scoring)
        - Ongoing monitoring (watchlist baseline)
        """
        logger.info(f"📋 Performing KYC screening for {customer_id}")
        
        state = self.active_workflows.get(customer_id)
        if not state:
            raise ValueError(f"No active workflow for customer {customer_id}")
        
        start_time = datetime.utcnow()
        
        # Perform screening (simulated)
        await asyncio.sleep(0.05)
        
        screening_result = {
            "screening_id": f"KYC_{customer_id}",
            "status": "PASS",
            "risk_level": "LOW",
            "watchlist_hits": 0,
            "pep_match": False,
            "sanctions_match": False,
            "adverse_media": False,
            "screened_at": datetime.utcnow().isoformat()
        }
        
        # Publish to Data Mesh - KYC domain
        await self._publish_to_mesh(
            DataDomain.KYC,
            {
                "event": "kyc_completed",
                "customer_id": customer_id,
                "screening_result": screening_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Store in state
        state.data_products[DataDomain.KYC] = screening_result
        state.current_stage = WorkflowStage.KYC_SCREENING
        state.completed_stages.append(WorkflowStage.KYC_SCREENING)
        
        # Log audit
        if self.audit_service:
            from modules.audit_reporting.audit_system import EventType, EventCategory
            await self.audit_service.log_event(
                event_type=EventType.SCREENING_COMPLETED,
                category=EventCategory.KYC,
                customer_id=customer_id,
                details=screening_result
            )
        
        # Record performance
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if self.performance_monitor:
            from modules.performance.performance_system import ServiceType, MetricType
            await self.performance_monitor.record_metric(
                ServiceType.KYC_SCREENING,
                MetricType.LATENCY,
                duration_ms
            )
        
        logger.info(f"✅ KYC screening completed for {customer_id}")
        return screening_result
    
    async def perform_fraud_check(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Perform fraud detection check
        
        Fraud results flow through Data Mesh to:
        - Compliance (case management)
        - Account opening (approval decision)
        - Ongoing monitoring (behavior baseline)
        """
        logger.info(f"🛡️ Performing fraud check for {customer_id}")
        
        state = self.active_workflows.get(customer_id)
        if not state:
            raise ValueError(f"No active workflow for customer {customer_id}")
        
        # Get identity and KYC data from mesh
        identity_data = await self._consume_from_mesh(DataDomain.IDENTITY, customer_id)
        kyc_data = await self._consume_from_mesh(DataDomain.KYC, customer_id)
        
        # Perform fraud check (simulated)
        await asyncio.sleep(0.03)
        
        fraud_result = {
            "check_id": f"FRAUD_{customer_id}",
            "risk_score": 15,  # Low risk
            "risk_level": "LOW",
            "fraud_indicators": [],
            "synthetic_identity_score": 0.05,
            "velocity_check": "PASS",
            "device_fingerprint": "trusted",
            "checked_at": datetime.utcnow().isoformat()
        }
        
        # Publish to Data Mesh - Fraud domain
        await self._publish_to_mesh(
            DataDomain.FRAUD,
            {
                "event": "fraud_check_completed",
                "customer_id": customer_id,
                "fraud_result": fraud_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Store in state
        state.data_products[DataDomain.FRAUD] = fraud_result
        state.current_stage = WorkflowStage.FRAUD_CHECK
        state.completed_stages.append(WorkflowStage.FRAUD_CHECK)
        
        logger.info(f"✅ Fraud check completed for {customer_id}")
        return fraud_result
    
    async def setup_ongoing_monitoring(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Setup ongoing monitoring
        
        Monitoring subscribes to data from:
        - Identity (profile changes)
        - KYC (watchlist updates)
        - Fraud (behavior patterns)
        - Account (transaction patterns)
        """
        logger.info(f"👁️ Setting up ongoing monitoring for {customer_id}")
        
        state = self.active_workflows.get(customer_id)
        if not state:
            raise ValueError(f"No active workflow for customer {customer_id}")
        
        # Get baseline data from all domains
        identity_baseline = await self._consume_from_mesh(DataDomain.IDENTITY, customer_id)
        kyc_baseline = await self._consume_from_mesh(DataDomain.KYC, customer_id)
        fraud_baseline = await self._consume_from_mesh(DataDomain.FRAUD, customer_id)
        
        monitoring_config = {
            "monitoring_id": f"MON_{customer_id}",
            "status": "ACTIVE",
            "kyc_refresh_schedule": "annual",
            "watchlist_check_frequency": "daily",
            "transaction_monitoring": "enabled",
            "behavior_analysis": "enabled",
            "baselines": {
                "identity": identity_baseline,
                "kyc": kyc_baseline,
                "fraud": fraud_baseline
            },
            "setup_at": datetime.utcnow().isoformat()
        }
        
        # Publish to Data Mesh - Monitoring domain
        await self._publish_to_mesh(
            DataDomain.MONITORING,
            {
                "event": "monitoring_setup_completed",
                "customer_id": customer_id,
                "monitoring_config": monitoring_config,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Store in state
        state.data_products[DataDomain.MONITORING] = monitoring_config
        state.current_stage = WorkflowStage.ONGOING_MONITORING_SETUP
        state.completed_stages.append(WorkflowStage.ONGOING_MONITORING_SETUP)
        
        logger.info(f"✅ Ongoing monitoring setup for {customer_id}")
        return monitoring_config
    
    async def open_account(
        self,
        customer_id: str,
        account_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Open customer account
        
        Account opening requires data from:
        - Identity (verified identity)
        - KYC (compliance approval)
        - Fraud (risk assessment)
        - Compliance (final approval)
        """
        logger.info(f"🏦 Opening account for {customer_id}")
        
        state = self.active_workflows.get(customer_id)
        if not state:
            raise ValueError(f"No active workflow for customer {customer_id}")
        
        start_time = datetime.utcnow()
        
        # Verify all requirements are met
        required_domains = [DataDomain.IDENTITY, DataDomain.KYC, DataDomain.FRAUD]
        for domain in required_domains:
            if domain not in state.data_products:
                raise ValueError(f"Missing required data from {domain.value} domain")
        
        # Open account (simulated)
        await asyncio.sleep(0.05)
        
        import uuid
        account_result = {
            "account_id": f"ACC_{uuid.uuid4().hex[:12].upper()}",
            "account_number": f"{uuid.uuid4().hex[:12].upper()}",
            "account_type": account_data.get("account_type", "STANDARD"),
            "status": "ACTIVE",
            "opened_at": datetime.utcnow().isoformat(),
            "initial_balance": 0.00
        }
        
        # Publish to Data Mesh - Account domain
        await self._publish_to_mesh(
            DataDomain.ACCOUNT,
            {
                "event": "account_opened",
                "customer_id": customer_id,
                "account_result": account_result,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Store in state
        state.data_products[DataDomain.ACCOUNT] = account_result
        state.current_stage = WorkflowStage.ACCOUNT_ACTIVATED
        state.completed_stages.append(WorkflowStage.ACCOUNT_OPENING)
        state.completed_stages.append(WorkflowStage.ACCOUNT_ACTIVATED)
        
        # Log audit
        if self.audit_service:
            from modules.audit_reporting.audit_system import EventType, EventCategory
            await self.audit_service.log_event(
                event_type=EventType.ACCOUNT_CREATED,
                category=EventCategory.ACCOUNT,
                customer_id=customer_id,
                details=account_result
            )
        
        # Record performance
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        if self.performance_monitor:
            from modules.performance.performance_system import ServiceType, MetricType
            await self.performance_monitor.record_metric(
                ServiceType.ACCOUNT_OPENING,
                MetricType.LATENCY,
                duration_ms
            )
        
        logger.info(f"✅ Account opened for {customer_id}: {account_result['account_id']}")
        return account_result
    
    # ========================================================================
    # DATA MESH OPERATIONS
    # ========================================================================
    
    async def _publish_to_mesh(
        self,
        domain: DataDomain,
        data: Dict[str, Any]
    ):
        """Publish data product to mesh"""
        product = DataProduct(
            domain=domain,
            product_id=f"{domain.value}_{data.get('customer_id', 'unknown')}",
            data=data,
            version="1.0",
            created_at=datetime.utcnow(),
            quality_score=1.0
        )
        
        self.data_mesh[domain].append(product)
        
        # Also publish to audit mesh if available
        if self.mesh_audit:
            from modules.audit_reporting.data_mesh_integration import AuditDataDomain
            
            # Map to audit domain
            audit_domain_map = {
                DataDomain.IDENTITY: AuditDataDomain.IDENTITY_VERIFICATION,
                DataDomain.KYC: AuditDataDomain.KYC_COMPLIANCE,
                DataDomain.ACCOUNT: AuditDataDomain.ACCOUNT_LIFECYCLE,
                DataDomain.FRAUD: AuditDataDomain.FRAUD_DETECTION
            }
            
            if domain in audit_domain_map:
                await self.mesh_audit.publish_to_domain_stream(
                    audit_domain_map[domain],
                    data
                )
        
        logger.debug(f"📤 Published to {domain.value} domain: {data.get('event')}")
    
    async def _consume_from_mesh(
        self,
        domain: DataDomain,
        customer_id: str
    ) -> Optional[Dict[str, Any]]:
        """Consume data product from mesh"""
        products = self.data_mesh.get(domain, [])
        
        # Find latest product for customer
        customer_products = [
            p for p in products
            if p.data.get("customer_id") == customer_id
        ]
        
        if customer_products:
            latest = max(customer_products, key=lambda p: p.created_at)
            logger.debug(f"📥 Consumed from {domain.value} domain for {customer_id}")
            return latest.data
        
        return None
    
    async def get_customer_360_view(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Get complete 360° customer view using federated data from all domains
        
        This demonstrates the power of Data Mesh - we can aggregate
        data from all domains to create a complete customer view
        """
        logger.info(f"📊 Building 360° view for {customer_id}")
        
        view = {
            "customer_id": customer_id,
            "generated_at": datetime.utcnow().isoformat(),
            "domains": {}
        }
        
        # Collect data from all domains
        for domain in DataDomain:
            data = await self._consume_from_mesh(domain, customer_id)
            if data:
                view["domains"][domain.value] = data
        
        # Add workflow state
        if customer_id in self.active_workflows:
            state = self.active_workflows[customer_id]
            view["workflow"] = {
                "current_stage": state.current_stage.value,
                "completion": f"{state.completion_percentage:.1f}%",
                "completed_stages": [s.value for s in state.completed_stages]
            }
        
        logger.info(f"✅ 360° view created with {len(view['domains'])} domains")
        return view
    
    # ========================================================================
    # PLATFORM OPERATIONS
    # ========================================================================
    
    async def get_platform_health(self) -> Dict[str, Any]:
        """Get overall platform health"""
        health = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check each component
        if self.audit_service:
            audit_stats = self.audit_service.get_audit_statistics()
            health["components"]["audit"] = {
                "status": "healthy",
                "total_events": audit_stats["total_events"]
            }
        
        if self.performance_monitor:
            from modules.performance.performance_system import ServiceType
            api_status = self.performance_monitor.get_performance_status(ServiceType.API_GATEWAY)
            health["components"]["performance"] = {
                "status": api_status.value,
                "monitoring": "active"
            }
        
        if self.cache_manager:
            cache_stats = self.cache_manager.get_cache_stats()
            health["components"]["cache"] = {
                "status": "healthy",
                "hit_rate": cache_stats["hit_rate"]
            }
        
        # Data mesh health
        mesh_products = sum(len(products) for products in self.data_mesh.values())
        health["components"]["data_mesh"] = {
            "status": "healthy",
            "total_products": mesh_products,
            "domains_active": len(self.data_mesh)
        }
        
        # Active workflows
        health["components"]["workflows"] = {
            "status": "healthy",
            "active_count": len(self.active_workflows)
        }
        
        return health
    
    def get_platform_metrics(self) -> Dict[str, Any]:
        """Get platform-wide metrics"""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "workflows": {
                "active": len(self.active_workflows),
                "total_processed": len(self.active_workflows)
            },
            "data_mesh": {
                domain.value: len(products)
                for domain, products in self.data_mesh.items()
            }
        }
        
        if self.performance_monitor:
            benchmarks = self.performance_monitor.get_benchmark_status()
            metrics["performance_benchmarks"] = benchmarks
        
        if self.cache_manager:
            metrics["cache"] = self.cache_manager.get_cache_stats()
        
        if self.audit_service:
            metrics["audit"] = self.audit_service.get_audit_statistics()
        
        return metrics


# ============================================================================
# DEMO - COMPLETE END-TO-END WORKFLOW
# ============================================================================

async def demo_complete_onboarding():
    """
    Demonstrate complete end-to-end onboarding workflow
    with full Data Mesh integration
    """
    print("\n" + "=" * 70)
    print("ULTRA PLATFORM - COMPLETE INTEGRATION DEMO")
    print("Onboarding with Data Mesh Architecture")
    print("=" * 70)
    
    # Initialize platform
    print("\n🚀 Initializing Ultra Platform...")
    platform = UltraPlatform()
    
    # Start onboarding
    print("\n" + "=" * 70)
    print("STAGE 1: APPLICATION START")
    print("=" * 70)
    
    customer_data = {
        "email": "john.doe@example.com",
        "phone": "+61-4-1234-5678",
        "first_name": "John",
        "last_name": "Doe"
    }
    
    state = await platform.start_onboarding(customer_data)
    print(f"✅ Application started: {state.application_id}")
    print(f"   Customer ID: {state.customer_id}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    
    # Identity verification
    print("\n" + "=" * 70)
    print("STAGE 2: IDENTITY VERIFICATION")
    print("=" * 70)
    
    identity_data = {
        "document_type": "PASSPORT",
        "document_number": "N1234567",
        "document_front": "base64_image_data",
        "selfie": "base64_selfie_data"
    }
    
    identity_result = await platform.verify_identity(state.customer_id, identity_data)
    print(f"✅ Identity verified")
    print(f"   Confidence: {identity_result['confidence_score']}")
    print(f"   Status: {identity_result['status']}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    print(f"   📤 Data published to IDENTITY domain in mesh")
    
    # KYC screening
    print("\n" + "=" * 70)
    print("STAGE 3: KYC SCREENING")
    print("=" * 70)
    
    kyc_data = {
        "first_name": "John",
        "last_name": "Doe",
        "date_of_birth": "1985-06-15",
        "nationality": "AU"
    }
    
    kyc_result = await platform.perform_kyc_screening(state.customer_id, kyc_data)
    print(f"✅ KYC screening completed")
    print(f"   Risk Level: {kyc_result['risk_level']}")
    print(f"   Status: {kyc_result['status']}")
    print(f"   PEP Match: {kyc_result['pep_match']}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    print(f"   📤 Data published to KYC domain in mesh")
    
    # Fraud check
    print("\n" + "=" * 70)
    print("STAGE 4: FRAUD DETECTION")
    print("=" * 70)
    print("   📥 Consuming IDENTITY data from mesh...")
    print("   📥 Consuming KYC data from mesh...")
    
    fraud_result = await platform.perform_fraud_check(state.customer_id)
    print(f"✅ Fraud check completed")
    print(f"   Risk Score: {fraud_result['risk_score']}/100")
    print(f"   Risk Level: {fraud_result['risk_level']}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    print(f"   📤 Data published to FRAUD domain in mesh")
    
    # Setup monitoring
    print("\n" + "=" * 70)
    print("STAGE 5: ONGOING MONITORING SETUP")
    print("=" * 70)
    print("   📥 Consuming baseline data from all domains...")
    
    monitoring_config = await platform.setup_ongoing_monitoring(state.customer_id)
    print(f"✅ Ongoing monitoring configured")
    print(f"   Status: {monitoring_config['status']}")
    print(f"   KYC Refresh: {monitoring_config['kyc_refresh_schedule']}")
    print(f"   Watchlist Check: {monitoring_config['watchlist_check_frequency']}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    print(f"   📤 Data published to MONITORING domain in mesh")
    
    # Open account
    print("\n" + "=" * 70)
    print("STAGE 6: ACCOUNT OPENING")
    print("=" * 70)
    print("   📥 Verifying all requirements from mesh...")
    print("   ✓ Identity verified")
    print("   ✓ KYC passed")
    print("   ✓ Fraud check passed")
    
    account_data = {
        "account_type": "STANDARD"
    }
    
    account_result = await platform.open_account(state.customer_id, account_data)
    print(f"✅ Account opened successfully!")
    print(f"   Account ID: {account_result['account_id']}")
    print(f"   Account Number: {account_result['account_number']}")
    print(f"   Status: {account_result['status']}")
    print(f"   Progress: {state.completion_percentage:.1f}%")
    print(f"   📤 Data published to ACCOUNT domain in mesh")
    
    # Get 360° view
    print("\n" + "=" * 70)
    print("CUSTOMER 360° VIEW (Data Mesh Aggregation)")
    print("=" * 70)
    
    view_360 = await platform.get_customer_360_view(state.customer_id)
    print(f"📊 Complete customer view from {len(view_360['domains'])} domains:")
    
    for domain, data in view_360["domains"].items():
        print(f"\n   {domain.upper()} Domain:")
        event = data.get("event", "N/A")
        print(f"     Event: {event}")
        
        if domain == "identity":
            print(f"     Status: {data.get('verification_result', {}).get('status')}")
        elif domain == "kyc":
            print(f"     Risk: {data.get('screening_result', {}).get('risk_level')}")
        elif domain == "fraud":
            print(f"     Score: {data.get('fraud_result', {}).get('risk_score')}")
        elif domain == "account":
            print(f"     Account: {data.get('account_result', {}).get('account_id')}")
    
    # Platform health
    print("\n" + "=" * 70)
    print("PLATFORM HEALTH CHECK")
    print("=" * 70)
    
    health = await platform.get_platform_health()
    print(f"Overall Status: {health['status'].upper()}")
    print(f"\nComponents:")
    for component, status in health["components"].items():
        print(f"  • {component}: {status['status']}")
    
    # Platform metrics
    print("\n" + "=" * 70)
    print("PLATFORM METRICS")
    print("=" * 70)
    
    metrics = platform.get_platform_metrics()
    print(f"Active Workflows: {metrics['workflows']['active']}")
    print(f"\nData Mesh Products by Domain:")
    for domain, count in metrics['data_mesh'].items():
        if count > 0:
            print(f"  • {domain}: {count} products")
    
    print("\n" + "=" * 70)
    print("✅ COMPLETE ONBOARDING WORKFLOW FINISHED!")
    print(f"⏱️  Total Time: {(datetime.utcnow() - state.started_at).total_seconds():.2f}s")
    print(f"📊 Completion: {state.completion_percentage:.0f}%")
    print(f"🔗 Domains Integrated: {len(view_360['domains'])}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_complete_onboarding())
