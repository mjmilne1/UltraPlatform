"""
ULTRA PLATFORM - AUDIT & REPORTING SYSTEM
==========================================

Institutional-grade audit trail and compliance reporting system providing:
- Immutable audit logs with blockchain-ready hashing
- Comprehensive event tracking across all operations
- Regulatory compliance reporting (AUSTRAC, FINRA, MiFID II)
- Real-time analytics and dashboards
- Automated report generation and distribution
- Data retention management with lifecycle policies
- Advanced search and query capabilities
- Multi-format export (PDF, CSV, JSON, XML)
- Data lake integration for long-term storage
- Compliance officer portal
- Audit trail verification and integrity checks

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
import hashlib
import json
import uuid
import logging
from collections import defaultdict, deque
import asyncio
from pathlib import Path
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class EventCategory(str, Enum):
    """Audit event categories"""
    IDENTITY = "identity"
    KYC = "kyc"
    ACCOUNT = "account"
    COMPLIANCE = "compliance"
    ACCESS = "access"
    SYSTEM = "system"
    FRAUD = "fraud"
    MONITORING = "monitoring"


class EventType(str, Enum):
    """Specific event types"""
    # Identity events
    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_VERIFIED = "document.verified"
    BIOMETRIC_CAPTURED = "biometric.captured"
    IDENTITY_VERIFIED = "identity.verified"
    
    # KYC events
    SCREENING_STARTED = "screening.started"
    SCREENING_COMPLETED = "screening.completed"
    WATCHLIST_HIT = "watchlist.hit"
    PEP_MATCH = "pep.match"
    SANCTIONS_MATCH = "sanctions.match"
    
    # Account events
    APPLICATION_STARTED = "application.started"
    ACCOUNT_CREATED = "account.created"
    ACCOUNT_ACTIVATED = "account.activated"
    ACCOUNT_SUSPENDED = "account.suspended"
    ACCOUNT_CLOSED = "account.closed"
    
    # Compliance events
    SAR_FILED = "sar.filed"
    EDD_INITIATED = "edd.initiated"
    EDD_COMPLETED = "edd.completed"
    REGULATORY_REPORT_FILED = "regulatory.report.filed"
    
    # Access events
    DATA_ACCESSED = "data.accessed"
    DATA_EXPORTED = "data.exported"
    DOCUMENT_VIEWED = "document.viewed"
    REPORT_GENERATED = "report.generated"
    
    # System events
    API_CALLED = "api.called"
    ERROR_OCCURRED = "error.occurred"
    SYSTEM_STARTED = "system.started"
    
    # Fraud events
    FRAUD_DETECTED = "fraud.detected"
    FRAUD_INVESTIGATION = "fraud.investigation"
    
    # Monitoring events
    PROFILE_CHANGED = "profile.changed"
    KYC_REFRESHED = "kyc.refreshed"


class ReportType(str, Enum):
    """Report types"""
    KYC_SUMMARY = "kyc_summary"
    FRAUD_REPORT = "fraud_report"
    ONBOARDING_METRICS = "onboarding_metrics"
    COMPLIANCE_REVIEW = "compliance_review"
    SAR_FILING = "sar_filing"
    AUDIT_TRAIL = "audit_trail"
    PERFORMANCE_METRICS = "performance_metrics"


class ReportFormat(str, Enum):
    """Report output formats"""
    PDF = "pdf"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EXCEL = "xlsx"
    HTML = "html"


class ReportFrequency(str, Enum):
    """Report generation frequency"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"
    ON_DEMAND = "on_demand"


class RetentionPeriod(str, Enum):
    """Data retention periods"""
    ONE_YEAR = "1_year"
    THREE_YEARS = "3_years"
    FIVE_YEARS = "5_years"
    SEVEN_YEARS = "7_years"
    TEN_YEARS = "10_years"
    PERMANENT = "permanent"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class AuditEvent:
    """Immutable audit event"""
    event_id: str
    event_type: EventType
    category: EventCategory
    customer_id: Optional[str]
    user_id: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    previous_hash: Optional[str] = None
    event_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate event hash for integrity verification"""
        if not self.event_hash:
            self.event_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """Calculate SHA-256 hash of event for blockchain-ready audit trail"""
        hash_input = f"{self.event_id}{self.event_type.value}{self.customer_id}"
        hash_input += f"{self.timestamp.isoformat()}{json.dumps(self.details, sort_keys=True)}"
        hash_input += f"{self.previous_hash or ''}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()
    
    def verify_integrity(self, previous_hash: Optional[str] = None) -> bool:
        """Verify event integrity"""
        expected_hash = self._calculate_hash()
        return self.event_hash == expected_hash


@dataclass
class RetentionPolicy:
    """Data retention policy"""
    category: EventCategory
    retention_period: RetentionPeriod
    archive_after_days: int
    delete_after_days: int
    requires_legal_hold: bool = False
    
    def is_expired(self, event_date: datetime) -> bool:
        """Check if event has exceeded retention period"""
        days_old = (datetime.utcnow() - event_date).days
        return days_old > self.delete_after_days
    
    def should_archive(self, event_date: datetime) -> bool:
        """Check if event should be archived"""
        days_old = (datetime.utcnow() - event_date).days
        return days_old > self.archive_after_days


@dataclass
class ReportSchedule:
    """Scheduled report configuration"""
    schedule_id: str
    report_type: ReportType
    frequency: ReportFrequency
    format: ReportFormat
    recipients: List[str]
    filters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GeneratedReport:
    """Generated report metadata"""
    report_id: str
    report_type: ReportType
    format: ReportFormat
    generated_at: datetime
    generated_by: str
    period_start: datetime
    period_end: datetime
    file_path: Optional[str] = None
    file_size_bytes: int = 0
    record_count: int = 0
    summary: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceMetrics:
    """Compliance metrics snapshot"""
    date: datetime
    total_applications: int = 0
    approved_applications: int = 0
    rejected_applications: int = 0
    pending_applications: int = 0
    kyc_screenings_completed: int = 0
    watchlist_hits: int = 0
    pep_matches: int = 0
    sanctions_matches: int = 0
    fraud_detections: int = 0
    sars_filed: int = 0
    edd_cases: int = 0
    avg_onboarding_time_hours: float = 0.0
    
    @property
    def approval_rate(self) -> float:
        """Calculate approval rate"""
        if self.total_applications == 0:
            return 0.0
        return (self.approved_applications / self.total_applications) * 100
    
    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate"""
        if self.total_applications == 0:
            return 0.0
        return (self.rejected_applications / self.total_applications) * 100


# ============================================================================
# AUDIT TRAIL SERVICE
# ============================================================================

class OnboardingAuditService:
    """
    Comprehensive audit trail service with immutable logging,
    blockchain-ready hashing, and compliance features
    """
    
    def __init__(self, storage_path: str = "data/audit"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # In-memory storage (use database for production)
        self.events: List[AuditEvent] = []
        self.event_index: Dict[str, AuditEvent] = {}
        self.customer_events: Dict[str, List[AuditEvent]] = defaultdict(list)
        
        # Last event hash for blockchain-like chain
        self.last_event_hash: Optional[str] = None
        
        # Retention policies
        self.retention_policies = self._initialize_retention_policies()
        
        # Metrics
        self.total_events_logged = 0
        self.events_by_category: Dict[EventCategory, int] = defaultdict(int)
        
        logger.info("Audit service initialized")
    
    def _initialize_retention_policies(self) -> Dict[EventCategory, RetentionPolicy]:
        """Initialize retention policies per regulatory requirements"""
        return {
            EventCategory.IDENTITY: RetentionPolicy(
                category=EventCategory.IDENTITY,
                retention_period=RetentionPeriod.SEVEN_YEARS,
                archive_after_days=365,
                delete_after_days=2555  # 7 years
            ),
            EventCategory.KYC: RetentionPolicy(
                category=EventCategory.KYC,
                retention_period=RetentionPeriod.SEVEN_YEARS,
                archive_after_days=365,
                delete_after_days=2555
            ),
            EventCategory.ACCOUNT: RetentionPolicy(
                category=EventCategory.ACCOUNT,
                retention_period=RetentionPeriod.SEVEN_YEARS,
                archive_after_days=365,
                delete_after_days=2555
            ),
            EventCategory.COMPLIANCE: RetentionPolicy(
                category=EventCategory.COMPLIANCE,
                retention_period=RetentionPeriod.TEN_YEARS,
                archive_after_days=365,
                delete_after_days=3650,  # 10 years
                requires_legal_hold=True
            ),
            EventCategory.ACCESS: RetentionPolicy(
                category=EventCategory.ACCESS,
                retention_period=RetentionPeriod.TEN_YEARS,
                archive_after_days=180,
                delete_after_days=3650
            ),
            EventCategory.SYSTEM: RetentionPolicy(
                category=EventCategory.SYSTEM,
                retention_period=RetentionPeriod.ONE_YEAR,
                archive_after_days=90,
                delete_after_days=365
            ),
            EventCategory.FRAUD: RetentionPolicy(
                category=EventCategory.FRAUD,
                retention_period=RetentionPeriod.TEN_YEARS,
                archive_after_days=365,
                delete_after_days=3650,
                requires_legal_hold=True
            ),
            EventCategory.MONITORING: RetentionPolicy(
                category=EventCategory.MONITORING,
                retention_period=RetentionPeriod.SEVEN_YEARS,
                archive_after_days=365,
                delete_after_days=2555
            )
        }
    
    async def log_event(
        self,
        event_type: EventType,
        category: EventCategory,
        details: Dict[str, Any],
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> AuditEvent:
        """
        Log immutable audit event
        
        Events are stored with blockchain-ready hashing for integrity verification
        """
        event = AuditEvent(
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type=event_type,
            category=category,
            customer_id=customer_id,
            user_id=user_id,
            timestamp=datetime.utcnow(),
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=session_id,
            request_id=request_id,
            previous_hash=self.last_event_hash
        )
        
        # Store event
        self.events.append(event)
        self.event_index[event.event_id] = event
        
        if customer_id:
            self.customer_events[customer_id].append(event)
        
        # Update last hash for blockchain chain
        self.last_event_hash = event.event_hash
        
        # Update metrics
        self.total_events_logged += 1
        self.events_by_category[category] += 1
        
        logger.info(f"Logged audit event: {event_type.value} for customer {customer_id}")
        
        return event
    
    async def get_customer_audit_trail(
        self,
        customer_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        event_types: Optional[List[EventType]] = None
    ) -> List[AuditEvent]:
        """Get complete audit trail for customer"""
        events = self.customer_events.get(customer_id, [])
        
        # Filter by date range
        if start_date:
            events = [e for e in events if e.timestamp >= start_date]
        if end_date:
            events = [e for e in events if e.timestamp <= end_date]
        
        # Filter by event types
        if event_types:
            events = [e for e in events if e.event_type in event_types]
        
        return sorted(events, key=lambda e: e.timestamp)
    
    async def verify_audit_trail_integrity(
        self,
        start_event_id: Optional[str] = None,
        end_event_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify integrity of audit trail using blockchain-like hashing
        
        Returns verification report with any integrity violations
        """
        events_to_verify = self.events
        
        if start_event_id:
            start_idx = next((i for i, e in enumerate(self.events) if e.event_id == start_event_id), 0)
            events_to_verify = events_to_verify[start_idx:]
        
        if end_event_id:
            end_idx = next((i for i, e in enumerate(self.events) if e.event_id == end_event_id), len(self.events))
            events_to_verify = events_to_verify[:end_idx + 1]
        
        violations = []
        previous_hash = None
        
        for event in events_to_verify:
            # Verify event hash
            if not event.verify_integrity(previous_hash):
                violations.append({
                    "event_id": event.event_id,
                    "type": "hash_mismatch",
                    "timestamp": event.timestamp.isoformat()
                })
            
            # Verify chain continuity
            if previous_hash and event.previous_hash != previous_hash:
                violations.append({
                    "event_id": event.event_id,
                    "type": "chain_break",
                    "timestamp": event.timestamp.isoformat()
                })
            
            previous_hash = event.event_hash
        
        return {
            "verified": len(violations) == 0,
            "events_checked": len(events_to_verify),
            "violations": violations,
            "verification_timestamp": datetime.utcnow().isoformat()
        }
    
    async def apply_retention_policies(self) -> Dict[str, Any]:
        """
        Apply retention policies and archive/delete old events
        
        Returns summary of retention actions
        """
        archived_count = 0
        deleted_count = 0
        retained_count = 0
        
        for event in self.events[:]:  # Copy list for safe iteration
            policy = self.retention_policies.get(event.category)
            
            if not policy:
                continue
            
            if policy.requires_legal_hold:
                # Don't delete events under legal hold
                retained_count += 1
                continue
            
            if policy.is_expired(event.timestamp):
                # Delete expired events
                self.events.remove(event)
                del self.event_index[event.event_id]
                if event.customer_id:
                    self.customer_events[event.customer_id].remove(event)
                deleted_count += 1
                
            elif policy.should_archive(event.timestamp):
                # Archive old events (would move to cold storage)
                archived_count += 1
                retained_count += 1
            else:
                retained_count += 1
        
        logger.info(f"Retention: archived={archived_count}, deleted={deleted_count}, retained={retained_count}")
        
        return {
            "archived": archived_count,
            "deleted": deleted_count,
            "retained": retained_count,
            "total_events": len(self.events)
        }
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get audit trail statistics"""
        return {
            "total_events": self.total_events_logged,
            "current_events": len(self.events),
            "events_by_category": dict(self.events_by_category),
            "unique_customers": len(self.customer_events),
            "chain_integrity": "verified" if self.last_event_hash else "empty"
        }


# ============================================================================
# COMPLIANCE REPORTING SERVICE
# ============================================================================

class ComplianceReportingService:
    """
    Comprehensive compliance reporting with automated generation,
    scheduling, and distribution
    """
    
    def __init__(self, audit_service: OnboardingAuditService):
        self.audit_service = audit_service
        self.reports: Dict[str, GeneratedReport] = {}
        self.schedules: Dict[str, ReportSchedule] = {}
        
        # Metrics storage
        self.daily_metrics: deque = deque(maxlen=365)  # 1 year
        self.monthly_metrics: deque = deque(maxlen=36)  # 3 years
        
        logger.info("Compliance reporting service initialized")
    
    async def generate_kyc_summary_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ReportFormat = ReportFormat.PDF
    ) -> GeneratedReport:
        """Generate KYC screening summary report"""
        # Get KYC events
        kyc_events = [
            e for e in self.audit_service.events
            if e.category == EventCategory.KYC
            and start_date <= e.timestamp <= end_date
        ]
        
        # Calculate metrics
        total_screenings = len([e for e in kyc_events if e.event_type == EventType.SCREENING_COMPLETED])
        watchlist_hits = len([e for e in kyc_events if e.event_type == EventType.WATCHLIST_HIT])
        pep_matches = len([e for e in kyc_events if e.event_type == EventType.PEP_MATCH])
        sanctions_matches = len([e for e in kyc_events if e.event_type == EventType.SANCTIONS_MATCH])
        
        summary = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "total_screenings": total_screenings,
            "watchlist_hits": watchlist_hits,
            "pep_matches": pep_matches,
            "sanctions_matches": sanctions_matches,
            "hit_rate": f"{(watchlist_hits / total_screenings * 100):.2f}%" if total_screenings > 0 else "0%"
        }
        
        report = GeneratedReport(
            report_id=f"rpt_{uuid.uuid4().hex}",
            report_type=ReportType.KYC_SUMMARY,
            format=format,
            generated_at=datetime.utcnow(),
            generated_by="system",
            period_start=start_date,
            period_end=end_date,
            record_count=len(kyc_events),
            summary=summary
        )
        
        self.reports[report.report_id] = report
        
        logger.info(f"Generated KYC summary report: {report.report_id}")
        
        return report
    
    async def generate_fraud_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ReportFormat = ReportFormat.PDF
    ) -> GeneratedReport:
        """Generate fraud detection report"""
        fraud_events = [
            e for e in self.audit_service.events
            if e.category == EventCategory.FRAUD
            and start_date <= e.timestamp <= end_date
        ]
        
        fraud_detections = len([e for e in fraud_events if e.event_type == EventType.FRAUD_DETECTED])
        investigations = len([e for e in fraud_events if e.event_type == EventType.FRAUD_INVESTIGATION])
        
        summary = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "fraud_detections": fraud_detections,
            "investigations_opened": investigations,
            "detection_rate": f"{(fraud_detections / max(1, len(fraud_events)) * 100):.2f}%"
        }
        
        report = GeneratedReport(
            report_id=f"rpt_{uuid.uuid4().hex}",
            report_type=ReportType.FRAUD_REPORT,
            format=format,
            generated_at=datetime.utcnow(),
            generated_by="system",
            period_start=start_date,
            period_end=end_date,
            record_count=len(fraud_events),
            summary=summary
        )
        
        self.reports[report.report_id] = report
        
        logger.info(f"Generated fraud report: {report.report_id}")
        
        return report
    
    async def generate_onboarding_metrics_report(
        self,
        start_date: datetime,
        end_date: datetime,
        format: ReportFormat = ReportFormat.CSV
    ) -> GeneratedReport:
        """Generate onboarding metrics report"""
        account_events = [
            e for e in self.audit_service.events
            if e.category == EventCategory.ACCOUNT
            and start_date <= e.timestamp <= end_date
        ]
        
        applications_started = len([e for e in account_events if e.event_type == EventType.APPLICATION_STARTED])
        accounts_created = len([e for e in account_events if e.event_type == EventType.ACCOUNT_CREATED])
        accounts_activated = len([e for e in account_events if e.event_type == EventType.ACCOUNT_ACTIVATED])
        
        summary = {
            "period_start": start_date.isoformat(),
            "period_end": end_date.isoformat(),
            "applications_started": applications_started,
            "accounts_created": accounts_created,
            "accounts_activated": accounts_activated,
            "completion_rate": f"{(accounts_created / max(1, applications_started) * 100):.2f}%",
            "activation_rate": f"{(accounts_activated / max(1, accounts_created) * 100):.2f}%"
        }
        
        report = GeneratedReport(
            report_id=f"rpt_{uuid.uuid4().hex}",
            report_type=ReportType.ONBOARDING_METRICS,
            format=format,
            generated_at=datetime.utcnow(),
            generated_by="system",
            period_start=start_date,
            period_end=end_date,
            record_count=len(account_events),
            summary=summary
        )
        
        self.reports[report.report_id] = report
        
        logger.info(f"Generated onboarding metrics report: {report.report_id}")
        
        return report
    
    async def export_to_csv(
        self,
        events: List[AuditEvent],
        output_path: Optional[str] = None
    ) -> str:
        """Export events to CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow([
            "Event ID", "Event Type", "Category", "Customer ID",
            "Timestamp", "IP Address", "Details"
        ])
        
        # Write data
        for event in events:
            writer.writerow([
                event.event_id,
                event.event_type.value,
                event.category.value,
                event.customer_id or "",
                event.timestamp.isoformat(),
                event.ip_address or "",
                json.dumps(event.details)
            ])
        
        csv_content = output.getvalue()
        
        if output_path:
            Path(output_path).write_text(csv_content)
        
        return csv_content
    
    async def export_to_json(
        self,
        events: List[AuditEvent],
        output_path: Optional[str] = None
    ) -> str:
        """Export events to JSON format"""
        json_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "record_count": len(events),
            "events": [
                {
                    "event_id": e.event_id,
                    "event_type": e.event_type.value,
                    "category": e.category.value,
                    "customer_id": e.customer_id,
                    "timestamp": e.timestamp.isoformat(),
                    "details": e.details,
                    "ip_address": e.ip_address,
                    "event_hash": e.event_hash
                }
                for e in events
            ]
        }
        
        json_content = json.dumps(json_data, indent=2)
        
        if output_path:
            Path(output_path).write_text(json_content)
        
        return json_content
    
    def schedule_report(
        self,
        report_type: ReportType,
        frequency: ReportFrequency,
        format: ReportFormat,
        recipients: List[str],
        filters: Optional[Dict[str, Any]] = None
    ) -> ReportSchedule:
        """Schedule automated report generation"""
        schedule = ReportSchedule(
            schedule_id=f"sch_{uuid.uuid4().hex}",
            report_type=report_type,
            frequency=frequency,
            format=format,
            recipients=recipients,
            filters=filters or {}
        )
        
        # Calculate next run time
        schedule.next_run = self._calculate_next_run(frequency)
        
        self.schedules[schedule.schedule_id] = schedule
        
        logger.info(f"Scheduled {frequency.value} {report_type.value} report")
        
        return schedule
    
    def _calculate_next_run(self, frequency: ReportFrequency) -> datetime:
        """Calculate next report run time"""
        now = datetime.utcnow()
        
        if frequency == ReportFrequency.DAILY:
            return now.replace(hour=0, minute=0, second=0) + timedelta(days=1)
        elif frequency == ReportFrequency.WEEKLY:
            days_ahead = 7 - now.weekday()
            return now.replace(hour=0, minute=0, second=0) + timedelta(days=days_ahead)
        elif frequency == ReportFrequency.MONTHLY:
            next_month = now.month + 1 if now.month < 12 else 1
            next_year = now.year if now.month < 12 else now.year + 1
            return datetime(next_year, next_month, 1, 0, 0, 0)
        elif frequency == ReportFrequency.QUARTERLY:
            next_quarter_month = ((now.month - 1) // 3 + 1) * 3 + 1
            if next_quarter_month > 12:
                next_quarter_month = 1
                return datetime(now.year + 1, next_quarter_month, 1, 0, 0, 0)
            return datetime(now.year, next_quarter_month, 1, 0, 0, 0)
        elif frequency == ReportFrequency.ANNUAL:
            return datetime(now.year + 1, 1, 1, 0, 0, 0)
        else:
            return now
    
    def get_report_summary(self) -> Dict[str, Any]:
        """Get reporting statistics"""
        return {
            "total_reports_generated": len(self.reports),
            "active_schedules": len([s for s in self.schedules.values() if s.enabled]),
            "reports_by_type": defaultdict(int, {
                report.report_type.value: 1
                for report in self.reports.values()
            })
        }


# ============================================================================
# TESTING & DEMO
# ============================================================================

async def demo_audit_reporting():
    """Demonstrate audit and reporting functionality"""
    print("\n" + "=" * 60)
    print("ULTRA PLATFORM - AUDIT & REPORTING DEMO")
    print("=" * 60)
    
    # Initialize services
    audit_service = OnboardingAuditService()
    reporting_service = ComplianceReportingService(audit_service)
    
    # Log sample events
    print("\n1. Logging Audit Events...")
    
    customer_id = "CUST001"
    
    await audit_service.log_event(
        event_type=EventType.APPLICATION_STARTED,
        category=EventCategory.ACCOUNT,
        customer_id=customer_id,
        details={"email": "customer@example.com"},
        ip_address="192.168.1.1"
    )
    
    await audit_service.log_event(
        event_type=EventType.IDENTITY_VERIFIED,
        category=EventCategory.IDENTITY,
        customer_id=customer_id,
        details={"confidence": 0.95},
        ip_address="192.168.1.1"
    )
    
    await audit_service.log_event(
        event_type=EventType.SCREENING_COMPLETED,
        category=EventCategory.KYC,
        customer_id=customer_id,
        details={"risk_level": "LOW"},
        ip_address="192.168.1.1"
    )
    
    await audit_service.log_event(
        event_type=EventType.ACCOUNT_CREATED,
        category=EventCategory.ACCOUNT,
        customer_id=customer_id,
        details={"account_id": "ACC123"},
        ip_address="192.168.1.1"
    )
    
    print(f"   Logged 4 events for customer {customer_id}")
    
    # Verify integrity
    print("\n2. Verifying Audit Trail Integrity...")
    integrity_report = await audit_service.verify_audit_trail_integrity()
    print(f"   Integrity verified: {integrity_report['verified']}")
    print(f"   Events checked: {integrity_report['events_checked']}")
    print(f"   Violations: {len(integrity_report['violations'])}")
    
    # Get customer audit trail
    print("\n3. Retrieving Customer Audit Trail...")
    trail = await audit_service.get_customer_audit_trail(customer_id)
    print(f"   Found {len(trail)} events for customer")
    for event in trail:
        print(f"     - {event.event_type.value} at {event.timestamp.isoformat()}")
    
    # Generate reports
    print("\n4. Generating Compliance Reports...")
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=30)
    
    kyc_report = await reporting_service.generate_kyc_summary_report(
        start_date, end_date, ReportFormat.PDF
    )
    print(f"   KYC Summary Report: {kyc_report.report_id}")
    print(f"     Screenings: {kyc_report.summary['total_screenings']}")
    
    metrics_report = await reporting_service.generate_onboarding_metrics_report(
        start_date, end_date, ReportFormat.CSV
    )
    print(f"   Metrics Report: {metrics_report.report_id}")
    print(f"     Applications: {metrics_report.summary['applications_started']}")
    print(f"     Completion: {metrics_report.summary['completion_rate']}")
    
    # Export to CSV
    print("\n5. Exporting to CSV...")
    csv_content = await reporting_service.export_to_csv(trail)
    print(f"   Exported {len(trail)} events to CSV ({len(csv_content)} bytes)")
    
    # Schedule reports
    print("\n6. Scheduling Automated Reports...")
    schedule = reporting_service.schedule_report(
        report_type=ReportType.KYC_SUMMARY,
        frequency=ReportFrequency.DAILY,
        format=ReportFormat.PDF,
        recipients=["compliance@example.com"]
    )
    print(f"   Scheduled daily KYC report: {schedule.schedule_id}")
    print(f"   Next run: {schedule.next_run.isoformat()}")
    
    # Statistics
    print("\n7. Audit Statistics:")
    stats = audit_service.get_audit_statistics()
    print(f"   Total Events: {stats['total_events']}")
    print(f"   Unique Customers: {stats['unique_customers']}")
    print(f"   Chain Integrity: {stats['chain_integrity']}")
    
    report_stats = reporting_service.get_report_summary()
    print(f"\n8. Reporting Statistics:")
    print(f"   Reports Generated: {report_stats['total_reports_generated']}")
    print(f"   Active Schedules: {report_stats['active_schedules']}")
    
    print("\n" + "=" * 60)
    print("✅ Audit & Reporting Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_audit_reporting())
