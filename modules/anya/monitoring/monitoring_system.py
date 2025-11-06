"""
ANYA OPERATIONS & MONITORING SYSTEM
====================================

Comprehensive monitoring and observability:
- Prometheus metrics
- Structured logging
- Alerting system
- Performance dashboards
- Cost tracking

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional
from enum import Enum
import time
import logging
import json
import hashlib
from collections import defaultdict, deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class MetricType(str, Enum):
    """Metric types"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class InteractionLog:
    """Complete interaction log entry"""
    timestamp: datetime
    client_id_hash: str
    session_id: str
    query_text: str
    query_intent: str
    response_text: str
    generation_time_ms: float
    token_usage: Dict[str, int]
    retrieved_documents: List[Dict[str, Any]]
    moderation_flags: List[str]
    user_feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot"""
    timestamp: datetime
    request_count: int
    avg_response_time: float
    p50_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_rate: float
    satisfaction_score: float
    active_sessions: int


@dataclass
class Alert:
    """Monitoring alert"""
    alert_id: str
    severity: AlertSeverity
    title: str
    description: str
    metric: str
    threshold: float
    current_value: float
    duration_minutes: int
    triggered_at: datetime
    resolved_at: Optional[datetime] = None


# ============================================================================
# PROMETHEUS METRICS COLLECTOR
# ============================================================================

class PrometheusMetrics:
    """
    Prometheus metrics collection
    
    Tracks:
    - Request metrics
    - Retrieval metrics
    - LLM metrics
    - Quality metrics
    - Safety metrics
    """
    
    def __init__(self):
        try:
            from prometheus_client import Counter, Histogram, Gauge, Summary
            
            # Request metrics
            self.requests_total = Counter(
                'anya_requests_total',
                'Total Anya requests',
                ['intent', 'status']
            )
            
            self.response_time = Histogram(
                'anya_response_time_seconds',
                'Response generation time in seconds',
                buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
            )
            
            # Retrieval metrics
            self.retrieval_documents = Histogram(
                'anya_retrieval_documents_count',
                'Number of documents retrieved per query',
                buckets=[1, 3, 5, 10, 20, 50]
            )
            
            self.retrieval_time = Histogram(
                'anya_retrieval_time_seconds',
                'Retrieval latency in seconds',
                buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
            )
            
            # LLM metrics
            self.llm_tokens = Counter(
                'anya_llm_tokens_total',
                'Total LLM tokens used',
                ['type']  # prompt or completion
            )
            
            self.llm_cost = Counter(
                'anya_llm_cost_dollars',
                'Total LLM API cost in dollars'
            )
            
            # Quality metrics
            self.satisfaction_score = Gauge(
                'anya_user_satisfaction_score',
                'User satisfaction score (0-1)'
            )
            
            self.hallucination_rate = Gauge(
                'anya_hallucination_rate',
                'Hallucination detection rate'
            )
            
            # Safety metrics
            self.moderation_flags = Counter(
                'anya_moderation_flags_total',
                'Moderation flags by category',
                ['category']
            )
            
            # Active sessions
            self.active_sessions = Gauge(
                'anya_active_sessions',
                'Number of active chat sessions'
            )
            
            self.prometheus_available = True
            logger.info("Prometheus metrics initialized")
            
        except ImportError:
            logger.warning("prometheus_client not available, metrics disabled")
            self.prometheus_available = False
    
    def record_request(
        self,
        intent: str,
        status: str,
        response_time: float
    ):
        """Record request metrics"""
        if not self.prometheus_available:
            return
        
        self.requests_total.labels(intent=intent, status=status).inc()
        self.response_time.observe(response_time)
    
    def record_retrieval(
        self,
        document_count: int,
        retrieval_time: float
    ):
        """Record retrieval metrics"""
        if not self.prometheus_available:
            return
        
        self.retrieval_documents.observe(document_count)
        self.retrieval_time.observe(retrieval_time)
    
    def record_llm_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cost: float
    ):
        """Record LLM usage metrics"""
        if not self.prometheus_available:
            return
        
        self.llm_tokens.labels(type='prompt').inc(prompt_tokens)
        self.llm_tokens.labels(type='completion').inc(completion_tokens)
        self.llm_cost.inc(cost)
    
    def record_moderation_flag(self, category: str):
        """Record moderation flag"""
        if not self.prometheus_available:
            return
        
        self.moderation_flags.labels(category=category).inc()
    
    def update_satisfaction_score(self, score: float):
        """Update satisfaction score"""
        if not self.prometheus_available:
            return
        
        self.satisfaction_score.set(score)
    
    def update_active_sessions(self, count: int):
        """Update active sessions count"""
        if not self.prometheus_available:
            return
        
        self.active_sessions.set(count)


# ============================================================================
# INTERACTION LOGGER
# ============================================================================

class InteractionLogger:
    """
    Comprehensive interaction logging
    
    Logs:
    - All user queries
    - All AI responses
    - Performance data
    - Safety checks
    - User feedback
    """
    
    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self.logs: deque = deque(maxlen=10000)  # In-memory for demo
        
        # In production: Use time-series DB like InfluxDB or TimescaleDB
        logger.info(f"Interaction Logger initialized (retention: {retention_days} days)")
    
    def log_interaction(
        self,
        client_id: str,
        session_id: str,
        query: str,
        intent: str,
        response: str,
        generation_time_ms: float,
        token_usage: Dict[str, int],
        retrieved_docs: List[Dict[str, Any]],
        moderation_flags: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Log complete interaction
        
        Note: client_id is hashed for privacy
        """
        # Hash client ID for privacy
        client_id_hash = hashlib.sha256(client_id.encode()).hexdigest()[:16]
        
        log_entry = InteractionLog(
            timestamp=datetime.now(UTC),
            client_id_hash=client_id_hash,
            session_id=session_id,
            query_text=query,
            query_intent=intent,
            response_text=response,
            generation_time_ms=generation_time_ms,
            token_usage=token_usage,
            retrieved_documents=retrieved_docs,
            moderation_flags=moderation_flags,
            metadata=metadata or {}
        )
        
        self.logs.append(log_entry)
        
        # Also log to file/database
        self._write_to_storage(log_entry)
    
    def _write_to_storage(self, log_entry: InteractionLog):
        """Write log entry to persistent storage"""
        # In production: Write to TimescaleDB, InfluxDB, or similar
        # For demo: Just log to file
        log_dict = {
            "timestamp": log_entry.timestamp.isoformat(),
            "client_id_hash": log_entry.client_id_hash,
            "session_id": log_entry.session_id,
            "query_intent": log_entry.query_intent,
            "generation_time_ms": log_entry.generation_time_ms,
            "token_usage": log_entry.token_usage,
            "moderation_flags": log_entry.moderation_flags
        }
        
        logger.info(f"Interaction logged: {json.dumps(log_dict)}")
    
    def get_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        session_id: Optional[str] = None
    ) -> List[InteractionLog]:
        """Query logs with filters"""
        filtered_logs = list(self.logs)
        
        if start_time:
            filtered_logs = [l for l in filtered_logs if l.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [l for l in filtered_logs if l.timestamp <= end_time]
        
        if session_id:
            filtered_logs = [l for l in filtered_logs if l.session_id == session_id]
        
        return filtered_logs


# ============================================================================
# ALERT MANAGER
# ============================================================================

class AlertManager:
    """
    Monitoring alert system
    
    Manages:
    - Alert rules
    - Alert state
    - Alert notifications
    """
    
    def __init__(self):
        self.alert_rules = self._initialize_alert_rules()
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        logger.info("Alert Manager initialized")
    
    def _initialize_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize alert rules"""
        return {
            "high_latency": {
                "metric": "response_time_p95",
                "threshold": 5.0,
                "duration_minutes": 5,
                "severity": AlertSeverity.WARNING,
                "title": "High Response Latency",
                "description": "P95 response time exceeds 5 seconds"
            },
            "high_moderation_rate": {
                "metric": "moderation_flags_per_second",
                "threshold": 0.1,
                "duration_minutes": 15,
                "severity": AlertSeverity.WARNING,
                "title": "High Moderation Flag Rate",
                "description": "Moderation flags exceed normal threshold"
            },
            "low_satisfaction": {
                "metric": "satisfaction_score_24h",
                "threshold": 0.7,
                "duration_minutes": 60,
                "severity": AlertSeverity.WARNING,
                "title": "Low User Satisfaction",
                "description": "24-hour satisfaction score below acceptable level"
            },
            "high_hallucination_rate": {
                "metric": "hallucination_rate",
                "threshold": 0.05,
                "duration_minutes": 30,
                "severity": AlertSeverity.CRITICAL,
                "title": "High Hallucination Rate",
                "description": "Hallucination detection rate exceeds 5%"
            },
            "high_error_rate": {
                "metric": "error_rate",
                "threshold": 0.01,
                "duration_minutes": 10,
                "severity": AlertSeverity.CRITICAL,
                "title": "High Error Rate",
                "description": "Error rate exceeds 1%"
            }
        }
    
    def check_alert(
        self,
        rule_name: str,
        current_value: float
    ):
        """Check if alert should be triggered"""
        if rule_name not in self.alert_rules:
            return
        
        rule = self.alert_rules[rule_name]
        
        # Check if threshold exceeded
        if current_value > rule["threshold"]:
            # Trigger or update alert
            if rule_name not in self.active_alerts:
                alert = Alert(
                    alert_id=f"{rule_name}_{datetime.now(UTC).timestamp()}",
                    severity=rule["severity"],
                    title=rule["title"],
                    description=rule["description"],
                    metric=rule["metric"],
                    threshold=rule["threshold"],
                    current_value=current_value,
                    duration_minutes=rule["duration_minutes"],
                    triggered_at=datetime.now(UTC)
                )
                
                self.active_alerts[rule_name] = alert
                self._send_alert_notification(alert)
                
                logger.warning(f"Alert triggered: {alert.title} ({current_value:.3f} > {rule['threshold']})")
        
        else:
            # Resolve alert if active
            if rule_name in self.active_alerts:
                alert = self.active_alerts[rule_name]
                alert.resolved_at = datetime.now(UTC)
                
                self.alert_history.append(alert)
                del self.active_alerts[rule_name]
                
                logger.info(f"Alert resolved: {alert.title}")
    
    def _send_alert_notification(self, alert: Alert):
        """Send alert notification"""
        # In production: Send to PagerDuty, Slack, email, etc.
        notification = {
            "severity": alert.severity.value,
            "title": alert.title,
            "description": alert.description,
            "metric": alert.metric,
            "threshold": alert.threshold,
            "current_value": alert.current_value,
            "triggered_at": alert.triggered_at.isoformat()
        }
        
        logger.warning(f"ALERT: {json.dumps(notification)}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())


# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """
    Real-time performance monitoring
    
    Tracks:
    - Request volume
    - Response times
    - Error rates
    - Satisfaction scores
    - Resource usage
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        
        # Rolling windows for metrics
        self.response_times: deque = deque(maxlen=window_size)
        self.request_statuses: deque = deque(maxlen=window_size)
        self.satisfaction_scores: deque = deque(maxlen=window_size)
        
        # Counters
        self.total_requests = 0
        self.total_errors = 0
        
        logger.info("Performance Monitor initialized")
    
    def record_request(
        self,
        response_time: float,
        status: str,
        satisfaction_score: Optional[float] = None
    ):
        """Record request metrics"""
        self.response_times.append(response_time)
        self.request_statuses.append(status)
        
        self.total_requests += 1
        
        if status == "error":
            self.total_errors += 1
        
        if satisfaction_score is not None:
            self.satisfaction_scores.append(satisfaction_score)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        if not self.response_times:
            return PerformanceMetrics(
                timestamp=datetime.now(UTC),
                request_count=0,
                avg_response_time=0,
                p50_response_time=0,
                p95_response_time=0,
                p99_response_time=0,
                error_rate=0,
                satisfaction_score=0,
                active_sessions=0
            )
        
        sorted_times = sorted(self.response_times)
        n = len(sorted_times)
        
        return PerformanceMetrics(
            timestamp=datetime.now(UTC),
            request_count=self.total_requests,
            avg_response_time=sum(sorted_times) / n,
            p50_response_time=sorted_times[int(n * 0.5)],
            p95_response_time=sorted_times[int(n * 0.95)],
            p99_response_time=sorted_times[int(n * 0.99)],
            error_rate=self.total_errors / self.total_requests if self.total_requests > 0 else 0,
            satisfaction_score=sum(self.satisfaction_scores) / len(self.satisfaction_scores) if self.satisfaction_scores else 0,
            active_sessions=0  # Would be updated by session manager
        )


# ============================================================================
# COST TRACKER
# ============================================================================

class CostTracker:
    """
    Track API costs for LLM usage
    
    Pricing:
    - GPT-4 Turbo: $10/1M prompt tokens, $30/1M completion tokens
    - Embedding: $0.13/1M tokens
    """
    
    # Pricing per 1M tokens
    PRICING = {
        "gpt4_turbo_prompt": 10.0,
        "gpt4_turbo_completion": 30.0,
        "embedding": 0.13
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.cost_by_type: Dict[str, float] = defaultdict(float)
        self.token_usage: Dict[str, int] = defaultdict(int)
        
        logger.info("Cost Tracker initialized")
    
    def record_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        embedding_tokens: int = 0
    ) -> float:
        """
        Record token usage and calculate cost
        
        Returns: Cost for this interaction
        """
        # Calculate costs
        prompt_cost = (prompt_tokens / 1_000_000) * self.PRICING["gpt4_turbo_prompt"]
        completion_cost = (completion_tokens / 1_000_000) * self.PRICING["gpt4_turbo_completion"]
        embedding_cost = (embedding_tokens / 1_000_000) * self.PRICING["embedding"]
        
        interaction_cost = prompt_cost + completion_cost + embedding_cost
        
        # Track usage
        self.total_cost += interaction_cost
        self.cost_by_type["prompt"] += prompt_cost
        self.cost_by_type["completion"] += completion_cost
        self.cost_by_type["embedding"] += embedding_cost
        
        self.token_usage["prompt"] += prompt_tokens
        self.token_usage["completion"] += completion_tokens
        self.token_usage["embedding"] += embedding_tokens
        
        return interaction_cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost summary"""
        return {
            "total_cost": f"${self.total_cost:.4f}",
            "cost_by_type": {
                k: f"${v:.4f}" for k, v in self.cost_by_type.items()
            },
            "token_usage": dict(self.token_usage),
            "cost_per_interaction": f"${self.total_cost / max(1, sum(self.token_usage.values()) / 1000):.4f}"
        }


# ============================================================================
# MONITORING DASHBOARD
# ============================================================================

class MonitoringDashboard:
    """
    Real-time monitoring dashboard
    
    Provides:
    - Current metrics
    - Alert status
    - Cost tracking
    - Performance trends
    """
    
    def __init__(
        self,
        prometheus_metrics: PrometheusMetrics,
        interaction_logger: InteractionLogger,
        alert_manager: AlertManager,
        performance_monitor: PerformanceMonitor,
        cost_tracker: CostTracker
    ):
        self.prometheus = prometheus_metrics
        self.logger = interaction_logger
        self.alerts = alert_manager
        self.performance = performance_monitor
        self.costs = cost_tracker
        
        logger.info("Monitoring Dashboard initialized")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get complete dashboard data"""
        metrics = self.performance.get_metrics()
        
        return {
            "timestamp": datetime.now(UTC).isoformat(),
            "performance": {
                "request_count": metrics.request_count,
                "avg_response_time": f"{metrics.avg_response_time:.3f}s",
                "p50_response_time": f"{metrics.p50_response_time:.3f}s",
                "p95_response_time": f"{metrics.p95_response_time:.3f}s",
                "p99_response_time": f"{metrics.p99_response_time:.3f}s",
                "error_rate": f"{metrics.error_rate * 100:.2f}%",
                "satisfaction_score": f"{metrics.satisfaction_score:.2f}",
                "active_sessions": metrics.active_sessions
            },
            "costs": self.costs.get_cost_summary(),
            "alerts": {
                "active_count": len(self.alerts.active_alerts),
                "active_alerts": [
                    {
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "current_value": alert.current_value,
                        "threshold": alert.threshold
                    }
                    for alert in self.alerts.get_active_alerts()
                ]
            },
            "recent_logs": len(self.logger.logs)
        }
    
    def print_dashboard(self):
        """Print dashboard to console"""
        data = self.get_dashboard_data()
        
        print("\n" + "=" * 70)
        print("ANYA MONITORING DASHBOARD")
        print("=" * 70)
        print(f"Timestamp: {data['timestamp']}")
        
        print("\n📊 PERFORMANCE METRICS:")
        perf = data['performance']
        print(f"  Total Requests: {perf['request_count']}")
        print(f"  Avg Response Time: {perf['avg_response_time']}")
        print(f"  P95 Response Time: {perf['p95_response_time']}")
        print(f"  P99 Response Time: {perf['p99_response_time']}")
        print(f"  Error Rate: {perf['error_rate']}")
        print(f"  Satisfaction Score: {perf['satisfaction_score']}")
        print(f"  Active Sessions: {perf['active_sessions']}")
        
        print("\n💰 COST TRACKING:")
        costs = data['costs']
        print(f"  Total Cost: {costs['total_cost']}")
        print(f"  Cost per Interaction: {costs['cost_per_interaction']}")
        
        print("\n🚨 ALERTS:")
        alerts = data['alerts']
        print(f"  Active Alerts: {alerts['active_count']}")
        
        if alerts['active_alerts']:
            for alert in alerts['active_alerts']:
                print(f"    [{alert['severity'].upper()}] {alert['title']}")
                print(f"      Current: {alert['current_value']:.3f} / Threshold: {alert['threshold']:.3f}")
        
        print("\n" + "=" * 70 + "\n")


# ============================================================================
# COMPLETE MONITORING SYSTEM
# ============================================================================

class AnyaMonitoringSystem:
    """
    Complete monitoring and operations system
    
    Integrates:
    - Prometheus metrics
    - Interaction logging
    - Alert management
    - Performance monitoring
    - Cost tracking
    - Dashboard
    """
    
    def __init__(self):
        self.prometheus = PrometheusMetrics()
        self.interaction_logger = InteractionLogger()
        self.alert_manager = AlertManager()
        self.performance_monitor = PerformanceMonitor()
        self.cost_tracker = CostTracker()
        
        self.dashboard = MonitoringDashboard(
            self.prometheus,
            self.interaction_logger,
            self.alert_manager,
            self.performance_monitor,
            self.cost_tracker
        )
        
        logger.info("✅ Anya Monitoring System initialized")
    
    def record_interaction(
        self,
        client_id: str,
        session_id: str,
        query: str,
        intent: str,
        response: str,
        start_time: float,
        token_usage: Dict[str, int],
        retrieved_docs: List[Dict[str, Any]],
        moderation_flags: List[str],
        status: str = "success",
        satisfaction_score: Optional[float] = None
    ):
        """
        Record complete interaction with all monitoring
        
        This is the main entry point for monitoring
        """
        # Calculate metrics
        generation_time = time.time() - start_time
        generation_time_ms = generation_time * 1000
        
        # Log interaction
        self.interaction_logger.log_interaction(
            client_id=client_id,
            session_id=session_id,
            query=query,
            intent=intent,
            response=response,
            generation_time_ms=generation_time_ms,
            token_usage=token_usage,
            retrieved_docs=retrieved_docs,
            moderation_flags=moderation_flags
        )
        
        # Record Prometheus metrics
        self.prometheus.record_request(intent, status, generation_time)
        self.prometheus.record_retrieval(len(retrieved_docs), 0.1)  # Mock retrieval time
        
        if token_usage:
            cost = self.cost_tracker.record_usage(
                token_usage.get("prompt_tokens", 0),
                token_usage.get("completion_tokens", 0)
            )
            self.prometheus.record_llm_usage(
                token_usage.get("prompt_tokens", 0),
                token_usage.get("completion_tokens", 0),
                cost
            )
        
        for flag in moderation_flags:
            self.prometheus.record_moderation_flag(flag)
        
        # Record performance
        self.performance_monitor.record_request(
            generation_time,
            status,
            satisfaction_score
        )
        
        # Check alerts
        metrics = self.performance_monitor.get_metrics()
        self.alert_manager.check_alert("high_latency", metrics.p95_response_time)
        self.alert_manager.check_alert("low_satisfaction", metrics.satisfaction_score)
        self.alert_manager.check_alert("high_error_rate", metrics.error_rate)
    
    def get_dashboard(self) -> Dict[str, Any]:
        """Get dashboard data"""
        return self.dashboard.get_dashboard_data()
    
    def print_dashboard(self):
        """Print dashboard"""
        self.dashboard.print_dashboard()


# ============================================================================
# DEMO
# ============================================================================

def demo_monitoring():
    """Demonstrate monitoring system"""
    print("\n" + "=" * 70)
    print("ANYA MONITORING SYSTEM DEMO")
    print("=" * 70)
    
    monitoring = AnyaMonitoringSystem()
    
    # Simulate some interactions
    print("\n📊 Simulating interactions...")
    
    import random
    
    for i in range(20):
        monitoring.record_interaction(
            client_id=f"customer_{i % 3}",
            session_id=f"session_{i % 5}",
            query=f"What is diversification? (query {i})",
            intent="educational",
            response="Diversification reduces risk by spreading investments...",
            start_time=time.time() - random.uniform(0.1, 3.0),
            token_usage={
                "prompt_tokens": random.randint(100, 500),
                "completion_tokens": random.randint(50, 200)
            },
            retrieved_docs=[{"source": "Guide", "relevance": 0.9}],
            moderation_flags=[],
            status="success",
            satisfaction_score=random.uniform(0.7, 1.0)
        )
    
    # Add some problematic interactions
    monitoring.record_interaction(
        client_id="customer_problem",
        session_id="session_problem",
        query="Should I buy stocks?",
        intent="advice_request",
        response="I can't provide investment advice...",
        start_time=time.time() - 6.0,  # Slow
        token_usage={"prompt_tokens": 200, "completion_tokens": 100},
        retrieved_docs=[],
        moderation_flags=["financial_advice"],
        status="success",
        satisfaction_score=0.5
    )
    
    # Print dashboard
    monitoring.print_dashboard()
    
    # Show active alerts
    print("📢 Active Alerts:")
    active_alerts = monitoring.alert_manager.get_active_alerts()
    
    if active_alerts:
        for alert in active_alerts:
            print(f"  • [{alert.severity.value.upper()}] {alert.title}")
            print(f"    {alert.description}")
            print(f"    Current: {alert.current_value:.3f} / Threshold: {alert.threshold}")
    else:
        print("  No active alerts")
    
    print("\n" + "=" * 70)
    print("✅ Monitoring System Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    demo_monitoring()
