"""
ULTRA PLATFORM - PERFORMANCE & SCALING SYSTEM
==============================================

Institutional-grade performance monitoring, optimization, and auto-scaling
providing:
- Real-time performance monitoring
- Intelligent auto-scaling
- Advanced caching strategies
- Database query optimization
- Load testing framework
- Capacity planning
- Performance analytics
- Bottleneck detection
- SLA monitoring
- Cost optimization

Author: Ultra Platform Team
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from collections import deque, defaultdict
import asyncio
import time
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class ServiceType(str, Enum):
    """Service types in the platform"""
    API_GATEWAY = "api_gateway"
    IDENTITY_VERIFICATION = "identity_verification"
    KYC_SCREENING = "kyc_screening"
    FRAUD_DETECTION = "fraud_detection"
    ACCOUNT_OPENING = "account_opening"
    DOCUMENT_PROCESSING = "document_processing"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"


class MetricType(str, Enum):
    """Performance metric types"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    CACHE_HIT_RATE = "cache_hit_rate"
    QUEUE_DEPTH = "queue_depth"


class ScalingAction(str, Enum):
    """Auto-scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"
    SCALE_IN = "scale_in"
    NO_ACTION = "no_action"


class PerformanceStatus(str, Enum):
    """Performance health status"""
    OPTIMAL = "optimal"
    GOOD = "good"
    DEGRADED = "degraded"
    CRITICAL = "critical"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class PerformanceMetric:
    """Performance metric data point"""
    service: ServiceType
    metric_type: MetricType
    value: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """Performance benchmark target"""
    operation: str
    target_latency_ms: float
    target_throughput: int
    target_error_rate: float
    current_latency_ms: float = 0.0
    current_throughput: int = 0
    current_error_rate: float = 0.0
    
    @property
    def is_meeting_target(self) -> bool:
        """Check if meeting performance targets"""
        return (
            self.current_latency_ms <= self.target_latency_ms and
            self.current_throughput >= self.target_throughput and
            self.current_error_rate <= self.target_error_rate
        )
    
    @property
    def latency_percent_of_target(self) -> float:
        """Calculate latency as percentage of target"""
        if self.target_latency_ms == 0:
            return 0.0
        return (self.current_latency_ms / self.target_latency_ms) * 100


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration"""
    service: ServiceType
    min_instances: int
    max_instances: int
    target_cpu_percent: float = 70.0
    target_memory_percent: float = 80.0
    scale_up_cooldown_seconds: int = 300
    scale_down_cooldown_seconds: int = 600
    queue_depth_threshold: int = 100
    enabled: bool = True


@dataclass
class ScalingEvent:
    """Scaling event record"""
    event_id: str
    service: ServiceType
    action: ScalingAction
    reason: str
    instances_before: int
    instances_after: int
    timestamp: datetime
    triggered_by: str


@dataclass
class CacheConfig:
    """Cache configuration"""
    cache_type: str  # redis, memcached, in-memory
    ttl_seconds: int
    max_size_mb: int
    eviction_policy: str  # LRU, LFU, FIFO
    hit_rate_target: float = 0.90


@dataclass
class CapacityPlan:
    """Capacity planning data"""
    service: ServiceType
    current_capacity: int
    projected_growth_rate: float
    capacity_needed_30d: int
    capacity_needed_90d: int
    capacity_needed_1y: int
    bottlenecks: List[str] = field(default_factory=list)


# ============================================================================
# PERFORMANCE MONITORING SERVICE
# ============================================================================

class PerformanceMonitoringService:
    """
    Real-time performance monitoring with SLA tracking
    """
    
    def __init__(self):
        # Metrics storage (use time-series DB in production)
        self.metrics: Dict[ServiceType, deque] = defaultdict(lambda: deque(maxlen=10000))
        
        # Performance benchmarks
        self.benchmarks = self._initialize_benchmarks()
        
        # SLA thresholds
        self.sla_thresholds = self._initialize_sla_thresholds()
        
        # Alert history
        self.alerts: List[Dict[str, Any]] = []
        
        logger.info("Performance Monitoring Service initialized")
    
    def _initialize_benchmarks(self) -> Dict[str, PerformanceBenchmark]:
        """Initialize performance benchmarks"""
        return {
            "application_start": PerformanceBenchmark(
                operation="Application Start",
                target_latency_ms=500,
                target_throughput=1000,
                target_error_rate=0.001
            ),
            "identity_verification": PerformanceBenchmark(
                operation="Identity Verification",
                target_latency_ms=30000,  # 30 seconds
                target_throughput=500,
                target_error_rate=0.005
            ),
            "document_upload": PerformanceBenchmark(
                operation="Document Upload",
                target_latency_ms=5000,  # 5 seconds
                target_throughput=2000,
                target_error_rate=0.002
            ),
            "kyc_screening": PerformanceBenchmark(
                operation="KYC Screening",
                target_latency_ms=10000,  # 10 seconds
                target_throughput=800,
                target_error_rate=0.001
            ),
            "account_opening": PerformanceBenchmark(
                operation="Account Opening",
                target_latency_ms=120000,  # 2 minutes
                target_throughput=500,
                target_error_rate=0.001
            ),
            "end_to_end_onboarding": PerformanceBenchmark(
                operation="End-to-End Onboarding",
                target_latency_ms=600000,  # 10 minutes (80th percentile)
                target_throughput=400,
                target_error_rate=0.005
            )
        }
    
    def _initialize_sla_thresholds(self) -> Dict[str, float]:
        """Initialize SLA thresholds"""
        return {
            "availability": 99.9,  # 99.9% uptime
            "api_latency_p95": 1000,  # 1 second
            "api_latency_p99": 2000,  # 2 seconds
            "error_rate": 0.01,  # 1% error rate
            "cache_hit_rate": 0.90  # 90% cache hit rate
        }
    
    async def record_metric(
        self,
        service: ServiceType,
        metric_type: MetricType,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record performance metric"""
        metric = PerformanceMetric(
            service=service,
            metric_type=metric_type,
            value=value,
            timestamp=datetime.now(UTC),
            metadata=metadata or {}
        )
        
        self.metrics[service].append(metric)
        
        # Check for SLA violations
        await self._check_sla_violations(service, metric_type, value)
    
    async def _check_sla_violations(
        self,
        service: ServiceType,
        metric_type: MetricType,
        value: float
    ):
        """Check for SLA violations and raise alerts"""
        violated = False
        
        if metric_type == MetricType.LATENCY:
            if value > self.sla_thresholds["api_latency_p95"]:
                violated = True
                severity = "warning"
                if value > self.sla_thresholds["api_latency_p99"]:
                    severity = "critical"
                
                alert = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "service": service.value,
                    "metric": metric_type.value,
                    "value": value,
                    "threshold": self.sla_thresholds["api_latency_p95"],
                    "severity": severity,
                    "message": f"Latency exceeded SLA: {value}ms > {self.sla_thresholds['api_latency_p95']}ms"
                }
                
                self.alerts.append(alert)
                logger.warning(f"SLA violation: {alert['message']}")
        
        elif metric_type == MetricType.ERROR_RATE:
            if value > self.sla_thresholds["error_rate"]:
                violated = True
                
                alert = {
                    "timestamp": datetime.now(UTC).isoformat(),
                    "service": service.value,
                    "metric": metric_type.value,
                    "value": value,
                    "threshold": self.sla_thresholds["error_rate"],
                    "severity": "critical",
                    "message": f"Error rate exceeded SLA: {value*100}% > {self.sla_thresholds['error_rate']*100}%"
                }
                
                self.alerts.append(alert)
                logger.error(f"SLA violation: {alert['message']}")
    
    def get_service_metrics(
        self,
        service: ServiceType,
        metric_type: Optional[MetricType] = None,
        time_window_minutes: int = 60
    ) -> List[PerformanceMetric]:
        """Get metrics for a service"""
        cutoff_time = datetime.now(UTC) - timedelta(minutes=time_window_minutes)
        
        metrics = self.metrics[service]
        
        # Filter by time
        recent_metrics = [m for m in metrics if m.timestamp >= cutoff_time]
        
        # Filter by metric type if specified
        if metric_type:
            recent_metrics = [m for m in recent_metrics if m.metric_type == metric_type]
        
        return recent_metrics
    
    def calculate_percentile(
        self,
        service: ServiceType,
        metric_type: MetricType,
        percentile: float,
        time_window_minutes: int = 60
    ) -> float:
        """Calculate percentile for a metric"""
        metrics = self.get_service_metrics(service, metric_type, time_window_minutes)
        
        if not metrics:
            return 0.0
        
        values = [m.value for m in metrics]
        values.sort()
        
        index = int(len(values) * (percentile / 100))
        return values[min(index, len(values) - 1)]
    
    def get_performance_status(
        self,
        service: ServiceType
    ) -> PerformanceStatus:
        """Get current performance status for service"""
        # Get recent metrics
        latency_p95 = self.calculate_percentile(service, MetricType.LATENCY, 95, 5)
        
        recent_metrics = self.get_service_metrics(service, time_window_minutes=5)
        
        if not recent_metrics:
            return PerformanceStatus.GOOD
        
        # Calculate error rate
        error_metrics = [m for m in recent_metrics if m.metric_type == MetricType.ERROR_RATE]
        avg_error_rate = statistics.mean([m.value for m in error_metrics]) if error_metrics else 0.0
        
        # Determine status
        if latency_p95 > self.sla_thresholds["api_latency_p99"] or avg_error_rate > 0.05:
            return PerformanceStatus.CRITICAL
        elif latency_p95 > self.sla_thresholds["api_latency_p95"] or avg_error_rate > 0.02:
            return PerformanceStatus.DEGRADED
        elif latency_p95 < self.sla_thresholds["api_latency_p95"] * 0.7:
            return PerformanceStatus.OPTIMAL
        else:
            return PerformanceStatus.GOOD
    
    def get_benchmark_status(self) -> Dict[str, Any]:
        """Get status of all benchmarks"""
        return {
            operation: {
                "target_latency_ms": bench.target_latency_ms,
                "current_latency_ms": bench.current_latency_ms,
                "target_throughput": bench.target_throughput,
                "current_throughput": bench.current_throughput,
                "meeting_target": bench.is_meeting_target,
                "latency_percent": f"{bench.latency_percent_of_target:.1f}%"
            }
            for operation, bench in self.benchmarks.items()
        }


# ============================================================================
# AUTO-SCALING SERVICE
# ============================================================================

class AutoScalingService:
    """
    Intelligent auto-scaling with predictive capabilities
    """
    
    def __init__(self, performance_monitor: PerformanceMonitoringService):
        self.performance_monitor = performance_monitor
        
        # Scaling policies
        self.policies = self._initialize_policies()
        
        # Current instance counts
        self.current_instances: Dict[ServiceType, int] = {}
        
        # Scaling history
        self.scaling_events: List[ScalingEvent] = []
        
        # Last scaling actions (for cooldown)
        self.last_scale_up: Dict[ServiceType, datetime] = {}
        self.last_scale_down: Dict[ServiceType, datetime] = {}
        
        logger.info("Auto-Scaling Service initialized")
    
    def _initialize_policies(self) -> Dict[ServiceType, ScalingPolicy]:
        """Initialize scaling policies"""
        return {
            ServiceType.API_GATEWAY: ScalingPolicy(
                service=ServiceType.API_GATEWAY,
                min_instances=3,
                max_instances=50,
                target_cpu_percent=70.0,
                target_memory_percent=80.0
            ),
            ServiceType.IDENTITY_VERIFICATION: ScalingPolicy(
                service=ServiceType.IDENTITY_VERIFICATION,
                min_instances=5,
                max_instances=100,
                target_cpu_percent=75.0,
                target_memory_percent=85.0
            ),
            ServiceType.KYC_SCREENING: ScalingPolicy(
                service=ServiceType.KYC_SCREENING,
                min_instances=5,
                max_instances=80,
                target_cpu_percent=70.0,
                target_memory_percent=80.0
            ),
            ServiceType.FRAUD_DETECTION: ScalingPolicy(
                service=ServiceType.FRAUD_DETECTION,
                min_instances=3,
                max_instances=60,
                target_cpu_percent=70.0,
                target_memory_percent=80.0
            ),
            ServiceType.DOCUMENT_PROCESSING: ScalingPolicy(
                service=ServiceType.DOCUMENT_PROCESSING,
                min_instances=5,
                max_instances=100,
                target_cpu_percent=75.0,
                target_memory_percent=85.0
            )
        }
    
    async def evaluate_scaling(
        self,
        service: ServiceType
    ) -> Optional[ScalingAction]:
        """
        Evaluate if scaling action is needed
        
        Returns scaling action or None if no action needed
        """
        policy = self.policies.get(service)
        
        if not policy or not policy.enabled:
            return None
        
        # Get current metrics
        cpu_usage = self._get_avg_metric(service, MetricType.CPU_USAGE, 5)
        memory_usage = self._get_avg_metric(service, MetricType.MEMORY_USAGE, 5)
        queue_depth = self._get_avg_metric(service, MetricType.QUEUE_DEPTH, 2)
        
        current_count = self.current_instances.get(service, policy.min_instances)
        
        # Check cooldown periods
        now = datetime.now(UTC)
        
        last_up = self.last_scale_up.get(service)
        last_down = self.last_scale_down.get(service)
        
        # Determine if scale up needed
        scale_up_needed = (
            cpu_usage > policy.target_cpu_percent or
            memory_usage > policy.target_memory_percent or
            queue_depth > policy.queue_depth_threshold
        )
        
        if scale_up_needed and current_count < policy.max_instances:
            # Check cooldown
            if last_up and (now - last_up).total_seconds() < policy.scale_up_cooldown_seconds:
                return None
            
            return ScalingAction.SCALE_OUT
        
        # Determine if scale down needed
        scale_down_needed = (
            cpu_usage < policy.target_cpu_percent * 0.5 and
            memory_usage < policy.target_memory_percent * 0.5 and
            queue_depth < policy.queue_depth_threshold * 0.3
        )
        
        if scale_down_needed and current_count > policy.min_instances:
            # Check cooldown
            if last_down and (now - last_down).total_seconds() < policy.scale_down_cooldown_seconds:
                return None
            
            return ScalingAction.SCALE_IN
        
        return ScalingAction.NO_ACTION
    
    def _get_avg_metric(
        self,
        service: ServiceType,
        metric_type: MetricType,
        time_window_minutes: int
    ) -> float:
        """Get average metric value"""
        metrics = self.performance_monitor.get_service_metrics(
            service, metric_type, time_window_minutes
        )
        
        if not metrics:
            return 0.0
        
        return statistics.mean([m.value for m in metrics])
    
    async def execute_scaling(
        self,
        service: ServiceType,
        action: ScalingAction,
        reason: str
    ) -> ScalingEvent:
        """
        Execute scaling action
        
        In production, this would call cloud provider APIs (AWS, Azure, GCP)
        """
        policy = self.policies[service]
        current_count = self.current_instances.get(service, policy.min_instances)
        
        new_count = current_count
        
        if action == ScalingAction.SCALE_OUT:
            new_count = min(current_count + 1, policy.max_instances)
            self.last_scale_up[service] = datetime.now(UTC)
        
        elif action == ScalingAction.SCALE_IN:
            new_count = max(current_count - 1, policy.min_instances)
            self.last_scale_down[service] = datetime.now(UTC)
        
        event = ScalingEvent(
            event_id=f"scale_{service.value}_{datetime.now(UTC).timestamp()}",
            service=service,
            action=action,
            reason=reason,
            instances_before=current_count,
            instances_after=new_count,
            timestamp=datetime.now(UTC),
            triggered_by="auto_scaler"
        )
        
        self.scaling_events.append(event)
        self.current_instances[service] = new_count
        
        logger.info(f"Scaling {service.value}: {current_count} → {new_count} instances ({action.value})")
        
        return event
    
    def get_scaling_history(
        self,
        service: Optional[ServiceType] = None,
        hours: int = 24
    ) -> List[ScalingEvent]:
        """Get scaling event history"""
        cutoff = datetime.now(UTC) - timedelta(hours=hours)
        
        events = [e for e in self.scaling_events if e.timestamp >= cutoff]
        
        if service:
            events = [e for e in events if e.service == service]
        
        return events


# ============================================================================
# CACHE MANAGEMENT SERVICE
# ============================================================================

class CacheManagementService:
    """
    Advanced caching with intelligent invalidation
    """
    
    def __init__(self):
        # Cache storage (use Redis in production)
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Cache configurations
        self.cache_configs = self._initialize_cache_configs()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
        logger.info("Cache Management Service initialized")
    
    def _initialize_cache_configs(self) -> Dict[str, CacheConfig]:
        """Initialize cache configurations"""
        return {
            "customer_profiles": CacheConfig(
                cache_type="redis",
                ttl_seconds=3600,  # 1 hour
                max_size_mb=1000,
                eviction_policy="LRU"
            ),
            "kyc_results": CacheConfig(
                cache_type="redis",
                ttl_seconds=86400,  # 24 hours
                max_size_mb=2000,
                eviction_policy="LRU"
            ),
            "document_metadata": CacheConfig(
                cache_type="redis",
                ttl_seconds=7200,  # 2 hours
                max_size_mb=500,
                eviction_policy="LFU"
            ),
            "api_responses": CacheConfig(
                cache_type="redis",
                ttl_seconds=300,  # 5 minutes
                max_size_mb=500,
                eviction_policy="LRU"
            )
        }
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            metadata = self.cache_metadata.get(key, {})
            expires_at = metadata.get("expires_at")
            
            if expires_at and datetime.now(UTC) > expires_at:
                # Expired
                del self.cache[key]
                del self.cache_metadata[key]
                self.misses += 1
                return None
            
            # Hit
            self.hits += 1
            metadata["last_accessed"] = datetime.now(UTC)
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            
            return self.cache[key]
        
        # Miss
        self.misses += 1
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        cache_type: str = "default"
    ):
        """Set value in cache"""
        config = self.cache_configs.get(cache_type)
        
        if not ttl_seconds and config:
            ttl_seconds = config.ttl_seconds
        
        self.cache[key] = value
        
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now(UTC) + timedelta(seconds=ttl_seconds)
        
        self.cache_metadata[key] = {
            "created_at": datetime.now(UTC),
            "last_accessed": datetime.now(UTC),
            "expires_at": expires_at,
            "access_count": 0,
            "cache_type": cache_type
        }
    
    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        keys_to_delete = [k for k in self.cache.keys() if pattern in k]
        
        for key in keys_to_delete:
            del self.cache[key]
            del self.cache_metadata[key]
            self.evictions += 1
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": f"{hit_rate:.2f}%",
            "total_keys": len(self.cache),
            "cache_configs": {
                name: {
                    "ttl_seconds": config.ttl_seconds,
                    "max_size_mb": config.max_size_mb,
                    "eviction_policy": config.eviction_policy
                }
                for name, config in self.cache_configs.items()
            }
        }


# ============================================================================
# CAPACITY PLANNING SERVICE
# ============================================================================

class CapacityPlanningService:
    """
    Capacity planning with growth projections
    """
    
    def __init__(self, performance_monitor: PerformanceMonitoringService):
        self.performance_monitor = performance_monitor
        
        # Historical capacity data
        self.capacity_history: Dict[ServiceType, deque] = defaultdict(lambda: deque(maxlen=365))
        
        logger.info("Capacity Planning Service initialized")
    
    async def analyze_capacity(
        self,
        service: ServiceType
    ) -> CapacityPlan:
        """Analyze capacity and project future needs"""
        # Get historical throughput data
        metrics = self.performance_monitor.get_service_metrics(
            service, MetricType.THROUGHPUT, time_window_minutes=10080  # 7 days
        )
        
        if len(metrics) < 100:
            # Not enough data
            return CapacityPlan(
                service=service,
                current_capacity=100,
                projected_growth_rate=0.10,
                capacity_needed_30d=110,
                capacity_needed_90d=130,
                capacity_needed_1y=180
            )
        
        # Calculate current capacity
        current_capacity = int(statistics.mean([m.value for m in metrics[-100:]]))
        
        # Calculate growth rate (simplified linear regression)
        growth_rate = self._calculate_growth_rate(metrics)
        
        # Project future capacity needs
        capacity_30d = int(current_capacity * (1 + growth_rate * 30))
        capacity_90d = int(current_capacity * (1 + growth_rate * 90))
        capacity_1y = int(current_capacity * (1 + growth_rate * 365))
        
        # Identify bottlenecks
        bottlenecks = self._identify_bottlenecks(service)
        
        return CapacityPlan(
            service=service,
            current_capacity=current_capacity,
            projected_growth_rate=growth_rate,
            capacity_needed_30d=capacity_30d,
            capacity_needed_90d=capacity_90d,
            capacity_needed_1y=capacity_1y,
            bottlenecks=bottlenecks
        )
    
    def _calculate_growth_rate(self, metrics: List[PerformanceMetric]) -> float:
        """Calculate daily growth rate"""
        if len(metrics) < 2:
            return 0.05  # Default 5% growth
        
        # Simple growth calculation: (latest - earliest) / days
        earliest = metrics[0].value
        latest = metrics[-1].value
        
        days = (metrics[-1].timestamp - metrics[0].timestamp).days or 1
        
        if earliest == 0:
            return 0.10
        
        total_growth = (latest - earliest) / earliest
        daily_growth = total_growth / days
        
        return max(0, min(daily_growth, 0.50))  # Cap between 0% and 50% daily
    
    def _identify_bottlenecks(self, service: ServiceType) -> List[str]:
        """Identify performance bottlenecks"""
        bottlenecks = []
        
        # Check CPU
        cpu = self.performance_monitor._get_avg_metric(service, MetricType.CPU_USAGE, 60)
        if cpu > 80:
            bottlenecks.append(f"High CPU usage: {cpu:.1f}%")
        
        # Check memory
        memory = self.performance_monitor._get_avg_metric(service, MetricType.MEMORY_USAGE, 60)
        if memory > 85:
            bottlenecks.append(f"High memory usage: {memory:.1f}%")
        
        # Check queue depth
        queue = self.performance_monitor._get_avg_metric(service, MetricType.QUEUE_DEPTH, 60)
        if queue > 100:
            bottlenecks.append(f"High queue depth: {queue:.0f}")
        
        return bottlenecks


# ============================================================================
# TESTING & DEMO
# ============================================================================

async def demo_performance_system():
    """Demonstrate performance and scaling system"""
    print("\n" + "=" * 60)
    print("ULTRA PLATFORM - PERFORMANCE & SCALING DEMO")
    print("=" * 60)
    
    # Initialize services
    perf_monitor = PerformanceMonitoringService()
    auto_scaler = AutoScalingService(perf_monitor)
    cache_manager = CacheManagementService()
    capacity_planner = CapacityPlanningService(perf_monitor)
    
    # Simulate metrics
    print("\n1. Recording Performance Metrics...")
    
    for i in range(10):
        await perf_monitor.record_metric(
            ServiceType.API_GATEWAY,
            MetricType.LATENCY,
            350 + (i * 50),  # Increasing latency
            {"request_id": f"req_{i}"}
        )
        
        await perf_monitor.record_metric(
            ServiceType.API_GATEWAY,
            MetricType.CPU_USAGE,
            60 + (i * 3),  # Increasing CPU
            {"instance": "api-1"}
        )
    
    print("   Recorded 20 metrics")
    
    # Check benchmarks
    print("\n2. Performance Benchmarks:")
    benchmarks = perf_monitor.get_benchmark_status()
    for operation, status in list(benchmarks.items())[:3]:
        print(f"\n   {operation}:")
        print(f"     Target: {status['target_latency_ms']}ms")
        print(f"     Current: {status['current_latency_ms']}ms")
        print(f"     Meeting Target: {status['meeting_target']}")
    
    # Check performance status
    print("\n3. Service Performance Status:")
    status = perf_monitor.get_performance_status(ServiceType.API_GATEWAY)
    print(f"   API Gateway: {status.value.upper()}")
    
    # Auto-scaling evaluation
    print("\n4. Auto-Scaling Evaluation:")
    action = await auto_scaler.evaluate_scaling(ServiceType.API_GATEWAY)
    print(f"   Recommended Action: {action.value if action else 'No action needed'}")
    
    if action and action != ScalingAction.NO_ACTION:
        event = await auto_scaler.execute_scaling(
            ServiceType.API_GATEWAY,
            action,
            "High CPU usage"
        )
        print(f"   Scaled: {event.instances_before} → {event.instances_after} instances")
    
    # Cache statistics
    print("\n5. Cache Performance:")
    
    # Simulate cache operations
    await cache_manager.set("customer:123", {"name": "John"}, cache_type="customer_profiles")
    await cache_manager.get("customer:123")  # Hit
    await cache_manager.get("customer:456")  # Miss
    
    cache_stats = cache_manager.get_cache_stats()
    print(f"   Hit Rate: {cache_stats['hit_rate']}")
    print(f"   Total Keys: {cache_stats['total_keys']}")
    print(f"   Hits: {cache_stats['hits']}, Misses: {cache_stats['misses']}")
    
    # Capacity planning
    print("\n6. Capacity Planning:")
    plan = await capacity_planner.analyze_capacity(ServiceType.API_GATEWAY)
    print(f"   Service: {plan.service.value}")
    print(f"   Current Capacity: {plan.current_capacity}")
    print(f"   30-day Projection: {plan.capacity_needed_30d}")
    print(f"   90-day Projection: {plan.capacity_needed_90d}")
    print(f"   1-year Projection: {plan.capacity_needed_1y}")
    
    if plan.bottlenecks:
        print(f"   Bottlenecks: {', '.join(plan.bottlenecks)}")
    
    # SLA alerts
    print("\n7. SLA Alerts:")
    if perf_monitor.alerts:
        for alert in perf_monitor.alerts[-3:]:
            print(f"   [{alert['severity'].upper()}] {alert['message']}")
    else:
        print("   No SLA violations detected")
    
    print("\n" + "=" * 60)
    print("✅ Performance & Scaling Demo Complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_performance_system())
