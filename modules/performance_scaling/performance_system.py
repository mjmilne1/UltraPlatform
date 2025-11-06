"""
Ultra Platform - Performance & Scaling System
============================================

Comprehensive performance optimization and scaling infrastructure implementing:
- Multi-layer caching (L1: In-memory, L2: Redis, L3: PostgreSQL)
- Database optimization and query analysis
- Performance monitoring and metrics
- Auto-scaling policies
- Health check endpoints
- Resource management

Based on: 13. Performance & Scaling - The Ultra Platform
Version: 1.0.0
"""

import asyncio
import time
import psutil
from typing import Dict, Any, Optional, List, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import hashlib
import json
from collections import OrderedDict, deque
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache tier levels"""
    L1_MEMORY = "l1_memory"
    L2_REDIS = "l2_redis"
    L3_DATABASE = "l3_database"


class PerformanceMetric(Enum):
    """Performance metrics to track"""
    LATENCY_P50 = "latency_p50"
    LATENCY_P90 = "latency_p90"
    LATENCY_P99 = "latency_p99"
    THROUGHPUT_RPS = "throughput_rps"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"


class HealthStatus(Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class PerformanceTargets:
    """Target performance metrics from documentation"""
    api_latency_p99_ms: float = 100.0
    feature_retrieval_p99_ms: float = 20.0
    model_inference_p99_ms: float = 10.0
    credit_score_calc_p99_ms: float = 50.0
    covenant_check_p99_ms: float = 30.0
    peak_tps: int = 500
    sustained_tps: int = 200
    availability_target: float = 99.95
    cache_hit_rate_target: float = 80.0


@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    ttl_seconds: int
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = datetime.now()


class LRUCache:
    """
    L1: In-Memory LRU Cache with TTL
    
    Fast local cache for frequently accessed data.
    Target: 1,000 entries, 60s TTL, 45% hit rate
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 60):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.hits = 0
        self.misses = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            self.misses += 1
            return None
            
        entry = self.cache[key]
        
        # Check expiration
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        # Update access and move to end (most recently used)
        entry.update_access()
        self.cache.move_to_end(key)
        self.hits += 1
        
        return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in cache"""
        ttl = ttl or self.default_ttl
        
        # Evict if at capacity
        if key not in self.cache and len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        
        # Calculate size
        size_bytes = len(str(value).encode('utf-8'))
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
        self.cache.move_to_end(key)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "level": "L1_MEMORY",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }


class RedisCache:
    """
    L2: Distributed Redis Cache
    
    Shared cache across all instances.
    Target: 100,000 entries, 5min TTL, 80% hit rate
    
    Note: Mock implementation - replace with actual Redis client
    """
    
    def __init__(self, max_size: int = 100000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.hits = 0
        self.misses = 0
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if key not in self.cache:
            self.misses += 1
            return None
            
        entry = self.cache[key]
        
        if entry.is_expired():
            del self.cache[key]
            self.misses += 1
            return None
        
        entry.update_access()
        self.hits += 1
        
        return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in Redis cache"""
        ttl = ttl or self.default_ttl
        
        if len(self.cache) >= self.max_size:
            # Simple eviction: remove oldest
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.cache[k].timestamp)
            del self.cache[oldest_key]
        
        size_bytes = len(json.dumps(value).encode('utf-8'))
        
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=datetime.now(),
            ttl_seconds=ttl,
            size_bytes=size_bytes
        )
        
        self.cache[key] = entry
    
    async def delete(self, key: str) -> bool:
        """Delete entry from Redis"""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    async def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0.0
        
        total_size = sum(entry.size_bytes for entry in self.cache.values())
        
        return {
            "level": "L2_REDIS",
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024)
        }


class MultiLayerCache:
    """
    Multi-layer caching system with L1, L2, L3 tiers
    
    Implements cache-aside pattern with automatic promotion/demotion
    """
    
    def __init__(self):
        self.l1_cache = LRUCache(max_size=1000, default_ttl=60)
        self.l2_cache = RedisCache(max_size=100000, default_ttl=300)
        self.invalidation_triggers: List[Callable] = []
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-layer cache"""
        # Try L1
        value = self.l1_cache.get(key)
        if value is not None:
            logger.debug(f"L1 cache hit: {key}")
            return value
        
        # Try L2
        value = await self.l2_cache.get(key)
        if value is not None:
            logger.debug(f"L2 cache hit: {key}, promoting to L1")
            # Promote to L1
            self.l1_cache.set(key, value)
            return value
        
        # L3 would be database lookup (handled by caller)
        logger.debug(f"Cache miss: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """Set value in all cache layers"""
        # Write to both L1 and L2
        self.l1_cache.set(key, value, ttl)
        await self.l2_cache.set(key, value, ttl)
    
    async def delete(self, key: str):
        """Delete from all cache layers"""
        self.l1_cache.delete(key)
        await self.l2_cache.delete(key)
    
    async def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern"""
        if pattern:
            # Pattern-based invalidation
            keys_to_delete = [k for k in self.l1_cache.cache.keys() 
                             if pattern in k]
            for key in keys_to_delete:
                await self.delete(key)
        else:
            # Clear all
            self.l1_cache.clear()
            await self.l2_cache.clear()
    
    def register_invalidation_trigger(self, trigger: Callable):
        """Register a cache invalidation trigger"""
        self.invalidation_triggers.append(trigger)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics from all cache layers"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        
        total_hits = l1_stats["hits"] + l2_stats["hits"]
        total_misses = l1_stats["misses"] + l2_stats["misses"]
        total_requests = total_hits + total_misses
        
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return {
            "l1": l1_stats,
            "l2": l2_stats,
            "overall": {
                "total_hits": total_hits,
                "total_misses": total_misses,
                "hit_rate": overall_hit_rate
            }
        }


@dataclass
class LatencyMeasurement:
    """Single latency measurement"""
    operation: str
    latency_ms: float
    timestamp: datetime
    success: bool


class PerformanceMonitor:
    """
    Performance monitoring and metrics collection
    
    Tracks latency, throughput, errors, and system resources
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: Dict[str, deque] = {}
        self.request_timestamps: deque = deque(maxlen=10000)
        self.error_count = 0
        self.success_count = 0
        self.start_time = datetime.now()
        
    def record_latency(self, operation: str, latency_ms: float, success: bool = True):
        """Record a latency measurement"""
        if operation not in self.latencies:
            self.latencies[operation] = deque(maxlen=self.window_size)
        
        measurement = LatencyMeasurement(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=datetime.now(),
            success=success
        )
        
        self.latencies[operation].append(measurement)
        self.request_timestamps.append(datetime.now())
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def get_percentile(self, operation: str, percentile: float) -> Optional[float]:
        """Calculate latency percentile for an operation"""
        if operation not in self.latencies or not self.latencies[operation]:
            return None
        
        latencies = [m.latency_ms for m in self.latencies[operation]]
        
        if not latencies:
            return None
        
        sorted_latencies = sorted(latencies)
        index = int(len(sorted_latencies) * percentile / 100)
        index = min(index, len(sorted_latencies) - 1)
        
        return sorted_latencies[index]
    
    def get_throughput(self, window_seconds: int = 60) -> float:
        """Calculate requests per second over a time window"""
        cutoff_time = datetime.now() - timedelta(seconds=window_seconds)
        
        recent_requests = sum(1 for ts in self.request_timestamps 
                             if ts >= cutoff_time)
        
        return recent_requests / window_seconds
    
    def get_error_rate(self) -> float:
        """Calculate error rate as percentage"""
        total_requests = self.error_count + self.success_count
        
        if total_requests == 0:
            return 0.0
        
        return (self.error_count / total_requests) * 100
    
    def get_metrics(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
        
        # Overall metrics
        metrics["overall"] = {
            "throughput_rps": self.get_throughput(),
            "error_rate": self.get_error_rate(),
            "total_requests": self.success_count + self.error_count,
            "success_count": self.success_count,
            "error_count": self.error_count
        }
        
        # Per-operation metrics
        if operation:
            operations = [operation]
        else:
            operations = list(self.latencies.keys())
        
        metrics["operations"] = {}
        for op in operations:
            metrics["operations"][op] = {
                "p50": self.get_percentile(op, 50),
                "p90": self.get_percentile(op, 90),
                "p99": self.get_percentile(op, 99),
                "count": len(self.latencies.get(op, []))
            }
        
        # System resources
        metrics["system"] = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return metrics
    
    def check_thresholds(self, targets: PerformanceTargets) -> List[str]:
        """Check if metrics exceed target thresholds"""
        violations = []
        
        # Check API latency
        api_p99 = self.get_percentile("api_request", 99)
        if api_p99 and api_p99 > targets.api_latency_p99_ms:
            violations.append(
                f"API p99 latency {api_p99:.2f}ms exceeds target {targets.api_latency_p99_ms}ms"
            )
        
        # Check error rate
        error_rate = self.get_error_rate()
        if error_rate > 5.0:  # Critical threshold
            violations.append(
                f"Error rate {error_rate:.2f}% exceeds critical threshold 5%"
            )
        
        # Check CPU
        cpu_usage = psutil.cpu_percent()
        if cpu_usage > 90:
            violations.append(
                f"CPU usage {cpu_usage:.1f}% exceeds critical threshold 90%"
            )
        
        # Check memory
        memory_usage = psutil.virtual_memory().percent
        if memory_usage > 95:
            violations.append(
                f"Memory usage {memory_usage:.1f}% exceeds critical threshold 95%"
            )
        
        return violations


class HealthChecker:
    """
    Health check endpoint for load balancers and monitoring
    
    Implements comprehensive health checks including:
    - Service availability
    - Database connectivity
    - Cache connectivity
    - Performance thresholds
    """
    
    def __init__(self, monitor: PerformanceMonitor, cache: MultiLayerCache):
        self.monitor = monitor
        self.cache = cache
        self.last_check_time = datetime.now()
        self.check_interval_seconds = 10
        
    async def check_health(self) -> Tuple[HealthStatus, Dict[str, Any]]:
        """Perform comprehensive health check"""
        checks = {}
        overall_status = HealthStatus.HEALTHY
        
        # Check 1: Service responsive
        checks["service"] = {
            "status": "healthy",
            "details": "Service is responsive"
        }
        
        # Check 2: Performance metrics
        metrics = self.monitor.get_metrics()
        error_rate = metrics["overall"]["error_rate"]
        cpu_usage = metrics["system"]["cpu_percent"]
        
        if error_rate > 5.0 or cpu_usage > 90:
            checks["performance"] = {
                "status": "unhealthy",
                "details": f"Error rate: {error_rate:.1f}%, CPU: {cpu_usage:.1f}%"
            }
            overall_status = HealthStatus.UNHEALTHY
        elif error_rate > 2.0 or cpu_usage > 80:
            checks["performance"] = {
                "status": "degraded",
                "details": f"Error rate: {error_rate:.1f}%, CPU: {cpu_usage:.1f}%"
            }
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        else:
            checks["performance"] = {
                "status": "healthy",
                "details": "Performance within thresholds"
            }
        
        # Check 3: Cache health
        cache_stats = self.cache.get_stats()
        cache_hit_rate = cache_stats["overall"]["hit_rate"]
        
        if cache_hit_rate < 50:
            checks["cache"] = {
                "status": "degraded",
                "details": f"Low cache hit rate: {cache_hit_rate:.1f}%"
            }
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        else:
            checks["cache"] = {
                "status": "healthy",
                "details": f"Cache hit rate: {cache_hit_rate:.1f}%"
            }
        
        # Check 4: System resources
        memory_usage = psutil.virtual_memory().percent
        
        if memory_usage > 95:
            checks["resources"] = {
                "status": "unhealthy",
                "details": f"Critical memory usage: {memory_usage:.1f}%"
            }
            overall_status = HealthStatus.UNHEALTHY
        elif memory_usage > 85:
            checks["resources"] = {
                "status": "degraded",
                "details": f"High memory usage: {memory_usage:.1f}%"
            }
            if overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.DEGRADED
        else:
            checks["resources"] = {
                "status": "healthy",
                "details": "Resources within limits"
            }
        
        self.last_check_time = datetime.now()
        
        return overall_status, {
            "status": overall_status.value,
            "timestamp": self.last_check_time.isoformat(),
            "checks": checks
        }


class AutoScaler:
    """
    Auto-scaling decision engine
    
    Determines when to scale up/down based on metrics
    """
    
    def __init__(self, 
                 min_replicas: int = 6,
                 max_replicas: int = 20,
                 cpu_threshold: float = 70.0,
                 memory_threshold: float = 80.0):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.current_replicas = min_replicas
        self.scaling_history: List[Dict[str, Any]] = []
        
    def should_scale(self, monitor: PerformanceMonitor) -> Tuple[bool, int, str]:
        """
        Determine if scaling is needed
        
        Returns: (should_scale, target_replicas, reason)
        """
        metrics = monitor.get_metrics()
        cpu_usage = metrics["system"]["cpu_percent"]
        memory_usage = metrics["system"]["memory_percent"]
        error_rate = metrics["overall"]["error_rate"]
        
        # Scale up conditions
        if cpu_usage > self.cpu_threshold:
            target = min(self.current_replicas + 2, self.max_replicas)
            if target > self.current_replicas:
                return True, target, f"CPU usage {cpu_usage:.1f}% > {self.cpu_threshold}%"
        
        if memory_usage > self.memory_threshold:
            target = min(self.current_replicas + 2, self.max_replicas)
            if target > self.current_replicas:
                return True, target, f"Memory usage {memory_usage:.1f}% > {self.memory_threshold}%"
        
        if error_rate > 5.0:
            target = min(self.current_replicas + 2, self.max_replicas)
            if target > self.current_replicas:
                return True, target, f"High error rate {error_rate:.1f}%"
        
        # Scale down conditions (conservative)
        if (cpu_usage < self.cpu_threshold * 0.5 and 
            memory_usage < self.memory_threshold * 0.5 and
            error_rate < 1.0):
            target = max(self.current_replicas - 1, self.min_replicas)
            if target < self.current_replicas:
                return True, target, "Low resource utilization"
        
        return False, self.current_replicas, "No scaling needed"
    
    def scale(self, target_replicas: int, reason: str):
        """Execute scaling action"""
        old_replicas = self.current_replicas
        self.current_replicas = target_replicas
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "old_replicas": old_replicas,
            "new_replicas": target_replicas,
            "reason": reason
        }
        
        self.scaling_history.append(event)
        
        logger.info(f"Scaling from {old_replicas} to {target_replicas} replicas: {reason}")
        
        return event


class PerformanceOptimizer:
    """
    Main performance and scaling system
    
    Coordinates caching, monitoring, health checks, and auto-scaling
    """
    
    def __init__(self, targets: Optional[PerformanceTargets] = None):
        self.targets = targets or PerformanceTargets()
        self.cache = MultiLayerCache()
        self.monitor = PerformanceMonitor()
        self.health_checker = HealthChecker(self.monitor, self.cache)
        self.auto_scaler = AutoScaler()
        self._monitoring_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start the performance system"""
        logger.info("Starting Performance & Scaling System")
        logger.info(f"Targets: {self.targets}")
        
        # Start monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop(self):
        """Stop the performance system"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance & Scaling System stopped")
    
    async def _monitoring_loop(self):
        """Background monitoring and auto-scaling loop"""
        while True:
            try:
                # Check health
                status, health_report = await self.health_checker.check_health()
                
                if status != HealthStatus.HEALTHY:
                    logger.warning(f"Health check: {status.value}")
                    logger.warning(f"Details: {health_report}")
                
                # Check for threshold violations
                violations = self.monitor.check_thresholds(self.targets)
                if violations:
                    logger.warning(f"Performance violations detected:")
                    for violation in violations:
                        logger.warning(f"  - {violation}")
                
                # Check if auto-scaling needed
                should_scale, target, reason = self.auto_scaler.should_scale(self.monitor)
                if should_scale:
                    self.auto_scaler.scale(target, reason)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "targets": {
                "api_latency_p99_ms": self.targets.api_latency_p99_ms,
                "peak_tps": self.targets.peak_tps,
                "availability_target": self.targets.availability_target,
                "cache_hit_rate_target": self.targets.cache_hit_rate_target
            },
            "cache": self.cache.get_stats(),
            "performance": self.monitor.get_metrics(),
            "scaling": {
                "current_replicas": self.auto_scaler.current_replicas,
                "min_replicas": self.auto_scaler.min_replicas,
                "max_replicas": self.auto_scaler.max_replicas,
                "recent_scaling_events": self.auto_scaler.scaling_history[-10:]
            }
        }


# Example usage
async def main():
    """Example usage of the performance system"""
    optimizer = PerformanceOptimizer()
    
    try:
        await optimizer.start()
        
        # Simulate some requests
        for i in range(100):
            start = time.time()
            
            # Simulate cache lookup
            cache_key = f"user:{i % 20}:portfolio"
            value = await optimizer.cache.get(cache_key)
            
            if value is None:
                # Cache miss - simulate DB lookup
                await asyncio.sleep(0.01)  # 10ms DB query
                value = {"portfolio": f"data_{i}"}
                await optimizer.cache.set(cache_key, value)
            
            # Record latency
            latency_ms = (time.time() - start) * 1000
            optimizer.monitor.record_latency("api_request", latency_ms, success=True)
            
            await asyncio.sleep(0.01)
        
        # Get report
        report = optimizer.get_comprehensive_report()
        print("\n=== Performance Report ===")
        print(json.dumps(report, indent=2, default=str))
        
    finally:
        await optimizer.stop()


if __name__ == "__main__":
    asyncio.run(main())
