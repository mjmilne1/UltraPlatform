"""
Tests for Ultra Platform Performance & Scaling System
====================================================

Comprehensive test suite covering:
- Multi-layer caching (L1, L2)
- Performance monitoring
- Health checks
- Auto-scaling logic
- Metrics collection
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta

# Import classes to test - FIXED IMPORT
from modules.performance_scaling.performance_system import (
    LRUCache,
    RedisCache,
    MultiLayerCache,
    PerformanceMonitor,
    HealthChecker,
    AutoScaler,
    PerformanceOptimizer,
    PerformanceTargets,
    HealthStatus,
    CacheEntry
)


class TestLRUCache:
    """Tests for L1 in-memory LRU cache"""
    
    def test_cache_initialization(self):
        """Test cache initialization with defaults"""
        cache = LRUCache(max_size=100, default_ttl=60)
        
        assert cache.max_size == 100
        assert cache.default_ttl == 60
        assert len(cache.cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_cache_set_and_get(self):
        """Test basic set and get operations"""
        cache = LRUCache()
        
        cache.set("key1", "value1")
        value = cache.get("key1")
        
        assert value == "value1"
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_cache_miss(self):
        """Test cache miss behavior"""
        cache = LRUCache()
        
        value = cache.get("nonexistent")
        
        assert value is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_cache_ttl_expiration(self):
        """Test TTL expiration"""
        cache = LRUCache(default_ttl=1)  # 1 second TTL
        
        cache.set("key1", "value1")
        
        # Should hit immediately
        value1 = cache.get("key1")
        assert value1 == "value1"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should miss after expiration
        value2 = cache.get("key1")
        assert value2 is None
        assert cache.misses == 1
    
    def test_cache_stats(self):
        """Test cache statistics"""
        cache = LRUCache(max_size=100)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["level"] == "L1_MEMORY"
        assert stats["size"] == 1
        assert stats["max_size"] == 100
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 50.0


class TestRedisCache:
    """Tests for L2 Redis cache"""
    
    @pytest.mark.asyncio
    async def test_redis_cache_initialization(self):
        """Test Redis cache initialization"""
        cache = RedisCache(max_size=1000, default_ttl=300)
        
        assert cache.max_size == 1000
        assert cache.default_ttl == 300
        assert len(cache.cache) == 0
    
    @pytest.mark.asyncio
    async def test_redis_set_and_get(self):
        """Test Redis set and get operations"""
        cache = RedisCache()
        
        await cache.set("key1", {"data": "value1"})
        value = await cache.get("key1")
        
        assert value == {"data": "value1"}
        assert cache.hits == 1


class TestMultiLayerCache:
    """Tests for multi-layer cache system"""
    
    @pytest.mark.asyncio
    async def test_multilayer_initialization(self):
        """Test multi-layer cache initialization"""
        cache = MultiLayerCache()
        
        assert cache.l1_cache is not None
        assert cache.l2_cache is not None
    
    @pytest.mark.asyncio
    async def test_multilayer_l1_hit(self):
        """Test L1 cache hit"""
        cache = MultiLayerCache()
        
        await cache.set("key1", "value1")
        value = await cache.get("key1")
        
        assert value == "value1"
        assert cache.l1_cache.hits > 0
    
    @pytest.mark.asyncio
    async def test_multilayer_stats(self):
        """Test multi-layer statistics"""
        cache = MultiLayerCache()
        
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert "l1" in stats
        assert "l2" in stats
        assert "overall" in stats
        assert stats["overall"]["total_hits"] >= 1


class TestPerformanceMonitor:
    """Tests for performance monitoring"""
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        monitor = PerformanceMonitor(window_size=100)
        
        assert monitor.window_size == 100
        assert monitor.error_count == 0
        assert monitor.success_count == 0
    
    def test_record_latency(self):
        """Test latency recording"""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("api_request", 50.0, success=True)
        monitor.record_latency("api_request", 75.0, success=True)
        
        assert monitor.success_count == 2
        assert "api_request" in monitor.latencies
        assert len(monitor.latencies["api_request"]) == 2
    
    def test_get_percentile(self):
        """Test percentile calculation"""
        monitor = PerformanceMonitor()
        
        # Record latencies: 10, 20, 30, 40, 50
        for i in range(1, 6):
            monitor.record_latency("test_op", i * 10.0)
        
        p50 = monitor.get_percentile("test_op", 50)
        p90 = monitor.get_percentile("test_op", 90)
        p99 = monitor.get_percentile("test_op", 99)
        
        assert p50 == 30.0  # Median
        assert p90 == 50.0  # 90th percentile
        assert p99 == 50.0  # 99th percentile
    
    def test_get_error_rate(self):
        """Test error rate calculation"""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("api", 10.0, success=True)
        monitor.record_latency("api", 10.0, success=True)
        monitor.record_latency("api", 10.0, success=False)
        monitor.record_latency("api", 10.0, success=False)
        
        error_rate = monitor.get_error_rate()
        
        assert error_rate == 50.0  # 2 errors out of 4 requests
    
    def test_get_metrics(self):
        """Test comprehensive metrics"""
        monitor = PerformanceMonitor()
        
        monitor.record_latency("api_request", 50.0, success=True)
        monitor.record_latency("db_query", 10.0, success=True)
        
        metrics = monitor.get_metrics()
        
        assert "timestamp" in metrics
        assert "overall" in metrics
        assert "operations" in metrics
        assert "system" in metrics
        assert "api_request" in metrics["operations"]


class TestHealthChecker:
    """Tests for health check system"""
    
    @pytest.mark.asyncio
    async def test_health_checker_initialization(self):
        """Test health checker initialization"""
        monitor = PerformanceMonitor()
        cache = MultiLayerCache()
        checker = HealthChecker(monitor, cache)
        
        assert checker.monitor == monitor
        assert checker.cache == cache
    
    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test healthy status"""
        monitor = PerformanceMonitor()
        cache = MultiLayerCache()
        checker = HealthChecker(monitor, cache)
        
        # Record good metrics
        for _ in range(10):
            monitor.record_latency("api", 10.0, success=True)
        
        status, report = await checker.check_health()
        
        assert status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        assert "checks" in report
        assert "status" in report


class TestAutoScaler:
    """Tests for auto-scaling logic"""
    
    def test_autoscaler_initialization(self):
        """Test auto-scaler initialization"""
        scaler = AutoScaler(min_replicas=6, max_replicas=20)
        
        assert scaler.min_replicas == 6
        assert scaler.max_replicas == 20
        assert scaler.current_replicas == 6
    
    def test_scale_action(self):
        """Test scale action"""
        scaler = AutoScaler(min_replicas=5, max_replicas=10)
        scaler.current_replicas = 5
        
        event = scaler.scale(8, "High CPU usage")
        
        assert scaler.current_replicas == 8
        assert event["old_replicas"] == 5
        assert event["new_replicas"] == 8
        assert event["reason"] == "High CPU usage"
        assert len(scaler.scaling_history) == 1


class TestPerformanceOptimizer:
    """Tests for main performance optimizer"""
    
    @pytest.mark.asyncio
    async def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = PerformanceOptimizer()
        
        assert optimizer.cache is not None
        assert optimizer.monitor is not None
        assert optimizer.health_checker is not None
        assert optimizer.auto_scaler is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_report(self):
        """Test comprehensive report generation"""
        optimizer = PerformanceOptimizer()
        
        # Record some activity
        await optimizer.cache.set("key1", "value1")
        optimizer.monitor.record_latency("api", 10.0, success=True)
        
        report = optimizer.get_comprehensive_report()
        
        assert "timestamp" in report
        assert "targets" in report
        assert "cache" in report
        assert "performance" in report
        assert "scaling" in report


class TestIntegration:
    """Integration tests for complete system"""
    
    @pytest.mark.asyncio
    async def test_full_request_cycle(self):
        """Test complete request handling cycle"""
        optimizer = PerformanceOptimizer()
        
        # Simulate request
        start = time.time()
        
        # Try cache
        value = await optimizer.cache.get("user:123:portfolio")
        
        if value is None:
            # Simulate DB lookup
            await asyncio.sleep(0.01)
            value = {"portfolio": "data"}
            await optimizer.cache.set("user:123:portfolio", value)
        
        # Record metrics
        latency_ms = (time.time() - start) * 1000
        optimizer.monitor.record_latency("api_request", latency_ms, success=True)
        
        # Verify
        cached_value = await optimizer.cache.get("user:123:portfolio")
        assert cached_value == value
        
        metrics = optimizer.monitor.get_metrics()
        assert metrics["overall"]["success_count"] == 1


def test_performance_targets():
    """Test PerformanceTargets dataclass"""
    targets = PerformanceTargets()
    
    assert targets.api_latency_p99_ms == 100.0
    assert targets.peak_tps == 500
    assert targets.sustained_tps == 200
    assert targets.availability_target == 99.95
    assert targets.cache_hit_rate_target == 80.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
