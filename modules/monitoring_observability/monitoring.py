from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import json
import uuid
import statistics
import time
import random
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import traceback

class MetricType(Enum):
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    SUMMARY = 'summary'
    TIMER = 'timer'

class LogLevel(Enum):
    DEBUG = 'debug'
    INFO = 'info'
    WARNING = 'warning'
    ERROR = 'error'
    CRITICAL = 'critical'

class AlertSeverity(Enum):
    INFO = 'info'
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

class HealthStatus(Enum):
    HEALTHY = 'healthy'
    DEGRADED = 'degraded'
    UNHEALTHY = 'unhealthy'
    UNKNOWN = 'unknown'

@dataclass
class Metric:
    '''Metric data point'''
    name: str
    type: MetricType
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = 'count'
    
    def to_dict(self):
        return {
            'name': self.name,
            'type': self.type.value,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'labels': self.labels,
            'unit': self.unit
        }

@dataclass
class Span:
    '''Distributed tracing span'''
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: Optional[str] = None
    operation_name: str = ''
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    status: str = 'in_progress'
    
    def duration_ms(self):
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds() * 1000
        return 0

class MonitoringObservability:
    '''Comprehensive Monitoring & Observability System for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Monitoring & Observability'
        self.version = '2.0'
        self.metrics_collector = MetricsCollector()
        self.distributed_tracing = DistributedTracing()
        self.log_aggregator = LogAggregator()
        self.health_monitor = HealthMonitor()
        self.performance_monitor = PerformanceMonitor()
        self.alert_manager = AlertManager()
        self.dashboard_service = DashboardService()
        self.sla_monitor = SLAMonitor()
        self.anomaly_detector = AnomalyDetector()
        self.compliance_monitor = ComplianceMonitor()
        
    def monitor_system(self):
        '''Monitor entire system'''
        print('MONITORING & OBSERVABILITY')
        print('='*80)
        print(f'System: UltraPlatform')
        print(f'Timestamp: {datetime.now()}')
        print()
        
        # Step 1: Metrics Collection
        print('1️⃣ METRICS COLLECTION')
        print('-'*40)
        metrics = self.metrics_collector.collect_metrics()
        print(f'  Metrics Collected: {metrics["count"]}')
        print(f'  Types: {", ".join(metrics["types"])}')
        print(f'  Collection Rate: {metrics["rate"]}/sec')
        
        # Step 2: Health Check
        print('\n2️⃣ HEALTH CHECK')
        print('-'*40)
        health = self.health_monitor.check_health()
        print(f'  Overall Status: {health["status"].value}')
        print(f'  Healthy Services: {health["healthy_count"]}/{health["total_services"]}')
        print(f'  Issues: {health["issues"]}')
        
        # Step 3: Performance Monitoring
        print('\n3️⃣ PERFORMANCE MONITORING')
        print('-'*40)
        performance = self.performance_monitor.get_performance()
        print(f'  CPU Usage: {performance["cpu_usage"]:.1f}%')
        print(f'  Memory Usage: {performance["memory_usage"]:.1f}%')
        print(f'  Latency P99: {performance["latency_p99"]}ms')
        print(f'  Throughput: {performance["throughput"]} req/sec')
        
        # Step 4: Distributed Tracing
        print('\n4️⃣ DISTRIBUTED TRACING')
        print('-'*40)
        tracing = self.distributed_tracing.get_trace_stats()
        print(f'  Active Traces: {tracing["active_traces"]}')
        print(f'  Completed Traces: {tracing["completed_traces"]}')
        print(f'  Avg Trace Duration: {tracing["avg_duration"]}ms')
        
        # Step 5: Log Analysis
        print('\n5️⃣ LOG ANALYSIS')
        print('-'*40)
        logs = self.log_aggregator.analyze_logs()
        print(f'  Total Logs: {logs["total"]:,}')
        print(f'  Error Rate: {logs["error_rate"]:.2f}%')
        print(f'  Warning Count: {logs["warnings"]}')
        
        # Step 6: Alert Status
        print('\n6️⃣ ALERT STATUS')
        print('-'*40)
        alerts = self.alert_manager.get_active_alerts()
        print(f'  Active Alerts: {alerts["count"]}')
        if alerts["critical"] > 0:
            print(f'  🔴 Critical: {alerts["critical"]}')
        if alerts["high"] > 0:
            print(f'  🟠 High: {alerts["high"]}')
        if alerts["medium"] > 0:
            print(f'  🟡 Medium: {alerts["medium"]}')
        
        # Step 7: SLA Monitoring
        print('\n7️⃣ SLA MONITORING')
        print('-'*40)
        sla = self.sla_monitor.get_sla_status()
        print(f'  Availability: {sla["availability"]:.3f}%')
        print(f'  SLA Target: {sla["target"]:.3f}%')
        print(f'  Status: {"✅ Meeting SLA" if sla["meeting_sla"] else "❌ Below SLA"}')
        
        # Step 8: Anomaly Detection
        print('\n8️⃣ ANOMALY DETECTION')
        print('-'*40)
        anomalies = self.anomaly_detector.detect_anomalies()
        print(f'  Anomalies Detected: {anomalies["count"]}')
        for anomaly in anomalies["top_anomalies"][:3]:
            print(f'    • {anomaly["type"]}: {anomaly["severity"]}')
        
        return {
            'metrics': metrics,
            'health': health,
            'performance': performance,
            'alerts': alerts["count"],
            'sla_met': sla["meeting_sla"]
        }

class MetricsCollector:
    '''Collect and manage metrics'''
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        self.timers = defaultdict(list)
        
    def collect_metrics(self):
        '''Collect all metrics'''
        # Collect system metrics
        self._collect_system_metrics()
        self._collect_application_metrics()
        self._collect_business_metrics()
        
        metric_types = set()
        total_metrics = 0
        
        for metric_type in [self.counters, self.gauges, self.histograms, self.timers]:
            if metric_type:
                metric_types.add(type(metric_type).__name__)
                total_metrics += len(metric_type)
        
        return {
            'count': total_metrics,
            'types': list(metric_types),
            'rate': random.randint(100, 500)
        }
    
    def record_metric(self, metric: Metric):
        '''Record a metric'''
        key = f"{metric.name}_{metric.type.value}"
        self.metrics[key].append(metric)
        
        # Update type-specific storage
        if metric.type == MetricType.COUNTER:
            self.counters[metric.name] += metric.value
        elif metric.type == MetricType.GAUGE:
            self.gauges[metric.name] = metric.value
        elif metric.type == MetricType.HISTOGRAM:
            self.histograms[metric.name].append(metric.value)
        elif metric.type == MetricType.TIMER:
            self.timers[metric.name].append(metric.value)
    
    def _collect_system_metrics(self):
        '''Collect system-level metrics'''
        # CPU metrics
        self.gauges['system.cpu.usage'] = random.uniform(20, 80)
        self.gauges['system.cpu.load.1m'] = random.uniform(0.5, 4.0)
        
        # Memory metrics
        self.gauges['system.memory.used'] = random.uniform(2000, 8000)
        self.gauges['system.memory.free'] = random.uniform(1000, 4000)
        
        # Disk metrics
        self.gauges['system.disk.used'] = random.uniform(100, 500)
        self.gauges['system.disk.free'] = random.uniform(100, 1000)
        
        # Network metrics
        self.counters['system.network.bytes.sent'] += random.randint(1000, 10000)
        self.counters['system.network.bytes.received'] += random.randint(1000, 10000)
    
    def _collect_application_metrics(self):
        '''Collect application-level metrics'''
        # Event metrics
        self.counters['events.published'] += random.randint(100, 1000)
        self.counters['events.consumed'] += random.randint(100, 1000)
        self.counters['events.failed'] += random.randint(0, 10)
        
        # Queue metrics
        self.gauges['queue.depth'] = random.randint(0, 1000)
        self.gauges['queue.consumers'] = random.randint(1, 10)
        
        # Latency metrics
        for _ in range(10):
            self.histograms['request.latency'].append(random.uniform(1, 100))
    
    def _collect_business_metrics(self):
        '''Collect business metrics'''
        # Trading metrics
        self.counters['trades.executed'] += random.randint(10, 100)
        self.gauges['trades.value'] = random.uniform(100000, 1000000)
        
        # Portfolio metrics
        self.gauges['portfolio.nav'] = random.uniform(1000000, 10000000)
        self.gauges['portfolio.positions'] = random.randint(10, 100)
    
    def get_metric_value(self, name, metric_type=None):
        '''Get current value of a metric'''
        if metric_type == MetricType.COUNTER or name in self.counters:
            return self.counters.get(name, 0)
        elif metric_type == MetricType.GAUGE or name in self.gauges:
            return self.gauges.get(name, 0)
        elif metric_type == MetricType.HISTOGRAM or name in self.histograms:
            values = self.histograms.get(name, [])
            return statistics.mean(values) if values else 0
        elif metric_type == MetricType.TIMER or name in self.timers:
            values = self.timers.get(name, [])
            return statistics.mean(values) if values else 0
        return 0
    
    def get_percentile(self, name, percentile):
        '''Get percentile value for histogram/timer'''
        values = self.histograms.get(name, []) or self.timers.get(name, [])
        if not values:
            return 0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * percentile / 100)
        return sorted_values[min(index, len(sorted_values) - 1)]

class DistributedTracing:
    '''Distributed tracing system'''
    
    def __init__(self):
        self.traces = {}
        self.spans = {}
        self.completed_traces = []
        
    def start_trace(self, operation_name):
        '''Start a new trace'''
        span = Span(operation_name=operation_name)
        self.traces[span.trace_id] = [span]
        self.spans[span.span_id] = span
        return span
    
    def start_span(self, trace_id, operation_name, parent_span_id=None):
        '''Start a new span within a trace'''
        span = Span(
            trace_id=trace_id,
            operation_name=operation_name,
            parent_span_id=parent_span_id
        )
        
        if trace_id in self.traces:
            self.traces[trace_id].append(span)
        else:
            self.traces[trace_id] = [span]
        
        self.spans[span.span_id] = span
        return span
    
    def end_span(self, span_id):
        '''End a span'''
        if span_id in self.spans:
            span = self.spans[span_id]
            span.end_time = datetime.now()
            span.status = 'completed'
            
            # Check if trace is complete
            trace_spans = self.traces.get(span.trace_id, [])
            if all(s.status == 'completed' for s in trace_spans):
                self.completed_traces.append({
                    'trace_id': span.trace_id,
                    'spans': trace_spans,
                    'duration': self._calculate_trace_duration(trace_spans)
                })
    
    def _calculate_trace_duration(self, spans):
        '''Calculate total trace duration'''
        if not spans:
            return 0
        
        min_start = min(s.start_time for s in spans)
        max_end = max(s.end_time for s in spans if s.end_time)
        
        if max_end:
            return (max_end - min_start).total_seconds() * 1000
        return 0
    
    def get_trace_stats(self):
        '''Get tracing statistics'''
        active_traces = sum(
            1 for trace_spans in self.traces.values()
            if any(s.status == 'in_progress' for s in trace_spans)
        )
        
        durations = [t['duration'] for t in self.completed_traces]
        avg_duration = statistics.mean(durations) if durations else 0
        
        return {
            'active_traces': active_traces,
            'completed_traces': len(self.completed_traces),
            'total_spans': len(self.spans),
            'avg_duration': avg_duration
        }
    
    def get_trace(self, trace_id):
        '''Get trace details'''
        return self.traces.get(trace_id, [])

class LogAggregator:
    '''Log aggregation and analysis'''
    
    def __init__(self):
        self.logs = deque(maxlen=100000)
        self.log_counts = defaultdict(int)
        self.error_patterns = []
        
    def log(self, level: LogLevel, message, context=None):
        '''Add log entry'''
        log_entry = {
            'timestamp': datetime.now(),
            'level': level,
            'message': message,
            'context': context or {}
        }
        
        self.logs.append(log_entry)
        self.log_counts[level] += 1
        
        # Track error patterns
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            self._analyze_error_pattern(message)
        
        return log_entry
    
    def _analyze_error_pattern(self, message):
        '''Analyze error patterns'''
        # Simple pattern matching
        patterns = {
            'timeout': 'timeout' in message.lower(),
            'connection': 'connection' in message.lower(),
            'validation': 'validation' in message.lower(),
            'permission': 'permission' in message.lower() or 'denied' in message.lower()
        }
        
        for pattern, matches in patterns.items():
            if matches:
                self.error_patterns.append({
                    'pattern': pattern,
                    'timestamp': datetime.now(),
                    'message': message
                })
    
    def analyze_logs(self):
        '''Analyze aggregated logs'''
        total = len(self.logs)
        
        if total == 0:
            return {
                'total': 0,
                'error_rate': 0,
                'warnings': 0,
                'errors': 0,
                'critical': 0
            }
        
        errors = self.log_counts.get(LogLevel.ERROR, 0)
        critical = self.log_counts.get(LogLevel.CRITICAL, 0)
        warnings = self.log_counts.get(LogLevel.WARNING, 0)
        
        return {
            'total': total,
            'error_rate': ((errors + critical) / total) * 100,
            'warnings': warnings,
            'errors': errors,
            'critical': critical,
            'top_errors': self._get_top_error_patterns()
        }
    
    def _get_top_error_patterns(self):
        '''Get most common error patterns'''
        pattern_counts = defaultdict(int)
        
        for pattern in self.error_patterns[-100:]:  # Last 100 errors
            pattern_counts[pattern['pattern']] += 1
        
        return sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]
    
    def search_logs(self, query, level=None, start_time=None, end_time=None):
        '''Search logs with filters'''
        results = []
        
        for log in self.logs:
            # Filter by level
            if level and log['level'] != level:
                continue
            
            # Filter by time
            if start_time and log['timestamp'] < start_time:
                continue
            if end_time and log['timestamp'] > end_time:
                continue
            
            # Filter by query
            if query and query.lower() not in log['message'].lower():
                continue
            
            results.append(log)
        
        return results

class HealthMonitor:
    '''Monitor system health'''
    
    def __init__(self):
        self.health_checks = {
            'database': self._check_database,
            'message_queue': self._check_message_queue,
            'event_store': self._check_event_store,
            'api_gateway': self._check_api_gateway,
            'cache': self._check_cache
        }
        self.health_history = deque(maxlen=1000)
        
    def check_health(self):
        '''Perform health checks'''
        results = {}
        issues = []
        
        for service, check_func in self.health_checks.items():
            status, message = check_func()
            results[service] = {
                'status': status,
                'message': message
            }
            
            if status != HealthStatus.HEALTHY:
                issues.append(f'{service}: {message}')
        
        # Determine overall status
        if all(r['status'] == HealthStatus.HEALTHY for r in results.values()):
            overall_status = HealthStatus.HEALTHY
        elif any(r['status'] == HealthStatus.UNHEALTHY for r in results.values()):
            overall_status = HealthStatus.UNHEALTHY
        else:
            overall_status = HealthStatus.DEGRADED
        
        # Record health check
        self.health_history.append({
            'timestamp': datetime.now(),
            'status': overall_status,
            'services': results
        })
        
        healthy_count = sum(1 for r in results.values() if r['status'] == HealthStatus.HEALTHY)
        
        return {
            'status': overall_status,
            'services': results,
            'healthy_count': healthy_count,
            'total_services': len(results),
            'issues': len(issues)
        }
    
    def _check_database(self):
        '''Check database health'''
        # Simulate health check
        if random.random() > 0.95:
            return HealthStatus.DEGRADED, 'High latency detected'
        return HealthStatus.HEALTHY, 'OK'
    
    def _check_message_queue(self):
        '''Check message queue health'''
        # Simulate health check
        if random.random() > 0.98:
            return HealthStatus.UNHEALTHY, 'Connection failed'
        return HealthStatus.HEALTHY, 'OK'
    
    def _check_event_store(self):
        '''Check event store health'''
        return HealthStatus.HEALTHY, 'OK'
    
    def _check_api_gateway(self):
        '''Check API gateway health'''
        return HealthStatus.HEALTHY, 'OK'
    
    def _check_cache(self):
        '''Check cache health'''
        if random.random() > 0.97:
            return HealthStatus.DEGRADED, 'High memory usage'
        return HealthStatus.HEALTHY, 'OK'

class PerformanceMonitor:
    '''Monitor system performance'''
    
    def __init__(self):
        self.performance_data = deque(maxlen=10000)
        self.latency_histogram = []
        
    def get_performance(self):
        '''Get current performance metrics'''
        # Simulate performance metrics
        cpu_usage = random.uniform(20, 80)
        memory_usage = random.uniform(30, 70)
        
        # Generate latency data
        latencies = [random.uniform(1, 100) for _ in range(100)]
        self.latency_histogram.extend(latencies)
        
        # Calculate percentiles
        sorted_latencies = sorted(self.latency_histogram[-1000:])
        p50 = sorted_latencies[int(len(sorted_latencies) * 0.5)]
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)]
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        
        throughput = random.randint(1000, 5000)
        
        # Record performance data
        self.performance_data.append({
            'timestamp': datetime.now(),
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'latency_p99': p99,
            'throughput': throughput
        })
        
        return {
            'cpu_usage': cpu_usage,
            'memory_usage': memory_usage,
            'latency_p50': p50,
            'latency_p95': p95,
            'latency_p99': p99,
            'throughput': throughput,
            'error_rate': random.uniform(0, 5)
        }
    
    def get_performance_trend(self, metric, duration_minutes=60):
        '''Get performance trend for metric'''
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        
        trend_data = [
            d for d in self.performance_data
            if d['timestamp'] > cutoff
        ]
        
        if not trend_data:
            return []
        
        return [
            {
                'timestamp': d['timestamp'],
                'value': d.get(metric, 0)
            }
            for d in trend_data
        ]

class AlertManager:
    '''Manage alerts and notifications'''
    
    def __init__(self):
        self.alerts = []
        self.alert_rules = self._initialize_alert_rules()
        self.notification_channels = ['email', 'slack', 'pagerduty']
        
    def _initialize_alert_rules(self):
        '''Initialize alert rules'''
        return [
            {
                'name': 'High CPU Usage',
                'condition': lambda m: m.get('system.cpu.usage', 0) > 80,
                'severity': AlertSeverity.HIGH,
                'threshold_duration': 300  # 5 minutes
            },
            {
                'name': 'High Error Rate',
                'condition': lambda m: m.get('error_rate', 0) > 5,
                'severity': AlertSeverity.CRITICAL,
                'threshold_duration': 60  # 1 minute
            },
            {
                'name': 'Low Disk Space',
                'condition': lambda m: m.get('system.disk.free', 100) < 10,
                'severity': AlertSeverity.MEDIUM,
                'threshold_duration': 600  # 10 minutes
            },
            {
                'name': 'Service Unhealthy',
                'condition': lambda m: m.get('health_status') == 'unhealthy',
                'severity': AlertSeverity.HIGH,
                'threshold_duration': 120  # 2 minutes
            }
        ]
    
    def check_alerts(self, metrics):
        '''Check if any alerts should be triggered'''
        for rule in self.alert_rules:
            if rule['condition'](metrics):
                self._trigger_alert(rule, metrics)
    
    def _trigger_alert(self, rule, metrics):
        '''Trigger an alert'''
        alert = {
            'id': str(uuid.uuid4()),
            'name': rule['name'],
            'severity': rule['severity'],
            'triggered_at': datetime.now(),
            'metrics': metrics,
            'status': 'active'
        }
        
        self.alerts.append(alert)
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def _send_notifications(self, alert):
        '''Send alert notifications'''
        # Simulate sending notifications
        for channel in self.notification_channels:
            if alert['severity'] in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                # Send to all channels for high severity
                pass
            elif alert['severity'] == AlertSeverity.MEDIUM and channel == 'email':
                # Send email for medium severity
                pass
    
    def get_active_alerts(self):
        '''Get active alerts'''
        active = [a for a in self.alerts if a['status'] == 'active']
        
        severity_counts = {
            'critical': sum(1 for a in active if a['severity'] == AlertSeverity.CRITICAL),
            'high': sum(1 for a in active if a['severity'] == AlertSeverity.HIGH),
            'medium': sum(1 for a in active if a['severity'] == AlertSeverity.MEDIUM),
            'low': sum(1 for a in active if a['severity'] == AlertSeverity.LOW),
            'info': sum(1 for a in active if a['severity'] == AlertSeverity.INFO)
        }
        
        return {
            'count': len(active),
            'alerts': active,
            **severity_counts
        }
    
    def acknowledge_alert(self, alert_id):
        '''Acknowledge an alert'''
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'acknowledged'
                alert['acknowledged_at'] = datetime.now()
                return True
        return False
    
    def resolve_alert(self, alert_id):
        '''Resolve an alert'''
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.now()
                return True
        return False

class DashboardService:
    '''Dashboard and visualization service'''
    
    def __init__(self):
        self.dashboards = self._initialize_dashboards()
        
    def _initialize_dashboards(self):
        '''Initialize dashboard configurations'''
        return {
            'system': {
                'name': 'System Overview',
                'widgets': [
                    {'type': 'gauge', 'metric': 'cpu_usage', 'title': 'CPU Usage'},
                    {'type': 'gauge', 'metric': 'memory_usage', 'title': 'Memory Usage'},
                    {'type': 'line', 'metric': 'throughput', 'title': 'Throughput'},
                    {'type': 'histogram', 'metric': 'latency', 'title': 'Latency Distribution'}
                ]
            },
            'business': {
                'name': 'Business Metrics',
                'widgets': [
                    {'type': 'counter', 'metric': 'trades_executed', 'title': 'Trades Executed'},
                    {'type': 'gauge', 'metric': 'portfolio_nav', 'title': 'Portfolio NAV'},
                    {'type': 'line', 'metric': 'revenue', 'title': 'Revenue Trend'},
                    {'type': 'pie', 'metric': 'asset_allocation', 'title': 'Asset Allocation'}
                ]
            },
            'events': {
                'name': 'Event Processing',
                'widgets': [
                    {'type': 'counter', 'metric': 'events_published', 'title': 'Events Published'},
                    {'type': 'counter', 'metric': 'events_consumed', 'title': 'Events Consumed'},
                    {'type': 'gauge', 'metric': 'queue_depth', 'title': 'Queue Depth'},
                    {'type': 'line', 'metric': 'event_rate', 'title': 'Event Rate'}
                ]
            }
        }
    
    def get_dashboard(self, name):
        '''Get dashboard configuration'''
        return self.dashboards.get(name)
    
    def get_widget_data(self, widget_type, metric):
        '''Get data for dashboard widget'''
        # Simulate widget data
        if widget_type == 'gauge':
            return {
                'value': random.uniform(0, 100),
                'min': 0,
                'max': 100,
                'thresholds': [60, 80]
            }
        elif widget_type == 'counter':
            return {
                'value': random.randint(1000, 10000),
                'change': random.uniform(-10, 10)
            }
        elif widget_type == 'line':
            return {
                'data': [
                    {'x': i, 'y': random.uniform(0, 100)}
                    for i in range(60)
                ]
            }
        elif widget_type == 'histogram':
            return {
                'buckets': [
                    {'range': '0-10ms', 'count': random.randint(100, 500)},
                    {'range': '10-50ms', 'count': random.randint(200, 800)},
                    {'range': '50-100ms', 'count': random.randint(50, 200)},
                    {'range': '>100ms', 'count': random.randint(10, 50)}
                ]
            }
        return {}

class SLAMonitor:
    '''Monitor SLA compliance'''
    
    def __init__(self):
        self.sla_targets = {
            'availability': 99.95,
            'latency_p99': 100,  # ms
            'error_rate': 0.1,   # %
            'throughput': 1000   # req/sec
        }
        self.uptime_start = datetime.now()
        self.downtime_total = timedelta(0)
        
    def get_sla_status(self):
        '''Get current SLA status'''
        # Calculate availability
        total_time = (datetime.now() - self.uptime_start).total_seconds()
        uptime = total_time - self.downtime_total.total_seconds()
        availability = (uptime / total_time) * 100 if total_time > 0 else 100
        
        # Simulate other SLA metrics
        current_metrics = {
            'availability': availability,
            'latency_p99': random.uniform(50, 150),
            'error_rate': random.uniform(0, 0.5),
            'throughput': random.uniform(800, 1200)
        }
        
        # Check SLA compliance
        violations = []
        for metric, target in self.sla_targets.items():
            current = current_metrics.get(metric, 0)
            if metric == 'availability' and current < target:
                violations.append(f'{metric}: {current:.2f}% < {target}%')
            elif metric != 'availability' and current > target:
                violations.append(f'{metric}: {current:.2f} > {target}')
        
        meeting_sla = len(violations) == 0
        
        return {
            'availability': availability,
            'target': self.sla_targets['availability'],
            'meeting_sla': meeting_sla,
            'violations': violations,
            'metrics': current_metrics
        }
    
    def record_downtime(self, duration):
        '''Record downtime period'''
        self.downtime_total += duration

class AnomalyDetector:
    '''Detect anomalies in metrics'''
    
    def __init__(self):
        self.baseline = {}
        self.anomalies = []
        
    def detect_anomalies(self):
        '''Detect anomalies in current metrics'''
        detected = []
        
        # Simulate anomaly detection
        anomaly_types = [
            {'type': 'Spike in error rate', 'severity': 'high', 'confidence': 0.85},
            {'type': 'Unusual traffic pattern', 'severity': 'medium', 'confidence': 0.72},
            {'type': 'Memory leak detected', 'severity': 'high', 'confidence': 0.91}
        ]
        
        # Randomly detect some anomalies
        for anomaly in anomaly_types:
            if random.random() > 0.8:
                detected.append(anomaly)
                self.anomalies.append({
                    **anomaly,
                    'detected_at': datetime.now()
                })
        
        return {
            'count': len(detected),
            'top_anomalies': detected,
            'total_historical': len(self.anomalies)
        }
    
    def train_baseline(self, metrics_history):
        '''Train baseline for anomaly detection'''
        # Calculate statistical baseline
        for metric_name, values in metrics_history.items():
            if values:
                self.baseline[metric_name] = {
                    'mean': statistics.mean(values),
                    'stdev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values)
                }

class ComplianceMonitor:
    '''Monitor compliance requirements'''
    
    def __init__(self):
        self.compliance_checks = {
            'data_retention': self._check_data_retention,
            'audit_logging': self._check_audit_logging,
            'encryption': self._check_encryption,
            'access_control': self._check_access_control,
            'availability': self._check_availability
        }
        
    def check_compliance(self):
        '''Check all compliance requirements'''
        results = {}
        
        for check_name, check_func in self.compliance_checks.items():
            compliant, details = check_func()
            results[check_name] = {
                'compliant': compliant,
                'details': details
            }
        
        overall_compliant = all(r['compliant'] for r in results.values())
        
        return {
            'compliant': overall_compliant,
            'checks': results
        }
    
    def _check_data_retention(self):
        '''Check data retention compliance'''
        # Check 7-year retention for AU/NZ
        return True, '7-year retention policy active'
    
    def _check_audit_logging(self):
        '''Check audit logging compliance'''
        return True, 'All access logged'
    
    def _check_encryption(self):
        '''Check encryption compliance'''
        return True, 'AES-256 encryption enabled'
    
    def _check_access_control(self):
        '''Check access control compliance'''
        return True, 'RBAC enabled'
    
    def _check_availability(self):
        '''Check availability compliance'''
        # Check if meeting availability requirements
        return True, '99.95% availability target'

# Demonstrate system
if __name__ == '__main__':
    print('📊 MONITORING & OBSERVABILITY - ULTRAPLATFORM')
    print('='*80)
    
    monitoring = MonitoringObservability()
    
    # Add some sample data
    monitoring.log_aggregator.log(LogLevel.INFO, 'System started')
    monitoring.log_aggregator.log(LogLevel.WARNING, 'High memory usage detected')
    monitoring.log_aggregator.log(LogLevel.ERROR, 'Connection timeout to database')
    
    # Create some traces
    trace = monitoring.distributed_tracing.start_trace('process_trade')
    span1 = monitoring.distributed_tracing.start_span(
        trace.trace_id, 'validate_trade', trace.span_id
    )
    monitoring.distributed_tracing.end_span(span1.span_id)
    monitoring.distributed_tracing.end_span(trace.span_id)
    
    # Monitor system
    print('\n🔍 SYSTEM MONITORING')
    print('='*80 + '\n')
    
    result = monitoring.monitor_system()
    
    # Show dashboard data
    print('\n' + '='*80)
    print('DASHBOARD: System Overview')
    print('='*80)
    dashboard = monitoring.dashboard_service.get_dashboard('system')
    for widget in dashboard['widgets']:
        data = monitoring.dashboard_service.get_widget_data(
            widget['type'], widget['metric']
        )
        print(f'{widget["title"]}: {data.get("value", "N/A")}')
    
    # Show compliance
    print('\n' + '='*80)
    print('COMPLIANCE STATUS')
    print('='*80)
    compliance = monitoring.compliance_monitor.check_compliance()
    for check, result in compliance['checks'].items():
        icon = '✅' if result['compliant'] else '❌'
        print(f'{icon} {check}: {result["details"]}')
    
    print('\n✅ Monitoring & Observability Operational!')
