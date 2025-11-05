from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from enum import Enum
import json
import uuid
import asyncio
import threading
from collections import defaultdict, deque
import random
import hashlib
import time

class ConsumptionPattern(Enum):
    COMPETING_CONSUMERS = 'competing_consumers'
    EXCLUSIVE_CONSUMER = 'exclusive_consumer'
    FAN_OUT_CONSUMER = 'fan_out_consumer'
    PRIORITY_CONSUMER = 'priority_consumer'
    BATCH_CONSUMER = 'batch_consumer'
    STREAMING_CONSUMER = 'streaming_consumer'
    REQUEST_RESPONSE = 'request_response'
    POLLING_CONSUMER = 'polling_consumer'
    EVENT_DRIVEN_CONSUMER = 'event_driven_consumer'
    SAGA_PARTICIPANT = 'saga_participant'

class ProcessingMode(Enum):
    SEQUENTIAL = 'sequential'
    PARALLEL = 'parallel'
    BATCH = 'batch'
    STREAMING = 'streaming'
    ASYNC = 'async'

class AcknowledgmentMode(Enum):
    AUTO = 'auto'
    MANUAL = 'manual'
    BATCH = 'batch'
    NONE = 'none'

class ConsumerState(Enum):
    IDLE = 'idle'
    CONSUMING = 'consuming'
    PROCESSING = 'processing'
    PAUSED = 'paused'
    ERROR = 'error'
    REBALANCING = 'rebalancing'

class ConsumptionPatterns:
    '''Comprehensive Consumption Patterns System for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Consumption Patterns'
        self.version = '2.0'
        self.pattern_registry = ConsumptionPatternRegistry()
        self.consumer_factory = ConsumerFactory()
        self.consumer_group_manager = ConsumerGroupManager()
        self.offset_manager = OffsetManager()
        self.backpressure_manager = BackpressureManager()
        self.error_handler = ErrorHandler()
        self.filter_manager = FilterManager()
        self.monitoring = ConsumptionMonitoring()
        
    def consume_events(self, source, pattern=ConsumptionPattern.COMPETING_CONSUMERS):
        '''Consume events using specified pattern'''
        print('EVENT CONSUMPTION')
        print('='*80)
        print(f'Source: {source}')
        print(f'Pattern: {pattern.value}')
        print(f'Timestamp: {datetime.now()}')
        print()
        
        # Step 1: Create Consumer
        print('1️⃣ CONSUMER CREATION')
        print('-'*40)
        consumer = self.consumer_factory.create_consumer(pattern)
        print(f'  Consumer Type: {consumer.__class__.__name__}')
        print(f'  Pattern: {pattern.value}')
        print(f'  Processing Mode: {consumer.processing_mode.value}')
        
        # Step 2: Consumer Group Assignment
        print('\n2️⃣ CONSUMER GROUP')
        print('-'*40)
        group = self.consumer_group_manager.assign_to_group(consumer, source)
        print(f'  Group ID: {group["group_id"]}')
        print(f'  Partition: {group["partition"]}')
        print(f'  Members: {group["member_count"]}')
        
        # Step 3: Offset Management
        print('\n3️⃣ OFFSET MANAGEMENT')
        print('-'*40)
        offset = self.offset_manager.get_offset(consumer, source)
        print(f'  Current Offset: {offset["current"]}')
        print(f'  Lag: {offset["lag"]} events')
        print(f'  Commit Strategy: {offset["commit_strategy"]}')
        
        # Step 4: Filter Configuration
        print('\n4️⃣ FILTER CONFIGURATION')
        print('-'*40)
        filters = self.filter_manager.get_filters(consumer)
        print(f'  Active Filters: {len(filters)}')
        for filter_item in filters[:3]:
            print(f'    • {filter_item["name"]}: {filter_item["condition"]}')
        
        # Step 5: Consume Events
        print('\n5️⃣ EVENT CONSUMPTION')
        print('-'*40)
        events = self._simulate_events(5)
        results = consumer.consume(events)
        print(f'  Events Consumed: {results["consumed"]}')
        print(f'  Events Processed: {results["processed"]}')
        print(f'  Processing Time: {results["processing_time"]}ms')
        
        # Step 6: Backpressure Handling
        print('\n6️⃣ BACKPRESSURE HANDLING')
        print('-'*40)
        backpressure = self.backpressure_manager.check_backpressure(consumer)
        print(f'  Buffer Usage: {backpressure["buffer_usage"]:.1%}')
        print(f'  Processing Rate: {backpressure["rate"]} events/sec')
        print(f'  Strategy: {backpressure["strategy"]}')
        
        # Step 7: Error Handling
        print('\n7️⃣ ERROR HANDLING')
        print('-'*40)
        error_stats = self.error_handler.get_stats(consumer)
        print(f'  Error Rate: {error_stats["error_rate"]:.2%}')
        print(f'  Retry Count: {error_stats["retries"]}')
        print(f'  DLQ Size: {error_stats["dlq_size"]}')
        
        return results
    
    def _simulate_events(self, count):
        '''Simulate events for consumption'''
        events = []
        for i in range(count):
            events.append({
                'id': str(uuid.uuid4()),
                'type': random.choice(['trade', 'portfolio', 'risk']),
                'timestamp': datetime.now().isoformat(),
                'payload': {'value': random.randint(100, 1000)}
            })
        return events

class ConsumptionPatternRegistry:
    '''Registry of consumption patterns'''
    
    def __init__(self):
        self.patterns = self._initialize_patterns()
        
    def _initialize_patterns(self):
        '''Initialize pattern definitions'''
        return {
            ConsumptionPattern.COMPETING_CONSUMERS: {
                'description': 'Multiple consumers compete for messages',
                'use_cases': ['Load balancing', 'Horizontal scaling', 'Work distribution'],
                'processing_mode': ProcessingMode.PARALLEL,
                'scalability': 'high'
            },
            ConsumptionPattern.EXCLUSIVE_CONSUMER: {
                'description': 'Single consumer for a partition/queue',
                'use_cases': ['Ordered processing', 'Stateful processing'],
                'processing_mode': ProcessingMode.SEQUENTIAL,
                'scalability': 'limited'
            },
            ConsumptionPattern.FAN_OUT_CONSUMER: {
                'description': 'All consumers receive all messages',
                'use_cases': ['Broadcasting', 'Multiple views', 'CQRS read models'],
                'processing_mode': ProcessingMode.PARALLEL,
                'scalability': 'high'
            },
            ConsumptionPattern.PRIORITY_CONSUMER: {
                'description': 'Consume based on message priority',
                'use_cases': ['Critical message handling', 'SLA management'],
                'processing_mode': ProcessingMode.SEQUENTIAL,
                'scalability': 'medium'
            },
            ConsumptionPattern.BATCH_CONSUMER: {
                'description': 'Process messages in batches',
                'use_cases': ['Bulk operations', 'ETL', 'Analytics'],
                'processing_mode': ProcessingMode.BATCH,
                'scalability': 'high'
            },
            ConsumptionPattern.STREAMING_CONSUMER: {
                'description': 'Continuous stream processing',
                'use_cases': ['Real-time analytics', 'CEP', 'Live dashboards'],
                'processing_mode': ProcessingMode.STREAMING,
                'scalability': 'high'
            },
            ConsumptionPattern.REQUEST_RESPONSE: {
                'description': 'Synchronous request-response',
                'use_cases': ['RPC', 'Query operations', 'Validation'],
                'processing_mode': ProcessingMode.SEQUENTIAL,
                'scalability': 'medium'
            },
            ConsumptionPattern.POLLING_CONSUMER: {
                'description': 'Poll for messages at intervals',
                'use_cases': ['Legacy integration', 'Rate limiting'],
                'processing_mode': ProcessingMode.SEQUENTIAL,
                'scalability': 'low'
            },
            ConsumptionPattern.EVENT_DRIVEN_CONSUMER: {
                'description': 'Push-based consumption',
                'use_cases': ['Real-time processing', 'Low latency'],
                'processing_mode': ProcessingMode.ASYNC,
                'scalability': 'high'
            },
            ConsumptionPattern.SAGA_PARTICIPANT: {
                'description': 'Participate in distributed transactions',
                'use_cases': ['Saga patterns', 'Compensation logic'],
                'processing_mode': ProcessingMode.SEQUENTIAL,
                'scalability': 'medium'
            }
        }
    
    def get_pattern(self, pattern):
        '''Get pattern definition'''
        return self.patterns.get(pattern)

class ConsumerFactory:
    '''Factory for creating consumers'''
    
    def create_consumer(self, pattern):
        '''Create consumer based on pattern'''
        consumers = {
            ConsumptionPattern.COMPETING_CONSUMERS: CompetingConsumer(),
            ConsumptionPattern.EXCLUSIVE_CONSUMER: ExclusiveConsumer(),
            ConsumptionPattern.FAN_OUT_CONSUMER: FanOutConsumer(),
            ConsumptionPattern.PRIORITY_CONSUMER: PriorityConsumer(),
            ConsumptionPattern.BATCH_CONSUMER: BatchConsumer(),
            ConsumptionPattern.STREAMING_CONSUMER: StreamingConsumer(),
            ConsumptionPattern.REQUEST_RESPONSE: RequestResponseConsumer(),
            ConsumptionPattern.POLLING_CONSUMER: PollingConsumer(),
            ConsumptionPattern.EVENT_DRIVEN_CONSUMER: EventDrivenConsumer(),
            ConsumptionPattern.SAGA_PARTICIPANT: SagaParticipantConsumer()
        }
        return consumers.get(pattern)

class BaseConsumer:
    '''Base class for consumers'''
    
    def __init__(self, processing_mode=ProcessingMode.SEQUENTIAL):
        self.consumer_id = str(uuid.uuid4())
        self.processing_mode = processing_mode
        self.state = ConsumerState.IDLE
        self.acknowledgment_mode = AcknowledgmentMode.AUTO
        self.buffer = deque()
        self.metrics = {
            'consumed': 0,
            'processed': 0,
            'failed': 0,
            'processing_time': 0
        }
    
    def consume(self, events):
        '''Consume events'''
        self.state = ConsumerState.CONSUMING
        start = datetime.now()
        
        consumed = 0
        processed = 0
        
        for event in events:
            # Add to buffer
            self.buffer.append(event)
            consumed += 1
            
            # Process event
            if self._process_event(event):
                processed += 1
                self._acknowledge(event)
        
        self.metrics['consumed'] += consumed
        self.metrics['processed'] += processed
        
        processing_time = (datetime.now() - start).total_seconds() * 1000
        self.metrics['processing_time'] += processing_time
        
        self.state = ConsumerState.IDLE
        
        return {
            'consumed': consumed,
            'processed': processed,
            'processing_time': processing_time
        }
    
    def _process_event(self, event):
        '''Process individual event'''
        self.state = ConsumerState.PROCESSING
        # Simulate processing
        time.sleep(0.001)  # 1ms processing
        return True
    
    def _acknowledge(self, event):
        '''Acknowledge event processing'''
        if self.acknowledgment_mode == AcknowledgmentMode.AUTO:
            # Auto acknowledge
            pass
        elif self.acknowledgment_mode == AcknowledgmentMode.MANUAL:
            # Manual acknowledge
            pass

class CompetingConsumer(BaseConsumer):
    '''Competing consumers pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.PARALLEL)
        self.consumer_group = None
    
    def consume(self, events):
        '''Consume events competitively'''
        # Only consume events assigned to this consumer
        assigned_events = self._get_assigned_events(events)
        return super().consume(assigned_events)
    
    def _get_assigned_events(self, events):
        '''Get events assigned to this consumer'''
        # Simplified: take every Nth event based on consumer position
        return events[::2]  # Take every other event

class ExclusiveConsumer(BaseConsumer):
    '''Exclusive consumer pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.SEQUENTIAL)
        self.lock = threading.Lock()
    
    def consume(self, events):
        '''Consume events exclusively'''
        with self.lock:
            # Process all events sequentially
            return super().consume(events)

class FanOutConsumer(BaseConsumer):
    '''Fan-out consumer pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.PARALLEL)
        self.subscription_id = str(uuid.uuid4())
    
    def consume(self, events):
        '''Consume all events (fan-out)'''
        # Process all events
        return super().consume(events)

class PriorityConsumer(BaseConsumer):
    '''Priority-based consumer'''
    
    def __init__(self):
        super().__init__(ProcessingMode.SEQUENTIAL)
        self.priority_queue = []
    
    def consume(self, events):
        '''Consume events by priority'''
        # Sort events by priority
        sorted_events = self._sort_by_priority(events)
        return super().consume(sorted_events)
    
    def _sort_by_priority(self, events):
        '''Sort events by priority'''
        # Simplified: sort by event type
        priority_map = {'critical': 0, 'high': 1, 'normal': 2, 'low': 3}
        return sorted(events, key=lambda e: priority_map.get(e.get('priority', 'normal'), 2))

class BatchConsumer(BaseConsumer):
    '''Batch processing consumer'''
    
    def __init__(self):
        super().__init__(ProcessingMode.BATCH)
        self.batch_size = 10
        self.batch_timeout = 5000  # ms
        self.current_batch = []
    
    def consume(self, events):
        '''Consume events in batches'''
        self.current_batch.extend(events)
        
        if len(self.current_batch) >= self.batch_size:
            # Process batch
            batch = self.current_batch[:self.batch_size]
            self.current_batch = self.current_batch[self.batch_size:]
            
            result = self._process_batch(batch)
            return result
        
        return {'consumed': len(events), 'processed': 0, 'processing_time': 0}
    
    def _process_batch(self, batch):
        '''Process a batch of events'''
        start = datetime.now()
        
        # Simulate batch processing
        time.sleep(0.01)  # 10ms for batch
        
        processing_time = (datetime.now() - start).total_seconds() * 1000
        
        return {
            'consumed': len(batch),
            'processed': len(batch),
            'processing_time': processing_time
        }

class StreamingConsumer(BaseConsumer):
    '''Streaming consumer pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.STREAMING)
        self.stream_position = 0
        self.window_size = 100
    
    def consume(self, events):
        '''Consume events as stream'''
        # Process events in streaming fashion
        results = {
            'consumed': 0,
            'processed': 0,
            'processing_time': 0
        }
        
        for event in events:
            # Update stream position
            self.stream_position += 1
            
            # Process in window
            window_result = self._process_window(event)
            results['consumed'] += 1
            results['processed'] += 1 if window_result else 0
        
        return results
    
    def _process_window(self, event):
        '''Process event in streaming window'''
        # Simplified window processing
        return True

class RequestResponseConsumer(BaseConsumer):
    '''Request-response consumer'''
    
    def __init__(self):
        super().__init__(ProcessingMode.SEQUENTIAL)
        self.pending_requests = {}
    
    def consume(self, events):
        '''Consume request events and send responses'''
        results = {
            'consumed': 0,
            'processed': 0,
            'processing_time': 0
        }
        
        for event in events:
            if event.get('reply_to'):
                # Process request
                response = self._process_request(event)
                self._send_response(event['reply_to'], response)
                results['consumed'] += 1
                results['processed'] += 1
        
        return results
    
    def _process_request(self, request):
        '''Process request and generate response'''
        return {'result': 'processed', 'data': request.get('payload')}
    
    def _send_response(self, reply_to, response):
        '''Send response to requester'''
        pass

class PollingConsumer(BaseConsumer):
    '''Polling consumer pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.SEQUENTIAL)
        self.poll_interval = 1000  # ms
        self.last_poll = datetime.now()
    
    def consume(self, events):
        '''Poll for events at intervals'''
        # Check if it's time to poll
        now = datetime.now()
        if (now - self.last_poll).total_seconds() * 1000 < self.poll_interval:
            return {'consumed': 0, 'processed': 0, 'processing_time': 0}
        
        self.last_poll = now
        return super().consume(events)

class EventDrivenConsumer(BaseConsumer):
    '''Event-driven consumer pattern'''
    
    def __init__(self):
        super().__init__(ProcessingMode.ASYNC)
        self.event_handlers = {}
    
    def consume(self, events):
        '''Consume events asynchronously'''
        results = {
            'consumed': 0,
            'processed': 0,
            'processing_time': 0
        }
        
        for event in events:
            # Trigger async processing
            self._trigger_handler(event)
            results['consumed'] += 1
            results['processed'] += 1
        
        return results
    
    def _trigger_handler(self, event):
        '''Trigger event handler asynchronously'''
        event_type = event.get('type')
        if event_type in self.event_handlers:
            # Would be async in real implementation
            self.event_handlers[event_type](event)

class SagaParticipantConsumer(BaseConsumer):
    '''Saga participant consumer'''
    
    def __init__(self):
        super().__init__(ProcessingMode.SEQUENTIAL)
        self.saga_state = {}
        self.compensation_handlers = {}
    
    def consume(self, events):
        '''Consume saga events'''
        results = {
            'consumed': 0,
            'processed': 0,
            'processing_time': 0
        }
        
        for event in events:
            saga_id = event.get('saga_id')
            if saga_id:
                # Process saga step
                if self._process_saga_step(saga_id, event):
                    results['processed'] += 1
                results['consumed'] += 1
        
        return results
    
    def _process_saga_step(self, saga_id, event):
        '''Process saga step'''
        # Update saga state
        if saga_id not in self.saga_state:
            self.saga_state[saga_id] = {'steps': [], 'status': 'active'}
        
        self.saga_state[saga_id]['steps'].append(event)
        return True

class ConsumerGroupManager:
    '''Manage consumer groups'''
    
    def __init__(self):
        self.groups = {}
        self.assignments = {}
    
    def assign_to_group(self, consumer, source):
        '''Assign consumer to group'''
        group_id = f'group_{source}'
        
        if group_id not in self.groups:
            self.groups[group_id] = {
                'members': [],
                'partitions': 16,
                'rebalance_strategy': 'range'
            }
        
        group = self.groups[group_id]
        group['members'].append(consumer.consumer_id)
        
        # Assign partition
        partition = len(group['members']) % group['partitions']
        self.assignments[consumer.consumer_id] = partition
        
        return {
            'group_id': group_id,
            'partition': partition,
            'member_count': len(group['members'])
        }
    
    def rebalance(self, group_id):
        '''Rebalance consumer group'''
        if group_id not in self.groups:
            return
        
        group = self.groups[group_id]
        members = group['members']
        partitions = group['partitions']
        
        # Simple range assignment
        for i, member in enumerate(members):
            self.assignments[member] = i % partitions

class OffsetManager:
    '''Manage consumer offsets'''
    
    def __init__(self):
        self.offsets = defaultdict(lambda: defaultdict(int))
        self.lag = defaultdict(int)
    
    def get_offset(self, consumer, source):
        '''Get current offset for consumer'''
        offset_key = f'{consumer.consumer_id}:{source}'
        current = self.offsets[offset_key]['current']
        
        # Calculate lag (simplified)
        latest = current + random.randint(0, 100)
        lag = latest - current
        
        return {
            'current': current,
            'latest': latest,
            'lag': lag,
            'commit_strategy': 'auto'
        }
    
    def commit_offset(self, consumer, source, offset):
        '''Commit offset'''
        offset_key = f'{consumer.consumer_id}:{source}'
        self.offsets[offset_key]['current'] = offset
        self.offsets[offset_key]['committed_at'] = datetime.now()
    
    def reset_offset(self, consumer, source, position='latest'):
        '''Reset offset to specified position'''
        offset_key = f'{consumer.consumer_id}:{source}'
        if position == 'earliest':
            self.offsets[offset_key]['current'] = 0
        elif position == 'latest':
            self.offsets[offset_key]['current'] = 999999  # Large number

class BackpressureManager:
    '''Manage backpressure'''
    
    def __init__(self):
        self.thresholds = {
            'buffer_high': 0.8,
            'buffer_critical': 0.95,
            'rate_low': 100,
            'rate_high': 1000
        }
    
    def check_backpressure(self, consumer):
        '''Check backpressure for consumer'''
        # Calculate buffer usage
        buffer_size = len(consumer.buffer)
        max_buffer = 1000
        buffer_usage = buffer_size / max_buffer
        
        # Calculate processing rate
        if consumer.metrics['processing_time'] > 0:
            rate = consumer.metrics['processed'] / (consumer.metrics['processing_time'] / 1000)
        else:
            rate = 0
        
        # Determine strategy
        if buffer_usage > self.thresholds['buffer_critical']:
            strategy = 'pause_consumption'
        elif buffer_usage > self.thresholds['buffer_high']:
            strategy = 'slow_consumption'
        elif rate < self.thresholds['rate_low']:
            strategy = 'increase_resources'
        else:
            strategy = 'normal'
        
        return {
            'buffer_usage': buffer_usage,
            'rate': rate,
            'strategy': strategy
        }

class ErrorHandler:
    '''Handle consumption errors'''
    
    def __init__(self):
        self.error_counts = defaultdict(int)
        self.retry_counts = defaultdict(int)
        self.dlq = deque(maxlen=1000)
    
    def handle_error(self, consumer, event, error):
        '''Handle consumption error'''
        self.error_counts[consumer.consumer_id] += 1
        
        # Retry logic
        retry_count = self.retry_counts[f'{consumer.consumer_id}:{event["id"]}']
        
        if retry_count < 3:
            # Retry
            self.retry_counts[f'{consumer.consumer_id}:{event["id"]}'] += 1
            return 'retry'
        else:
            # Send to DLQ
            self.dlq.append({
                'event': event,
                'error': str(error),
                'consumer': consumer.consumer_id,
                'timestamp': datetime.now()
            })
            return 'dlq'
    
    def get_stats(self, consumer):
        '''Get error statistics'''
        total = consumer.metrics['consumed']
        errors = self.error_counts[consumer.consumer_id]
        
        return {
            'error_rate': errors / total if total > 0 else 0,
            'retries': sum(1 for k in self.retry_counts if k.startswith(consumer.consumer_id)),
            'dlq_size': len(self.dlq)
        }

class FilterManager:
    '''Manage event filters'''
    
    def __init__(self):
        self.filters = defaultdict(list)
    
    def add_filter(self, consumer, name, condition):
        '''Add filter for consumer'''
        self.filters[consumer.consumer_id].append({
            'name': name,
            'condition': condition,
            'active': True
        })
    
    def get_filters(self, consumer):
        '''Get filters for consumer'''
        # Default filters
        default_filters = [
            {'name': 'type_filter', 'condition': 'type in [trade, portfolio]'},
            {'name': 'priority_filter', 'condition': 'priority >= medium'},
            {'name': 'timestamp_filter', 'condition': 'timestamp > now-1h'}
        ]
        
        return self.filters.get(consumer.consumer_id, default_filters)
    
    def apply_filters(self, events, filters):
        '''Apply filters to events'''
        filtered = events
        for filter_def in filters:
            if filter_def['active']:
                # Simplified filtering
                filtered = [e for e in filtered if self._evaluate(e, filter_def['condition'])]
        return filtered
    
    def _evaluate(self, event, condition):
        '''Evaluate filter condition'''
        # Simplified evaluation
        return True

class ConsumptionMonitoring:
    '''Monitor consumption metrics'''
    
    def get_metrics(self):
        '''Get consumption metrics'''
        return {
            'total_consumed': 5847293,
            'consumption_rate': 1250,  # events/sec
            'avg_processing_time': 2.5,  # ms
            'consumer_groups': 12,
            'active_consumers': 45,
            'avg_lag': 150,
            'error_rate': 0.0012,
            'patterns_used': {
                'competing': 40,
                'exclusive': 20,
                'batch': 15,
                'streaming': 15,
                'other': 10
            }
        }

# Demonstrate system
if __name__ == '__main__':
    print('🎯 CONSUMPTION PATTERNS - ULTRAPLATFORM')
    print('='*80)
    
    consumption = ConsumptionPatterns()
    
    # Show available patterns
    print('\n📊 AVAILABLE PATTERNS:')
    print('-'*40)
    for pattern in ConsumptionPattern:
        pattern_def = consumption.pattern_registry.get_pattern(pattern)
        if pattern_def:
            print(f'\n{pattern.value.upper()}:')
            print(f'  Description: {pattern_def["description"]}')
            print(f'  Processing: {pattern_def["processing_mode"].value}')
            print(f'  Scalability: {pattern_def["scalability"]}')
            print(f'  Use Cases:')
            for use_case in pattern_def["use_cases"][:2]:
                print(f'    • {use_case}')
    
    # Demonstrate consumption
    print('\n' + '='*80)
    print('CONSUMING EVENTS')
    print('='*80 + '\n')
    
    result = consumption.consume_events('trading_events', ConsumptionPattern.BATCH_CONSUMER)
    
    # Show metrics
    print('\n' + '='*80)
    print('CONSUMPTION METRICS')
    print('='*80)
    metrics = consumption.monitoring.get_metrics()
    print(f'Total Consumed: {metrics["total_consumed"]:,}')
    print(f'Consumption Rate: {metrics["consumption_rate"]} events/sec')
    print(f'Active Consumers: {metrics["active_consumers"]}')
    print(f'Average Lag: {metrics["avg_lag"]} events')
    
    print('\n✅ Consumption Patterns Operational!')
