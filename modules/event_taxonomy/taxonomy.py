from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json
import uuid
from dataclasses import dataclass, field

class EventDomain(Enum):
    TRADING = 'trading'
    PORTFOLIO = 'portfolio'
    RISK = 'risk'
    MARKET = 'market'
    COMPLIANCE = 'compliance'
    SYSTEM = 'system'
    ANALYTICS = 'analytics'
    CLIENT = 'client'

class EventCategory(Enum):
    COMMAND = 'command'
    QUERY = 'query'
    FACT = 'fact'
    NOTIFICATION = 'notification'
    INTEGRATION = 'integration'

class EventSeverity(Enum):
    CRITICAL = 'critical'
    HIGH = 'high'
    MEDIUM = 'medium'
    LOW = 'low'
    INFO = 'info'

class EventTaxonomy:
    '''Complete Event Types and Taxonomy System for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Event Taxonomy'
        self.version = '2.0'
        self.event_catalog = EventCatalog()
        self.schema_registry = SchemaRegistry()
        self.relationship_manager = RelationshipManager()
        self.validator = EventValidator()
        self.metadata_manager = MetadataManager()
        self.lifecycle_manager = LifecycleManager()
        
    def display_taxonomy(self):
        '''Display complete event taxonomy'''
        print('EVENT TYPES AND TAXONOMY')
        print('='*80)
        print(f'Total Event Types: {len(self.event_catalog.get_all_events())}')
        print(f'Domains: {len(EventDomain)}')
        print(f'Categories: {len(EventCategory)}')
        print()
        
        # Display hierarchy
        print('📊 EVENT HIERARCHY:')
        print('-'*40)
        
        for domain in EventDomain:
            events = self.event_catalog.get_events_by_domain(domain)
            print(f'\n{domain.value.upper()} DOMAIN ({len(events)} events):')
            
            # Group by category
            by_category = {}
            for event in events:
                cat = event.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(event)
            
            for category, cat_events in by_category.items():
                print(f'  {category.value}:')
                for event in cat_events[:3]:  # Show first 3
                    print(f'    • {event.name} ({event.severity.value})')
                if len(cat_events) > 3:
                    print(f'    ... and {len(cat_events)-3} more')

class EventCatalog:
    '''Catalog of all event types'''
    
    def __init__(self):
        self.events = self._initialize_events()
        
    def _initialize_events(self):
        '''Initialize complete event catalog'''
        events = []
        
        # Trading Events
        events.extend([
            EventType('trade.requested', EventDomain.TRADING, EventCategory.COMMAND, EventSeverity.HIGH),
            EventType('trade.validated', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('trade.executed', EventDomain.TRADING, EventCategory.FACT, EventSeverity.HIGH),
            EventType('trade.failed', EventDomain.TRADING, EventCategory.FACT, EventSeverity.HIGH),
            EventType('trade.cancelled', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('trade.settled', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('order.placed', EventDomain.TRADING, EventCategory.COMMAND, EventSeverity.HIGH),
            EventType('order.modified', EventDomain.TRADING, EventCategory.COMMAND, EventSeverity.MEDIUM),
            EventType('order.cancelled', EventDomain.TRADING, EventCategory.COMMAND, EventSeverity.MEDIUM),
            EventType('order.filled', EventDomain.TRADING, EventCategory.FACT, EventSeverity.HIGH),
            EventType('order.partially_filled', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('signal.generated', EventDomain.TRADING, EventCategory.FACT, EventSeverity.HIGH),
            EventType('signal.approved', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('signal.rejected', EventDomain.TRADING, EventCategory.FACT, EventSeverity.MEDIUM)
        ])
        
        # Portfolio Events
        events.extend([
            EventType('portfolio.created', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('portfolio.updated', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('portfolio.rebalanced', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('position.opened', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('position.closed', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('position.updated', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('nav.calculated', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('pnl.calculated', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('allocation.changed', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('cash.deposited', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH),
            EventType('cash.withdrawn', EventDomain.PORTFOLIO, EventCategory.FACT, EventSeverity.HIGH)
        ])
        
        # Risk Events
        events.extend([
            EventType('risk.calculated', EventDomain.RISK, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('risk.limit.breached', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.CRITICAL),
            EventType('risk.limit.warning', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('risk.limit.cleared', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.INFO),
            EventType('var.calculated', EventDomain.RISK, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('var.exceeded', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('drawdown.exceeded', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('exposure.calculated', EventDomain.RISK, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('stress.test.completed', EventDomain.RISK, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('margin.call', EventDomain.RISK, EventCategory.NOTIFICATION, EventSeverity.CRITICAL)
        ])
        
        # Market Events
        events.extend([
            EventType('market.opened', EventDomain.MARKET, EventCategory.FACT, EventSeverity.HIGH),
            EventType('market.closed', EventDomain.MARKET, EventCategory.FACT, EventSeverity.HIGH),
            EventType('market.halted', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.CRITICAL),
            EventType('price.updated', EventDomain.MARKET, EventCategory.FACT, EventSeverity.LOW),
            EventType('price.gap.detected', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('volume.spike', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.MEDIUM),
            EventType('volatility.spike', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('bid.updated', EventDomain.MARKET, EventCategory.FACT, EventSeverity.LOW),
            EventType('ask.updated', EventDomain.MARKET, EventCategory.FACT, EventSeverity.LOW),
            EventType('spread.widened', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.MEDIUM),
            EventType('liquidity.low', EventDomain.MARKET, EventCategory.NOTIFICATION, EventSeverity.HIGH)
        ])
        
        # Compliance Events
        events.extend([
            EventType('compliance.check.initiated', EventDomain.COMPLIANCE, EventCategory.COMMAND, EventSeverity.HIGH),
            EventType('compliance.check.passed', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('compliance.check.failed', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.CRITICAL),
            EventType('compliance.breach.detected', EventDomain.COMPLIANCE, EventCategory.NOTIFICATION, EventSeverity.CRITICAL),
            EventType('audit.initiated', EventDomain.COMPLIANCE, EventCategory.COMMAND, EventSeverity.HIGH),
            EventType('audit.completed', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('report.generated', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('report.submitted', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.HIGH),
            EventType('kyc.initiated', EventDomain.COMPLIANCE, EventCategory.COMMAND, EventSeverity.HIGH),
            EventType('kyc.completed', EventDomain.COMPLIANCE, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('aml.alert', EventDomain.COMPLIANCE, EventCategory.NOTIFICATION, EventSeverity.CRITICAL)
        ])
        
        # System Events
        events.extend([
            EventType('system.started', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.HIGH),
            EventType('system.stopped', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.HIGH),
            EventType('component.started', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('component.stopped', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('component.failed', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.CRITICAL),
            EventType('component.recovered', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('performance.degraded', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('memory.high', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('cpu.high', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('disk.low', EventDomain.SYSTEM, EventCategory.NOTIFICATION, EventSeverity.HIGH),
            EventType('backup.completed', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('deployment.started', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.HIGH),
            EventType('deployment.completed', EventDomain.SYSTEM, EventCategory.FACT, EventSeverity.HIGH)
        ])
        
        # Analytics Events
        events.extend([
            EventType('analytics.calculated', EventDomain.ANALYTICS, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('report.requested', EventDomain.ANALYTICS, EventCategory.COMMAND, EventSeverity.MEDIUM),
            EventType('report.generated', EventDomain.ANALYTICS, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('metrics.updated', EventDomain.ANALYTICS, EventCategory.FACT, EventSeverity.LOW),
            EventType('performance.calculated', EventDomain.ANALYTICS, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('benchmark.compared', EventDomain.ANALYTICS, EventCategory.FACT, EventSeverity.MEDIUM)
        ])
        
        # Client Events
        events.extend([
            EventType('client.connected', EventDomain.CLIENT, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('client.disconnected', EventDomain.CLIENT, EventCategory.FACT, EventSeverity.MEDIUM),
            EventType('client.authenticated', EventDomain.CLIENT, EventCategory.FACT, EventSeverity.HIGH),
            EventType('client.request', EventDomain.CLIENT, EventCategory.COMMAND, EventSeverity.MEDIUM),
            EventType('client.notification', EventDomain.CLIENT, EventCategory.NOTIFICATION, EventSeverity.MEDIUM)
        ])
        
        return events
    
    def get_all_events(self):
        return self.events
    
    def get_events_by_domain(self, domain):
        return [e for e in self.events if e.domain == domain]
    
    def get_event_by_name(self, name):
        for event in self.events:
            if event.name == name:
                return event
        return None

@dataclass
class EventType:
    '''Event type definition'''
    name: str
    domain: EventDomain
    category: EventCategory
    severity: EventSeverity
    schema: Optional[Dict] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.schema is None:
            self.schema = self._default_schema()
    
    def _default_schema(self):
        '''Generate default schema for event'''
        return {
            'type': 'object',
            'required': ['event_id', 'event_type', 'timestamp', 'payload'],
            'properties': {
                'event_id': {'type': 'string', 'format': 'uuid'},
                'event_type': {'type': 'string', 'enum': [self.name]},
                'timestamp': {'type': 'string', 'format': 'date-time'},
                'version': {'type': 'string'},
                'domain': {'type': 'string', 'enum': [self.domain.value]},
                'category': {'type': 'string', 'enum': [self.category.value]},
                'severity': {'type': 'string', 'enum': [self.severity.value]},
                'payload': self._get_payload_schema(),
                'metadata': {'type': 'object'},
                'correlation_id': {'type': 'string'},
                'causation_id': {'type': 'string'}
            }
        }
    
    def _get_payload_schema(self):
        '''Get payload schema based on event type'''
        if 'trade' in self.name:
            return {
                'type': 'object',
                'required': ['symbol', 'action', 'quantity', 'price'],
                'properties': {
                    'trade_id': {'type': 'string'},
                    'symbol': {'type': 'string'},
                    'action': {'type': 'string', 'enum': ['BUY', 'SELL']},
                    'quantity': {'type': 'number'},
                    'price': {'type': 'number'},
                    'strategy': {'type': 'string'}
                }
            }
        elif 'portfolio' in self.name:
            return {
                'type': 'object',
                'properties': {
                    'portfolio_id': {'type': 'string'},
                    'nav': {'type': 'number'},
                    'total_value': {'type': 'number'},
                    'positions': {'type': 'array'}
                }
            }
        elif 'risk' in self.name:
            return {
                'type': 'object',
                'properties': {
                    'risk_type': {'type': 'string'},
                    'value': {'type': 'number'},
                    'threshold': {'type': 'number'},
                    'status': {'type': 'string'}
                }
            }
        else:
            return {'type': 'object'}

class SchemaRegistry:
    '''Registry for event schemas'''
    
    def __init__(self):
        self.schemas = {}
        self.versions = {}
        
    def register_schema(self, event_type, schema, version='1.0'):
        '''Register event schema'''
        key = f'{event_type}_{version}'
        self.schemas[key] = schema
        
        if event_type not in self.versions:
            self.versions[event_type] = []
        self.versions[event_type].append(version)
        
        return key
    
    def get_schema(self, event_type, version=None):
        '''Get schema for event type'''
        if version is None:
            # Get latest version
            if event_type in self.versions:
                version = self.versions[event_type][-1]
            else:
                version = '1.0'
        
        key = f'{event_type}_{version}'
        return self.schemas.get(key)
    
    def validate_schema(self, event, schema):
        '''Validate event against schema'''
        # Simplified validation
        required_fields = schema.get('required', [])
        for field in required_fields:
            if field not in event:
                return False, f'Missing required field: {field}'
        
        return True, 'Valid'

class RelationshipManager:
    '''Manage relationships between events'''
    
    def __init__(self):
        self.relationships = self._initialize_relationships()
        self.event_chains = []
        
    def _initialize_relationships(self):
        '''Initialize event relationships'''
        return {
            'trade.requested': {
                'produces': ['trade.validated', 'trade.failed'],
                'requires': ['signal.approved'],
                'triggers': ['risk.calculated']
            },
            'trade.validated': {
                'produces': ['trade.executed', 'trade.cancelled'],
                'requires': ['compliance.check.passed'],
                'triggers': ['portfolio.updated']
            },
            'trade.executed': {
                'produces': ['trade.settled'],
                'triggers': ['position.updated', 'pnl.calculated', 'nav.calculated'],
                'notifies': ['client.notification']
            },
            'risk.limit.breached': {
                'triggers': ['trade.cancelled', 'system.stopped'],
                'notifies': ['compliance.breach.detected', 'client.notification']
            },
            'portfolio.rebalanced': {
                'produces': ['trade.requested'],
                'triggers': ['nav.calculated', 'allocation.changed']
            }
        }
    
    def get_relationships(self, event_type):
        '''Get relationships for event type'''
        return self.relationships.get(event_type, {})
    
    def get_event_chain(self, start_event):
        '''Get complete event chain from start event'''
        chain = [start_event]
        current = start_event
        
        while current in self.relationships:
            rel = self.relationships[current]
            next_events = rel.get('produces', [])
            if next_events:
                current = next_events[0]
                chain.append(current)
            else:
                break
        
        return chain
    
    def get_dependencies(self, event_type):
        '''Get dependencies for event type'''
        rel = self.relationships.get(event_type, {})
        return rel.get('requires', [])

class EventValidator:
    '''Validate events against taxonomy'''
    
    def validate(self, event, event_type_def):
        '''Validate event against type definition'''
        errors = []
        
        # Check required fields
        if 'event_id' not in event:
            errors.append('Missing event_id')
        if 'timestamp' not in event:
            errors.append('Missing timestamp')
        if 'payload' not in event:
            errors.append('Missing payload')
        
        # Check event type matches
        if event.get('event_type') != event_type_def.name:
            errors.append(f'Event type mismatch: expected {event_type_def.name}')
        
        # Check severity
        if event.get('severity') != event_type_def.severity.value:
            errors.append(f'Severity mismatch: expected {event_type_def.severity.value}')
        
        return len(errors) == 0, errors

class MetadataManager:
    '''Manage event metadata standards'''
    
    def __init__(self):
        self.required_metadata = [
            'source_system',
            'source_version',
            'environment',
            'region'
        ]
        self.optional_metadata = [
            'user_id',
            'session_id',
            'request_id',
            'trace_id',
            'span_id'
        ]
    
    def enrich_metadata(self, event):
        '''Enrich event with standard metadata'''
        if 'metadata' not in event:
            event['metadata'] = {}
        
        # Add standard metadata
        event['metadata'].update({
            'source_system': 'ultraplatform',
            'source_version': '2.0',
            'environment': 'production',
            'region': 'au-sydney',
            'processed_at': datetime.now().isoformat()
        })
        
        return event
    
    def validate_metadata(self, metadata):
        '''Validate metadata completeness'''
        missing = []
        for field in self.required_metadata:
            if field not in metadata:
                missing.append(field)
        
        return len(missing) == 0, missing

class LifecycleManager:
    '''Manage event lifecycle'''
    
    def __init__(self):
        self.lifecycle_states = [
            'created',
            'validated',
            'published',
            'processing',
            'processed',
            'archived'
        ]
        self.retention_policies = {
            EventDomain.TRADING: 7 * 365,  # 7 years
            EventDomain.COMPLIANCE: 7 * 365,  # 7 years
            EventDomain.RISK: 5 * 365,  # 5 years
            EventDomain.PORTFOLIO: 5 * 365,  # 5 years
            EventDomain.MARKET: 2 * 365,  # 2 years
            EventDomain.SYSTEM: 1 * 365,  # 1 year
            EventDomain.ANALYTICS: 1 * 365,  # 1 year
            EventDomain.CLIENT: 90  # 90 days
        }
    
    def get_retention_days(self, event_type_def):
        '''Get retention period for event type'''
        return self.retention_policies.get(event_type_def.domain, 365)
    
    def should_archive(self, event, event_type_def):
        '''Check if event should be archived'''
        if 'timestamp' not in event:
            return False
        
        event_date = datetime.fromisoformat(event['timestamp'])
        age_days = (datetime.now() - event_date).days
        retention_days = self.get_retention_days(event_type_def)
        
        return age_days > retention_days * 0.9  # Archive at 90% of retention

# Demonstrate taxonomy
if __name__ == '__main__':
    print('📋 EVENT TYPES AND TAXONOMY - ULTRAPLATFORM')
    print('='*80)
    
    taxonomy = EventTaxonomy()
    
    # Display taxonomy
    taxonomy.display_taxonomy()
    
    # Show event statistics
    print('\n' + '='*80)
    print('EVENT STATISTICS')
    print('='*80)
    
    catalog = taxonomy.event_catalog
    total = len(catalog.get_all_events())
    
    print(f'Total Event Types: {total}')
    print('\nBy Domain:')
    for domain in EventDomain:
        count = len(catalog.get_events_by_domain(domain))
        percentage = (count/total*100) if total > 0 else 0
        print(f'  {domain.value}: {count} ({percentage:.1f}%)')
    
    print('\nBy Category:')
    category_counts = {}
    for event in catalog.get_all_events():
        cat = event.category
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    for category, count in category_counts.items():
        percentage = (count/total*100) if total > 0 else 0
        print(f'  {category.value}: {count} ({percentage:.1f}%)')
    
    print('\nBy Severity:')
    severity_counts = {}
    for event in catalog.get_all_events():
        sev = event.severity
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
    
    for severity, count in severity_counts.items():
        percentage = (count/total*100) if total > 0 else 0
        print(f'  {severity.value}: {count} ({percentage:.1f}%)')
    
    # Show sample event chain
    print('\n' + '='*80)
    print('EVENT CHAIN EXAMPLE')
    print('='*80)
    
    chain = taxonomy.relationship_manager.get_event_chain('trade.requested')
    print('Trade Execution Chain:')
    for i, event in enumerate(chain):
        print(f'  {i+1}. {event}')
    
    print('\n✅ Event Taxonomy Complete!')
