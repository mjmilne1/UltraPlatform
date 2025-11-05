from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
import heapq
import json

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

class TaskType(Enum):
    TRADING = 'trading'
    PORTFOLIO = 'portfolio'
    RISK = 'risk'
    COMPLIANCE = 'compliance'
    ANALYTICS = 'analytics'
    DATA = 'data'
    MAINTENANCE = 'maintenance'

class TaskStatus(Enum):
    PENDING = 'pending'
    QUEUED = 'queued'
    ROUTING = 'routing'
    SCHEDULED = 'scheduled'
    EXECUTING = 'executing'
    COMPLETED = 'completed'
    FAILED = 'failed'
    CANCELLED = 'cancelled'

class TaskRoutingSchedulingSystem:
    '''Intelligent Task Routing & Scheduling for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Task Routing & Scheduling'
        self.version = '2.0'
        self.router = IntelligentRouter()
        self.scheduler = AdvancedScheduler()
        self.queue_manager = QueueManager()
        self.worker_pool = WorkerPool()
        self.load_balancer = LoadBalancer()
        self.task_optimizer = TaskOptimizer()
        self.monitoring = RoutingMonitoring()
        
    def process_task(self, task):
        '''Process a task through the routing and scheduling system'''
        print('TASK ROUTING & SCHEDULING')
        print('='*70)
        print(f'Task ID: {task.id}')
        print(f'Task Type: {task.task_type.value}')
        print(f'Priority: {task.priority.value}')
        print()
        
        # Step 1: Route task
        print('1️⃣ ROUTING TASK')
        print('-'*40)
        routing_decision = self.router.route_task(task)
        print(f'  Target Worker: {routing_decision["worker"]}')
        print(f'  Routing Strategy: {routing_decision["strategy"]}')
        print(f'  Estimated Time: {routing_decision["estimated_time"]}s')
        
        # Step 2: Queue management
        print('\n2️⃣ QUEUE MANAGEMENT')
        print('-'*40)
        queue_position = self.queue_manager.enqueue(task, routing_decision['worker'])
        print(f'  Queue: {routing_decision["worker"]}')
        print(f'  Position: {queue_position}')
        print(f'  Queue Length: {self.queue_manager.get_queue_length(routing_decision["worker"])}')
        
        # Step 3: Schedule task
        print('\n3️⃣ SCHEDULING')
        print('-'*40)
        schedule = self.scheduler.schedule_task(task)
        print(f'  Scheduled Time: {schedule["start_time"]}')
        print(f'  Execution Window: {schedule["window"]}')
        print(f'  Dependencies: {len(schedule["dependencies"])} tasks')
        
        # Step 4: Load balancing
        print('\n4️⃣ LOAD BALANCING')
        print('-'*40)
        balanced = self.load_balancer.balance_load(routing_decision['worker'])
        print(f'  Current Load: {balanced["current_load"]:.1%}')
        print(f'  Target Load: {balanced["target_load"]:.1%}')
        print(f'  Status: {"✅ Balanced" if balanced["is_balanced"] else "⚠️ Rebalancing"}')
        
        # Step 5: Optimization
        print('\n5️⃣ OPTIMIZATION')
        print('-'*40)
        optimized = self.task_optimizer.optimize_execution(task, routing_decision)
        print(f'  Batch Processing: {"Yes" if optimized["batch"] else "No"}')
        print(f'  Parallelization: {"Yes" if optimized["parallel"] else "No"}')
        print(f'  Resource Allocation: {optimized["resources"]}')
        
        # Step 6: Execute
        print('\n6️⃣ EXECUTION')
        print('-'*40)
        worker = self.worker_pool.get_worker(routing_decision['worker'])
        result = worker.execute(task)
        print(f'  Status: {result["status"]}')
        print(f'  Execution Time: {result["execution_time"]}ms')
        print(f'  Result: {result["message"]}')
        
        return result

class Task:
    '''Task object'''
    
    def __init__(self, task_id, name, task_type, priority=TaskPriority.NORMAL):
        self.id = task_id
        self.name = name
        self.task_type = task_type
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.created = datetime.now()
        self.dependencies = []
        self.metadata = {}
        
    def __lt__(self, other):
        '''For priority queue comparison'''
        return self.priority.value < other.priority.value

class IntelligentRouter:
    '''Intelligent task routing'''
    
    def __init__(self):
        self.routing_rules = self._initialize_routing_rules()
        self.routing_history = []
        
    def _initialize_routing_rules(self):
        '''Initialize routing rules'''
        return {
            TaskType.TRADING: {
                'preferred_workers': ['trading_worker_1', 'trading_worker_2'],
                'strategy': 'least_loaded',
                'requirements': ['low_latency', 'high_throughput']
            },
            TaskType.PORTFOLIO: {
                'preferred_workers': ['portfolio_worker'],
                'strategy': 'dedicated',
                'requirements': ['high_accuracy']
            },
            TaskType.RISK: {
                'preferred_workers': ['risk_worker'],
                'strategy': 'dedicated',
                'requirements': ['high_compute']
            },
            TaskType.COMPLIANCE: {
                'preferred_workers': ['compliance_worker'],
                'strategy': 'dedicated',
                'requirements': ['audit_trail']
            },
            TaskType.ANALYTICS: {
                'preferred_workers': ['analytics_worker_1', 'analytics_worker_2'],
                'strategy': 'round_robin',
                'requirements': ['batch_processing']
            },
            TaskType.DATA: {
                'preferred_workers': ['data_worker_1', 'data_worker_2', 'data_worker_3'],
                'strategy': 'load_balanced',
                'requirements': ['high_memory']
            }
        }
    
    def route_task(self, task):
        '''Route task to appropriate worker'''
        rules = self.routing_rules.get(task.task_type, {})
        
        # Select worker based on strategy
        if rules.get('strategy') == 'least_loaded':
            worker = self._select_least_loaded_worker(rules['preferred_workers'])
        elif rules.get('strategy') == 'round_robin':
            worker = self._select_round_robin(rules['preferred_workers'])
        elif rules.get('strategy') == 'dedicated':
            worker = rules['preferred_workers'][0]
        else:
            worker = self._select_load_balanced(rules['preferred_workers'])
        
        routing_decision = {
            'worker': worker,
            'strategy': rules.get('strategy', 'default'),
            'estimated_time': self._estimate_execution_time(task),
            'requirements': rules.get('requirements', [])
        }
        
        # Track routing decision
        self.routing_history.append({
            'task_id': task.id,
            'worker': worker,
            'timestamp': datetime.now()
        })
        
        return routing_decision
    
    def _select_least_loaded_worker(self, workers):
        '''Select worker with least load'''
        # Simulated selection
        return workers[0]
    
    def _select_round_robin(self, workers):
        '''Round robin selection'''
        # Simulated selection
        return workers[len(self.routing_history) % len(workers)]
    
    def _select_load_balanced(self, workers):
        '''Load balanced selection'''
        # Simulated selection
        return workers[0]
    
    def _estimate_execution_time(self, task):
        '''Estimate task execution time'''
        estimates = {
            TaskType.TRADING: 50,
            TaskType.PORTFOLIO: 200,
            TaskType.RISK: 500,
            TaskType.COMPLIANCE: 300,
            TaskType.ANALYTICS: 1000,
            TaskType.DATA: 2000
        }
        return estimates.get(task.task_type, 100)

class AdvancedScheduler:
    '''Advanced task scheduler'''
    
    def __init__(self):
        self.schedule = []
        self.scheduling_policies = {
            'market_hours': {'start': '09:00', 'end': '16:00'},
            'batch_window': {'start': '00:00', 'end': '06:00'},
            'maintenance_window': {'start': '02:00', 'end': '04:00'}
        }
        
    def schedule_task(self, task):
        '''Schedule task execution'''
        # Determine scheduling window
        if task.task_type == TaskType.TRADING:
            window = 'market_hours'
        elif task.task_type in [TaskType.DATA, TaskType.ANALYTICS]:
            window = 'batch_window'
        elif task.task_type == TaskType.MAINTENANCE:
            window = 'maintenance_window'
        else:
            window = 'anytime'
        
        # Calculate start time
        if task.priority == TaskPriority.CRITICAL:
            start_time = datetime.now()
        elif task.priority == TaskPriority.HIGH:
            start_time = datetime.now() + timedelta(minutes=5)
        else:
            start_time = datetime.now() + timedelta(minutes=15)
        
        schedule_entry = {
            'task_id': task.id,
            'start_time': start_time,
            'window': window,
            'dependencies': task.dependencies,
            'retry_policy': self._get_retry_policy(task)
        }
        
        self.schedule.append(schedule_entry)
        
        return schedule_entry
    
    def _get_retry_policy(self, task):
        '''Get retry policy for task'''
        if task.priority == TaskPriority.CRITICAL:
            return {'max_retries': 5, 'backoff': 'exponential'}
        elif task.priority == TaskPriority.HIGH:
            return {'max_retries': 3, 'backoff': 'linear'}
        else:
            return {'max_retries': 1, 'backoff': 'none'}

class QueueManager:
    '''Task queue management'''
    
    def __init__(self):
        self.queues = {}
        self.priority_queues = {}
        
    def enqueue(self, task, worker):
        '''Add task to queue'''
        if worker not in self.priority_queues:
            self.priority_queues[worker] = []
        
        heapq.heappush(self.priority_queues[worker], (task.priority.value, task))
        
        return len(self.priority_queues[worker])
    
    def dequeue(self, worker):
        '''Remove task from queue'''
        if worker in self.priority_queues and self.priority_queues[worker]:
            _, task = heapq.heappop(self.priority_queues[worker])
            return task
        return None
    
    def get_queue_length(self, worker):
        '''Get queue length for worker'''
        if worker in self.priority_queues:
            return len(self.priority_queues[worker])
        return 0
    
    def get_queue_status(self):
        '''Get status of all queues'''
        status = {}
        for worker, queue in self.priority_queues.items():
            status[worker] = {
                'length': len(queue),
                'oldest_task': queue[0][1].created if queue else None
            }
        return status

class WorkerPool:
    '''Pool of worker processes'''
    
    def __init__(self):
        self.workers = self._initialize_workers()
        
    def _initialize_workers(self):
        '''Initialize worker pool'''
        return {
            'trading_worker_1': Worker('trading_worker_1', TaskType.TRADING),
            'trading_worker_2': Worker('trading_worker_2', TaskType.TRADING),
            'portfolio_worker': Worker('portfolio_worker', TaskType.PORTFOLIO),
            'risk_worker': Worker('risk_worker', TaskType.RISK),
            'compliance_worker': Worker('compliance_worker', TaskType.COMPLIANCE),
            'analytics_worker_1': Worker('analytics_worker_1', TaskType.ANALYTICS),
            'analytics_worker_2': Worker('analytics_worker_2', TaskType.ANALYTICS),
            'data_worker_1': Worker('data_worker_1', TaskType.DATA),
            'data_worker_2': Worker('data_worker_2', TaskType.DATA),
            'data_worker_3': Worker('data_worker_3', TaskType.DATA)
        }
    
    def get_worker(self, worker_name):
        '''Get worker by name'''
        return self.workers.get(worker_name)
    
    def get_available_workers(self):
        '''Get available workers'''
        return [w for w in self.workers.values() if w.is_available()]

class Worker:
    '''Worker process'''
    
    def __init__(self, name, specialization):
        self.name = name
        self.specialization = specialization
        self.status = 'idle'
        self.current_task = None
        self.completed_tasks = 0
        self.failed_tasks = 0
        
    def execute(self, task):
        '''Execute task'''
        self.status = 'busy'
        self.current_task = task
        
        # Simulate execution
        start_time = datetime.now()
        
        # Task execution logic
        result = {
            'status': 'completed',
            'execution_time': 125,
            'message': f'Task {task.id} completed successfully',
            'output': {}
        }
        
        # Update stats
        self.completed_tasks += 1
        self.status = 'idle'
        self.current_task = None
        
        return result
    
    def is_available(self):
        '''Check if worker is available'''
        return self.status == 'idle'

class LoadBalancer:
    '''Load balancing across workers'''
    
    def __init__(self):
        self.load_thresholds = {
            'low': 0.3,
            'normal': 0.6,
            'high': 0.8,
            'critical': 0.95
        }
        
    def balance_load(self, worker):
        '''Balance load for worker'''
        # Simulated load calculation
        current_load = 0.45  # 45% load
        
        return {
            'current_load': current_load,
            'target_load': 0.6,
            'is_balanced': 0.3 < current_load < 0.8,
            'recommendation': self._get_recommendation(current_load)
        }
    
    def _get_recommendation(self, load):
        '''Get load balancing recommendation'''
        if load < self.load_thresholds['low']:
            return 'Assign more tasks'
        elif load > self.load_thresholds['high']:
            return 'Redistribute tasks'
        return 'Optimal load'

class TaskOptimizer:
    '''Task execution optimization'''
    
    def optimize_execution(self, task, routing_decision):
        '''Optimize task execution'''
        optimization = {
            'batch': False,
            'parallel': False,
            'resources': 'standard',
            'cache': False
        }
        
        # Optimization logic
        if task.task_type == TaskType.ANALYTICS:
            optimization['batch'] = True
            optimization['resources'] = 'high_memory'
        elif task.task_type == TaskType.TRADING:
            optimization['cache'] = True
            optimization['resources'] = 'low_latency'
        elif task.task_type == TaskType.DATA:
            optimization['parallel'] = True
            optimization['resources'] = 'high_compute'
        
        return optimization

class RoutingMonitoring:
    '''Monitor routing and scheduling performance'''
    
    def get_metrics(self):
        '''Get routing metrics'''
        return {
            'total_tasks_routed': 5847,
            'avg_routing_time': 2.5,  # ms
            'routing_success_rate': 99.8,
            'queue_depths': {
                'trading': 5,
                'portfolio': 2,
                'risk': 1,
                'analytics': 8,
                'data': 12
            },
            'worker_utilization': {
                'trading_worker_1': 0.65,
                'trading_worker_2': 0.70,
                'portfolio_worker': 0.45,
                'risk_worker': 0.55,
                'analytics_worker_1': 0.80
            }
        }

# Demonstrate the system
if __name__ == '__main__':
    print('📅 TASK ROUTING & SCHEDULING - ULTRAPLATFORM')
    print('='*70)
    
    system = TaskRoutingSchedulingSystem()
    
    # Create sample tasks
    tasks = [
        Task('TSK-001', 'Execute Trade', TaskType.TRADING, TaskPriority.HIGH),
        Task('TSK-002', 'Calculate NAV', TaskType.PORTFOLIO, TaskPriority.CRITICAL),
        Task('TSK-003', 'Risk Assessment', TaskType.RISK, TaskPriority.NORMAL),
        Task('TSK-004', 'Generate Report', TaskType.ANALYTICS, TaskPriority.LOW),
        Task('TSK-005', 'Process Market Data', TaskType.DATA, TaskPriority.HIGH)
    ]
    
    # Process first task
    print(f'\nProcessing Task: {tasks[0].name}')
    print('='*70 + '\n')
    result = system.process_task(tasks[0])
    
    # Show system metrics
    print('\n' + '='*70)
    print('SYSTEM METRICS')
    print('='*70)
    metrics = system.monitoring.get_metrics()
    print(f'Total Tasks Routed: {metrics["total_tasks_routed"]:,}')
    print(f'Routing Success Rate: {metrics["routing_success_rate"]:.1f}%')
    print(f'Average Routing Time: {metrics["avg_routing_time"]}ms')
    
    print('\n✅ Task Routing & Scheduling System Operational!')
