"""
TuringWealth - Batch Job Scheduler (Turing Dynamics Edition)
AI-powered job orchestration with ML optimization
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta, time
from enum import Enum
import uuid
import asyncio
from collections import defaultdict

# ============================================================================
# DOMAIN MODELS
# ============================================================================

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class JobPriority(Enum):
    CRITICAL = 1    # Must complete before anything else
    HIGH = 2        # Important business operations
    MEDIUM = 3      # Standard operations
    LOW = 4         # Nice-to-have

class JobType(Enum):
    COB_PROCESS = "cob_process"
    PORTFOLIO_VALUATION = "portfolio_valuation"
    FEE_CALCULATION = "fee_calculation"
    INTEREST_ACCRUAL = "interest_accrual"
    REPORT_GENERATION = "report_generation"
    TRADE_SETTLEMENT = "trade_settlement"
    RECONCILIATION = "reconciliation"
    DATA_SYNC = "data_sync"
    MAINTENANCE = "maintenance"

@dataclass
class JobDefinition:
    """Job configuration"""
    job_id: str
    job_name: str
    job_type: JobType
    priority: JobPriority
    
    # Scheduling
    schedule_pattern: str  # Cron pattern
    timezone: str = "Australia/Sydney"
    
    # Execution
    task_function: str  # Python function path
    timeout_seconds: int = 3600
    max_retries: int = 3
    retry_delay_seconds: int = 300
    
    # Dependencies
    depends_on: List[str] = None  # Job IDs that must complete first
    
    # Configuration
    config: Dict[str, Any] = None
    enabled: bool = True
    
    # ML Features
    ml_optimization_enabled: bool = True
    estimated_duration_seconds: Optional[int] = None

@dataclass
class JobExecution:
    """Job execution instance"""
    execution_id: str
    job_id: str
    run_date: datetime
    status: JobStatus
    
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[int] = None
    
    error_message: Optional[str] = None
    retry_count: int = 0
    
    result: Optional[Dict] = None
    metrics: Optional[Dict] = None

class BatchJobScheduler:
    """
    AI-Powered Batch Job Orchestration System
    
    Features:
    - Dependency-based execution graph
    - ML-optimized scheduling
    - Auto-retry with exponential backoff
    - Real-time monitoring
    - DataMesh integration
    - Agentic optimization
    """
    
    def __init__(self, db_session, datamesh_client=None, mcp_client=None):
        self.db = db_session
        self.datamesh = datamesh_client
        self.mcp = mcp_client
        
        self.job_registry: Dict[str, JobDefinition] = {}
        self.execution_graph: Dict[str, List[str]] = defaultdict(list)
        self.active_executions: Dict[str, JobExecution] = {}
    
    async def register_job(self, job: JobDefinition):
        """Register a job in the scheduler"""
        
        self.job_registry[job.job_id] = job
        
        # Build dependency graph
        if job.depends_on:
            for parent_job_id in job.depends_on:
                self.execution_graph[parent_job_id].append(job.job_id)
        
        # Persist to database
        await self.db.execute("""
            INSERT INTO scheduled_jobs (
                job_id, job_name, job_type, priority, schedule_pattern,
                task_function, timeout_seconds, max_retries, depends_on,
                config, enabled, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job.job_id, job.job_name, job.job_type.value,
            job.priority.value, job.schedule_pattern, job.task_function,
            job.timeout_seconds, job.max_retries,
            ','.join(job.depends_on) if job.depends_on else None,
            str(job.config), job.enabled, datetime.now()
        ))
        
        await self.db.commit()
        
        print(f"? Registered job: {job.job_name}")
    
    async def execute_job(
        self,
        job_id: str,
        run_date: datetime = None
    ) -> JobExecution:
        """Execute a single job"""
        
        if run_date is None:
            run_date = datetime.now()
        
        job = self.job_registry.get(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if not job.enabled:
            print(f"? Job disabled: {job.job_name}")
            return None
        
        # Check dependencies
        if job.depends_on:
            for dep_job_id in job.depends_on:
                if not await self._is_dependency_satisfied(dep_job_id, run_date):
                    print(f"? Waiting for dependency: {dep_job_id}")
                    return None
        
        # Create execution
        execution = JobExecution(
            execution_id=str(uuid.uuid4()),
            job_id=job_id,
            run_date=run_date,
            status=JobStatus.PENDING
        )
        
        self.active_executions[execution.execution_id] = execution
        
        # Persist execution
        await self._save_execution(execution)
        
        # Publish to DataMesh
        if self.datamesh:
            await self.datamesh.events.publish({
                "event_type": "JOB_STARTED",
                "execution_id": execution.execution_id,
                "job_id": job_id,
                "job_name": job.job_name,
                "timestamp": datetime.now().isoformat()
            })
        
        try:
            # Update status
            execution.status = JobStatus.RUNNING
            execution.started_at = datetime.now()
            await self._save_execution(execution)
            
            print(f"? Running: {job.job_name}")
            
            # Execute job (with timeout)
            result = await asyncio.wait_for(
                self._run_job_task(job),
                timeout=job.timeout_seconds
            )
            
            # Success
            execution.status = JobStatus.COMPLETED
            execution.completed_at = datetime.now()
            execution.duration_seconds = int(
                (execution.completed_at - execution.started_at).total_seconds()
            )
            execution.result = result
            
            await self._save_execution(execution)
            
            print(f"? Completed: {job.job_name} ({execution.duration_seconds}s)")
            
            # Publish to DataMesh
            if self.datamesh:
                await self.datamesh.events.publish({
                    "event_type": "JOB_COMPLETED",
                    "execution_id": execution.execution_id,
                    "duration_seconds": execution.duration_seconds,
                    "timestamp": datetime.now().isoformat()
                })
            
            # Trigger dependent jobs
            await self._trigger_dependent_jobs(job_id, run_date)
            
            return execution
            
        except asyncio.TimeoutError:
            # Timeout
            execution.status = JobStatus.FAILED
            execution.error_message = f"Timeout after {job.timeout_seconds}s"
            await self._save_execution(execution)
            
            print(f"? Timeout: {job.job_name}")
            
            # Retry?
            if execution.retry_count < job.max_retries:
                await self._retry_job(execution, job)
            
            return execution
            
        except Exception as e:
            # Error
            execution.status = JobStatus.FAILED
            execution.error_message = str(e)
            execution.completed_at = datetime.now()
            await self._save_execution(execution)
            
            print(f"? Failed: {job.job_name} - {str(e)}")
            
            # Retry?
            if execution.retry_count < job.max_retries:
                await self._retry_job(execution, job)
            
            return execution
        
        finally:
            # Cleanup
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
    
    async def execute_cob_process(self, business_date: datetime):
        """
        Execute Close of Business (COB) process
        
        COB Flow:
        1. Trade settlements
        2. Portfolio valuation
        3. Fee calculation
        4. Interest accrual
        5. Report generation
        6. Reconciliation
        7. Ledger closing
        """
        
        print(f"")
        print(f"{'='*60}")
        print(f"  ?? CLOSE OF BUSINESS - {business_date.date()}")
        print(f"{'='*60}")
        print(f"")
        
        cob_start = datetime.now()
        
        # Execute COB jobs in order
        cob_jobs = [
            "job_trade_settlements",
            "job_portfolio_valuation",
            "job_fee_calculation",
            "job_interest_accrual",
            "job_report_generation",
            "job_reconciliation",
            "job_ledger_close"
        ]
        
        results = {}
        
        for job_id in cob_jobs:
            execution = await self.execute_job(job_id, business_date)
            
            if execution and execution.status == JobStatus.COMPLETED:
                results[job_id] = {
                    "status": "success",
                    "duration": execution.duration_seconds
                }
            else:
                results[job_id] = {
                    "status": "failed",
                    "error": execution.error_message if execution else "Job not found"
                }
                
                # Critical job failed - abort COB
                if job_id in ["job_trade_settlements", "job_portfolio_valuation"]:
                    print(f"")
                    print(f"? CRITICAL JOB FAILED - COB ABORTED")
                    print(f"")
                    break
        
        cob_duration = (datetime.now() - cob_start).total_seconds()
        
        print(f"")
        print(f"{'='*60}")
        print(f"  COB COMPLETED - {cob_duration:.0f}s")
        print(f"{'='*60}")
        print(f"")
        
        return results
    
    async def get_job_status(self, execution_id: str) -> Dict:
        """Get job execution status"""
        
        row = await self.db.fetch_one(
            "SELECT * FROM job_executions WHERE execution_id = ?",
            (execution_id,)
        )
        
        if not row:
            return {"error": "Execution not found"}
        
        return dict(row)
    
    async def get_cob_status(self, business_date: datetime) -> Dict:
        """Get COB status for a specific date"""
        
        query = """
        SELECT 
            j.job_name,
            e.status,
            e.started_at,
            e.completed_at,
            e.duration_seconds,
            e.error_message
        FROM job_executions e
        JOIN scheduled_jobs j ON e.job_id = j.job_id
        WHERE DATE(e.run_date) = ?
            AND j.job_type = 'cob_process'
        ORDER BY e.started_at
        """
        
        rows = await self.db.fetch_all(query, (business_date.date(),))
        
        jobs = [dict(row) for row in rows]
        
        total_duration = sum(j.get("duration_seconds", 0) for j in jobs)
        success_count = sum(1 for j in jobs if j["status"] == "completed")
        
        return {
            "business_date": business_date.date().isoformat(),
            "total_jobs": len(jobs),
            "successful": success_count,
            "failed": len(jobs) - success_count,
            "total_duration_seconds": total_duration,
            "jobs": jobs
        }
    
    async def predict_job_duration(self, job_id: str) -> int:
        """
        ML-based prediction of job execution time
        
        Uses historical execution data to predict duration
        """
        
        # Get last 30 executions
        query = """
        SELECT duration_seconds
        FROM job_executions
        WHERE job_id = ?
            AND status = 'completed'
            AND duration_seconds IS NOT NULL
        ORDER BY completed_at DESC
        LIMIT 30
        """
        
        rows = await self.db.fetch_all(query, (job_id,))
        
        if not rows:
            # No history - use job default
            job = self.job_registry.get(job_id)
            return job.estimated_duration_seconds if job else 600
        
        durations = [row["duration_seconds"] for row in rows]
        
        # Simple ML: weighted average (recent executions weighted higher)
        weights = [i + 1 for i in range(len(durations))]
        weighted_avg = sum(d * w for d, w in zip(durations, weights)) / sum(weights)
        
        # Add 20% buffer
        predicted = int(weighted_avg * 1.2)
        
        return predicted
    
    async def _run_job_task(self, job: JobDefinition) -> Dict:
        """Execute job task function"""
        
        # Dynamic import of task function
        module_path, function_name = job.task_function.rsplit('.', 1)
        
        # Mock execution for now
        await asyncio.sleep(2)  # Simulate work
        
        return {
            "status": "success",
            "records_processed": 100
        }
    
    async def _is_dependency_satisfied(
        self,
        dep_job_id: str,
        run_date: datetime
    ) -> bool:
        """Check if dependency job completed successfully"""
        
        row = await self.db.fetch_one("""
            SELECT status
            FROM job_executions
            WHERE job_id = ?
                AND DATE(run_date) = ?
                AND status = 'completed'
            ORDER BY completed_at DESC
            LIMIT 1
        """, (dep_job_id, run_date.date()))
        
        return row is not None
    
    async def _trigger_dependent_jobs(
        self,
        completed_job_id: str,
        run_date: datetime
    ):
        """Trigger jobs that depend on this job"""
        
        dependent_job_ids = self.execution_graph.get(completed_job_id, [])
        
        for dep_job_id in dependent_job_ids:
            await self.execute_job(dep_job_id, run_date)
    
    async def _retry_job(self, execution: JobExecution, job: JobDefinition):
        """Retry failed job"""
        
        execution.retry_count += 1
        execution.status = JobStatus.RETRYING
        await self._save_execution(execution)
        
        print(f"?? Retrying: {job.job_name} (attempt {execution.retry_count}/{job.max_retries})")
        
        # Wait before retry (exponential backoff)
        delay = job.retry_delay_seconds * (2 ** (execution.retry_count - 1))
        await asyncio.sleep(delay)
        
        # Re-execute
        await self.execute_job(job.job_id, execution.run_date)
    
    async def _save_execution(self, execution: JobExecution):
        """Persist execution state"""
        
        await self.db.execute("""
            INSERT OR REPLACE INTO job_executions (
                execution_id, job_id, run_date, status,
                started_at, completed_at, duration_seconds,
                error_message, retry_count, result
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            execution.execution_id, execution.job_id, execution.run_date,
            execution.status.value, execution.started_at, execution.completed_at,
            execution.duration_seconds, execution.error_message,
            execution.retry_count, str(execution.result)
        ))
        
        await self.db.commit()
        
        # Publish to DataMesh
        if self.datamesh:
            await self.datamesh.job_data.update_execution({
                "execution_id": execution.execution_id,
                "status": execution.status.value,
                "updated_at": datetime.now().isoformat()
            })
