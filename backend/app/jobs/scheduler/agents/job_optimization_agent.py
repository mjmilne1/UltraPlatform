"""
AI Job Optimization Agent
Autonomous optimization of job scheduling and execution
"""

from typing import Dict, List
from datetime import datetime, time

class JobOptimizationAgent:
    """
    Autonomous agent for job scheduling optimization
    
    Optimizations:
    - Dynamic schedule adjustment
    - Resource-based prioritization
    - Execution time optimization
    - Dependency reordering
    - Load balancing
    """
    
    def __init__(self, scheduler, memory):
        self.scheduler = scheduler
        self.memory = memory
        self.name = "JobOptimizationAgent"
    
    async def run_optimization(self):
        """Run daily optimization"""
        
        print(f"[{self.name}] Starting job optimization...")
        
        # Analyze job performance
        insights = await self._analyze_job_performance()
        
        # Optimize schedules
        adjustments = []
        
        for job_id, metrics in insights.items():
            # Check if job consistently takes longer
            if metrics["avg_duration"] > metrics["expected_duration"] * 1.5:
                adjustment = await self._suggest_schedule_adjustment(
                    job_id,
                    metrics
                )
                if adjustment:
                    adjustments.append(adjustment)
        
        # Apply optimizations
        for adjustment in adjustments:
            await self._apply_adjustment(adjustment)
        
        print(f"[{self.name}] Optimization complete. {len(adjustments)} adjustments made.")
    
    async def predict_optimal_start_time(
        self,
        job_id: str,
        target_completion_time: time
    ) -> time:
        """
        Predict optimal start time to meet target completion
        
        Uses ML to predict:
        - Expected duration
        - System load at different times
        - Historical success rates
        """
        
        # Get predicted duration
        predicted_duration = await self.scheduler.predict_job_duration(job_id)
        
        # Calculate start time
        target_datetime = datetime.combine(datetime.today(), target_completion_time)
        optimal_start = target_datetime - timedelta(seconds=predicted_duration)
        
        # Add buffer (10%)
        buffer = predicted_duration * 0.1
        optimal_start = optimal_start - timedelta(seconds=buffer)
        
        return optimal_start.time()
    
    async def optimize_cob_sequence(self) -> List[str]:
        """
        Optimize COB job execution sequence
        
        Considers:
        - Job dependencies
        - Historical durations
        - Resource requirements
        - Parallel execution opportunities
        """
        
        # Current COB sequence
        current_sequence = [
            "job_trade_settlements",
            "job_portfolio_valuation",
            "job_fee_calculation",
            "job_interest_accrual",
            "job_report_generation",
            "job_reconciliation",
            "job_ledger_close"
        ]
        
        # Analyze which jobs can run in parallel
        parallel_opportunities = await self._find_parallel_jobs(current_sequence)
        
        # Reorder for optimal performance
        optimized_sequence = await self._reorder_for_speed(
            current_sequence,
            parallel_opportunities
        )
        
        return optimized_sequence
    
    async def _analyze_job_performance(self) -> Dict:
        """Analyze job performance metrics"""
        
        insights = {}
        
        for job_id in self.scheduler.job_registry.keys():
            # Get last 30 executions
            executions = await self._get_recent_executions(job_id, limit=30)
            
            if executions:
                avg_duration = sum(e["duration_seconds"] for e in executions) / len(executions)
                success_rate = sum(1 for e in executions if e["status"] == "completed") / len(executions)
                
                insights[job_id] = {
                    "avg_duration": avg_duration,
                    "expected_duration": self.scheduler.job_registry[job_id].estimated_duration_seconds or 600,
                    "success_rate": success_rate,
                    "total_executions": len(executions)
                }
        
        return insights
    
    async def _suggest_schedule_adjustment(
        self,
        job_id: str,
        metrics: Dict
    ) -> Optional[Dict]:
        """Suggest schedule adjustment"""
        
        if metrics["avg_duration"] > metrics["expected_duration"] * 2:
            return {
                "job_id": job_id,
                "action": "INCREASE_TIMEOUT",
                "new_timeout": int(metrics["avg_duration"] * 1.5),
                "reason": f"Job consistently exceeds timeout"
            }
        
        return None
    
    async def _apply_adjustment(self, adjustment: Dict):
        """Apply optimization adjustment"""
        
        job_id = adjustment["job_id"]
        job = self.scheduler.job_registry[job_id]
        
        if adjustment["action"] == "INCREASE_TIMEOUT":
            job.timeout_seconds = adjustment["new_timeout"]
            print(f"  ? Increased timeout for {job.job_name}: {adjustment['new_timeout']}s")
    
    async def _get_recent_executions(self, job_id: str, limit: int = 30) -> List[Dict]:
        """Get recent job executions"""
        # Mock - replace with actual query
        return []
    
    async def _find_parallel_jobs(self, sequence: List[str]) -> Dict:
        """Find jobs that can run in parallel"""
        return {}
    
    async def _reorder_for_speed(
        self,
        sequence: List[str],
        parallel_opportunities: Dict
    ) -> List[str]:
        """Reorder jobs for optimal speed"""
        return sequence
