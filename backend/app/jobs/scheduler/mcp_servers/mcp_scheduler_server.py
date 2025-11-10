"""
MCP Server for Batch Job Scheduler
Standardized interface for job control operations
"""

from typing import Dict, List
from datetime import datetime

class SchedulerMCPServer:
    """MCP Server exposing scheduler tools"""
    
    def __init__(self, scheduler_service):
        self.service = scheduler_service
        self.server_uri = "mcp://turingwealth/scheduler"
    
    async def register_tools(self) -> List[Dict]:
        """Register available MCP tools"""
        return [
            {
                "name": "execute_job",
                "description": "Execute a scheduled job manually",
                "parameters": {
                    "job_id": {"type": "string", "required": True},
                    "run_date": {"type": "datetime", "required": False}
                }
            },
            {
                "name": "execute_cob",
                "description": "Execute Close of Business process",
                "parameters": {
                    "business_date": {"type": "datetime", "required": True}
                }
            },
            {
                "name": "get_job_status",
                "description": "Get status of job execution",
                "parameters": {
                    "execution_id": {"type": "string", "required": True}
                }
            },
            {
                "name": "get_cob_status",
                "description": "Get COB status for a date",
                "parameters": {
                    "business_date": {"type": "datetime", "required": True}
                }
            },
            {
                "name": "predict_job_duration",
                "description": "ML-based prediction of job runtime",
                "parameters": {
                    "job_id": {"type": "string", "required": True}
                }
            },
            {
                "name": "cancel_job",
                "description": "Cancel running job",
                "parameters": {
                    "execution_id": {"type": "string", "required": True}
                }
            },
            {
                "name": "list_active_jobs",
                "description": "List all currently running jobs",
                "parameters": {}
            },
            {
                "name": "get_job_history",
                "description": "Get execution history for a job",
                "parameters": {
                    "job_id": {"type": "string", "required": True},
                    "days": {"type": "integer", "required": False, "default": 7}
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute MCP tool"""
        
        if tool_name == "execute_job":
            return await self.service.execute_job(**params)
        elif tool_name == "execute_cob":
            return await self.service.execute_cob_process(**params)
        elif tool_name == "get_job_status":
            return await self.service.get_job_status(**params)
        elif tool_name == "get_cob_status":
            return await self.service.get_cob_status(**params)
        elif tool_name == "predict_job_duration":
            duration = await self.service.predict_job_duration(**params)
            return {"predicted_duration_seconds": duration}
        elif tool_name == "list_active_jobs":
            return {"active_jobs": list(self.service.active_executions.values())}
        elif tool_name == "cancel_job":
            return await self._cancel_job(params["execution_id"])
        elif tool_name == "get_job_history":
            return await self._get_job_history(**params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _cancel_job(self, execution_id: str) -> Dict:
        """Cancel running job"""
        if execution_id in self.service.active_executions:
            execution = self.service.active_executions[execution_id]
            execution.status = "cancelled"
            return {"status": "cancelled", "execution_id": execution_id}
        return {"error": "Job not found or not running"}
    
    async def _get_job_history(self, job_id: str, days: int = 7) -> Dict:
        """Get job execution history"""
        # Mock - replace with actual query
        return {
            "job_id": job_id,
            "executions": []
        }
