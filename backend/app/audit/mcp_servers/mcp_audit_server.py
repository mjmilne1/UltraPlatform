"""
MCP Server for Audit & Approval System
Standardized interface for compliance operations
"""

from typing import Dict, List
from datetime import datetime

class AuditMCPServer:
    """MCP Server exposing audit and approval tools"""
    
    def __init__(self, audit_service):
        self.service = audit_service
        self.server_uri = "mcp://turingwealth/audit"
    
    async def register_tools(self) -> List[Dict]:
        """Register available MCP tools"""
        return [
            {
                "name": "create_approval_request",
                "description": "Create maker-checker approval request",
                "parameters": {
                    "maker_id": {"type": "string", "required": True},
                    "maker_email": {"type": "string", "required": True},
                    "entity_type": {"type": "string", "required": True},
                    "entity_id": {"type": "string", "required": True},
                    "change_type": {"type": "string", "required": True},
                    "proposed_changes": {"type": "object", "required": True},
                    "reason": {"type": "string", "required": False}
                }
            },
            {
                "name": "approve_request",
                "description": "Approve pending request (checker)",
                "parameters": {
                    "request_id": {"type": "string", "required": True},
                    "checker_id": {"type": "string", "required": True},
                    "checker_email": {"type": "string", "required": True},
                    "comment": {"type": "string", "required": False}
                }
            },
            {
                "name": "reject_request",
                "description": "Reject pending request (checker)",
                "parameters": {
                    "request_id": {"type": "string", "required": True},
                    "checker_id": {"type": "string", "required": True},
                    "checker_email": {"type": "string", "required": True},
                    "reason": {"type": "string", "required": True}
                }
            },
            {
                "name": "get_pending_approvals",
                "description": "Get pending approval requests",
                "parameters": {
                    "checker_id": {"type": "string", "required": False}
                }
            },
            {
                "name": "get_audit_trail",
                "description": "Query audit trail",
                "parameters": {
                    "entity_type": {"type": "string", "required": False},
                    "entity_id": {"type": "string", "required": False},
                    "user_id": {"type": "string", "required": False},
                    "start_date": {"type": "datetime", "required": False},
                    "end_date": {"type": "datetime", "required": False},
                    "limit": {"type": "integer", "required": False, "default": 100}
                }
            },
            {
                "name": "verify_audit_chain",
                "description": "Verify audit trail integrity",
                "parameters": {}
            },
            {
                "name": "get_risk_score",
                "description": "Get ML risk score for proposed change",
                "parameters": {
                    "entity_type": {"type": "string", "required": True},
                    "entity_id": {"type": "string", "required": True},
                    "change_type": {"type": "string", "required": True},
                    "proposed_changes": {"type": "object", "required": True}
                }
            },
            {
                "name": "get_compliance_report",
                "description": "Generate compliance report",
                "parameters": {
                    "start_date": {"type": "date", "required": True},
                    "end_date": {"type": "date", "required": True},
                    "report_type": {"type": "string", "required": False}
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute MCP tool"""
        
        if tool_name == "create_approval_request":
            return await self.service.create_approval_request(**params)
        elif tool_name == "approve_request":
            return await self.service.approve_request(**params)
        elif tool_name == "reject_request":
            return await self.service.reject_request(**params)
        elif tool_name == "get_pending_approvals":
            requests = await self.service.get_pending_approvals(**params)
            return {"pending_requests": [r.__dict__ for r in requests]}
        elif tool_name == "get_audit_trail":
            entries = await self.service.get_audit_trail(**params)
            return {"audit_entries": [e.__dict__ for e in entries]}
        elif tool_name == "verify_audit_chain":
            return await self.service.verify_audit_chain()
        elif tool_name == "get_risk_score":
            score = await self.service._calculate_change_risk(**params)
            return {"risk_score": score}
        elif tool_name == "get_compliance_report":
            return await self._generate_compliance_report(**params)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _generate_compliance_report(
        self,
        start_date,
        end_date,
        report_type="summary"
    ) -> Dict:
        """Generate compliance report"""
        
        # Get audit entries for period
        entries = await self.service.get_audit_trail(
            start_date=start_date,
            end_date=end_date,
            limit=10000
        )
        
        return {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "total_changes": len(entries),
            "high_risk_changes": sum(1 for e in entries if e.risk_level == "high"),
            "approval_requests": 0,  # Would query separately
            "rejected_requests": 0
        }
