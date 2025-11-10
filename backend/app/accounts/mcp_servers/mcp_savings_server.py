"""
MCP Server for Savings Account Operations
Standardized interface via Model Context Protocol
"""

from typing import Dict, List
from datetime import datetime

class SavingsAccountMCPServer:
    """MCP Server exposing savings account tools"""
    
    def __init__(self, savings_service):
        self.service = savings_service
        self.server_uri = "mcp://turingwealth/savings-accounts"
    
    async def register_tools(self) -> List[Dict]:
        """Register available MCP tools"""
        return [
            {
                "name": "create_savings_account",
                "description": "Create new savings/CMA account",
                "parameters": {
                    "client_id": {"type": "string", "required": True},
                    "product_id": {"type": "string", "required": True},
                    "initial_deposit": {"type": "decimal", "required": False}
                }
            },
            {
                "name": "calculate_interest",
                "description": "Calculate daily interest for account",
                "parameters": {
                    "account_id": {"type": "string", "required": True},
                    "as_of_date": {"type": "date", "required": True}
                }
            },
            {
                "name": "post_interest",
                "description": "Post accrued interest to account",
                "parameters": {
                    "account_id": {"type": "string", "required": True}
                }
            },
            {
                "name": "get_account_balance",
                "description": "Get current account balance",
                "parameters": {
                    "account_id": {"type": "string", "required": True}
                }
            }
        ]
    
    async def execute_tool(self, tool_name: str, params: Dict) -> Dict:
        """Execute MCP tool"""
        
        if tool_name == "create_savings_account":
            return await self.service.create_account(**params)
        elif tool_name == "calculate_interest":
            return await self.service.calculate_daily_interest(**params)
        elif tool_name == "post_interest":
            return await self.service.post_monthly_interest(**params)
        elif tool_name == "get_account_balance":
            return await self._get_balance(params["account_id"])
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    async def _get_balance(self, account_id: str) -> Dict:
        """Get account balance"""
        account = await self.service.datamesh.account_data.get_account(account_id)
        return {
            "account_id": account_id,
            "balance": account["balance"],
            "available_balance": account["available_balance"],
            "accrued_interest": account["accrued_interest_ytd"]
        }
