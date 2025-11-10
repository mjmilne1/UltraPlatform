"""
Agentic AI - Interest Rate Optimization
Autonomous agent for dynamic rate management
"""

from typing import Dict
from datetime import datetime

class InterestOptimizationAgent:
    """
    Autonomous agent for interest rate optimization
    
    ReAct Pattern:
    - Thought: Analyze market + account data
    - Action: Adjust rates
    - Observation: Monitor results
    - Learn: Update strategy
    """
    
    def __init__(self, mcp_client, memory):
        self.mcp = mcp_client
        self.memory = memory
        self.name = "InterestOptimizationAgent"
    
    async def run_daily_optimization(self):
        """Daily optimization run"""
        
        print(f"[{self.name}] Starting daily rate optimization...")
        
        # Get market rates
        market_rates = await self._get_market_rates()
        
        # Get all accounts
        accounts = await self._get_all_accounts()
        
        adjustments_made = 0
        
        for account in accounts:
            # Decide if rate adjustment needed
            decision = await self._decide_rate_adjustment(
                account=account,
                market_rates=market_rates
            )
            
            if decision["should_adjust"]:
                # Adjust rate
                await self._adjust_rate(
                    account_id=account["account_id"],
                    new_rate=decision["recommended_rate"],
                    reason=decision["reason"]
                )
                adjustments_made += 1
                
                # Store in memory
                self.memory.append({
                    "action": "RATE_ADJUSTMENT",
                    "account_id": account["account_id"],
                    "new_rate": decision["recommended_rate"],
                    "reason": decision["reason"],
                    "timestamp": datetime.now()
                })
        
        print(f"[{self.name}] Optimization complete. Adjusted {adjustments_made} accounts.")
    
    async def _get_market_rates(self) -> Dict:
        """Get current market rates"""
        # Mock data - replace with actual market data API
        return {
            "rba_cash_rate": 0.0435,
            "competitor_average": 0.038,
            "competitor_high": 0.042,
            "competitor_low": 0.032
        }
    
    async def _get_all_accounts(self) -> list:
        """Get all active accounts"""
        # Mock - replace with actual DataMesh query
        return []
    
    async def _decide_rate_adjustment(
        self, 
        account: Dict, 
        market_rates: Dict
    ) -> Dict:
        """Decide if rate adjustment needed"""
        
        current_rate = account.get("optimized_interest_rate", 0.035)
        balance = account["balance"]
        competitor_avg = market_rates["competitor_average"]
        
        # Balance-based tiering
        if balance >= 100000:
            target_rate = competitor_avg + 0.005
        elif balance >= 50000:
            target_rate = competitor_avg
        else:
            target_rate = competitor_avg - 0.005
        
        # Check if adjustment needed
        rate_diff = abs(current_rate - target_rate)
        
        if rate_diff > 0.001:  # 0.1% threshold
            return {
                "should_adjust": True,
                "recommended_rate": target_rate,
                "reason": f"Market alignment: {competitor_avg:.2%}"
            }
        
        return {"should_adjust": False}
    
    async def _adjust_rate(
        self, 
        account_id: str, 
        new_rate: float, 
        reason: str
    ):
        """Apply rate adjustment"""
        await self.mcp.call_tool("update_account_rate", {
            "account_id": account_id,
            "new_rate": new_rate,
            "reason": reason
        })
