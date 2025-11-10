"""
AI Insights Generation Agent
Autonomous generation of investment insights
"""

from typing import Dict, List
from datetime import datetime

class InsightsGenerationAgent:
    """
    Autonomous agent for generating investment insights
    
    Generates:
    - Performance insights
    - Risk alerts
    - Opportunity identification
    - Portfolio recommendations
    - Market commentary
    """
    
    def __init__(self, datamesh_client, memory):
        self.datamesh = datamesh_client
        self.memory = memory
        self.name = "InsightsGenerationAgent"
    
    async def generate_insights_for_report(
        self,
        client_id: str,
        portfolio_data: Dict,
        performance_data: Dict
    ) -> List[Dict]:
        """Generate AI insights for client report"""
        
        print(f"[{self.name}] Generating insights for client {client_id}")
        
        insights = []
        
        # Performance insights
        perf_insights = await self._analyze_performance(performance_data)
        insights.extend(perf_insights)
        
        # Risk insights
        risk_insights = await self._analyze_risk(portfolio_data)
        insights.extend(risk_insights)
        
        # Opportunity insights
        opp_insights = await self._identify_opportunities(portfolio_data)
        insights.extend(opp_insights)
        
        # Market context
        market_insights = await self._analyze_market_context()
        insights.extend(market_insights)
        
        print(f"[{self.name}] Generated {len(insights)} insights")
        
        return insights
    
    async def _analyze_performance(self, performance: Dict) -> List[Dict]:
        """Analyze performance and generate insights"""
        
        insights = []
        
        total_return = performance.get("total_return", 0)
        benchmark_return = performance.get("benchmark_return", 0)
        outperformance = total_return - benchmark_return
        
        if outperformance > 0.02:  # > 2%
            insights.append({
                "type": "STRONG_PERFORMANCE",
                "title": "Strong Outperformance",
                "message": f"Your portfolio outperformed the benchmark by {outperformance*100:.1f}% this period.",
                "sentiment": "positive",
                "priority": "high"
            })
        elif outperformance > 0:
            insights.append({
                "type": "POSITIVE_PERFORMANCE",
                "title": "Positive Returns",
                "message": f"Your portfolio achieved a {total_return*100:.1f}% return, beating the benchmark.",
                "sentiment": "positive",
                "priority": "medium"
            })
        elif outperformance < -0.02:  # < -2%
            insights.append({
                "type": "UNDERPERFORMANCE",
                "title": "Underperformance Alert",
                "message": f"Portfolio underperformed benchmark by {abs(outperformance)*100:.1f}%. Consider reviewing strategy.",
                "sentiment": "negative",
                "priority": "high"
            })
        
        return insights
    
    async def _analyze_risk(self, portfolio: Dict) -> List[Dict]:
        """Analyze portfolio risk"""
        
        insights = []
        
        # Check concentration risk
        holdings = portfolio.get("holdings", [])
        if holdings:
            max_holding = max(h.get("weight", 0) for h in holdings)
            if max_holding > 0.15:  # > 15%
                insights.append({
                    "type": "CONCENTRATION_RISK",
                    "title": "Concentration Risk",
                    "message": f"Your largest holding represents {max_holding*100:.1f}% of the portfolio. Consider diversification.",
                    "sentiment": "warning",
                    "priority": "medium"
                })
        
        return insights
    
    async def _identify_opportunities(self, portfolio: Dict) -> List[Dict]:
        """Identify investment opportunities"""
        
        insights = []
        
        # Example opportunity
        insights.append({
            "type": "REBALANCING_OPPORTUNITY",
            "title": "Rebalancing Recommended",
            "message": "Current equity allocation is 5% above target. Consider rebalancing to manage risk.",
            "sentiment": "neutral",
            "priority": "medium",
            "action": "Review asset allocation"
        })
        
        return insights
    
    async def _analyze_market_context(self) -> List[Dict]:
        """Provide market context"""
        
        insights = []
        
        # Would integrate with market data
        insights.append({
            "type": "MARKET_COMMENTARY",
            "title": "Market Update",
            "message": "ASX 200 gained 2.3% this month driven by resource sector strength.",
            "sentiment": "neutral",
            "priority": "low"
        })
        
        return insights
    
    async def generate_recommendations(
        self,
        client_id: str,
        portfolio_data: Dict,
        risk_profile: str
    ) -> List[Dict]:
        """Generate personalized recommendations"""
        
        recommendations = []
        
        # Based on risk profile
        if risk_profile == "conservative":
            recommendations.append({
                "type": "DEFENSIVE_POSITIONING",
                "title": "Consider Defensive Assets",
                "message": "Given market volatility, consider increasing allocation to bonds and cash.",
                "action": "Increase fixed income by 5%",
                "rationale": "Matches conservative risk profile"
            })
        elif risk_profile == "growth":
            recommendations.append({
                "type": "GROWTH_OPPORTUNITY",
                "title": "Growth Opportunity",
                "message": "Market conditions favor growth stocks. Consider increasing equity exposure.",
                "action": "Increase equity allocation by 5%",
                "rationale": "Aligns with growth objective"
            })
        
        return recommendations
