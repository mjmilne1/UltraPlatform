"""
ANYA EXPLANATION GENERATION SYSTEM
===================================

Intelligent financial decision transparency engine that transforms
complex financial decisions into clear, personalized explanations.

Integrates with Ultra Platform to explain:
- Portfolio rebalancing
- Trade executions
- Goal progress
- Market impacts
- Proactive insights

Author: Ultra Platform Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, List, Any, Optional
from enum import Enum
import asyncio
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS
# ============================================================================

class SophisticationLevel(str, Enum):
    """User sophistication levels"""
    NOVICE = "novice"
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ExplanationType(str, Enum):
    """Types of explanations"""
    REBALANCING = "rebalancing"
    TRADE_EXECUTION = "trade_execution"
    GOAL_PROGRESS = "goal_progress"
    MARKET_IMPACT = "market_impact"
    TAX_OPPORTUNITY = "tax_opportunity"
    RISK_ALERT = "risk_alert"
    PORTFOLIO_DRIFT = "portfolio_drift"


class InsightSeverity(str, Enum):
    """Severity levels for proactive insights"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeliveryChannel(str, Enum):
    """Delivery channels for explanations"""
    IN_APP = "in_app"
    EMAIL = "email"
    SMS = "sms"
    PUSH = "push"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class UserProfile:
    """Complete user profile for personalization"""
    user_id: str
    sophistication_level: SophisticationLevel = SophisticationLevel.BEGINNER
    
    # Communication preferences
    preferred_channel: DeliveryChannel = DeliveryChannel.IN_APP
    detail_preference: str = "standard"  # summary, standard, detailed
    format_preference: str = "mixed"  # text, visual, mixed
    
    # Learning style
    learning_style: str = "balanced"  # visual, textual, interactive, example-based
    
    # Behavioral patterns
    engagement_level: str = "active"  # passive, active, highly-engaged
    decision_making: str = "validator"  # delegator, validator, controller
    risk_response: str = "moderate"  # conservative, moderate, aggressive
    
    # Profile metadata
    investment_experience_years: int = 0
    last_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class PortfolioEvent:
    """Portfolio event requiring explanation"""
    event_id: str
    event_type: ExplanationType
    timestamp: datetime
    customer_id: str
    
    # Event details
    trigger: str
    actions: List[Dict[str, Any]]
    impact: Dict[str, Any]
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedExplanation:
    """Generated explanation with metadata"""
    explanation_id: str
    event_id: str
    text: str
    sophistication_level: SophisticationLevel
    generation_time_ms: float
    
    # Metadata
    template_used: str
    personalization_applied: bool
    sources: List[str] = field(default_factory=list)
    
    # Delivery info
    delivery_channels: List[DeliveryChannel] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass
class ProactiveInsight:
    """Proactive insight notification"""
    insight_id: str
    insight_type: ExplanationType
    severity: InsightSeverity
    customer_id: str
    
    # Content
    title: str
    summary: str
    detailed_explanation: str
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadata
    triggered_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    delivered: bool = False


# ============================================================================
# EXPLANATION TEMPLATES
# ============================================================================

class ExplanationTemplates:
    """
    Multi-level explanation templates
    
    Templates for three sophistication levels:
    - Beginner: Simple, conversational
    - Intermediate: Standard financial terms
    - Advanced: Technical, quantitative
    """
    
    REBALANCING_TEMPLATES = {
        SophisticationLevel.BEGINNER: """
**Your Portfolio Was Adjusted**

We made some changes to keep your investments balanced:

**What Changed:**
{changes}

**Why We Did This:**
{reason}

**What This Means for You:**
{benefit}

Think of it like adjusting a recipe - we're keeping the right mix of ingredients to reach your goals!
""",
        
        SophisticationLevel.INTERMEDIATE: """
**Portfolio Rebalancing Summary**

Your portfolio was rebalanced to maintain your target allocation:

**Actions Taken:**
{changes}

**Trigger:**
{reason}

**Benefits:**
{benefit}

**Impact on Goals:**
{goal_impact}

This ensures your portfolio stays aligned with your risk tolerance and investment timeline.
""",
        
        SophisticationLevel.ADVANCED: """
**Portfolio Rebalancing Execution**

**Rebalancing Details:**
{changes}

**Triggering Factors:**
{reason}

**Risk-Return Analysis:**
{risk_analysis}

**Performance Impact:**
{performance_impact}

**Goal Alignment:**
{goal_impact}

Rebalancing executed to optimize risk-adjusted returns while maintaining strategic asset allocation.
"""
    }
    
    TRADE_TEMPLATES = {
        SophisticationLevel.BEGINNER: """
**We Made a Trade in Your Portfolio**

{action_summary}

**Why We Did This:**
{rationale}

**How This Helps:**
{benefit}

This keeps your portfolio working toward your goals!
""",
        
        SophisticationLevel.INTERMEDIATE: """
**Trade Execution Summary**

**Transaction Details:**
{action_summary}

**Investment Rationale:**
{rationale}

**Market Context:**
{market_context}

**Portfolio Impact:**
{benefit}

This trade aligns with your investment strategy and current market conditions.
""",
        
        SophisticationLevel.ADVANCED: """
**Trade Execution Report**

**Transaction:**
{action_summary}

**Investment Thesis:**
{rationale}

**Market Analysis:**
{market_context}

**Expected Alpha:**
{alpha_analysis}

**Risk Contribution:**
{risk_impact}

Trade executed within compliance parameters and strategic allocation guidelines.
"""
    }
    
    GOAL_PROGRESS_TEMPLATES = {
        SophisticationLevel.BEGINNER: """
**Your {goal_name} Progress Update**

📊 **Current Progress:** {progress_percentage}%

{status_message}

**What's Next:**
{next_steps}

You're {on_track} to reach your goal!
""",
        
        SophisticationLevel.INTERMEDIATE: """
**{goal_name} Progress Report**

**Current Status:**
- Progress: {progress_percentage}%
- Target: ${target_amount:,.2f}
- Current: ${current_amount:,.2f}

**Trajectory Analysis:**
{trajectory}

**Projected Completion:**
{completion_date}

**Recommended Actions:**
{next_steps}
""",
        
        SophisticationLevel.ADVANCED: """
**{goal_name} Quantitative Analysis**

**Performance Metrics:**
- Progress: {progress_percentage}%
- Actual vs Required Return: {return_analysis}
- Probability of Success: {success_probability}%

**Trajectory Analysis:**
{trajectory}

**Monte Carlo Projection:**
{monte_carlo_results}

**Optimization Recommendations:**
{next_steps}
"""
    }


# ============================================================================
# TEMPLATE ENGINE
# ============================================================================

class TemplateEngine:
    """
    Template rendering engine with variable substitution
    """
    
    def __init__(self):
        self.templates = ExplanationTemplates()
        logger.info("Template Engine initialized")
    
    def render(
        self,
        template_type: ExplanationType,
        sophistication: SophisticationLevel,
        variables: Dict[str, Any]
    ) -> str:
        """
        Render template with variables
        
        Args:
            template_type: Type of explanation
            sophistication: User sophistication level
            variables: Template variables to substitute
        
        Returns:
            Rendered explanation text
        """
        # Get appropriate template
        template = self._get_template(template_type, sophistication)
        
        # Render with variables
        try:
            rendered = template.format(**variables)
            return rendered
        
        except KeyError as e:
            logger.error(f"Missing template variable: {e}")
            return self._render_fallback(template_type, sophistication)
    
    def _get_template(
        self,
        template_type: ExplanationType,
        sophistication: SophisticationLevel
    ) -> str:
        """Get template for type and sophistication level"""
        template_map = {
            ExplanationType.REBALANCING: self.templates.REBALANCING_TEMPLATES,
            ExplanationType.TRADE_EXECUTION: self.templates.TRADE_TEMPLATES,
            ExplanationType.GOAL_PROGRESS: self.templates.GOAL_PROGRESS_TEMPLATES
        }
        
        templates = template_map.get(template_type, {})
        return templates.get(sophistication, templates.get(SophisticationLevel.INTERMEDIATE, ""))
    
    def _render_fallback(
        self,
        template_type: ExplanationType,
        sophistication: SophisticationLevel
    ) -> str:
        """Render fallback explanation"""
        return f"An {template_type.value} event occurred in your portfolio. Details are being processed."


# ============================================================================
# PERSONALIZATION ENGINE
# ============================================================================

class PersonalizationEngine:
    """
    Personalization engine that adapts explanations to user profiles
    """
    
    def __init__(self):
        # User profiles cache
        self.profiles: Dict[str, UserProfile] = {}
        logger.info("Personalization Engine initialized")
    
    async def get_profile(self, user_id: str) -> UserProfile:
        """Get or create user profile"""
        if user_id not in self.profiles:
            # In production: fetch from database
            self.profiles[user_id] = UserProfile(user_id=user_id)
        
        return self.profiles[user_id]
    
    async def personalize_explanation(
        self,
        explanation: str,
        profile: UserProfile,
        event: PortfolioEvent
    ) -> str:
        """
        Personalize explanation based on user profile
        
        Applies:
        - Style adjustments
        - Detail level
        - Educational context
        - Examples and analogies
        """
        personalized = explanation
        
        # Add user name if appropriate
        if profile.engagement_level in ["active", "highly-engaged"]:
            personalized = self._add_personal_greeting(personalized, profile)
        
        # Adjust detail level
        if profile.detail_preference == "summary":
            personalized = self._create_summary(personalized)
        elif profile.detail_preference == "detailed":
            personalized = self._add_details(personalized, event)
        
        # Add educational context for learners
        if profile.learning_style == "example-based":
            personalized = self._add_examples(personalized, event.event_type)
        
        return personalized
    
    def _add_personal_greeting(self, text: str, profile: UserProfile) -> str:
        """Add personal greeting to explanation"""
        # In production: fetch actual user name
        return f"Hi there! {text}"
    
    def _create_summary(self, text: str) -> str:
        """Create concise summary of explanation"""
        # Extract first paragraph or create TL;DR
        lines = text.split('\n')
        summary_lines = [line for line in lines if line.strip()][:3]
        return '\n'.join(summary_lines)
    
    def _add_details(self, text: str, event: PortfolioEvent) -> str:
        """Add additional details to explanation"""
        details = f"\n\n**Additional Details:**\n"
        details += f"- Event ID: {event.event_id}\n"
        details += f"- Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return text + details
    
    def _add_examples(self, text: str, event_type: ExplanationType) -> str:
        """Add educational examples"""
        examples = {
            ExplanationType.REBALANCING: "\n\n💡 **Example:** It's like rebalancing a seesaw - we adjust the weights to keep things level.",
            ExplanationType.TRADE_EXECUTION: "\n\n💡 **Example:** Think of buying stocks like shopping for groceries - we look for good value and quality.",
            ExplanationType.GOAL_PROGRESS: "\n\n💡 **Example:** Tracking your goal is like following a GPS - we show you how close you are and if you need to adjust your route."
        }
        
        return text + examples.get(event_type, "")


# ============================================================================
# PROACTIVE INSIGHTS ENGINE
# ============================================================================

class ProactiveInsightsEngine:
    """
    Proactive insights generation and delivery
    
    Monitors portfolio and generates timely notifications:
    - Portfolio drift alerts
    - Goal progress updates
    - Tax opportunities
    - Market impact notifications
    """
    
    def __init__(self):
        self.insights_queue: List[ProactiveInsight] = []
        logger.info("Proactive Insights Engine initialized")
    
    async def analyze_portfolio(
        self,
        customer_id: str,
        portfolio_data: Dict[str, Any]
    ) -> List[ProactiveInsight]:
        """
        Analyze portfolio and generate insights
        
        Returns list of actionable insights
        """
        insights = []
        
        # Check for portfolio drift
        drift_insight = await self._check_portfolio_drift(customer_id, portfolio_data)
        if drift_insight:
            insights.append(drift_insight)
        
        # Check goal progress
        goal_insights = await self._check_goal_progress(customer_id, portfolio_data)
        insights.extend(goal_insights)
        
        # Check for tax opportunities
        tax_insight = await self._check_tax_opportunities(customer_id, portfolio_data)
        if tax_insight:
            insights.append(tax_insight)
        
        return insights
    
    async def _check_portfolio_drift(
        self,
        customer_id: str,
        portfolio_data: Dict[str, Any]
    ) -> Optional[ProactiveInsight]:
        """Check if portfolio has drifted from target allocation"""
        target_allocation = portfolio_data.get("target_allocation", {})
        current_allocation = portfolio_data.get("current_allocation", {})
        
        # Calculate drift
        max_drift = 0
        drifted_asset = None
        
        for asset_class, target in target_allocation.items():
            current = current_allocation.get(asset_class, 0)
            drift = abs(current - target)
            
            if drift > max_drift:
                max_drift = drift
                drifted_asset = asset_class
        
        # Generate insight if drift > 5%
        if max_drift > 5:
            return ProactiveInsight(
                insight_id=f"drift_{customer_id}_{datetime.now(UTC).timestamp()}",
                insight_type=ExplanationType.PORTFOLIO_DRIFT,
                severity=InsightSeverity.MEDIUM,
                customer_id=customer_id,
                title="Portfolio Drift Detected",
                summary=f"Your {drifted_asset} allocation has drifted {max_drift:.1f}% from target.",
                detailed_explanation=f"""
Your portfolio allocation has shifted from your target:

**Current Drift:**
- {drifted_asset}: {max_drift:.1f}% away from target

**Why This Happened:**
Market movements have changed the relative values of your holdings.

**Recommended Action:**
Consider rebalancing to restore your target allocation and maintain your desired risk level.
""",
                recommended_actions=[
                    "Review current allocation",
                    "Consider rebalancing",
                    "Maintain target risk level"
                ]
            )
        
        return None
    
    async def _check_goal_progress(
        self,
        customer_id: str,
        portfolio_data: Dict[str, Any]
    ) -> List[ProactiveInsight]:
        """Check progress toward financial goals"""
        insights = []
        
        goals = portfolio_data.get("goals", [])
        
        for goal in goals:
            progress = goal.get("progress_percentage", 0)
            
            # Monthly update or significant change
            if progress % 10 < 2:  # Near 10% milestone
                insights.append(ProactiveInsight(
                    insight_id=f"goal_{goal['id']}_{datetime.now(UTC).timestamp()}",
                    insight_type=ExplanationType.GOAL_PROGRESS,
                    severity=InsightSeverity.LOW,
                    customer_id=customer_id,
                    title=f"{goal['name']} Progress Update",
                    summary=f"You're {progress:.0f}% of the way to your goal!",
                    detailed_explanation=f"""
**{goal['name']} Status:**
- Progress: {progress:.1f}%
- Current Amount: ${goal.get('current_amount', 0):,.2f}
- Target Amount: ${goal.get('target_amount', 0):,.2f}

You're making great progress toward your goal!
""",
                    recommended_actions=["Keep contributing regularly", "Review timeline"]
                ))
        
        return insights
    
    async def _check_tax_opportunities(
        self,
        customer_id: str,
        portfolio_data: Dict[str, Any]
    ) -> Optional[ProactiveInsight]:
        """Check for tax loss harvesting opportunities"""
        holdings = portfolio_data.get("holdings", [])
        
        harvestable_losses = sum(
            holding.get("unrealized_loss", 0)
            for holding in holdings
            if holding.get("unrealized_loss", 0) > 0
        )
        
        if harvestable_losses > 1000:
            return ProactiveInsight(
                insight_id=f"tax_{customer_id}_{datetime.now(UTC).timestamp()}",
                insight_type=ExplanationType.TAX_OPPORTUNITY,
                severity=InsightSeverity.MEDIUM,
                customer_id=customer_id,
                title="Tax Loss Harvesting Opportunity",
                summary=f"Potential tax savings: ${harvestable_losses * 0.25:,.2f}",
                detailed_explanation=f"""
**Tax Loss Harvesting Opportunity Detected**

**Potential Savings:**
- Harvestable Losses: ${harvestable_losses:,.2f}
- Estimated Tax Savings: ${harvestable_losses * 0.25:,.2f}

**What This Means:**
You can sell investments at a loss to offset gains and reduce your tax bill.

**Next Steps:**
Review the opportunity and consider harvesting before year-end.
""",
                recommended_actions=[
                    "Review losing positions",
                    "Consider tax impact",
                    "Avoid wash sales"
                ]
            )
        
        return None


# ============================================================================
# EXPLANATION GENERATOR
# ============================================================================

class ExplanationGenerator:
    """
    Complete explanation generation system
    
    Orchestrates:
    - Template rendering
    - Personalization
    - NLG enhancement
    - Quality validation
    """
    
    def __init__(self):
        self.template_engine = TemplateEngine()
        self.personalization_engine = PersonalizationEngine()
        self.insights_engine = ProactiveInsightsEngine()
        
        logger.info("✅ Explanation Generator initialized")
    
    async def generate_explanation(
        self,
        event: PortfolioEvent,
        user_id: str
    ) -> GeneratedExplanation:
        """
        Generate complete explanation for portfolio event
        
        Full pipeline:
        1. Get user profile
        2. Render appropriate template
        3. Personalize content
        4. Validate quality
        5. Return explanation
        """
        import time
        start_time = time.time()
        
        # Get user profile
        profile = await self.personalization_engine.get_profile(user_id)
        
        # Prepare template variables
        variables = self._prepare_variables(event)
        
        # Render template
        explanation_text = self.template_engine.render(
            template_type=event.event_type,
            sophistication=profile.sophistication_level,
            variables=variables
        )
        
        # Personalize
        personalized_text = await self.personalization_engine.personalize_explanation(
            explanation=explanation_text,
            profile=profile,
            event=event
        )
        
        # Calculate generation time
        generation_time_ms = (time.time() - start_time) * 1000
        
        return GeneratedExplanation(
            explanation_id=f"exp_{event.event_id}",
            event_id=event.event_id,
            text=personalized_text,
            sophistication_level=profile.sophistication_level,
            generation_time_ms=generation_time_ms,
            template_used=event.event_type.value,
            personalization_applied=True,
            delivery_channels=[profile.preferred_channel]
        )
    
    def _prepare_variables(self, event: PortfolioEvent) -> Dict[str, Any]:
        """Prepare template variables from event data"""
        variables = {}
        
        if event.event_type == ExplanationType.REBALANCING:
            variables = {
                "changes": self._format_changes(event.actions),
                "reason": event.trigger,
                "benefit": self._format_benefit(event.impact),
                "goal_impact": event.impact.get("goal_impact", "Goals remain on track"),
                "risk_analysis": event.context.get("risk_analysis", "Risk maintained at target level"),
                "performance_impact": event.context.get("performance", "Minimal performance impact expected")
            }
        
        elif event.event_type == ExplanationType.TRADE_EXECUTION:
            variables = {
                "action_summary": self._format_trade(event.actions[0]) if event.actions else "Trade executed",
                "rationale": event.trigger,
                "benefit": self._format_benefit(event.impact),
                "market_context": event.context.get("market", "Current market conditions favorable"),
                "alpha_analysis": event.context.get("alpha", "Expected to contribute positive alpha"),
                "risk_impact": event.context.get("risk", "Within risk parameters")
            }
        
        elif event.event_type == ExplanationType.GOAL_PROGRESS:
            variables = {
                "goal_name": event.context.get("goal_name", "Financial Goal"),
                "progress_percentage": event.context.get("progress", 0),
                "target_amount": event.context.get("target_amount", 0),
                "current_amount": event.context.get("current_amount", 0),
                "status_message": self._get_status_message(event.context.get("progress", 0)),
                "on_track": "on track" if event.context.get("on_track", True) else "slightly behind",
                "trajectory": event.context.get("trajectory", "On pace to meet goal"),
                "completion_date": event.context.get("completion_date", "As scheduled"),
                "next_steps": event.context.get("next_steps", "Continue current strategy"),
                "return_analysis": event.context.get("return_analysis", "Returns meeting expectations"),
                "success_probability": event.context.get("success_probability", 85),
                "monte_carlo_results": event.context.get("monte_carlo", "85% confidence interval")
            }
        
        return variables
    
    def _format_changes(self, actions: List[Dict[str, Any]]) -> str:
        """Format list of portfolio changes"""
        if not actions:
            return "Portfolio adjustments made"
        
        formatted = []
        for action in actions:
            asset = action.get("asset_class", "assets")
            change = action.get("change", 0)
            formatted.append(f"- {asset}: {change:+.1f}%")
        
        return "\n".join(formatted)
    
    def _format_trade(self, trade: Dict[str, Any]) -> str:
        """Format trade action"""
        action = trade.get("action", "Traded")
        quantity = trade.get("quantity", 0)
        symbol = trade.get("symbol", "securities")
        
        return f"{action} {quantity} shares of {symbol}"
    
    def _format_benefit(self, impact: Dict[str, Any]) -> str:
        """Format benefit description"""
        benefits = []
        
        if "risk_reduction" in impact:
            benefits.append(f"Reduced risk by {impact['risk_reduction']:.1f}%")
        
        if "return_improvement" in impact:
            benefits.append(f"Improved expected returns by {impact['return_improvement']:.1f}%")
        
        if not benefits:
            benefits.append("Maintains target allocation and risk level")
        
        return ", ".join(benefits)
    
    def _get_status_message(self, progress: float) -> str:
        """Get status message based on progress"""
        if progress >= 90:
            return "🎉 Almost there! You're close to reaching your goal."
        elif progress >= 75:
            return "📈 Great progress! You're well on your way."
        elif progress >= 50:
            return "✅ Halfway there! Keep up the good work."
        elif progress >= 25:
            return "🚀 Good start! You're making steady progress."
        else:
            return "🌱 Just getting started! Stay consistent."


# ============================================================================
# DEMO
# ============================================================================

async def demo_explanation_generation():
    """Demonstrate explanation generation system"""
    print("\n" + "=" * 70)
    print("ANYA EXPLANATION GENERATION SYSTEM DEMO")
    print("=" * 70)
    
    generator = ExplanationGenerator()
    
    # Test 1: Rebalancing explanation
    print("\n" + "=" * 70)
    print("TEST 1: PORTFOLIO REBALANCING")
    print("=" * 70)
    
    rebalance_event = PortfolioEvent(
        event_id="rebal_001",
        event_type=ExplanationType.REBALANCING,
        timestamp=datetime.now(UTC),
        customer_id="customer_001",
        trigger="Portfolio drift exceeded 5% threshold",
        actions=[
            {"asset_class": "Stocks", "change": -2.0},
            {"asset_class": "Bonds", "change": +2.0}
        ],
        impact={
            "risk_reduction": 1.5,
            "goal_impact": "Retirement goal remains on track"
        }
    )
    
    for level in [SophisticationLevel.BEGINNER, SophisticationLevel.INTERMEDIATE, SophisticationLevel.ADVANCED]:
        print(f"\n{'─' * 70}")
        print(f"{level.value.upper()} LEVEL:")
        print(f"{'─' * 70}")
        
        # Create user with this sophistication level
        generator.personalization_engine.profiles["test_user"] = UserProfile(
            user_id="test_user",
            sophistication_level=level
        )
        
        explanation = await generator.generate_explanation(rebalance_event, "test_user")
        print(explanation.text)
        print(f"\nGeneration Time: {explanation.generation_time_ms:.1f}ms")
    
    # Test 2: Proactive Insights
    print("\n" + "=" * 70)
    print("TEST 2: PROACTIVE INSIGHTS")
    print("=" * 70)
    
    portfolio_data = {
        "target_allocation": {"Stocks": 60, "Bonds": 30, "Real Estate": 10},
        "current_allocation": {"Stocks": 66, "Bonds": 28, "Real Estate": 6},
        "goals": [
            {
                "id": "retirement_001",
                "name": "Retirement",
                "progress_percentage": 48.5,
                "current_amount": 145000,
                "target_amount": 300000
            }
        ],
        "holdings": [
            {"symbol": "AAPL", "unrealized_loss": -500},
            {"symbol": "TSLA", "unrealized_loss": -800}
        ]
    }
    
    insights = await generator.insights_engine.analyze_portfolio("customer_001", portfolio_data)
    
    print(f"\nGenerated {len(insights)} insights:\n")
    
    for insight in insights:
        print(f"{'═' * 70}")
        print(f"🔔 {insight.title}")
        print(f"Severity: {insight.severity.value.upper()}")
        print(f"{'═' * 70}")
        print(insight.detailed_explanation)
        print(f"\n📋 Recommended Actions:")
        for action in insight.recommended_actions:
            print(f"   • {action}")
        print()
    
    print("\n" + "=" * 70)
    print("✅ Explanation Generation Demo Complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(demo_explanation_generation())
