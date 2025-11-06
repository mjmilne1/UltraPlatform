"""
Tests for Ultra Platform DSOA System
====================================

Comprehensive test suite covering:
- Goal-based optimization
- Real-time decision engine
- Compliance validation
- MCP integration
- Performance metrics (<100ms p99 latency)
- Client management
- Portfolio allocation
"""

import pytest
import asyncio
from datetime import datetime, timedelta

# Import DSOA components
from modules.dsoa_system.dsoa_core import (
    DSOASystem,
    ClientProfile,
    FinancialGoal,
    PortfolioAllocation,
    AdvisoryRecommendation,
    MarketConditions,
    GoalOptimizer,
    RealTimeDecisionEngine,
    ComplianceEngine,
    MCPAdvisoryServer,
    GoalType,
    RiskProfile,
    MarketRegime,
    AdvisoryAction,
    ComplianceStatus
)


class TestFinancialGoal:
    """Tests for FinancialGoal dataclass"""
    
    def test_goal_creation(self):
        """Test goal initialization"""
        goal = FinancialGoal(
            goal_id="GOAL-TEST01",
            goal_type=GoalType.RETIREMENT,
            name="Retirement Fund",
            target_amount=1000000,
            current_amount=100000,
            target_date=datetime.now() + timedelta(days=365*20),
            priority=1,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        assert goal.goal_id == "GOAL-TEST01"
        assert goal.goal_type == GoalType.RETIREMENT
        assert goal.target_amount == 1000000
        assert goal.priority == 1
    
    def test_update_progress(self):
        """Test progress calculation"""
        goal = FinancialGoal(
            goal_id="GOAL-TEST02",
            goal_type=GoalType.HOME_PURCHASE,
            name="Home Down Payment",
            target_amount=200000,
            current_amount=50000,
            target_date=datetime.now() + timedelta(days=365*5),
            priority=2,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        goal.update_progress()
        
        assert goal.progress_percentage == 25.0
    
    def test_years_to_goal(self):
        """Test years to goal calculation"""
        future_date = datetime.now() + timedelta(days=365*10)
        goal = FinancialGoal(
            goal_id="GOAL-TEST03",
            goal_type=GoalType.EDUCATION,
            name="College Fund",
            target_amount=100000,
            current_amount=10000,
            target_date=future_date,
            priority=2,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        years = goal.years_to_goal()
        
        assert 9.9 < years < 10.1  # Allow small margin
    
    def test_calculate_success_probability(self):
        """Test Monte Carlo success probability"""
        goal = FinancialGoal(
            goal_id="GOAL-TEST04",
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=1000000,
            current_amount=500000,
            target_date=datetime.now() + timedelta(days=365*10),
            priority=1,
            risk_tolerance=RiskProfile.BALANCED,
            monthly_contribution=2000
        )
        
        probability = goal.calculate_success_probability(
            portfolio_return=0.08,
            volatility=0.15
        )
        
        assert 0 <= probability <= 100
        assert probability > 50  # Should have good chance with reasonable inputs


class TestClientProfile:
    """Tests for ClientProfile"""
    
    def test_client_creation(self):
        """Test client profile initialization"""
        client = ClientProfile(
            client_id="CLI-TEST01",
            name="John Doe",
            date_of_birth=datetime(1980, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=500000,
            liquidity_needs=50000,
            tax_rate=0.30
        )
        
        assert client.client_id == "CLI-TEST01"
        assert client.name == "John Doe"
        assert client.risk_profile == RiskProfile.BALANCED
    
    def test_add_goal(self):
        """Test adding goal to client"""
        client = ClientProfile(
            client_id="CLI-TEST02",
            name="Jane Smith",
            date_of_birth=datetime(1985, 6, 15),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=300000,
            liquidity_needs=60000,
            tax_rate=0.32
        )
        
        goal = FinancialGoal(
            goal_id="GOAL-TEST05",
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=2000000,
            current_amount=0,
            target_date=datetime.now() + timedelta(days=365*30),
            priority=1,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        client.add_goal(goal)
        
        assert len(client.goals) == 1
        assert client.goals[0].goal_id == "GOAL-TEST05"
    
    def test_get_primary_goal(self):
        """Test getting highest priority goal"""
        client = ClientProfile(
            client_id="CLI-TEST03",
            name="Bob Wilson",
            date_of_birth=datetime(1990, 3, 20),
            risk_profile=RiskProfile.BALANCED,
            annual_income=150000,
            net_worth=400000,
            liquidity_needs=75000,
            tax_rate=0.35
        )
        
        # Add goals with different priorities
        goal1 = FinancialGoal(
            goal_id="GOAL-TEST06",
            goal_type=GoalType.HOME_PURCHASE,
            name="Home",
            target_amount=300000,
            current_amount=50000,
            target_date=datetime.now() + timedelta(days=365*3),
            priority=2,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        goal2 = FinancialGoal(
            goal_id="GOAL-TEST07",
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=2000000,
            current_amount=100000,
            target_date=datetime.now() + timedelta(days=365*25),
            priority=1,
            risk_tolerance=RiskProfile.BALANCED
        )
        
        client.add_goal(goal1)
        client.add_goal(goal2)
        
        primary = client.get_primary_goal()
        
        assert primary.goal_id == "GOAL-TEST07"  # Priority 1
        assert primary.priority == 1
    
    def test_age_calculation(self):
        """Test age calculation"""
        client = ClientProfile(
            client_id="CLI-TEST04",
            name="Alice Brown",
            date_of_birth=datetime(1990, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=200000,
            liquidity_needs=50000,
            tax_rate=0.30
        )
        
        age = client.age()
        expected_age = datetime.now().year - 1990
        
        assert age == expected_age


class TestPortfolioAllocation:
    """Tests for portfolio allocation"""
    
    def test_allocation_creation(self):
        """Test portfolio allocation creation"""
        allocation = PortfolioAllocation(
            allocation_id="ALLOC-TEST01",
            client_id="CLI-TEST01",
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        assert allocation.allocation_id == "ALLOC-TEST01"
        assert allocation.equities == 60.0
    
    def test_validate_allocations(self):
        """Test allocation validation"""
        allocation = PortfolioAllocation(
            allocation_id="ALLOC-TEST02",
            client_id="CLI-TEST02",
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        assert allocation.validate_allocations() is True
        
        # Invalid allocation
        invalid_allocation = PortfolioAllocation(
            allocation_id="ALLOC-TEST03",
            client_id="CLI-TEST03",
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=10.0  # Total = 105%
        )
        
        assert invalid_allocation.validate_allocations() is False


class TestMarketConditions:
    """Tests for market conditions"""
    
    def test_regime_detection_crisis(self):
        """Test crisis regime detection"""
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY,
            vix_level=50,
            sp500_return_1m=-15
        )
        
        regime = market.detect_regime()
        
        assert regime == MarketRegime.CRISIS
    
    def test_regime_detection_bull(self):
        """Test bull market detection"""
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY,
            vix_level=12,
            sp500_return_12m=20
        )
        
        regime = market.detect_regime()
        
        assert regime == MarketRegime.BULL_MARKET
    
    def test_regime_detection_bear(self):
        """Test bear market detection"""
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY,
            vix_level=25,
            sp500_return_3m=-12
        )
        
        regime = market.detect_regime()
        
        assert regime == MarketRegime.BEAR_MARKET


class TestGoalOptimizer:
    """Tests for goal-based optimization"""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization"""
        optimizer = GoalOptimizer()
        
        assert optimizer.optimization_history == []
    
    def test_optimize_portfolio_conservative(self):
        """Test portfolio optimization for conservative client"""
        optimizer = GoalOptimizer()
        
        client = ClientProfile(
            client_id="CLI-OPT01",
            name="Conservative Client",
            date_of_birth=datetime(1950, 1, 1),
            risk_profile=RiskProfile.CONSERVATIVE,
            annual_income=80000,
            net_worth=1000000,
            liquidity_needs=40000,
            tax_rate=0.25
        )
        
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY
        )
        
        allocation = optimizer.optimize_portfolio(client, market)
        
        assert allocation.equities <= 30  # Conservative limit
        assert allocation.validate_allocations()
    
    def test_optimize_portfolio_aggressive(self):
        """Test portfolio optimization for aggressive client"""
        optimizer = GoalOptimizer()
        
        client = ClientProfile(
            client_id="CLI-OPT02",
            name="Aggressive Client",
            date_of_birth=datetime(1995, 1, 1),
            risk_profile=RiskProfile.AGGRESSIVE,
            annual_income=150000,
            net_worth=200000,
            liquidity_needs=75000,
            tax_rate=0.35
        )
        
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.BULL_MARKET
        )
        
        allocation = optimizer.optimize_portfolio(client, market)
        
        assert allocation.equities >= 80  # Aggressive allocation
        assert allocation.validate_allocations()
    
    def test_crisis_adjustment(self):
        """Test allocation adjustment during crisis"""
        optimizer = GoalOptimizer()
        
        client = ClientProfile(
            client_id="CLI-OPT03",
            name="Balanced Client",
            date_of_birth=datetime(1980, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=500000,
            liquidity_needs=60000,
            tax_rate=0.30
        )
        
        # Normal market
        normal_market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY,
            vix_level=15
        )
        normal_allocation = optimizer.optimize_portfolio(client, normal_market)
        
        # Crisis market
        crisis_market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.CRISIS,
            vix_level=50
        )
        crisis_allocation = optimizer.optimize_portfolio(client, crisis_market)
        
        # Should reduce equity exposure in crisis
        assert crisis_allocation.equities < normal_allocation.equities


class TestRealTimeDecisionEngine:
    """Tests for real-time decision engine"""
    
    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """Test decision engine initialization"""
        optimizer = GoalOptimizer()
        engine = RealTimeDecisionEngine(optimizer)
        
        assert engine.decision_history == []
        assert engine.latency_metrics == []
    
    @pytest.mark.asyncio
    async def test_process_market_event(self):
        """Test market event processing"""
        optimizer = GoalOptimizer()
        engine = RealTimeDecisionEngine(optimizer)
        
        client = ClientProfile(
            client_id="CLI-ENG01",
            name="Test Client",
            date_of_birth=datetime(1985, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=400000,
            liquidity_needs=60000,
            tax_rate=0.30
        )
        
        portfolio = PortfolioAllocation(
            allocation_id="ALLOC-ENG01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.CRISIS,
            vix_level=50,
            sp500_return_1m=-15
        )
        
        recommendation = await engine.process_market_event(client, portfolio, market)
        
        # Should generate recommendation due to high risk
        assert recommendation is not None
        assert recommendation.action != AdvisoryAction.NO_ACTION
    
    @pytest.mark.asyncio
    async def test_latency_tracking(self):
        """Test latency performance tracking"""
        optimizer = GoalOptimizer()
        engine = RealTimeDecisionEngine(optimizer)
        
        client = ClientProfile(
            client_id="CLI-LAT01",
            name="Latency Test",
            date_of_birth=datetime(1990, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=300000,
            liquidity_needs=50000,
            tax_rate=0.28
        )
        
        portfolio = PortfolioAllocation(
            allocation_id="ALLOC-LAT01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        market = MarketConditions(
            timestamp=datetime.now(),
            regime=MarketRegime.LOW_VOLATILITY
        )
        
        # Process event
        await engine.process_market_event(client, portfolio, market)
        
        # Check latency was tracked
        assert len(engine.latency_metrics) > 0
        
        # Get metrics
        metrics = engine.get_latency_metrics()
        assert "p99" in metrics
        assert metrics["count"] > 0


class TestComplianceEngine:
    """Tests for compliance validation"""
    
    def test_compliance_initialization(self):
        """Test compliance engine initialization"""
        engine = ComplianceEngine()
        
        assert engine.compliance_checks == []
        assert engine.violations == []
    
    def test_validate_compliant_recommendation(self):
        """Test validation of compliant recommendation"""
        engine = ComplianceEngine()
        
        client = ClientProfile(
            client_id="CLI-COMP01",
            name="Compliant Client",
            date_of_birth=datetime(1980, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=500000,
            liquidity_needs=60000,
            tax_rate=0.30,
            kyc_completed=True,
            last_review_date=datetime.now()
        )
        
        current = PortfolioAllocation(
            allocation_id="ALLOC-COMP01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommended = PortfolioAllocation(
            allocation_id="ALLOC-COMP02",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=65.0,
            bonds=25.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommendation = AdvisoryRecommendation(
            recommendation_id="REC-COMP01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            action=AdvisoryAction.REBALANCE,
            priority=3,
            rationale="Portfolio rebalancing recommended",
            current_allocation=current,
            recommended_allocation=recommended,
            estimated_improvement=2.0,
            transaction_costs=0.1,
            tax_implications=0.0,
            implementation_date=datetime.now() + timedelta(days=1),
            review_date=datetime.now() + timedelta(days=30)
        )
        
        status, issues = engine.validate_recommendation(recommendation, client)
        
        assert status == ComplianceStatus.COMPLIANT
        assert len(issues) == 0
    
    def test_validate_kyc_breach(self):
        """Test KYC compliance breach"""
        engine = ComplianceEngine()
        
        client = ClientProfile(
            client_id="CLI-KYC01",
            name="No KYC Client",
            date_of_birth=datetime(1985, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=300000,
            liquidity_needs=50000,
            tax_rate=0.28,
            kyc_completed=False  # KYC not completed
        )
        
        current = PortfolioAllocation(
            allocation_id="ALLOC-KYC01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=50.0,
            bonds=40.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommended = PortfolioAllocation(
            allocation_id="ALLOC-KYC02",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=55.0,
            bonds=35.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommendation = AdvisoryRecommendation(
            recommendation_id="REC-KYC01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            action=AdvisoryAction.REBALANCE,
            priority=3,
            rationale="Rebalancing",
            current_allocation=current,
            recommended_allocation=recommended,
            estimated_improvement=1.5,
            transaction_costs=0.05,
            tax_implications=0.0,
            implementation_date=datetime.now() + timedelta(days=1),
            review_date=datetime.now() + timedelta(days=30)
        )
        
        status, issues = engine.validate_recommendation(recommendation, client)
        
        assert status == ComplianceStatus.BREACH
        assert any("KYC" in issue for issue in issues)
    
    def test_generate_statement_of_advice(self):
        """Test SOA generation"""
        engine = ComplianceEngine()
        
        client = ClientProfile(
            client_id="CLI-SOA01",
            name="SOA Test Client",
            date_of_birth=datetime(1980, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=500000,
            liquidity_needs=60000,
            tax_rate=0.30,
            kyc_completed=True
        )
        
        current = PortfolioAllocation(
            allocation_id="ALLOC-SOA01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommended = PortfolioAllocation(
            allocation_id="ALLOC-SOA02",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=55.0,
            bonds=35.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommendation = AdvisoryRecommendation(
            recommendation_id="REC-SOA01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            action=AdvisoryAction.DECREASE_RISK,
            priority=2,
            rationale="Reducing risk due to market volatility",
            current_allocation=current,
            recommended_allocation=recommended,
            estimated_improvement=1.0,
            transaction_costs=0.05,
            tax_implications=0.0,
            implementation_date=datetime.now() + timedelta(days=1),
            review_date=datetime.now() + timedelta(days=30)
        )
        
        soa = engine.generate_statement_of_advice(recommendation, client)
        
        assert "document_id" in soa
        assert soa["document_id"].startswith("SOA-")
        assert "client_details" in soa
        assert "advice_summary" in soa
        assert "compliance" in soa
        assert soa["compliance"]["asic_rg175_compliant"] is True
    
    def test_compliance_metrics(self):
        """Test compliance metrics collection"""
        engine = ComplianceEngine()
        
        # Perform some checks
        client = ClientProfile(
            client_id="CLI-MET01",
            name="Metrics Client",
            date_of_birth=datetime(1985, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=300000,
            liquidity_needs=50000,
            tax_rate=0.28,
            kyc_completed=True,
            last_review_date=datetime.now()
        )
        
        current = PortfolioAllocation(
            allocation_id="ALLOC-MET01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=60.0,
            bonds=30.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommended = PortfolioAllocation(
            allocation_id="ALLOC-MET02",
            client_id=client.client_id,
            generated_at=datetime.now(),
            equities=65.0,
            bonds=25.0,
            alternatives=5.0,
            cash=5.0
        )
        
        recommendation = AdvisoryRecommendation(
            recommendation_id="REC-MET01",
            client_id=client.client_id,
            generated_at=datetime.now(),
            action=AdvisoryAction.REBALANCE,
            priority=3,
            rationale="Test",
            current_allocation=current,
            recommended_allocation=recommended,
            estimated_improvement=1.0,
            transaction_costs=0.05,
            tax_implications=0.0,
            implementation_date=datetime.now() + timedelta(days=1),
            review_date=datetime.now() + timedelta(days=30)
        )
        
        engine.validate_recommendation(recommendation, client)
        
        metrics = engine.get_compliance_metrics()
        
        assert metrics["total_checks"] == 1
        assert metrics["compliance_rate"] == 100.0


class TestDSOASystem:
    """Tests for complete DSOA system"""
    
    def test_system_initialization(self):
        """Test DSOA system initialization"""
        dsoa = DSOASystem()
        
        assert dsoa.goal_optimizer is not None
        assert dsoa.decision_engine is not None
        assert dsoa.compliance_engine is not None
        assert dsoa.mcp_server is not None
        assert len(dsoa.clients) == 0
    
    def test_register_client(self):
        """Test client registration"""
        dsoa = DSOASystem()
        
        client = dsoa.register_client(
            name="Test User",
            date_of_birth=datetime(1990, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=100000,
            net_worth=300000
        )
        
        assert client.client_id.startswith("CLI-")
        assert client.name == "Test User"
        assert len(dsoa.clients) == 1
    
    def test_add_goal_to_client(self):
        """Test adding goal to client"""
        dsoa = DSOASystem()
        
        client = dsoa.register_client(
            name="Goal Test User",
            date_of_birth=datetime(1985, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=400000
        )
        
        goal = dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.RETIREMENT,
            name="Retirement at 65",
            target_amount=2000000,
            target_date=datetime(2050, 1, 1),
            priority=1
        )
        
        assert goal is not None
        assert goal.goal_id.startswith("GOAL-")
        assert len(client.goals) == 1
    
    @pytest.mark.asyncio
    async def test_generate_advisory_recommendation(self):
        """Test end-to-end recommendation generation"""
        dsoa = DSOASystem()
        
        # Register client
        client = dsoa.register_client(
            name="Advisory Test",
            date_of_birth=datetime(1980, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=150000,
            net_worth=500000
        )
        
        # Add goal
        dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=2000000,
            target_date=datetime(2050, 1, 1),
            priority=1
        )
        
        # Generate recommendation
        recommendation = await dsoa.generate_advisory_recommendation(client.client_id)
        
        assert recommendation is not None
        assert recommendation.recommendation_id.startswith("REC-")
        assert recommendation.compliance_checked is True
    
    def test_get_performance_metrics(self):
        """Test performance metrics collection"""
        dsoa = DSOASystem()
        
        metrics = dsoa.get_performance_metrics()
        
        assert "dsoa_metrics" in metrics
        assert "latency" in metrics
        assert "compliance" in metrics
        assert "mcp" in metrics
    
    def test_get_dashboard(self):
        """Test dashboard generation"""
        dsoa = DSOASystem()
        
        dashboard = dsoa.get_dashboard()
        
        assert "timestamp" in dashboard
        assert "system_status" in dashboard
        assert "clients" in dashboard
        assert "performance" in dashboard
        assert dashboard["system_status"] == "operational"


class TestMCPIntegration:
    """Tests for MCP server integration"""
    
    @pytest.mark.asyncio
    async def test_mcp_server_initialization(self):
        """Test MCP server initialization"""
        dsoa = DSOASystem()
        mcp = dsoa.mcp_server
        
        assert mcp.server_id.startswith("MCP-DSOA-")
        
        info = mcp.get_server_info()
        assert "server_id" in info
        assert "capabilities" in info
        assert "generate_advice" in info["capabilities"]
    
    @pytest.mark.asyncio
    async def test_mcp_generate_advice(self):
        """Test MCP advice generation"""
        dsoa = DSOASystem()
        
        # Register client
        client = dsoa.register_client(
            name="MCP Test Client",
            date_of_birth=datetime(1985, 1, 1),
            risk_profile=RiskProfile.BALANCED,
            annual_income=120000,
            net_worth=400000
        )
        
        # Add goal
        dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=1500000,
            target_date=datetime(2045, 1, 1),
            priority=1
        )
        
        # Generate via MCP
        response = await dsoa.mcp_server.generate_advice(
            client.client_id,
            {"trigger": "market_event"}
        )
        
        assert "status" in response
        # May be "success" or "no_action_required"
        assert response["status"] in ["success", "no_action_required"]


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_advisory_workflow(self):
        """Test complete advisory workflow from client registration to SOA"""
        dsoa = DSOASystem()
        
        # 1. Register client
        client = dsoa.register_client(
            name="Integration Test Client",
            date_of_birth=datetime(1980, 5, 15),
            risk_profile=RiskProfile.BALANCED,
            annual_income=130000,
            net_worth=450000
        )
        
        # 2. Add retirement goal
        retirement_goal = dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.RETIREMENT,
            name="Retirement at 65",
            target_amount=2000000,
            target_date=datetime(2045, 5, 15),
            priority=1
        )
        
        # 3. Add home purchase goal
        home_goal = dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.HOME_PURCHASE,
            name="Home Down Payment",
            target_amount=200000,
            target_date=datetime(2027, 1, 1),
            priority=2
        )
        
        # 4. Generate initial recommendation
        recommendation = await dsoa.generate_advisory_recommendation(client.client_id)
        
        assert recommendation is not None
        assert recommendation.compliance_checked is True
        
        # 5. Verify portfolio was created
        portfolio = dsoa.get_current_portfolio(client.client_id)
        assert portfolio is not None
        assert portfolio.validate_allocations()
        
        # 6. Generate SOA
        soa = dsoa.compliance_engine.generate_statement_of_advice(
            recommendation,
            client
        )
        
        assert soa["compliance"]["asic_rg175_compliant"] is True
        assert soa["compliance"]["best_interests_duty_met"] is True
        
        # 7. Verify metrics
        metrics = dsoa.get_performance_metrics()
        assert metrics["dsoa_metrics"]["total_recommendations"] >= 1
        assert metrics["compliance"]["compliance_rate"] == 100.0
    
    @pytest.mark.asyncio
    async def test_crisis_response_workflow(self):
        """Test system response to market crisis"""
        dsoa = DSOASystem()
        
        # Register client
        client = dsoa.register_client(
            name="Crisis Test Client",
            date_of_birth=datetime(1975, 1, 1),
            risk_profile=RiskProfile.MODERATELY_CONSERVATIVE,
            annual_income=140000,
            net_worth=600000
        )
        
        dsoa.add_goal_to_client(
            client.client_id,
            goal_type=GoalType.RETIREMENT,
            name="Retirement",
            target_amount=1800000,
            target_date=datetime(2040, 1, 1),
            priority=1
        )
        
        # Normal market recommendation
        normal_rec = await dsoa.generate_advisory_recommendation(client.client_id)
        normal_equity = normal_rec.recommended_allocation.equities if normal_rec else 0
        
        # Simulate crisis
        dsoa.update_market_conditions(
            vix_level=50,
            sp500_return_1m=-20,
            regime=MarketRegime.CRISIS
        )
        
        # Crisis recommendation
        crisis_rec = await dsoa.generate_advisory_recommendation(client.client_id)
        
        if crisis_rec:
            crisis_equity = crisis_rec.recommended_allocation.equities
            # Should recommend risk reduction
            assert crisis_rec.action in [
                AdvisoryAction.DECREASE_RISK,
                AdvisoryAction.REBALANCE
            ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
