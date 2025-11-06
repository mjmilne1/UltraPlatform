"""
Tests for Enterprise Risk Management Framework
==============================================

Comprehensive test suite covering:
- Risk identification and assessment
- Market risk analysis (VaR, stress testing)
- Risk controls and effectiveness
- Incident response and management
- Risk reporting and analytics

Standards:
- ISO 31000 (Risk Management)
- COSO ERM Framework
- Basel III/IV
- APRA CPS 220

Performance Targets:
- Incident Response: <15 minutes detection
- Resolution Time: <4 hours
- Control Effectiveness: 95%+
- Risk Coverage: 100%
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta

from modules.risk_management.risk_engine import (
    RiskManagementFramework,
    RiskIdentificationEngine,
    MarketRiskAnalyzer,
    RiskControlManager,
    IncidentResponseManager,
    RiskReportingEngine,
    RiskCategory,
    RiskLevel,
    RiskStatus,
    ControlEffectiveness,
    IncidentSeverity,
    RiskIdentification,
    RiskControl,
    RiskIncident,
    MarketRiskMetrics
)


class TestRiskIdentificationEngine:
    """Tests for risk identification and assessment"""
    
    def test_engine_initialization(self):
        """Test risk engine initialization"""
        engine = RiskIdentificationEngine()
        
        assert len(engine.risks) == 0
        assert engine.risk_count == 0
        assert len(engine.risk_matrix) > 0
    
    def test_risk_identification(self):
        """Test risk identification"""
        engine = RiskIdentificationEngine()
        
        risk = engine.identify_risk(
            risk_name="Market Volatility",
            category=RiskCategory.MARKET,
            description="Portfolio value fluctuation",
            likelihood="high",
            impact="major",
            risk_owner="CIO"
        )
        
        assert risk.risk_id.startswith("RISK-")
        assert risk.category == RiskCategory.MARKET
        assert risk.risk_level == RiskLevel.HIGH
        assert engine.risk_count == 1
    
    def test_critical_risk_identification(self):
        """Test critical risk identification"""
        engine = RiskIdentificationEngine()
        
        risk = engine.identify_risk(
            risk_name="Cyber Security Breach",
            category=RiskCategory.TECHNOLOGY,
            description="Critical data breach",
            likelihood="very_high",
            impact="catastrophic",
            risk_owner="CTO"
        )
        
        assert risk.risk_level == RiskLevel.CRITICAL
        assert risk.inherent_risk_score == 25
    
    def test_risk_assessment_with_controls(self):
        """Test risk assessment with controls"""
        engine = RiskIdentificationEngine()
        
        risk = engine.identify_risk(
            risk_name="Operational Error",
            category=RiskCategory.OPERATIONAL,
            description="Process failure",
            likelihood="medium",
            impact="moderate",
            risk_owner="COO"
        )
        
        initial_score = risk.residual_risk_score
        
        # Apply controls (70% effective)
        assessed_risk = engine.assess_risk(risk.risk_id, 0.70)
        
        assert assessed_risk.residual_risk_score < initial_score
        assert assessed_risk.status == RiskStatus.ASSESSED
        assert assessed_risk.last_reviewed is not None
    
    def test_get_critical_risks(self):
        """Test filtering critical and high risks"""
        engine = RiskIdentificationEngine()
        
        # Create various risk levels
        engine.identify_risk("Critical Risk", RiskCategory.MARKET, "Test", 
                           "very_high", "catastrophic", "Owner1")
        engine.identify_risk("High Risk", RiskCategory.OPERATIONAL, "Test",
                           "high", "major", "Owner2")
        engine.identify_risk("Low Risk", RiskCategory.COMPLIANCE, "Test",
                           "low", "minor", "Owner3")
        
        critical_risks = engine.get_critical_risks()
        
        assert len(critical_risks) == 2
        assert all(r.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH] 
                  for r in critical_risks)
    
    def test_risks_by_category(self):
        """Test filtering risks by category"""
        engine = RiskIdentificationEngine()
        
        engine.identify_risk("Market Risk 1", RiskCategory.MARKET, "Test",
                           "medium", "moderate", "Owner")
        engine.identify_risk("Market Risk 2", RiskCategory.MARKET, "Test",
                           "medium", "moderate", "Owner")
        engine.identify_risk("Tech Risk", RiskCategory.TECHNOLOGY, "Test",
                           "medium", "moderate", "Owner")
        
        market_risks = engine.get_risks_by_category(RiskCategory.MARKET)
        
        assert len(market_risks) == 2
        assert all(r.category == RiskCategory.MARKET for r in market_risks)
    
    def test_risk_heatmap_generation(self):
        """Test risk heatmap generation"""
        engine = RiskIdentificationEngine()
        
        engine.identify_risk("Risk 1", RiskCategory.MARKET, "Test",
                           "high", "major", "Owner")
        engine.identify_risk("Risk 2", RiskCategory.MARKET, "Test",
                           "high", "major", "Owner")
        
        heatmap = engine.generate_risk_heatmap()
        
        assert len(heatmap) > 0
        assert "high_major" in heatmap
        assert len(heatmap["high_major"]) == 2


class TestMarketRiskAnalyzer:
    """Tests for market risk analysis"""
    
    def test_analyzer_initialization(self):
        """Test market risk analyzer initialization"""
        analyzer = MarketRiskAnalyzer()
        
        assert len(analyzer.risk_calculations) == 0
    
    def test_var_calculation(self):
        """Test Value at Risk calculation"""
        analyzer = MarketRiskAnalyzer()
        
        # Simulate portfolio returns
        returns = np.random.normal(0.0005, 0.02, 252).tolist()
        
        metrics = analyzer.calculate_var(
            portfolio_id="TEST-PORT",
            portfolio_value=10000000,
            returns=returns
        )
        
        assert metrics.portfolio_id == "TEST-PORT"
        assert metrics.var_1day_95 > 0
        assert metrics.var_1day_99 > metrics.var_1day_95
        assert metrics.var_10day_95 > metrics.var_1day_95
        assert metrics.portfolio_volatility > 0
    
    def test_var_percentile_logic(self):
        """Test VaR uses correct percentiles"""
        analyzer = MarketRiskAnalyzer()
        
        # Create known distribution
        returns = [-0.05, -0.03, -0.01, 0.01, 0.02, 0.03] * 42  # 252 days
        
        metrics = analyzer.calculate_var(
            portfolio_id="TEST",
            portfolio_value=1000000,
            returns=returns
        )
        
        # VaR should be positive (loss amount)
        assert metrics.var_1day_95 > 0
        assert metrics.var_1day_99 > 0
    
    def test_stress_testing(self):
        """Test portfolio stress testing"""
        analyzer = MarketRiskAnalyzer()
        
        # First calculate VaR
        returns = np.random.normal(0.0005, 0.02, 252).tolist()
        analyzer.calculate_var("PORT-001", 5000000, returns)
        
        # Stress test scenarios
        scenarios = {
            "market_crash": -0.20,
            "rate_spike": -0.05,
            "liquidity_crisis": -0.15
        }
        
        results = analyzer.stress_test_portfolio("PORT-001", scenarios)
        
        assert len(results) == 3
        assert results["market_crash"] < 0
        assert results["rate_spike"] < 0
    
    def test_concentration_risk_check(self):
        """Test concentration risk checking"""
        analyzer = MarketRiskAnalyzer()
        
        returns = np.random.normal(0.0005, 0.02, 252).tolist()
        metrics = analyzer.calculate_var("PORT-002", 10000000, returns)
        
        # Check with 10% threshold
        exceeds, concentration = analyzer.check_concentration_risk("PORT-002", 0.10)
        
        assert isinstance(exceeds, bool)
        assert concentration >= 0
        assert concentration <= 1.0
    
    def test_portfolio_metrics(self):
        """Test portfolio risk metrics calculation"""
        analyzer = MarketRiskAnalyzer()
        
        returns = np.random.normal(0.001, 0.015, 252).tolist()
        metrics = analyzer.calculate_var("PORT-003", 20000000, returns)
        
        assert metrics.portfolio_value == 20000000
        assert metrics.portfolio_volatility > 0
        assert metrics.sharpe_ratio != 0
        assert metrics.portfolio_beta > 0


class TestRiskControlManager:
    """Tests for risk control management"""
    
    def test_manager_initialization(self):
        """Test control manager initialization"""
        manager = RiskControlManager()
        
        assert len(manager.controls) == 0
        assert manager.control_count == 0
    
    def test_control_implementation(self):
        """Test implementing risk control"""
        manager = RiskControlManager()
        
        control = manager.implement_control(
            control_name="Daily Monitoring",
            risk_id="RISK-123",
            control_type="detective",
            description="Monitor daily",
            control_owner="Manager",
            frequency="daily"
        )
        
        assert control.control_id.startswith("CTRL-")
        assert control.risk_id == "RISK-123"
        assert control.status == "active"
        assert manager.control_count == 1
    
    def test_control_effectiveness_testing(self):
        """Test control effectiveness testing"""
        manager = RiskControlManager()
        
        control = manager.implement_control(
            "Test Control", "RISK-001", "preventive",
            "Test", "Owner", "continuous"
        )
        
        tested_control = manager.test_control_effectiveness(
            control.control_id,
            "Control working as expected",
            ControlEffectiveness.EFFECTIVE
        )
        
        assert tested_control.effectiveness == ControlEffectiveness.EFFECTIVE
        assert tested_control.test_date is not None
        assert tested_control.test_results != ""
    
    def test_get_controls_for_risk(self):
        """Test getting controls for specific risk"""
        manager = RiskControlManager()
        
        manager.implement_control("Control 1", "RISK-001", "preventive", "Test", "Owner")
        manager.implement_control("Control 2", "RISK-001", "detective", "Test", "Owner")
        manager.implement_control("Control 3", "RISK-002", "corrective", "Test", "Owner")
        
        controls = manager.get_controls_for_risk("RISK-001")
        
        assert len(controls) == 2
        assert all(c.risk_id == "RISK-001" for c in controls)
    
    def test_control_effectiveness_rate(self):
        """Test overall control effectiveness rate"""
        manager = RiskControlManager()
        
        # Create 10 controls
        for i in range(10):
            control = manager.implement_control(
                f"Control {i}", f"RISK-{i}", "preventive", "Test", "Owner"
            )
            
            # Mark 8 as effective
            if i < 8:
                manager.test_control_effectiveness(
                    control.control_id,
                    "Effective",
                    ControlEffectiveness.EFFECTIVE
                )
        
        rate = manager.calculate_control_effectiveness_rate()
        
        assert rate == 80.0
    
    def test_control_gap_identification(self):
        """Test identifying control gaps"""
        manager = RiskControlManager()
        engine = RiskIdentificationEngine()
        
        # Create high risk
        risk = engine.identify_risk(
            "High Risk", RiskCategory.MARKET, "Test",
            "high", "major", "Owner"
        )
        
        # Add only 1 control (need 2)
        control = manager.implement_control(
            "Single Control", risk.risk_id, "preventive", "Test", "Owner"
        )
        manager.test_control_effectiveness(
            control.control_id, "OK", ControlEffectiveness.EFFECTIVE
        )
        
        gaps = manager.identify_control_gaps(list(engine.risks.values()))
        
        assert len(gaps) > 0
        assert "High Risk" in gaps[0]


class TestIncidentResponseManager:
    """Tests for incident response management"""
    
    def test_manager_initialization(self):
        """Test incident manager initialization"""
        manager = IncidentResponseManager()
        
        assert len(manager.incidents) == 0
        assert manager.incident_count == 0
    
    @pytest.mark.asyncio
    async def test_incident_detection(self):
        """Test incident detection and logging"""
        manager = IncidentResponseManager()
        
        incident = await manager.detect_incident(
            incident_type="cyber_attack",
            severity=IncidentSeverity.MAJOR,
            title="Security Breach",
            description="Unauthorized access detected"
        )
        
        assert incident.incident_id.startswith("INC-")
        assert incident.severity == IncidentSeverity.MAJOR
        assert incident.status == "open"
        assert len(incident.response_actions) > 0
    
    @pytest.mark.asyncio
    async def test_automated_response_actions(self):
        """Test automated incident response"""
        manager = IncidentResponseManager()
        
        incident = await manager.detect_incident(
            "cyber_attack",
            IncidentSeverity.CATASTROPHIC,
            "Critical Breach",
            "Major security incident"
        )
        
        # Should have response actions
        assert len(incident.response_actions) >= 4
        assert any("Alert executive" in action for action in incident.response_actions)
    
    def test_incident_resolution(self):
        """Test incident resolution"""
        manager = IncidentResponseManager()
        
        # Create incident
        incident = RiskIncident(
            incident_id="INC-TEST",
            incident_type="operational",
            severity=IncidentSeverity.MODERATE,
            title="Test Incident",
            description="Test",
            root_cause="",
            financial_impact=0,
            operational_impact="",
            reputational_impact="",
            detected_at=datetime.now(),
            reported_at=datetime.now()
        )
        manager.incidents[incident.incident_id] = incident
        
        # Resolve incident
        resolved = manager.resolve_incident(
            incident.incident_id,
            "Human error",
            5000.0,
            "Improve training"
        )
        
        assert resolved.status == "resolved"
        assert resolved.resolved_at is not None
        assert resolved.financial_impact == 5000.0
    
    def test_get_open_incidents(self):
        """Test getting open incidents"""
        manager = IncidentResponseManager()
        
        # Create mix of open and resolved
        for i in range(3):
            incident = RiskIncident(
                incident_id=f"INC-{i}",
                incident_type="test",
                severity=IncidentSeverity.MINOR,
                title=f"Incident {i}",
                description="Test",
                root_cause="",
                financial_impact=0,
                operational_impact="",
                reputational_impact="",
                detected_at=datetime.now(),
                reported_at=datetime.now(),
                status="open" if i < 2 else "resolved"
            )
            manager.incidents[incident.incident_id] = incident
        
        open_incidents = manager.get_open_incidents()
        
        assert len(open_incidents) == 2
    
    def test_resolution_time_tracking(self):
        """Test average resolution time calculation"""
        manager = IncidentResponseManager()
        
        # Create and resolve incidents
        for i in range(3):
            incident = RiskIncident(
                incident_id=f"INC-{i}",
                incident_type="test",
                severity=IncidentSeverity.MINOR,
                title="Test",
                description="Test",
                root_cause="",
                financial_impact=0,
                operational_impact="",
                reputational_impact="",
                detected_at=datetime.now() - timedelta(hours=2),
                reported_at=datetime.now() - timedelta(hours=2)
            )
            manager.incidents[incident.incident_id] = incident
            manager.resolve_incident(incident.incident_id, "Root cause", 0, "Lessons")
        
        avg_time = manager.calculate_average_resolution_time()
        
        assert avg_time >= 0
        assert avg_time <= 3  # Should be around 2 hours


class TestRiskReportingEngine:
    """Tests for risk reporting"""
    
    def test_engine_initialization(self):
        """Test reporting engine initialization"""
        risk_engine = RiskIdentificationEngine()
        control_manager = RiskControlManager()
        incident_manager = IncidentResponseManager()
        
        reporting = RiskReportingEngine(
            risk_engine, control_manager, incident_manager
        )
        
        assert reporting.risk_engine == risk_engine
        assert reporting.control_manager == control_manager
        assert reporting.incident_manager == incident_manager
    
    def test_comprehensive_report_generation(self):
        """Test generating comprehensive risk report"""
        risk_engine = RiskIdentificationEngine()
        control_manager = RiskControlManager()
        incident_manager = IncidentResponseManager()
        reporting = RiskReportingEngine(risk_engine, control_manager, incident_manager)
        
        # Create some data
        risk = risk_engine.identify_risk(
            "Test Risk", RiskCategory.MARKET, "Test",
            "high", "major", "Owner"
        )
        control_manager.implement_control(
            "Test Control", risk.risk_id, "preventive", "Test", "Owner"
        )
        
        report = reporting.generate_comprehensive_report()
        
        assert report.report_id.startswith("RPT-")
        assert report.total_risks > 0
        assert report.total_controls > 0
        assert report.overall_risk_rating in list(RiskLevel)
    
    def test_executive_summary_generation(self):
        """Test executive summary content"""
        risk_engine = RiskIdentificationEngine()
        control_manager = RiskControlManager()
        incident_manager = IncidentResponseManager()
        reporting = RiskReportingEngine(risk_engine, control_manager, incident_manager)
        
        # Create critical risk
        risk_engine.identify_risk(
            "Critical Risk", RiskCategory.TECHNOLOGY, "Test",
            "very_high", "catastrophic", "Owner"
        )
        
        report = reporting.generate_comprehensive_report()
        
        assert len(report.executive_summary) > 0
        assert "CRITICAL" in report.executive_summary
    
    def test_dashboard_data_generation(self):
        """Test real-time dashboard data"""
        risk_engine = RiskIdentificationEngine()
        control_manager = RiskControlManager()
        incident_manager = IncidentResponseManager()
        reporting = RiskReportingEngine(risk_engine, control_manager, incident_manager)
        
        # Add data
        risk_engine.identify_risk("Risk 1", RiskCategory.MARKET, "Test",
                                 "high", "major", "Owner")
        
        dashboard = reporting.generate_risk_dashboard_data()
        
        assert "timestamp" in dashboard
        assert "risk_summary" in dashboard
        assert "control_status" in dashboard
        assert "incident_status" in dashboard
        assert "top_risks" in dashboard


class TestRiskManagementFramework:
    """Tests for integrated risk management framework"""
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        framework = RiskManagementFramework()
        
        assert framework.risk_engine is not None
        assert framework.market_analyzer is not None
        assert framework.control_manager is not None
        assert framework.incident_manager is not None
        assert framework.reporting_engine is not None
    
    @pytest.mark.asyncio
    async def test_comprehensive_risk_assessment(self):
        """Test complete risk assessment workflow"""
        framework = RiskManagementFramework()
        
        assessment = await framework.perform_risk_assessment(
            entity="Test Business",
            entity_type="financial_services"
        )
        
        assert assessment["assessment_id"].startswith("ASSESS-")
        assert assessment["risks_identified"] > 0
        assert assessment["controls_implemented"] > 0
        assert assessment["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_real_time_monitoring(self):
        """Test real-time risk monitoring"""
        framework = RiskManagementFramework()
        
        # Add some risks
        await framework.perform_risk_assessment("Test", "business")
        
        dashboard = await framework.monitor_real_time_risks()
        
        assert "timestamp" in dashboard
        assert "risk_summary" in dashboard
    
    def test_comprehensive_status(self):
        """Test comprehensive status reporting"""
        framework = RiskManagementFramework()
        
        status = framework.get_comprehensive_status()
        
        assert "timestamp" in status
        assert "risk_assessment" in status
        assert "control_environment" in status
        assert "incident_management" in status
        assert "market_risk" in status
        assert "compliance" in status


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_risk_workflow(self):
        """Test end-to-end risk management workflow"""
        framework = RiskManagementFramework()
        
        # Step 1: Perform assessment
        assessment = await framework.perform_risk_assessment("Business", "entity")
        assert assessment["status"] == "completed"
        
        # Step 2: Detect incident
        incident = await framework.incident_manager.detect_incident(
            "operational",
            IncidentSeverity.MODERATE,
            "Process Failure",
            "System error"
        )
        assert incident.incident_id.startswith("INC-")
        
        # Step 3: Calculate market risk
        returns = np.random.normal(0.0005, 0.02, 252).tolist()
        var_metrics = framework.market_analyzer.calculate_var(
            "PORT-001", 10000000, returns
        )
        assert var_metrics.var_1day_95 > 0
        
        # Step 4: Generate report
        status = framework.get_comprehensive_status()
        assert status["risk_assessment"]["total_risks"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_targets_met(self):
        """Test all performance targets are met"""
        framework = RiskManagementFramework()
        
        # Perform assessment
        await framework.perform_risk_assessment("Test", "entity")
        
        # Get status
        status = framework.get_comprehensive_status()
        
        # Verify targets
        assert status["risk_assessment"]["risk_coverage"] >= 100.0
        assert status["control_environment"]["effectiveness_rate"] >= 0
        assert status["incident_management"]["target_hours"] == 4.0
        assert status["compliance"]["compliance_score"] >= 95.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
