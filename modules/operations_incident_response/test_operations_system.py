"""
Tests for Ultra Platform Operations & Incident Response System
==============================================================

Comprehensive test suite covering:
- Incident lifecycle management
- SLA compliance tracking
- Runbook automation
- Deployment orchestration
- Post-incident reviews
- On-call scheduling
"""

import pytest
import asyncio
from datetime import datetime, timedelta

# Import classes to test
from modules.operations_incident_response.operations_system import (
    IncidentManager,
    RunbookExecutor,
    DeploymentOrchestrator,
    PostIncidentReviewManager,
    OnCallManager,
    OperationsCenter,
    Severity,
    IncidentStatus,
    DeploymentStrategy,
    SLARequirements,
    Incident,
    Runbook,
    RunbookStep,
    RunbookStatus
)


class TestSLARequirements:
    """Tests for SLA requirements"""
    
    def test_p1_sla_requirements(self):
        """Test P1 critical SLA requirements"""
        sla = SLARequirements.for_severity(Severity.P1_CRITICAL)
        
        assert sla.response_time_minutes == 5
        assert sla.resolution_time_minutes == 30
        assert sla.escalation_time_minutes == 15
    
    def test_p2_sla_requirements(self):
        """Test P2 high SLA requirements"""
        sla = SLARequirements.for_severity(Severity.P2_HIGH)
        
        assert sla.response_time_minutes == 15
        assert sla.resolution_time_minutes == 120
        assert sla.escalation_time_minutes == 60
    
    def test_all_severity_levels_have_sla(self):
        """Test all severity levels have SLA definitions"""
        for severity in Severity:
            sla = SLARequirements.for_severity(severity)
            assert sla is not None
            assert sla.response_time_minutes > 0
            assert sla.resolution_time_minutes > 0


class TestIncident:
    """Tests for Incident dataclass"""
    
    def test_incident_creation(self):
        """Test incident initialization"""
        incident = Incident(
            incident_id="INC-TEST01",
            title="Test Incident",
            description="Test description",
            severity=Severity.P2_HIGH,
            status=IncidentStatus.DETECTED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        assert incident.incident_id == "INC-TEST01"
        assert incident.severity == Severity.P2_HIGH
        assert incident.status == IncidentStatus.DETECTED
        assert not incident.sla_breached
    
    def test_add_timeline_event(self):
        """Test adding timeline events"""
        incident = Incident(
            incident_id="INC-TEST02",
            title="Test",
            description="Test",
            severity=Severity.P1_CRITICAL,
            status=IncidentStatus.DETECTED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        incident.add_timeline_event("Test event", {"key": "value"})
        
        assert len(incident.timeline) == 1
        assert incident.timeline[0]["event"] == "Test event"
        assert incident.timeline[0]["details"]["key"] == "value"
    
    def test_add_action(self):
        """Test recording actions"""
        incident = Incident(
            incident_id="INC-TEST03",
            title="Test",
            description="Test",
            severity=Severity.P1_CRITICAL,
            status=IncidentStatus.DETECTED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        incident.add_action("Restarted service")
        
        assert len(incident.actions_taken) == 1
        assert "Restarted service" in incident.actions_taken[0]
    
    def test_get_duration(self):
        """Test duration calculation"""
        now = datetime.now()
        incident = Incident(
            incident_id="INC-TEST04",
            title="Test",
            description="Test",
            severity=Severity.P1_CRITICAL,
            status=IncidentStatus.RESOLVED,
            created_at=now,
            detected_at=now,
            resolved_at=now + timedelta(minutes=30)
        )
        
        duration = incident.get_duration()
        
        assert duration is not None
        assert duration.total_seconds() == 1800  # 30 minutes


class TestIncidentManager:
    """Tests for incident management"""
    
    def test_create_incident(self):
        """Test incident creation"""
        manager = IncidentManager()
        
        incident = manager.create_incident(
            title="API Error Spike",
            description="High error rate detected",
            severity=Severity.P1_CRITICAL,
            affected_services=["api-service"],
            users_affected=100
        )
        
        assert incident.incident_id.startswith("INC-")
        assert incident.title == "API Error Spike"
        assert incident.severity == Severity.P1_CRITICAL
        assert incident.status == IncidentStatus.DETECTED
        assert len(manager.active_incidents) == 1
        assert manager.metrics["total_incidents"] == 1
        assert manager.metrics["p1_incidents"] == 1
    
    def test_acknowledge_incident(self):
        """Test incident acknowledgment"""
        manager = IncidentManager()
        incident = manager.create_incident(
            "Test Incident",
            "Description",
            Severity.P2_HIGH,
            ["service1"]
        )
        
        success = manager.acknowledge_incident(incident.incident_id, "alice@ultra.com")
        
        assert success is True
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_at is not None
        assert incident.assigned_to == "alice@ultra.com"
    
    def test_update_status(self):
        """Test incident status updates"""
        manager = IncidentManager()
        incident = manager.create_incident(
            "Test",
            "Test",
            Severity.P2_HIGH,
            ["service1"]
        )
        
        success = manager.update_status(
            incident.incident_id,
            IncidentStatus.INVESTIGATING,
            "Running diagnostics"
        )
        
        assert success is True
        assert incident.status == IncidentStatus.INVESTIGATING
    
    def test_resolve_incident(self):
        """Test incident resolution"""
        manager = IncidentManager()
        incident = manager.create_incident(
            "Test",
            "Test",
            Severity.P2_HIGH,
            ["service1"]
        )
        
        manager.update_status(incident.incident_id, IncidentStatus.RESOLVED)
        
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.incident_id not in manager.active_incidents
    
    def test_escalate_incident(self):
        """Test incident escalation"""
        manager = IncidentManager()
        incident = manager.create_incident(
            "Test",
            "Test",
            Severity.P1_CRITICAL,
            ["service1"]
        )
        
        success = manager.escalate_incident(
            incident.incident_id,
            "engineering-lead@ultra.com",
            "Requires senior expertise"
        )
        
        assert success is True
        assert incident.escalated_to == "engineering-lead@ultra.com"
    
    def test_list_active_incidents(self):
        """Test listing active incidents"""
        manager = IncidentManager()
        
        inc1 = manager.create_incident("Inc1", "Desc", Severity.P1_CRITICAL, ["s1"])
        inc2 = manager.create_incident("Inc2", "Desc", Severity.P2_HIGH, ["s2"])
        manager.create_incident("Inc3", "Desc", Severity.P3_MEDIUM, ["s3"])
        
        # Resolve one
        manager.update_status(inc1.incident_id, IncidentStatus.RESOLVED)
        
        active = manager.list_active_incidents()
        assert len(active) == 2
        
        # Filter by severity
        p2_incidents = manager.list_active_incidents(severity=Severity.P2_HIGH)
        assert len(p2_incidents) == 1
        assert p2_incidents[0].incident_id == inc2.incident_id
    
    def test_get_metrics(self):
        """Test metrics collection"""
        manager = IncidentManager()
        
        manager.create_incident("Test1", "Desc", Severity.P1_CRITICAL, ["s1"])
        manager.create_incident("Test2", "Desc", Severity.P2_HIGH, ["s2"])
        
        metrics = manager.get_metrics()
        
        assert metrics["total_incidents"] == 2
        assert metrics["p1_incidents"] == 1
        assert metrics["active_incidents"] == 2
        assert "sla_compliance_rate" in metrics


class TestRunbookExecutor:
    """Tests for runbook execution"""
    
    def test_default_runbooks_registered(self):
        """Test default runbooks are registered"""
        executor = RunbookExecutor()
        
        runbooks = executor.list_runbooks()
        
        assert len(runbooks) >= 3  # At least 3 default runbooks
        assert "RB001" in executor.runbooks
        assert "RB002" in executor.runbooks
        assert "RB003" in executor.runbooks
    
    def test_register_custom_runbook(self):
        """Test registering custom runbook"""
        executor = RunbookExecutor()
        
        runbook = Runbook(
            runbook_id="RB999",
            name="Custom Runbook",
            description="Test runbook",
            category="test",
            severity_triggers=[Severity.P3_MEDIUM],
            steps=[
                RunbookStep(
                    step_id="RB999-S1",
                    title="Test Step",
                    description="Test",
                    command="echo test"
                )
            ]
        )
        
        executor.register_runbook(runbook)
        
        assert "RB999" in executor.runbooks
        assert executor.get_runbook("RB999").name == "Custom Runbook"
    
    @pytest.mark.asyncio
    async def test_execute_runbook_dry_run(self):
        """Test runbook execution in dry-run mode"""
        executor = RunbookExecutor()
        
        result = await executor.execute_runbook("RB001", dry_run=True)
        
        assert result["execution_id"].startswith("EXEC-")
        assert result["runbook_id"] == "RB001"
        assert result["dry_run"] is True
        assert len(result["steps"]) > 0
        assert all(s["status"] == RunbookStatus.SUCCESS.value for s in result["steps"])
    
    @pytest.mark.asyncio
    async def test_execute_runbook_with_incident(self):
        """Test runbook execution linked to incident"""
        executor = RunbookExecutor()
        
        result = await executor.execute_runbook(
            "RB001",
            incident_id="INC-TEST123",
            dry_run=True
        )
        
        assert result["incident_id"] == "INC-TEST123"
        assert result["success"] is True
    
    def test_list_runbooks_by_category(self):
        """Test filtering runbooks by category"""
        executor = RunbookExecutor()
        
        api_runbooks = executor.list_runbooks(category="api")
        
        assert len(api_runbooks) > 0
        assert all(rb.category == "api" for rb in api_runbooks)


class TestDeploymentOrchestrator:
    """Tests for deployment orchestration"""
    
    @pytest.mark.asyncio
    async def test_standard_deployment(self):
        """Test standard canary deployment"""
        orchestrator = DeploymentOrchestrator()
        
        deployment = await orchestrator.deploy(
            version="v1.2.3",
            environment="production",
            strategy=DeploymentStrategy.STANDARD,
            approved_by="tech-lead@ultra.com"
        )
        
        assert deployment.deployment_id.startswith("DEP-")
        assert deployment.version == "v1.2.3"
        assert deployment.environment == "production"
        assert deployment.status == "completed"
        assert deployment.canary_percentage == 100
    
    @pytest.mark.asyncio
    async def test_emergency_deployment(self):
        """Test emergency deployment"""
        orchestrator = DeploymentOrchestrator()
        
        deployment = await orchestrator.deploy(
            version="v1.2.4-hotfix",
            environment="production",
            strategy=DeploymentStrategy.EMERGENCY,
            emergency=True,
            approved_by="cto@ultra.com"
        )
        
        assert deployment.emergency is True
        assert deployment.status == "completed"
        assert deployment.canary_percentage == 100
    
    @pytest.mark.asyncio
    async def test_deployment_logging(self):
        """Test deployment audit logging"""
        orchestrator = DeploymentOrchestrator()
        
        deployment = await orchestrator.deploy(
            version="v1.2.5",
            environment="staging",
            strategy=DeploymentStrategy.STANDARD
        )
        
        assert len(deployment.deployment_log) > 0
        assert any("initiated" in log.lower() for log in deployment.deployment_log)
    
    def test_list_deployments(self):
        """Test listing deployments"""
        orchestrator = DeploymentOrchestrator()
        
        # Create deployment without async execution
        dep = orchestrator.deployments["TEST-DEP"] = orchestrator.deployments.get("TEST-DEP")
        
        deployments = orchestrator.list_deployments()
        
        # Should have at least the test deployment
        assert isinstance(deployments, list)


class TestPostIncidentReviewManager:
    """Tests for PIR management"""
    
    def test_create_pir(self):
        """Test PIR creation"""
        manager = PostIncidentReviewManager()
        
        incident = Incident(
            incident_id="INC-TEST99",
            title="Test Incident",
            description="Test",
            severity=Severity.P1_CRITICAL,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now(),
            resolved_at=datetime.now() + timedelta(minutes=30),
            users_affected=50,
            revenue_impact=1000.0
        )
        
        pir = manager.create_pir(
            incident=incident,
            incident_commander="alice@ultra.com",
            schedule_days_from_now=2
        )
        
        assert pir.pir_id.startswith("PIR-")
        assert pir.incident_id == "INC-TEST99"
        assert pir.severity == Severity.P1_CRITICAL
        assert pir.users_affected == 50
        assert pir.revenue_impact == 1000.0
    
    def test_conduct_pir(self):
        """Test conducting PIR"""
        manager = PostIncidentReviewManager()
        
        incident = Incident(
            incident_id="INC-TEST100",
            title="Test",
            description="Test",
            severity=Severity.P2_HIGH,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        pir = manager.create_pir(incident, "bob@ultra.com")
        
        manager.conduct_pir(
            pir.pir_id,
            attendees=["alice@ultra.com", "bob@ultra.com", "charlie@ultra.com"],
            summary="Database connection pool exhausted",
            root_cause="Insufficient pool size for load",
            resolution="Increased pool size from 50 to 100"
        )
        
        assert pir.conducted_date is not None
        assert len(pir.attendees) == 3
        assert pir.summary != ""
        assert pir.root_cause != ""
    
    def test_add_retrospective_items(self):
        """Test adding retrospective analysis"""
        manager = PostIncidentReviewManager()
        
        incident = Incident(
            incident_id="INC-TEST101",
            title="Test",
            description="Test",
            severity=Severity.P2_HIGH,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        pir = manager.create_pir(incident, "alice@ultra.com")
        
        manager.add_retrospective_items(
            pir.pir_id,
            what_went_well=["Fast detection", "Clear runbooks"],
            what_went_wrong=["Slow escalation", "Missing monitoring"],
            lessons_learned=["Need better alerts", "Update runbooks"]
        )
        
        assert len(pir.what_went_well) == 2
        assert len(pir.what_went_wrong) == 2
        assert len(pir.lessons_learned) == 2
    
    def test_add_action_items(self):
        """Test adding PIR action items"""
        manager = PostIncidentReviewManager()
        
        incident = Incident(
            incident_id="INC-TEST102",
            title="Test",
            description="Test",
            severity=Severity.P2_HIGH,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        pir = manager.create_pir(incident, "alice@ultra.com")
        
        pir.add_action_item(
            action="Implement database connection monitoring",
            owner="bob@ultra.com",
            due_date=datetime.now() + timedelta(days=7),
            priority="high"
        )
        
        assert len(pir.action_items) == 1
        assert pir.action_items[0]["action"] == "Implement database connection monitoring"
        assert pir.action_items[0]["priority"] == "high"
    
    def test_list_pending_pirs(self):
        """Test listing pending PIRs"""
        manager = PostIncidentReviewManager()
        
        incident1 = Incident(
            incident_id="INC-TEST103",
            title="Test1",
            description="Test",
            severity=Severity.P2_HIGH,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        incident2 = Incident(
            incident_id="INC-TEST104",
            title="Test2",
            description="Test",
            severity=Severity.P3_MEDIUM,
            status=IncidentStatus.RESOLVED,
            created_at=datetime.now(),
            detected_at=datetime.now()
        )
        
        pir1 = manager.create_pir(incident1, "alice@ultra.com")
        pir2 = manager.create_pir(incident2, "bob@ultra.com")
        
        # Conduct one
        manager.conduct_pir(pir1.pir_id, [], "Summary", "Root cause", "Resolution")
        
        pending = manager.list_pirs(pending_only=True)
        
        assert len(pending) == 1
        assert pending[0].pir_id == pir2.pir_id


class TestOnCallManager:
    """Tests for on-call management"""
    
    def test_create_schedule(self):
        """Test creating on-call schedule"""
        manager = OnCallManager()
        
        schedule = manager.create_schedule(
            name="Platform Team",
            team_members=["alice@ultra.com", "bob@ultra.com", "charlie@ultra.com"],
            rotation_type="weekly"
        )
        
        assert schedule.schedule_id.startswith("SCH-")
        assert schedule.name == "Platform Team"
        assert len(schedule.team_members) == 3
        assert schedule.current_primary == "alice@ultra.com"
        assert schedule.current_secondary == "bob@ultra.com"
    
    def test_get_current_oncall(self):
        """Test getting current on-call engineers"""
        manager = OnCallManager()
        
        schedule = manager.create_schedule(
            name="Test Team",
            team_members=["engineer1@ultra.com", "engineer2@ultra.com"]
        )
        
        primary, secondary = manager.get_current_oncall(schedule.schedule_id)
        
        assert primary == "engineer1@ultra.com"
        assert secondary == "engineer2@ultra.com"
    
    def test_rotate_schedule(self):
        """Test rotating on-call schedule"""
        manager = OnCallManager()
        
        schedule = manager.create_schedule(
            name="Test Team",
            team_members=["a@ultra.com", "b@ultra.com", "c@ultra.com"]
        )
        
        original_primary = schedule.current_primary
        
        success = manager.rotate_schedule(schedule.schedule_id)
        
        assert success is True
        assert schedule.current_primary != original_primary
        assert schedule.current_primary == "b@ultra.com"


class TestOperationsCenter:
    """Tests for central operations system"""
    
    def test_initialization(self):
        """Test operations center initialization"""
        ops = OperationsCenter()
        
        assert ops.incident_manager is not None
        assert ops.runbook_executor is not None
        assert ops.deployment_orchestrator is not None
        assert ops.pir_manager is not None
        assert ops.oncall_manager is not None
    
    def test_get_dashboard(self):
        """Test dashboard generation"""
        ops = OperationsCenter()
        
        # Create some data
        ops.incident_manager.create_incident(
            "Test Incident",
            "Description",
            Severity.P2_HIGH,
            ["service1"]
        )
        
        dashboard = ops.get_dashboard()
        
        assert "timestamp" in dashboard
        assert "incidents" in dashboard
        assert "deployments" in dashboard
        assert "pirs" in dashboard
        assert "runbooks" in dashboard
        assert dashboard["incidents"]["active"] == 1
    
    @pytest.mark.asyncio
    async def test_integrated_workflow(self):
        """Test complete operational workflow"""
        ops = OperationsCenter()
        
        # Create on-call schedule
        schedule = ops.oncall_manager.create_schedule(
            "Test Team",
            ["alice@ultra.com", "bob@ultra.com"]
        )
        primary, _ = ops.oncall_manager.get_current_oncall(schedule.schedule_id)
        
        # Create incident
        incident = ops.incident_manager.create_incident(
            "Integration Test Incident",
            "Testing full workflow",
            Severity.P2_HIGH,
            ["test-service"]
        )
        
        # Acknowledge
        ops.incident_manager.acknowledge_incident(incident.incident_id, primary)
        
        # Execute runbook
        result = await ops.runbook_executor.execute_runbook(
            "RB001",
            incident_id=incident.incident_id,
            dry_run=True
        )
        
        # Resolve
        ops.incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.RESOLVED
        )
        
        # Create PIR
        pir = ops.pir_manager.create_pir(incident, primary)
        
        # Verify complete workflow
        assert incident.status == IncidentStatus.RESOLVED
        assert result["success"] is True
        assert pir.incident_id == incident.incident_id
        
        # Check dashboard
        dashboard = ops.get_dashboard()
        assert dashboard["incidents"]["metrics"]["total_incidents"] == 1


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_p1_incident_workflow(self):
        """Test complete P1 incident workflow"""
        ops = OperationsCenter()
        
        # Incident detection
        incident = ops.incident_manager.create_incident(
            "Production Outage - API Unavailable",
            "All API endpoints returning 503",
            Severity.P1_CRITICAL,
            ["api-gateway", "backend-service"],
            users_affected=5000
        )
        
        assert incident.severity == Severity.P1_CRITICAL
        
        # On-call acknowledges
        ops.incident_manager.acknowledge_incident(incident.incident_id, "oncall@ultra.com")
        
        # Execute diagnostic runbook
        runbook_result = await ops.runbook_executor.execute_runbook(
            "RB001",
            incident_id=incident.incident_id,
            dry_run=True
        )
        
        assert runbook_result["success"] is True
        
        # Incident resolved
        ops.incident_manager.update_status(
            incident.incident_id,
            IncidentStatus.RESOLVED,
            "Service restarted, traffic recovered"
        )
        
        # PIR scheduled
        pir = ops.pir_manager.create_pir(incident, "oncall@ultra.com")
        
        assert pir is not None
        assert incident.status == IncidentStatus.RESOLVED


def test_severity_enum():
    """Test severity enumeration"""
    assert len(list(Severity)) == 4
    assert Severity.P1_CRITICAL.value == "p1_critical"


def test_incident_status_enum():
    """Test incident status enumeration"""
    assert IncidentStatus.DETECTED.value == "detected"
    assert IncidentStatus.RESOLVED.value == "resolved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
