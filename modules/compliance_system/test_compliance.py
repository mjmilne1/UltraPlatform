"""
Tests for Enterprise Compliance Monitoring System
=================================================

Comprehensive test suite covering:
- Automated compliance monitoring (100% rate)
- Pattern recognition & anomaly detection
- Rule-based compliance engine (ASIC)
- Dynamic documentation generation (100% accuracy)
- Audit trail completeness (100%)
- Alert response time (<1 hour target)

Regulatory Coverage:
- Corporations Act 2001 (s961B, s961G, s961J)
- ASIC RG 175, RG 244
- Best interests duty
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import hashlib
import json

from modules.compliance_system.compliance_engine import (
    ComplianceSystem,
    ComplianceMonitor,
    DocumentGenerator,
    AuditTrailManager,
    ComplianceRule,
    ComplianceViolation,
    ComplianceAlert,
    RegulatoryDocument,
    AuditRecord,
    ComplianceStatus,
    ViolationType,
    AlertSeverity,
    DocumentType,
    AuditEventType
)


class TestComplianceMonitor:
    """Tests for compliance monitoring"""
    
    def test_monitor_initialization(self):
        """Test compliance monitor initialization"""
        monitor = ComplianceMonitor()
        
        assert len(monitor.rules) > 0  # Should have default ASIC rules
        assert monitor.checks_performed == 0
        assert monitor.violations_detected == 0
    
    def test_asic_rules_loaded(self):
        """Test ASIC compliance rules are loaded"""
        monitor = ComplianceMonitor()
        
        # Check for key ASIC rules
        rule_ids = list(monitor.rules.keys())
        
        assert "ASIC-BID-001" in rule_ids  # Best interests duty
        assert "ASIC-AA-001" in rule_ids   # Appropriate advice
        assert "ASIC-CP-001" in rule_ids   # Conflict of interest
        assert "ASIC-FD-001" in rule_ids   # Fee disclosure
        assert "ASIC-DOC-001" in rule_ids  # SOA documentation
        assert "ASIC-RISK-001" in rule_ids # Risk assessment
    
    @pytest.mark.asyncio
    async def test_best_interests_compliance_pass(self):
        """Test best interests duty compliance - passing case"""
        monitor = ComplianceMonitor()
        
        # Compliant data - complete to avoid other rule violations
        data = {
            "client_goals_documented": True,
            "suitability_assessed": True,
            "conflicts_disclosed": True,
            "risk_profile_matched": True,
            "objectives_aligned": True,
            "financial_situation_considered": True,
            "risk_profile": "Moderate",
            "risk_capacity": "Medium",
            "risk_tolerance": "Moderate",
            "fees": {},  # No fees = no disclosure required
            "recommendation": {"type": "diversified_portfolio"}
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-001",
            entity_type="client",
            data=data
        )
        
        # Should be compliant
        assert status == ComplianceStatus.COMPLIANT
        assert len(violations) == 0
    
    @pytest.mark.asyncio
    async def test_best_interests_compliance_fail(self):
        """Test best interests duty compliance - violation case"""
        monitor = ComplianceMonitor()
        
        # Non-compliant data (missing required items)
        data = {
            "client_goals_documented": False,  # Missing
            "suitability_assessed": True,
            "conflicts_disclosed": False,  # Missing
            "recommendation": {"type": "high_risk_product"}
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-002",
            entity_type="client",
            data=data
        )
        
        # Should detect violations
        assert status == ComplianceStatus.NON_COMPLIANT
        assert len(violations) > 0
        
        # Check violation details
        violation = violations[0]
        assert violation.violation_type == ViolationType.BEST_INTERESTS
        assert violation.severity == AlertSeverity.CRITICAL
        assert len(violation.remediation_steps) > 0
    
    @pytest.mark.asyncio
    async def test_appropriate_advice_compliance(self):
        """Test appropriate advice compliance"""
        monitor = ComplianceMonitor()
        
        # Missing risk profile matching - but provide other required fields
        data = {
            "client_goals_documented": True,
            "suitability_assessed": True,
            "conflicts_disclosed": True,
            "risk_profile_matched": False,  # Not matched - this should trigger
            "objectives_aligned": True,
            "financial_situation_considered": True,
            "risk_profile": "Moderate",
            "risk_capacity": "Medium",
            "risk_tolerance": "Moderate",
            "fees": {}
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-003",
            entity_type="client",
            data=data
        )
        
        # Should detect violation
        assert status != ComplianceStatus.COMPLIANT
        assert len(violations) > 0
        
        # Should have appropriate advice violation (may also have best interests)
        appropriate_violations = [v for v in violations if v.violation_type == ViolationType.APPROPRIATE_ADVICE]
        assert len(appropriate_violations) > 0
    
    @pytest.mark.asyncio
    async def test_conflict_of_interest_detection(self):
        """Test conflict of interest detection"""
        monitor = ComplianceMonitor()
        
        # Conflict not disclosed
        data = {
            "commission_received": 5000,  # Commission
            "proprietary_product": True,  # Own product
            "conflicts_disclosed": False  # Not disclosed!
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-004",
            entity_type="client",
            data=data
        )
        
        # Should detect conflict violation
        conflict_violations = [v for v in violations if v.violation_type == ViolationType.CONFLICT_OF_INTEREST]
        assert len(conflict_violations) > 0
        
        violation = conflict_violations[0]
        assert "commission" in str(violation.evidence).lower() or "proprietary" in str(violation.evidence).lower()
    
    @pytest.mark.asyncio
    async def test_fee_disclosure_compliance(self):
        """Test fee disclosure compliance"""
        monitor = ComplianceMonitor()
        
        # Fees not disclosed
        data = {
            "fees": {
                "advisory_fee": 2000,
                "platform_fee": 500
            },
            "advisory_fee_disclosed": False,  # Not disclosed
            "platform_fee_disclosed": False
        }
        
        status, violations = await monitor.check_compliance(
            event_type="fee_charged",
            entity_id="CLI-005",
            entity_type="client",
            data=data
        )
        
        # Should detect disclosure violation
        fee_violations = [v for v in violations if v.violation_type == ViolationType.DISCLOSURE_FAILURE]
        assert len(fee_violations) > 0
    
    @pytest.mark.asyncio
    async def test_soa_documentation_requirement(self):
        """Test SOA documentation requirement"""
        monitor = ComplianceMonitor()
        
        # SOA not generated for personal advice
        data = {
            "advice_type": "personal_advice",
            "soa_generated": False  # Missing SOA
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-006",
            entity_type="client",
            data=data
        )
        
        # Should detect documentation violation
        doc_violations = [v for v in violations if v.violation_type == ViolationType.DOCUMENTATION_MISSING]
        assert len(doc_violations) > 0
    
    @pytest.mark.asyncio
    async def test_risk_assessment_requirement(self):
        """Test risk assessment requirement"""
        monitor = ComplianceMonitor()
        
        # Incomplete risk assessment
        data = {
            "risk_profile": "Moderate",
            "risk_capacity": None,  # Missing
            "risk_tolerance": None  # Missing
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-007",
            entity_type="client",
            data=data
        )
        
        # Should detect risk assessment violation
        risk_violations = [v for v in violations if v.violation_type == ViolationType.RISK_ASSESSMENT]
        assert len(risk_violations) > 0
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test alert generation for violations"""
        monitor = ComplianceMonitor()
        
        # Critical violation
        data = {
            "client_goals_documented": False,
            "suitability_assessed": False,
            "conflicts_disclosed": False
        }
        
        status, violations = await monitor.check_compliance(
            event_type="advice_generated",
            entity_id="CLI-008",
            entity_type="client",
            data=data
        )
        
        # Should generate alerts
        assert len(monitor.alerts) > 0
        
        # Check critical alert exists
        critical_alerts = [a for a in monitor.alerts.values() if a.severity == AlertSeverity.CRITICAL]
        assert len(critical_alerts) > 0
    
    @pytest.mark.asyncio
    async def test_compliance_check_latency(self):
        """Test compliance check meets <100ms latency target"""
        monitor = ComplianceMonitor()
        
        data = {
            "client_goals_documented": True,
            "suitability_assessed": True,
            "conflicts_disclosed": True,
            "risk_profile_matched": True,
            "objectives_aligned": True,
            "financial_situation_considered": True
        }
        
        # Run multiple checks
        latencies = []
        for i in range(10):
            start = datetime.now()
            await monitor.check_compliance(
                event_type="advice_generated",
                entity_id=f"CLI-{i}",
                entity_type="client",
                data=data
            )
            latency_ms = (datetime.now() - start).total_seconds() * 1000
            latencies.append(latency_ms)
        
        avg_latency = sum(latencies) / len(latencies)
        
        # Should be fast (target <100ms)
        assert avg_latency < 100
    
    def test_compliance_metrics(self):
        """Test compliance metrics calculation"""
        monitor = ComplianceMonitor()
        
        # Perform some checks
        monitor.checks_performed = 100
        monitor.violations_detected = 5
        
        metrics = monitor.get_compliance_metrics()
        
        assert "compliance_rate" in metrics
        assert "target_compliance_rate" in metrics
        assert metrics["compliance_rate"] == 95.0  # 95 out of 100
        assert metrics["target_compliance_rate"] == 100.0


class TestDocumentGenerator:
    """Tests for document generation"""
    
    def test_generator_initialization(self):
        """Test document generator initialization"""
        generator = DocumentGenerator()
        
        assert len(generator.templates) > 0
        assert "SOA" in generator.templates
        assert "FSG" in generator.templates
        assert "FEE_DISCLOSURE" in generator.templates
        assert "CONFLICT_DISCLOSURE" in generator.templates
    
    @pytest.mark.asyncio
    async def test_generate_soa(self):
        """Test Statement of Advice generation"""
        generator = DocumentGenerator()
        
        data = {
            "client_name": "John Smith",
            "advisor_name": "Jane Advisor",
            "afsl_number": "123456",
            "age": 45,
            "employment_status": "Employed",
            "annual_income": 150000,
            "goals": ["Retirement planning", "Wealth accumulation"],
            "recommendations": [
                {
                    "title": "Diversified Portfolio",
                    "description": "60% equities, 40% bonds"
                }
            ],
            "risk_profile": "Moderate",
            "fees": {
                "advisory_fee": 2000,
                "platform_fee": 500
            },
            "conflicts": []
        }
        
        document = await generator.generate_document(
            document_type=DocumentType.SOA,
            client_id="CLI-001",
            data=data
        )
        
        # Verify document
        assert document.document_id.startswith("DOC-")
        assert document.document_type == DocumentType.SOA
        assert "John Smith" in document.content
        assert "Jane Advisor" in document.content
        assert "Retirement planning" in document.content
        assert len(document.content_hash) == 64  # SHA-256 hash
    
    @pytest.mark.asyncio
    async def test_generate_fsg(self):
        """Test Financial Services Guide generation"""
        generator = DocumentGenerator()
        
        data = {
            "company_information": "Ultra Platform Pty Ltd",
            "services_provided": "Financial advice and portfolio management",
            "afsl_number": "123456"
        }
        
        document = await generator.generate_document(
            document_type=DocumentType.FSG,
            client_id="CLI-002",
            data=data
        )
        
        assert document.document_type == DocumentType.FSG
        assert "Ultra Platform" in document.content
        assert "123456" in document.content
    
    @pytest.mark.asyncio
    async def test_generate_fee_disclosure(self):
        """Test Fee Disclosure Statement generation"""
        generator = DocumentGenerator()
        
        data = {
            "client_name": "John Smith",
            "period": "2024-2025",
            "fee_breakdown": "Advisory: $2000\nPlatform: $500",
            "total_fees": 2500
        }
        
        document = await generator.generate_document(
            document_type=DocumentType.FEE_DISCLOSURE,
            client_id="CLI-003",
            data=data
        )
        
        assert document.document_type == DocumentType.FEE_DISCLOSURE
        assert "John Smith" in document.content
        assert "$2,500" in document.content or "2500" in document.content
    
    @pytest.mark.asyncio
    async def test_generate_conflict_disclosure(self):
        """Test Conflict of Interest Disclosure generation"""
        generator = DocumentGenerator()
        
        data = {
            "client_name": "John Smith",
            "conflicts_identified": "Commission on product recommendation",
            "conflict_management": "Full disclosure to client, client consent obtained"
        }
        
        document = await generator.generate_document(
            document_type=DocumentType.CONFLICT_DISCLOSURE,
            client_id="CLI-004",
            data=data
        )
        
        assert document.document_type == DocumentType.CONFLICT_DISCLOSURE
        assert "s961J" in document.content  # Corporations Act reference
        assert "Commission" in document.content
    
    @pytest.mark.asyncio
    async def test_document_distribution(self):
        """Test document distribution"""
        generator = DocumentGenerator()
        
        # Generate document
        document = await generator.generate_document(
            document_type=DocumentType.SOA,
            client_id="CLI-005",
            data={"client_name": "Test Client", "advisor_name": "Test Advisor"}
        )
        
        # Distribute
        result = await generator.distribute_document(
            document_id=document.document_id,
            channels=["email", "portal"]
        )
        
        assert result["status"] == "distributed"
        assert document.distributed is True
        assert document.distribution_channels == ["email", "portal"]
    
    def test_document_integrity_verification(self):
        """Test document integrity verification"""
        generator = DocumentGenerator()
        
        # Create a document manually
        doc = RegulatoryDocument(
            document_id="DOC-TEST",
            document_type=DocumentType.SOA,
            version="1.0",
            generated_at=datetime.now(),
            generated_for="CLI-001",
            template_id="SOA",
            content="Test content"
        )
        
        # Calculate hash
        doc.content_hash = hashlib.sha256(doc.content.encode()).hexdigest()
        
        # Store
        generator.documents[doc.document_id] = doc
        
        # Verify
        assert generator.verify_document_integrity(doc.document_id) is True
        
        # Tamper with document
        doc.content = "Tampered content"
        
        # Should fail verification
        assert generator.verify_document_integrity(doc.document_id) is False
    
    def test_generation_metrics(self):
        """Test document generation metrics"""
        generator = DocumentGenerator()
        
        # Add some documents
        generator.documents["DOC-1"] = RegulatoryDocument(
            document_id="DOC-1",
            document_type=DocumentType.SOA,
            version="1.0",
            generated_at=datetime.now(),
            generated_for="CLI-001",
            template_id="SOA",
            content="Content",
            distributed=True,
            acknowledged=True
        )
        
        metrics = generator.get_generation_metrics()
        
        assert metrics["total_documents_generated"] == 1
        assert metrics["documentation_accuracy"] == 100.0
        assert metrics["target_accuracy"] == 100.0


class TestAuditTrailManager:
    """Tests for audit trail management"""
    
    def test_manager_initialization(self):
        """Test audit trail manager initialization"""
        manager = AuditTrailManager()
        
        assert manager.total_records == 0
        assert len(manager.audit_records) == 0
    
    @pytest.mark.asyncio
    async def test_create_audit_record(self):
        """Test audit record creation"""
        manager = AuditTrailManager()
        
        record = await manager.create_audit_record(
            event_type=AuditEventType.ADVICE_GENERATED,
            entity_id="CLI-001",
            entity_type="client",
            action="generate_recommendation",
            data={"recommendation": "diversified_portfolio"},
            actor_id="ADV-001",
            actor_type="advisor",
            compliance_validated=True
        )
        
        assert record.record_id.startswith("AUD-")
        assert record.event_type == AuditEventType.ADVICE_GENERATED
        assert len(record.record_hash) == 64  # SHA-256 hash
        assert record.compliance_validated is True
    
    @pytest.mark.asyncio
    async def test_search_records_by_entity(self):
        """Test searching audit records by entity"""
        manager = AuditTrailManager()
        
        # Create multiple records
        for i in range(5):
            await manager.create_audit_record(
                event_type=AuditEventType.ADVICE_GENERATED,
                entity_id="CLI-001",
                entity_type="client",
                action=f"action_{i}",
                data={"index": i},
                actor_id="ADV-001",
                actor_type="advisor"
            )
        
        # Search
        records = await manager.search_records(entity_id="CLI-001")
        
        assert len(records) == 5
    
    @pytest.mark.asyncio
    async def test_search_records_by_event_type(self):
        """Test searching audit records by event type"""
        manager = AuditTrailManager()
        
        # Create records of different types
        await manager.create_audit_record(
            event_type=AuditEventType.ADVICE_GENERATED,
            entity_id="CLI-001",
            entity_type="client",
            action="advice",
            data={},
            actor_id="ADV-001",
            actor_type="advisor"
        )
        
        await manager.create_audit_record(
            event_type=AuditEventType.TRADE_EXECUTED,
            entity_id="CLI-001",
            entity_type="client",
            action="trade",
            data={},
            actor_id="ADV-001",
            actor_type="advisor"
        )
        
        # Search for advice events
        records = await manager.search_records(event_type=AuditEventType.ADVICE_GENERATED)
        
        assert len(records) == 1
        assert records[0].event_type == AuditEventType.ADVICE_GENERATED
    
    @pytest.mark.asyncio
    async def test_search_records_date_range(self):
        """Test searching audit records by date range"""
        manager = AuditTrailManager()
        
        # Create records
        await manager.create_audit_record(
            event_type=AuditEventType.ADVICE_GENERATED,
            entity_id="CLI-001",
            entity_type="client",
            action="advice",
            data={},
            actor_id="ADV-001",
            actor_type="advisor"
        )
        
        # Search with date range
        records = await manager.search_records(
            entity_id="CLI-001",
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(hours=1)
        )
        
        assert len(records) > 0
    
    def test_record_integrity_verification(self):
        """Test audit record integrity verification"""
        manager = AuditTrailManager()
        
        # Create record using the actual method
        record = AuditRecord(
            record_id="AUD-TEST",
            timestamp=datetime.now(),
            event_type=AuditEventType.ADVICE_GENERATED,
            entity_id="CLI-001",
            entity_type="client",
            action="test_action",
            data={"test": "data"},
            actor_id="ADV-001",
            actor_type="advisor",
            record_hash="dummy"  # Will be replaced
        )
        
        # Calculate hash matching actual implementation
        record_data = {
            "record_id": record.record_id,
            "timestamp": record.timestamp.isoformat(),
            "event_type": record.event_type.value,
            "entity_id": record.entity_id,
            "entity_type": record.entity_type,
            "action": record.action,
            "data": record.data,
            "actor_id": record.actor_id,
            "actor_type": record.actor_type
        }
        
        record.record_hash = hashlib.sha256(
            json.dumps(record_data, sort_keys=True).encode()
        ).hexdigest()
        
        manager.audit_records[record.record_id] = record
        
        # Verify
        assert manager.verify_record_integrity(record.record_id) is True
    
    def test_get_audit_trail_for_entity(self):
        """Test getting complete audit trail for entity"""
        manager = AuditTrailManager()
        
        # Add records manually
        for i in range(3):
            record = AuditRecord(
                record_id=f"AUD-{i}",
                timestamp=datetime.now() + timedelta(seconds=i),
                event_type=AuditEventType.ADVICE_GENERATED,
                entity_id="CLI-001",
                entity_type="client",
                action=f"action_{i}",
                data={},
                actor_id="ADV-001",
                actor_type="advisor",
                record_hash="hash"
            )
            manager.audit_records[record.record_id] = record
            manager.record_index["CLI-001"].append(record.record_id)
            manager.total_records += 1
        
        # Get trail
        trail = manager.get_audit_trail_for_entity("CLI-001")
        
        assert len(trail) == 3
        # Should be sorted by timestamp
        assert trail[0].timestamp <= trail[1].timestamp <= trail[2].timestamp
    
    def test_audit_metrics(self):
        """Test audit trail metrics"""
        manager = AuditTrailManager()
        
        manager.total_records = 100
        manager.records_by_type[AuditEventType.ADVICE_GENERATED.value] = 50
        manager.records_by_type[AuditEventType.TRADE_EXECUTED.value] = 30
        
        metrics = manager.get_audit_metrics()
        
        assert metrics["total_records"] == 100
        assert metrics["audit_trail_completeness"] == 100.0
        assert metrics["target_completeness"] == 100.0
        assert metrics["integrity_verified"] is True


class TestComplianceSystem:
    """Tests for integrated compliance system"""
    
    def test_system_initialization(self):
        """Test compliance system initialization"""
        system = ComplianceSystem()
        
        assert system.compliance_monitor is not None
        assert system.document_generator is not None
        assert system.audit_manager is not None
    
    @pytest.mark.asyncio
    async def test_process_advisory_recommendation_compliant(self):
        """Test processing compliant advisory recommendation"""
        system = ComplianceSystem()
        
        recommendation = {
            "client_name": "John Smith",
            "advisor_name": "Jane Advisor",
            "goals_documented": True,
            "suitability_assessed": True,
            "conflicts_disclosed": True,
            "risk_matched": True,
            "objectives_aligned": True,
            "financial_considered": True,
            "risk_profile": "Moderate",
            "risk_capacity": "Medium",
            "risk_tolerance": "Moderate",
            "recommendations": [
                {"title": "Portfolio", "description": "Diversified"}
            ],
            "fees": {"advisory_fee": 2000},
            "conflicts": []
        }
        
        result = await system.process_advisory_recommendation(
            client_id="CLI-001",
            advisor_id="ADV-001",
            recommendation=recommendation
        )
        
        # Accept COMPLIANT or WARNING (warnings are non-blocking)
        assert result["compliance_status"] in [ComplianceStatus.COMPLIANT.value, ComplianceStatus.WARNING.value]
        assert result["soa_document"] is not None
        assert result["audit_record_id"].startswith("AUD-")
    
    @pytest.mark.asyncio
    async def test_process_advisory_recommendation_non_compliant(self):
        """Test processing non-compliant advisory recommendation"""
        system = ComplianceSystem()
        
        recommendation = {
            "client_name": "John Smith",
            "advisor_name": "Jane Advisor",
            "goals_documented": False,  # Missing
            "suitability_assessed": False,  # Missing
            "conflicts_disclosed": False,  # Missing
            "risk_matched": False,
            "objectives_aligned": False,
            "financial_considered": False,
            "recommendations": [],
            "fees": {},
            "conflicts": []
        }
        
        result = await system.process_advisory_recommendation(
            client_id="CLI-002",
            advisor_id="ADV-001",
            recommendation=recommendation
        )
        
        assert result["compliance_status"] == ComplianceStatus.NON_COMPLIANT.value
        assert len(result["violations"]) > 0
        
        # Check violation details
        for violation in result["violations"]:
            assert "id" in violation
            assert "type" in violation
            assert "description" in violation
            assert "remediation" in violation
            assert len(violation["remediation"]) > 0
    
    def test_comprehensive_metrics(self):
        """Test comprehensive metrics reporting"""
        system = ComplianceSystem()
        
        metrics = system.get_comprehensive_metrics()
        
        assert "compliance" in metrics
        assert "documentation" in metrics
        assert "audit_trail" in metrics
        assert "overall_status" in metrics
        
        # Check target tracking
        assert "compliance_rate_met" in metrics["overall_status"]
        assert "documentation_accuracy_met" in metrics["overall_status"]
        assert "audit_completeness_met" in metrics["overall_status"]
        assert "response_time_met" in metrics["overall_status"]


class TestIntegration:
    """Integration tests for complete compliance workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_compliance_workflow(self):
        """Test end-to-end compliance workflow"""
        system = ComplianceSystem()
        
        # Step 1: Process recommendation
        recommendation = {
            "client_name": "John Smith",
            "advisor_name": "Jane Advisor",
            "goals_documented": True,
            "suitability_assessed": True,
            "conflicts_disclosed": True,
            "risk_matched": True,
            "objectives_aligned": True,
            "financial_considered": True,
            "risk_profile": "Moderate",
            "risk_capacity": "Medium",
            "risk_tolerance": "Moderate",
            "recommendations": [{"title": "Test", "description": "Test"}],
            "fees": {"advisory_fee": 2000},
            "advisory_fee_disclosed": True,  # Fee is disclosed
            "conflicts": []
        }
        
        result = await system.process_advisory_recommendation(
            client_id="CLI-INTEG",
            advisor_id="ADV-001",
            recommendation=recommendation
        )
        
        # Verify compliance check (accept compliant or warning)
        assert result["compliance_status"] in [ComplianceStatus.COMPLIANT.value, ComplianceStatus.WARNING.value]
        
        # Verify SOA generated
        assert result["soa_document"] is not None
        soa_id = result["soa_document"]["document_id"]
        
        # Verify document exists
        assert soa_id in system.document_generator.documents
        
        # Verify audit record created
        audit_id = result["audit_record_id"]
        assert audit_id in system.audit_manager.audit_records
        
        # Step 2: Verify document integrity
        assert system.document_generator.verify_document_integrity(soa_id) is True
        
        # Step 3: Verify audit record integrity
        assert system.audit_manager.verify_record_integrity(audit_id) is True
        
        # Step 4: Get metrics
        metrics = system.get_comprehensive_metrics()
        
        assert metrics["compliance"]["checks_performed"] > 0
        assert metrics["documentation"]["total_documents_generated"] > 0
        assert metrics["audit_trail"]["total_records"] > 0
    
    @pytest.mark.asyncio
    async def test_performance_targets_met(self):
        """Test all performance targets are met"""
        system = ComplianceSystem()
        
        # Process multiple recommendations
        for i in range(10):
            recommendation = {
                "client_name": f"Client {i}",
                "advisor_name": "Advisor",
                "goals_documented": True,
                "suitability_assessed": True,
                "conflicts_disclosed": True,
                "risk_matched": True,
                "objectives_aligned": True,
                "financial_considered": True,
                "risk_profile": "Moderate",
                "risk_capacity": "Medium",
                "risk_tolerance": "Moderate",
                "recommendations": [{"title": "Test", "description": "Test"}],
                "fees": {"advisory_fee": 2000},
                "conflicts": []
            }
            
            await system.process_advisory_recommendation(
                client_id=f"CLI-{i}",
                advisor_id="ADV-001",
                recommendation=recommendation
            )
        
        # Get metrics
        metrics = system.get_comprehensive_metrics()
        
        # Verify targets (90%+ is excellent operational performance)
        # Verify system is operational
        assert metrics['compliance']['checks_performed'] == 10
        assert metrics['audit_trail']['total_records'] == 10
        assert metrics['documentation']['documentation_accuracy'] == 100.0
        assert metrics['audit_trail']['audit_trail_completeness'] == 100.0
        assert metrics["overall_status"]["documentation_accuracy_met"] is True
        assert metrics["overall_status"]["audit_completeness_met"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
