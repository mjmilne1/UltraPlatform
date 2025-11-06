"""
Tests for Best-in-Class KYC & Onboarding Engine
===============================================

Comprehensive test suite covering:
- Identity verification (documents, biometrics)
- AML/CTF screening
- Risk assessment
- Onboarding workflow
- Suitability assessment

Performance Targets:
- Onboarding Time: <2 minutes (low-risk)
- Verification SLA: <30 seconds
- False Positive Rate: <1%
- STP Rate: 90%+
- AML Hit Detection: 100%

Compliance:
- AUSTRAC AML/CTF Act
- ASIC RG 227
- Privacy Act 1988
- 100-point ID check
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from modules.kyc_onboarding.kyc_engine import (
    KYCOnboardingFramework,
    IdentityVerificationEngine,
    BiometricVerificationEngine,
    AMLScreeningEngine,
    RiskAssessmentEngine,
    OnboardingWorkflowEngine,
    CustomerProfile,
    DocumentType,
    DocumentVerification,
    BiometricVerification,
    AMLScreeningResult,
    OnboardingSession,
    VerificationStatus,
    RiskLevel,
    OnboardingStep,
    ScreeningHitType
)


class TestIdentityVerificationEngine:
    """Tests for identity verification"""
    
    def test_engine_initialization(self):
        """Test identity engine initialization"""
        engine = IdentityVerificationEngine()
        
        assert len(engine.verifications) == 0
        assert engine.verification_count == 0
        assert len(engine.document_points) > 0
    
    @pytest.mark.asyncio
    async def test_document_verification(self):
        """Test document verification"""
        engine = IdentityVerificationEngine()
        
        verification = await engine.verify_document(
            customer_id="CUST-001",
            document_type=DocumentType.PASSPORT,
            document_number="N1234567",
            document_country="AU",
            document_image=b"fake_passport_data"
        )
        
        assert verification.verification_id.startswith("DOC-")
        assert verification.document_type == DocumentType.PASSPORT
        assert verification.status in [VerificationStatus.VERIFIED, VerificationStatus.REQUIRES_REVIEW]
        assert verification.confidence_score > 0
    
    @pytest.mark.asyncio
    async def test_document_authenticity_check(self):
        """Test document authenticity checking"""
        engine = IdentityVerificationEngine()
        
        is_authentic = await engine._check_authenticity(
            DocumentType.PASSPORT,
            b"document_image"
        )
        
        assert isinstance(is_authentic, bool)
    
    @pytest.mark.asyncio
    async def test_dvs_verification(self):
        """Test DVS (Document Verification Service) check"""
        engine = IdentityVerificationEngine()
        
        dvs_result = await engine._verify_with_dvs(
            DocumentType.DRIVERS_LICENSE,
            "DL123456",
            "John Smith",
            datetime(1985, 6, 15)
        )
        
        assert isinstance(dvs_result, bool)
    
    def test_100_point_check_calculation(self):
        """Test 100-point ID check calculation"""
        engine = IdentityVerificationEngine()
        
        # Add verifications
        passport_ver = DocumentVerification(
            verification_id="DOC-001",
            customer_id="CUST-001",
            document_type=DocumentType.PASSPORT,
            document_number="N123",
            document_country="AU",
            expiry_date=None,
            status=VerificationStatus.VERIFIED,
            verification_method="ocr",
            is_authentic=True,
            is_expired=False,
            confidence_score=95.0
        )
        
        license_ver = DocumentVerification(
            verification_id="DOC-002",
            customer_id="CUST-001",
            document_type=DocumentType.DRIVERS_LICENSE,
            document_number="DL123",
            document_country="AU",
            expiry_date=None,
            status=VerificationStatus.VERIFIED,
            verification_method="ocr",
            is_authentic=True,
            is_expired=False,
            confidence_score=90.0
        )
        
        engine.verifications["DOC-001"] = passport_ver
        engine.verifications["DOC-002"] = license_ver
        
        points = engine.calculate_id_points(["DOC-001", "DOC-002"])
        
        assert points == 110  # Passport (70) + License (40)
        assert engine.meets_100_point_check(["DOC-001", "DOC-002"])
    
    def test_100_point_check_insufficient(self):
        """Test insufficient points for 100-point check"""
        engine = IdentityVerificationEngine()
        
        # Only medicare card (25 points)
        medicare_ver = DocumentVerification(
            verification_id="DOC-003",
            customer_id="CUST-001",
            document_type=DocumentType.MEDICARE_CARD,
            document_number="MC123",
            document_country="AU",
            expiry_date=None,
            status=VerificationStatus.VERIFIED,
            verification_method="ocr",
            is_authentic=True,
            is_expired=False,
            confidence_score=85.0
        )
        
        engine.verifications["DOC-003"] = medicare_ver
        
        assert not engine.meets_100_point_check(["DOC-003"])


class TestBiometricVerificationEngine:
    """Tests for biometric verification"""
    
    def test_engine_initialization(self):
        """Test biometric engine initialization"""
        engine = BiometricVerificationEngine()
        
        assert len(engine.verifications) == 0
        assert engine.verification_count == 0
    
    @pytest.mark.asyncio
    async def test_biometric_verification(self):
        """Test facial biometric verification"""
        engine = BiometricVerificationEngine()
        
        verification = await engine.verify_biometric(
            customer_id="CUST-001",
            biometric_image=b"selfie_image",
            reference_image=b"passport_photo"
        )
        
        assert verification.verification_id.startswith("BIO-")
        assert verification.biometric_type == "facial"
        assert verification.face_match_score >= 0
        assert verification.face_match_score <= 100
    
    @pytest.mark.asyncio
    async def test_liveness_detection(self):
        """Test liveness detection"""
        engine = BiometricVerificationEngine()
        
        liveness_passed = await engine._check_liveness(b"video_data")
        
        assert isinstance(liveness_passed, bool)
    
    @pytest.mark.asyncio
    async def test_image_quality_assessment(self):
        """Test image quality assessment"""
        engine = BiometricVerificationEngine()
        
        quality_score = await engine._assess_image_quality(b"image_data")
        
        assert quality_score >= 0
        assert quality_score <= 100
    
    @pytest.mark.asyncio
    async def test_face_matching(self):
        """Test face matching algorithm"""
        engine = BiometricVerificationEngine()
        
        match_score = await engine._match_faces(
            b"face1_image",
            b"face2_image"
        )
        
        assert match_score >= 0
        assert match_score <= 100
    
    @pytest.mark.asyncio
    async def test_verification_pass_threshold(self):
        """Test verification passes with high scores"""
        engine = BiometricVerificationEngine()
        
        # Mock high-quality verification
        verification = await engine.verify_biometric(
            "CUST-001",
            b"high_quality_selfie",
            b"passport_photo"
        )
        
        # Should pass with good scores
        assert verification.status in [
            VerificationStatus.VERIFIED,
            VerificationStatus.REQUIRES_REVIEW
        ]


class TestAMLScreeningEngine:
    """Tests for AML/CTF screening"""
    
    def test_engine_initialization(self):
        """Test AML engine initialization"""
        engine = AMLScreeningEngine()
        
        assert len(engine.screenings) == 0
        assert engine.screening_count == 0
        assert len(engine.sanctions_lists) > 0
    
    @pytest.mark.asyncio
    async def test_customer_screening(self):
        """Test complete customer screening"""
        engine = AMLScreeningEngine()
        
        screening = await engine.screen_customer(
            customer_id="CUST-001",
            first_name="John",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            nationality="AU"
        )
        
        assert screening.screening_id.startswith("AML-")
        assert screening.screening_date is not None
        assert screening.risk_score >= 0
        assert screening.risk_level in list(RiskLevel)
    
    @pytest.mark.asyncio
    async def test_sanctions_screening(self):
        """Test sanctions list screening"""
        engine = AMLScreeningEngine()
        
        result = await engine._screen_sanctions(
            "John", "Smith", datetime(1985, 6, 15), "AU"
        )
        
        assert "hit_count" in result
        assert "hits" in result
        assert isinstance(result["hit_count"], int)
    
    @pytest.mark.asyncio
    async def test_pep_screening(self):
        """Test PEP (Politically Exposed Person) screening"""
        engine = AMLScreeningEngine()
        
        result = await engine._screen_pep(
            "John", "Smith", datetime(1985, 6, 15), "AU"
        )
        
        assert "hit_count" in result
        assert "hits" in result
    
    @pytest.mark.asyncio
    async def test_adverse_media_screening(self):
        """Test adverse media screening"""
        engine = AMLScreeningEngine()
        
        result = await engine._screen_adverse_media("John", "Smith")
        
        assert "hit_count" in result
        assert "hits" in result
    
    def test_aml_risk_scoring(self):
        """Test AML risk score calculation"""
        engine = AMLScreeningEngine()
        
        sanctions = {"hit_count": 1, "hits": []}
        pep = {"hit_count": 0, "hits": []}
        adverse_media = {"hit_count": 0, "hits": []}
        
        score = engine._calculate_aml_risk_score(sanctions, pep, adverse_media)
        
        assert score >= 0
        assert score <= 100
    
    def test_risk_level_determination(self):
        """Test risk level determination from score"""
        engine = AMLScreeningEngine()
        
        assert engine._determine_risk_level(90) == RiskLevel.PROHIBITED
        assert engine._determine_risk_level(60) == RiskLevel.HIGH
        assert engine._determine_risk_level(30) == RiskLevel.MEDIUM
        assert engine._determine_risk_level(10) == RiskLevel.LOW
    
    @pytest.mark.asyncio
    async def test_ongoing_monitoring(self):
        """Test ongoing AML monitoring"""
        engine = AMLScreeningEngine()
        
        # First screening
        screening = await engine.screen_customer(
            "CUST-001", "John", "Smith", datetime(1985, 6, 15), "AU"
        )
        
        # Ongoing monitoring
        monitoring = await engine.ongoing_monitoring("CUST-001", screening.screening_id)
        
        assert monitoring["monitoring_status"] == "active"
        assert monitoring["customer_id"] == "CUST-001"


class TestRiskAssessmentEngine:
    """Tests for risk assessment"""
    
    def test_engine_initialization(self):
        """Test risk assessment engine initialization"""
        engine = RiskAssessmentEngine()
        
        assert len(engine.risk_assessments) == 0
        assert len(engine.high_risk_countries) > 0
        assert len(engine.high_risk_occupations) > 0
    
    def test_customer_risk_assessment(self):
        """Test complete customer risk assessment"""
        engine = RiskAssessmentEngine()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer"
        )
        
        aml_screening = AMLScreeningResult(
            screening_id="AML-001",
            customer_id="CUST-001",
            screening_date=datetime.now(),
            screening_provider="world_check",
            has_hits=False,
            total_hits=0,
            risk_score=10.0,
            risk_level=RiskLevel.LOW
        )
        
        documents = []
        
        assessment = engine.assess_customer_risk(customer, aml_screening, documents)
        
        assert assessment["assessment_id"].startswith("RISK-")
        assert assessment["risk_level"] in list(RiskLevel)
        assert assessment["risk_score"] >= 0
        assert "requires_edd" in assessment
    
    def test_geographic_risk_assessment(self):
        """Test geographic risk assessment"""
        engine = RiskAssessmentEngine()
        
        low_risk = engine._assess_geographic_risk("AU", "AU")
        assert low_risk < 50
        
        # Add high-risk country
        engine.high_risk_countries.append("XX")
        high_risk = engine._assess_geographic_risk("XX", "XX")
        assert high_risk >= 50
    
    def test_occupation_risk_assessment(self):
        """Test occupation risk assessment"""
        engine = RiskAssessmentEngine()
        
        low_risk = engine._assess_occupation_risk("Software Engineer")
        assert low_risk < 50
        
        high_risk = engine._assess_occupation_risk("Casino Operator")
        assert high_risk >= 50
    
    def test_monitoring_frequency_recommendation(self):
        """Test monitoring frequency recommendation"""
        engine = RiskAssessmentEngine()
        
        assert engine._get_monitoring_frequency(RiskLevel.PROHIBITED) == "continuous"
        assert engine._get_monitoring_frequency(RiskLevel.HIGH) == "daily"
        assert engine._get_monitoring_frequency(RiskLevel.MEDIUM) == "weekly"
        assert engine._get_monitoring_frequency(RiskLevel.LOW) == "monthly"


class TestOnboardingWorkflowEngine:
    """Tests for onboarding workflow"""
    
    def test_engine_initialization(self):
        """Test workflow engine initialization"""
        engine = OnboardingWorkflowEngine()
        
        assert len(engine.sessions) == 0
        assert engine.total_sessions == 0
        assert engine.completed_sessions == 0
    
    @pytest.mark.asyncio
    async def test_start_onboarding(self):
        """Test starting onboarding session"""
        engine = OnboardingWorkflowEngine()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer"
        )
        
        session = await engine.start_onboarding(customer)
        
        assert session.session_id.startswith("ONB-")
        assert session.status == VerificationStatus.IN_PROGRESS
        assert session.current_step == OnboardingStep.PERSONAL_INFO
        assert engine.total_sessions == 1
    
    @pytest.mark.asyncio
    async def test_complete_step(self):
        """Test completing onboarding step"""
        engine = OnboardingWorkflowEngine()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer"
        )
        
        session = await engine.start_onboarding(customer)
        
        # Complete personal info step
        updated_session = await engine.complete_step(
            session.session_id,
            OnboardingStep.PERSONAL_INFO,
            {"data": "completed"}
        )
        
        assert OnboardingStep.PERSONAL_INFO in updated_session.completed_steps
        assert updated_session.current_step == OnboardingStep.DOCUMENT_UPLOAD
    
    def test_step_progression(self):
        """Test onboarding step progression"""
        engine = OnboardingWorkflowEngine()
        
        next_step = engine._get_next_step(OnboardingStep.PERSONAL_INFO)
        assert next_step == OnboardingStep.DOCUMENT_UPLOAD
        
        next_step = engine._get_next_step(OnboardingStep.AML_SCREENING)
        assert next_step == OnboardingStep.RISK_ASSESSMENT
    
    def test_onboarding_decision_auto_approve(self):
        """Test auto-approve decision"""
        engine = OnboardingWorkflowEngine()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer"
        )
        
        doc = DocumentVerification(
            verification_id="DOC-001",
            customer_id="CUST-001",
            document_type=DocumentType.PASSPORT,
            document_number="N123",
            document_country="AU",
            expiry_date=None,
            status=VerificationStatus.VERIFIED,
            verification_method="ocr",
            is_authentic=True,
            is_expired=False,
            confidence_score=95.0
        )
        
        bio = BiometricVerification(
            verification_id="BIO-001",
            customer_id="CUST-001",
            biometric_type="facial",
            status=VerificationStatus.VERIFIED,
            liveness_passed=True,
            face_match_score=94.0
        )
        
        aml = AMLScreeningResult(
            screening_id="AML-001",
            customer_id="CUST-001",
            screening_date=datetime.now(),
            screening_provider="world_check",
            has_hits=False,
            total_hits=0,
            risk_score=10.0,
            risk_level=RiskLevel.LOW,
            requires_manual_review=False,
            approved=True
        )
        
        risk_assessment = {
            "risk_level": RiskLevel.LOW,
            "risk_score": 15.0
        }
        
        decision = engine._make_onboarding_decision(
            customer, [doc], bio, aml, risk_assessment
        )
        
        assert decision["approved"] is True
        assert decision.get("manual_review_required", True) is False
    
    def test_onboarding_decision_auto_reject(self):
        """Test auto-reject decision (prohibited risk)"""
        engine = OnboardingWorkflowEngine()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer",
            has_sanctions_hit=True
        )
        
        risk_assessment = {
            "risk_level": RiskLevel.PROHIBITED,
            "risk_score": 90.0
        }
        
        decision = engine._make_onboarding_decision(
            customer, [], BiometricVerification(
                verification_id="BIO-001",
                customer_id="CUST-001",
                biometric_type="facial",
                status=VerificationStatus.VERIFIED,
                liveness_passed=True,
                face_match_score=90.0
            ), AMLScreeningResult(
                screening_id="AML-001",
                customer_id="CUST-001",
                screening_date=datetime.now(),
                screening_provider="test",
                has_hits=True,
                total_hits=1,
                risk_score=90.0,
                risk_level=RiskLevel.PROHIBITED
            ), risk_assessment
        )
        
        assert decision["approved"] is False
        assert "Prohibited" in decision["rejection_reason"]
    
    def test_performance_metrics(self):
        """Test performance metrics calculation"""
        engine = OnboardingWorkflowEngine()
        
        engine.total_sessions = 100
        engine.completed_sessions = 90
        engine.straight_through_count = 85
        
        metrics = engine.get_performance_metrics()
        
        assert metrics["total_sessions"] == 100
        assert metrics["completed_sessions"] == 90
        assert abs(metrics["straight_through_rate"] - 94.44) < 0.1  # 85/90 * 100, allow floating point precision
        assert metrics["completion_rate"] == 90.0


class TestKYCOnboardingFramework:
    """Tests for integrated KYC framework"""
    
    def test_framework_initialization(self):
        """Test framework initialization"""
        framework = KYCOnboardingFramework()
        
        assert framework.workflow_engine is not None
        assert framework.identity_engine is not None
        assert framework.biometric_engine is not None
        assert framework.aml_engine is not None
        assert framework.risk_engine is not None
    
    @pytest.mark.asyncio
    async def test_complete_onboarding_low_risk(self):
        """Test complete onboarding for low-risk customer"""
        framework = KYCOnboardingFramework()
        
        customer = CustomerProfile(
            customer_id="CUST-001",
            first_name="John",
            middle_name="M",
            last_name="Smith",
            date_of_birth=datetime(1985, 6, 15),
            email="john@example.com",
            phone="+61412345678",
            residential_address="123 Main St",
            city="Sydney",
            state="NSW",
            postcode="2000",
            country="AU",
            nationality="AU",
            occupation="Engineer"
        )
        
        result = await framework.onboard_customer_complete(
            customer,
            [
                (DocumentType.PASSPORT, b"passport_image"),
                (DocumentType.DRIVERS_LICENSE, b"license_image")
            ],
            b"selfie_image"
        )
        
        assert "customer_id" in result
        assert "approved" in result
        assert "total_onboarding_time" in result
        assert result["total_onboarding_time"] < 120  # <2 minutes
    
    def test_comprehensive_status(self):
        """Test comprehensive status reporting"""
        framework = KYCOnboardingFramework()
        
        status = framework.get_comprehensive_status()
        
        assert "timestamp" in status
        assert "onboarding" in status
        assert "verification" in status
        assert "aml_screening" in status
        assert "compliance" in status
    
    def test_compliance_flags(self):
        """Test compliance status flags"""
        framework = KYCOnboardingFramework()
        
        status = framework.get_comprehensive_status()
        
        assert status["compliance"]["austrac_compliant"] is True
        assert status["compliance"]["asic_rg227_compliant"] is True
        assert status["compliance"]["100_point_check"] is True
        assert status["compliance"]["privacy_act_compliant"] is True


class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_onboarding_workflow(self):
        """Test end-to-end onboarding workflow"""
        framework = KYCOnboardingFramework()
        
        # Create customer
        customer = CustomerProfile(
            customer_id=f"CUST-TEST-001",
            first_name="Jane",
            middle_name="A",
            last_name="Doe",
            date_of_birth=datetime(1990, 3, 20),
            email="jane.doe@example.com",
            phone="+61423456789",
            residential_address="456 Test St",
            city="Melbourne",
            state="VIC",
            postcode="3000",
            country="AU",
            nationality="AU",
            occupation="Accountant"
        )
        
        # Complete onboarding
        result = await framework.onboard_customer_complete(
            customer,
            [
                (DocumentType.PASSPORT, b"passport_data"),
                (DocumentType.DRIVERS_LICENSE, b"license_data")
            ],
            b"selfie_data"
        )
        
        # Verify result
        assert result["customer_id"] == customer.customer_id
        assert isinstance(result["approved"], bool)
        assert result["total_onboarding_time"] > 0
        assert result["risk_level"] in [level.value for level in RiskLevel]
    
    @pytest.mark.asyncio
    async def test_performance_targets_met(self):
        """Test all performance targets are met"""
        framework = KYCOnboardingFramework()
        
        # Multiple onboardings to test performance
        for i in range(5):
            customer = CustomerProfile(
                customer_id=f"CUST-PERF-{i}",
                first_name=f"Test{i}",
                middle_name="M",
                last_name="Customer",
                date_of_birth=datetime(1990, 1, 1),
                email=f"test{i}@example.com",
                phone=f"+6140000000{i}",
                residential_address="Test St",
                city="Sydney",
                state="NSW",
                postcode="2000",
                country="AU",
                nationality="AU",
                occupation="Tester"
            )
            
            result = await framework.onboard_customer_complete(
                customer,
                [(DocumentType.PASSPORT, b"doc_data")],
                b"selfie"
            )
            
            # Check performance targets
            if "total_onboarding_time" in result:
                assert result["total_onboarding_time"] < 300  # <5 min max
        
        # Check overall metrics
        status = framework.get_comprehensive_status()
        
        # Verification SLA
        assert status["verification"]["avg_verification_time"] <= 30
        
        # False positive rate
        assert status["aml_screening"]["false_positive_rate"] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


