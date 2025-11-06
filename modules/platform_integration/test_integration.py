"""
ULTRA PLATFORM - COMPLETE INTEGRATION TESTS
============================================

Comprehensive test suite for end-to-end platform integration
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.platform_integration.ultra_platform import (
    UltraPlatform,
    WorkflowStage,
    DataDomain,
    CustomerOnboardingState
)


class TestUltraPlatformIntegration:
    """Test suite for complete platform integration"""
    
    @pytest.mark.asyncio
    async def test_platform_initialization(self):
        """Test platform initialization"""
        platform = UltraPlatform()
        
        assert platform is not None
        assert platform.data_mesh is not None
        assert len(platform.data_mesh) == len(DataDomain)
        assert len(platform.active_workflows) == 0
    
    @pytest.mark.asyncio
    async def test_start_onboarding(self):
        """Test starting onboarding workflow"""
        platform = UltraPlatform()
        
        customer_data = {
            "email": "test@example.com",
            "phone": "+61-4-1234-5678"
        }
        
        state = await platform.start_onboarding(customer_data)
        
        assert state.customer_id.startswith("CUST_")
        assert state.application_id.startswith("APP_")
        assert state.current_stage == WorkflowStage.APPLICATION_STARTED
        assert WorkflowStage.APPLICATION_STARTED in state.completed_stages
        assert state.completion_percentage > 0
    
    @pytest.mark.asyncio
    async def test_verify_identity(self):
        """Test identity verification"""
        platform = UltraPlatform()
        
        # Start onboarding first
        customer_data = {"email": "test@example.com"}
        state = await platform.start_onboarding(customer_data)
        
        # Verify identity
        identity_data = {
            "document_type": "PASSPORT",
            "document_number": "N1234567"
        }
        
        result = await platform.verify_identity(state.customer_id, identity_data)
        
        assert result["status"] == "VERIFIED"
        assert result["confidence_score"] > 0
        assert "checks" in result
        assert DataDomain.IDENTITY in state.data_products
    
    @pytest.mark.asyncio
    async def test_kyc_screening(self):
        """Test KYC screening"""
        platform = UltraPlatform()
        
        # Setup
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        
        # KYC screening
        kyc_data = {
            "first_name": "John",
            "last_name": "Doe",
            "date_of_birth": "1985-06-15"
        }
        
        result = await platform.perform_kyc_screening(state.customer_id, kyc_data)
        
        assert result["status"] in ["PASS", "REVIEW", "FAIL"]
        assert "risk_level" in result
        assert DataDomain.KYC in state.data_products
    
    @pytest.mark.asyncio
    async def test_fraud_check(self):
        """Test fraud detection"""
        platform = UltraPlatform()
        
        # Setup
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        await platform.perform_kyc_screening(state.customer_id, {})
        
        # Fraud check
        result = await platform.perform_fraud_check(state.customer_id)
        
        assert "risk_score" in result
        assert "risk_level" in result
        assert DataDomain.FRAUD in state.data_products
    
    @pytest.mark.asyncio
    async def test_ongoing_monitoring_setup(self):
        """Test ongoing monitoring setup"""
        platform = UltraPlatform()
        
        # Setup
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        await platform.perform_kyc_screening(state.customer_id, {})
        await platform.perform_fraud_check(state.customer_id)
        
        # Setup monitoring
        result = await platform.setup_ongoing_monitoring(state.customer_id)
        
        assert result["status"] == "ACTIVE"
        assert "kyc_refresh_schedule" in result
        assert "baselines" in result
        assert DataDomain.MONITORING in state.data_products
    
    @pytest.mark.asyncio
    async def test_open_account(self):
        """Test account opening"""
        platform = UltraPlatform()
        
        # Complete prerequisite steps
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        await platform.perform_kyc_screening(state.customer_id, {})
        await platform.perform_fraud_check(state.customer_id)
        
        # Open account
        account_data = {"account_type": "STANDARD"}
        result = await platform.open_account(state.customer_id, account_data)
        
        assert result["account_id"].startswith("ACC_")
        assert result["status"] == "ACTIVE"
        assert "account_number" in result
        assert DataDomain.ACCOUNT in state.data_products
    
    @pytest.mark.asyncio
    async def test_data_mesh_publish_consume(self):
        """Test data mesh publish and consume"""
        platform = UltraPlatform()
        
        customer_id = "TEST_CUSTOMER"
        
        # Publish to mesh
        await platform._publish_to_mesh(
            DataDomain.IDENTITY,
            {
                "event": "test_event",
                "customer_id": customer_id,
                "data": "test_data"
            }
        )
        
        # Consume from mesh
        data = await platform._consume_from_mesh(DataDomain.IDENTITY, customer_id)
        
        assert data is not None
        assert data["customer_id"] == customer_id
        assert data["event"] == "test_event"
    
    @pytest.mark.asyncio
    async def test_customer_360_view(self):
        """Test customer 360° view"""
        platform = UltraPlatform()
        
        # Complete full workflow
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        await platform.perform_kyc_screening(state.customer_id, {})
        await platform.perform_fraud_check(state.customer_id)
        await platform.setup_ongoing_monitoring(state.customer_id)
        await platform.open_account(state.customer_id, {})
        
        # Get 360° view
        view = await platform.get_customer_360_view(state.customer_id)
        
        assert view["customer_id"] == state.customer_id
        assert "domains" in view
        assert len(view["domains"]) >= 4  # At least 4 domains
        assert "workflow" in view
    
    @pytest.mark.asyncio
    async def test_platform_health(self):
        """Test platform health check"""
        platform = UltraPlatform()
        
        health = await platform.get_platform_health()
        
        assert health["status"] in ["healthy", "degraded", "critical"]
        assert "components" in health
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_platform_metrics(self):
        """Test platform metrics"""
        platform = UltraPlatform()
        
        # Add some workflows
        await platform.start_onboarding({"email": "test1@example.com"})
        await platform.start_onboarding({"email": "test2@example.com"})
        
        metrics = platform.get_platform_metrics()
        
        assert "workflows" in metrics
        assert metrics["workflows"]["active"] == 2
        assert "data_mesh" in metrics
    
    @pytest.mark.asyncio
    async def test_complete_workflow_progression(self):
        """Test complete workflow stage progression"""
        platform = UltraPlatform()
        
        state = await platform.start_onboarding({"email": "test@example.com"})
        initial_completion = state.completion_percentage
        
        await platform.verify_identity(state.customer_id, {})
        assert state.completion_percentage > initial_completion
        
        await platform.perform_kyc_screening(state.customer_id, {})
        await platform.perform_fraud_check(state.customer_id)
        await platform.setup_ongoing_monitoring(state.customer_id)
        await platform.open_account(state.customer_id, {})
        
        # Should have completed multiple stages
        assert len(state.completed_stages) >= 6
        assert state.completion_percentage > 50
    
    @pytest.mark.asyncio
    async def test_data_product_versioning(self):
        """Test data product versioning in mesh"""
        platform = UltraPlatform()
        
        customer_id = "TEST_CUSTOMER"
        
        # Publish multiple versions
        await platform._publish_to_mesh(
            DataDomain.IDENTITY,
            {"customer_id": customer_id, "version": 1}
        )
        
        await platform._publish_to_mesh(
            DataDomain.IDENTITY,
            {"customer_id": customer_id, "version": 2}
        )
        
        # Should get latest version
        data = await platform._consume_from_mesh(DataDomain.IDENTITY, customer_id)
        
        assert data is not None
        assert data["version"] == 2
    
    @pytest.mark.asyncio
    async def test_cross_domain_data_flow(self):
        """Test data flowing across domains"""
        platform = UltraPlatform()
        
        state = await platform.start_onboarding({"email": "test@example.com"})
        
        # Publish to identity domain
        await platform.verify_identity(state.customer_id, {})
        identity_data = await platform._consume_from_mesh(DataDomain.IDENTITY, state.customer_id)
        assert identity_data is not None
        
        # Fraud check should consume identity data
        await platform.perform_fraud_check(state.customer_id)
        fraud_data = await platform._consume_from_mesh(DataDomain.FRAUD, state.customer_id)
        assert fraud_data is not None
        
        # Verify cross-domain data access
        assert identity_data["customer_id"] == fraud_data["customer_id"]


class TestWorkflowValidation:
    """Test workflow validation and error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_customer_workflow(self):
        """Test accessing non-existent workflow"""
        platform = UltraPlatform()
        
        with pytest.raises(ValueError):
            await platform.verify_identity("INVALID_CUSTOMER", {})
    
    @pytest.mark.asyncio
    async def test_account_opening_without_prerequisites(self):
        """Test account opening without completing prerequisites"""
        platform = UltraPlatform()
        
        state = await platform.start_onboarding({"email": "test@example.com"})
        
        # Try to open account without completing other steps
        with pytest.raises(ValueError):
            await platform.open_account(state.customer_id, {})


class TestDataMeshIntegration:
    """Test Data Mesh specific functionality"""
    
    @pytest.mark.asyncio
    async def test_multiple_customers_in_mesh(self):
        """Test multiple customers in data mesh"""
        platform = UltraPlatform()
        
        # Onboard multiple customers
        state1 = await platform.start_onboarding({"email": "cust1@example.com"})
        state2 = await platform.start_onboarding({"email": "cust2@example.com"})
        
        await platform.verify_identity(state1.customer_id, {})
        await platform.verify_identity(state2.customer_id, {})
        
        # Verify both customers' data in mesh
        data1 = await platform._consume_from_mesh(DataDomain.IDENTITY, state1.customer_id)
        data2 = await platform._consume_from_mesh(DataDomain.IDENTITY, state2.customer_id)
        
        assert data1["customer_id"] != data2["customer_id"]
    
    @pytest.mark.asyncio
    async def test_data_mesh_domains_populated(self):
        """Test that data mesh domains get populated"""
        platform = UltraPlatform()
        
        state = await platform.start_onboarding({"email": "test@example.com"})
        await platform.verify_identity(state.customer_id, {})
        await platform.perform_kyc_screening(state.customer_id, {})
        
        # Check data mesh has products
        identity_products = platform.data_mesh[DataDomain.IDENTITY]
        kyc_products = platform.data_mesh[DataDomain.KYC]
        
        assert len(identity_products) > 0
        assert len(kyc_products) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
