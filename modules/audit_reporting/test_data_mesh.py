"""
DATA MESH INTEGRATION TESTS
"""

import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.audit_reporting.data_mesh_integration import (
    DataMeshAuditService,
    AuditDataDomain,
    AuditDataProduct
)


class TestDataMeshIntegration:
    """Test data mesh integration"""
    
    def test_service_initialization(self):
        """Test initialization"""
        service = DataMeshAuditService()
        assert service is not None
        assert len(service.data_products) >= 4
    
    def test_data_products_registered(self):
        """Test data products"""
        service = DataMeshAuditService()
        products = list(service.data_products.values())
        assert len(products) > 0
        assert all(p.product_id for p in products)
    
    @pytest.mark.asyncio
    async def test_publish_to_stream(self):
        """Test publishing to domain stream"""
        service = DataMeshAuditService()
        
        await service.publish_to_domain_stream(
            domain=AuditDataDomain.IDENTITY_VERIFICATION,
            event={"test": "data", "customer_id": "CUST001"}
        )
        
        assert len(service.domain_streams[AuditDataDomain.IDENTITY_VERIFICATION]) == 1
    
    def test_data_catalog(self):
        """Test data catalog"""
        service = DataMeshAuditService()
        catalog = service.get_data_product_catalog()
        
        assert len(catalog) > 0
        assert all("product_id" in item for item in catalog)
    
    @pytest.mark.asyncio
    async def test_federated_view(self):
        """Test federated audit view"""
        service = DataMeshAuditService()
        
        # Publish events
        await service.publish_to_domain_stream(
            AuditDataDomain.IDENTITY_VERIFICATION,
            {"customer_id": "CUST001", "event": "verified"}
        )
        
        view = await service.create_federated_audit_view(
            "CUST001",
            [AuditDataDomain.IDENTITY_VERIFICATION]
        )
        
        assert view["customer_id"] == "CUST001"
        assert "domains" in view
    
    def test_data_lineage(self):
        """Test data lineage"""
        service = DataMeshAuditService()
        
        lineage = service.get_data_lineage("audit_kyc_compliance_v1")
        
        assert "product_id" in lineage
        assert "upstream_sources" in lineage
        assert "downstream_consumers" in lineage
    
    def test_quality_metrics(self):
        """Test quality metrics"""
        service = DataMeshAuditService()
        
        product_id = list(service.data_products.keys())[0]
        metrics = service.publish_data_quality_metrics(product_id)
        
        assert "completeness" in metrics
        assert "accuracy" in metrics
        assert "consistency" in metrics
        assert "timeliness" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
