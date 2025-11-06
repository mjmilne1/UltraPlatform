"""
Tests for Enterprise Security and Data Protection Framework
===========================================================

Comprehensive test suite covering:
- Encryption management (AES-256)
- Authentication & authorization (MFA, RBAC, ABAC)
- Security monitoring (24/7 SOC)
- Business continuity (RTO < 2h, RPO < 15min)
- Incident response

Security Standards:
- OWASP Top 10
- SOC 2 Type II
- ISO 27001
- Australian Privacy Principles
"""

import pytest
import asyncio
from datetime import datetime, timedelta

from modules.security_framework.security_engine import (
    SecurityFramework,
    EncryptionManager,
    AuthenticationManager,
    AuthorizationManager,
    SecurityMonitor,
    BusinessContinuityManager,
    SecurityLevel,
    AuthenticationMethod,
    IncidentSeverity,
    AccessDecision,
    UserIdentity,
    AccessToken,
    SecurityEvent,
    SecurityIncident
)


class TestEncryptionManager:
    """Tests for encryption management"""
    
    def test_manager_initialization(self):
        """Test encryption manager initialization"""
        manager = EncryptionManager()
        
        assert len(manager.keys) > 0
        assert manager.master_key is not None
    
    def test_key_creation(self):
        """Test encryption key creation"""
        manager = EncryptionManager()
        
        key = manager.create_key(
            purpose="test_encryption",
            key_type="AES-256"
        )
        
        assert key.key_id.startswith("KEY-")
        assert key.key_type == "AES-256"
        assert key.purpose == "test_encryption"
        assert key.status == "active"
    
    def test_data_encryption_decryption(self):
        """Test data encryption and decryption"""
        manager = EncryptionManager()
        
        original_data = b"Sensitive client financial data"
        
        # Encrypt
        encrypted_data, key_id = manager.encrypt_data(
            original_data,
            purpose="data_encryption"
        )
        
        assert encrypted_data != original_data
        assert key_id.startswith("KEY-")
        
        # Decrypt
        decrypted_data = manager.decrypt_data(encrypted_data, key_id)
        
        assert decrypted_data == original_data
    
    def test_password_hashing(self):
        """Test password hashing with bcrypt"""
        manager = EncryptionManager()
        
        password = "SecureP@ssw0rd!123"
        
        # Hash password
        password_hash = manager.hash_password(password)
        
        assert password_hash != password
        assert len(password_hash) > 0
        
        # Verify correct password
        assert manager.verify_password(password, password_hash) is True
        
        # Verify incorrect password
        assert manager.verify_password("WrongPassword", password_hash) is False
    
    def test_key_rotation(self):
        """Test encryption key rotation"""
        manager = EncryptionManager()
        
        # Create key
        old_key = manager.create_key(
            purpose="rotation_test",
            key_type="AES-256"
        )
        
        # Rotate key
        new_key = manager.rotate_key(old_key.key_id)
        
        assert new_key.key_id != old_key.key_id
        assert new_key.purpose == old_key.purpose
        assert old_key.status == "retiring"
        assert new_key.status == "active"
    
    def test_secure_token_generation(self):
        """Test cryptographically secure token generation"""
        manager = EncryptionManager()
        
        token1 = manager.generate_secure_token(32)
        token2 = manager.generate_secure_token(32)
        
        assert len(token1) > 0
        assert len(token2) > 0
        assert token1 != token2  # Should be unique


class TestAuthenticationManager:
    """Tests for authentication management"""
    
    def test_manager_initialization(self):
        """Test authentication manager initialization"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        assert auth_mgr.max_failed_attempts == 5
        assert len(auth_mgr.users) == 0
    
    def test_user_creation(self):
        """Test user account creation"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        user = auth_mgr.create_user(
            username="test.user",
            email="test@example.com",
            password="SecurePass123!",
            roles=["client"]
        )
        
        assert user.user_id.startswith("USR-")
        assert user.username == "test.user"
        assert user.password_hash is not None
        assert "client" in user.roles
    
    @pytest.mark.asyncio
    async def test_successful_authentication(self):
        """Test successful user authentication"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user
        user = auth_mgr.create_user(
            username="john.doe",
            email="john@example.com",
            password="ValidPassword123!"
        )
        
        # Authenticate
        decision, token = await auth_mgr.authenticate(
            username="john.doe",
            password="ValidPassword123!"
        )
        
        assert decision == AccessDecision.ALLOW
        assert token is not None
        assert token.user_id == user.user_id
        assert user.failed_login_attempts == 0
    
    @pytest.mark.asyncio
    async def test_failed_authentication(self):
        """Test failed authentication with wrong password"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user
        auth_mgr.create_user(
            username="jane.doe",
            email="jane@example.com",
            password="CorrectPassword123!"
        )
        
        # Authenticate with wrong password
        decision, token = await auth_mgr.authenticate(
            username="jane.doe",
            password="WrongPassword"
        )
        
        assert decision == AccessDecision.DENY
        assert token is None
    
    @pytest.mark.asyncio
    async def test_account_lockout(self):
        """Test account lockout after max failed attempts"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user
        user = auth_mgr.create_user(
            username="lockout.test",
            email="lockout@example.com",
            password="CorrectPassword123!"
        )
        
        # Attempt login with wrong password 5 times
        for i in range(5):
            await auth_mgr.authenticate(
                username="lockout.test",
                password="WrongPassword"
            )
        
        # Account should be locked
        assert user.account_status == "locked"
    
    @pytest.mark.asyncio
    async def test_mfa_challenge(self):
        """Test MFA authentication flow"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user and enable MFA
        user = auth_mgr.create_user(
            username="mfa.user",
            email="mfa@example.com",
            password="Password123!"
        )
        
        mfa_secret = auth_mgr.enable_mfa(user.user_id)
        
        assert user.mfa_enabled is True
        assert mfa_secret is not None
        
        # Authenticate without MFA code - should challenge
        decision, token = await auth_mgr.authenticate(
            username="mfa.user",
            password="Password123!"
        )
        
        assert decision == AccessDecision.CHALLENGE
        assert token is None
    
    def test_token_validation(self):
        """Test access token validation"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user and token
        user = auth_mgr.create_user(
            username="token.test",
            email="token@example.com",
            password="Password123!"
        )
        
        token = auth_mgr._generate_access_token(user)
        
        # Validate token
        is_valid, user_id = auth_mgr.validate_token(token.token)
        
        assert is_valid is True
        assert user_id == user.user_id
    
    def test_token_revocation(self):
        """Test token revocation"""
        encryption_mgr = EncryptionManager()
        auth_mgr = AuthenticationManager(encryption_mgr)
        
        # Create user and token
        user = auth_mgr.create_user(
            username="revoke.test",
            email="revoke@example.com",
            password="Password123!"
        )
        
        token = auth_mgr._generate_access_token(user)
        
        # Revoke token
        auth_mgr.revoke_token(token.token_id)
        
        assert token.revoked is True
        
        # Validation should fail
        is_valid, user_id = auth_mgr.validate_token(token.token)
        assert is_valid is False


class TestAuthorizationManager:
    """Tests for authorization management"""
    
    def test_manager_initialization(self):
        """Test authorization manager initialization"""
        authz_mgr = AuthorizationManager()
        
        # Default roles should be created
        assert "client" in authz_mgr.roles
        assert "advisor" in authz_mgr.roles
        assert "compliance_officer" in authz_mgr.roles
        assert "admin" in authz_mgr.roles
    
    def test_role_creation(self):
        """Test custom role creation"""
        authz_mgr = AuthorizationManager()
        
        authz_mgr.create_role("analyst", {
            "view_reports",
            "generate_analytics"
        })
        
        assert "analyst" in authz_mgr.roles
        assert "view_reports" in authz_mgr.roles["analyst"]
    
    def test_role_assignment(self):
        """Test role assignment to user"""
        authz_mgr = AuthorizationManager()
        
        user_id = "USR-12345"
        
        authz_mgr.assign_role(user_id, "advisor")
        
        assert "advisor" in authz_mgr.user_roles[user_id]
    
    def test_permission_check_allowed(self):
        """Test permission check - allowed"""
        authz_mgr = AuthorizationManager()
        
        user_id = "USR-12345"
        authz_mgr.assign_role(user_id, "advisor")
        
        # Advisor should have view_client_data permission
        has_permission = authz_mgr.check_permission(
            user_id,
            "view_client_data"
        )
        
        assert has_permission is True
    
    def test_permission_check_denied(self):
        """Test permission check - denied"""
        authz_mgr = AuthorizationManager()
        
        user_id = "USR-12345"
        authz_mgr.assign_role(user_id, "client")
        
        # Client should not have manage_users permission
        has_permission = authz_mgr.check_permission(
            user_id,
            "manage_users"
        )
        
        assert has_permission is False
    
    def test_context_based_access(self):
        """Test attribute-based access control (ABAC)"""
        authz_mgr = AuthorizationManager()
        
        user_id = "USR-ADVISOR-1"
        authz_mgr.assign_role(user_id, "advisor")
        
        # Should be allowed - advisor viewing their own client
        has_permission = authz_mgr.check_permission(
            user_id,
            "view_client_data",
            context={
                "client_id": "CLI-123",
                "assigned_advisor": user_id
            }
        )
        
        assert has_permission is True
        
        # Should be denied - advisor viewing someone else's client
        has_permission = authz_mgr.check_permission(
            user_id,
            "view_client_data",
            context={
                "client_id": "CLI-456",
                "assigned_advisor": "USR-ADVISOR-2"
            }
        )
        
        assert has_permission is False
    
    def test_get_user_permissions(self):
        """Test getting all user permissions"""
        authz_mgr = AuthorizationManager()
        
        user_id = "USR-12345"
        authz_mgr.assign_role(user_id, "advisor")
        authz_mgr.assign_role(user_id, "client")
        
        permissions = authz_mgr.get_user_permissions(user_id)
        
        assert len(permissions) > 0
        assert "view_client_data" in permissions
        assert "view_own_data" in permissions


class TestSecurityMonitor:
    """Tests for security monitoring"""
    
    def test_monitor_initialization(self):
        """Test security monitor initialization"""
        monitor = SecurityMonitor()
        
        assert len(monitor.threat_rules) > 0
        assert monitor.event_count == 0
        assert monitor.incident_count == 0
    
    @pytest.mark.asyncio
    async def test_event_logging(self):
        """Test security event logging"""
        monitor = SecurityMonitor()
        
        event = await monitor.log_event(
            event_type="login",
            user_id="USR-12345",
            resource="authentication_service",
            action="login",
            success=True,
            severity=IncidentSeverity.LOW
        )
        
        assert event.event_id.startswith("EVT-")
        assert event.success is True
        assert monitor.event_count == 1
    
    @pytest.mark.asyncio
    async def test_brute_force_detection(self):
        """Test brute force attack detection"""
        monitor = SecurityMonitor()
        
        user_id = "USR-BRUTE"
        
        # Simulate 6 failed login attempts
        for i in range(6):
            await monitor.log_event(
                event_type="login",
                user_id=user_id,
                resource="authentication_service",
                action="login",
                success=False,
                severity=IncidentSeverity.LOW
            )
        
        # Should have created incident
        assert monitor.incident_count > 0
        
        incidents = [
            i for i in monitor.incidents.values()
            if "Brute Force" in i.title
        ]
        
        assert len(incidents) > 0
    
    @pytest.mark.asyncio
    async def test_incident_creation(self):
        """Test security incident creation"""
        monitor = SecurityMonitor()
        
        incident = await monitor.create_incident(
            title="Test Security Incident",
            description="Test incident for validation",
            incident_type="test",
            severity=IncidentSeverity.HIGH,
            detected_by="automated",
            detection_method="test_rule"
        )
        
        assert incident.incident_id.startswith("INC-")
        assert incident.severity == IncidentSeverity.HIGH
        assert incident.status == "open"
    
    def test_incident_resolution(self):
        """Test security incident resolution"""
        monitor = SecurityMonitor()
        
        # Create incident manually
        incident = SecurityIncident(
            incident_id="INC-TEST",
            timestamp=datetime.now(),
            severity=IncidentSeverity.MEDIUM,
            title="Test Incident",
            description="Test",
            incident_type="test",
            detected_by="manual",
            detection_method="test"
        )
        
        monitor.incidents[incident.incident_id] = incident
        monitor.incident_count += 1
        
        # Resolve incident
        monitor.resolve_incident(
            incident.incident_id,
            "False alarm - resolved"
        )
        
        assert incident.status == "resolved"
        assert incident.resolved_at is not None
    
    def test_security_metrics(self):
        """Test security metrics reporting"""
        monitor = SecurityMonitor()
        
        monitor.event_count = 100
        monitor.incident_count = 5
        
        metrics = monitor.get_security_metrics()
        
        assert metrics["total_events"] == 100
        assert metrics["total_incidents"] == 5
        assert "open_incidents" in metrics


class TestBusinessContinuityManager:
    """Tests for business continuity and disaster recovery"""
    
    def test_manager_initialization(self):
        """Test BC/DR manager initialization"""
        bc_mgr = BusinessContinuityManager()
        
        assert bc_mgr.rto_target == timedelta(hours=2)
        assert bc_mgr.rpo_target == timedelta(minutes=15)
        assert bc_mgr.backup_count == 0
    
    @pytest.mark.asyncio
    async def test_backup_creation(self):
        """Test encrypted backup creation"""
        bc_mgr = BusinessContinuityManager()
        
        backup = await bc_mgr.create_backup(
            backup_type="full",
            data_source="client_database",
            data={"clients": 1000, "transactions": 5000}
        )
        
        assert backup["backup_id"].startswith("BKP-")
        assert backup["encrypted"] is True
        assert bc_mgr.backup_count == 1
    
    @pytest.mark.asyncio
    async def test_backup_verification(self):
        """Test backup integrity verification"""
        bc_mgr = BusinessContinuityManager()
        
        # Create backup
        backup = await bc_mgr.create_backup(
            backup_type="full",
            data_source="test_data",
            data={"test": "data"}
        )
        
        # Verify backup
        verified = await bc_mgr.verify_backup(backup["backup_id"])
        
        assert verified is True
        assert backup["verified"] is True
    
    @pytest.mark.asyncio
    async def test_backup_restoration(self):
        """Test backup restoration"""
        bc_mgr = BusinessContinuityManager()
        
        original_data = {"important": "data", "value": 12345}
        
        # Create backup
        backup = await bc_mgr.create_backup(
            backup_type="full",
            data_source="test_data",
            data=original_data
        )
        
        # Restore backup
        restored_data = await bc_mgr.restore_backup(backup["backup_id"])
        
        assert restored_data == original_data
    
    @pytest.mark.asyncio
    async def test_disaster_recovery_failover(self):
        """Test disaster recovery failover"""
        bc_mgr = BusinessContinuityManager()
        
        result = await bc_mgr.initiate_failover(
            primary_region="us-east-1",
            dr_region="us-west-2"
        )
        
        assert result["failover_id"].startswith("FO-")
        assert result["status"] == "completed"
        assert result["rto_met"] is True
        assert result["duration_seconds"] < bc_mgr.rto_target.total_seconds()
    
    def test_dr_metrics(self):
        """Test disaster recovery metrics"""
        bc_mgr = BusinessContinuityManager()
        
        bc_mgr.backup_count = 50
        bc_mgr.last_backup = datetime.now() - timedelta(minutes=5)
        
        metrics = bc_mgr.get_dr_metrics()
        
        assert metrics["rto_target_hours"] == 2.0
        assert metrics["rpo_target_minutes"] == 15.0
        assert metrics["total_backups"] == 50
        assert metrics["rpo_met"] is True


class TestSecurityFramework:
    """Tests for integrated security framework"""
    
    def test_framework_initialization(self):
        """Test security framework initialization"""
        framework = SecurityFramework()
        
        assert framework.encryption_manager is not None
        assert framework.auth_manager is not None
        assert framework.authz_manager is not None
        assert framework.security_monitor is not None
        assert framework.bc_manager is not None
    
    @pytest.mark.asyncio
    async def test_secure_operation_authorized(self):
        """Test secure operation - authorized"""
        framework = SecurityFramework()
        
        # Create user and assign role
        user = framework.auth_manager.create_user(
            username="test.advisor",
            email="advisor@test.com",
            password="Password123!"
        )
        
        framework.authz_manager.assign_role(user.user_id, "advisor")
        
        # Execute secure operation
        authorized, message = await framework.secure_operation(
            user_id=user.user_id,
            operation="view_client_data",
            resource="client_portfolio"
        )
        
        assert authorized is True
        assert "authorized" in message.lower()
    
    @pytest.mark.asyncio
    async def test_secure_operation_denied(self):
        """Test secure operation - access denied"""
        framework = SecurityFramework()
        
        # Create user with limited permissions
        user = framework.auth_manager.create_user(
            username="test.client",
            email="client@test.com",
            password="Password123!"
        )
        
        framework.authz_manager.assign_role(user.user_id, "client")
        
        # Try to perform admin operation
        authorized, message = await framework.secure_operation(
            user_id=user.user_id,
            operation="manage_users",
            resource="user_management"
        )
        
        assert authorized is False
        assert "denied" in message.lower()
    
    @pytest.mark.asyncio
    async def test_data_encryption_by_classification(self):
        """Test data encryption based on classification"""
        framework = SecurityFramework()
        
        sensitive_data = "Client SSN: 123-45-6789"
        
        # Encrypt highly confidential data
        encrypted, key_id = await framework.encrypt_sensitive_data(
            sensitive_data,
            SecurityLevel.HIGHLY_CONFIDENTIAL
        )
        
        assert encrypted != sensitive_data.encode('utf-8')
        assert key_id.startswith("KEY-")
    
    def test_comprehensive_security_status(self):
        """Test comprehensive security status reporting"""
        framework = SecurityFramework()
        
        # Create some activity
        user = framework.auth_manager.create_user(
            username="status.test",
            email="status@test.com",
            password="Password123!"
        )
        
        framework.authz_manager.assign_role(user.user_id, "advisor")
        
        # Get status
        status = framework.get_comprehensive_security_status()
        
        assert "encryption" in status
        assert "authentication" in status
        assert "authorization" in status
        assert "monitoring" in status
        assert "business_continuity" in status
        assert "compliance_status" in status
        
        # Check compliance
        assert status["compliance_status"]["owasp_top_10"] == "compliant"
        assert status["compliance_status"]["soc2_type2"] == "compliant"


class TestIntegration:
    """Integration tests for complete security workflows"""
    
    @pytest.mark.asyncio
    async def test_complete_security_workflow(self):
        """Test end-to-end security workflow"""
        framework = SecurityFramework()
        
        # Step 1: Create and authenticate user
        user = framework.auth_manager.create_user(
            username="integration.test",
            email="integration@test.com",
            password="SecurePass123!",
            roles=["advisor"]
        )
        
        framework.authz_manager.assign_role(user.user_id, "advisor")
        
        decision, token = await framework.auth_manager.authenticate(
            username="integration.test",
            password="SecurePass123!"
        )
        
        assert decision == AccessDecision.ALLOW
        assert token is not None
        
        # Step 2: Perform authorized operation
        authorized, _ = await framework.secure_operation(
            user_id=user.user_id,
            operation="create_advice",
            resource="advisory_system"
        )
        
        assert authorized is True
        
        # Step 3: Create encrypted backup
        backup = await framework.bc_manager.create_backup(
            backup_type="full",
            data_source="integration_test",
            data={"test": "complete"}
        )
        
        assert backup["encrypted"] is True
        
        # Step 4: Verify monitoring captured events
        assert framework.security_monitor.event_count > 0
    
    @pytest.mark.asyncio
    async def test_security_incident_handling(self):
        """Test security incident detection and handling"""
        framework = SecurityFramework()
        
        user_id = "USR-INCIDENT-TEST"
        
        # Simulate suspicious activity (multiple failed logins)
        for i in range(6):
            await framework.security_monitor.log_event(
                event_type="login",
                user_id=user_id,
                success=False,
                severity=IncidentSeverity.LOW
            )
        
        # Should have created incident
        open_incidents = framework.security_monitor.get_open_incidents()
        assert len(open_incidents) > 0
    
    @pytest.mark.asyncio
    async def test_performance_targets_met(self):
        """Test all security performance targets are met"""
        framework = SecurityFramework()
        
        # Get status
        status = framework.get_comprehensive_security_status()
        
        # Verify encryption
        assert status["encryption"]["active_keys"] > 0
        
        # Verify BC/DR targets
        bc_status = status["business_continuity"]
        assert bc_status["rto_target_hours"] <= 2.0  # RTO < 2 hours
        assert bc_status["rpo_target_minutes"] <= 15.0  # RPO < 15 minutes
        
        # Verify compliance
        compliance = status["compliance_status"]
        assert compliance["owasp_top_10"] == "compliant"
        assert compliance["soc2_type2"] == "compliant"
        assert compliance["iso_27001"] == "compliant"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
