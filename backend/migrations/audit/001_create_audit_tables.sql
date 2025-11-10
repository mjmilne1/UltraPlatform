-- ============================================================================
-- Audit Trail & Maker-Checker System - Database Schema
-- ============================================================================

-- Audit Trail (Immutable)
CREATE TABLE IF NOT EXISTS audit_trail (
    audit_id VARCHAR(36) PRIMARY KEY,
    timestamp TIMESTAMP(6) NOT NULL,
    
    -- Who
    user_id VARCHAR(36) NOT NULL,
    user_email VARCHAR(200) NOT NULL,
    user_role VARCHAR(50) NOT NULL,
    
    -- What
    entity_type ENUM('client', 'account', 'transaction', 'journal_entry', 'user', 'portfolio', 'settings', 'fee_structure') NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    change_type ENUM('create', 'update', 'delete', 'approve', 'reject', 'cancel') NOT NULL,
    
    -- Changes
    old_value JSON,
    new_value JSON,
    
    -- Context
    ip_address VARCHAR(45),
    user_agent TEXT,
    session_id VARCHAR(100),
    reason TEXT,
    
    -- Risk
    risk_score DECIMAL(5, 4),
    risk_level ENUM('low', 'medium', 'high', 'critical'),
    
    -- Integrity
    checksum VARCHAR(64) NOT NULL,
    previous_audit_id VARCHAR(36),
    
    INDEX idx_timestamp (timestamp),
    INDEX idx_user (user_id),
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_risk_level (risk_level),
    INDEX idx_change_type (change_type),
    INDEX idx_checksum (checksum),
    FOREIGN KEY (previous_audit_id) REFERENCES audit_trail(audit_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Approval Requests (Maker-Checker)
CREATE TABLE IF NOT EXISTS approval_requests (
    request_id VARCHAR(36) PRIMARY KEY,
    created_at TIMESTAMP NOT NULL,
    
    -- Maker
    maker_id VARCHAR(36) NOT NULL,
    maker_email VARCHAR(200) NOT NULL,
    
    -- What needs approval
    entity_type ENUM('client', 'account', 'transaction', 'journal_entry', 'user', 'portfolio', 'settings', 'fee_structure') NOT NULL,
    entity_id VARCHAR(100) NOT NULL,
    change_type ENUM('create', 'update', 'delete', 'approve', 'reject', 'cancel') NOT NULL,
    proposed_changes JSON NOT NULL,
    
    -- Approval status
    status ENUM('pending', 'approved', 'rejected', 'cancelled', 'expired') DEFAULT 'pending',
    requires_approval_count INT DEFAULT 1,
    approved_by JSON,  -- Array of checker IDs
    
    -- Rejection
    rejected_by VARCHAR(36),
    rejection_timestamp TIMESTAMP,
    rejection_reason TEXT,
    
    -- Approval
    approval_timestamp TIMESTAMP,
    
    -- Risk
    risk_score DECIMAL(5, 4),
    risk_level ENUM('low', 'medium', 'high', 'critical'),
    
    -- Expiry
    expires_at TIMESTAMP,
    
    -- Context
    reason TEXT,
    business_justification TEXT,
    
    INDEX idx_status (status),
    INDEX idx_maker (maker_id),
    INDEX idx_entity (entity_type, entity_id),
    INDEX idx_created (created_at),
    INDEX idx_expires (expires_at),
    INDEX idx_risk (risk_level)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Approval Actions (Audit of approval workflow)
CREATE TABLE IF NOT EXISTS approval_actions (
    action_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    action_type ENUM('approved', 'rejected', 'cancelled', 'expired') NOT NULL,
    action_by VARCHAR(36) NOT NULL,
    action_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    comment TEXT,
    
    FOREIGN KEY (request_id) REFERENCES approval_requests(request_id),
    INDEX idx_request (request_id),
    INDEX idx_timestamp (action_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Compliance Violations
CREATE TABLE IF NOT EXISTS compliance_violations (
    violation_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    violation_type VARCHAR(100) NOT NULL,
    severity ENUM('low', 'medium', 'high', 'critical') NOT NULL,
    
    -- Related entities
    user_id VARCHAR(36),
    entity_type VARCHAR(50),
    entity_id VARCHAR(100),
    
    -- Details
    description TEXT NOT NULL,
    evidence JSON,
    
    -- Resolution
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolved_by VARCHAR(36),
    resolution_notes TEXT,
    
    INDEX idx_detected (detected_at),
    INDEX idx_severity (severity),
    INDEX idx_resolved (resolved),
    INDEX idx_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Segregation of Duties Rules
CREATE TABLE IF NOT EXISTS sod_rules (
    rule_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    rule_name VARCHAR(200) NOT NULL,
    description TEXT,
    
    -- Conflicting permissions
    permission_a VARCHAR(100) NOT NULL,
    permission_b VARCHAR(100) NOT NULL,
    
    -- Enforcement
    enabled BOOLEAN DEFAULT TRUE,
    enforcement_level ENUM('warning', 'block') DEFAULT 'block',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    UNIQUE KEY unique_permissions (permission_a, permission_b),
    INDEX idx_enabled (enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- User Risk Scores (ML-based)
CREATE TABLE IF NOT EXISTS user_risk_scores (
    score_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Scores
    overall_risk_score DECIMAL(5, 4) NOT NULL,
    approval_pattern_score DECIMAL(5, 4),
    violation_history_score DECIMAL(5, 4),
    access_pattern_score DECIMAL(5, 4),
    
    -- Metrics
    total_approvals INT DEFAULT 0,
    approval_rate DECIMAL(5, 4),
    avg_time_to_approve_seconds INT,
    violations_count INT DEFAULT 0,
    
    -- Risk factors
    risk_factors JSON,
    
    INDEX idx_user (user_id),
    INDEX idx_calculated (calculated_at),
    INDEX idx_risk_score (overall_risk_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Views
-- ============================================================================

-- Pending Approvals View
CREATE OR REPLACE VIEW v_pending_approvals AS
SELECT 
    r.request_id,
    r.created_at,
    r.maker_email,
    r.entity_type,
    r.entity_id,
    r.change_type,
    r.risk_level,
    r.requires_approval_count,
    JSON_LENGTH(r.approved_by) as current_approvals,
    DATEDIFF(r.expires_at, NOW()) as days_until_expiry
FROM approval_requests r
WHERE r.status = 'pending'
    AND r.expires_at > NOW()
ORDER BY r.risk_level DESC, r.created_at;

-- High Risk Changes View
CREATE OR REPLACE VIEW v_high_risk_changes AS
SELECT 
    audit_id,
    timestamp,
    user_email,
    entity_type,
    entity_id,
    change_type,
    risk_score,
    risk_level,
    reason
FROM audit_trail
WHERE risk_level IN ('high', 'critical')
ORDER BY timestamp DESC;

-- Compliance Dashboard View
CREATE OR REPLACE VIEW v_compliance_dashboard AS
SELECT 
    DATE(detected_at) as date,
    severity,
    COUNT(*) as violation_count,
    SUM(CASE WHEN resolved THEN 1 ELSE 0 END) as resolved_count
FROM compliance_violations
WHERE detected_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY DATE(detected_at), severity
ORDER BY date DESC, severity;

-- ============================================================================
-- Stored Procedures
-- ============================================================================

DELIMITER //

CREATE PROCEDURE sp_verify_audit_integrity()
BEGIN
    DECLARE chain_broken BOOLEAN DEFAULT FALSE;
    DECLARE last_id VARCHAR(36);
    
    SELECT 
        CASE 
            WHEN COUNT(*) = COUNT(DISTINCT checksum) THEN FALSE
            ELSE TRUE
        END INTO chain_broken
    FROM audit_trail;
    
    SELECT chain_broken as integrity_compromised;
END//

CREATE PROCEDURE sp_expire_old_approvals()
BEGIN
    UPDATE approval_requests
    SET status = 'expired'
    WHERE status = 'pending'
        AND expires_at < NOW();
    
    SELECT ROW_COUNT() as expired_count;
END//

CREATE PROCEDURE sp_get_user_compliance_report(
    IN p_user_id VARCHAR(36),
    IN p_days INT
)
BEGIN
    SELECT 
        DATE(timestamp) as date,
        COUNT(*) as total_changes,
        SUM(CASE WHEN risk_level = 'high' THEN 1 ELSE 0 END) as high_risk_changes,
        SUM(CASE WHEN risk_level = 'critical' THEN 1 ELSE 0 END) as critical_risk_changes
    FROM audit_trail
    WHERE user_id = p_user_id
        AND timestamp >= DATE_SUB(CURDATE(), INTERVAL p_days DAY)
    GROUP BY DATE(timestamp)
    ORDER BY date DESC;
END//

DELIMITER ;

-- ============================================================================
-- Triggers
-- ============================================================================

-- Trigger: Log approval actions
DELIMITER //

CREATE TRIGGER tr_log_approval_action
AFTER UPDATE ON approval_requests
FOR EACH ROW
BEGIN
    IF NEW.status != OLD.status THEN
        INSERT INTO approval_actions (
            request_id,
            action_type,
            action_by,
            action_timestamp
        ) VALUES (
            NEW.request_id,
            NEW.status,
            COALESCE(NEW.rejected_by, 'system'),
            CASE 
                WHEN NEW.status = 'approved' THEN NEW.approval_timestamp
                WHEN NEW.status = 'rejected' THEN NEW.rejection_timestamp
                ELSE NOW()
            END
        );
    END IF;
END//

DELIMITER ;

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Default SOD Rules
INSERT INTO sod_rules (rule_name, description, permission_a, permission_b, enforcement_level) VALUES
('Maker-Checker: Client Creation', 'Same user cannot create and approve clients', 'client.create', 'client.approve', 'block'),
('Maker-Checker: Transactions', 'Same user cannot create and approve transactions', 'transaction.create', 'transaction.approve', 'block'),
('Maker-Checker: Journal Entries', 'Same user cannot create and approve journal entries', 'journal.create', 'journal.approve', 'block'),
('SOD: Trading vs Settlement', 'Same user cannot execute and settle trades', 'trade.execute', 'trade.settle', 'block'),
('SOD: Custody vs Trading', 'Custody staff cannot execute trades', 'custody.access', 'trade.execute', 'block');

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

-- Composite indexes for common queries
CREATE INDEX idx_audit_user_entity ON audit_trail(user_id, entity_type, entity_id);
CREATE INDEX idx_audit_entity_time ON audit_trail(entity_type, entity_id, timestamp);
CREATE INDEX idx_approval_maker_status ON approval_requests(maker_id, status);
CREATE INDEX idx_violation_user_resolved ON compliance_violations(user_id, resolved, detected_at);
