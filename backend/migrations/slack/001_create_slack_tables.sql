-- ============================================================================
-- Slack Integration - Database Schema
-- ============================================================================

-- Slack Workspaces
CREATE TABLE IF NOT EXISTS slack_workspaces (
    workspace_id VARCHAR(50) PRIMARY KEY,
    team_name VARCHAR(200) NOT NULL,
    team_id VARCHAR(50) NOT NULL,
    
    -- Tokens
    bot_token TEXT NOT NULL,
    app_token TEXT,
    webhook_url TEXT,
    signing_secret VARCHAR(100),
    
    -- Connection
    connected_at TIMESTAMP NOT NULL,
    last_activity_at TIMESTAMP,
    active BOOLEAN DEFAULT TRUE,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_active (active),
    INDEX idx_team_id (team_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Slack Messages
CREATE TABLE IF NOT EXISTS slack_messages (
    message_id VARCHAR(36) PRIMARY KEY,
    channel VARCHAR(100) NOT NULL,
    message_type ENUM(
        'notification', 'alert', 'approval_request',
        'report_ready', 'compliance_issue', 'system_status'
    ) NOT NULL,
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    
    -- Content
    title VARCHAR(500),
    text TEXT NOT NULL,
    blocks JSON,
    attachments JSON,
    
    -- Threading
    thread_ts VARCHAR(50),  -- Thread timestamp
    
    -- Status
    status ENUM('pending', 'sent', 'failed', 'delivered') DEFAULT 'pending',
    created_at TIMESTAMP NOT NULL,
    sent_at TIMESTAMP,
    error_message TEXT,
    
    -- Metadata
    workspace_id VARCHAR(50),
    
    FOREIGN KEY (workspace_id) REFERENCES slack_workspaces(workspace_id),
    INDEX idx_channel (channel),
    INDEX idx_status (status),
    INDEX idx_priority (priority),
    INDEX idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Slack Commands
CREATE TABLE IF NOT EXISTS slack_commands (
    command_id VARCHAR(36) PRIMARY KEY,
    command VARCHAR(50) NOT NULL,
    text TEXT,
    
    -- User
    user_id VARCHAR(50) NOT NULL,
    user_name VARCHAR(200),
    
    -- Channel/Team
    channel_id VARCHAR(50),
    channel_name VARCHAR(200),
    team_id VARCHAR(50),
    
    -- Response
    trigger_id VARCHAR(100),
    response_url TEXT,
    
    -- Processing
    received_at TIMESTAMP NOT NULL,
    processed_at TIMESTAMP,
    
    -- Classification
    classified_intent VARCHAR(50),
    intent_confidence DECIMAL(5, 4),
    
    -- Status
    status ENUM('received', 'processing', 'completed', 'failed') DEFAULT 'received',
    error_message TEXT,
    
    INDEX idx_command (command),
    INDEX idx_user (user_id),
    INDEX idx_received (received_at),
    INDEX idx_intent (classified_intent)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Slack Interactions (button clicks, modal submissions)
CREATE TABLE IF NOT EXISTS slack_interactions (
    interaction_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    interaction_type ENUM('block_actions', 'view_submission', 'shortcut') NOT NULL,
    
    -- Action details
    action_id VARCHAR(100),
    callback_id VARCHAR(100),
    
    -- User
    user_id VARCHAR(50) NOT NULL,
    user_name VARCHAR(200),
    
    -- Message
    message_ts VARCHAR(50),
    channel_id VARCHAR(50),
    
    -- Payload
    payload JSON NOT NULL,
    
    -- Processing
    received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    response_sent BOOLEAN DEFAULT FALSE,
    
    INDEX idx_action_id (action_id),
    INDEX idx_user (user_id),
    INDEX idx_received (received_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Slack Channels Configuration
CREATE TABLE IF NOT EXISTS slack_channels (
    channel_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    channel_name VARCHAR(100) NOT NULL UNIQUE,
    channel_slack_id VARCHAR(50),
    
    -- Purpose
    purpose ENUM(
        'operations', 'alerts', 'compliance', 'reports',
        'support', 'general', 'approvals'
    ) NOT NULL,
    
    -- Notification settings
    enabled BOOLEAN DEFAULT TRUE,
    priority_filter ENUM('all', 'medium_and_above', 'high_and_above', 'critical_only') DEFAULT 'all',
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_purpose (purpose),
    INDEX idx_enabled (enabled)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Message Templates
CREATE TABLE IF NOT EXISTS slack_message_templates (
    template_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_name VARCHAR(200) NOT NULL UNIQUE,
    message_type ENUM(
        'notification', 'alert', 'approval_request',
        'report_ready', 'compliance_issue', 'system_status'
    ) NOT NULL,
    
    -- Template
    blocks_template JSON NOT NULL,
    default_priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    
    -- Usage
    active BOOLEAN DEFAULT TRUE,
    usage_count INT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_message_type (message_type),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Command History Analytics
CREATE TABLE IF NOT EXISTS slack_command_analytics (
    analytics_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    command VARCHAR(50) NOT NULL,
    intent VARCHAR(50),
    
    -- Metrics
    total_uses INT DEFAULT 0,
    successful_uses INT DEFAULT 0,
    failed_uses INT DEFAULT 0,
    avg_response_time_ms INT,
    
    -- Period
    analytics_date DATE NOT NULL,
    
    UNIQUE KEY unique_command_date (command, intent, analytics_date),
    INDEX idx_date (analytics_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Views
-- ============================================================================

-- Recent Messages View
CREATE OR REPLACE VIEW v_recent_slack_messages AS
SELECT 
    m.message_id,
    m.channel,
    m.message_type,
    m.priority,
    m.title,
    m.status,
    m.created_at,
    m.sent_at
FROM slack_messages m
WHERE m.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY m.created_at DESC;

-- Command Usage Stats
CREATE OR REPLACE VIEW v_slack_command_stats AS
SELECT 
    command,
    classified_intent,
    COUNT(*) as total_commands,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(TIMESTAMPDIFF(SECOND, received_at, processed_at)) as avg_processing_time
FROM slack_commands
WHERE received_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY command, classified_intent;

-- ============================================================================
-- Stored Procedures
-- ============================================================================

DELIMITER //

CREATE PROCEDURE sp_get_pending_messages()
BEGIN
    SELECT *
    FROM slack_messages
    WHERE status = 'pending'
    ORDER BY priority DESC, created_at
    LIMIT 100;
END//

CREATE PROCEDURE sp_update_command_analytics(
    IN p_date DATE
)
BEGIN
    INSERT INTO slack_command_analytics (
        command, intent, analytics_date,
        total_uses, successful_uses, failed_uses
    )
    SELECT 
        command,
        classified_intent,
        DATE(received_at),
        COUNT(*),
        SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END),
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END)
    FROM slack_commands
    WHERE DATE(received_at) = p_date
    GROUP BY command, classified_intent, DATE(received_at)
    ON DUPLICATE KEY UPDATE
        total_uses = VALUES(total_uses),
        successful_uses = VALUES(successful_uses),
        failed_uses = VALUES(failed_uses);
END//

DELIMITER ;

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Default Channels
INSERT INTO slack_channels (channel_name, purpose, priority_filter) VALUES
('#operations', 'operations', 'all'),
('#alerts', 'alerts', 'high_and_above'),
('#compliance', 'compliance', 'medium_and_above'),
('#reports', 'reports', 'all'),
('#approvals', 'approvals', 'all'),
('#support', 'support', 'all');

-- Default Message Templates
INSERT INTO slack_message_templates (template_name, message_type, blocks_template) VALUES
('simple_notification', 'notification', '[]'),
('critical_alert', 'alert', '[]'),
('approval_request', 'approval_request', '[]');
