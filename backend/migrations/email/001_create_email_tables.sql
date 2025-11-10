-- ============================================================================
-- Email Integration - Database Schema
-- ============================================================================

-- Email Messages
CREATE TABLE IF NOT EXISTS email_messages (
    message_id VARCHAR(36) PRIMARY KEY,
    email_type ENUM(
        'transactional', 'notification', 'report',
        'marketing', 'alert', 'approval'
    ) NOT NULL,
    priority ENUM('low', 'medium', 'high', 'urgent') DEFAULT 'medium',
    
    -- Recipients
    to_addresses TEXT NOT NULL,
    cc_addresses TEXT,
    bcc_addresses TEXT,
    
    -- Content
    subject VARCHAR(500) NOT NULL,
    body_html TEXT NOT NULL,
    body_text TEXT,
    
    -- Sender
    from_email VARCHAR(200) DEFAULT 'noreply@turingwealth.com',
    from_name VARCHAR(200) DEFAULT 'TuringWealth',
    reply_to VARCHAR(200),
    
    -- Template
    template_id VARCHAR(50),
    template_data JSON,
    
    -- Tracking
    track_opens BOOLEAN DEFAULT TRUE,
    track_clicks BOOLEAN DEFAULT TRUE,
    
    -- Status
    status ENUM(
        'pending', 'sending', 'sent', 'delivered',
        'opened', 'clicked', 'bounced', 'failed'
    ) DEFAULT 'pending',
    
    -- Provider
    provider ENUM('sendgrid', 'aws_ses', 'smtp'),
    provider_message_id VARCHAR(200),
    
    -- Timestamps
    created_at TIMESTAMP NOT NULL,
    sent_at TIMESTAMP,
    delivered_at TIMESTAMP,
    opened_at TIMESTAMP,
    
    -- Error
    error_message TEXT,
    
    INDEX idx_status (status),
    INDEX idx_email_type (email_type),
    INDEX idx_priority (priority),
    INDEX idx_created (created_at),
    INDEX idx_template (template_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Templates
CREATE TABLE IF NOT EXISTS email_templates (
    template_id VARCHAR(50) PRIMARY KEY,
    template_name VARCHAR(200) NOT NULL UNIQUE,
    email_type ENUM(
        'transactional', 'notification', 'report',
        'marketing', 'alert', 'approval'
    ) NOT NULL,
    
    -- Template content
    subject_template TEXT NOT NULL,
    html_template TEXT NOT NULL,
    text_template TEXT,
    
    -- Variables
    required_variables JSON,
    
    -- Versioning
    version VARCHAR(20) DEFAULT '1.0',
    
    -- Usage
    active BOOLEAN DEFAULT TRUE,
    usage_count INT DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_email_type (email_type),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Clicks
CREATE TABLE IF NOT EXISTS email_clicks (
    click_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    url TEXT NOT NULL,
    clicked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address VARCHAR(45),
    user_agent TEXT,
    
    FOREIGN KEY (message_id) REFERENCES email_messages(message_id),
    INDEX idx_message (message_id),
    INDEX idx_clicked (clicked_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Bounces
CREATE TABLE IF NOT EXISTS email_bounces (
    bounce_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    email_address VARCHAR(200) NOT NULL,
    bounce_type ENUM('hard', 'soft', 'complaint') NOT NULL,
    bounce_reason TEXT,
    bounced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (message_id) REFERENCES email_messages(message_id),
    INDEX idx_message (message_id),
    INDEX idx_email (email_address),
    INDEX idx_bounced (bounced_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Unsubscribes
CREATE TABLE IF NOT EXISTS email_unsubscribes (
    unsubscribe_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email_address VARCHAR(200) NOT NULL UNIQUE,
    reason VARCHAR(500),
    unsubscribed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_email (email_address),
    INDEX idx_unsubscribed (unsubscribed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Campaigns
CREATE TABLE IF NOT EXISTS email_campaigns (
    campaign_id VARCHAR(50) PRIMARY KEY,
    campaign_name VARCHAR(200) NOT NULL,
    template_id VARCHAR(50) NOT NULL,
    
    -- Targeting
    target_segment VARCHAR(100),
    target_count INT,
    
    -- Schedule
    scheduled_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Status
    status ENUM('draft', 'scheduled', 'sending', 'completed', 'cancelled') DEFAULT 'draft',
    
    -- Stats
    sent_count INT DEFAULT 0,
    delivered_count INT DEFAULT 0,
    opened_count INT DEFAULT 0,
    clicked_count INT DEFAULT 0,
    bounced_count INT DEFAULT 0,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    
    FOREIGN KEY (template_id) REFERENCES email_templates(template_id),
    INDEX idx_status (status),
    INDEX idx_scheduled (scheduled_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Recipients (for personalization)
CREATE TABLE IF NOT EXISTS email_recipients (
    recipient_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    email_address VARCHAR(200) NOT NULL UNIQUE,
    
    -- Profile
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    timezone VARCHAR(50) DEFAULT 'Australia/Sydney',
    
    -- Preferences
    preferred_send_time TIME DEFAULT '09:00:00',
    email_frequency VARCHAR(50) DEFAULT 'all',
    
    -- Engagement
    total_sent INT DEFAULT 0,
    total_opened INT DEFAULT 0,
    total_clicked INT DEFAULT 0,
    last_opened_at TIMESTAMP,
    engagement_score DECIMAL(5, 4) DEFAULT 0.5,
    
    -- Status
    active BOOLEAN DEFAULT TRUE,
    subscribed BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_email (email_address),
    INDEX idx_active (active),
    INDEX idx_engagement (engagement_score)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Email Analytics
CREATE TABLE IF NOT EXISTS email_analytics (
    analytics_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    analytics_date DATE NOT NULL,
    email_type VARCHAR(50),
    
    -- Metrics
    total_sent INT DEFAULT 0,
    total_delivered INT DEFAULT 0,
    total_opened INT DEFAULT 0,
    total_clicked INT DEFAULT 0,
    total_bounced INT DEFAULT 0,
    total_failed INT DEFAULT 0,
    
    -- Rates
    delivery_rate DECIMAL(5, 4),
    open_rate DECIMAL(5, 4),
    click_rate DECIMAL(5, 4),
    bounce_rate DECIMAL(5, 4),
    
    UNIQUE KEY unique_date_type (analytics_date, email_type),
    INDEX idx_date (analytics_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Views
-- ============================================================================

-- Recent Emails View
CREATE OR REPLACE VIEW v_recent_emails AS
SELECT 
    m.message_id,
    m.email_type,
    m.priority,
    m.to_addresses,
    m.subject,
    m.status,
    m.created_at,
    m.sent_at,
    m.delivered_at
FROM email_messages m
WHERE m.created_at >= DATE_SUB(NOW(), INTERVAL 7 DAY)
ORDER BY m.created_at DESC;

-- Email Performance Stats
CREATE OR REPLACE VIEW v_email_performance AS
SELECT 
    email_type,
    COUNT(*) as total_sent,
    SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as delivered,
    SUM(CASE WHEN status = 'opened' THEN 1 ELSE 0 END) as opened,
    SUM(CASE WHEN status = 'clicked' THEN 1 ELSE 0 END) as clicked,
    SUM(CASE WHEN status = 'bounced' THEN 1 ELSE 0 END) as bounced,
    ROUND(SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as delivery_rate,
    ROUND(SUM(CASE WHEN status = 'opened' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as open_rate
FROM email_messages
WHERE created_at >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)
GROUP BY email_type;

-- Top Engaged Recipients
CREATE OR REPLACE VIEW v_top_engaged_recipients AS
SELECT 
    email_address,
    first_name,
    last_name,
    engagement_score,
    total_sent,
    total_opened,
    total_clicked,
    ROUND(total_opened * 100.0 / NULLIF(total_sent, 0), 2) as open_rate,
    last_opened_at
FROM email_recipients
WHERE active = TRUE AND total_sent > 0
ORDER BY engagement_score DESC
LIMIT 100;

-- ============================================================================
-- Stored Procedures
-- ============================================================================

DELIMITER //

CREATE PROCEDURE sp_update_email_analytics(
    IN p_date DATE
)
BEGIN
    INSERT INTO email_analytics (
        analytics_date, email_type,
        total_sent, total_delivered, total_opened,
        total_clicked, total_bounced, total_failed,
        delivery_rate, open_rate, click_rate, bounce_rate
    )
    SELECT 
        DATE(created_at) as analytics_date,
        email_type,
        COUNT(*) as total_sent,
        SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) as total_delivered,
        SUM(CASE WHEN status = 'opened' THEN 1 ELSE 0 END) as total_opened,
        SUM(CASE WHEN status = 'clicked' THEN 1 ELSE 0 END) as total_clicked,
        SUM(CASE WHEN status = 'bounced' THEN 1 ELSE 0 END) as total_bounced,
        SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as total_failed,
        ROUND(SUM(CASE WHEN status = 'delivered' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as delivery_rate,
        ROUND(SUM(CASE WHEN status = 'opened' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as open_rate,
        ROUND(SUM(CASE WHEN status = 'clicked' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as click_rate,
        ROUND(SUM(CASE WHEN status = 'bounced' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 4) as bounce_rate
    FROM email_messages
    WHERE DATE(created_at) = p_date
    GROUP BY DATE(created_at), email_type
    ON DUPLICATE KEY UPDATE
        total_sent = VALUES(total_sent),
        total_delivered = VALUES(total_delivered),
        total_opened = VALUES(total_opened),
        total_clicked = VALUES(total_clicked),
        total_bounced = VALUES(total_bounced),
        total_failed = VALUES(total_failed),
        delivery_rate = VALUES(delivery_rate),
        open_rate = VALUES(open_rate),
        click_rate = VALUES(click_rate),
        bounce_rate = VALUES(bounce_rate);
END//

CREATE PROCEDURE sp_update_recipient_engagement(
    IN p_email VARCHAR(200)
)
BEGIN
    UPDATE email_recipients
    SET 
        total_sent = (
            SELECT COUNT(*)
            FROM email_messages
            WHERE to_addresses LIKE CONCAT('%', p_email, '%')
        ),
        total_opened = (
            SELECT COUNT(*)
            FROM email_messages
            WHERE to_addresses LIKE CONCAT('%', p_email, '%')
                AND status IN ('opened', 'clicked')
        ),
        total_clicked = (
            SELECT COUNT(*)
            FROM email_messages
            WHERE to_addresses LIKE CONCAT('%', p_email, '%')
                AND status = 'clicked'
        ),
        engagement_score = (
            SELECT ROUND(
                (SUM(CASE WHEN status IN ('opened', 'clicked') THEN 1 ELSE 0 END) * 1.0 / COUNT(*)),
                4
            )
            FROM email_messages
            WHERE to_addresses LIKE CONCAT('%', p_email, '%')
        )
    WHERE email_address = p_email;
END//

DELIMITER ;

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Default Email Templates
INSERT INTO email_templates (template_id, template_name, email_type, subject_template, html_template, text_template) VALUES
('welcome', 'Welcome Email', 'transactional', 'Welcome to TuringWealth, {{ client_name }}!', '<h1>Welcome!</h1>', 'Welcome!'),
('report_ready', 'Report Ready', 'report', '{{ report_name }} is Ready', '<h1>Your report is ready</h1>', 'Your report is ready'),
('approval_request', 'Approval Request', 'approval', 'Approval Required: {{ request_type }}', '<h1>Approval Required</h1>', 'Approval Required'),
('alert', 'Alert Notification', 'alert', '?? {{ alert_title }}', '<h1>Alert</h1>', 'Alert'),
('monthly_statement', 'Monthly Statement', 'report', 'Your {{ month }} Statement - {{ client_name }}', '<h1>Monthly Statement</h1>', 'Monthly Statement');

-- ============================================================================
-- Triggers
-- ============================================================================

DELIMITER //

CREATE TRIGGER tr_update_template_usage
AFTER INSERT ON email_messages
FOR EACH ROW
BEGIN
    IF NEW.template_id IS NOT NULL THEN
        UPDATE email_templates
        SET usage_count = usage_count + 1
        WHERE template_id = NEW.template_id;
    END IF;
END//

DELIMITER ;
