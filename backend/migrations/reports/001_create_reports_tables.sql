-- ============================================================================
-- Reporting Engine - Database Schema
-- ============================================================================

-- Report Definitions
CREATE TABLE IF NOT EXISTS report_definitions (
    report_id VARCHAR(50) PRIMARY KEY,
    report_name VARCHAR(200) NOT NULL,
    report_type ENUM(
        'monthly_statement', 'quarterly_statement', 'annual_statement',
        'performance_summary', 'portfolio_valuation', 'transaction_history',
        'tax_summary', 'aum_report', 'fee_revenue_report', 'client_activity',
        'trading_summary', 'audit_trail_report', 'compliance_summary',
        'risk_report', 'balance_sheet', 'income_statement', 'cash_flow',
        'trial_balance'
    ) NOT NULL,
    description TEXT,
    
    -- Configuration
    required_data_sources JSON,
    optional_parameters JSON,
    default_format ENUM('pdf', 'excel', 'csv', 'html', 'json') DEFAULT 'pdf',
    supported_formats JSON,
    
    -- Scheduling
    can_be_scheduled BOOLEAN DEFAULT TRUE,
    default_schedule VARCHAR(100),
    
    -- AI Features
    ai_insights_enabled BOOLEAN DEFAULT TRUE,
    ml_recommendations_enabled BOOLEAN DEFAULT TRUE,
    
    -- Access
    requires_approval BOOLEAN DEFAULT FALSE,
    allowed_roles JSON,
    
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_report_type (report_type),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Report Requests
CREATE TABLE IF NOT EXISTS report_requests (
    request_id VARCHAR(36) PRIMARY KEY,
    report_type ENUM(
        'monthly_statement', 'quarterly_statement', 'annual_statement',
        'performance_summary', 'portfolio_valuation', 'transaction_history',
        'tax_summary', 'aum_report', 'fee_revenue_report', 'client_activity',
        'trading_summary', 'audit_trail_report', 'compliance_summary',
        'risk_report', 'balance_sheet', 'income_statement', 'cash_flow',
        'trial_balance'
    ) NOT NULL,
    requested_by VARCHAR(36) NOT NULL,
    requested_at TIMESTAMP NOT NULL,
    
    -- Parameters
    start_date DATE,
    end_date DATE,
    client_id VARCHAR(36),
    account_id VARCHAR(36),
    parameters JSON,
    
    -- Output
    format ENUM('pdf', 'excel', 'csv', 'html', 'json') DEFAULT 'pdf',
    
    -- Status
    status ENUM('pending', 'generating', 'completed', 'failed', 'delivered') DEFAULT 'pending',
    generated_at TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Result
    output_path TEXT,
    file_size_bytes BIGINT,
    error_message TEXT,
    
    INDEX idx_requested_by (requested_by),
    INDEX idx_status (status),
    INDEX idx_report_type (report_type),
    INDEX idx_client (client_id),
    INDEX idx_requested_at (requested_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Scheduled Reports
CREATE TABLE IF NOT EXISTS scheduled_reports (
    schedule_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    report_type VARCHAR(50) NOT NULL,
    schedule_name VARCHAR(200) NOT NULL,
    
    -- Schedule
    schedule_pattern VARCHAR(100) NOT NULL,  -- Cron pattern
    timezone VARCHAR(50) DEFAULT 'Australia/Sydney',
    
    -- Parameters
    client_id VARCHAR(36),
    account_id VARCHAR(36),
    parameters JSON,
    format ENUM('pdf', 'excel', 'csv', 'html', 'json') DEFAULT 'pdf',
    
    -- Recipients
    recipients JSON,  -- Array of email addresses
    
    -- Status
    enabled BOOLEAN DEFAULT TRUE,
    last_run_at TIMESTAMP,
    next_run_at TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(36),
    
    INDEX idx_enabled (enabled),
    INDEX idx_next_run (next_run_at),
    INDEX idx_client (client_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Report Templates
CREATE TABLE IF NOT EXISTS report_templates (
    template_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    template_name VARCHAR(200) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    
    -- Template content
    template_version VARCHAR(20) DEFAULT '1.0',
    template_html TEXT,
    template_config JSON,
    
    -- Styling
    css_styles TEXT,
    header_html TEXT,
    footer_html TEXT,
    
    -- Status
    is_default BOOLEAN DEFAULT FALSE,
    active BOOLEAN DEFAULT TRUE,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    
    INDEX idx_report_type (report_type),
    INDEX idx_active (active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Report Insights (AI-generated)
CREATE TABLE IF NOT EXISTS report_insights (
    insight_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    
    -- Insight
    insight_type VARCHAR(50) NOT NULL,
    title VARCHAR(200) NOT NULL,
    message TEXT NOT NULL,
    
    -- Classification
    sentiment ENUM('positive', 'negative', 'neutral', 'warning') DEFAULT 'neutral',
    priority ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
    
    -- AI metadata
    confidence_score DECIMAL(5, 4),
    generated_by VARCHAR(50) DEFAULT 'AI',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (request_id) REFERENCES report_requests(request_id),
    INDEX idx_request (request_id),
    INDEX idx_type (insight_type),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Performance Attribution (ML results)
CREATE TABLE IF NOT EXISTS performance_attribution (
    attribution_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    client_id VARCHAR(36) NOT NULL,
    period_start DATE NOT NULL,
    period_end DATE NOT NULL,
    
    -- Attribution effects
    asset_allocation_effect DECIMAL(10, 6),
    security_selection_effect DECIMAL(10, 6),
    timing_effect DECIMAL(10, 6),
    currency_effect DECIMAL(10, 6),
    interaction_effect DECIMAL(10, 6),
    
    -- Totals
    total_attribution DECIMAL(10, 6),
    unexplained_return DECIMAL(10, 6),
    
    -- Metadata
    calculation_method VARCHAR(50) DEFAULT 'ML',
    confidence_score DECIMAL(5, 4),
    
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    INDEX idx_client (client_id),
    INDEX idx_period (period_start, period_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Report Delivery Log
CREATE TABLE IF NOT EXISTS report_delivery_log (
    delivery_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    request_id VARCHAR(36) NOT NULL,
    
    -- Delivery
    delivery_method ENUM('email', 'download', 'api', 'scheduled') NOT NULL,
    recipient VARCHAR(200),
    delivered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    delivery_status ENUM('sent', 'delivered', 'failed', 'bounced') DEFAULT 'sent',
    error_message TEXT,
    
    FOREIGN KEY (request_id) REFERENCES report_requests(request_id),
    INDEX idx_request (request_id),
    INDEX idx_delivered (delivered_at),
    INDEX idx_status (delivery_status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Views
-- ============================================================================

-- Recent Reports View
CREATE OR REPLACE VIEW v_recent_reports AS
SELECT 
    r.request_id,
    r.report_type,
    r.requested_by,
    r.requested_at,
    r.client_id,
    r.status,
    r.format,
    r.output_path,
    TIMESTAMPDIFF(SECOND, r.generated_at, r.completed_at) as generation_time_seconds
FROM report_requests r
WHERE r.requested_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
ORDER BY r.requested_at DESC;

-- Scheduled Reports Due View
CREATE OR REPLACE VIEW v_scheduled_reports_due AS
SELECT 
    s.schedule_id,
    s.report_type,
    s.schedule_name,
    s.client_id,
    s.next_run_at,
    s.format,
    s.recipients
FROM scheduled_reports s
WHERE s.enabled = TRUE
    AND s.next_run_at <= NOW()
ORDER BY s.next_run_at;

-- Report Performance Stats View
CREATE OR REPLACE VIEW v_report_performance AS
SELECT 
    report_type,
    COUNT(*) as total_generated,
    AVG(TIMESTAMPDIFF(SECOND, generated_at, completed_at)) as avg_generation_time,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed,
    AVG(file_size_bytes) as avg_file_size
FROM report_requests
WHERE requested_at >= DATE_SUB(CURDATE(), INTERVAL 90 DAY)
GROUP BY report_type;

-- ============================================================================
-- Stored Procedures
-- ============================================================================

DELIMITER //

CREATE PROCEDURE sp_get_client_reports(
    IN p_client_id VARCHAR(36),
    IN p_days INT
)
BEGIN
    SELECT 
        request_id,
        report_type,
        requested_at,
        status,
        format,
        output_path
    FROM report_requests
    WHERE client_id = p_client_id
        AND requested_at >= DATE_SUB(CURDATE(), INTERVAL p_days DAY)
    ORDER BY requested_at DESC;
END//

CREATE PROCEDURE sp_update_scheduled_report_next_run(
    IN p_schedule_id BIGINT,
    IN p_next_run TIMESTAMP
)
BEGIN
    UPDATE scheduled_reports
    SET last_run_at = NOW(),
        next_run_at = p_next_run
    WHERE schedule_id = p_schedule_id;
END//

DELIMITER ;

-- ============================================================================
-- Initial Data
-- ============================================================================

-- Standard Report Definitions
INSERT INTO report_definitions (report_id, report_name, report_type, description, can_be_scheduled, ai_insights_enabled) VALUES
('rpt_monthly_stmt', 'Monthly Statement', 'monthly_statement', 'Professional monthly client statement with performance and holdings', TRUE, TRUE),
('rpt_performance', 'Performance Report', 'performance_summary', 'Detailed performance analysis with ML attribution', TRUE, TRUE),
('rpt_tax_summary', 'Tax Summary', 'tax_summary', 'Australian tax summary for financial year', TRUE, FALSE),
('rpt_portfolio_val', 'Portfolio Valuation', 'portfolio_valuation', 'Complete portfolio valuation with market prices', TRUE, TRUE),
('rpt_transaction', 'Transaction History', 'transaction_history', 'Detailed transaction history report', TRUE, FALSE),
('rpt_aum', 'AUM Report', 'aum_report', 'Assets under management summary', TRUE, FALSE),
('rpt_fee_revenue', 'Fee Revenue Report', 'fee_revenue_report', 'Management fee revenue analysis', TRUE, FALSE),
('rpt_audit_trail', 'Audit Trail Report', 'audit_trail_report', 'Complete audit trail for compliance', FALSE, FALSE),
('rpt_balance_sheet', 'Balance Sheet', 'balance_sheet', 'Statement of financial position', TRUE, FALSE),
('rpt_income_stmt', 'Income Statement', 'income_statement', 'Statement of profit and loss', TRUE, FALSE);

-- ============================================================================
-- Indexes for Performance
-- ============================================================================

CREATE INDEX idx_report_client_date ON report_requests(client_id, requested_at);
CREATE INDEX idx_report_type_status ON report_requests(report_type, status);
CREATE INDEX idx_scheduled_client_enabled ON scheduled_reports(client_id, enabled);
