-- ============================================================================
-- Batch Job Scheduler - Database Schema
-- ============================================================================

-- Scheduled Jobs (Job Definitions)
CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id VARCHAR(50) PRIMARY KEY,
    job_name VARCHAR(200) NOT NULL,
    job_type VARCHAR(50) NOT NULL,
    priority INT NOT NULL,
    schedule_pattern VARCHAR(100) NOT NULL,
    timezone VARCHAR(50) DEFAULT 'Australia/Sydney',
    task_function VARCHAR(200) NOT NULL,
    timeout_seconds INT DEFAULT 3600,
    max_retries INT DEFAULT 3,
    retry_delay_seconds INT DEFAULT 300,
    depends_on TEXT,
    config JSON,
    enabled BOOLEAN DEFAULT TRUE,
    ml_optimization_enabled BOOLEAN DEFAULT TRUE,
    estimated_duration_seconds INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_job_type (job_type),
    INDEX idx_enabled (enabled),
    INDEX idx_priority (priority)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Job Executions
CREATE TABLE IF NOT EXISTS job_executions (
    execution_id VARCHAR(36) PRIMARY KEY,
    job_id VARCHAR(50) NOT NULL,
    run_date DATETIME NOT NULL,
    status ENUM('pending', 'running', 'completed', 'failed', 'skipped', 'cancelled', 'retrying') NOT NULL,
    started_at DATETIME,
    completed_at DATETIME,
    duration_seconds INT,
    error_message TEXT,
    retry_count INT DEFAULT 0,
    result JSON,
    metrics JSON,
    triggered_by VARCHAR(100) DEFAULT 'scheduler',
    FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id),
    INDEX idx_job_run (job_id, run_date),
    INDEX idx_status (status),
    INDEX idx_run_date (run_date),
    INDEX idx_completed (completed_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Job Dependencies
CREATE TABLE IF NOT EXISTS job_dependencies (
    dependency_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    child_job_id VARCHAR(50) NOT NULL,
    parent_job_id VARCHAR(50) NOT NULL,
    is_blocking BOOLEAN DEFAULT TRUE,
    FOREIGN KEY (child_job_id) REFERENCES scheduled_jobs(job_id),
    FOREIGN KEY (parent_job_id) REFERENCES scheduled_jobs(job_id),
    UNIQUE KEY unique_dependency (child_job_id, parent_job_id),
    INDEX idx_child (child_job_id),
    INDEX idx_parent (parent_job_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Job Performance Metrics
CREATE TABLE IF NOT EXISTS job_performance_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    job_id VARCHAR(50) NOT NULL,
    metric_date DATE NOT NULL,
    avg_duration_seconds INT,
    min_duration_seconds INT,
    max_duration_seconds INT,
    success_count INT DEFAULT 0,
    failure_count INT DEFAULT 0,
    total_executions INT DEFAULT 0,
    success_rate DECIMAL(5, 2),
    FOREIGN KEY (job_id) REFERENCES scheduled_jobs(job_id),
    UNIQUE KEY unique_job_date (job_id, metric_date),
    INDEX idx_date (metric_date)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- COB Status Tracking
CREATE TABLE IF NOT EXISTS cob_status (
    cob_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    business_date DATE NOT NULL UNIQUE,
    started_at DATETIME,
    completed_at DATETIME,
    status ENUM('in_progress', 'completed', 'failed', 'aborted') NOT NULL,
    total_jobs INT,
    successful_jobs INT,
    failed_jobs INT,
    total_duration_seconds INT,
    error_summary TEXT,
    INDEX idx_business_date (business_date),
    INDEX idx_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Job Execution Logs
CREATE TABLE IF NOT EXISTS job_execution_logs (
    log_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    execution_id VARCHAR(36) NOT NULL,
    log_level ENUM('debug', 'info', 'warning', 'error', 'critical') NOT NULL,
    log_message TEXT NOT NULL,
    logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (execution_id) REFERENCES job_executions(execution_id) ON DELETE CASCADE,
    INDEX idx_execution (execution_id),
    INDEX idx_logged_at (logged_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- ============================================================================
-- Views
-- ============================================================================

-- Active Jobs View
CREATE OR REPLACE VIEW v_active_jobs AS
SELECT 
    e.execution_id,
    j.job_name,
    j.job_type,
    e.status,
    e.started_at,
    TIMESTAMPDIFF(SECOND, e.started_at, NOW()) as running_seconds,
    j.timeout_seconds
FROM job_executions e
JOIN scheduled_jobs j ON e.job_id = j.job_id
WHERE e.status IN ('running', 'retrying')
ORDER BY e.started_at;

-- Job Success Rate View
CREATE OR REPLACE VIEW v_job_success_rates AS
SELECT 
    j.job_id,
    j.job_name,
    COUNT(*) as total_executions,
    SUM(CASE WHEN e.status = 'completed' THEN 1 ELSE 0 END) as successful,
    SUM(CASE WHEN e.status = 'failed' THEN 1 ELSE 0 END) as failed,
    ROUND(SUM(CASE WHEN e.status = 'completed' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as success_rate,
    AVG(e.duration_seconds) as avg_duration
FROM scheduled_jobs j
LEFT JOIN job_executions e ON j.job_id = e.job_id
WHERE e.completed_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
GROUP BY j.job_id, j.job_name;

-- ============================================================================
-- Stored Procedures
-- ============================================================================

DELIMITER //

CREATE PROCEDURE sp_get_job_history(
    IN p_job_id VARCHAR(50),
    IN p_days INT
)
BEGIN
    SELECT 
        execution_id,
        run_date,
        status,
        started_at,
        completed_at,
        duration_seconds,
        error_message
    FROM job_executions
    WHERE job_id = p_job_id
        AND run_date >= DATE_SUB(CURDATE(), INTERVAL p_days DAY)
    ORDER BY run_date DESC;
END//

DELIMITER ;

-- ============================================================================
-- Initial Job Definitions
-- ============================================================================

-- COB Jobs
INSERT INTO scheduled_jobs (job_id, job_name, job_type, priority, schedule_pattern, task_function, timeout_seconds, estimated_duration_seconds, enabled) VALUES
('job_trade_settlements', 'Trade Settlements', 'cob_process', 1, '0 18 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_trade_settlements', 1800, 120, TRUE),
('job_portfolio_valuation', 'Portfolio Valuation', 'cob_process', 1, '0 18 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_portfolio_valuation', 3600, 180, TRUE),
('job_fee_calculation', 'Fee Calculation', 'cob_process', 2, '0 18 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_fee_calculation', 1800, 60, TRUE),
('job_interest_accrual', 'Interest Accrual', 'cob_process', 2, '0 0 * * *', 'app.jobs.cob_tasks.cob_jobs.job_interest_accrual', 1800, 60, TRUE),
('job_report_generation', 'Report Generation', 'cob_process', 3, '0 19 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_report_generation', 3600, 120, TRUE),
('job_reconciliation', 'Reconciliation', 'cob_process', 2, '0 20 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_reconciliation', 1800, 120, TRUE),
('job_ledger_close', 'Ledger Close', 'cob_process', 1, '0 21 * * 1-5', 'app.jobs.cob_tasks.cob_jobs.job_ledger_close', 900, 60, TRUE);

-- Dependencies
INSERT INTO job_dependencies (child_job_id, parent_job_id) VALUES
('job_portfolio_valuation', 'job_trade_settlements'),
('job_fee_calculation', 'job_portfolio_valuation'),
('job_report_generation', 'job_portfolio_valuation'),
('job_ledger_close', 'job_reconciliation');
