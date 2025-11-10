-- ============================================================================
-- Savings Module Database Schema
-- ============================================================================

-- Savings Products
CREATE TABLE IF NOT EXISTS savings_products (
    product_id VARCHAR(36) PRIMARY KEY,
    product_name VARCHAR(200) NOT NULL,
    base_interest_rate DECIMAL(10, 6) NOT NULL,
    calculation_method VARCHAR(50) NOT NULL,
    rate_tiers JSON,
    min_balance DECIMAL(19, 4) DEFAULT 0,
    min_opening_balance DECIMAL(19, 4) DEFAULT 100,
    interest_posting_frequency VARCHAR(20) DEFAULT 'monthly',
    interest_calculation_frequency VARCHAR(20) DEFAULT 'daily',
    ml_optimization_enabled BOOLEAN DEFAULT TRUE,
    dynamic_rate_adjustment BOOLEAN DEFAULT TRUE,
    monthly_maintenance_fee DECIMAL(19, 4) DEFAULT 0,
    active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_active (active)
);

-- Savings Accounts
CREATE TABLE IF NOT EXISTS savings_accounts (
    account_id VARCHAR(36) PRIMARY KEY,
    client_id VARCHAR(36) NOT NULL,
    product_id VARCHAR(36) NOT NULL,
    account_number VARCHAR(20) UNIQUE NOT NULL,
    balance DECIMAL(19, 4) DEFAULT 0,
    available_balance DECIMAL(19, 4) DEFAULT 0,
    hold_balance DECIMAL(19, 4) DEFAULT 0,
    accrued_interest_ytd DECIMAL(19, 4) DEFAULT 0,
    posted_interest_ytd DECIMAL(19, 4) DEFAULT 0,
    last_interest_calculation TIMESTAMP,
    last_interest_posting TIMESTAMP,
    status VARCHAR(20) DEFAULT 'ACTIVE',
    activated_date TIMESTAMP,
    optimized_interest_rate DECIMAL(10, 6),
    predicted_next_month_balance DECIMAL(19, 4),
    cash_flow_pattern VARCHAR(50),
    linked_brokerage_account VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (client_id) REFERENCES clients(client_id),
    FOREIGN KEY (product_id) REFERENCES savings_products(product_id),
    INDEX idx_client (client_id),
    INDEX idx_status (status),
    INDEX idx_account_number (account_number)
);

-- Interest Transactions
CREATE TABLE IF NOT EXISTS interest_transactions (
    transaction_id VARCHAR(36) PRIMARY KEY,
    account_id VARCHAR(36) NOT NULL,
    transaction_date DATE NOT NULL,
    transaction_type VARCHAR(20) NOT NULL,
    opening_balance DECIMAL(19, 4) NOT NULL,
    interest_rate DECIMAL(10, 6) NOT NULL,
    days INT NOT NULL,
    interest_amount DECIMAL(19, 4) NOT NULL,
    posted BOOLEAN DEFAULT FALSE,
    posted_at TIMESTAMP,
    journal_entry_id VARCHAR(36),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES savings_accounts(account_id),
    INDEX idx_account_date (account_id, transaction_date),
    INDEX idx_posted (posted)
);

-- Rate Adjustments (Audit Trail)
CREATE TABLE IF NOT EXISTS rate_adjustments (
    adjustment_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    account_id VARCHAR(36) NOT NULL,
    old_rate DECIMAL(10, 6),
    new_rate DECIMAL(10, 6) NOT NULL,
    reason TEXT,
    adjusted_by VARCHAR(100),
    adjusted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (account_id) REFERENCES savings_accounts(account_id),
    INDEX idx_account (account_id),
    INDEX idx_adjusted_at (adjusted_at)
);

-- Insert default product
INSERT INTO savings_products (
    product_id,
    product_name,
    base_interest_rate,
    calculation_method,
    min_balance,
    min_opening_balance,
    interest_posting_frequency,
    ml_optimization_enabled,
    active
) VALUES (
    'CMA_STANDARD',
    'Standard Cash Management Account',
    0.035,
    'compound_daily',
    0,
    100,
    'monthly',
    TRUE,
    TRUE
);
