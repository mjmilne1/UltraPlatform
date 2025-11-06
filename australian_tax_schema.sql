CREATE TABLE IF NOT EXISTS tax_lots (
    id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    purchase_date DATE NOT NULL,
    quantity DECIMAL(18,8) NOT NULL,
    cost_base DECIMAL(18,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cgt_events (
    id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    event_date DATE NOT NULL,
    capital_gain DECIMAL(18,2),
    discount_applied DECIMAL(18,2),
    holding_days INTEGER,
    financial_year VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS wash_sale_monitoring (
    id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    asset_id VARCHAR(100) NOT NULL,
    sale_date DATE NOT NULL,
    proposed_rebuy_date DATE,
    days_between INTEGER,
    risk_level VARCHAR(20),
    risk_score INTEGER,
    compliance_status VARCHAR(50),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_tax_lots_client ON tax_lots(client_id);
CREATE INDEX IF NOT EXISTS idx_cgt_client ON cgt_events(client_id);
CREATE INDEX IF NOT EXISTS idx_wash_sale_client ON wash_sale_monitoring(client_id);
CREATE INDEX IF NOT EXISTS idx_wash_sale_date ON wash_sale_monitoring(sale_date);