-- ============================================================================
-- CAPSULES PLATFORM - INSTITUTIONAL SCHEMA
-- ASIC RG 255 Compliance Tables
-- ============================================================================

-- Client KYC Table
CREATE TABLE IF NOT EXISTS client_kyc (
    id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    legal_name VARCHAR(200) NOT NULL,
    date_of_birth DATE NOT NULL,
    residential_address TEXT NOT NULL,
    is_australian_tax_resident BOOLEAN NOT NULL,
    tax_identification_number VARCHAR(50),
    kyc_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk Assessment Table
CREATE TABLE IF NOT EXISTS risk_assessment (
    id VARCHAR(50) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    session_id VARCHAR(100),
    questionnaire_version VARCHAR(20) DEFAULT '1.0',
    r1_score INTEGER CHECK (r1_score BETWEEN 1 AND 5),
    r2_score INTEGER CHECK (r2_score BETWEEN 1 AND 5),
    r3_score INTEGER CHECK (r3_score BETWEEN 1 AND 5),
    r4_score INTEGER CHECK (r4_score BETWEEN 1 AND 5),
    r5_score INTEGER CHECK (r5_score BETWEEN 1 AND 5),
    total_risk_score INTEGER,
    risk_tolerance_band VARCHAR(20),
    assessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Onboarding Sessions Table
CREATE TABLE IF NOT EXISTS onboarding_sessions (
    id VARCHAR(100) PRIMARY KEY,
    client_id VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'started',
    current_step INTEGER DEFAULT 1,
    total_steps INTEGER DEFAULT 7,
    completion_percentage DECIMAL(5,2) DEFAULT 0.00,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Audit Log Table
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type VARCHAR(100) NOT NULL,
    client_id VARCHAR(100),
    session_id VARCHAR(100),
    action VARCHAR(200) NOT NULL,
    event_data TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create Indexes
CREATE INDEX IF NOT EXISTS idx_risk_assessment_client ON risk_assessment(client_id);
CREATE INDEX IF NOT EXISTS idx_onboarding_client ON onboarding_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp DESC);

SELECT 'Institutional schema ready' as status;
