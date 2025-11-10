# TuringWealth - Audit Trail & Maker-Checker System

## ?? Overview

AFSL-compliant audit and approval system with:
- Immutable audit trail with chain verification
- 4-eyes approval workflow (maker-checker)
- AI-powered risk scoring
- ML-based compliance monitoring
- CQRS pattern (Command/Query separation)
- DataMesh event sourcing
- Real-time violation detection

## ?? Quick Start

### 1. Run Database Migration
```bash
mysql -u root -p < backend/migrations/audit/001_create_audit_tables.sql
```

### 2. Initialize System
```python
from app.audit.audit_trail_system import AuditTrailSystem

audit = AuditTrailSystem(db_session, datamesh_client)
```

### 3. Log Your First Change
```python
await audit.log_change(
    user_id="user-123",
    user_email="john@turingwealth.com",
    user_role="admin",
    entity_type="CLIENT",
    entity_id="client-456",
    change_type="UPDATE",
    old_value={"status": "pending"},
    new_value={"status": "active"},
    reason="Client onboarding complete"
)
```

### 4. Create Approval Request
```python
request = await audit.create_approval_request(
    maker_id="maker-123",
    maker_email="maker@turingwealth.com",
    entity_type="TRANSACTION",
    entity_id="txn-789",
    change_type="CREATE",
    proposed_changes={"amount": 150000, "type": "withdrawal"},
    reason="Large withdrawal request"
)
```

## ?? Features

### Immutable Audit Trail
- **Chain of Custody** - SHA-256 checksum verification
- **Tamper-Proof** - Blockchain-style linking
- **7-Year Retention** - ASIC compliant
- **Complete Context** - User, IP, timestamp, reason

### Maker-Checker Workflow
1. **Maker** creates change request
2. **System** calculates risk score
3. **Checker** approves or rejects
4. **System** executes if approved
5. **Audit** logs entire workflow

### Risk Levels
- **LOW** - Routine changes (auto-approved)
- **MEDIUM** - Requires 1 approval
- **HIGH** - Requires 2 approvals  
- **CRITICAL** - Requires 3 approvals + justification

### AI/ML Features

#### Risk Scoring Model
```python
# ML-based risk prediction
risk = await audit._calculate_risk_score(change)

Factors:
- Change type (delete = high risk)
- Entity type (financial = high risk)
- Amount/magnitude
- Time of day (after hours)
- User history
```

#### Compliance Agent
```python
# Autonomous monitoring
agent = ComplianceMonitoringAgent(audit, memory)
await agent.run_daily_monitoring()

Detects:
- Maker-checker violations
- After-hours approvals
- Segregation of duties breaches
- Rubber stamping
- Unusual patterns
```

### CQRS Pattern

#### Commands (Write)
```python
from app.audit.commands import CreateClientCommand

command = CreateClientCommand(
    client_data={"name": "ABC Corp"},
    user_id="user-123"
)

result = await command_handler.handle_create_client(command)
```

#### Queries (Read)
```python
from app.audit.queries import QueryHandler

# Read operations don't modify state
clients = await query_handler.list_clients(limit=100)
```

## ?? MCP Tools

- `create_approval_request` - Submit change for approval
- `approve_request` - Approve pending request
- `reject_request` - Reject request with reason
- `get_pending_approvals` - List pending approvals
- `get_audit_trail` - Query audit history
- `verify_audit_chain` - Check integrity
- `get_risk_score` - ML risk prediction
- `get_compliance_report` - Generate report

## ??? AFSL Compliance

### Requirements Met
? **7-year audit retention**  
? **Maker-checker for financial transactions**  
? **Segregation of duties enforcement**  
? **Change justification mandatory**  
? **Approval evidence trail**  
? **Immutable audit log**  
? **Chain of custody verification**  

### Segregation of Duties Rules
- Maker cannot approve own requests
- Trading staff cannot settle trades
- Custody staff cannot execute trades
- Same person cannot create and approve journal entries

### Compliance Reports
```python
# Monthly compliance report
report = await audit_mcp.generate_compliance_report(
    start_date=date(2025, 1, 1),
    end_date=date(2025, 1, 31)
)
```

## ?? Risk Scoring

### Factors
- **Change Type** (25%) - Delete > Update > Create
- **Entity Type** (20%) - Transaction > Account > Settings
- **Amount** (30%) - Higher amounts = higher risk
- **Time Context** (10%) - After hours, weekends
- **User History** (15%) - Past violations, patterns

### Thresholds
- < 0.3 = LOW
- 0.3-0.5 = MEDIUM
- 0.5-0.7 = HIGH
- > 0.7 = CRITICAL

## ?? AI Agents

### Compliance Monitoring Agent
Runs daily at 8:00 AM:
- Scans for violations
- Analyzes approval patterns
- Detects suspicious activity
- Generates alerts

### Risk Analysis Agent
Continuous monitoring:
- Real-time risk assessment
- Pattern detection
- Anomaly identification
- Predictive compliance

## ?? Support

- Documentation: `/docs/audit_module`
- Issues: GitHub issues
- Slack: #turingwealth-compliance

## ? Next Steps

1. ? Run database migration
2. ? Configure approval thresholds
3. ? Set up SOD rules
4. ? Train ML risk model
5. ? Enable compliance monitoring
6. ? Test maker-checker workflow
