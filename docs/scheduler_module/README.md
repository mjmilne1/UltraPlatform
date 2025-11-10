# TuringWealth - Batch Job Scheduler (Turing Dynamics Edition)

## ?? Overview

AI-powered batch job orchestration with:
- Dependency-based execution graphs
- ML-optimized scheduling
- Real-time monitoring
- Auto-retry with exponential backoff
- DataMesh state synchronization
- MCP control interface
- Agentic optimization

## ?? Quick Start

### 1. Run Database Migration
```bash
mysql -u root -p < backend/migrations/scheduler/001_create_scheduler_tables.sql
```

### 2. Register Jobs
```python
from app.jobs.scheduler.batch_job_scheduler import BatchJobScheduler, JobDefinition

scheduler = BatchJobScheduler(db_session, datamesh_client)

# Register COB jobs (already in database via migration)
```

### 3. Execute COB Process
```python
from datetime import datetime

# Manual COB execution
await scheduler.execute_cob_process(datetime.today())

# Or via MCP
await mcp_client.call_tool("execute_cob", {
    "business_date": datetime.today()
})
```

## ?? Features

### Close of Business (COB) Automation
1. **Trade Settlements** - Settle all pending trades
2. **Portfolio Valuation** - Mark-to-market all positions
3. **Fee Calculation** - Accrue management fees
4. **Interest Accrual** - Calculate CMA interest
5. **Report Generation** - Daily reports
6. **Reconciliation** - Account reconciliations
7. **Ledger Close** - Lock accounting day

### ML Optimization
- **Duration Prediction** - ML-based runtime estimation
- **Schedule Optimization** - AI-driven timing adjustments
- **Resource Allocation** - Smart job prioritization
- **Failure Prediction** - Proactive issue detection

### Real-Time Monitoring
- Live job status dashboard
- Execution metrics
- Performance analytics
- Alert notifications

## ?? MCP Tools

- `execute_job` - Run job manually
- `execute_cob` - Trigger COB process
- `get_job_status` - Check execution status
- `predict_job_duration` - ML prediction
- `cancel_job` - Stop running job
- `list_active_jobs` - View active executions

## ?? Default Schedule

### Daily (Mon-Fri)
- **18:00** - COB Process starts
- **18:00** - Trade settlements
- **18:15** - Portfolio valuation
- **18:30** - Fee calculation
- **19:00** - Report generation
- **20:00** - Reconciliation
- **21:00** - Ledger close

### Daily
- **00:05** - Interest accrual
- **07:00** - Job optimization

## ?? AI Features

### Job Optimization Agent
- Analyzes job performance
- Suggests schedule improvements
- Auto-adjusts timeouts
- Reorders COB sequence

### ML Duration Predictor
- Predicts execution time
- Confidence intervals
- Historical analysis
- Context-aware predictions

## ?? Performance Metrics

- Success rate tracking
- Duration analytics
- Failure pattern detection
- Resource utilization

## ??? Reliability

- **Auto-Retry** - 3 attempts with exponential backoff
- **Timeout Protection** - Prevents hung jobs
- **Dependency Management** - Enforces execution order
- **Failure Recovery** - Graceful error handling

## ?? Support

- Documentation: `/docs/scheduler_module`
- Issues: GitHub issues
- Slack: #turingwealth-scheduler
