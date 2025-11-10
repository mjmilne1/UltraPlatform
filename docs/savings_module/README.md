# TuringWealth - Savings/CMA Module

## ?? Overview

Next-generation savings account system with:
- TuringMachines DataMesh integration
- MCP standardized operations
- Agentic AI for rate optimization
- ML-based predictive analytics

## ?? Quick Start

### 1. Run Database Migration
```bash
mysql -u root -p < backend/migrations/savings/001_create_savings_tables.sql
```

### 2. Start Celery
```bash
celery -A app.jobs.tasks.savings_tasks worker --loglevel=info
celery -A app.jobs.tasks.savings_tasks beat --loglevel=info
```

### 3. Test
```python
from app.accounts.savings_service import SavingsAccountService
service = SavingsAccountService(datamesh, ledger)
result = await service.create_account("client-123", "CMA_STANDARD")
```

## ?? Batch Jobs

- 12:05 AM - Daily interest accrual
- 6:00 AM - AI rate optimization
- Month-end 11:50 PM - Interest posting

## ?? MCP Tools

- create_savings_account
- calculate_interest
- post_interest
- get_account_balance
