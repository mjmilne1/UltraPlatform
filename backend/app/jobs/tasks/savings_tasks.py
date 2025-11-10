"""
Celery Tasks for Savings Module
Scheduled batch jobs
"""

from celery import Celery
from celery.schedules import crontab
from datetime import datetime
import asyncio

app = Celery('turingwealth')

@app.task
def daily_interest_accrual():
    """
    Daily interest accrual job
    Runs at 12:05 AM daily
    """
    print(f"[{datetime.now()}] Starting daily interest accrual...")
    asyncio.run(_run_daily_interest_accrual())
    print(f"[{datetime.now()}] Daily interest accrual complete.")

async def _run_daily_interest_accrual():
    """Async implementation"""
    from app.accounts.savings_service import SavingsAccountService
    
    # Get service (mock - replace with actual initialization)
    service = SavingsAccountService(None, None)
    
    # Get all active accounts
    # accounts = await get_all_active_accounts()
    
    # for account in accounts:
    #     await service.calculate_daily_interest(
    #         account_id=account["account_id"],
    #         as_of_date=datetime.now()
    #     )

@app.task
def monthly_interest_posting():
    """
    Monthly interest posting job
    Runs on last day of month at 11:50 PM
    """
    print(f"[{datetime.now()}] Starting monthly interest posting...")
    asyncio.run(_run_monthly_interest_posting())
    print(f"[{datetime.now()}] Monthly interest posting complete.")

async def _run_monthly_interest_posting():
    """Async implementation"""
    from app.accounts.savings_service import SavingsAccountService
    
    service = SavingsAccountService(None, None)
    
    # Get all accounts with accrued interest
    # accounts = await get_accounts_with_accrued_interest()
    
    # for account in accounts:
    #     await service.post_monthly_interest(
    #         account_id=account["account_id"]
    #     )

@app.task
def daily_rate_optimization():
    """
    AI-driven rate optimization job
    Runs at 6:00 AM daily
    """
    print(f"[{datetime.now()}] Starting rate optimization...")
    asyncio.run(_run_rate_optimization())
    print(f"[{datetime.now()}] Rate optimization complete.")

async def _run_rate_optimization():
    """Run optimization agent"""
    from app.accounts.agents.interest_optimization_agent import InterestOptimizationAgent
    
    agent = InterestOptimizationAgent(None, [])
    await agent.run_daily_optimization()

# Celery Beat Schedule
app.conf.beat_schedule = {
    'daily-interest-accrual': {
        'task': 'app.jobs.tasks.savings_tasks.daily_interest_accrual',
        'schedule': crontab(hour=0, minute=5),
    },
    'monthly-interest-posting': {
        'task': 'app.jobs.tasks.savings_tasks.monthly_interest_posting',
        'schedule': crontab(day_of_month=31, hour=23, minute=50),
    },
    'daily-rate-optimization': {
        'task': 'app.jobs.tasks.savings_tasks.daily_rate_optimization',
        'schedule': crontab(hour=6, minute=0),
    },
}
