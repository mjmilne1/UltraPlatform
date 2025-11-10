"""
TuringWealth - Savings & CMA Service
Next-gen savings account system with AI/ML optimization
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from enum import Enum
import uuid
import asyncio

# ============================================================================
# DOMAIN MODELS
# ============================================================================

class InterestCalculationMethod(Enum):
    SIMPLE = "simple"
    COMPOUND_DAILY = "compound_daily"
    COMPOUND_MONTHLY = "compound_monthly"
    TIERED = "tiered"
    ML_OPTIMIZED = "ml_optimized"

class AccountStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    DORMANT = "dormant"
    FROZEN = "frozen"
    CLOSED = "closed"

@dataclass
class SavingsProduct:
    product_id: str
    product_name: str
    base_interest_rate: Decimal
    calculation_method: InterestCalculationMethod
    min_balance: Decimal = Decimal("0")
    min_opening_balance: Decimal = Decimal("100")
    interest_posting_frequency: str = "monthly"
    ml_optimization_enabled: bool = True
    active: bool = True

@dataclass
class SavingsAccount:
    account_id: str
    client_id: str
    product_id: str
    account_number: str
    balance: Decimal
    available_balance: Decimal
    accrued_interest_ytd: Decimal
    posted_interest_ytd: Decimal
    status: AccountStatus
    optimized_interest_rate: Optional[Decimal] = None

class SavingsAccountService:
    """Core savings account operations"""
    
    def __init__(self, datamesh_client, ledger_service):
        self.datamesh = datamesh_client
        self.ledger = ledger_service
    
    async def create_account(
        self,
        client_id: str,
        product_id: str,
        initial_deposit: Decimal = Decimal("0")
    ) -> Dict:
        """Create new savings account"""
        
        account_id = str(uuid.uuid4())
        account_number = self._generate_account_number()
        
        account_data = {
            "account_id": account_id,
            "client_id": client_id,
            "product_id": product_id,
            "account_number": account_number,
            "balance": float(initial_deposit),
            "available_balance": float(initial_deposit),
            "accrued_interest_ytd": 0.0,
            "posted_interest_ytd": 0.0,
            "status": "ACTIVE",
            "created_at": datetime.now().isoformat()
        }
        
        # Publish to DataMesh
        await self.datamesh.account_data.create_account(account_data)
        
        if initial_deposit > 0:
            await self.ledger.post_deposit(
                account_id=account_id,
                amount=initial_deposit,
                description="Initial deposit"
            )
        
        return {
            "success": True,
            "account_id": account_id,
            "account_number": account_number
        }
    
    async def calculate_daily_interest(
        self,
        account_id: str,
        as_of_date: datetime
    ) -> Dict:
        """Calculate daily interest"""
        
        account = await self.datamesh.account_data.get_account(account_id)
        product = await self._get_product(account["product_id"])
        
        # Get interest rate (ML-optimized or base)
        if product["ml_optimization_enabled"]:
            interest_rate = await self._get_ml_optimized_rate(
                account_id, 
                account["balance"]
            )
        else:
            interest_rate = Decimal(str(product["base_interest_rate"]))
        
        # Calculate daily interest
        balance = Decimal(str(account["balance"]))
        daily_rate = interest_rate / 365
        daily_interest = balance * daily_rate
        
        # Update accrued interest
        new_accrued = Decimal(str(account["accrued_interest_ytd"])) + daily_interest
        
        await self.datamesh.account_data.update_account(account_id, {
            "accrued_interest_ytd": float(new_accrued),
            "last_interest_calculation": as_of_date.isoformat()
        })
        
        return {
            "success": True,
            "daily_interest": float(daily_interest),
            "total_accrued": float(new_accrued)
        }
    
    async def post_monthly_interest(self, account_id: str) -> Dict:
        """Post accrued interest to account balance"""
        
        account = await self.datamesh.account_data.get_account(account_id)
        accrued = Decimal(str(account["accrued_interest_ytd"]))
        
        if accrued <= 0:
            return {"success": False, "message": "No interest to post"}
        
        new_balance = Decimal(str(account["balance"])) + accrued
        new_posted = Decimal(str(account["posted_interest_ytd"])) + accrued
        
        await self.datamesh.account_data.update_account(account_id, {
            "balance": float(new_balance),
            "available_balance": float(new_balance),
            "posted_interest_ytd": float(new_posted),
            "accrued_interest_ytd": 0.0,
            "last_interest_posting": datetime.now().isoformat()
        })
        
        # Post to ledger
        await self.ledger.post_journal_entry({
            "transaction_type": "INTEREST_PAYMENT",
            "debits": [{"account": "5200", "amount": float(accrued)}],
            "credits": [{"account": "2000", "amount": float(accrued)}],
            "description": f"Monthly interest - {account['account_number']}"
        })
        
        return {
            "success": True,
            "interest_posted": float(accrued),
            "new_balance": float(new_balance)
        }
    
    def _generate_account_number(self) -> str:
        """Generate unique account number"""
        import random
        return f"CMA{random.randint(10000000, 99999999)}"
    
    async def _get_product(self, product_id: str) -> Dict:
        """Get product configuration"""
        # Mock - replace with actual DB query
        return {
            "product_id": product_id,
            "base_interest_rate": 0.035,
            "ml_optimization_enabled": True
        }
    
    async def _get_ml_optimized_rate(
        self, 
        account_id: str, 
        balance: float
    ) -> Decimal:
        """Get ML-optimized interest rate"""
        base_rate = Decimal("0.035")
        
        # Balance-based adjustment
        if balance >= 100000:
            return base_rate + Decimal("0.005")  # +0.5%
        elif balance >= 50000:
            return base_rate
        else:
            return base_rate - Decimal("0.005")  # -0.5%
