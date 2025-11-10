"""
TuringWealth - Double-Entry Accounting System
Production-grade General Ledger with automated journal entries
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date
from enum import Enum
import uuid
import asyncio

# ============================================================================
# DOMAIN MODELS
# ============================================================================

class AccountType(Enum):
    """Account types following standard accounting"""
    ASSET = "asset"
    LIABILITY = "liability"
    EQUITY = "equity"
    INCOME = "income"
    EXPENSE = "expense"

class AccountCategory(Enum):
    """Detailed account categories"""
    # Assets
    CASH = "cash"
    INVESTMENTS = "investments"
    RECEIVABLES = "receivables"
    PREPAID = "prepaid"
    FIXED_ASSETS = "fixed_assets"
    
    # Liabilities
    PAYABLES = "payables"
    CLIENT_BALANCES = "client_balances"
    ACCRUED_EXPENSES = "accrued_expenses"
    
    # Equity
    SHARE_CAPITAL = "share_capital"
    RETAINED_EARNINGS = "retained_earnings"
    CURRENT_YEAR_EARNINGS = "current_year_earnings"
    
    # Income
    MANAGEMENT_FEES = "management_fees"
    PERFORMANCE_FEES = "performance_fees"
    INTEREST_INCOME = "interest_income"
    
    # Expenses
    OPERATING_EXPENSES = "operating_expenses"
    BROKERAGE_FEES = "brokerage_fees"
    INTEREST_EXPENSE = "interest_expense"
    TECHNOLOGY_COSTS = "technology_costs"

class EntryStatus(Enum):
    DRAFT = "draft"
    POSTED = "posted"
    REVERSED = "reversed"
    VOID = "void"

@dataclass
class GLAccount:
    """General Ledger Account"""
    account_code: str
    account_name: str
    account_type: AccountType
    account_category: AccountCategory
    parent_account: Optional[str] = None
    is_active: bool = True
    is_manual_entries_allowed: bool = True
    is_header_account: bool = False  # Summary account (can't post to)
    description: str = ""
    created_at: datetime = None
    updated_at: datetime = None

@dataclass
class JournalEntryLine:
    """Individual line in journal entry"""
    line_id: Optional[int]
    entry_id: str
    account_code: str
    debit_amount: Decimal
    credit_amount: Decimal
    description: Optional[str] = None

@dataclass
class JournalEntry:
    """Double-entry journal entry (header)"""
    entry_id: str
    transaction_id: str
    entry_date: date
    transaction_date: date
    entry_type: str
    reference: str
    description: str
    
    # Lines (debits and credits)
    lines: List[JournalEntryLine]
    
    # Metadata
    created_by: str
    created_at: datetime
    status: EntryStatus = EntryStatus.DRAFT
    posted_at: Optional[datetime] = None
    posted_by: Optional[str] = None
    reversed_by: Optional[str] = None
    reversal_entry_id: Optional[str] = None
    
    def validate(self) -> Tuple[bool, str]:
        """Validate journal entry balances"""
        total_debits = sum(line.debit_amount for line in self.lines)
        total_credits = sum(line.credit_amount for line in self.lines)
        
        diff = abs(total_debits - total_credits)
        if diff > Decimal("0.01"):  # Allow 1 cent rounding
            return False, f"Entry does not balance: DR {total_debits} vs CR {total_credits}"
        
        if len(self.lines) < 2:
            return False, "Entry must have at least 2 lines"
        
        return True, "Valid"

class DoubleEntryLedger:
    """
    Core double-entry accounting system
    
    Features:
    - Chart of Accounts management
    - Journal entry posting
    - Trial balance generation
    - Financial statements
    - Period closures
    - DataMesh integration
    """
    
    def __init__(self, db_session, datamesh_client=None):
        self.db = db_session
        self.datamesh = datamesh_client
        self.chart_of_accounts = {}
    
    async def initialize_chart_of_accounts(self) -> Dict[str, GLAccount]:
        """
        Initialize standard Chart of Accounts for wealth management
        
        Structure:
        1000-1999: Assets
        2000-2999: Liabilities
        3000-3999: Equity
        4000-4999: Income
        5000-5999: Expenses
        """
        
        accounts = {
            # ========== ASSETS ==========
            "1000": GLAccount(
                account_code="1000",
                account_name="ASSETS",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.CASH,
                is_header_account=True,
                description="Total Assets"
            ),
            
            # Cash Accounts
            "1010": GLAccount(
                account_code="1010",
                account_name="Cash - Operating Account",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.CASH,
                parent_account="1000",
                description="Main operating cash account"
            ),
            "1020": GLAccount(
                account_code="1020",
                account_name="Cash - Client Trust Account",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.CASH,
                parent_account="1000",
                description="Client trust account (Zepto/Cuscal)"
            ),
            "1030": GLAccount(
                account_code="1030",
                account_name="Cash - Reserve Account",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.CASH,
                parent_account="1000",
                description="Reserve/emergency fund"
            ),
            
            # Investment Accounts
            "1100": GLAccount(
                account_code="1100",
                account_name="Investments - Client Holdings",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.INVESTMENTS,
                parent_account="1000",
                description="Client portfolio holdings (at cost)"
            ),
            "1110": GLAccount(
                account_code="1110",
                account_name="Investments - Unsettled Trades",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.INVESTMENTS,
                parent_account="1000",
                description="Trades pending settlement"
            ),
            
            # Receivables
            "1200": GLAccount(
                account_code="1200",
                account_name="Receivables - Brokerage Settlements",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.RECEIVABLES,
                parent_account="1000",
                description="Amounts receivable from broker"
            ),
            "1210": GLAccount(
                account_code="1210",
                account_name="Receivables - Management Fees",
                account_type=AccountType.ASSET,
                account_category=AccountCategory.RECEIVABLES,
                parent_account="1000",
                description="Accrued management fees not yet collected"
            ),
            
            # ========== LIABILITIES ==========
            "2000": GLAccount(
                account_code="2000",
                account_name="LIABILITIES",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.CLIENT_BALANCES,
                is_header_account=True,
                description="Total Liabilities"
            ),
            
            # Client Balances
            "2010": GLAccount(
                account_code="2010",
                account_name="Client Portfolio Balances",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.CLIENT_BALANCES,
                parent_account="2000",
                description="Total value owed to clients"
            ),
            "2020": GLAccount(
                account_code="2020",
                account_name="Client Cash Balances",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.CLIENT_BALANCES,
                parent_account="2000",
                description="Client cash in trust accounts"
            ),
            
            # Payables
            "2100": GLAccount(
                account_code="2100",
                account_name="Payables - Brokerage",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.PAYABLES,
                parent_account="2000",
                description="Amounts owed to OpenMarkets"
            ),
            "2110": GLAccount(
                account_code="2110",
                account_name="Payables - Trade Settlements",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.PAYABLES,
                parent_account="2000",
                description="Trade settlements payable"
            ),
            
            # Accrued Expenses
            "2200": GLAccount(
                account_code="2200",
                account_name="Accrued Interest Payable",
                account_type=AccountType.LIABILITY,
                account_category=AccountCategory.ACCRUED_EXPENSES,
                parent_account="2000",
                description="Interest accrued on client accounts"
            ),
            
            # ========== EQUITY ==========
            "3000": GLAccount(
                account_code="3000",
                account_name="EQUITY",
                account_type=AccountType.EQUITY,
                account_category=AccountCategory.SHARE_CAPITAL,
                is_header_account=True,
                description="Total Equity"
            ),
            
            "3010": GLAccount(
                account_code="3010",
                account_name="Share Capital",
                account_type=AccountType.EQUITY,
                account_category=AccountCategory.SHARE_CAPITAL,
                parent_account="3000",
                description="Issued share capital"
            ),
            "3100": GLAccount(
                account_code="3100",
                account_name="Retained Earnings",
                account_type=AccountType.EQUITY,
                account_category=AccountCategory.RETAINED_EARNINGS,
                parent_account="3000",
                description="Accumulated profits from prior years"
            ),
            "3200": GLAccount(
                account_code="3200",
                account_name="Current Year Earnings",
                account_type=AccountType.EQUITY,
                account_category=AccountCategory.CURRENT_YEAR_EARNINGS,
                parent_account="3000",
                description="Profit/loss for current year"
            ),
            
            # ========== INCOME ==========
            "4000": GLAccount(
                account_code="4000",
                account_name="INCOME",
                account_type=AccountType.INCOME,
                account_category=AccountCategory.MANAGEMENT_FEES,
                is_header_account=True,
                description="Total Income"
            ),
            
            "4010": GLAccount(
                account_code="4010",
                account_name="Management Fee Income",
                account_type=AccountType.INCOME,
                account_category=AccountCategory.MANAGEMENT_FEES,
                parent_account="4000",
                description="Recurring management fees (0.40-0.60%)"
            ),
            "4020": GLAccount(
                account_code="4020",
                account_name="Performance Fee Income",
                account_type=AccountType.INCOME,
                account_category=AccountCategory.PERFORMANCE_FEES,
                parent_account="4000",
                description="Performance-based fees"
            ),
            "4100": GLAccount(
                account_code="4100",
                account_name="Interest Income",
                account_type=AccountType.INCOME,
                account_category=AccountCategory.INTEREST_INCOME,
                parent_account="4000",
                description="Interest earned on cash balances"
            ),
            
            # ========== EXPENSES ==========
            "5000": GLAccount(
                account_code="5000",
                account_name="EXPENSES",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.OPERATING_EXPENSES,
                is_header_account=True,
                description="Total Expenses"
            ),
            
            "5010": GLAccount(
                account_code="5010",
                account_name="Brokerage Fees",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.BROKERAGE_FEES,
                parent_account="5000",
                description="OpenMarkets trading costs"
            ),
            "5020": GLAccount(
                account_code="5020",
                account_name="Interest Expense",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.INTEREST_EXPENSE,
                parent_account="5000",
                description="Interest paid to clients on CMA"
            ),
            "5100": GLAccount(
                account_code="5100",
                account_name="Technology Costs",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.TECHNOLOGY_COSTS,
                parent_account="5000",
                description="AWS, DataMesh, MCP infrastructure"
            ),
            "5110": GLAccount(
                account_code="5110",
                account_name="Data Provider Costs",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.TECHNOLOGY_COSTS,
                parent_account="5000",
                description="Market data subscriptions"
            ),
            "5200": GLAccount(
                account_code="5200",
                account_name="Operating Expenses",
                account_type=AccountType.EXPENSE,
                account_category=AccountCategory.OPERATING_EXPENSES,
                parent_account="5000",
                description="General operating costs"
            ),
        }
        
        # Store in memory
        self.chart_of_accounts = accounts
        
        # Persist to database
        for account in accounts.values():
            await self._save_account(account)
        
        return accounts
    
    async def post_journal_entry(
        self,
        entry: JournalEntry,
        auto_post: bool = True
    ) -> str:
        """
        Post journal entry to ledger
        
        Args:
            entry: Journal entry to post
            auto_post: Automatically post (vs save as draft)
        
        Returns:
            entry_id
        """
        
        # Validate entry
        is_valid, error_msg = entry.validate()
        if not is_valid:
            raise ValueError(f"Invalid journal entry: {error_msg}")
        
        # Validate accounts exist
        for line in entry.lines:
            if line.account_code not in self.chart_of_accounts:
                account = await self._get_account(line.account_code)
                if not account:
                    raise ValueError(f"Account does not exist: {line.account_code}")
        
        # Save header
        await self.db.execute("""
            INSERT INTO journal_entries (
                entry_id, transaction_id, entry_date, transaction_date,
                entry_type, reference, description, created_by, created_at,
                status, posted_at, posted_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.entry_id, entry.transaction_id, entry.entry_date,
            entry.transaction_date, entry.entry_type, entry.reference,
            entry.description, entry.created_by, entry.created_at,
            EntryStatus.POSTED.value if auto_post else EntryStatus.DRAFT.value,
            datetime.now() if auto_post else None,
            entry.created_by if auto_post else None
        ))
        
        # Save lines
        for line in entry.lines:
            await self.db.execute("""
                INSERT INTO journal_entry_lines (
                    entry_id, account_code, debit_amount, credit_amount, description
                ) VALUES (?, ?, ?, ?, ?)
            """, (entry.entry_id, line.account_code, line.debit_amount,
                  line.credit_amount, line.description))
        
        await self.db.commit()
        
        # Publish to DataMesh
        if self.datamesh and auto_post:
            await self.datamesh.events.publish({
                "event_type": "JOURNAL_ENTRY_POSTED",
                "entry_id": entry.entry_id,
                "entry_type": entry.entry_type,
                "total_amount": sum(line.debit_amount for line in entry.lines),
                "timestamp": datetime.now().isoformat()
            })
        
        return entry.entry_id
    
    async def post_trade_settlement(
        self,
        trade_id: str,
        client_id: str,
        ticker: str,
        action: str,  # BUY or SELL
        shares: int,
        price: Decimal,
        brokerage_fee: Decimal,
        settlement_date: date
    ) -> str:
        """
        Automatically post journal entries for trade settlement
        
        BUY Trade:
        DR: Investments - Client Holdings (1100)  [shares * price]
        DR: Brokerage Fees (5010)                  [brokerage_fee]
        CR: Cash - Client Trust (1020)             [total]
        
        SELL Trade:
        DR: Cash - Client Trust (1020)             [proceeds - brokerage]
        DR: Brokerage Fees (5010)                  [brokerage_fee]
        CR: Investments - Client Holdings (1100)   [shares * price]
        """
        
        entry_id = str(uuid.uuid4())
        total_cost = shares * price
        
        lines = []
        
        if action == "BUY":
            # Debit: Investment
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="1100",
                debit_amount=total_cost,
                credit_amount=Decimal("0"),
                description=f"Buy {shares} {ticker} @ ${price}"
            ))
            
            # Debit: Brokerage expense
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="5010",
                debit_amount=brokerage_fee,
                credit_amount=Decimal("0"),
                description="Brokerage fee"
            ))
            
            # Credit: Cash
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="1020",
                debit_amount=Decimal("0"),
                credit_amount=total_cost + brokerage_fee,
                description="Trade settlement"
            ))
            
        else:  # SELL
            # Debit: Cash
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="1020",
                debit_amount=total_cost - brokerage_fee,
                credit_amount=Decimal("0"),
                description="Trade proceeds"
            ))
            
            # Debit: Brokerage expense
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="5010",
                debit_amount=brokerage_fee,
                credit_amount=Decimal("0"),
                description="Brokerage fee"
            ))
            
            # Credit: Investment
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="1100",
                debit_amount=Decimal("0"),
                credit_amount=total_cost,
                description=f"Sell {shares} {ticker} @ ${price}"
            ))
        
        entry = JournalEntry(
            entry_id=entry_id,
            transaction_id=trade_id,
            entry_date=date.today(),
            transaction_date=settlement_date,
            entry_type=f"TRADE_{action}",
            reference=f"{action} {shares} {ticker}",
            description=f"Trade settlement for client {client_id}",
            lines=lines,
            created_by="system",
            created_at=datetime.now()
        )
        
        return await self.post_journal_entry(entry, auto_post=True)
    
    async def post_management_fee(
        self,
        client_id: str,
        portfolio_value: Decimal,
        fee_rate: Decimal,
        period: str
    ) -> str:
        """
        Post management fee accrual
        
        DR: Client Portfolio Balances (2010)  [fee reduces client balance]
        CR: Management Fee Income (4010)      [company revenue]
        """
        
        fee_amount = portfolio_value * fee_rate
        entry_id = str(uuid.uuid4())
        
        lines = [
            JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="2010",
                debit_amount=fee_amount,
                credit_amount=Decimal("0"),
                description=f"Management fee - {period}"
            ),
            JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="4010",
                debit_amount=Decimal("0"),
                credit_amount=fee_amount,
                description=f"Management fee revenue - {period}"
            )
        ]
        
        entry = JournalEntry(
            entry_id=entry_id,
            transaction_id=f"FEE-{client_id}-{period}",
            entry_date=date.today(),
            transaction_date=date.today(),
            entry_type="MANAGEMENT_FEE",
            reference=f"Management fee {period}",
            description=f"Client {client_id} - {fee_rate*100:.2f}% of ${portfolio_value:,.2f}",
            lines=lines,
            created_by="system",
            created_at=datetime.now()
        )
        
        return await self.post_journal_entry(entry, auto_post=True)
    
    async def post_interest_payment(
        self,
        account_id: str,
        interest_amount: Decimal
    ) -> str:
        """
        Post interest payment to client CMA
        
        DR: Interest Expense (5020)
        CR: Client Cash Balances (2020)
        """
        
        entry_id = str(uuid.uuid4())
        
        lines = [
            JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="5020",
                debit_amount=interest_amount,
                credit_amount=Decimal("0"),
                description="Interest paid to client"
            ),
            JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="2020",
                debit_amount=Decimal("0"),
                credit_amount=interest_amount,
                description=f"Interest credit - Account {account_id}"
            )
        ]
        
        entry = JournalEntry(
            entry_id=entry_id,
            transaction_id=f"INTEREST-{account_id}",
            entry_date=date.today(),
            transaction_date=date.today(),
            entry_type="INTEREST_PAYMENT",
            reference=f"Interest payment",
            description=f"Monthly interest to account {account_id}",
            lines=lines,
            created_by="system",
            created_at=datetime.now()
        )
        
        return await self.post_journal_entry(entry, auto_post=True)
    
    async def generate_trial_balance(
        self,
        as_of_date: date
    ) -> Dict:
        """
        Generate trial balance report
        
        Returns summary of all account balances
        """
        
        query = """
        SELECT 
            a.account_code,
            a.account_name,
            a.account_type,
            COALESCE(SUM(l.debit_amount), 0) as total_debits,
            COALESCE(SUM(l.credit_amount), 0) as total_credits
        FROM gl_accounts a
        LEFT JOIN journal_entry_lines l ON a.account_code = l.account_code
        LEFT JOIN journal_entries e ON l.entry_id = e.entry_id
        WHERE e.status = 'posted'
            AND e.entry_date <= ?
            AND a.is_header_account = 0
        GROUP BY a.account_code, a.account_name, a.account_type
        ORDER BY a.account_code
        """
        
        rows = await self.db.fetch_all(query, (as_of_date,))
        
        trial_balance = []
        total_debits = Decimal("0")
        total_credits = Decimal("0")
        
        for row in rows:
            net_balance = Decimal(str(row["total_debits"])) - Decimal(str(row["total_credits"]))
            debit_balance = net_balance if net_balance > 0 else Decimal("0")
            credit_balance = abs(net_balance) if net_balance < 0 else Decimal("0")
            
            trial_balance.append({
                "account_code": row["account_code"],
                "account_name": row["account_name"],
                "account_type": row["account_type"],
                "debit_balance": float(debit_balance),
                "credit_balance": float(credit_balance)
            })
            
            total_debits += debit_balance
            total_credits += credit_balance
        
        return {
            "as_of_date": as_of_date.isoformat(),
            "accounts": trial_balance,
            "total_debits": float(total_debits),
            "total_credits": float(total_credits),
            "balanced": abs(total_debits - total_credits) < Decimal("0.01")
        }
    
    async def generate_balance_sheet(
        self,
        as_of_date: date
    ) -> Dict:
        """Generate Balance Sheet"""
        
        trial_balance = await self.generate_trial_balance(as_of_date)
        
        assets = []
        liabilities = []
        equity = []
        
        for account in trial_balance["accounts"]:
            if account["account_type"] == "asset":
                assets.append(account)
            elif account["account_type"] == "liability":
                liabilities.append(account)
            elif account["account_type"] == "equity":
                equity.append(account)
        
        total_assets = sum(a["debit_balance"] for a in assets)
        total_liabilities = sum(l["credit_balance"] for l in liabilities)
        total_equity = sum(e["credit_balance"] for e in equity)
        
        return {
            "as_of_date": as_of_date.isoformat(),
            "assets": {
                "accounts": assets,
                "total": total_assets
            },
            "liabilities": {
                "accounts": liabilities,
                "total": total_liabilities
            },
            "equity": {
                "accounts": equity,
                "total": total_equity
            },
            "balanced": abs(total_assets - (total_liabilities + total_equity)) < 0.01
        }
    
    async def generate_income_statement(
        self,
        start_date: date,
        end_date: date
    ) -> Dict:
        """Generate Income Statement (P&L)"""
        
        query = """
        SELECT 
            a.account_code,
            a.account_name,
            a.account_type,
            COALESCE(SUM(l.credit_amount - l.debit_amount), 0) as net_amount
        FROM gl_accounts a
        LEFT JOIN journal_entry_lines l ON a.account_code = l.account_code
        LEFT JOIN journal_entries e ON l.entry_id = e.entry_id
        WHERE e.status = 'posted'
            AND e.entry_date >= ?
            AND e.entry_date <= ?
            AND a.account_type IN ('income', 'expense')
            AND a.is_header_account = 0
        GROUP BY a.account_code, a.account_name, a.account_type
        ORDER BY a.account_code
        """
        
        rows = await self.db.fetch_all(query, (start_date, end_date))
        
        income_accounts = []
        expense_accounts = []
        
        for row in rows:
            account_data = {
                "account_code": row["account_code"],
                "account_name": row["account_name"],
                "amount": float(row["net_amount"])
            }
            
            if row["account_type"] == "income":
                income_accounts.append(account_data)
            else:
                expense_accounts.append(account_data)
        
        total_income = sum(a["amount"] for a in income_accounts)
        total_expenses = sum(a["amount"] for a in expense_accounts)
        net_income = total_income - total_expenses
        
        return {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "income": {
                "accounts": income_accounts,
                "total": total_income
            },
            "expenses": {
                "accounts": expense_accounts,
                "total": total_expenses
            },
            "net_income": net_income
        }
    
    async def close_accounting_period(
        self,
        period_end_date: date
    ) -> str:
        """
        Close accounting period (month-end, year-end)
        
        Process:
        1. Calculate net income (revenue - expenses)
        2. Transfer to Current Year Earnings (3200)
        3. Mark period as closed
        4. Prevent further postings to closed period
        """
        
        # Get net income
        income_stmt = await self.generate_income_statement(
            start_date=date(period_end_date.year, 1, 1),
            end_date=period_end_date
        )
        
        net_income = Decimal(str(income_stmt["net_income"]))
        
        # Create closing entry
        entry_id = str(uuid.uuid4())
        lines = []
        
        # Close income accounts (debit income, credit summary)
        for income_account in income_stmt["income"]["accounts"]:
            if income_account["amount"] != 0:
                lines.append(JournalEntryLine(
                    line_id=None,
                    entry_id=entry_id,
                    account_code=income_account["account_code"],
                    debit_amount=Decimal(str(income_account["amount"])),
                    credit_amount=Decimal("0"),
                    description="Period closing entry"
                ))
        
        # Close expense accounts (debit summary, credit expenses)
        for expense_account in income_stmt["expenses"]["accounts"]:
            if expense_account["amount"] != 0:
                lines.append(JournalEntryLine(
                    line_id=None,
                    entry_id=entry_id,
                    account_code=expense_account["account_code"],
                    debit_amount=Decimal("0"),
                    credit_amount=Decimal(str(expense_account["amount"])),
                    description="Period closing entry"
                ))
        
        # Transfer net income to Current Year Earnings
        if net_income != 0:
            lines.append(JournalEntryLine(
                line_id=None,
                entry_id=entry_id,
                account_code="3200",
                debit_amount=Decimal("0") if net_income > 0 else abs(net_income),
                credit_amount=net_income if net_income > 0 else Decimal("0"),
                description="Net income for period"
            ))
        
        entry = JournalEntry(
            entry_id=entry_id,
            transaction_id=f"CLOSE-{period_end_date.isoformat()}",
            entry_date=period_end_date,
            transaction_date=period_end_date,
            entry_type="PERIOD_CLOSE",
            reference=f"Period closing {period_end_date}",
            description="Close income and expense accounts",
            lines=lines,
            created_by="system",
            created_at=datetime.now()
        )
        
        closing_entry_id = await self.post_journal_entry(entry, auto_post=True)
        
        # Mark period as closed
        await self.db.execute("""
            INSERT INTO accounting_closures (
                closure_date, net_income, closed_by, closed_at
            ) VALUES (?, ?, ?, ?)
        """, (period_end_date, float(net_income), "system", datetime.now()))
        
        await self.db.commit()
        
        return closing_entry_id
    
    async def _save_account(self, account: GLAccount):
        """Save GL account to database"""
        await self.db.execute("""
            INSERT OR REPLACE INTO gl_accounts (
                account_code, account_name, account_type, account_category,
                parent_account, is_active, is_manual_entries_allowed,
                is_header_account, description, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            account.account_code, account.account_name,
            account.account_type.value, account.account_category.value,
            account.parent_account, account.is_active,
            account.is_manual_entries_allowed, account.is_header_account,
            account.description, datetime.now(), datetime.now()
        ))
    
    async def _get_account(self, account_code: str) -> Optional[GLAccount]:
        """Get GL account from database"""
        row = await self.db.fetch_one(
            "SELECT * FROM gl_accounts WHERE account_code = ?",
            (account_code,)
        )
        if row:
            return GLAccount(**dict(row))
        return None
