"""
CQRS Query Handlers
Queries that read state (reads only)
"""

from typing import Dict, List, Optional
from datetime import datetime

class QueryHandler:
    """
    CQRS Query Handler
    
    All read operations go through queries
    Queries:
    - Never modify state
    - Can be cached
    - Can use read replicas
    - Don't require audit logging
    """
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def get_client(self, client_id: str) -> Optional[Dict]:
        """Get client by ID"""
        
        row = await self.db.fetch_one(
            "SELECT * FROM clients WHERE client_id = ?",
            (client_id,)
        )
        
        return dict(row) if row else None
    
    async def get_account(self, account_id: str) -> Optional[Dict]:
        """Get account by ID"""
        
        row = await self.db.fetch_one(
            "SELECT * FROM accounts WHERE account_id = ?",
            (account_id,)
        )
        
        return dict(row) if row else None
    
    async def list_clients(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """List clients"""
        
        rows = await self.db.fetch_all(
            "SELECT * FROM clients ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        
        return [dict(row) for row in rows]
    
    async def search_transactions(
        self,
        account_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Search transactions"""
        
        query = "SELECT * FROM transactions WHERE 1=1"
        params = []
        
        if account_id:
            query += " AND account_id = ?"
            params.append(account_id)
        
        if start_date:
            query += " AND transaction_date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND transaction_date <= ?"
            params.append(end_date)
        
        query += " ORDER BY transaction_date DESC LIMIT ?"
        params.append(limit)
        
        rows = await self.db.fetch_all(query, tuple(params))
        
        return [dict(row) for row in rows]
    
    async def get_account_balance(self, account_id: str) -> Dict:
        """Get account balance (materialized view)"""
        
        row = await self.db.fetch_one(
            "SELECT * FROM account_balances WHERE account_id = ?",
            (account_id,)
        )
        
        return dict(row) if row else {"balance": 0}
