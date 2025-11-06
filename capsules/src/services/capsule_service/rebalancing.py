from models import SessionLocal, Capsule, Allocation, Transaction
from datetime import datetime
import uuid

class RebalancingEngine:
    """Portfolio rebalancing logic"""
    
    def __init__(self, threshold=5.0):
        self.threshold = threshold  # Rebalance if drift > 5%
    
    def check_rebalance_needed(self, capsule_id):
        """Check if rebalancing is needed"""
        db = SessionLocal()
        try:
            allocations = db.query(Allocation).filter(
                Allocation.capsule_id == capsule_id
            ).all()
            
            max_drift = max([abs(a.current_percentage - a.target_percentage) 
                           for a in allocations], default=0)
            
            return max_drift > self.threshold, max_drift
        finally:
            db.close()
    
    def rebalance(self, capsule_id):
        """Perform rebalancing"""
        db = SessionLocal()
        try:
            capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
            if not capsule:
                return None
            
            allocations = db.query(Allocation).filter(
                Allocation.capsule_id == capsule_id
            ).all()
            
            total_value = capsule.current_value
            rebalance_actions = []
            
            for allocation in allocations:
                target_value = total_value * (allocation.target_percentage / 100)
                current_value = allocation.current_value
                difference = target_value - current_value
                
                if abs(difference) > 0:
                    # Create rebalance transaction
                    trans_id = f"trans_{uuid.uuid4().hex[:8]}"
                    transaction = Transaction(
                        id=trans_id,
                        capsule_id=capsule_id,
                        transaction_type="rebalance",
                        amount=difference,
                        description=f"Rebalance {allocation.asset_class}: ${difference:.2f}",
                        transaction_date=datetime.utcnow()
                    )
                    db.add(transaction)
                    
                    # Update allocation
                    allocation.current_value = target_value
                    allocation.current_percentage = allocation.target_percentage
                    allocation.updated_at = datetime.utcnow()
                    
                    rebalance_actions.append({
                        'asset_class': allocation.asset_class,
                        'action': 'buy' if difference > 0 else 'sell',
                        'amount': abs(difference)
                    })
            
            capsule.updated_at = datetime.utcnow()
            db.commit()
            
            return {
                'capsule_id': capsule_id,
                'rebalanced': True,
                'actions': rebalance_actions,
                'timestamp': datetime.utcnow().isoformat()
            }
        finally:
            db.close()

def calculate_performance(capsule_id, period='daily'):
    """Calculate performance metrics"""
    db = SessionLocal()
    try:
        from sqlalchemy import func
        
        transactions = db.query(Transaction).filter(
            Transaction.capsule_id == capsule_id
        ).order_by(Transaction.transaction_date).all()
        
        if len(transactions) < 2:
            return None
        
        start_value = 0
        end_value = 0
        
        for trans in transactions:
            if trans.transaction_type in ['deposit', 'dividend', 'interest']:
                end_value += trans.amount
            elif trans.transaction_type in ['withdrawal', 'fee']:
                end_value -= trans.amount
        
        if start_value == 0:
            start_value = transactions[0].amount if transactions else 0
        
        capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
        end_value = capsule.current_value if capsule else end_value
        
        return_pct = ((end_value - start_value) / start_value * 100) if start_value > 0 else 0
        
        return {
            'start_value': start_value,
            'end_value': end_value,
            'return_percentage': round(return_pct, 2),
            'absolute_return': end_value - start_value
        }
    finally:
        db.close()
