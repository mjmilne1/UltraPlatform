from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging
from models import Capsule, Transaction, Allocation, Performance, SessionLocal
from rebalancing import RebalancingEngine, calculate_performance
from sqlalchemy.exc import SQLAlchemyError
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

rebalancer = RebalancingEngine(threshold=5.0)

with app.app_context():
    try:
        from models import init_db
        init_db()
        logger.info("✓ PostgreSQL connected")
    except Exception as e:
        logger.error(f"✗ Database error: {e}")

def get_db():
    return SessionLocal()

# ===== EXISTING ENDPOINTS =====

@app.route('/')
def home():
    return jsonify({
        'service': 'Capsules Platform',
        'version': '3.0.0',
        'database': 'PostgreSQL',
        'features': ['transactions', 'allocations', 'rebalancing', 'performance'],
        'status': 'operational'
    })

@app.route('/health')
def health():
    db = get_db()
    try:
        db.execute('SELECT 1')
        db_status = 'healthy'
    except:
        db_status = 'unhealthy'
    finally:
        db.close()
    return jsonify({'status': 'healthy', 'database': db_status}), 200

@app.route('/api/v1/capsules', methods=['GET', 'POST'])
def capsules():
    if request.method == 'GET':
        db = get_db()
        try:
            client_id = request.args.get('client_id')
            query = db.query(Capsule)
            if client_id:
                query = query.filter(Capsule.client_id == client_id)
            return jsonify([c.to_dict() for c in query.all()])
        finally:
            db.close()
    
    else:  # POST
        db = get_db()
        try:
            data = request.get_json()
            count = db.query(Capsule).count()
            capsule = Capsule(
                id=f"cap_{count + 1}",
                client_id=data['client_id'],
                capsule_type=data['capsule_type'],
                goal_amount=float(data['goal_amount']),
                goal_date=data['goal_date'],
                current_value=data.get('current_value', 0.0),
                status='created'
            )
            db.add(capsule)
            db.commit()
            db.refresh(capsule)
            return jsonify(capsule.to_dict()), 201
        finally:
            db.close()

@app.route('/api/v1/capsules/<capsule_id>', methods=['GET', 'PUT', 'DELETE'])
def capsule_detail(capsule_id):
    db = get_db()
    try:
        capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
        if not capsule:
            return jsonify({'error': 'Not found'}), 404
        
        if request.method == 'GET':
            return jsonify(capsule.to_dict())
        
        elif request.method == 'PUT':
            data = request.get_json()
            if 'current_value' in data:
                capsule.current_value = float(data['current_value'])
            if 'status' in data:
                capsule.status = data['status']
            capsule.updated_at = datetime.utcnow()
            db.commit()
            db.refresh(capsule)
            return jsonify(capsule.to_dict())
        
        else:  # DELETE
            db.delete(capsule)
            db.commit()
            return '', 204
    finally:
        db.close()

# ===== NEW TRANSACTION ENDPOINTS =====

@app.route('/api/v1/capsules/<capsule_id>/transactions', methods=['GET', 'POST'])
def transactions(capsule_id):
    if request.method == 'GET':
        db = get_db()
        try:
            trans = db.query(Transaction).filter(
                Transaction.capsule_id == capsule_id
            ).order_by(Transaction.transaction_date.desc()).all()
            return jsonify([t.to_dict() for t in trans])
        finally:
            db.close()
    
    else:  # POST - Create transaction
        db = get_db()
        try:
            data = request.get_json()
            trans_id = f"trans_{uuid.uuid4().hex[:8]}"
            
            transaction = Transaction(
                id=trans_id,
                capsule_id=capsule_id,
                transaction_type=data['transaction_type'],
                amount=float(data['amount']),
                description=data.get('description', ''),
                transaction_date=datetime.utcnow()
            )
            
            # Update capsule value
            capsule = db.query(Capsule).filter(Capsule.id == capsule_id).first()
            if capsule:
                if data['transaction_type'] in ['deposit', 'dividend', 'interest']:
                    capsule.current_value += float(data['amount'])
                elif data['transaction_type'] in ['withdrawal', 'fee']:
                    capsule.current_value -= float(data['amount'])
                capsule.updated_at = datetime.utcnow()
            
            db.add(transaction)
            db.commit()
            db.refresh(transaction)
            
            return jsonify(transaction.to_dict()), 201
        finally:
            db.close()

# ===== NEW ALLOCATION ENDPOINTS =====

@app.route('/api/v1/capsules/<capsule_id>/allocations', methods=['GET', 'POST'])
def allocations(capsule_id):
    if request.method == 'GET':
        db = get_db()
        try:
            allocs = db.query(Allocation).filter(
                Allocation.capsule_id == capsule_id
            ).all()
            return jsonify([a.to_dict() for a in allocs])
        finally:
            db.close()
    
    else:  # POST - Set allocation
        db = get_db()
        try:
            data = request.get_json()
            alloc_id = f"alloc_{uuid.uuid4().hex[:8]}"
            
            allocation = Allocation(
                id=alloc_id,
                capsule_id=capsule_id,
                asset_class=data['asset_class'],
                target_percentage=float(data['target_percentage']),
                current_percentage=float(data.get('current_percentage', 0)),
                current_value=float(data.get('current_value', 0))
            )
            
            db.add(allocation)
            db.commit()
            db.refresh(allocation)
            
            return jsonify(allocation.to_dict()), 201
        finally:
            db.close()

# ===== NEW REBALANCING ENDPOINTS =====

@app.route('/api/v1/capsules/<capsule_id>/rebalance/check', methods=['GET'])
def check_rebalance(capsule_id):
    """Check if rebalancing is needed"""
    needed, drift = rebalancer.check_rebalance_needed(capsule_id)
    return jsonify({
        'capsule_id': capsule_id,
        'rebalance_needed': needed,
        'max_drift': round(drift, 2),
        'threshold': rebalancer.threshold
    })

@app.route('/api/v1/capsules/<capsule_id>/rebalance', methods=['POST'])
def do_rebalance(capsule_id):
    """Execute rebalancing"""
    result = rebalancer.rebalance(capsule_id)
    if result:
        return jsonify(result), 200
    return jsonify({'error': 'Rebalance failed'}), 500

# ===== NEW PERFORMANCE ENDPOINTS =====

@app.route('/api/v1/capsules/<capsule_id>/performance', methods=['GET'])
def performance(capsule_id):
    """Get performance metrics"""
    period = request.args.get('period', 'daily')
    perf = calculate_performance(capsule_id, period)
    if perf:
        return jsonify(perf)
    return jsonify({'error': 'Insufficient data'}), 404

if __name__ == '__main__':
    logger.info("Starting Capsules Service v3.0...")
    app.run(host='0.0.0.0', port=8000, debug=True)
