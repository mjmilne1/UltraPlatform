"""
Capsules Service - Flask REST API
Institutional-grade portfolio management
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# In-memory storage (will add database later)
capsules_db = {}

@app.route('/')
def home():
    return jsonify({
        'service': 'Capsules Platform',
        'version': '1.0.0',
        'status': 'operational',
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/api/v1/capsules', methods=['GET'])
def list_capsules():
    """List all capsules"""
    client_id = request.args.get('client_id')
    
    if client_id:
        filtered = {k: v for k, v in capsules_db.items() if v.get('client_id') == client_id}
        return jsonify(list(filtered.values()))
    
    return jsonify(list(capsules_db.values()))

@app.route('/api/v1/capsules/<capsule_id>', methods=['GET'])
def get_capsule(capsule_id):
    """Get a specific capsule"""
    capsule = capsules_db.get(capsule_id)
    
    if not capsule:
        return jsonify({'error': 'Capsule not found'}), 404
    
    return jsonify(capsule)

@app.route('/api/v1/capsules', methods=['POST'])
def create_capsule():
    """Create a new capsule"""
    data = request.get_json()
    
    # Validate required fields
    required = ['client_id', 'capsule_type', 'goal_amount', 'goal_date']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
    
    # Generate ID
    capsule_id = f"cap_{len(capsules_db) + 1}"
    
    # Create capsule
    capsule = {
        'id': capsule_id,
        'client_id': data['client_id'],
        'capsule_type': data['capsule_type'],
        'goal_amount': data['goal_amount'],
        'goal_date': data['goal_date'],
        'current_value': 0.0,
        'status': 'created',
        'created_at': datetime.utcnow().isoformat(),
        'updated_at': datetime.utcnow().isoformat()
    }
    
    capsules_db[capsule_id] = capsule
    
    logger.info(f"Created capsule: {capsule_id} for client: {data['client_id']}")
    
    return jsonify(capsule), 201

@app.route('/api/v1/capsules/<capsule_id>', methods=['PUT'])
def update_capsule(capsule_id):
    """Update a capsule"""
    capsule = capsules_db.get(capsule_id)
    
    if not capsule:
        return jsonify({'error': 'Capsule not found'}), 404
    
    data = request.get_json()
    
    # Update fields
    for key in ['goal_amount', 'goal_date', 'current_value', 'status']:
        if key in data:
            capsule[key] = data[key]
    
    capsule['updated_at'] = datetime.utcnow().isoformat()
    
    logger.info(f"Updated capsule: {capsule_id}")
    
    return jsonify(capsule)

@app.route('/api/v1/capsules/<capsule_id>', methods=['DELETE'])
def delete_capsule(capsule_id):
    """Delete a capsule"""
    if capsule_id not in capsules_db:
        return jsonify({'error': 'Capsule not found'}), 404
    
    del capsules_db[capsule_id]
    
    logger.info(f"Deleted capsule: {capsule_id}")
    
    return '', 204

if __name__ == '__main__':
    logger.info("Starting Capsules Service...")
    app.run(host='0.0.0.0', port=8000, debug=True)
