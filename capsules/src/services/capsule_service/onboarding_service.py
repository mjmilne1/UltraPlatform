from flask import Blueprint, request, jsonify
import uuid

onboarding_bp = Blueprint('onboarding', __name__, url_prefix='/api/v1/onboarding')

@onboarding_bp.route('/start', methods=['POST'])
def start_onboarding():
    session_id = f'session_{uuid.uuid4().hex[:12]}'
    return jsonify({
        'session_id': session_id,
        'status': 'started',
        'current_step': 1,
        'total_steps': 7,
        'message': 'Onboarding started successfully'
    }), 201

@onboarding_bp.route('/<session_id>/risk-assessment', methods=['POST'])
def risk_assessment(session_id):
    data = request.get_json() or {}
    r1 = data.get('r1_score', 3)
    r2 = data.get('r2_score', 3)
    r3 = data.get('r3_score', 3)
    r4 = data.get('r4_score', 3)
    r5 = data.get('r5_score', 3)
    
    total = r1 + r2 + (6 - r3) + r4 + (6 - r5)
    
    # Map to risk tolerance band (5 bands)
    if total_score <= 10:
        tolerance_band = 'Very Conservative'
    elif total_score <= 14:
        tolerance_band = 'Conservative'
    elif total_score <= 17:
        tolerance_band = 'Balanced'
    elif total_score <= 21:
        tolerance_band = 'Growth'
    else:
        tolerance_band = 'Aggressive'

@onboarding_bp.route('/<session_id>/status', methods=['GET'])
def get_status(session_id):
    return jsonify({
        'session_id': session_id,
        'status': 'in_progress',
        'current_step': 3,
        'total_steps': 7
    })
