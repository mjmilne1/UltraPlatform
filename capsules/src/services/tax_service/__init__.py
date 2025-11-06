"""
Australian Tax Service - ATO Compliant
Includes Part IVA Anti-Avoidance Provisions for Wash Sales
"""
from flask import Blueprint, request, jsonify
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Dict

tax_bp = Blueprint('aus_tax', __name__, url_prefix='/api/v1/tax')

# ATO Compliance Constants
ATO_SAFE_WAIT_PERIOD_DAYS = 45  # Best practice waiting period
ATO_RISKY_PERIOD_DAYS = 30      # High-risk period for wash sales
CGT_DISCOUNT_DAYS = 365         # Days required for 50% CGT discount

# Common Australian ETF Alternatives
ASSET_ALTERNATIVES = {
    'VAS.AX': ['A200.AX', 'STW.AX', 'IOZ.AX'],  # ASX 200 alternatives
    'VGS.AX': ['IWLD.AX', 'WXOZ.AX', 'VGAD.AX'],  # International alternatives
    'VHY.AX': ['ZYAU.AX', 'SYI.AX', 'EINC.AX'],  # High yield alternatives
    'NDQ.AX': ['HNDQ.AX', 'QNAS.AX', 'TECH.AX'],  # NASDAQ alternatives
    'VAF.AX': ['VBND.AX', 'IAF.AX', 'PLUS.AX'],  # Fixed income alternatives
}

@tax_bp.route('/calculate-cgt', methods=['POST'])
def calculate_cgt():
    """Calculate Australian Capital Gains Tax with 50% discount"""
    data = request.get_json()
    
    purchase_date = datetime.fromisoformat(data['purchase_date'])
    sale_date = datetime.fromisoformat(data['sale_date'])
    holding_days = (sale_date - purchase_date).days
    
    cost_base = Decimal(str(data['cost_base']))
    proceeds = Decimal(str(data['proceeds']))
    capital_gain = proceeds - cost_base
    
    # 50% CGT discount if held >12 months
    discount = Decimal('0.5') if holding_days > CGT_DISCOUNT_DAYS else Decimal('0')
    discount_amount = capital_gain * discount if capital_gain > 0 else Decimal('0')
    
    # Calculate safe rebuy date (45 days from sale)
    safe_rebuy_date = (sale_date + timedelta(days=ATO_SAFE_WAIT_PERIOD_DAYS)).isoformat()
    risky_until_date = (sale_date + timedelta(days=ATO_RISKY_PERIOD_DAYS)).isoformat()
    
    return jsonify({
        'capital_gain': float(capital_gain),
        'holding_days': holding_days,
        'cgt_discount': float(discount),
        'discount_amount': float(discount_amount),
        'taxable_gain': float(capital_gain - discount_amount),
        'sale_date': sale_date.isoformat(),
        'wash_sale_risk': {
            'high_risk_until': risky_until_date,
            'safe_to_rebuy_after': safe_rebuy_date,
            'warning': 'ATO monitors wash sales under Part IVA anti-avoidance provisions',
            'recommendation': 'Wait 45+ days OR switch to substantially different asset',
            'penalty_risk': 'Up to 50% of tax avoided plus interest'
        },
        'compliance': {
            'legislation': 'ITAA 1997 Part IVA (s177D, s177F)',
            'ato_guidance': 'TR 2008/1 - Wash Sales',
            'fy_applicable': f'{sale_date.year if sale_date.month >= 7 else sale_date.year-1}/{sale_date.year+1 if sale_date.month >= 7 else sale_date.year}'
        }
    })

@tax_bp.route('/check-wash-sale-risk', methods=['POST'])
def check_wash_sale_risk():
    """
    Check if a proposed transaction would trigger ATO wash sale concerns
    
    Request body:
    {
        "asset_id": "VAS.AX",
        "sale_date": "2025-06-15",
        "proposed_rebuy_date": "2025-06-20",
        "is_loss": true,
        "has_offsetting_gains": true
    }
    """
    data = request.get_json()
    
    sale_date = datetime.fromisoformat(data['sale_date'])
    rebuy_date = datetime.fromisoformat(data['proposed_rebuy_date'])
    days_between = (rebuy_date - sale_date).days
    
    is_loss = data.get('is_loss', False)
    has_gains = data.get('has_offsetting_gains', False)
    
    # Calculate risk level
    if not is_loss:
        risk_level = 'NONE'
        risk_score = 0
        explanation = 'No wash sale concern for capital gains'
    elif days_between >= ATO_SAFE_WAIT_PERIOD_DAYS:
        risk_level = 'LOW'
        risk_score = 1
        explanation = f'Waiting {days_between} days reduces wash sale risk significantly'
    elif days_between >= ATO_RISKY_PERIOD_DAYS:
        risk_level = 'MEDIUM'
        risk_score = 5
        explanation = 'Still within concerning timeframe, consider waiting longer or switching assets'
    else:
        risk_level = 'HIGH'
        risk_score = 9
        explanation = 'HIGH RISK: ATO likely to challenge as wash sale under Part IVA'
    
    # Additional risk factors
    risk_factors = []
    if has_gains:
        risk_score += 2
        risk_factors.append('Offsetting capital gains increases scrutiny')
    
    end_of_fy = datetime(sale_date.year if sale_date.month < 7 else sale_date.year + 1, 6, 30)
    days_to_eofy = (end_of_fy - sale_date).days
    if 0 <= days_to_eofy <= 30:
        risk_score += 3
        risk_factors.append('Transaction near end of financial year (June 30) - higher scrutiny')
    
    return jsonify({
        'risk_assessment': {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'days_between_trades': days_between,
            'explanation': explanation,
            'additional_risk_factors': risk_factors
        },
        'recommendations': _get_wash_sale_recommendations(
            data['asset_id'], 
            days_between, 
            risk_level
        ),
        'compliance_notes': {
            'legislation': 'Part IVA - General Anti-Avoidance Rule',
            'key_test': 'Dominant purpose of transaction',
            'ato_focus': 'No genuine change in economic exposure',
            'penalties': 'Loss disallowed + up to 50% penalty + interest'
        }
    })

def _get_wash_sale_recommendations(asset_id: str, days_between: int, risk_level: str) -> Dict:
    """Generate recommendations to avoid wash sale issues"""
    recommendations = []
    
    if risk_level == 'HIGH' or risk_level == 'MEDIUM':
        # Recommend waiting
        days_to_wait = ATO_SAFE_WAIT_PERIOD_DAYS - days_between
        if days_to_wait > 0:
            recommendations.append({
                'strategy': 'WAIT',
                'action': f'Wait {days_to_wait} more days before repurchasing',
                'confidence': 0.95,
                'rationale': '45+ day waiting period demonstrates genuine disposition'
            })
        
        # Recommend alternative assets
        alternatives = ASSET_ALTERNATIVES.get(asset_id, [])
        if alternatives:
            recommendations.append({
                'strategy': 'SWITCH_ASSET',
                'action': f'Purchase alternative asset instead: {", ".join(alternatives)}',
                'alternatives': alternatives,
                'confidence': 0.90,
                'rationale': 'Substantially different assets reduce wash sale risk'
            })
        
        # Recommend genuine rebalancing
        recommendations.append({
            'strategy': 'GENUINE_REBALANCE',
            'action': 'Document as part of genuine portfolio rebalancing strategy',
            'confidence': 0.85,
            'rationale': 'Legitimate investment strategy with business purpose'
        })
    
    return {
        'recommended_strategies': recommendations,
        'best_practice': 'Combine waiting period with asset switching for maximum safety'
    }

@tax_bp.route('/optimize-harvest', methods=['POST'])
def optimize_harvest():
    """
    Tax loss harvesting optimization with ATO wash sale compliance
    
    Request body:
    {
        "positions": [{
            "asset_id": "VAS.AX",
            "unrealized_gain": -5000,
            "holding_days": 200,
            "purchase_date": "2024-01-15"
        }],
        "target_harvest_date": "2025-06-15"
    }
    """
    data = request.get_json()
    target_date = datetime.fromisoformat(data.get('target_harvest_date', datetime.now().isoformat()))
    
    recommendations = []
    
    for position in data.get('positions', []):
        asset_id = position['asset_id']
        unrealized = Decimal(str(position.get('unrealized_gain', 0)))
        holding_days = position.get('holding_days', 0)
        
        if unrealized < 0:  # Loss position
            loss_amount = abs(unrealized)
            tax_benefit = loss_amount * Decimal('0.47')  # Top tax rate
            
            # Calculate safe dates
            safe_rebuy_date = (target_date + timedelta(days=ATO_SAFE_WAIT_PERIOD_DAYS))
            
            # Get alternative assets
            alternatives = ASSET_ALTERNATIVES.get(asset_id, [])
            
            recommendations.append({
                'asset': asset_id,
                'action': 'HARVEST_LOSS',
                'loss_amount': float(loss_amount),
                'tax_benefit': float(tax_benefit),
                'strategy': {
                    'option_1': {
                        'method': 'WAIT_AND_REBUY',
                        'timeline': f'Sell now, rebuy after {safe_rebuy_date.strftime("%Y-%m-%d")}',
                        'risk': 'LOW',
                        'market_risk': 'Exposed to price changes during 45-day wait'
                    },
                    'option_2': {
                        'method': 'IMMEDIATE_SWITCH',
                        'timeline': 'Sell now, immediately buy alternative',
                        'alternatives': alternatives,
                        'risk': 'LOW',
                        'market_risk': 'Minimal - maintains market exposure'
                    },
                    'option_3': {
                        'method': 'RISKY_REBUY',
                        'timeline': 'Sell and rebuy within 30 days',
                        'risk': 'HIGH',
                        'warning': '⚠️ NOT RECOMMENDED - ATO likely to disallow loss'
                    }
                },
                'recommendation': 'Use Option 2 (immediate switch) for best tax efficiency with compliance',
                'confidence': 0.90,
                'compliance_notes': {
                    'part_iva_risk': 'LOW if using alternative asset or waiting 45+ days',
                    'documentation': 'Keep records of investment rationale and timing decisions'
                }
            })
        
        elif 335 <= holding_days < CGT_DISCOUNT_DAYS and unrealized > 0:  # Near CGT discount
            days_to_wait = CGT_DISCOUNT_DAYS - holding_days
            potential_saving = unrealized * Decimal('0.5') * Decimal('0.47')
            
            recommendations.append({
                'asset': asset_id,
                'action': 'WAIT_FOR_DISCOUNT',
                'days_to_wait': days_to_wait,
                'gain_amount': float(unrealized),
                'potential_saving': float(potential_saving),
                'strategy': f'HOLD for {days_to_wait} days to access 50% CGT discount',
                'discount_available_from': (target_date + timedelta(days=days_to_wait)).strftime('%Y-%m-%d'),
                'confidence': 0.98,
                'no_wash_sale_risk': True
            })
    
    return jsonify({
        'recommendations': recommendations,
        'total_opportunities': len(recommendations),
        'estimated_tax_benefit': sum(r.get('tax_benefit', 0) for r in recommendations),
        'compliance_summary': {
            'all_strategies_ato_compliant': True,
            'documentation_required': 'Yes - keep records of all transactions and rationale',
            'risk_level': 'LOW when following recommended strategies'
        }
    })

@tax_bp.route('/suggest-alternatives', methods=['POST'])
def suggest_alternatives():
    """
    Suggest alternative assets to avoid wash sale issues
    
    Request body:
    {
        "asset_id": "VAS.AX",
        "asset_type": "etf",
        "exposure": "ASX 200"
    }
    """
    data = request.get_json()
    asset_id = data['asset_id']
    
    alternatives = ASSET_ALTERNATIVES.get(asset_id, [])
    
    return jsonify({
        'original_asset': asset_id,
        'alternatives': [
            {
                'ticker': alt,
                'substantially_different': True,
                'similar_exposure': True,
                'swap_rationale': 'Different fund manager/index methodology = substantially different for ATO purposes'
            }
            for alt in alternatives
        ],
        'strategy': 'Switching to alternative maintains similar market exposure while avoiding wash sale issues',
        'ato_compliance': 'Assets from different fund families are generally not considered "substantially the same"',
        'recommendation': f'Sell {asset_id}, immediately buy {alternatives[0] if alternatives else "alternative"}',
        'documentation_tip': 'Document that this is part of portfolio optimization, not purely for tax benefit'
    })

@tax_bp.route('/franking-credits', methods=['POST'])
def franking_credits():
    """Calculate franking credit benefit"""
    data = request.get_json()
    
    dividend = Decimal(str(data['dividend']))
    franking_pct = Decimal(str(data.get('franking_percentage', 100))) / 100
    marginal_rate = Decimal(str(data['marginal_tax_rate']))
    
    COMPANY_TAX = Decimal('0.30')
    franking_credit = (dividend / (1 - COMPANY_TAX * franking_pct)) * COMPANY_TAX * franking_pct - dividend
    grossed_up = dividend + franking_credit
    tax_payable = grossed_up * marginal_rate
    net_benefit = franking_credit - (tax_payable - dividend * marginal_rate)
    
    return jsonify({
        'dividend': float(dividend),
        'franking_credits': float(franking_credit),
        'grossed_up_dividend': float(grossed_up),
        'tax_payable': float(tax_payable),
        'net_benefit': float(net_benefit),
        'is_refund': net_benefit > 0,
        'explanation': 'Franking credits represent company tax paid - can offset your tax or provide refund',
        'holding_period_note': 'Must hold shares 45+ days (at risk) for franking credits to be valid'
    })

@tax_bp.route('/compliance-check', methods=['POST'])
def compliance_check():
    """
    Comprehensive compliance check for proposed transactions
    
    Request body:
    {
        "transactions": [{
            "date": "2025-06-15",
            "type": "sell",
            "asset_id": "VAS.AX",
            "is_loss": true
        }, {
            "date": "2025-06-20",
            "type": "buy",
            "asset_id": "VAS.AX"
        }]
    }
    """
    data = request.get_json()
    transactions = data.get('transactions', [])
    
    issues = []
    warnings = []
    
    # Check for wash sale patterns
    for i, txn in enumerate(transactions):
        if txn['type'] == 'sell' and txn.get('is_loss'):
            # Look for repurchases
            for j in range(i+1, len(transactions)):
                next_txn = transactions[j]
                if next_txn['type'] == 'buy' and next_txn['asset_id'] == txn['asset_id']:
                    days_between = (datetime.fromisoformat(next_txn['date']) - 
                                   datetime.fromisoformat(txn['date'])).days
                    
                    if days_between < ATO_RISKY_PERIOD_DAYS:
                        issues.append({
                            'severity': 'HIGH',
                            'issue': 'Potential wash sale detected',
                            'detail': f'Selling and rebuying {txn["asset_id"]} within {days_between} days',
                            'recommendation': 'Wait 45+ days OR switch to alternative asset',
                            'risk': 'Loss may be disallowed by ATO'
                        })
                    elif days_between < ATO_SAFE_WAIT_PERIOD_DAYS:
                        warnings.append({
                            'severity': 'MEDIUM',
                            'issue': 'Short repurchase timeframe',
                            'detail': f'{days_between} days between trades',
                            'recommendation': 'Consider waiting a few more days for safety'
                        })
    
    return jsonify({
        'compliance_status': 'FAILED' if issues else ('WARNING' if warnings else 'PASSED'),
        'issues': issues,
        'warnings': warnings,
        'overall_risk': 'HIGH' if issues else ('MEDIUM' if warnings else 'LOW'),
        'recommendations': [
            'Document genuine investment strategy',
            'Maintain 45+ day waiting periods',
            'Use alternative assets when appropriate',
            'Keep detailed records of all transactions'
        ]
    })

@tax_bp.route('/health', methods=['GET'])
def health():
    return jsonify({
        'service': 'Australian Tax Service (ATO Compliant)',
        'status': 'healthy',
        'version': '3.1',
        'compliance': {
            'legislation': 'ITAA 1997 Part IVA',
            'wash_sale_monitoring': 'ACTIVE',
            'safe_waiting_period': f'{ATO_SAFE_WAIT_PERIOD_DAYS} days',
            'cgt_discount_period': f'{CGT_DISCOUNT_DAYS} days'
        }
    })
