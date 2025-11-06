import hashlib
import jwt
import json
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional
from enum import Enum

class SecurityLevel(Enum):
    PUBLIC = 0
    AUTHENTICATED = 1
    TRADER = 2
    RISK_MANAGER = 3
    ADMIN = 4
    SUPER_ADMIN = 5

class SecurityFramework:
    '''Enterprise Security Framework for UltraPlatform'''
    
    def __init__(self):
        self.name = 'UltraPlatform Security Framework'
        self.version = '2.0'
        self.secret_key = 'ultra-secret-key-change-in-production'
        self.active_sessions = {}
        self.failed_attempts = {}
        self.max_attempts = 5
        
    # Authentication
    def authenticate_user(self, username, password):
        '''Authenticate user credentials'''
        # Check failed attempts
        if self.failed_attempts.get(username, 0) >= self.max_attempts:
            return {
                'success': False,
                'error': 'Account locked due to multiple failed attempts'
            }
        
        # Hash password (simplified - use bcrypt in production)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Check credentials (simplified)
        valid_users = {
            'trader': {'password_hash': hashlib.sha256('password123'.encode()).hexdigest(), 'role': 'TRADER'},
            'admin': {'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(), 'role': 'ADMIN'}
        }
        
        if username in valid_users:
            if valid_users[username]['password_hash'] == password_hash:
                # Generate token
                token = self.generate_token(username, valid_users[username]['role'])
                
                # Reset failed attempts
                self.failed_attempts[username] = 0
                
                return {
                    'success': True,
                    'token': token,
                    'role': valid_users[username]['role']
                }
        
        # Track failed attempt
        self.failed_attempts[username] = self.failed_attempts.get(username, 0) + 1
        
        return {
            'success': False,
            'error': 'Invalid credentials'
        }
    
    def generate_token(self, username, role):
        '''Generate JWT token'''
        payload = {
            'username': username,
            'role': role,
            'exp': datetime.now(UTC) + timedelta(hours=8),
            'iat': datetime.now(UTC)
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        # Store session
        self.active_sessions[username] = {
            'token': token,
            'login_time': datetime.now(),
            'role': role
        }
        
        return token
    
    # Authorization
    def authorize_action(self, token, action, resource):
        '''Authorize user action'''
        try:
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            username = payload['username']
            role = payload['role']
            
            # Check permissions
            permissions = self.get_role_permissions(role)
            
            if action in permissions.get(resource, []):
                return {
                    'authorized': True,
                    'username': username,
                    'role': role
                }
            else:
                return {
                    'authorized': False,
                    'reason': 'Insufficient permissions'
                }
                
        except jwt.ExpiredSignatureError:
            return {
                'authorized': False,
                'reason': 'Token expired'
            }
        except:
            return {
                'authorized': False,
                'reason': 'Invalid token'
            }
    
    def get_role_permissions(self, role):
        '''Get permissions for role'''
        permissions = {
            'TRADER': {
                'portfolio': ['read'],
                'trading': ['read', 'execute'],
                'analytics': ['read']
            },
            'RISK_MANAGER': {
                'portfolio': ['read'],
                'trading': ['read'],
                'analytics': ['read'],
                'risk': ['read', 'modify']
            },
            'ADMIN': {
                'portfolio': ['read', 'write'],
                'trading': ['read', 'execute', 'cancel'],
                'analytics': ['read', 'write'],
                'risk': ['read', 'modify'],
                'users': ['read', 'write']
            }
        }
        
        return permissions.get(role, {})
    
    # Risk Controls
    def check_trading_limits(self, username, trade):
        '''Check if trade exceeds limits'''
        limits = {
            'max_trade_size': 100000,
            'max_daily_trades': 100,
            'max_position_size': 500000,
            'max_daily_loss': 50000
        }
        
        # Check trade size
        trade_value = trade.get('quantity', 0) * trade.get('price', 0)
        
        if trade_value > limits['max_trade_size']:
            return {
                'allowed': False,
                'reason': f'Trade exceeds size limit: '
            }
        
        # Check other limits...
        
        return {
            'allowed': True
        }
    
    # Audit
    def log_activity(self, username, action, resource, result):
        '''Log user activity for audit'''
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'username': username,
            'action': action,
            'resource': resource,
            'result': result,
            'ip_address': '127.0.0.1',  # Would get real IP
            'session_id': self.active_sessions.get(username, {}).get('token', '')[:20]
        }
        
        # Would write to secure audit log
        print(f'AUDIT: {log_entry}')
        
        return log_entry
    
    # Data Protection
    def encrypt_sensitive_data(self, data):
        '''Encrypt sensitive data'''
        # Simplified - use proper encryption in production
        import base64
        encrypted = base64.b64encode(json.dumps(data).encode()).decode()
        return encrypted
    
    def decrypt_sensitive_data(self, encrypted_data):
        '''Decrypt sensitive data'''
        import base64
        decrypted = json.loads(base64.b64decode(encrypted_data.encode()).decode())
        return decrypted

# Test security framework
print('Security Framework')
print('='*50)

security = SecurityFramework()

# Test authentication
print('\n1. Authentication Test:')
result = security.authenticate_user('trader', 'password123')
if result['success']:
    print(f'   ✅ Authenticated as {result["role"]}')
    token = result['token']
    print(f'   Token: {token[:30]}...')
else:
    print(f'   ❌ {result["error"]}')

# Test authorization
print('\n2. Authorization Test:')
auth = security.authorize_action(token, 'execute', 'trading')
if auth['authorized']:
    print(f'   ✅ Authorized for trading')
else:
    print(f'   ❌ {auth["reason"]}')

# Test risk controls
print('\n3. Risk Control Test:')
trade = {'quantity': 100, 'price': 500, 'symbol': 'GOOGL'}
risk_check = security.check_trading_limits('trader', trade)
if risk_check['allowed']:
    print(f'   ✅ Trade within limits')
else:
    print(f'   ❌ {risk_check["reason"]}')

# Log activity
print('\n4. Audit Log:')
security.log_activity('trader', 'execute_trade', 'GOOGL', 'success')

print('\n✅ Security Framework operational!')
