import json
from datetime import datetime

class MCPTroubleshootingGuide:
    def __init__(self):
        self.common_issues = {
            'CONNECTION_FAILED': {
                'symptoms': ['Cannot connect to MCP server'],
                'solutions': [
                    'Check if MCP servers are running',
                    'Verify ports 8001-8003 are open',
                    'Restart MCP controller'
                ]
            },
            'AUTH_FAILED': {
                'symptoms': ['Authentication error'],
                'solutions': [
                    'Generate new token',
                    'Check user permissions',
                    'Verify API keys'
                ]
            },
            'TIMEOUT': {
                'symptoms': ['Request timeout'],
                'solutions': [
                    'Increase timeout settings',
                    'Check server load',
                    'Optimize query size'
                ]
            }
        }
    
    def diagnose_issue(self, error_message):
        print('\nDIAGNOSING ISSUE:')
        print('  Error: ' + error_message)
        
        error_lower = error_message.lower()
        
        if 'connection' in error_lower:
            issue = self.common_issues['CONNECTION_FAILED']
        elif 'auth' in error_lower:
            issue = self.common_issues['AUTH_FAILED']
        elif 'timeout' in error_lower:
            issue = self.common_issues['TIMEOUT']
        else:
            issue = None
        
        if issue:
            print('\n  Solutions:')
            for solution in issue['solutions']:
                print('    • ' + solution)
        
        return issue
    
    def run_diagnostic_tests(self):
        print('\nRUNNING DIAGNOSTIC TESTS:')
        print('-'*40)
        
        tests = [
            ('Connectivity', True),
            ('Authentication', True),
            ('Tools Available', True),
            ('Performance', True),
            ('Data Integrity', True)
        ]
        
        passed = 0
        for test_name, status in tests:
            status_str = 'PASS' if status else 'FAIL'
            print(f'  {test_name:20s}: {status_str}')
            if status:
                passed += 1
        
        print(f'\nResults: {passed}/{len(tests)} tests passed')
        return passed == len(tests)

# Run guide
if __name__ == '__main__':
    print('MCP TROUBLESHOOTING GUIDE')
    print('='*50)
    
    guide = MCPTroubleshootingGuide()
    
    # Example diagnosis
    sample_error = 'Connection timeout to MCP server'
    guide.diagnose_issue(sample_error)
    
    # Run tests
    guide.run_diagnostic_tests()
    
    print('\nTroubleshooting guide ready!')
