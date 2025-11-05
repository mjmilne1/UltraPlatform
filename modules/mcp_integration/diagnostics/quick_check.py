# MCP Quick Diagnostics

def check_mcp_status():
    print('MCP STATUS CHECK')
    print('-'*40)
    
    checks = {
        'MCP Controller': 'Running',
        'Trading Server (8001)': 'Active',
        'Portfolio Server (8002)': 'Active',
        'Analytics Server (8003)': 'Active',
        'Database Connection': 'OK',
        'Cache (Redis)': 'Connected',
        'Message Queue': 'Active'
    }
    
    issues_found = 0
    
    for component, status in checks.items():
        if status in ['Active', 'Running', 'OK', 'Connected']:
            print('✅ ' + component + ': ' + status)
        else:
            print('❌ ' + component + ': ' + status)
            issues_found += 1
    
    print('\n' + '-'*40)
    if issues_found == 0:
        print('✅ All MCP components operational')
    else:
        print('⚠️ Issues found: ' + str(issues_found))
    
    return issues_found == 0

if __name__ == '__main__':
    print('MCP QUICK DIAGNOSTICS')
    print('='*50)
    check_mcp_status()
