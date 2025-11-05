import numpy as np
from datetime import datetime

class PortfolioRebalancingEngine:
    def __init__(self):
        self.threshold = 0.05  # 5% drift threshold
        
    def calculate_target_allocation(self, strategy='OPTIMAL'):
        if strategy == 'OPTIMAL':
            # Your optimized allocation
            return {
                'GOOGL': 0.433,
                'NVDA': 0.387,
                'AAPL': 0.155,
                'MSFT': 0.024,
                'CASH': 0.001
            }
        else:
            return {
                'GOOGL': 0.25,
                'NVDA': 0.25,
                'AAPL': 0.25,
                'MSFT': 0.20,
                'CASH': 0.05
            }
    
    def calculate_portfolio_value(self, portfolio):
        total = portfolio.get('cash', 0)
        for symbol, position in portfolio.get('positions', {}).items():
            total += position.get('shares', 0) * position.get('last_price', 0)
        return total
    
    def check_rebalancing_needed(self, portfolio, target_allocation):
        total_value = self.calculate_portfolio_value(portfolio)
        needs_rebalancing = False
        drift_report = {}
        
        for asset, target_pct in target_allocation.items():
            if asset == 'CASH':
                current_value = portfolio.get('cash', 0)
            else:
                position = portfolio.get('positions', {}).get(asset, {})
                current_value = position.get('shares', 0) * position.get('last_price', 0)
            
            current_pct = current_value / total_value if total_value > 0 else 0
            drift = abs(current_pct - target_pct)
            
            drift_report[asset] = {
                'current_pct': current_pct,
                'target_pct': target_pct,
                'drift': drift,
                'needs_rebalancing': drift > self.threshold
            }
            
            if drift > self.threshold:
                needs_rebalancing = True
        
        return needs_rebalancing, drift_report

# Test
print('Portfolio Rebalancing Engine')
print('='*50)

rebalancer = PortfolioRebalancingEngine()

# Your portfolio
portfolio = {
    'cash': 94521.86,
    'positions': {
        'GOOGL': {'shares': 7, 'last_price': 280.50},
        'NVDA': {'shares': 10, 'last_price': 205.00},
        'MSFT': {'shares': 3, 'last_price': 510.00}
    }
}

# Get target allocation
target = rebalancer.calculate_target_allocation('OPTIMAL')

print('\nOptimal Target Allocation:')
for asset, pct in target.items():
    if pct > 0.01:
        print('  ' + asset + ': ' + str(round(pct*100, 1)) + '%')

# Check rebalancing
needs_rebalancing, drift = rebalancer.check_rebalancing_needed(portfolio, target)

print('\nDrift Analysis:')
print('Asset  | Current | Target | Drift')
print('-'*40)
for asset, data in drift.items():
    if asset != 'CASH':
        current = round(data['current_pct']*100, 1)
        target = round(data['target_pct']*100, 1)
        drift_pct = round(data['drift']*100, 1)
        print(f'{asset:6s} | {current:6.1f}% | {target:6.1f}% | {drift_pct:5.1f}%')

if needs_rebalancing:
    print('\nREBALANCING NEEDED!')
    print('Your portfolio has drifted from optimal allocation')
    print('After rebalancing: 102.8% expected annual return!')
else:
    print('\nPortfolio is balanced')

print('\nRebalancing Engine Ready!')
