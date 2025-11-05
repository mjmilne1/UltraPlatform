import numpy as np
from datetime import datetime

class NAVComputationEngine:
    def __init__(self):
        self.nav_history = []
        self.benchmark_nav = 100.0
        
    def calculate_nav(self, portfolio, market_prices, shares=1000000):
        # Calculate total assets
        cash = portfolio.get('cash', 0)
        positions_value = 0
        
        for symbol, position in portfolio.get('positions', {}).items():
            if symbol in market_prices:
                positions_value += position['shares'] * market_prices[symbol]
        
        total_assets = cash + positions_value
        liabilities = portfolio.get('liabilities', 0)
        net_value = total_assets - liabilities
        nav_per_share = net_value / shares
        
        return {
            'nav_per_share': nav_per_share,
            'net_value': net_value,
            'total_assets': total_assets
        }

# Test
print('NAV Computation Engine')
print('='*50)

nav_engine = NAVComputationEngine()

# Your portfolio
portfolio = {
    'cash': 94521.86,
    'positions': {
        'GOOGL': {'shares': 7, 'avg_cost': 277.82},
        'NVDA': {'shares': 10, 'avg_cost': 198.89},
        'MSFT': {'shares': 3, 'avg_cost': 514.84}
    },
    'liabilities': 0
}

# Market prices
market_prices = {
    'GOOGL': 280.50,
    'NVDA': 205.00,
    'MSFT': 510.00
}

# Calculate NAV
result = nav_engine.calculate_nav(portfolio, market_prices)

print('\nNAV Calculation:')
print('  NAV per Share: '.format(result['nav_per_share']))
print('  Total Net Assets: '.format(result['net_value']))
print('  Total Assets: '.format(result['total_assets']))

print('\nNAV Computation Ready!')
