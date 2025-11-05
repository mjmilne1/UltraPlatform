from datetime import datetime

class CorporateActionsProcessor:
    def __init__(self):
        self.action_types = ['DIVIDEND', 'STOCK_SPLIT', 'MERGER']
        self.processed_actions = []
        
    def process_dividend(self, portfolio, symbol, dividend_per_share):
        if symbol in portfolio.get('positions', {}):
            shares = portfolio['positions'][symbol]['shares']
            payment = shares * dividend_per_share
            return {'symbol': symbol, 'payment': payment, 'shares': shares}
        return None
    
    def process_stock_split(self, portfolio, symbol, split_ratio):
        if symbol in portfolio.get('positions', {}):
            shares = portfolio['positions'][symbol]['shares']
            ratio_parts = split_ratio.split(':')
            multiplier = float(ratio_parts[0]) / float(ratio_parts[1])
            new_shares = int(shares * multiplier)
            return {'symbol': symbol, 'before': shares, 'after': new_shares}
        return None

# Test
print('Corporate Actions Processor')
print('='*50)

processor = CorporateActionsProcessor()

portfolio = {
    'cash': 94521.86,
    'positions': {
        'GOOGL': {'shares': 7, 'avg_cost': 277.82},
        'NVDA': {'shares': 10, 'avg_cost': 198.89},
        'MSFT': {'shares': 3, 'avg_cost': 514.84}
    }
}

# Process dividend
print('\nMSFT Dividend (.75/share):')
result = processor.process_dividend(portfolio, 'MSFT', 0.75)
if result:
    print(f'  Shares: {result["shares"]}')
    print(f'  Payment: ')

# Process split
print('\nNVDA Stock Split (4:1):')
result = processor.process_stock_split(portfolio, 'NVDA', '4:1')
if result:
    print(f'  Before: {result["before"]} shares')
    print(f'  After: {result["after"]} shares')

print('\nCorporate Actions Ready!')
