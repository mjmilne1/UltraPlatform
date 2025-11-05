print('💰 P&L Engine Test')
print('='*50)

# Test with your actual positions
positions = {
    'GOOGL': {'shares': 7, 'avg_cost': 277.82},
    'NVDA': {'shares': 10, 'avg_cost': 198.89},
    'MSFT': {'shares': 3, 'avg_cost': 514.84}
}

# Current prices (example)
current_prices = {
    'GOOGL': 280.50,
    'NVDA': 205.00,
    'MSFT': 510.00
}

# Calculate simple P&L
total_pnl = 0
for symbol, pos in positions.items():
    cost = pos['shares'] * pos['avg_cost']
    value = pos['shares'] * current_prices[symbol]
    pnl = value - cost
    pct = (pnl / cost) * 100
    print(f'{symbol}:  ({pct:+.2f}%)')
    total_pnl += pnl

print(f'\nTotal P&L: ')
