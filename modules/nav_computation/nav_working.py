print('NAV Computation Engine')
print('='*50)

# Portfolio data
cash = 94521.86
positions = {
    'GOOGL': {'shares': 7, 'price': 280.50},
    'NVDA': {'shares': 10, 'price': 205.00},
    'MSFT': {'shares': 3, 'price': 510.00}
}

# Calculate values
positions_value = 0
print('\nPositions:')
for symbol, data in positions.items():
    value = data['shares'] * data['price']
    positions_value += value
    print('  ' + symbol + ': ' + str(data['shares']) + ' shares = $' + str(value))

total_assets = cash + positions_value
shares_outstanding = 1000000
nav_per_share = total_assets / shares_outstanding

print('\nNAV Summary:')
print('  Cash: $' + str(cash))
print('  Positions Value: $' + str(positions_value))
print('  Total Assets: $' + str(total_assets))
print('  Shares Outstanding: ' + str(shares_outstanding))
print('  NAV per Share: $' + str(round(nav_per_share, 4)))

print('\nNAV Computation Complete!')
