import numpy as np
from datetime import datetime

class NAVComputationEngine:
    def __init__(self):
        self.nav_history = []
        self.benchmark_nav = 100.0
        
    def calculate_nav(self, portfolio, market_prices, shares_outstanding=1000000):
        # Calculate total assets
        total_assets = portfolio.get("cash", 0)
        
        for symbol, position in portfolio.get("positions", {}).items():
            if symbol in market_prices:
                total_assets += position["shares"] * market_prices[symbol]
        
        # Calculate NAV
        total_liabilities = portfolio.get("liabilities", 0)
        net_asset_value = total_assets - total_liabilities
        nav_per_share = net_asset_value / shares_outstanding
        
        # Calculate changes
        previous_nav = self.nav_history[-1]["nav_per_share"] if self.nav_history else self.benchmark_nav
        nav_change = nav_per_share - previous_nav
        nav_change_pct = (nav_change / previous_nav * 100) if previous_nav > 0 else 0
        
        nav_data = {
            "timestamp": datetime.now().isoformat(),
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "net_asset_value": net_asset_value,
            "shares_outstanding": shares_outstanding,
            "nav_per_share": nav_per_share,
            "nav_change": nav_change,
            "nav_change_pct": nav_change_pct
        }
        
        self.nav_history.append(nav_data)
        return nav_data
    
    def generate_nav_report(self, portfolio, market_prices):
        nav_data = self.calculate_nav(portfolio, market_prices)
        
        report = []
        report.append("=" * 50)
        report.append("NAV COMPUTATION REPORT")
        report.append("=" * 50)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("NAV SUMMARY:")
        report.append(f"  NAV per Share: ${nav_data['nav_per_share']:.4f}")
        report.append(f"  Change: ${nav_data['nav_change']:+.4f} ({nav_data['nav_change_pct']:+.2f}%)")
        report.append(f"  Total Net Assets: ${nav_data['net_asset_va
# Create a working NAV engine without issues
$navCode = @'
import numpy as np
from datetime import datetime

class NAVComputationEngine:
    def __init__(self):
        self.nav_history = []
        self.benchmark_nav = 100.0
        
    def calculate_nav(self, portfolio, market_prices, shares_outstanding=1000000):
        # Calculate total assets
        total_assets = portfolio.get("cash", 0)
        
        for symbol, position in portfolio.get("positions", {}).items():
            if symbol in market_prices:
                total_assets += position["shares"] * market_prices[symbol]
        
        # Calculate NAV
        total_liabilities = portfolio.get("liabilities", 0)
        net_asset_value = total_assets - total_liabilities
        nav_per_share = net_asset_value / shares_outstanding
        
        # Calculate changes
        previous_nav = self.nav_history[-1]["nav_per_share"] if self.nav_history else self.benchmark_nav
        nav_change = nav_per_share - previous_nav
        nav_change_pct = (nav_change / previous_nav * 100) if previous_nav > 0 else 0
        
        nav_data = {
            "timestamp": datetime.now().isoformat(),
            "total_assets": total_assets,
            "total_liabilities": total_liabilities,
            "net_asset_value": net_asset_value,
            "shares_outstanding": shares_outstanding,
            "nav_per_share": nav_per_share,
            "nav_change": nav_change,
            "nav_change_pct": nav_change_pct
        }
        
        self.nav_history.append(nav_data)
        return nav_data
    
    def generate_nav_report(self, portfolio, market_prices):
        nav_data = self.calculate_nav(portfolio, market_prices)
        
        report = []
        report.append("=" * 50)
        report.append("NAV COMPUTATION REPORT")
        report.append("=" * 50)
        report.append(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("NAV SUMMARY:")
        report.append(f"  NAV per Share: ${nav_data['nav_per_share']:.4f}")
        report.append(f"  Change: ${nav_data['nav_change']:+.4f} ({nav_data['nav_change_pct']:+.2f}%)")
        report.append(f"  Total Net Assets: ${nav_data['net_asset_value']:,.2f}")
        
        return "\n".join(report)

# Test
if __name__ == "__main__":
    print("NAV Computation Engine")
    print("=" * 50)
    
    nav_engine = NAVComputationEngine()
    
    # Your portfolio
    portfolio = {
        "cash": 94521.86,
        "positions": {
            "GOOGL": {"shares": 7, "avg_cost": 277.82},
            "NVDA": {"shares": 10, "avg_cost": 198.89},
            "MSFT": {"shares": 3, "avg_cost": 514.84}
        },
        "liabilities": 0
    }
    
    # Market prices
    market_prices = {
        "GOOGL": 280.50,
        "NVDA": 205.00,
        "MSFT": 510.00
    }
    
    # Calculate NAV
    nav_data = nav_engine.calculate_nav(portfolio, market_prices)
    
    print(f"\nNAV Calculation:")
    print(f"  NAV per Share: ${nav_data['nav_per_share']:.4f}")
    print(f"  Total Net Assets: ${nav_data['net_asset_value']:,.2f}")
    print(f"  Change: {nav_data['nav_change_pct']:+.2f}%")
    
    # Report
    report = nav_engine.generate_nav_report(portfolio, market_prices)
    print("\n" + report)
    
    print("\nNAV Computation Engine Ready!")
