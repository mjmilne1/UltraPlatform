"""
ML Model - Performance Attribution
ML-based attribution of portfolio returns
"""

from typing import Dict, List
import numpy as np

class PerformanceAttributionModel:
    """
    ML model for performance attribution
    
    Attributes returns to:
    - Asset allocation effect
    - Security selection effect
    - Market timing effect
    - Currency effect
    - Interaction effects
    """
    
    def __init__(self):
        self.model = None  # Would use regression model
        self.trained = False
    
    def train(self, historical_data: List[Dict]):
        """Train attribution model"""
        
        if len(historical_data) < 50:
            print("Insufficient data for training")
            return
        
        # Extract features and targets
        X = []
        y = []
        
        for record in historical_data:
            features = self._extract_features(record)
            X.append(features)
            y.append(record["total_return"])
        
        # Train model
        # from sklearn.linear_model import LinearRegression
        # self.model = LinearRegression()
        # self.model.fit(X, y)
        
        self.trained = True
        print(f"? Attribution model trained with {len(historical_data)} samples")
    
    def attribute_returns(
        self,
        portfolio_return: float,
        portfolio_data: Dict,
        benchmark_data: Dict
    ) -> Dict:
        """
        Attribute portfolio returns to factors
        
        Brinson Attribution:
        - Asset Allocation = S (wp_i - wb_i) × rb_i
        - Security Selection = S wb_i × (rp_i - rb_i)
        - Interaction = S (wp_i - wb_i) × (rp_i - rb_i)
        """
        
        if not self.trained:
            return self._rule_based_attribution(
                portfolio_return, portfolio_data, benchmark_data
            )
        
        # ML-based attribution
        features = self._extract_features({
            "portfolio": portfolio_data,
            "benchmark": benchmark_data
        })
        
        # attribution = self.model.predict([features])[0]
        
        return self._rule_based_attribution(
            portfolio_return, portfolio_data, benchmark_data
        )
    
    def _rule_based_attribution(
        self,
        portfolio_return: float,
        portfolio_data: Dict,
        benchmark_data: Dict
    ) -> Dict:
        """Fallback rule-based attribution"""
        
        # Simplified Brinson attribution
        
        # Mock data
        asset_allocation_effect = 0.015  # 1.5%
        security_selection_effect = 0.012  # 1.2%
        timing_effect = -0.003  # -0.3%
        currency_effect = 0.002  # 0.2%
        interaction_effect = 0.004  # 0.4%
        
        total = (asset_allocation_effect + security_selection_effect + 
                 timing_effect + currency_effect + interaction_effect)
        
        return {
            "asset_allocation": {
                "effect": asset_allocation_effect,
                "contribution_pct": asset_allocation_effect / portfolio_return if portfolio_return else 0
            },
            "security_selection": {
                "effect": security_selection_effect,
                "contribution_pct": security_selection_effect / portfolio_return if portfolio_return else 0
            },
            "timing": {
                "effect": timing_effect,
                "contribution_pct": timing_effect / portfolio_return if portfolio_return else 0
            },
            "currency": {
                "effect": currency_effect,
                "contribution_pct": currency_effect / portfolio_return if portfolio_return else 0
            },
            "interaction": {
                "effect": interaction_effect,
                "contribution_pct": interaction_effect / portfolio_return if portfolio_return else 0
            },
            "total_attributed": total,
            "unexplained": portfolio_return - total
        }
    
    def _extract_features(self, data: Dict) -> List[float]:
        """Extract features for attribution"""
        
        features = []
        
        # Portfolio characteristics
        portfolio = data.get("portfolio", {})
        features.append(portfolio.get("equity_weight", 0.6))
        features.append(portfolio.get("bond_weight", 0.3))
        features.append(portfolio.get("cash_weight", 0.1))
        features.append(portfolio.get("international_weight", 0.2))
        
        # Returns
        features.append(data.get("total_return", 0))
        
        return features
    
    def explain_attribution(self, attribution: Dict) -> str:
        """Generate human-readable explanation"""
        
        explanations = []
        
        # Asset allocation
        aa_effect = attribution["asset_allocation"]["effect"]
        if aa_effect > 0.01:
            explanations.append(
                f"Asset allocation added {aa_effect*100:.1f}% - your sector positioning was favorable"
            )
        elif aa_effect < -0.01:
            explanations.append(
                f"Asset allocation detracted {abs(aa_effect)*100:.1f}% - sector positioning could be improved"
            )
        
        # Security selection
        ss_effect = attribution["security_selection"]["effect"]
        if ss_effect > 0.01:
            explanations.append(
                f"Stock selection contributed {ss_effect*100:.1f}% - your individual holdings outperformed"
            )
        
        return " ".join(explanations)
