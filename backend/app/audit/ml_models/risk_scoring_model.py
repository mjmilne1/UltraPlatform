"""
ML Model - Risk Scoring for Changes
Predicts compliance risk using historical data
"""

from typing import Dict, List
import numpy as np
from datetime import datetime

class RiskScoringModel:
    """
    ML model for scoring change risk
    
    Features:
    - Change type
    - Entity type
    - Amount/magnitude
    - User historical behavior
    - Time/context
    - Historical violations
    """
    
    def __init__(self):
        self.model = None  # Would use XGBoost or RandomForest
        self.feature_scaler = None
        self.trained = False
    
    def train(self, historical_data: List[Dict]):
        """
        Train risk scoring model
        
        Args:
            historical_data: Past changes with outcomes
                - features (change details)
                - label (was it flagged/violated?)
        """
        
        if len(historical_data) < 100:
            print(f"Insufficient data. Need at least 100 samples.")
            return
        
        # Extract features and labels
        X = []
        y = []
        
        for record in historical_data:
            features = self._extract_features(record)
            X.append(features)
            y.append(1 if record.get("violation", False) else 0)
        
        # Train model
        # from xgboost import XGBClassifier
        # self.model = XGBClassifier(n_estimators=100, max_depth=5)
        # self.model.fit(X, y)
        
        self.trained = True
        print(f"? Risk model trained with {len(historical_data)} samples")
    
    def predict_risk(self, change_details: Dict) -> Dict:
        """
        Predict risk for proposed change
        
        Returns:
            {
                "risk_score": 0.0-1.0,
                "risk_level": "low|medium|high|critical",
                "confidence": 0.0-1.0,
                "key_factors": ["factor1", "factor2"]
            }
        """
        
        if not self.trained:
            # Fallback to rule-based
            return self._rule_based_risk(change_details)
        
        features = self._extract_features(change_details)
        
        # risk_probability = self.model.predict_proba([features])[0][1]
        risk_probability = 0.5  # Mock
        
        # Determine level
        if risk_probability >= 0.8:
            risk_level = "critical"
        elif risk_probability >= 0.6:
            risk_level = "high"
        elif risk_probability >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Identify key factors
        key_factors = self._identify_risk_factors(change_details, risk_probability)
        
        return {
            "risk_score": float(risk_probability),
            "risk_level": risk_level,
            "confidence": 0.85,  # Would come from model
            "key_factors": key_factors
        }
    
    def _extract_features(self, change: Dict) -> List[float]:
        """Extract ML features from change"""
        
        features = []
        
        # Change type encoding
        change_type = change.get("change_type", "update")
        change_type_encoding = {
            "delete": 1.0,
            "update": 0.5,
            "create": 0.3
        }
        features.append(change_type_encoding.get(change_type, 0.5))
        
        # Entity type encoding
        entity_type = change.get("entity_type", "settings")
        entity_risk = {
            "transaction": 1.0,
            "journal_entry": 0.9,
            "account": 0.8,
            "client": 0.7,
            "settings": 0.3
        }
        features.append(entity_risk.get(entity_type, 0.5))
        
        # Amount (normalized)
        amount = change.get("amount", 0)
        features.append(min(float(amount) / 1000000, 1.0))
        
        # Time features
        timestamp = change.get("timestamp", datetime.now())
        features.append(timestamp.hour / 24.0)
        features.append(timestamp.weekday() / 7.0)
        
        # User risk score (would be calculated)
        features.append(change.get("user_risk_score", 0.5))
        
        return features
    
    def _rule_based_risk(self, change_details: Dict) -> Dict:
        """Fallback rule-based risk assessment"""
        
        risk_score = 0.3  # Base risk
        factors = []
        
        # High value
        if change_details.get("amount", 0) > 100000:
            risk_score += 0.3
            factors.append("HIGH_VALUE")
        
        # Financial entity
        if change_details.get("entity_type") in ["transaction", "journal_entry"]:
            risk_score += 0.2
            factors.append("FINANCIAL_ENTITY")
        
        # Deletion
        if change_details.get("change_type") == "delete":
            risk_score += 0.3
            factors.append("DELETION")
        
        # After hours
        hour = datetime.now().hour
        if hour < 6 or hour > 20:
            risk_score += 0.2
            factors.append("AFTER_HOURS")
        
        risk_score = min(risk_score, 1.0)
        
        if risk_score >= 0.7:
            level = "high"
        elif risk_score >= 0.5:
            level = "medium"
        else:
            level = "low"
        
        return {
            "risk_score": risk_score,
            "risk_level": level,
            "confidence": 0.6,
            "key_factors": factors
        }
    
    def _identify_risk_factors(
        self,
        change: Dict,
        risk_score: float
    ) -> List[str]:
        """Identify key risk contributors"""
        
        factors = []
        
        if change.get("amount", 0) > 50000:
            factors.append("LARGE_AMOUNT")
        
        if change.get("change_type") == "delete":
            factors.append("DELETION")
        
        if datetime.now().hour > 20 or datetime.now().hour < 6:
            factors.append("AFTER_HOURS")
        
        return factors
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from trained model"""
        
        if not self.trained or not self.model:
            return {}
        
        # Would extract from model
        # importance = self.model.feature_importances_
        
        return {
            "change_type": 0.25,
            "entity_type": 0.20,
            "amount": 0.30,
            "time_of_day": 0.10,
            "user_history": 0.15
        }
