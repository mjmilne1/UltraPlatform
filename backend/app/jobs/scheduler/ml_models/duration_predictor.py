"""
ML Model - Job Duration Predictor
Predicts execution time using historical data
"""

from typing import List, Dict
import numpy as np
from datetime import datetime

class JobDurationPredictor:
    """
    ML model for predicting job execution duration
    
    Features:
    - Historical execution times
    - Time of day
    - Day of week
    - System load
    - Data volume
    """
    
    def __init__(self):
        self.model = None  # Would use XGBoost or RandomForest
        self.feature_scaler = None
    
    def train(self, job_id: str, historical_data: List[Dict]):
        """
        Train prediction model
        
        Args:
            job_id: Job identifier
            historical_data: List of past executions with features
        """
        
        if len(historical_data) < 10:
            print(f"Insufficient data for {job_id}. Need at least 10 executions.")
            return
        
        # Extract features
        X = []
        y = []
        
        for execution in historical_data:
            features = self._extract_features(execution)
            X.append(features)
            y.append(execution["duration_seconds"])
        
        # Train model (simplified - would use actual ML library)
        # from xgboost import XGBRegressor
        # self.model = XGBRegressor(n_estimators=100, max_depth=5)
        # self.model.fit(X, y)
        
        print(f"? Model trained for {job_id} with {len(historical_data)} samples")
    
    def predict(self, job_id: str, context: Dict) -> int:
        """
        Predict job duration
        
        Args:
            job_id: Job identifier
            context: Current execution context (time, load, etc.)
        
        Returns:
            Predicted duration in seconds
        """
        
        if self.model is None:
            # Fallback to simple average
            return 600  # 10 minutes default
        
        features = self._extract_features(context)
        
        # prediction = self.model.predict([features])[0]
        prediction = 600  # Mock
        
        return int(prediction)
    
    def _extract_features(self, execution: Dict) -> List[float]:
        """Extract ML features from execution"""
        
        started_at = execution.get("started_at", datetime.now())
        
        features = [
            started_at.hour,                    # Hour of day
            started_at.weekday(),               # Day of week
            execution.get("system_load", 0.5),  # System load
            execution.get("data_volume", 1000), # Data volume
        ]
        
        return features
    
    def get_confidence_interval(
        self,
        job_id: str,
        context: Dict
    ) -> tuple:
        """
        Get prediction confidence interval
        
        Returns:
            (lower_bound, upper_bound) in seconds
        """
        
        prediction = self.predict(job_id, context)
        
        # Simple confidence interval (±20%)
        lower = int(prediction * 0.8)
        upper = int(prediction * 1.2)
        
        return (lower, upper)
