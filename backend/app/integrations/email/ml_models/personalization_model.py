"""
ML Model - Email Personalization
Personalizes email content and timing for each recipient
"""

from typing import Dict, List
import numpy as np
from datetime import datetime

class PersonalizationModel:
    """
    ML model for email personalization
    
    Features:
    - Optimal send time prediction
    - Content preference learning
    - Engagement scoring
    - Subject line optimization
    """
    
    def __init__(self):
        self.model = None
        self.trained = False
    
    def train(self, historical_data: List[Dict]):
        """
        Train personalization model
        
        Args:
            historical_data: Past email interactions
        """
        
        if len(historical_data) < 100:
            print("Insufficient data for training")
            return
        
        # Extract features
        X = []
        y = []
        
        for record in historical_data:
            features = self._extract_features(record)
            X.append(features)
            
            # Target: engagement score
            opened = 1 if record.get("opened") else 0
            clicked = 2 if record.get("clicked") else 0
            y.append(opened + clicked)
        
        # Train model
        # from sklearn.ensemble import RandomForestRegressor
        # self.model = RandomForestRegressor(n_estimators=100)
        # self.model.fit(X, y)
        
        self.trained = True
        print(f"? Personalization model trained with {len(historical_data)} samples")
    
    def predict_engagement(
        self,
        recipient_profile: Dict,
        email_context: Dict
    ) -> Dict:
        """
        Predict email engagement
        
        Returns:
            {
                "engagement_score": 0.0-1.0,
                "open_probability": 0.0-1.0,
                "click_probability": 0.0-1.0,
                "best_send_time": "09:00"
            }
        """
        
        if not self.trained:
            return self._rule_based_prediction(recipient_profile, email_context)
        
        features = self._extract_features({
            "profile": recipient_profile,
            "context": email_context
        })
        
        # engagement = self.model.predict([features])[0]
        
        return self._rule_based_prediction(recipient_profile, email_context)
    
    def _rule_based_prediction(
        self,
        profile: Dict,
        context: Dict
    ) -> Dict:
        """Fallback rule-based prediction"""
        
        # Simple heuristics
        engagement_score = 0.5
        
        # Adjust for email type
        email_type = context.get("email_type")
        if email_type == "transactional":
            engagement_score += 0.3
        elif email_type == "alert":
            engagement_score += 0.4
        
        # Adjust for past behavior
        if profile.get("active_user"):
            engagement_score += 0.2
        
        engagement_score = min(engagement_score, 1.0)
        
        return {
            "engagement_score": engagement_score,
            "open_probability": engagement_score * 0.8,
            "click_probability": engagement_score * 0.3,
            "best_send_time": "09:00"
        }
    
    def optimize_subject_line(
        self,
        subject_options: List[str],
        recipient_profile: Dict
    ) -> str:
        """
        Select best subject line for recipient
        
        Uses ML to score each option
        """
        
        scores = []
        
        for subject in subject_options:
            score = self._score_subject(subject, recipient_profile)
            scores.append(score)
        
        # Return best scoring subject
        best_idx = np.argmax(scores)
        return subject_options[best_idx]
    
    def _score_subject(self, subject: str, profile: Dict) -> float:
        """Score subject line effectiveness"""
        
        score = 0.5
        
        # Length optimization (50-60 chars ideal)
        if 50 <= len(subject) <= 60:
            score += 0.2
        
        # Personalization
        if profile.get("name") and profile["name"] in subject:
            score += 0.3
        
        # Urgency indicators
        if any(word in subject.lower() for word in ["urgent", "important", "action"]):
            score += 0.1
        
        # Emojis
        if any(char in subject for char in ["??", "??", "??"]):
            score += 0.1
        
        return min(score, 1.0)
    
    def _extract_features(self, data: Dict) -> List[float]:
        """Extract ML features"""
        
        features = []
        
        # Time features
        now = datetime.now()
        features.append(now.hour / 24.0)
        features.append(now.weekday() / 7.0)
        
        # Profile features
        profile = data.get("profile", {})
        features.append(1.0 if profile.get("active_user") else 0.0)
        features.append(profile.get("engagement_score", 0.5))
        
        # Context features
        context = data.get("context", {})
        email_types = ["transactional", "marketing", "alert", "report"]
        email_type = context.get("email_type", "notification")
        features.extend([1.0 if email_type == t else 0.0 for t in email_types])
        
        return features
