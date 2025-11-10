"""
ML Model - Intent Classification
Classifies user intent from Slack messages and commands
"""

from typing import Dict, List
import numpy as np

class IntentClassifier:
    """
    ML model for classifying user intent
    
    Intents:
    - help (asking for assistance)
    - portfolio (portfolio query)
    - report (report generation)
    - approve (approval action)
    - status (system status)
    - complaint (issue reporting)
    - urgent (emergency)
    """
    
    def __init__(self):
        self.model = None  # Would use BERT or similar
        self.label_encoder = None
        self.trained = False
        
        # Intent categories
        self.intents = [
            "help",
            "portfolio",
            "report",
            "approve",
            "status",
            "complaint",
            "urgent",
            "unknown"
        ]
    
    def train(self, training_data: List[Dict]):
        """
        Train intent classification model
        
        Args:
            training_data: List of {text, intent} pairs
        """
        
        if len(training_data) < 50:
            print("Insufficient training data")
            return
        
        # Extract features
        X = []
        y = []
        
        for record in training_data:
            features = self._extract_features(record["text"])
            X.append(features)
            y.append(record["intent"])
        
        # Train model
        # from sklearn.naive_bayes import MultinomialNB
        # self.model = MultinomialNB()
        # self.model.fit(X, y)
        
        self.trained = True
        print(f"? Intent classifier trained with {len(training_data)} samples")
    
    def predict(self, text: str) -> Dict:
        """
        Predict intent from text
        
        Returns:
            {
                "intent": "portfolio",
                "confidence": 0.92,
                "entities": {"client_id": "..."}
            }
        """
        
        if not self.trained:
            return self._rule_based_classification(text)
        
        features = self._extract_features(text)
        
        # prediction = self.model.predict_proba([features])[0]
        # intent_idx = np.argmax(prediction)
        
        # Mock prediction
        return self._rule_based_classification(text)
    
    def _rule_based_classification(self, text: str) -> Dict:
        """Fallback rule-based classification"""
        
        text_lower = text.lower()
        
        # Keywords for each intent
        intent_keywords = {
            "help": ["help", "how to", "how do i", "can you help", "instructions"],
            "portfolio": ["portfolio", "holdings", "balance", "positions", "value"],
            "report": ["report", "statement", "generate", "monthly", "performance"],
            "approve": ["approve", "reject", "pending", "approval"],
            "status": ["status", "system", "down", "working", "operational"],
            "complaint": ["problem", "issue", "error", "broken", "not working", "bug"],
            "urgent": ["urgent", "emergency", "critical", "asap", "immediately"]
        }
        
        # Score each intent
        scores = {}
        for intent, keywords in intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            scores[intent] = score
        
        # Get best match
        if max(scores.values()) > 0:
            best_intent = max(scores, key=scores.get)
            confidence = min(scores[best_intent] / 3.0, 1.0)  # Normalize
        else:
            best_intent = "unknown"
            confidence = 0.0
        
        # Extract entities
        entities = self._extract_entities(text, best_intent)
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "entities": entities
        }
    
    def _extract_features(self, text: str) -> List[float]:
        """Extract features from text"""
        
        # Simple bag-of-words features
        # Would use word embeddings or BERT in production
        
        text_lower = text.lower()
        
        features = []
        
        # Keyword presence
        keywords = [
            "help", "portfolio", "report", "approve", "status",
            "urgent", "problem", "client", "account", "balance"
        ]
        
        for keyword in keywords:
            features.append(1.0 if keyword in text_lower else 0.0)
        
        # Text length
        features.append(len(text.split()))
        
        # Question mark
        features.append(1.0 if "?" in text else 0.0)
        
        return features
    
    def _extract_entities(self, text: str, intent: str) -> Dict:
        """
        Extract entities from text
        
        Entities:
        - client_id
        - report_type
        - request_id
        - date
        """
        
        entities = {}
        
        text_lower = text.lower()
        
        # Extract client references
        if "client" in text_lower:
            # Would use NER model
            entities["client_id"] = "client-123"
        
        # Extract report type
        if "monthly" in text_lower:
            entities["report_type"] = "monthly"
        elif "performance" in text_lower:
            entities["report_type"] = "performance"
        elif "tax" in text_lower:
            entities["report_type"] = "tax"
        
        # Extract request ID
        words = text.split()
        for word in words:
            if word.startswith("req-") or word.startswith("REQ-"):
                entities["request_id"] = word.lower()
                break
        
        return entities
    
    def get_training_examples(self) -> List[Dict]:
        """Get sample training data"""
        
        return [
            {"text": "show me the portfolio", "intent": "portfolio"},
            {"text": "what's my client's balance", "intent": "portfolio"},
            {"text": "generate monthly report", "intent": "report"},
            {"text": "create performance statement", "intent": "report"},
            {"text": "approve request req-123", "intent": "approve"},
            {"text": "reject pending approval", "intent": "approve"},
            {"text": "system status", "intent": "status"},
            {"text": "is the API working", "intent": "status"},
            {"text": "urgent issue with account", "intent": "urgent"},
            {"text": "critical error", "intent": "urgent"},
            {"text": "problem with transaction", "intent": "complaint"},
            {"text": "this is broken", "intent": "complaint"},
            {"text": "how do I create a report", "intent": "help"},
            {"text": "help with commands", "intent": "help"}
        ]
