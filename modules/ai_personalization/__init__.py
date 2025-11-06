"""
Ultra Platform - AI-Driven Personalization Framework

Advanced machine learning and behavioral analytics for highly personalized
financial advice, content, and experiences.

Features:
- Behavioral Analytics & Pattern Recognition (>90% churn prediction)
- Life Event Prediction (>75% accuracy)
- NLP Engine (>95% intent recognition)
- Content Personalization (>85% relevance)
- Recommendation Intelligence (>60% acceptance)
- Real-time Processing (<100ms p99)

Based on: Section 5 - AI-Driven Personalization Framework
Version: 1.0.0
"""

from .personalization_engine import (
    # Core framework
    PersonalizationFramework,
    
    # Component engines
    BehavioralAnalyzer,
    LifeEventPredictor,
    NLPEngine,
    ContentPersonalizationEngine,
    RecommendationEngine,
    
    # Data classes
    BehaviorEvent,
    ClientBehaviorProfile,
    LifeEventPrediction,
    PersonalizedRecommendation,
    
    # Enums
    EngagementLevel,
    InvestmentStyle,
    RiskAppetite,
    LifeStage,
    LifeEventType,
    ContentType,
    SentimentType
)

__version__ = "1.0.0"

__all__ = [
    # Core framework
    "PersonalizationFramework",
    
    # Component engines
    "BehavioralAnalyzer",
    "LifeEventPredictor",
    "NLPEngine",
    "ContentPersonalizationEngine",
    "RecommendationEngine",
    
    # Data classes
    "BehaviorEvent",
    "ClientBehaviorProfile",
    "LifeEventPrediction",
    "PersonalizedRecommendation",
    
    # Enums
    "EngagementLevel",
    "InvestmentStyle",
    "RiskAppetite",
    "LifeStage",
    "LifeEventType",
    "ContentType",
    "SentimentType"
]
